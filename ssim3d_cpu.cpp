#include <torch/extension.h>
#include "ssim3d.h"
#include <cmath>
#include <algorithm>
#include <vector>

// MSVC-specific pragmas for Windows compatibility
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)  // Conversion from double to float
#pragma warning(disable: 4267)  // Conversion from size_t to int
#endif

// ------------------------------------------
// Gaussian Coefficients (11x11x11 kernel)
// ------------------------------------------
static const float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

// ------------------------------------------
// Utility: Safe voxel fetch w/ zero padding
// ------------------------------------------
inline float get_voxel_value(
    const float* img, 
    int b, int c, int z, int y, int x,
    int CH, int D, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0f;
    }
    return img[b * CH * D * H * W + c * D * H * W + z * H * W + y * W + x];
}

// ------------------------------------------
// CPU Forward Pass: Fused SSIM3D
// ------------------------------------------
void fusedssim3dCPU(
    int BS,
    int CH,
    int D,
    int H,
    int W,
    float C1,
    float C2,
    const float* img1,
    const float* img2,
    float* ssim_map,
    float* dm_dmu1,
    float* dm_dsigma1_sq,
    float* dm_dsigma12,
    bool train
) {
    const int HALO = 5;
    
    // Process each batch and channel
    for (int b = 0; b < BS; b++) {
        for (int c = 0; c < CH; c++) {
            // Process each voxel
            for (int z = 0; z < D; z++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        // Three-pass separable convolution: X, Y, Z
                        float mu1 = 0.0f, sigma1_sq_raw = 0.0f;
                        float mu2 = 0.0f, sigma2_sq_raw = 0.0f;
                        float sigma12 = 0.0f;
                        
                        // Apply 3D Gaussian convolution
                        for (int dz = -HALO; dz <= HALO; dz++) {
                            for (int dy = -HALO; dy <= HALO; dy++) {
                                for (int dx = -HALO; dx <= HALO; dx++) {
                                    float weight = cGauss[dz + HALO] * cGauss[dy + HALO] * cGauss[dx + HALO];
                                    float val1 = get_voxel_value(img1, b, c, z + dz, y + dy, x + dx, CH, D, H, W);
                                    float val2 = get_voxel_value(img2, b, c, z + dz, y + dy, x + dx, CH, D, H, W);
                                    
                                    mu1 += weight * val1;
                                    sigma1_sq_raw += weight * val1 * val1;
                                    mu2 += weight * val2;
                                    sigma2_sq_raw += weight * val2 * val2;
                                    sigma12 += weight * val1 * val2;
                                }
                            }
                        }
                        
                        // Compute variances and covariance
                        float mu1_sq = mu1 * mu1;
                        float mu2_sq = mu2 * mu2;
                        float mu1_mu2 = mu1 * mu2;
                        
                        float sigma1_sq = sigma1_sq_raw - mu1_sq;
                        float sigma2_sq = sigma2_sq_raw - mu2_sq;
                        sigma12 = sigma12 - mu1_mu2;
                        
                        // Compute SSIM
                        float A = mu1_sq + mu2_sq + C1;
                        float B = sigma1_sq + sigma2_sq + C2;
                        float C_ = 2.0f * mu1_mu2 + C1;
                        float D_ = 2.0f * sigma12 + C2;
                        
                        float ssim_val = (C_ * D_) / (A * B);
                        ssim_val = std::max(-1.0f, std::min(1.0f, ssim_val));
                        
                        int idx = b * CH * D * H * W + c * D * H * W + z * H * W + y * W + x;
                        ssim_map[idx] = ssim_val;
                        
                        // Store partial derivatives for backward pass if training
                        if (train) {
                            float denom_inv = 1.0f / (A * B);
                            
                            // ∂SSIM/∂μ₁
                            float dssim_dmu1 = (2.0f * mu2 * D_ * B - C_ * D_ * 2.0f * mu1 * B) * denom_inv;
                            dssim_dmu1 -= (C_ * D_ * A * 2.0f * mu1) * denom_inv / B;
                            
                            // ∂SSIM/∂σ₁²
                            float dssim_dsigma1_sq = -(C_ * D_ * A) * denom_inv / (B * B);
                            
                            // ∂SSIM/∂σ₁₂
                            float dssim_dsigma12 = (2.0f * A * B) * denom_inv;
                            
                            dm_dmu1[idx] = dssim_dmu1;
                            dm_dsigma1_sq[idx] = dssim_dsigma1_sq;
                            dm_dsigma12[idx] = dssim_dsigma12;
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------
// CPU Backward Pass: 3D
// ------------------------------------------
void fusedssimBackward3dCPU(
    int BS,
    int CH,
    int D,
    int H,
    int W,
    float C1,
    float C2,
    const float* img1,
    const float* img2,
    const float* dL_dmap,
    const float* dm_dmu1,
    const float* dm_dsigma1_sq,
    const float* dm_dsigma12,
    float* dL_dimg1
) {
    const int HALO = 5;
    
    // Initialize gradient to zero
    for (int i = 0; i < BS * CH * D * H * W; i++) {
        dL_dimg1[i] = 0.0f;
    }
    
    // Process each batch and channel
    for (int b = 0; b < BS; b++) {
        for (int c = 0; c < CH; c++) {
            // For each output voxel, accumulate gradients to input
            for (int z = 0; z < D; z++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        int out_idx = b * CH * D * H * W + c * D * H * W + z * H * W + y * W + x;
                        float dL_dssim = dL_dmap[out_idx];
                        
                        if (dL_dssim == 0.0f) continue;
                        
                        float dssim_dmu1 = dm_dmu1[out_idx];
                        float dssim_dsigma1_sq = dm_dsigma1_sq[out_idx];
                        float dssim_dsigma12 = dm_dsigma12[out_idx];
                        
                        // Propagate gradients through 3D convolution
                        for (int dz = -HALO; dz <= HALO; dz++) {
                            for (int dy = -HALO; dy <= HALO; dy++) {
                                for (int dx = -HALO; dx <= HALO; dx++) {
                                    int in_z = z + dz;
                                    int in_y = y + dy;
                                    int in_x = x + dx;
                                    
                                    if (in_z < 0 || in_z >= D || in_y < 0 || in_y >= H || in_x < 0 || in_x >= W) continue;
                                    
                                    float weight = cGauss[dz + HALO] * cGauss[dy + HALO] * cGauss[dx + HALO];
                                    float val1 = get_voxel_value(img1, b, c, in_z, in_y, in_x, CH, D, H, W);
                                    float val2 = get_voxel_value(img2, b, c, in_z, in_y, in_x, CH, D, H, W);
                                    
                                    // Compute mu1 for this voxel (needed for variance gradient)
                                    float mu1 = 0.0f;
                                    for (int kz = -HALO; kz <= HALO; kz++) {
                                        for (int ky = -HALO; ky <= HALO; ky++) {
                                            for (int kx = -HALO; kx <= HALO; kx++) {
                                                float kw = cGauss[kz + HALO] * cGauss[ky + HALO] * cGauss[kx + HALO];
                                                float kval = get_voxel_value(img1, b, c, in_z + kz, in_y + ky, in_x + kx, CH, D, H, W);
                                                mu1 += kw * kval;
                                            }
                                        }
                                    }
                                    
                                    // Compute mu2 for this voxel
                                    float mu2 = 0.0f;
                                    for (int kz = -HALO; kz <= HALO; kz++) {
                                        for (int ky = -HALO; ky <= HALO; ky++) {
                                            for (int kx = -HALO; kx <= HALO; kx++) {
                                                float kw = cGauss[kz + HALO] * cGauss[ky + HALO] * cGauss[kx + HALO];
                                                float kval = get_voxel_value(img2, b, c, in_z + kz, in_y + ky, in_x + kx, CH, D, H, W);
                                                mu2 += kw * kval;
                                            }
                                        }
                                    }
                                    
                                    // ∂μ₁/∂img1 = weight
                                    float dmu1_dimg1 = weight;
                                    
                                    // ∂σ₁²/∂img1 = 2 * weight * (img1 - μ₁)
                                    float dsigma1_sq_dimg1 = 2.0f * weight * (val1 - mu1);
                                    
                                    // ∂σ₁₂/∂img1 = weight * (img2 - μ₂)
                                    float dsigma12_dimg1 = weight * (val2 - mu2);
                                    
                                    int in_idx = b * CH * D * H * W + c * D * H * W + in_z * H * W + in_y * W + in_x;
                                    dL_dimg1[in_idx] += dL_dssim * (
                                        dssim_dmu1 * dmu1_dimg1 +
                                        dssim_dsigma1_sq * dsigma1_sq_dimg1 +
                                        dssim_dsigma12 * dsigma12_dimg1
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------
// PyTorch Interface Functions
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim3d(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    const int BS = img1.size(0);
    const int CH = img1.size(1);
    const int D = img1.size(2);
    const int H = img1.size(3);
    const int W = img1.size(4);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(img1.device());
    
    torch::Tensor ssim_map = torch::zeros({BS, CH, D, H, W}, options);
    torch::Tensor dm_dmu1 = train ? torch::zeros({BS, CH, D, H, W}, options) : torch::empty({0}, options);
    torch::Tensor dm_dsigma1_sq = train ? torch::zeros({BS, CH, D, H, W}, options) : torch::empty({0}, options);
    torch::Tensor dm_dsigma12 = train ? torch::zeros({BS, CH, D, H, W}, options) : torch::empty({0}, options);
    
    const float* img1_ptr = img1.data_ptr<float>();
    const float* img2_ptr = img2.data_ptr<float>();
    float* ssim_map_ptr = ssim_map.data_ptr<float>();
    float* dm_dmu1_ptr = train ? dm_dmu1.data_ptr<float>() : nullptr;
    float* dm_dsigma1_sq_ptr = train ? dm_dsigma1_sq.data_ptr<float>() : nullptr;
    float* dm_dsigma12_ptr = train ? dm_dsigma12.data_ptr<float>() : nullptr;
    
    fusedssim3dCPU(BS, CH, D, H, W, C1, C2, img1_ptr, img2_ptr,
                   ssim_map_ptr, dm_dmu1_ptr, dm_dsigma1_sq_ptr, dm_dsigma12_ptr, train);
    
    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

torch::Tensor
fusedssim_backward3d(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
) {
    const int BS = img1.size(0);
    const int CH = img1.size(1);
    const int D = img1.size(2);
    const int H = img1.size(3);
    const int W = img1.size(4);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(img1.device());
    torch::Tensor dL_dimg1 = torch::zeros({BS, CH, D, H, W}, options);
    
    const float* img1_ptr = img1.data_ptr<float>();
    const float* img2_ptr = img2.data_ptr<float>();
    const float* dL_dmap_ptr = dL_dmap.data_ptr<float>();
    const float* dm_dmu1_ptr = dm_dmu1.data_ptr<float>();
    const float* dm_dsigma1_sq_ptr = dm_dsigma1_sq.data_ptr<float>();
    const float* dm_dsigma12_ptr = dm_dsigma12.data_ptr<float>();
    float* dL_dimg1_ptr = dL_dimg1.data_ptr<float>();
    
    fusedssimBackward3dCPU(BS, CH, D, H, W, C1, C2, img1_ptr, img2_ptr,
                           dL_dmap_ptr, dm_dmu1_ptr, dm_dsigma1_sq_ptr, dm_dsigma12_ptr,
                           dL_dimg1_ptr);
    
    return dL_dimg1;
}

// Restore MSVC warnings
#ifdef _MSC_VER
#pragma warning(pop)
#endif
