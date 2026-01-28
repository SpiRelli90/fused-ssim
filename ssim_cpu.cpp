#include <torch/extension.h>
#include "ssim.h"
#include <cmath>
#include <algorithm>

// MSVC-specific pragmas for Windows compatibility
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)  // Conversion from double to float
#pragma warning(disable: 4267)  // Conversion from size_t to int
#endif

// ------------------------------------------
// Gaussian Coefficients (11x11 kernel)
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
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
inline float get_pix_value(
    const float* img, 
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// CPU Forward Pass: Fused SSIM
// ------------------------------------------
void fusedssimCPU(
    int BS,
    int CH,
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
            // Process each pixel
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    // First pass: Horizontal convolution
                    float h_sumX = 0.0f, h_sumX2 = 0.0f;
                    float h_sumY = 0.0f, h_sumY2 = 0.0f;
                    float h_sumXY = 0.0f;
                    
                    for (int dx = -HALO; dx <= HALO; dx++) {
                        float weight = cGauss[dx + HALO];
                        float val1 = get_pix_value(img1, b, c, y, x + dx, CH, H, W);
                        float val2 = get_pix_value(img2, b, c, y, x + dx, CH, H, W);
                        
                        h_sumX += weight * val1;
                        h_sumX2 += weight * val1 * val1;
                        h_sumY += weight * val2;
                        h_sumY2 += weight * val2 * val2;
                        h_sumXY += weight * val1 * val2;
                    }
                    
                    // Second pass: Vertical convolution
                    float mu1 = 0.0f, sigma1_sq_raw = 0.0f;
                    float mu2 = 0.0f, sigma2_sq_raw = 0.0f;
                    float sigma12 = 0.0f;
                    
                    for (int dy = -HALO; dy <= HALO; dy++) {
                        float weight = cGauss[dy + HALO];
                        
                        // Re-compute horizontal sums for this row
                        float row_sumX = 0.0f, row_sumX2 = 0.0f;
                        float row_sumY = 0.0f, row_sumY2 = 0.0f;
                        float row_sumXY = 0.0f;
                        
                        for (int dx = -HALO; dx <= HALO; dx++) {
                            float w = cGauss[dx + HALO];
                            float val1 = get_pix_value(img1, b, c, y + dy, x + dx, CH, H, W);
                            float val2 = get_pix_value(img2, b, c, y + dy, x + dx, CH, H, W);
                            
                            row_sumX += w * val1;
                            row_sumX2 += w * val1 * val1;
                            row_sumY += w * val2;
                            row_sumY2 += w * val2 * val2;
                            row_sumXY += w * val1 * val2;
                        }
                        
                        mu1 += weight * row_sumX;
                        sigma1_sq_raw += weight * row_sumX2;
                        mu2 += weight * row_sumY;
                        sigma2_sq_raw += weight * row_sumY2;
                        sigma12 += weight * row_sumXY;
                    }
                    
                    // Compute variances and covariance
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;
                    float mu1_mu2 = mu1 * mu2;
                    
                    float sigma1_sq = std::max(0.0f, sigma1_sq_raw - mu1_sq);
                    float sigma2_sq = std::max(0.0f, sigma2_sq_raw - mu2_sq);
                    sigma12 = sigma12 - mu1_mu2;
                    
                    // Compute SSIM
                    float A = mu1_sq + mu2_sq + C1;
                    float B = sigma1_sq + sigma2_sq + C2;
                    float C_ = 2.0f * mu1_mu2 + C1;
                    float D_ = 2.0f * sigma12 + C2;
                    
                    float ssim_val = (C_ * D_) / (A * B);
                    ssim_val = std::max(-1.0f, std::min(1.0f, ssim_val));
                    
                    int idx = b * CH * H * W + c * H * W + y * W + x;
                    ssim_map[idx] = ssim_val;
                    
                    // Store partial derivatives for backward pass if training
                    if (train) {
                        // Check if value was clamped - if so, derivatives are zero
                        bool clamped = (ssim_val <= -1.0f || ssim_val >= 1.0f);
                        
                        // ∂SSIM/∂μ₁ - matches CUDA implementation
                        float dssim_dmu1 = clamped ? 0.0f : (
                            (mu2 * 2.0f * D_) / (A * B)
                            - (mu2 * 2.0f * C_) / (A * B)
                            - (mu1 * 2.0f * C_ * D_) / (A * A * B)
                            + (mu1 * 2.0f * C_ * D_) / (A * B * B)
                        );
                        
                        // ∂SSIM/∂σ₁² - matches CUDA implementation
                        float dssim_dsigma1_sq = clamped ? 0.0f : (-C_ * D_) / (A * B * B);
                        
                        // ∂SSIM/∂σ₁₂ - matches CUDA implementation
                        float dssim_dsigma12 = clamped ? 0.0f : (2.0f * C_) / (A * B);
                        
                        dm_dmu1[idx] = dssim_dmu1;
                        dm_dsigma1_sq[idx] = dssim_dsigma1_sq;
                        dm_dsigma12[idx] = dssim_dsigma12;
                    }
                }
            }
        }
    }
}

// ------------------------------------------
// CPU Backward Pass
// Matches CUDA implementation approach:
// Convolve the fused derivatives and accumulate with pixel values
// ------------------------------------------
void fusedssimBackwardCPU(
    int BS,
    int CH,
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
    for (int i = 0; i < BS * CH * H * W; i++) {
        dL_dimg1[i] = 0.0f;
    }
    
    // Process each batch and channel
    for (int b = 0; b < BS; b++) {
        for (int c = 0; c < CH; c++) {
            // Process each pixel
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int idx = b * CH * H * W + c * H * W + y * W + x;
                    
                    // Get pixel values
                    float p1 = img1[idx];
                    float p2 = img2[idx];
                    
                    // Fuse the partial derivatives with dL_dmap before convolution
                    // This matches the CUDA approach
                    
                    // Perform 2D convolution on the fused derivatives
                    float sum0 = 0.0f;  // Conv(dm_dmu1 * dL_dmap)
                    float sum1 = 0.0f;  // Conv(dm_dsigma1_sq * dL_dmap)
                    float sum2 = 0.0f;  // Conv(dm_dsigma12 * dL_dmap)
                    
                    for (int dy = -HALO; dy <= HALO; dy++) {
                        for (int dx = -HALO; dx <= HALO; dx++) {
                            float weight = cGauss[dy + HALO] * cGauss[dx + HALO];
                            
                            int ky = y + dy;
                            int kx = x + dx;
                            
                            if (ky >= 0 && ky < H && kx >= 0 && kx < W) {
                                int k_idx = b * CH * H * W + c * H * W + ky * W + kx;
                                
                                float chain = dL_dmap[k_idx];
                                float vmu = dm_dmu1[k_idx];
                                float vs1 = dm_dsigma1_sq[k_idx];
                                float vs12 = dm_dsigma12[k_idx];
                                
                                sum0 += weight * vmu * chain;
                                sum1 += weight * vs1 * chain;
                                sum2 += weight * vs12 * chain;
                            }
                        }
                    }
                    
                    // Final gradient accumulation - matches CUDA line 422
                    float dL_dpix = sum0 + (2.0f * p1) * sum1 + p2 * sum2;
                    dL_dimg1[idx] = dL_dpix;
                }
            }
        }
    }
}

// ------------------------------------------
// PyTorch Interface Functions
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    const int BS = img1.size(0);
    const int CH = img1.size(1);
    const int H = img1.size(2);
    const int W = img1.size(3);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(img1.device());
    
    torch::Tensor ssim_map = torch::zeros({BS, CH, H, W}, options);
    torch::Tensor dm_dmu1 = train ? torch::zeros({BS, CH, H, W}, options) : torch::empty({0}, options);
    torch::Tensor dm_dsigma1_sq = train ? torch::zeros({BS, CH, H, W}, options) : torch::empty({0}, options);
    torch::Tensor dm_dsigma12 = train ? torch::zeros({BS, CH, H, W}, options) : torch::empty({0}, options);
    
    const float* img1_ptr = img1.data_ptr<float>();
    const float* img2_ptr = img2.data_ptr<float>();
    float* ssim_map_ptr = ssim_map.data_ptr<float>();
    float* dm_dmu1_ptr = train ? dm_dmu1.data_ptr<float>() : nullptr;
    float* dm_dsigma1_sq_ptr = train ? dm_dsigma1_sq.data_ptr<float>() : nullptr;
    float* dm_dsigma12_ptr = train ? dm_dsigma12.data_ptr<float>() : nullptr;
    
    fusedssimCPU(BS, CH, H, W, C1, C2, img1_ptr, img2_ptr,
                 ssim_map_ptr, dm_dmu1_ptr, dm_dsigma1_sq_ptr, dm_dsigma12_ptr, train);
    
    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

torch::Tensor
fusedssim_backward(
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
    const int H = img1.size(2);
    const int W = img1.size(3);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(img1.device());
    torch::Tensor dL_dimg1 = torch::zeros({BS, CH, H, W}, options);
    
    const float* img1_ptr = img1.data_ptr<float>();
    const float* img2_ptr = img2.data_ptr<float>();
    const float* dL_dmap_ptr = dL_dmap.data_ptr<float>();
    const float* dm_dmu1_ptr = dm_dmu1.data_ptr<float>();
    const float* dm_dsigma1_sq_ptr = dm_dsigma1_sq.data_ptr<float>();
    const float* dm_dsigma12_ptr = dm_dsigma12.data_ptr<float>();
    float* dL_dimg1_ptr = dL_dimg1.data_ptr<float>();
    
    fusedssimBackwardCPU(BS, CH, H, W, C1, C2, img1_ptr, img2_ptr,
                         dL_dmap_ptr, dm_dmu1_ptr, dm_dsigma1_sq_ptr, dm_dsigma12_ptr,
                         dL_dimg1_ptr);
    
    return dL_dimg1;
}

// Restore MSVC warnings
#ifdef _MSC_VER
#pragma warning(pop)
#endif
