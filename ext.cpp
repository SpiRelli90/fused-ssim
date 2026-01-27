#include <torch/extension.h>
#include "ssim.h"

// Only include 3D SSIM for CUDA and CPU builds
#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_CPU)
#include "ssim3d.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 2D SSIM (available on all backends)
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);

  // 3D SSIM (CUDA and CPU)
#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_CPU)
  m.def("fusedssim3d", &fusedssim3d);
  m.def("fusedssim_backward3d", &fusedssim_backward3d);
#endif
}
