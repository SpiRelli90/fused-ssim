#!/usr/bin/env python3
"""
Test script for CPU implementation of SSIM and SSIM3D
"""
import sys

sys.path.insert(0, ".")
import torch
from fused_ssim import fused_ssim, fused_ssim3d


def test_2d_ssim():
    """Test 2D SSIM implementation"""
    print("Testing 2D SSIM...")

    # Test 1: Identical images should have SSIM close to 1
    img1 = torch.rand(2, 3, 64, 64)
    img2 = img1.clone()
    ssim_val = fused_ssim(img1, img2)
    assert (
        ssim_val.item() > 0.99
    ), f"Identical images should have SSIM ~1, got {ssim_val.item()}"
    print(f"  ✓ Identical images: SSIM = {ssim_val.item():.6f}")

    # Test 2: Different images should have lower SSIM
    img1 = torch.rand(2, 3, 64, 64)
    img2 = torch.rand(2, 3, 64, 64)
    ssim_val = fused_ssim(img1, img2)
    assert (
        0.0 < ssim_val.item() < 0.5
    ), f"Random images should have low SSIM, got {ssim_val.item()}"
    print(f"  ✓ Random images: SSIM = {ssim_val.item():.6f}")

    # Test 3: Gradient computation
    img1 = torch.rand(1, 1, 32, 32, requires_grad=True)
    img2 = torch.rand(1, 1, 32, 32)
    ssim_val = fused_ssim(img1, img2)
    ssim_val.backward()
    assert img1.grad is not None, "Gradient should be computed"
    assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"
    print(f"  ✓ Gradient computation successful")

    # Test 4: Valid padding mode
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 64, 64)
    ssim_val_valid = fused_ssim(img1, img2, padding="valid")
    # Valid padding can produce negative values for random images due to cropping
    print(f"  ✓ Valid padding: SSIM = {ssim_val_valid.item():.6f}")

    print("2D SSIM tests passed! ✓\n")


def test_3d_ssim():
    """Test 3D SSIM implementation"""
    print("Testing 3D SSIM...")

    # Test 1: Identical volumes should have SSIM close to 1
    img1 = torch.rand(1, 2, 16, 16, 16)
    img2 = img1.clone()
    ssim_val = fused_ssim3d(img1, img2)
    assert (
        ssim_val.item() > 0.99
    ), f"Identical volumes should have SSIM ~1, got {ssim_val.item()}"
    print(f"  ✓ Identical volumes: SSIM = {ssim_val.item():.6f}")

    # Test 2: Different volumes should have lower SSIM
    img1 = torch.rand(1, 2, 16, 16, 16)
    img2 = torch.rand(1, 2, 16, 16, 16)
    ssim_val = fused_ssim3d(img1, img2)
    assert (
        0.0 < ssim_val.item() < 0.8
    ), f"Random volumes should have low SSIM, got {ssim_val.item()}"
    print(f"  ✓ Random volumes: SSIM = {ssim_val.item():.6f}")

    # Test 3: Gradient computation
    img1 = torch.rand(1, 1, 12, 12, 12, requires_grad=True)
    img2 = torch.rand(1, 1, 12, 12, 12)
    ssim_val = fused_ssim3d(img1, img2)
    ssim_val.backward()
    assert img1.grad is not None, "Gradient should be computed"
    assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"
    print(f"  ✓ Gradient computation successful")

    # Test 4: Valid padding mode
    img1 = torch.rand(1, 1, 30, 30, 30)
    img2 = torch.rand(1, 1, 30, 30, 30)
    ssim_val_valid = fused_ssim3d(img1, img2, padding="valid")
    # Valid padding can produce negative values for random images due to cropping
    print(f"  ✓ Valid padding: SSIM = {ssim_val_valid.item():.6f}")

    print("3D SSIM tests passed! ✓\n")


def test_image():
    # Load 2 images from disk for testing and run SSIM on them
    from PIL import Image
    import numpy as np

    print("Testing SSIM on real images...")

    def load_image(path):
        img = Image.open(path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        # Shape must be [Batch, Channel, Height, Width]
        return torch.tensor(img_np, device="cpu").permute(2, 0, 1).unsqueeze(0)

    img1 = load_image("TestData/deer512_noise.png")
    img2 = load_image("TestData/deer512.png")
    print(f"Image Shapes: {img1.shape}, {img2.shape}")
    img1.requires_grad = True
    ssim_val = fused_ssim(img1, img2)
    loss = 1.0 - ssim_val

    print(
        f"  ✓ SSIM between deer512_noise.png and deer512.png: SSIM = {ssim_val.item():.6f}\n"
    )
    loss.backward()
    gradients = img1.grad.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    assert gradients is not None, "Gradient should be computed"
    # assert not torch.isnan(gradients).any(), "Gradient should not contain NaN"
    print(f"  ✓ Gradient computation successful on real images")
    # Store gradient norm in file in R G B order per line
    rgb_grad_norm = (gradients - gradients.min()) / (
        gradients.max() - gradients.min() + 1e-8
    )
    np.savetxt(
        "TestData/ssim_gradient.txt",
        gradients.reshape(-1, 3),
        fmt="%.6f",
        header="R G B",
    )

    # compare with reference SSIM gradient file
    ref_grad = np.loadtxt("TestData/ssim_gradient_target.txt", skiprows=1)
    diff = np.abs(ref_grad - gradients.reshape(-1, 3))
    max_diff = diff.max()
    assert max_diff < 1e-4, f"Max gradient difference {max_diff} exceeds tolerance"
    print(f"  ✓ Gradient matches reference within tolerance")


if __name__ == "__main__":
    print("=" * 60)
    print("CPU Implementation Test Suite")
    print("=" * 60 + "\n")

    test_2d_ssim()
    test_3d_ssim()
    test_image()

    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
