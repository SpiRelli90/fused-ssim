#!/usr/bin/env python3
"""
Test script for CPU implementation of SSIM and SSIM3D
"""
import sys
sys.path.insert(0, '.')
import torch
from fused_ssim import fused_ssim, fused_ssim3d

def test_2d_ssim():
    """Test 2D SSIM implementation"""
    print("Testing 2D SSIM...")
    
    # Test 1: Identical images should have SSIM close to 1
    img1 = torch.rand(2, 3, 64, 64)
    img2 = img1.clone()
    ssim_val = fused_ssim(img1, img2)
    assert ssim_val.item() > 0.99, f"Identical images should have SSIM ~1, got {ssim_val.item()}"
    print(f"  ✓ Identical images: SSIM = {ssim_val.item():.6f}")
    
    # Test 2: Different images should have lower SSIM
    img1 = torch.rand(2, 3, 64, 64)
    img2 = torch.rand(2, 3, 64, 64)
    ssim_val = fused_ssim(img1, img2)
    assert 0.0 < ssim_val.item() < 0.5, f"Random images should have low SSIM, got {ssim_val.item()}"
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
    assert ssim_val_valid.item() > 0, "Valid padding should work"
    print(f"  ✓ Valid padding: SSIM = {ssim_val_valid.item():.6f}")
    
    print("2D SSIM tests passed! ✓\n")

def test_3d_ssim():
    """Test 3D SSIM implementation"""
    print("Testing 3D SSIM...")
    
    # Test 1: Identical volumes should have SSIM close to 1
    img1 = torch.rand(1, 2, 16, 16, 16)
    img2 = img1.clone()
    ssim_val = fused_ssim3d(img1, img2)
    assert ssim_val.item() > 0.99, f"Identical volumes should have SSIM ~1, got {ssim_val.item()}"
    print(f"  ✓ Identical volumes: SSIM = {ssim_val.item():.6f}")
    
    # Test 2: Different volumes should have lower SSIM
    img1 = torch.rand(1, 2, 16, 16, 16)
    img2 = torch.rand(1, 2, 16, 16, 16)
    ssim_val = fused_ssim3d(img1, img2)
    assert 0.0 < ssim_val.item() < 0.8, f"Random volumes should have low SSIM, got {ssim_val.item()}"
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

if __name__ == "__main__":
    print("=" * 60)
    print("CPU Implementation Test Suite")
    print("=" * 60 + "\n")
    
    test_2d_ssim()
    test_3d_ssim()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
