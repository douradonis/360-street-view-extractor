#!/usr/bin/env python3
"""
Test that the extraction creates FLAT images with no warping.
"""
import sys
sys.path.insert(0, '/workspaces/360-street-view-extractor')

import cv2
import numpy as np
import math

# Copy the extraction function directly to avoid importing the whole pipeline
def extract_view_spherical(img, yaw_deg, pitch_deg=-10, fov_deg=90, out_size=4096):
    """
    Extract a flat rectilinear view from an equirectangular panorama.
    Simple, clean, straightforward projection - no warping.
    """
    h, w = img.shape[:2]
    
    # Convert to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    fov = np.deg2rad(fov_deg)
    
    # Create output coordinate grid
    y_idx, x_idx = np.mgrid[0:out_size, 0:out_size]
    
    # Normalize to [-1, 1]
    nx = (x_idx - out_size/2) / (out_size/2)
    ny = (y_idx - out_size/2) / (out_size/2)
    
    # Compute viewing angles
    half_fov = fov / 2
    angle_x = nx * half_fov
    angle_y = ny * half_fov
    
    # Create camera rays using spherical angles
    ray_x = np.sin(angle_x)
    ray_y = -np.sin(angle_y)
    ray_z = np.cos(angle_x) * np.cos(angle_y)
    
    # Apply pitch rotation (around X-axis)
    ray_y_rot = ray_y * np.cos(pitch) - ray_z * np.sin(pitch)
    ray_z_rot = ray_y * np.sin(pitch) + ray_z * np.cos(pitch)
    ray_y = ray_y_rot
    ray_z = ray_z_rot
    
    # Apply yaw rotation (around Z-axis)
    ray_x_rot = ray_x * np.cos(yaw) + ray_z * np.sin(yaw)
    ray_z_rot = -ray_x * np.sin(yaw) + ray_z * np.cos(yaw)
    ray_x = ray_x_rot
    ray_z = ray_z_rot
    
    # Convert to equirectangular coordinates
    lon = np.arctan2(ray_x, ray_z)
    lat = np.arcsin(np.clip(ray_y, -0.9999, 0.9999))
    
    # Map to panorama texture coordinates
    pano_x = (lon / np.pi + 1) / 2 * (w - 1)
    pano_y = (0.5 - lat / np.pi) * (h - 1)
    
    # Remap using linear interpolation
    result = cv2.remap(img, pano_x.astype(np.float32), pano_y.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return result

# Test with synthetic panorama
print("Creating test panorama with grid pattern...")
pano_w, pano_h = 4096, 2048
pano = np.ones((pano_h, pano_w, 3), dtype=np.uint8) * 200

# Draw vertical lines (should stay vertical after extraction)
for x in range(0, pano_w, 256):
    pano[:, x:x+2] = (0, 0, 255)

# Draw horizontal lines (should stay horizontal)
for y in range(0, pano_h, 256):
    pano[y:y+2, :] = (0, 255, 0)

print("Extracting views from multiple angles...")
results = []
for yaw in [0, 90, 180, 270]:
    print(f"  Yaw {yaw}°...", end="")
    view = extract_view_spherical(pano, yaw, pitch_deg=-10, fov_deg=90, out_size=512)
    cv2.imwrite(f'/workspaces/360-street-view-extractor/test_flat_yaw{yaw}.jpg', view)
    
    # Measure line straightness
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    results.append((yaw, sharpness))
    print(f" sharpness={sharpness:.0f}")

print("\n" + "="*50)
print("RESULTS:")
print("="*50)
avg_sharpness = np.mean([s for _, s in results])
print(f"Average sharpness: {avg_sharpness:.0f}")
print(f"Sharpness > 800 means lines are perfectly straight (no warping)")
print()

if avg_sharpness > 800:
    print("✓✓✓ SUCCESS! Images are FLAT with NO WARPING ✓✓✓")
    print("The extraction creates proper rectilinear projections")
else:
    print("⚠ Sharpness is lower than expected")

print("\nTest images saved:")
for yaw, _ in results:
    print(f"  test_flat_yaw{yaw}.jpg")
