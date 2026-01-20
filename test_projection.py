#!/usr/bin/env python3
"""
Test the spherical projection to verify there's no warping.
Creates a synthetic equirectangular panorama with a grid pattern and tests extraction.
"""

import cv2
import numpy as np
import math

def extract_view_spherical(img, yaw_deg, pitch_deg=-10, fov_deg=90, out_size=4096):
    """Extract a rectilinear view from an equirectangular panorama."""
    h, w = img.shape[:2]
    
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    fov = np.deg2rad(fov_deg)
    half_fov = fov / 2.0
    
    f = (out_size / 2.0) / np.tan(half_fov)
    
    y_out, x_out = np.mgrid[0:out_size, 0:out_size]
    
    x_cam = (x_out - out_size/2.0) / f
    y_cam = (y_out - out_size/2.0) / f
    z_cam = np.ones_like(x_cam)
    
    r_cam = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= r_cam
    y_cam /= r_cam
    z_cam /= r_cam
    
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    x_rot = x_cam
    y_rot = y_cam * cos_p - z_cam * sin_p
    z_rot = y_cam * sin_p + z_cam * cos_p
    
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    x_world = x_rot * cos_y + z_rot * sin_y
    y_world = y_rot
    z_world = -x_rot * sin_y + z_rot * cos_y
    
    lon = np.arctan2(x_world, z_world)
    lat = np.arcsin(np.clip(y_world, -0.9999, 0.9999))
    
    u = (lon / (2 * np.pi) + 0.5) * (w - 1)
    v = (0.5 - lat / np.pi) * (h - 1)
    
    result = cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return result

# Create a synthetic equirectangular panorama with grid pattern
pano_w, pano_h = 4096, 2048
pano = np.ones((pano_h, pano_w, 3), dtype=np.uint8) * 255

# Draw grid lines (every 256 pixels)
for x in range(0, pano_w, 256):
    pano[:, x:x+2] = (0, 0, 255)  # Red vertical lines

for y in range(0, pano_h, 256):
    pano[y:y+2, :] = (0, 255, 0)  # Green horizontal lines

# Test extraction
print("Testing spherical projection...")
views = []
for yaw_offset in [0, 90, 180, 270]:
    print(f"Extracting view with yaw offset {yaw_offset}°...")
    view = extract_view_spherical(pano, yaw_offset, pitch_deg=-10, fov_deg=90, out_size=512)
    views.append(view)
    
    # Save for inspection
    cv2.imwrite(f"test_view_yaw{yaw_offset}.jpg", view)
    
    # Check if lines are straight (analyze vertical lines in the center)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    
    # Find vertical red lines (should be straight)
    red_channel = view[:, :, 2]
    vert_lines = np.where(red_channel > 200)[1]
    
    if len(vert_lines) > 0:
        # Check variance in line positions - should be low if lines are straight
        hist, _ = np.histogram(vert_lines, bins=512)
        
        # Lines should concentrate in certain columns
        concentrations = np.sum(hist > 10)
        print(f"  → Vertical line concentrations: {concentrations}")
        print(f"  → Mean line sharpness: {cv2.Laplacian(gray, cv2.CV_64F).var():.2f}")

print("\n✅ Test complete. Check test_view_yaw*.jpg files for visual inspection.")
print("Good projection: lines should appear straight without barrel/pincushion distortion")
