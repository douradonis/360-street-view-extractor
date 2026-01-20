#!/usr/bin/env python3
"""
Enhance the pipeline with automatic fisheye distortion correction for panoramas.
This handles the lens distortion that may be present in Street View images.
"""

import cv2
import numpy as np

def undistort_panorama_simple(img, k1=-0.05, k2=0.0):
    """
    Apply barrel/pincushion distortion correction to panorama.
    
    k1: primary radial distortion coefficient (negative = barrel, positive = pincushion)
    k2: secondary radial distortion coefficient
    
    For Street View panoramas, typical values are:
    k1 ~ -0.08 to -0.02 (slight barrel distortion)
    """
    h, w = img.shape[:2]
    
    # Camera matrix for panorama
    # For equirectangular, we use a virtual camera at center
    cx, cy = w / 2, h / 2
    # Focal length chosen so that corners map reasonably
    f = w / (2 * np.pi)
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    # Create undistortion maps
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w, h), cv2.CV_32F)
    
    # Apply undistortion
    undistorted = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    
    return undistorted

# Test it
if __name__ == "__main__":
    print("This module provides panorama undistortion for better extraction quality.")
    print("Add to pipeline2.py after downloading panorama:")
    print("  pano_img = undistort_panorama_simple(pano_img, k1=-0.04)")
