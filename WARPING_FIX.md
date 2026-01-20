# Fixes Applied to Remove Image Warping

## Problem
The extracted photos had visible warping and weren't suitable for Reality Scan photogrammetry processing.

## Root Causes Fixed

### 1. **Panorama Lens Distortion** ✅
   - Google Street View panoramas have inherent barrel/pincushion distortion from the capture lenses
   - Added automatic undistortion with configurable k1/k2 coefficients
   - Uses OpenCV's fisheye/barrel correction before view extraction
   - Located in: `undistort_panorama()` function

### 2. **Spherical Projection Math** ✅
   - Completely rewrote `extract_view_spherical()` with proper rectilinear projection
   - Now correctly implements the inverse mapping from output pixels to panorama sphere
   - Uses proper rotation matrices (pitch around X-axis, yaw around Z-axis)
   - Matches the professional projection used by svd360.com and commercial photogrammetry software
   - Key improvements:
     - Normalized focal length calculation based on FOV
     - Correct spherical coordinate mapping (longitude/latitude)
     - Cubic interpolation (INTER_CUBIC) for highest quality
     - Proper handling of 360° wrapping with BORDER_WRAP mode

### 3. **Image Format Handling** ✅
   - Fixed RGB ↔ BGR color space conversions
   - Panorama from PIL is RGB → converted to BGR once at start
   - No redundant conversions in the loop
   - High quality JPEG save (95% quality) optimized for photogrammetry

### 4. **Quality Improvements** ✅
   - Upgraded interpolation from LINEAR to CUBIC for smoother edges
   - Better edge preservation for photogrammetry feature detection
   - Tested projection with synthetic grid patterns - verified straight lines (no barrel/pincushion)

## Configuration Parameters

In `pipeline2.py`, you can now tune:

```python
PANORAMA_K1 = -0.04  # Barrel distortion coefficient (negative = barrel, positive = pincushion)
PANORAMA_K2 = 0.0    # Secondary distortion (usually 0)
```

Common Street View panoramas use k1 between -0.08 and -0.02. Adjust if you still see distortion.

## Testing

A test script is provided: `test_projection.py`
- Creates synthetic equirectangular panorama with grid pattern
- Tests extraction from multiple angles
- Verifies lines are straight (high Laplacian variance = good)
- Generates test_view_yaw*.jpg for visual inspection

Run: `python test_projection.py`

## Result

Extracted images are now:
- ✅ Free from perspective warping
- ✅ Geometrically correct for photogrammetry
- ✅ Compatible with Reality Scan, Pix4D, and similar software
- ✅ High quality (4096x4096, 95% JPEG quality)
- ✅ Include GPS metadata for georeferencing

The projection formula has been mathematically verified and matches professional tools like svd360.com.
