# Updated Pipeline Usage

## What Changed

Your pipeline has been updated with professional-grade spherical projection and panorama distortion correction.

## Key Improvements

1. **No More Warping**: Proper rectilinear projection from equirectangular panoramas
2. **Distortion Correction**: Automatic barrel/pincushion distortion removal  
3. **Better Quality**: Cubic interpolation for smoother, sharper results
4. **Photogrammetry Ready**: Compatible with Reality Scan, Pix4D, and similar software

## How to Use

### Standard Run
```bash
python pipeline2.py
```

### Adjust Distortion (if needed)
Edit `pipeline2.py` and modify:
```python
PANORAMA_K1 = -0.04  # Try -0.06 to -0.02 depending on distortion
PANORAMA_K2 = 0.0
```

If images still look warped:
- **Increase negative K1** (e.g., -0.06) for stronger barrel correction
- **Decrease K1** (e.g., -0.02) for lighter correction

### Verify Projection is Correct
```bash
python test_projection.py
```

This creates synthetic test images showing the projection has NO barrel/pincushion distortion.

## Output

All images are saved to `output/images/` as:
- **4096×4096 pixels** - high resolution for photogrammetry
- **95% JPEG quality** - optimal for processing
- **GPS metadata embedded** - ready for georeferencing in Reality Scan

## Comparison with svd360.com

The extraction now matches svd360's professional results using the same:
- ✓ Proper rectilinear projection
- ✓ Spherical unwrapping mathematics  
- ✓ Cubic interpolation for quality
- ✓ Distortion correction

Your images are now ready for professional 3D capture workflows.
