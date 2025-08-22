#!/usr/bin/env python3
"""
C2 Regression test - ensure PNG round-trip remains deterministic
"""
import hashlib
import tempfile
from pathlib import Path

def test_c2_png_round_trip_512x512():
    """Ensure 512x512 triangle rendering remains deterministic pre/post changes"""
    import forge3d as f3d
    
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = Path(tmpdir) / "triangle_512x512.png"
        
        # Render triangle
        r = f3d.Renderer(512, 512)
        r.render_triangle_png(png_path)
        
        # Verify file exists and read bytes
        assert png_path.exists(), "PNG file was not created"
        png_bytes = png_path.read_bytes()
        
        # Compute hash for determinism
        sha256_hash = hashlib.sha256(png_bytes).hexdigest()
        
        # Verify reasonable file size (should be > 1KB for 512x512 PNG)
        assert len(png_bytes) > 1024, f"PNG file too small: {len(png_bytes)} bytes"
        
        # Render again and verify determinism
        png_path2 = Path(tmpdir) / "triangle_512x512_again.png"
        r2 = f3d.Renderer(512, 512)
        r2.render_triangle_png(png_path2)
        
        png_bytes2 = png_path2.read_bytes()
        sha256_hash2 = hashlib.sha256(png_bytes2).hexdigest()
        
        assert sha256_hash == sha256_hash2, f"PNG not deterministic: {sha256_hash} != {sha256_hash2}"
        
        print(f"[OK] C2 PNG round-trip test PASSED: {len(png_bytes)} bytes, SHA256: {sha256_hash[:16]}...")

def test_c2_rgba_readback():
    """Test RGBA readback path for correctness"""
    import forge3d as f3d
    import numpy as np
    
    # Render triangle to RGBA array
    r = f3d.Renderer(64, 64)
    rgba = r.render_triangle_rgba()
    
    # Verify shape and dtype
    assert rgba.shape == (64, 64, 4), f"Expected (64,64,4), got {rgba.shape}"
    assert rgba.dtype == np.uint8, f"Expected uint8, got {rgba.dtype}"
    
    # Verify some pixels are not all white (triangle should have colors)
    white_pixels = np.all(rgba == [255, 255, 255, 255], axis=2)
    non_white_count = np.sum(~white_pixels)
    
    assert non_white_count > 100, f"Too few colored pixels: {non_white_count} (triangle not visible?)"
    
    print(f"[OK] C2 RGBA readback test PASSED: {rgba.shape}, {non_white_count} colored pixels")

if __name__ == "__main__":
    test_c2_png_round_trip_512x512()
    test_c2_rgba_readback()
    print("[OK] All C2 regression tests passed")