#!/usr/bin/env python3
"""
L3 Descriptor indexing tests for forge3d

Tests the descriptor indexing capability detection and terrain pipeline
texture array functionality when supported by the GPU.
"""

import pytest
import numpy as np


def test_device_caps_includes_descriptor_indexing():
    """Test that device capability reporting includes descriptor indexing fields"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    
    # Verify descriptor indexing fields are present
    expected_fields = {
        'descriptor_indexing', 'max_texture_array_layers',
        'max_sampler_array_size', 'vertex_shader_array_support'
    }
    
    actual_fields = set(device_info.keys())
    missing_fields = expected_fields - actual_fields
    
    assert not missing_fields, f"Missing descriptor indexing fields: {missing_fields}"
    
    # Verify field types
    assert isinstance(device_info['descriptor_indexing'], bool), "descriptor_indexing should be bool"
    assert isinstance(device_info['max_texture_array_layers'], int), "max_texture_array_layers should be int"
    assert isinstance(device_info['max_sampler_array_size'], int), "max_sampler_array_size should be int"
    assert isinstance(device_info['vertex_shader_array_support'], bool), "vertex_shader_array_support should be bool"
    
    # Verify reasonable limits
    assert device_info['max_texture_array_layers'] >= 1, "max_texture_array_layers should be >= 1"
    assert device_info['max_sampler_array_size'] >= 1, "max_sampler_array_size should be >= 1"
    
    print(f"[OK] Descriptor indexing: {device_info['descriptor_indexing']}")
    print(f"[OK] Max texture arrays: {device_info['max_texture_array_layers']}")
    print(f"[OK] Max sampler arrays: {device_info['max_sampler_array_size']}")
    print(f"[OK] Vertex array support: {device_info['vertex_shader_array_support']}")


def test_backend_specific_limits():
    """Test that different backends report appropriate descriptor indexing limits"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    backend = device_info['backend'].lower()
    
    # Backend-specific validation
    if backend == 'vulkan':
        if device_info['descriptor_indexing']:
            # Vulkan should have generous limits
            assert device_info['max_texture_array_layers'] >= 256, f"Vulkan should support >= 256 textures, got {device_info['max_texture_array_layers']}"
            assert device_info['vertex_shader_array_support'], "Vulkan should support vertex shader arrays"
    
    elif backend == 'metal':
        if device_info['descriptor_indexing']:
            # Metal has more conservative limits
            assert device_info['max_texture_array_layers'] >= 32, f"Metal should support >= 32 textures, got {device_info['max_texture_array_layers']}"
            assert device_info['vertex_shader_array_support'], "Metal should support vertex shader arrays"
    
    elif backend == 'dx12':
        if device_info['descriptor_indexing']:
            # DX12 variable support
            assert device_info['max_texture_array_layers'] >= 64, f"DX12 should support >= 64 textures, got {device_info['max_texture_array_layers']}"
    
    elif backend == 'gl':
        # OpenGL has limited support
        if device_info['descriptor_indexing']:
            assert device_info['max_texture_array_layers'] >= 16, f"GL should support >= 16 textures when indexing available"
            # GL often lacks vertex shader array support
            # This is backend-dependent so we don't assert it
    
    print(f"[OK] Backend-specific limits validated for {backend}")


def test_descriptor_indexing_runtime_detection():
    """Test that descriptor indexing detection works at runtime"""
    import forge3d as f3d
    
    # Test that we can detect descriptor indexing without crashing
    try:
        r = f3d.Renderer(128, 128)
        device_info = r.report_device()
        
        descriptor_support = device_info['descriptor_indexing']
        max_textures = device_info['max_texture_array_layers']
        
        # The values should be consistent
        if descriptor_support:
            assert max_textures > 1, "If descriptor indexing is supported, should allow > 1 texture"
        else:
            print("[INFO] Descriptor indexing not supported on this device - fallback mode will be used")
        
        print(f"[OK] Runtime detection successful: {descriptor_support}")
        
    except Exception as e:
        pytest.fail(f"Descriptor indexing detection failed: {e}")


def test_terrain_pipeline_creation():
    """Test that terrain pipeline creates successfully with descriptor indexing detection"""
    import forge3d as f3d
    
    r = f3d.Renderer(128, 128)
    device_info = r.report_device()
    
    # Test basic terrain rendering to ensure pipeline works
    # This will use either descriptor indexing or fallback internally
    try:
        # Create a simple heightmap
        heights = np.random.rand(64, 64).astype(np.float32) * 0.5
        
        # Upload height data
        r.upload_height_r32f(heights)
        
        # Render terrain
        rgba = r.render_terrain_rgba()
        
        # Verify output
        assert rgba.shape == (128, 128, 4), f"Expected (128,128,4), got {rgba.shape}"
        assert rgba.dtype == np.uint8, f"Expected uint8, got {rgba.dtype}"
        
        # Should have non-zero content (not all black)
        assert np.sum(rgba[:,:,0:3]) > 0, "Terrain render should have non-zero color content"
        
        print(f"[OK] Terrain pipeline works with descriptor indexing detection")
        print(f"[OK] Using {'descriptor indexing' if device_info['descriptor_indexing'] else 'fallback'} mode")
        
    except Exception as e:
        pytest.fail(f"Terrain pipeline creation with descriptor indexing failed: {e}")


def test_palette_switching_compatibility():
    """Test that palette switching works with both descriptor indexing and fallback modes"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    
    try:
        # Create a simple heightmap
        heights = np.random.rand(32, 32).astype(np.float32) * 0.3
        r.upload_height_r32f(heights)
        
        # Test palette switching
        available_palettes = f3d.list_palettes()
        assert len(available_palettes) > 1, "Should have multiple palettes available for testing"
        
        # Test switching between first two palettes
        palette1 = available_palettes[0]
        palette2 = available_palettes[1]
        
        # Render with first palette
        f3d.set_palette(palette1)
        current_palette = f3d.get_current_palette()
        assert current_palette == palette1, f"Palette should be {palette1}, got {current_palette}"
        
        rgba1 = r.render_terrain_rgba()
        assert rgba1.shape == (64, 64, 4), "First render should have correct shape"
        
        # Render with second palette
        f3d.set_palette(palette2)
        current_palette = f3d.get_current_palette()
        assert current_palette == palette2, f"Palette should be {palette2}, got {current_palette}"
        
        rgba2 = r.render_terrain_rgba()
        assert rgba2.shape == (64, 64, 4), "Second render should have correct shape"
        
        # Results should be different (different palettes)
        diff = np.sum(np.abs(rgba1.astype(np.int16) - rgba2.astype(np.int16)))
        assert diff > 0, "Different palettes should produce different colors"
        
        print(f"[OK] Palette switching works in {'descriptor indexing' if device_info['descriptor_indexing'] else 'fallback'} mode")
        print(f"[OK] Color difference between palettes: {diff}")
        
    except Exception as e:
        pytest.fail(f"Palette switching compatibility test failed: {e}")


def test_graceful_fallback():
    """Test that the system gracefully falls back when descriptor indexing is not available"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    
    if not device_info['descriptor_indexing']:
        print("[INFO] Testing fallback mode (descriptor indexing not supported)")
        
        # Should still work in fallback mode
        try:
            heights = np.random.rand(32, 32).astype(np.float32) * 0.4
            r.upload_height_r32f(heights)
            
            rgba = r.render_terrain_rgba()
            assert rgba.shape == (64, 64, 4), "Fallback mode should still work"
            assert np.sum(rgba[:,:,0:3]) > 0, "Should have visible content in fallback mode"
            
            print("[OK] Fallback mode works correctly")
            
        except Exception as e:
            pytest.fail(f"Fallback mode failed: {e}")
    else:
        print("[INFO] Descriptor indexing is supported - fallback test skipped")


@pytest.mark.gpu
def test_descriptor_indexing_performance():
    """Performance comparison between descriptor indexing and fallback modes (when available)"""
    import forge3d as f3d
    import time
    
    r = f3d.Renderer(256, 256)
    device_info = r.report_device()
    
    if not device_info['descriptor_indexing']:
        pytest.skip("Descriptor indexing not available - performance test skipped")
    
    try:
        # Create a larger heightmap for performance testing
        heights = np.random.rand(128, 128).astype(np.float32)
        r.upload_height_r32f(heights)
        
        # Warm up
        for _ in range(3):
            r.render_terrain_rgba()
        
        # Performance test: multiple palette switches
        available_palettes = f3d.list_palettes()
        if len(available_palettes) < 3:
            pytest.skip("Need at least 3 palettes for performance test")
        
        start_time = time.time()
        
        # Simulate rapid palette switching (this benefits from descriptor indexing)
        for i in range(10):
            palette_idx = i % len(available_palettes)
            f3d.set_palette(available_palettes[palette_idx])
            rgba = r.render_terrain_rgba()
            assert rgba.shape == (256, 256, 4), "Performance test render should work"
        
        elapsed_time = time.time() - start_time
        
        print(f"[PERF] 10 renders with palette switching took {elapsed_time:.3f}s")
        print(f"[PERF] Average per render: {elapsed_time/10:.3f}s")
        
        # Basic performance sanity check (should complete in reasonable time)
        assert elapsed_time < 5.0, f"Performance test took too long: {elapsed_time:.3f}s"
        
    except Exception as e:
        pytest.fail(f"Performance test failed: {e}")


if __name__ == "__main__":
    test_device_caps_includes_descriptor_indexing()
    test_backend_specific_limits()
    test_descriptor_indexing_runtime_detection()
    test_terrain_pipeline_creation()
    test_palette_switching_compatibility()
    test_graceful_fallback()
    print("[OK] All L3 descriptor indexing tests passed")