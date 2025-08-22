#!/usr/bin/env python3
"""
C3 Device capabilities and MSAA gating tests
"""

def test_report_device_method():
    """Test that Renderer.report_device() method exists and returns expected fields"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    
    # Verify expected fields are present
    expected_fields = {
        'backend', 'adapter_name', 'device_name', 
        'max_texture_dimension_2d', 'max_buffer_size',
        'msaa_supported', 'max_samples', 'device_type'
    }
    
    actual_fields = set(device_info.keys())
    missing_fields = expected_fields - actual_fields
    
    assert not missing_fields, f"Missing fields in report_device(): {missing_fields}"
    
    # Verify field types
    assert isinstance(device_info['backend'], str), f"backend should be str, got {type(device_info['backend'])}"
    assert isinstance(device_info['adapter_name'], str), f"adapter_name should be str"
    assert isinstance(device_info['msaa_supported'], bool), f"msaa_supported should be bool"
    assert isinstance(device_info['max_samples'], int), f"max_samples should be int"
    assert device_info['max_samples'] >= 1, f"max_samples should be >= 1, got {device_info['max_samples']}"
    
    print(f"[OK] Device report: {device_info['backend']}, MSAA: {device_info['msaa_supported']} (max: {device_info['max_samples']})")

def test_device_consistency_with_probe():
    """Test that report_device() is consistent with existing device_probe()"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    probe_info = f3d.device_probe()
    
    # These should be consistent (though probe might have more fields)
    if 'backend' in probe_info:
        assert device_info['backend'] in probe_info.get('backend', '').lower(), \
            f"Backend mismatch: {device_info['backend']} not in {probe_info.get('backend', '')}"
    
    # Verify both return successful status types
    assert isinstance(device_info, dict), "report_device() should return dict"
    assert 'status' in probe_info, "device_probe() should have status field"
    
    print(f"[OK] Consistency check passed between report_device() and device_probe()")

def test_msaa_gating():
    """Test that MSAA is properly gated based on device capabilities"""
    import forge3d as f3d
    
    r = f3d.Renderer(64, 64)
    device_info = r.report_device()
    
    msaa_supported = device_info['msaa_supported']
    max_samples = device_info['max_samples']
    
    if msaa_supported:
        # If MSAA is supported, max_samples should be > 1
        assert max_samples > 1, f"MSAA supported but max_samples={max_samples} <= 1"
        assert max_samples in [2, 4, 8], f"Unexpected max_samples: {max_samples}"
        print(f"[OK] MSAA gating: Supported with max_samples={max_samples}")
    else:
        # If MSAA not supported, max_samples should be 1
        assert max_samples == 1, f"MSAA not supported but max_samples={max_samples} != 1"
        print(f"[OK] MSAA gating: Not supported, max_samples={max_samples}")

def test_renderer_builds_without_msaa():
    """Test that renderer can initialize successfully regardless of MSAA support"""
    import forge3d as f3d
    
    # This should work on any device (MSAA gated automatically)
    r = f3d.Renderer(128, 128)
    device_info = r.report_device()
    
    # Render something to verify it works
    rgba = r.render_triangle_rgba()
    assert rgba.shape == (128, 128, 4), f"Expected (128,128,4), got {rgba.shape}"
    
    print(f"[OK] Renderer works without MSAA issues on {device_info['device_type']} device")

if __name__ == "__main__":
    test_report_device_method()
    test_device_consistency_with_probe()
    test_msaa_gating()
    test_renderer_builds_without_msaa()
    print("[OK] All C3 device capabilities tests passed")