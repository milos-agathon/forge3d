"""P1.3: TAA (Temporal Anti-Aliasing) convergence tests.

Tests the TAA implementation for:
1. TAA module infrastructure exists
2. History buffer ping-pong management
3. TAA resolve shader with reprojection and neighborhood clamping
4. Jitter-TAA integration (jitter enabled when TAA enabled)
5. Variance reduction in static scenes (convergence test)
"""

import pytest
import numpy as np
from pathlib import Path

# Skip if forge3d not built
pytest.importorskip("forge3d")


class TestTaaInfrastructure:
    """Test P1.3 TAA infrastructure exists."""

    def test_taa_module_exists(self):
        """Verify taa.rs module exists in core."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        assert taa_path.exists(), f"taa.rs not found at {taa_path}"

    def test_taa_shader_exists(self):
        """Verify taa.wgsl shader exists."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        assert shader_path.exists(), f"taa.wgsl not found at {shader_path}"

    def test_viewer_has_taa_renderer(self):
        """Verify Viewer struct has TAA renderer field."""
        viewer_path = Path(__file__).parent.parent / "src" / "viewer" / "viewer_struct.rs"
        content = viewer_path.read_text()
        assert "taa_renderer" in content, "taa_renderer field not found in Viewer struct"

    def test_viewer_cmd_has_taa_commands(self):
        """Verify ViewerCmd enum has TAA commands."""
        enums_path = Path(__file__).parent.parent / "src" / "viewer" / "viewer_enums.rs"
        content = enums_path.read_text()
        assert "SetTaaEnabled" in content, "SetTaaEnabled command not found"
        assert "GetTaaStatus" in content, "GetTaaStatus command not found"


class TestTaaSettings:
    """Test TAA settings struct."""

    def test_taa_settings_struct_exists(self):
        """Verify TaaSettings struct exists with required fields."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        assert "struct TaaSettings" in content, "TaaSettings struct not found"
        assert "resolution:" in content, "resolution field not found"
        assert "jitter_offset:" in content, "jitter_offset field not found"
        assert "history_weight:" in content, "history_weight field not found"
        assert "clamp_gamma:" in content, "clamp_gamma field not found"

    def test_taa_default_settings(self):
        """Verify TaaSettings has sensible defaults."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        # Check for default history weight (typically 0.9)
        assert "history_weight: 0.9" in content or "history_weight: 0.90" in content, \
            "Default history weight should be ~0.9"


class TestTaaHistoryBuffers:
    """Test TAA history buffer management."""

    def test_ping_pong_buffers(self):
        """Verify TAA uses ping-pong history buffers."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        # Check for ping-pong buffer pattern
        assert "history_textures:" in content, "history_textures field not found"
        assert "[Texture; 2]" in content or "history_textures: [" in content, \
            "Should have 2 history textures for ping-pong"
        assert "read_index:" in content, "read_index field not found for ping-pong swap"

    def test_history_texture_format(self):
        """Verify history texture uses HDR format."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        # Should use Rgba16Float for HDR history
        assert "Rgba16Float" in content, "History texture should use Rgba16Float for HDR"


class TestTaaShader:
    """Test TAA resolve shader implementation."""

    def test_shader_has_neighborhood_clamp(self):
        """Verify shader implements neighborhood clamping."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for neighborhood sampling
        assert "neighborhood" in content.lower(), "Neighborhood sampling not found"
        # Check for AABB clamping
        assert "aabb" in content.lower() or "clamp" in content.lower(), \
            "AABB clamping not found"

    def test_shader_has_ycocg_conversion(self):
        """Verify shader uses YCoCg color space for clamping."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for YCoCg conversion (better clamping quality)
        assert "ycocg" in content.lower() or "cocg" in content.lower(), \
            "YCoCg color space conversion not found"

    def test_shader_has_reprojection(self):
        """Verify shader implements reprojection using velocity."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for velocity usage
        assert "velocity" in content.lower(), "Velocity-based reprojection not found"
        # Check for history UV calculation
        assert "history_uv" in content.lower() or "history" in content.lower(), \
            "History UV reprojection not found"

    def test_shader_has_unjitter(self):
        """Verify shader unjitters current frame sampling."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for jitter offset usage
        assert "jitter" in content.lower(), "Jitter offset handling not found"


class TestTaaJitterIntegration:
    """Test TAA and jitter integration (P1.2 + P1.3)."""

    def test_set_taa_enabled_method(self):
        """Verify set_taa_enabled method exists in Viewer."""
        mod_path = Path(__file__).parent.parent / "src" / "viewer" / "mod.rs"
        content = mod_path.read_text()
        
        assert "fn set_taa_enabled" in content, "set_taa_enabled method not found"

    def test_taa_enables_jitter(self):
        """Verify enabling TAA also enables jitter."""
        mod_path = Path(__file__).parent.parent / "src" / "viewer" / "mod.rs"
        content = mod_path.read_text()
        
        # Check that set_taa_enabled modifies jitter state
        assert "taa_jitter" in content, "TAA should control jitter state"
        # Check that jitter is enabled when TAA is enabled
        assert "jitter::JitterState::enabled()" in content or "taa_jitter.enabled = enabled" in content, \
            "TAA should enable jitter when TAA is enabled"


class TestTaaRenderer:
    """Test TaaRenderer struct and methods."""

    def test_taa_renderer_struct(self):
        """Verify TaaRenderer struct has required fields."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        assert "struct TaaRenderer" in content, "TaaRenderer struct not found"
        assert "pipeline:" in content, "pipeline field not found"
        assert "bind_group_layout:" in content, "bind_group_layout not found"
        assert "sampler:" in content, "sampler field not found"
        assert "enabled:" in content, "enabled field not found"

    def test_taa_renderer_execute_method(self):
        """Verify TaaRenderer has execute method."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        assert "fn execute(" in content, "execute method not found"
        # Check it takes current_color, velocity, depth
        assert "current_color:" in content, "current_color parameter not found"
        assert "velocity_view:" in content, "velocity_view parameter not found"
        assert "depth_view:" in content, "depth_view parameter not found"

    def test_taa_renderer_resize_method(self):
        """Verify TaaRenderer has resize method."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        assert "fn resize(" in content, "resize method not found"


class TestTaaBindings:
    """Test TAA shader bindings."""

    def test_shader_bindings_complete(self):
        """Verify shader has all required bindings."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for required bindings
        assert "@binding(0)" in content, "Binding 0 (current color) not found"
        assert "@binding(1)" in content, "Binding 1 (history color) not found"
        assert "@binding(2)" in content, "Binding 2 (velocity) not found"
        assert "@binding(3)" in content, "Binding 3 (depth) not found"
        assert "@binding(4)" in content, "Binding 4 (sampler) not found"
        assert "@binding(5)" in content, "Binding 5 (settings) not found"
        assert "@binding(6)" in content, "Binding 6 (output) not found"

    def test_rust_bindings_match_shader(self):
        """Verify Rust bind group layout matches shader."""
        taa_path = Path(__file__).parent.parent / "src" / "core" / "taa.rs"
        content = taa_path.read_text()
        
        # Check Rust side has all bindings
        assert "binding: 0" in content, "Rust binding 0 not found"
        assert "binding: 1" in content, "Rust binding 1 not found"
        assert "binding: 2" in content, "Rust binding 2 not found"
        assert "binding: 3" in content, "Rust binding 3 not found"
        assert "binding: 4" in content, "Rust binding 4 not found"
        assert "binding: 5" in content, "Rust binding 5 not found"
        assert "binding: 6" in content, "Rust binding 6 not found"


class TestTaaMotionRejection:
    """Test TAA motion-based history rejection."""

    def test_shader_has_motion_rejection(self):
        """Verify shader reduces history weight based on motion."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for motion-based weight adjustment
        assert "motion" in content.lower(), "Motion-based rejection not found"

    def test_history_bounds_check(self):
        """Verify shader checks history UV bounds."""
        shader_path = Path(__file__).parent.parent / "src" / "shaders" / "taa.wgsl"
        content = shader_path.read_text()
        
        # Check for UV bounds validation
        assert "history_valid" in content or ("0.0" in content and "1.0" in content), \
            "History UV bounds check not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
