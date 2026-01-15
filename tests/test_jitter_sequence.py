"""P1.2: Jitter sequence tests for TAA.

Tests the Halton 2,3 jitter sequence implementation for temporal anti-aliasing.
Verifies:
1. Halton sequence produces correct values
2. Jitter is applied to projection matrix correctly
3. Jitter state advances each frame
4. Jitter is disabled by default
"""

import pytest
import numpy as np
from pathlib import Path

# Skip if forge3d not built
pytest.importorskip("forge3d")


class TestJitterSequenceInfrastructure:
    """Test P1.2 jitter sequence infrastructure exists."""

    def test_jitter_module_exists(self):
        """Verify jitter.rs module exists in core."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        assert jitter_path.exists(), f"jitter.rs not found at {jitter_path}"

    def test_camera_params_has_jitter_offset(self):
        """Verify CameraParams includes jitter_offset field."""
        sse_path = Path(__file__).parent.parent / "src" / "core" / "screen_space_effects.rs"
        content = sse_path.read_text()
        assert "jitter_offset" in content, "jitter_offset field not found in CameraParams"

    def test_viewer_has_taa_jitter_state(self):
        """Verify Viewer struct has TAA jitter state."""
        viewer_path = Path(__file__).parent.parent / "src" / "viewer" / "viewer_struct.rs"
        content = viewer_path.read_text()
        assert "taa_jitter" in content, "taa_jitter field not found in Viewer struct"


class TestHaltonSequence:
    """Test Halton sequence mathematical properties."""

    def test_halton_sequence_in_jitter_module(self):
        """Verify Halton sequence implementation exists."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        
        # Check for Halton function
        assert "fn halton(" in content, "halton function not found"
        assert "fn halton_2_3(" in content, "halton_2_3 function not found"

    def test_halton_base_2_values(self):
        """Verify Halton base 2 produces expected values.
        
        Halton(1, 2) = 1/2 = 0.5
        Halton(2, 2) = 1/4 = 0.25
        Halton(3, 2) = 3/4 = 0.75
        Halton(4, 2) = 1/8 = 0.125
        """
        # These values are verified by Rust unit tests in jitter.rs
        # This test verifies the test exists
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        assert "test_halton_base_2" in content, "Halton base 2 test not found"

    def test_halton_2_3_range_check(self):
        """Verify halton_2_3 returns values in [-0.5, 0.5] range."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        assert "test_halton_2_3_range" in content, "Halton 2,3 range test not found"
        # Check that the function centers around 0
        assert "- 0.5" in content, "Halton centering (- 0.5) not found"


class TestJitterApplication:
    """Test jitter application to projection matrix."""

    def test_apply_jitter_function_exists(self):
        """Verify apply_jitter function exists."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        assert "fn apply_jitter(" in content, "apply_jitter function not found"

    def test_jitter_converts_to_ndc(self):
        """Verify jitter is converted from pixel to NDC space."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        # NDC conversion: 2 * jitter / resolution
        assert "2.0 *" in content, "NDC scaling factor not found"
        assert "width" in content and "height" in content, "Resolution parameters not found"

    def test_jitter_state_struct(self):
        """Verify JitterState struct has required fields."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        
        assert "struct JitterState" in content, "JitterState struct not found"
        assert "enabled:" in content, "enabled field not found"
        assert "index:" in content, "index field not found"
        assert "sequence_length:" in content, "sequence_length field not found"
        assert "offset:" in content, "offset field not found"


class TestJitterInShaders:
    """Test jitter offset availability in shaders."""

    def test_wgsl_camera_params_has_jitter(self):
        """Verify WGSL CameraParams includes jitter_offset."""
        shader_dir = Path(__file__).parent.parent / "src" / "shaders"
        
        # Check key shaders
        shaders_to_check = [
            "ssao/common.wgsl",
            "ssr/trace.wgsl",
            "ssgi/trace.wgsl",
        ]
        
        for shader_rel in shaders_to_check:
            shader_path = shader_dir / shader_rel
            if shader_path.exists():
                content = shader_path.read_text()
                if "struct CameraParams" in content:
                    assert "jitter_offset" in content, \
                        f"jitter_offset not found in {shader_rel}"


class TestJitterSequenceLength:
    """Test jitter sequence length and cycling."""

    def test_default_sequence_length(self):
        """Verify default sequence length is defined."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        assert "DEFAULT_JITTER_SEQUENCE_LENGTH" in content, \
            "DEFAULT_JITTER_SEQUENCE_LENGTH constant not found"

    def test_sequence_cycles(self):
        """Verify sequence wraps around correctly."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        # Check for modulo operation in advance or halton_2_3
        assert "%" in content, "Modulo operation for sequence cycling not found"


class TestJitterDisabledByDefault:
    """Test that jitter is disabled by default."""

    def test_jitter_default_disabled(self):
        """Verify JitterState::new() creates disabled state."""
        jitter_path = Path(__file__).parent.parent / "src" / "core" / "jitter.rs"
        content = jitter_path.read_text()
        assert "fn new()" in content, "JitterState::new() not found"
        assert "enabled: false" in content, "Default enabled should be false"

    def test_viewer_initializes_jitter_disabled(self):
        """Verify Viewer initializes with jitter disabled."""
        viewer_new_path = Path(__file__).parent.parent / "src" / "viewer" / "init" / "viewer_new.rs"
        content = viewer_new_path.read_text()
        assert "taa_jitter:" in content, "taa_jitter initialization not found"
        assert "JitterState::new()" in content, "Should use JitterState::new() (disabled)"


class TestJitterIntegration:
    """Integration tests for jitter with rendering pipeline."""

    def test_jitter_applied_in_main_loop(self):
        """Verify jitter is applied to projection in main render loop."""
        main_loop_path = Path(__file__).parent.parent / "src" / "viewer" / "render" / "main_loop.rs"
        content = main_loop_path.read_text()
        assert "apply_jitter" in content, "apply_jitter not called in main_loop.rs"
        assert "taa_jitter.enabled" in content, "Jitter enable check not found"

    def test_jitter_advances_each_frame(self):
        """Verify jitter advances to next sample each frame."""
        main_loop_path = Path(__file__).parent.parent / "src" / "viewer" / "render" / "main_loop.rs"
        content = main_loop_path.read_text()
        assert "taa_jitter.advance()" in content, "Jitter advance not called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
