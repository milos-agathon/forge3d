# tests/test_animation_mvp.py
"""
Tests for Feature C: Camera Animation MVP (Plan 1)

Tests keyframe interpolation, animation evaluation, and render queue configuration.
"""

import pytest
import tempfile
from pathlib import Path


class TestCameraAnimation:
    """Tests for CameraAnimation class."""

    def test_create_empty_animation(self):
        """Empty animation has zero duration and no keyframes."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        assert anim.keyframe_count == 0
        assert anim.duration == 0.0

    def test_add_single_keyframe(self):
        """Single keyframe sets duration to keyframe time."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=2.0, phi=45.0, theta=30.0, radius=1000.0, fov=60.0)
        
        assert anim.keyframe_count == 1
        assert anim.duration == 2.0

    def test_add_multiple_keyframes_sorted(self):
        """Keyframes are sorted by time automatically."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=5.0, phi=180.0, theta=30.0, radius=3000.0, fov=60.0)
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=5000.0, fov=60.0)
        anim.add_keyframe(time=2.5, phi=90.0, theta=35.0, radius=4000.0, fov=60.0)
        
        assert anim.keyframe_count == 3
        assert anim.duration == 5.0

    def test_evaluate_single_keyframe(self):
        """Evaluating single keyframe returns that keyframe's values."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=45.0, theta=30.0, radius=1000.0, fov=60.0)
        
        state = anim.evaluate(0.0)
        assert state is not None
        assert abs(state.phi_deg - 45.0) < 0.1
        assert abs(state.theta_deg - 30.0) < 0.1
        assert abs(state.radius - 1000.0) < 0.1
        assert abs(state.fov_deg - 60.0) < 0.1

    def test_evaluate_at_keyframe_times(self):
        """Evaluation at keyframe times returns exact keyframe values."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=5000.0, fov=60.0)
        anim.add_keyframe(time=1.0, phi=90.0, theta=45.0, radius=5000.0, fov=60.0)
        
        # At t=0
        state = anim.evaluate(0.0)
        assert abs(state.phi_deg - 0.0) < 0.1
        
        # At t=1
        state = anim.evaluate(1.0)
        assert abs(state.phi_deg - 90.0) < 0.1

    def test_interpolation_midpoint(self):
        """Interpolation produces correct intermediate values."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=1.0, phi=90.0, theta=45.0, radius=1000.0, fov=60.0)
        
        state = anim.evaluate(0.5)
        # Midpoint should be approximately 45 degrees (cubic Hermite may vary slightly)
        assert abs(state.phi_deg - 45.0) < 5.0  # Allow some tolerance for cubic interpolation

    def test_interpolation_smooth(self):
        """Cubic Hermite interpolation is smooth (monotonic for linear input)."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=1.0, phi=100.0, theta=45.0, radius=1000.0, fov=60.0)
        
        # Sample at multiple points
        prev_phi = -1.0
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            state = anim.evaluate(t)
            assert state.phi_deg > prev_phi, f"phi should increase monotonically at t={t}"
            prev_phi = state.phi_deg

    def test_evaluate_clamped_to_bounds(self):
        """Evaluation outside animation bounds is clamped."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=1.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=2.0, phi=90.0, theta=45.0, radius=1000.0, fov=60.0)
        
        # Before start
        state = anim.evaluate(0.0)
        assert abs(state.phi_deg - 0.0) < 0.1
        
        # After end
        state = anim.evaluate(10.0)
        assert abs(state.phi_deg - 90.0) < 0.1

    def test_frame_count_calculation(self):
        """Frame count is calculated correctly from duration and fps."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=1.0, phi=90.0, theta=45.0, radius=1000.0, fov=60.0)
        
        # 1 second at 30 fps = 30 frames + 1 = 31 frames (inclusive of start and end)
        assert anim.get_frame_count(30) == 31
        
        # 1 second at 60 fps = 60 frames + 1 = 61 frames
        assert anim.get_frame_count(60) == 61

    def test_repr(self):
        """String representation shows keyframe count and duration."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=5.0, phi=180.0, theta=30.0, radius=3000.0, fov=60.0)
        
        repr_str = repr(anim)
        assert "2" in repr_str or "keyframes=2" in repr_str
        assert "5" in repr_str  # duration


class TestCameraState:
    """Tests for CameraState class."""

    def test_camera_state_attributes(self):
        """CameraState has expected attributes."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=45.0, theta=30.0, radius=1000.0, fov=60.0)
        
        state = anim.evaluate(0.0)
        assert hasattr(state, 'phi_deg')
        assert hasattr(state, 'theta_deg')
        assert hasattr(state, 'radius')
        assert hasattr(state, 'fov_deg')

    def test_camera_state_repr(self):
        """CameraState has readable string representation."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=45.0, theta=30.0, radius=1000.0, fov=60.0)
        
        state = anim.evaluate(0.0)
        repr_str = repr(state)
        assert "45" in repr_str or "phi" in repr_str.lower()


class TestRenderConfig:
    """Tests for RenderConfig class."""

    def test_default_config(self):
        """Default config has sensible values."""
        from forge3d.animation import RenderConfig
        
        config = RenderConfig()
        assert config.fps == 30
        assert config.width == 1920
        assert config.height == 1080
        assert config.filename_prefix == "frame"
        assert config.frame_digits == 4

    def test_custom_config(self):
        """Custom config values are stored correctly."""
        from forge3d.animation import RenderConfig
        
        config = RenderConfig(
            output_dir="/tmp/test_frames",
            fps=60,
            width=3840,
            height=2160,
        )
        assert config.fps == 60
        assert config.width == 3840
        assert config.height == 2160
        assert config.output_dir == Path("/tmp/test_frames")

    def test_frame_path_generation(self):
        """Frame paths are generated with correct formatting."""
        from forge3d.animation import RenderConfig
        
        config = RenderConfig(output_dir="/tmp/frames", frame_digits=4)
        
        path = config.frame_path(0)
        assert path.name == "frame_0000.png"
        
        path = config.frame_path(42)
        assert path.name == "frame_0042.png"
        
        path = config.frame_path(9999)
        assert path.name == "frame_9999.png"

    def test_ensure_output_dir(self):
        """Output directory is created if it doesn't exist."""
        from forge3d.animation import RenderConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            config = RenderConfig(output_dir=str(output_dir))
            
            assert not output_dir.exists()
            config.ensure_output_dir()
            assert output_dir.exists()


class TestRenderProgress:
    """Tests for RenderProgress class."""

    def test_progress_percent(self):
        """Progress percentage is calculated correctly."""
        from forge3d.animation import RenderProgress
        
        progress = RenderProgress(
            frame=50,
            total_frames=100,
            time=1.67,
            output_path=Path("/tmp/frame_0050.png"),
        )
        assert abs(progress.percent - 0.5) < 0.001

    def test_progress_zero_frames(self):
        """Zero total frames returns 0% progress."""
        from forge3d.animation import RenderProgress
        
        progress = RenderProgress(
            frame=0,
            total_frames=0,
            time=0.0,
            output_path=Path("/tmp/frame_0000.png"),
        )
        assert progress.percent == 0.0


class TestDeterminism:
    """Tests for deterministic animation evaluation."""

    def test_same_animation_same_result(self):
        """Same animation evaluated at same time produces identical results."""
        from forge3d.animation import CameraAnimation
        
        def create_animation():
            anim = CameraAnimation()
            anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=5000.0, fov=60.0)
            anim.add_keyframe(time=2.5, phi=90.0, theta=30.0, radius=3000.0, fov=60.0)
            anim.add_keyframe(time=5.0, phi=180.0, theta=45.0, radius=5000.0, fov=60.0)
            return anim
        
        anim1 = create_animation()
        anim2 = create_animation()
        
        for t in [0.0, 1.0, 2.5, 3.5, 5.0]:
            state1 = anim1.evaluate(t)
            state2 = anim2.evaluate(t)
            
            assert abs(state1.phi_deg - state2.phi_deg) < 1e-6
            assert abs(state1.theta_deg - state2.theta_deg) < 1e-6
            assert abs(state1.radius - state2.radius) < 1e-6
            assert abs(state1.fov_deg - state2.fov_deg) < 1e-6


class TestMultipleKeyframes:
    """Tests for animations with many keyframes."""

    def test_many_keyframes(self):
        """Animation with many keyframes interpolates correctly."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        # Add 10 keyframes at regular intervals
        for i in range(10):
            t = i * 1.0
            phi = i * 36.0  # 0, 36, 72, ... 324 degrees
            anim.add_keyframe(time=t, phi=phi, theta=45.0, radius=1000.0, fov=60.0)
        
        assert anim.keyframe_count == 10
        assert anim.duration == 9.0
        
        # Check some intermediate points
        state = anim.evaluate(4.5)  # Between keyframes 4 and 5
        # phi should be between 144 and 180
        assert 140.0 < state.phi_deg < 185.0

    def test_four_keyframe_spline(self):
        """Four keyframes produce smooth Catmull-Rom spline."""
        from forge3d.animation import CameraAnimation
        
        anim = CameraAnimation()
        anim.add_keyframe(time=0.0, phi=0.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=1.0, phi=30.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=2.0, phi=60.0, theta=45.0, radius=1000.0, fov=60.0)
        anim.add_keyframe(time=3.0, phi=90.0, theta=45.0, radius=1000.0, fov=60.0)
        
        # Sample middle segment
        samples = [anim.evaluate(1.0 + t * 0.1).phi_deg for t in range(11)]
        
        # Check monotonic increase
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i-1] - 0.1  # Allow tiny numerical error
