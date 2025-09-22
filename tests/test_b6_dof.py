import numpy as np
import pytest

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover - tests rely on python fallback
    pytest.skip(str(exc), allow_module_level=True)


def _make_scene(size: int = 128) -> f3d.Scene:
    scene = f3d.Scene(size, size, grid=64)
    coords = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    heightmap = np.outer(np.sin(coords * np.pi), np.cos(coords * np.pi)).astype(np.float32)
    scene.set_height_from_r32f(heightmap)
    return scene


def _luminance_std(img: np.ndarray, row_slice: slice) -> float:
    region = img[row_slice, :, :3].astype(np.float32)
    grey = region.mean(axis=2)
    return float(np.std(grey))


class TestDepthOfField:
    def test_enable_disable(self) -> None:
        scene = _make_scene(96)
        scene.set_msaa_samples(4)
        scene.enable_dof('medium')
        assert scene.dof_enabled() is True
        scene.disable_dof()
        assert scene.dof_enabled() is False

    def test_aperture_controls_blur(self) -> None:
        scene = _make_scene(128)
        scene.set_msaa_samples(4)
        scene.enable_dof('medium')
        scene.set_dof_focus_distance(0.6)
        scene.set_dof_focal_length(50.0)

        scene.set_dof_f_stop(11.0)
        sharp = scene.render_rgba()

        scene.set_dof_f_stop(2.0)
        shallow = scene.render_rgba()

        top_slice = slice(0, 32)
        sharp_std = _luminance_std(sharp, top_slice)
        shallow_std = _luminance_std(shallow, top_slice)
        assert shallow_std < sharp_std * 0.85

    def test_focus_distance_shifts_blur_region(self) -> None:
        scene = _make_scene(128)
        scene.set_msaa_samples(4)
        scene.enable_dof('medium')
        scene.set_dof_f_stop(2.0)

        scene.set_dof_focus_distance(0.25)
        near_focused = scene.render_rgba()

        scene.set_dof_focus_distance(0.85)
        far_focused = scene.render_rgba()

        top_slice = slice(0, 32)
        bottom_slice = slice(-32, None)

        near_top = _luminance_std(near_focused, top_slice)
        near_bottom = _luminance_std(near_focused, bottom_slice)
        far_top = _luminance_std(far_focused, top_slice)
        far_bottom = _luminance_std(far_focused, bottom_slice)

        assert near_top > near_bottom * 1.05
        assert far_bottom > far_top * 1.05

    def test_show_coc_and_method_toggle(self) -> None:
        scene = _make_scene(96)
        scene.set_msaa_samples(4)
        scene.enable_dof('high')
        scene.set_dof_f_stop(2.8)
        scene.set_dof_focus_distance(0.6)

        baseline = scene.render_rgba()
        scene.set_dof_show_coc(True)
        with_coc = scene.render_rgba()
        assert not np.array_equal(baseline, with_coc)

        scene.set_dof_show_coc(False)
        scene.set_dof_method('separable')
        separable = scene.render_rgba()
        assert separable.shape == baseline.shape

    def test_invalid_quality_rejected(self) -> None:
        scene = _make_scene(64)
        scene.set_msaa_samples(4)
        with pytest.raises(ValueError):
            scene.enable_dof('invalid-quality')
