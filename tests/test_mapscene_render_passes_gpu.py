"""Physical D04 example gate; existing required-terrain environment is honored."""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from forge3d._png import load_png_rgba

from _terrain_runtime import terrain_rendering_available


@pytest.mark.gpu_lane
@pytest.mark.skipif(not terrain_rendering_available(), reason="physical terrain renderer unavailable")
def test_actual_render_outputs_compose_and_replay(tmp_path):
    path = Path(__file__).resolve().parents[1] / "docs/examples/mapscene_render_passes.py"
    spec = importlib.util.spec_from_file_location("d04_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    result = example.compose_example(tmp_path)
    assert result["replay_pixel_and_report_equal"]
    assert result["composition_backend"] == "python_ordered_rgba_composition"
    assert result["source_metrics"]["color"]["backend"] == "gpu_terrain"
    assert result["source_metrics"]["relief"]["backend"] == "gpu_terrain"
    assert result["sha256"]["color"] != result["sha256"]["relief"]
    assert result["sha256"]["composed"] == result["sha256"]["replay"]
    # Independent source-over reference from the actual GPU PNG inputs. This
    # does not call the product's compiler, blend helpers or PNG writer.
    color = np.asarray(load_png_rgba(tmp_path / 'color.png'), dtype=np.float64) / 255
    relief = np.asarray(load_png_rgba(tmp_path / 'relief.png'), dtype=np.float64) / 255
    vectors = np.asarray(load_png_rgba(tmp_path / 'overlay.png'), dtype=np.float64) / 255

    def linear(rgb):
        return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

    def encoded(rgb):
        return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1/2.4) - .055)

    cb, cr = linear(color[..., :3]), linear(relief[..., :3])
    ab, ar = color[..., 3:4], relief[..., 3:4] * .5
    relief_alpha = ar + ab * (1 - ar)
    relief_premult = cb * ab * (1 - ar) + cr * ar * (1 - ab) + cb * cr * ar * ab
    relief_rgb = np.divide(relief_premult, relief_alpha, out=np.zeros_like(cb), where=relief_alpha > 0)
    av = vectors[..., 3:4]
    final_alpha = av + relief_alpha * (1 - av)
    final_premult = vectors[..., :3] * av + relief_premult * (1 - av)
    final_rgb = np.divide(final_premult, final_alpha, out=np.zeros_like(cb), where=final_alpha > 0)
    expected = (np.clip(np.concatenate((encoded(final_rgb), final_alpha), axis=2), 0, 1) * 255 + .5).astype(np.uint8)
    actual = np.asarray(load_png_rgba(tmp_path / 'composed.png'))
    np.testing.assert_array_equal(actual, expected)
    relief_only = (np.clip(np.concatenate((encoded(relief_rgb), relief_alpha), axis=2), 0, 1) * 255 + .5).astype(np.uint8)
    assert np.any(relief_only != np.asarray(load_png_rgba(tmp_path / 'color.png')))
    assert np.any(actual != relief_only)
    # The raw Rgba8Unorm vector target quantizes the supplied linear RGB
    # directly: an sRGB transfer would yield different channel values.
    covered = vectors[..., 3] > 0
    vector_rgb = np.asarray(load_png_rgba(tmp_path / 'overlay.png'))[..., :3][covered]
    np.testing.assert_array_equal(vector_rgb, np.broadcast_to([13, 102, 255], vector_rgb.shape))
