import numpy as np
import pytest

from forge3d import thematic


def test_equal_interval_classify_returns_bins_and_classes():
    result = thematic.classify([0, 5, 10, 15], scheme="equal_interval", k=3)

    np.testing.assert_allclose(result["bins"], np.array([5.0, 10.0], dtype=np.float64))
    np.testing.assert_array_equal(result["classes"], np.array([1, 2, 3, 3], dtype=np.uint16))
    assert result["scheme"] == "equal_interval"
    assert result["valid_count"] == 4


def test_quantile_classify_uses_distribution_breaks():
    result = thematic.classify([1, 2, 3, 100], scheme="quantile", k=2)

    np.testing.assert_allclose(result["bins"], np.array([2.5], dtype=np.float64))
    np.testing.assert_array_equal(result["classes"], np.array([1, 1, 2, 2], dtype=np.uint16))


def test_jenks_classify_finds_natural_gap():
    result = thematic.classify([1, 2, 2, 3, 4, 10, 11, 12, 12], scheme="jenks", k=2)

    np.testing.assert_allclose(result["bins"], np.array([4.0], dtype=np.float64))
    np.testing.assert_array_equal(result["classes"], np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]))


def test_classify_preserves_shape_and_nodata_zero_class():
    values = np.array([[1.0, -9999.0], [np.nan, 9.0]], dtype=np.float32)

    result = thematic.classify(values, scheme="equal_interval", k=2, nodata=-9999.0)

    assert result["classes"].shape == values.shape
    np.testing.assert_array_equal(result["classes"], np.array([[1, 0], [0, 2]], dtype=np.uint16))
    assert result["nodata_count"] == 2


def test_apply_palette_binds_class_ids_to_rgba():
    classes = np.array([[0, 1, 2]], dtype=np.uint16)
    colors = [(255, 0, 0), (0, 255, 0, 128)]

    rgba = thematic.apply_palette(classes, colors)

    np.testing.assert_array_equal(
        rgba,
        np.array([[[0, 0, 0, 0], [255, 0, 0, 255], [0, 255, 0, 128]]], dtype=np.uint8),
    )


def test_classify_rejects_bad_inputs():
    with pytest.raises(ValueError, match="k"):
        thematic.classify([1, 2, 3], k=1)
    with pytest.raises(ValueError, match="scheme"):
        thematic.classify([1, 2, 3], scheme="bad", k=2)
    with pytest.raises(ValueError, match="empty"):
        thematic.classify([np.nan], k=2)
