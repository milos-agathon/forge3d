import numpy as np
import pytest

from forge3d.colors import apply_luminance_preserving_tint, environment_mood_tint

W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def test_horizon_band_uses_only_central_rows():
    env = np.zeros((64, 8, 3), dtype=np.float64)
    env[:, :, 2] = 10.0
    env[27:37, :, :] = 0.0
    env[27:37, :, 0] = 2.0
    tint = environment_mood_tint(env)
    mean = env[27:37, :, :3].mean(axis=(0, 1))
    expected = np.clip(mean / float(mean @ W), 1.0 / 1.25, 1.25)
    assert np.allclose(tint, expected)
    assert tint[0] > tint[2]


def test_band_boundaries_are_exact():
    env = np.zeros((64, 4, 3), dtype=np.float64)
    env[27:37, :, :] = 1.0
    env[26, :, 0] = 1000.0
    env[37, :, 2] = 1000.0
    assert np.allclose(environment_mood_tint(env), [1.0, 1.0, 1.0])


def test_near_black_band_returns_identity():
    env = np.zeros((64, 4, 3), dtype=np.float64)
    env[27:37, :, :] = 1e-15
    assert np.array_equal(environment_mood_tint(env), np.array([1.0, 1.0, 1.0]))


def test_cool_warm_brown_ordering():
    def tint(rgb):
        env = np.zeros((64, 4, 3), dtype=np.float64)
        env[27:37, :, :] = rgb
        return environment_mood_tint(env, max_gain=4.0)

    warm = tint([1.0, 0.5, 0.2])
    brown = tint([0.5, 0.35, 0.25])
    cool = tint([0.2, 0.5, 1.0])
    assert warm[0] - warm[2] > brown[0] - brown[2] > cool[0] - cool[2]
    assert warm[0] > warm[2]
    assert cool[0] < cool[2]


def test_tint_gains_are_bounded():
    env = np.zeros((64, 4, 3), dtype=np.float64)
    env[27:37, :, 0] = 5.0
    tint = environment_mood_tint(env, max_gain=1.25)
    assert np.all(tint <= 1.25 + 1e-12)
    assert np.all(tint >= 1.0 / 1.25 - 1e-12)


def test_environment_mood_tint_dtype_rgba_and_validation():
    for dtype in (np.float16, np.float32, np.float64):
        env = np.zeros((64, 4, 4), dtype=dtype)
        env[27:37, :, 0] = 1.0
        env[..., 3] = 999
        tint = environment_mood_tint(env)
        assert tint.dtype == dtype
        assert tint.shape == (3,)
        assert tint[0] > tint[2]
    env_i = np.zeros((64, 4, 3), dtype=np.uint8)
    env_i[27:37, :, 0] = 200
    assert environment_mood_tint(env_i).dtype == np.float64
    bad = np.zeros((8, 8, 3), dtype=np.float64)
    for kwargs in ({"horizon_fraction": 0.0}, {"horizon_fraction": 1.5}, {"max_gain": 0.5}):
        with pytest.raises(ValueError):
            environment_mood_tint(bad, **kwargs)
    for arr in (np.zeros((8, 8)), np.zeros((8, 8, 2)), np.zeros((0, 8, 3)), np.zeros((8, 0, 3))):
        with pytest.raises(ValueError):
            environment_mood_tint(arr)
    with pytest.raises(ValueError):
        environment_mood_tint(np.full((8, 8, 3), np.nan))


def test_strength_zero_is_identity_copy():
    img = np.arange(5 * 7 * 4, dtype=np.float32).reshape(5, 7, 4) / 100.0
    out = apply_luminance_preserving_tint(img, [1.5, 0.8, 0.9], strength=0.0)
    assert out.dtype == img.dtype
    assert np.array_equal(out, img)
    assert out is not img


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_identity_tint_preserves_all_integer_dtypes_exactly(dtype):
    info = np.iinfo(dtype)
    values = [0, 1, info.max - 1, info.max]
    if np.issubdtype(dtype, np.signedinteger):
        values += [info.min, info.min + 1, -1]
        if np.dtype(dtype).itemsize >= 8:
            values += [-(2**53 - 1), -(2**53), -(2**53 + 1), -(2**60 + 1)]
    if np.dtype(dtype).itemsize >= 8:
        values += [2**53 - 1, 2**53, 2**53 + 1, 2**60 + 1]
    arr = np.array(values, dtype=dtype)
    img = np.stack([arr, arr, arr], axis=-1).reshape(-1, 1, 3)
    out = apply_luminance_preserving_tint(img, [1.0, 1.0, 1.0], strength=1.0)
    assert out.dtype == img.dtype
    assert np.array_equal(out, img)


def test_luminance_preserved_and_alpha_unchanged():
    rng = np.random.default_rng(1)
    img = rng.random((6, 6, 4)).astype(np.float64)
    img[..., 3] = np.linspace(0.0, 1.0, 36).reshape(6, 6)
    out = apply_luminance_preserving_tint(img, [1.4, 0.9, 0.7], strength=0.6)
    assert np.allclose(img[..., :3] @ W, out[..., :3] @ W, atol=1e-9)
    assert np.array_equal(out[..., 3], img[..., 3])
    assert not np.allclose(out[..., :3], img[..., :3])


def test_ordinary_values_match_literal_formula():
    img = np.array(
        [[[0.2, 0.4, 0.6, 0.1], [0.7, 0.3, 0.1, 0.9]]],
        dtype=np.float64,
    )
    tint = np.array([1.4, 0.8, 1.1], dtype=np.float64)
    strength = 0.625
    out = apply_luminance_preserving_tint(img, tint, strength=strength)
    mix = 1.0 + strength * (tint - 1.0)
    rgb = img[..., :3].astype(np.float64)
    expected = rgb * mix
    expected += ((rgb @ W) - (expected @ W))[..., None]
    assert np.allclose(out[..., :3], expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(out[..., 3], img[..., 3])


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_integer_rounding_saturation_and_alpha_extrema(dtype):
    info = np.iinfo(dtype)
    img = np.zeros((2, 2, 4), dtype=dtype)
    img[0, 0] = [10, 20, 30, info.max]
    img[0, 1] = [info.max, info.max, info.max, info.min]
    img[1, 0] = [info.min, info.min, info.min, info.max]

    rounded = apply_luminance_preserving_tint(img, [1.1, 0.9, 1.0], strength=0.5)
    rgb = img[0:1, 0:1, :3].astype(np.float64)
    mix = 1.0 + 0.5 * (np.array([1.1, 0.9, 1.0]) - 1.0)
    expected = rgb * mix
    expected += ((rgb @ W) - (expected @ W))[..., None]
    assert np.array_equal(rounded[0:1, 0:1, :3], np.clip(np.rint(expected), info.min, info.max).astype(dtype))

    saturated = apply_luminance_preserving_tint(img, [1000.0, 0.0, 0.0], strength=1.0)
    assert saturated.dtype == img.dtype
    assert saturated[0, 1, 0] == info.max
    assert saturated[0, 1, 1] == info.min
    if np.issubdtype(dtype, np.signedinteger):
        assert saturated[1, 0, 0] == info.min
        assert saturated[1, 0, 1] == info.max
    assert np.array_equal(saturated[..., 3], img[..., 3])

    identity = apply_luminance_preserving_tint(img, [1000.0, 0.0, 0.0], strength=0.0)
    assert np.array_equal(identity, img)


def test_integer_positive_and_negative_non_identity_tints():
    img = np.array([[[2**53, 2**53 + 1, 2**53 - 1], [-(2**53), -(2**53 + 1), -(2**53 - 1)]]], dtype=np.int64)
    for tint in ([1.2, 0.8, 1.0], [0.8, 1.2, 1.0]):
        out = apply_luminance_preserving_tint(img, tint, strength=0.5)
        rgb = img.astype(np.float64)
        mix = 1.0 + 0.5 * (np.array(tint, dtype=np.float64) - 1.0)
        expected = rgb * mix
        expected += ((rgb @ W) - (expected @ W))[..., None]
        info = np.iinfo(np.int64)
        assert np.array_equal(out, np.clip(np.rint(expected), info.min, info.max).astype(np.int64))


def test_int64_uint64_alpha_and_rgb_do_not_wrap():
    for dtype in (np.int64, np.uint64):
        info = np.iinfo(dtype)
        img = np.zeros((1, 2, 4), dtype=dtype)
        img[0, 0] = [info.max, info.max, info.max, info.max]
        img[0, 1] = [info.min, info.min, info.min, info.min]
        out = apply_luminance_preserving_tint(img, [1000.0, 0.0, 0.0], strength=1.0)
        assert out[0, 0, 0] == info.max
        assert out[0, 0, 1] == info.min
        assert np.array_equal(out[..., 3], img[..., 3])


def test_extreme_floating_values_stay_finite_and_preserve_alpha():
    for dtype in (np.float16, np.float32, np.float64):
        finfo = np.finfo(dtype)
        cases = [
            (np.full((2, 2, 4), finfo.max, dtype=dtype), [np.finfo(np.float64).max, 1.0, 1.0]),
            (np.full((2, 2, 4), finfo.max, dtype=dtype), [-np.finfo(np.float64).max, 1.0, 1.0]),
            (np.array([[[finfo.max, -finfo.max, finfo.max / 2, -finfo.max]]], dtype=dtype),
             [np.finfo(np.float64).max, -np.finfo(np.float64).max, 0.5]),
        ]
        for img, tint in cases:
            alpha = img[..., 3].copy()
            out = apply_luminance_preserving_tint(img, tint, strength=1.0)
            assert out.dtype == dtype
            assert np.isfinite(out).all()
            assert np.array_equal(out[..., 3], alpha)


def test_finite_tinting_respects_strict_numpy_errstate():
    img = np.ones((1, 1, 4), dtype=np.float64)
    img[..., 3] = -0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = apply_luminance_preserving_tint(img, [0.0, 0.0, 0.0], strength=1.0)
    assert np.isfinite(out).all()
    assert np.signbit(out[..., 3])[0, 0]


def test_apply_tint_validation():
    img = np.zeros((4, 4, 3), dtype=np.float64)
    bad_calls = [
        (img, [1.0, 1.0], 0.5),
        (img, [1.0, 1.0, 1.0], 1.5),
        (img, [1.0, 1.0, 1.0], -0.1),
        (np.zeros((4, 4, 2)), [1, 1, 1], 0.5),
        (np.zeros((4, 4)), [1, 1, 1], 0.5),
        (np.full((4, 4, 3), np.nan), [1, 1, 1], 0.5),
    ]
    for arr, tint, strength in bad_calls:
        with pytest.raises(ValueError):
            apply_luminance_preserving_tint(arr, tint, strength=strength)
