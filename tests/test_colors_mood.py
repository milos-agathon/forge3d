# tests/test_colors_mood.py
# Slice 2 of the 2026-07-13 HDR terrain mood design: opt-in, renderer-
# independent environment mood-grade utilities in forge3d.colors.
#
# All fixtures are small SYNTHETIC linear-RGB arrays — no HDR files, caches, or
# machine-specific paths. These lock the exact horizon rows, identity
# fallbacks, cool/warm/brown ordering, strength-zero identity, luminance
# preservation before output clipping, unchanged alpha, bounded gains, and the
# integer-dtype round-trip (rounding, clipping, return dtype).

import numpy as np
import pytest

from forge3d.colors import apply_luminance_preserving_tint, environment_mood_tint

# Rec.709 weights (sum to exactly 1.0) shared with the implementation.
W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


# ---------------------------------------------------------------------------
# environment_mood_tint
# ---------------------------------------------------------------------------
def test_horizon_band_uses_only_central_rows():
    # H=64 -> band_height=round(10/64*64)=10, start=(64-10)//2=27, stop=37.
    H, Wd = 64, 8
    env = np.zeros((H, Wd, 3), dtype=np.float64)
    env[:, :, 2] = 10.0  # strong blue fill would dominate a full-image mean
    band = slice(27, 37)
    env[band, :, :] = 0.0
    env[band, :, 0] = 2.0  # pure red horizon band

    tint = environment_mood_tint(env)
    assert tint[0] > tint[2], "tint must reflect the warm band, not the blue fill"

    mean = env[band, :, :3].astype(np.float64).mean(axis=(0, 1))
    lum = float(mean @ W)
    expected = np.clip(mean / lum, 1.0 / 1.25, 1.25)
    assert np.allclose(tint, expected)


def test_band_boundaries_are_exact():
    # Poisoning the rows immediately adjacent to [27, 37) must not leak in.
    H, Wd = 64, 4
    env = np.zeros((H, Wd, 3), dtype=np.float64)
    env[27:37, :, :] = 1.0  # neutral grey band -> identity tint
    env[26, :, 0] = 1000.0  # just above the band
    env[37, :, 2] = 1000.0  # just below the band (stop is exclusive)

    tint = environment_mood_tint(env)
    assert np.allclose(tint, [1.0, 1.0, 1.0], atol=1e-9)


def test_near_black_band_returns_identity():
    env = np.zeros((64, 4, 3), dtype=np.float64)
    env[27:37, :, :] = 1e-15  # band luminance below the 1e-12 floor
    tint = environment_mood_tint(env)
    assert np.array_equal(tint, np.array([1.0, 1.0, 1.0]))


def test_cool_warm_brown_ordering():
    def band_tint(rgb):
        env = np.zeros((64, 4, 3), dtype=np.float64)
        env[27:37, :, :] = rgb
        # max_gain high enough that the clamp does not saturate the ordering.
        return environment_mood_tint(env, max_gain=4.0)

    warm = band_tint([1.0, 0.5, 0.2])   # R >> B
    brown = band_tint([0.5, 0.35, 0.25])  # mildly warm
    cool = band_tint([0.2, 0.5, 1.0])   # B >> R

    warm_rb = warm[0] - warm[2]
    brown_rb = brown[0] - brown[2]
    cool_rb = cool[0] - cool[2]
    assert warm_rb > brown_rb > cool_rb
    assert warm[0] > warm[2], "warm horizon must read warm"
    assert cool[0] < cool[2], "cool horizon must read cool"


def test_tint_gains_are_bounded():
    env = np.zeros((64, 4, 3), dtype=np.float64)
    env[27:37, :, 0] = 5.0  # extreme warm would blow past the gain cap
    tint = environment_mood_tint(env, max_gain=1.25)
    assert np.all(tint <= 1.25 + 1e-12)
    assert np.all(tint >= 1.0 / 1.25 - 1e-12)


def test_environment_mood_tint_validation():
    env = np.zeros((8, 8, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        environment_mood_tint(env, horizon_fraction=0.0)
    with pytest.raises(ValueError):
        environment_mood_tint(env, horizon_fraction=1.5)
    with pytest.raises(ValueError):
        environment_mood_tint(env, max_gain=0.5)
    with pytest.raises(ValueError):
        environment_mood_tint(np.zeros((8, 8), dtype=np.float64))  # not 3D
    with pytest.raises(ValueError):
        environment_mood_tint(np.zeros((8, 8, 2), dtype=np.float64))  # <3 channels
    with pytest.raises(ValueError):
        environment_mood_tint(np.full((8, 8, 3), np.nan, dtype=np.float64))
    # Empty spatial axes have no band; must be rejected, not return NaNs.
    with pytest.raises(ValueError):
        environment_mood_tint(np.zeros((0, 8, 3), dtype=np.float64))
    with pytest.raises(ValueError):
        environment_mood_tint(np.zeros((8, 0, 3), dtype=np.float64))


def test_environment_mood_tint_preserves_floating_dtype():
    def make(dtype):
        env = np.zeros((64, 4, 3), dtype=dtype)
        env[27:37, :, 0] = 1.0  # warm band
        return environment_mood_tint(env)

    for dtype in (np.float16, np.float32, np.float64):
        tint = make(dtype)
        assert tint.dtype == dtype, f"{dtype} not preserved (got {tint.dtype})"
        assert tint[0] > tint[2]
    # Integer environments have no floating dtype to preserve -> float64 tint.
    env_i = np.zeros((64, 4, 3), dtype=np.uint8)
    env_i[27:37, :, 0] = 200
    tint_i = environment_mood_tint(env_i)
    assert tint_i.dtype == np.float64
    assert tint_i[0] > tint_i[2]


def test_environment_mood_tint_accepts_rgba_and_ignores_alpha():
    env = np.zeros((64, 4, 4), dtype=np.float64)
    env[27:37, :, 0] = 1.0
    env[:, :, 3] = 999.0  # alpha must be ignored (slice is [..., :3])
    tint = environment_mood_tint(env)
    assert tint.shape == (3,)
    assert tint[0] > tint[2]


# ---------------------------------------------------------------------------
# apply_luminance_preserving_tint
# ---------------------------------------------------------------------------
def test_strength_zero_is_identity_copy():
    img = (np.arange(5 * 7 * 4, dtype=np.float32).reshape(5, 7, 4) / 100.0)
    out = apply_luminance_preserving_tint(img, [1.5, 0.8, 0.9], strength=0.0)
    assert out.dtype == img.dtype
    assert np.array_equal(out, img)
    assert out is not img  # a copy, not the same buffer


def test_luminance_preserved_before_output_clipping():
    rng = np.random.default_rng(1)
    img = rng.random((6, 6, 3)).astype(np.float64) * 2.0  # HDR-ish, may exceed 1
    tint = np.array([1.4, 0.9, 0.7])
    out = apply_luminance_preserving_tint(img, tint, strength=0.6)
    lum_in = img @ W
    lum_out = out @ W
    assert np.allclose(lum_in, lum_out, atol=1e-9), "luminance must be preserved"
    assert not np.allclose(out, img), "the tint must actually change chroma"


def test_alpha_is_copied_unchanged():
    img = np.zeros((3, 3, 4), dtype=np.float64)
    img[..., :3] = 0.5
    img[..., 3] = np.linspace(0.0, 1.0, 9).reshape(3, 3)
    out = apply_luminance_preserving_tint(img, [1.3, 0.8, 0.9], strength=0.5)
    assert np.array_equal(out[..., 3], img[..., 3])


def test_integer_dtype_round_trip():
    img = np.array([[[100, 150, 200], [10, 20, 30]]], dtype=np.uint8)
    tint = np.array([1.3, 1.0, 0.7])
    out = apply_luminance_preserving_tint(img, tint, strength=0.5)

    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255

    rgb = img[..., :3].astype(np.float64)
    mix = 1.0 + 0.5 * (tint - 1.0)
    lum0 = rgb @ W
    tinted = rgb * mix
    lum1 = tinted @ W
    tinted = tinted + (lum0 - lum1)[..., None]
    expected = np.clip(np.rint(tinted), 0, 255).astype(np.uint8)
    assert np.array_equal(out, expected)


def test_integer_clipping_saturates_not_wraps():
    # A bright pixel pushed past 255 by a warm tint must SATURATE to 255, not
    # wrap around (uint8(303) would wrap to 47). uint8 output is trivially in
    # 0-255 even on wraparound, so assert the exact saturated pixel instead.
    img = np.full((1, 1, 3), 250, dtype=np.uint8)
    out = apply_luminance_preserving_tint(img, [1.25, 1.0, 0.8], strength=1.0)
    assert out.dtype == np.uint8
    # R is ~303 before clipping: it must saturate to 255, not wrap to 47.
    assert out[0, 0, 0] == 255
    assert out[0, 0, 0] != 47
    assert np.array_equal(out[0, 0], np.array([255, 240, 190], dtype=np.uint8))


def test_extreme_floating_values_stay_finite():
    # Every channel at the dtype's finite maximum with a >1 tint used to
    # overflow: float16/float32 on the narrowing cast, float64 in the float64
    # intermediates (inf - inf -> nan). Finite input must yield finite output,
    # saturated to the input dtype's finite range.
    tint = np.array([1.25, 0.8, 0.8])
    for dtype in (np.float16, np.float32, np.float64):
        finfo = np.finfo(dtype)
        img = np.full((2, 2, 3), finfo.max, dtype=dtype)
        out = apply_luminance_preserving_tint(img, tint, strength=1.0)
        assert out.dtype == dtype
        assert np.isfinite(out).all(), f"{dtype} produced non-finite output: {out[0, 0]}"
        assert np.all(np.abs(out) <= finfo.max)
        # The saturating channel clamps to the dtype maximum, not below it.
        assert out[0, 0, 0] == finfo.max


def test_extreme_value_path_is_bit_identical_for_normal_inputs():
    # The overflow guards (power-of-two downscale + finite-range clamp) must
    # be inert for ordinary HDR-range inputs: same bits as the plain formula.
    rng = np.random.default_rng(7)
    img = (rng.random((5, 5, 3)) * 4.0).astype(np.float32)
    tint = np.array([1.4, 0.9, 0.7])
    out = apply_luminance_preserving_tint(img, tint, strength=0.6)

    rgb = img.astype(np.float64)
    mix = 1.0 + 0.6 * (tint - 1.0)
    lum0 = rgb @ W
    tinted = rgb * mix
    lum1 = tinted @ W
    expected = (tinted + (lum0 - lum1)[..., None]).astype(np.float32)
    assert np.array_equal(out, expected)


def test_apply_tint_validation():
    img = np.zeros((4, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(img, [1.0, 1.0], strength=0.5)  # tint len 2
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(img, [1.0, 1.0, 1.0], strength=1.5)  # strength>1
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(img, [1.0, 1.0, 1.0], strength=-0.1)
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(np.zeros((4, 4, 2)), [1, 1, 1], strength=0.5)
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(np.zeros((4, 4)), [1, 1, 1], strength=0.5)
    with pytest.raises(ValueError):
        apply_luminance_preserving_tint(
            np.full((4, 4, 3), np.nan), [1, 1, 1], strength=0.5
        )
