"""
CIELAB conversion, CIEDE2000 (dE2000), and adjudication masks for golden testing.

Pure NumPy, no external color-science dependencies (companion to tests/_ssim.py).
sRGB decode/encode uses the piecewise IEC 61966-2-1 curves; XYZ uses the D65
Rec.709 primaries; CIELAB uses the D65 reference white. The dE2000 implementation
follows Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
Implementation Notes, Supplementary Test Data, and Mathematical Observations",
including the RT rotation term (this is CIEDE2000, NOT dE76).
"""

from __future__ import annotations

import numpy as np

# D65 reference white (2-degree observer), Y normalized to 1.0.
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# Linear sRGB (Rec.709 primaries, D65) -> CIEXYZ matrix.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# Rec.709 relative-luminance weights on linear RGB (row 2 of the matrix above).
_LUMA_WEIGHTS = _SRGB_TO_XYZ[1]


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values (uint8 0..255 or float 0..1) to linear 0..1."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (..., 3) [uint8 or 0..1 float] to CIELAB (..., 3) under D65."""
    rgb = np.asarray(rgb)
    if rgb.shape[-1] < 3:
        raise ValueError(f"Expected trailing RGB(A) axis, got shape {rgb.shape}")
    linear = srgb_to_linear(rgb[..., :3])
    xyz = linear @ _SRGB_TO_XYZ.T
    xyz_n = xyz / _D65_WHITE

    delta = 6.0 / 29.0
    f = np.where(
        xyz_n > delta**3,
        np.cbrt(xyz_n),
        xyz_n / (3.0 * delta**2) + 4.0 / 29.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )
    return lab


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference between two CIELAB arrays of shape (..., 3).

    Faithful to Sharma et al. (2005), incl. G-based a' rescale, hue-difference
    wrap rules, the T term, and the RT rotation term. kL = kC = kH = 1.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    # Hue angles in degrees, in [0, 360). h' = 0 when C' == 0.
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where(C1p == 0.0, 0.0, h1p)
    h2p = np.where(C2p == 0.0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    # Delta h' with wrap (Sharma eq. 10).
    dh = h2p - h1p
    dhp = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    # Mean hue h'_bar (Sharma eq. 14).
    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    hp_bar = np.where(
        C1p * C2p == 0.0,
        hsum,  # convention: sum (only one defined); T-term unaffected since dHp=0
        np.where(
            habs <= 180.0,
            0.5 * hsum,
            np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar**7
    RC = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0**7))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    Lm50 = (Lp_bar - 50.0) ** 2
    SL = 1.0 + 0.015 * Lm50 / np.sqrt(20.0 + Lm50)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T

    tL = dLp / SL
    tC = dCp / SC
    tH = dHp / SH
    return np.sqrt(tL**2 + tC**2 + tH**2 + RT * tC * tH)


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance of sRGB-encoded input, in linear 0..1."""
    linear = srgb_to_linear(np.asarray(rgb)[..., :3])
    return linear @ _LUMA_WEIGHTS


def lit_mask(reference_rgb: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    """Boolean (H, W) mask of lit pixels in the reference render.

    A pixel is 'lit' when its linear relative luminance exceeds `threshold`,
    excluding fully-shadowed/near-black pixels so dE claims are about lit
    surface color.
    """
    return relative_luminance(reference_rgb) > threshold


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation of a boolean mask via shifted ORs (pure NumPy)."""
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[yd, xd] = mask[ys, xs]
            out |= shifted
    return out


def shadow_boundary_band(
    reference_rgb: np.ndarray,
    lum_threshold: float = 0.12,
    band_px: int = 3,
) -> np.ndarray:
    """Boolean (H, W) band straddling the lit/shadow boundary of the reference.

    Thresholds the reference luminance to a binary lit/shadow mask, extracts the
    boundary (pixels whose 8-neighborhood is not uniform), and dilates it into a
    band of +/- `band_px` pixels. Compute SSIM on this ROI only.
    """
    lit = relative_luminance(reference_rgb) > lum_threshold
    interior = _binary_dilate(~lit, 1)  # lit pixels adjacent to shadow ...
    boundary = lit & interior  # ... form the raw boundary line
    return _binary_dilate(boundary, band_px)


def band_bbox(band: np.ndarray) -> tuple[slice, slice]:
    """Tight bounding-box slices of a boolean band mask (for SSIM ROI crops)."""
    ys, xs = np.nonzero(band)
    if ys.size == 0:
        raise ValueError("shadow boundary band is empty")
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
