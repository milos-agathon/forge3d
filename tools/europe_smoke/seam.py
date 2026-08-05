# tools/europe_smoke/seam.py
"""Gate 6: does the analysis/forecast join look like an ordinary time step?

Calibration priors from the two cached cubes: within-run p95 of the per-hour
burden change rate is 0.0277 (Greece, n=30) and 0.0307 (Iberia, n=42); pooled
p95 = 0.0313, p99 = 0.0500 h^-1. Measured power is only ~0.48 at a 5% false
alarm rate, so this catches gross misjoins, not subtle ones. The verdict
carries that caveat.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

POOLED_P95_PRIOR = 0.0313
POOLED_P99_PRIOR = 0.0500
CAVEAT = (
    "Measured power ~0.48 at 5% false alarm on the cached cubes; catches gross "
    "misjoins, not subtle ones. p99 of a small reference is an order statistic."
)


def burden(ds: xr.Dataset, var: str = "omaod550") -> np.ndarray:
    """B(t) = sum over cells of AOD * cos(lat) -- the cheap seam statistic."""
    w = np.cos(np.radians(ds["latitude"].values))[None, :, None]
    return (ds[var].values * w).sum(axis=(1, 2))


def rates(times: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-hour fractional change between consecutive steps."""
    dt_h = np.diff(times).astype("timedelta64[m]").astype("float64") / 60.0
    mean_b = 0.5 * (b[:-1] + b[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.diff(b)) / (mean_b * dt_h)


def evaluate(ds: xr.Dataset, t_now, var: str = "omaod550") -> dict:
    times = ds["time"].values
    arm = ds["arm"].values
    b = burden(ds, var)
    r = rates(times, b)

    same_arm = arm[:-1] == arm[1:]
    seam_idx = int(np.argmax((arm[:-1] == "an") & (arm[1:] == "fc")))
    r_seam = float(r[seam_idx])

    # Spec gate 6: the reference must be CADENCE-MATCHED. Per-hour
    # normalisation removes most, not all, of the dt dependence -- measured
    # within-run p95 at 3/6/9 h separation is 0.0277/0.0240/0.0182 on the
    # Greece cube, a 34% drift across the range. The analysis arm is 6-hourly
    # and the seam step is 3-hourly, so an unmatched reference drags the
    # threshold ~15% low, i.e. towards false alarms. Compare 3 h against 3 h.
    step_h = np.diff(times).astype("timedelta64[m]").astype("float64") / 60.0
    seam_dt = step_h[seam_idx]
    matched = same_arm & np.isfinite(r) & np.isclose(step_h, seam_dt, rtol=1e-6)
    reference = r[matched]

    if reference.size < 8:
        p95, p99 = POOLED_P95_PRIOR, POOLED_P99_PRIOR
        reference_source = "pooled_priors"
        cadence_note = (
            f"only {int(reference.size)} same-arm intervals at the seam's "
            f"{seam_dt:g} h cadence; using pooled priors without mixing cadences"
        )
    else:
        p95 = float(np.percentile(reference, 95))
        p99 = float(np.percentile(reference, 99))
        reference_source = "empirical"
        cadence_note = None

    out = {
        "r_seam": r_seam,
        "seam_dt_h": float(seam_dt),
        "n_reference": int(reference.size),
        "cadence_matched": True,
        "reference_source": reference_source,
        "cadence_note": cadence_note,
        "caveat": CAVEAT,
        "priors": {"p95": POOLED_P95_PRIOR, "p99": POOLED_P99_PRIOR},
        "p95": p95,
        "p99": p99,
    }

    a = ds[var].values[seam_idx]
    c = ds[var].values[seam_idx + 1]
    if np.array_equal(a, c):
        out.update(verdict="FAIL", reason="seam fields are bit-identical (frozen frame)")
        return out

    if r_seam <= p95:
        out["verdict"] = "PASS"
    elif r_seam <= p99:
        out["verdict"] = "WARN"
    else:
        out["verdict"] = "FAIL"
        out["reason"] = f"r_seam {r_seam:.4f} exceeds p99 {p99:.4f}"
    return out
