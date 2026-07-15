# EGM96 geoid coefficients (degree/order 120)

`egm96_n120.bin` is the committed coefficient set behind
`forge3d::geo::geoid` / `forge3d.crs.geoid_undulation`. It is evaluated by
spherical-harmonic synthesis on demand — it is never expanded into a grid.

## Provenance

Derived from the official NGA/NASA EGM96 release (public data):

- `EGM96` — the NASA/NIMA spherical-harmonic potential coefficient set,
  complete to degree and order 360, from the NGA "egm-96spherical" package
  (https://earth-info.nga.mil), truncated here to degree/order 120.
- `CORRCOEF` — NGA's spherical-harmonic correction coefficients (primarily the
  height-anomaly → geoid-undulation conversion term of Rapp 1996), same
  package, truncated to degree/order 120, stored in centimetres as published.

The evaluation convention follows NGA's reference program `F477.F` exactly:
WGS84(G873) constants, removal of the normal field's even zonal harmonics
(J2..J10), the correction-model sum, and the −0.53 m zero-degree term, so the
undulations refer to the WGS84 ellipsoid. Reference values for validation are
committed under `tests/data/egm96_test_values.txt` (NGA `OUTF477.DAT` points
plus `WW15MGH.GRD` grid nodes).

## Binary format (little-endian)

| offset | type      | value |
|--------|-----------|-------|
| 0      | `[u8; 8]` | magic `"F3DEGM96"` |
| 8      | `u32`     | format version (1) |
| 12     | `u32`     | nmax (120) |
| 16     | `u32`     | potential pair count (7378) |
| 20     | `u32`     | correction pair count (7381) |
| 24     | `(f64, f64) × 7378` | fully-normalized (C̄nm, S̄nm), n = 2..=120, m = 0..=n, n-major |
| …      | `(f64, f64) × 7381` | correction (Cnm, Snm) in centimetres, n = 0..=120, m = 0..=n, n-major |

Total size: 236,168 bytes (< 1 MiB).
