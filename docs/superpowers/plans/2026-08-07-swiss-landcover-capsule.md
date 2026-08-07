# Swiss Land Cover Verified Map Capsule — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/capsules/swiss-landcover/` — the first Verified Map Capsule: a hash-pinned, certificate-bearing, locally reproducible PROMETHEUS path-traced Swiss land-cover plate, per the approved spec `docs/superpowers/specs/2026-08-07-swiss-landcover-capsule-design.md`.

**Architecture:** The recipe feeds the two hash-pinned registry rasters (`fetch_dem("swiss")`, `fetch("swiss-land-cover")`) into the proven Swiss PT pipeline: overlay building via public functions of `examples/swiss_terrain_landcover_viewer.py`, light field + modulation via `examples/swiss_landcover_pt_3d.py` (constants verbatim), with a small ported light-field loop that adds `certificate=` on the final tile. A standalone `verify.py` compares a local run against the committed reference (SSIM + certificate fields).

**Tech Stack:** Python 3.10+, forge3d (installed via `maturin develop` in-repo; pinned wheel for external users), numpy, rasterio, PIL, scipy; pytest for tests. No Rust/WGSL changes.

## Global Constraints

- Capsule directory: `examples/capsules/swiss-landcover/` (spec §2, Milos 2026-08-07).
- NO network access at recipe runtime beyond the two registry dataset fetches (spec §4).
- NO Pro imports: nothing from `forge3d.map_plate`, export, buildings, style (spec §5). Clean run must succeed with no license key set.
- NO telemetry, beacons, or phoned-home data anywhere (spec §13).
- NO silent fallback: recipe exits nonzero if GPU/native path unavailable, tracer reports no terrain hits, or convergence fails (spec §4.6).
- Certificate: development-signed in v1, labeled honestly; production signing deferred (spec §6).
- Image comparison: SSIM + mean-abs, thresholds calibrated single-vendor NVIDIA and documented as such (spec §8).
- Attribution rendered verbatim where data is described: `Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft` and the Terrarium/AWS Terrain Tiles elevation notice (spec §3, §9).
- Preview budgets: hero 1920px ≤350 KB AVIF / ≤500 KB WebP; thumbnail 640px ≤60 KB (spec §11).
- PT constants come VERBATIM from `examples/swiss_landcover_pt_3d.py` (`RELIEF_WORLD=3.2`, `CAMERA_FOV_Y=8.0`, `CAMERA_MARGIN=1.06`, `SUN_AZIMUTH=135.0`, `SUN_ELEVATION=28.0`, `SUN_INTENSITY=2.6`, `ENV_INTENSITY=0.82`, `PT_ALBEDO=(0.62,0.62,0.62)`, `SPAN_X=100.0`) — do not re-derive.
- Repo test conventions: tests live in `tests/`, run `python -m pytest tests/<file> -v --tb=short`. GPU-dependent tests are marked and skipped when `forge3d.has_gpu()` is false.
- Commits: scoped, one per task, message style `feat(capsule): …` / `test(capsule): …` / `docs(capsule): …`.

---

### Task 1: Capsule scaffold, data manifest, and provenance

**Files:**
- Create: `examples/capsules/swiss-landcover/data-manifest.json`
- Create: `examples/capsules/swiss-landcover/provenance.json`
- Test: `tests/test_capsule_swiss_landcover.py`

**Interfaces:**
- Produces: `data-manifest.json` with a `datasets` list whose entries have keys `name`, `file`, `sha256`, `fetch` (exact registry call). `provenance.json` with one record per dataset, keys exactly: `dataset`, `distributed_file`, `distributed_sha256`, `upstream_provider`, `upstream_product`, `upstream_url`, `license_basis`, `required_attribution`, `acquired_date`, `transformations`, `notes`. Later tasks (README, verify) read these files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capsule_swiss_landcover.py
import json
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1] / "examples" / "capsules" / "swiss-landcover"


def _registry():
    from forge3d import datasets as ds
    return ds._REMOTE_DATASETS


def test_manifest_matches_registry_hashes():
    manifest = json.loads((CAPSULE / "data-manifest.json").read_text(encoding="utf-8"))
    reg = _registry()
    by_name = {d["name"]: d for d in manifest["datasets"]}
    assert set(by_name) == {"swiss", "swiss-land-cover"}
    for name, entry in by_name.items():
        assert entry["sha256"] == reg[name].known_hash.removeprefix("sha256:")
        assert entry["file"] == reg[name].filename


def test_provenance_complete_no_tbd():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    assert {r["dataset"] for r in prov["records"]} == {"swiss", "swiss-land-cover"}
    required = {
        "dataset", "distributed_file", "distributed_sha256", "upstream_provider",
        "upstream_product", "upstream_url", "license_basis", "required_attribution",
        "acquired_date", "transformations", "notes",
    }
    for record in prov["records"]:
        assert required <= set(record), f"missing keys in {record['dataset']}"
        for key in required:
            assert "TBD" not in json.dumps(record[key]), f"TBD left in {record['dataset']}.{key}"


def test_landcover_attribution_line_exact():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    lc = next(r for r in prov["records"] if r["dataset"] == "swiss-land-cover")
    assert lc["required_attribution"] == (
        "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short`
Expected: FAIL (FileNotFoundError — capsule files don't exist).

- [ ] **Step 3: Write the two JSON files**

`data-manifest.json` — copy the hashes from `python/forge3d/datasets.py` (`swiss`, `swiss-land-cover` entries), full 64-char values:

```json
{
  "capsule": "swiss-landcover",
  "version": "0.1.0",
  "datasets": [
    {
      "name": "swiss",
      "file": "switzerland_dem.tif",
      "sha256": "d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478",
      "fetch": "forge3d.datasets.fetch_dem('swiss')"
    },
    {
      "name": "swiss-land-cover",
      "file": "switzerland_land_cover.tif",
      "sha256": "6b254585be4982ed9e8da63b8536ecc2f5fa4c64c6545db06c73eb1fe39a8f7f",
      "fetch": "forge3d.datasets.fetch('swiss-land-cover')"
    }
  ]
}
```

`provenance.json` — Milos's answers (2026-08-07) plus the in-code attribution line. For fields not recoverable from the repo (`upstream_url` exact tile endpoints, `acquired_date`), consult the creating pipeline (`examples/swiss_terrain_landcover_viewer.py` `_build_dem` / `bosnia_terrain_landcover_viewer.py` `_download_landcover_tiles` — read the URL constants there and record them); dates: record `"2025 (from repo asset history: git log --follow assets/tif/switzerland_dem.tif)"` — run that command and put the actual commit date in. Structure:

```json
{
  "records": [
    {
      "dataset": "swiss",
      "distributed_file": "switzerland_dem.tif",
      "distributed_sha256": "d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478",
      "upstream_provider": "AWS Terrain Tiles (Mapzen Terrarium)",
      "upstream_product": "Terrarium-encoded elevation tiles, ~30 m (composite: EU-DEM, SRTM et al. for the Swiss extent)",
      "upstream_url": "<exact tile URL constant from the creating script>",
      "license_basis": "Mapzen/AWS Terrain Tiles open composite; underlying sources' notices apply",
      "required_attribution": "Elevation: Terrain Tiles (Mapzen/AWS Open Data), sources incl. EU-DEM (Copernicus) and SRTM (NASA)",
      "acquired_date": "<from git log --follow>",
      "transformations": ["tile mosaic", "clip to Switzerland boundary", "reproject", "GeoTIFF export"],
      "notes": "Produced by Milos Popovic; pinned by sha256 in forge3d.datasets registry."
    },
    {
      "dataset": "swiss-land-cover",
      "distributed_file": "switzerland_land_cover.tif",
      "distributed_sha256": "6b254585be4982ed9e8da63b8536ecc2f5fa4c64c6545db06c73eb1fe39a8f7f",
      "upstream_provider": "Esri / Impact Observatory / Microsoft",
      "upstream_product": "Sentinel-2 10m Land Use/Land Cover",
      "upstream_url": "<exact tile/download URL constant from the creating script>",
      "license_basis": "CC BY 4.0",
      "required_attribution": "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft",
      "acquired_date": "<from git log --follow>",
      "transformations": ["tile mosaic", "clip to Switzerland boundary", "RGB class raster export"],
      "notes": "Credit line matches examples/swiss_terrain_landcover_viewer.py:22 verbatim."
    }
  ]
}
```

Replace every `<...>` with the real value found in the named sources before committing — the test's no-TBD scan will not catch `<...>`, so also grep the final JSON for `<` and assert none remain.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/capsules/swiss-landcover/data-manifest.json examples/capsules/swiss-landcover/provenance.json tests/test_capsule_swiss_landcover.py
git commit -m "feat(capsule): swiss-landcover manifest and provenance with pinned hashes"
```

---

### Task 2: `recipe.py` — certified PT render from pinned datasets

**Files:**
- Create: `examples/capsules/swiss-landcover/recipe.py`
- Test: append to `tests/test_capsule_swiss_landcover.py`

**Interfaces:**
- Consumes: registry fetches; public functions of `examples/swiss_terrain_landcover_viewer.py` (`build_landcover_classes(landcover_path) -> Path`, `build_aligned_dem(dem_path) -> Path`, `despeckle_landcover_classes(classes) -> np.ndarray`, `classes_to_rgba(classes) -> np.ndarray`, `resample_raster_to_grid`, `_apply_class_aware_adjustments(img, class_mask)`, `compose_snapshot(...)`, `_downsample_snapshot(...)`, `_configure_base()`, `CLASS_ADJUST`, `MAP_SUBJECT_ROTATION_DEG`); `examples/swiss_landcover_pt_3d.py` (`_load_dem_grid`, `_camera_for_grid`, `_neutral_sky_env`, `_shade_on_overlay_grid`, `_modulate_overlay`, and its module constants); `forge3d.path_tracing.hybrid_render_terrain_reference(..., certificate=...)`.
- Produces: CLI-less script; running `python recipe.py` writes `out/swiss_landcover.png` and `out/local.certificate.json` under the capsule dir. Exit codes: `0` success, `2` no GPU/native path, `3` no terrain hits / aspect guard, `4` certificate missing after render. Module constants `GRID_MAX = 1536`, `FRAME = 1120`, `TILES = 2`, `MAX_FRAMES = 2048`, `SEED = 7`, `SNAPSHOT_SIZE = (1920, 1080)`, `SUPERSAMPLE = 2` (the proven Swiss PT defaults). Later tasks call `recipe.OUT_DIR`, `recipe.SNAPSHOT`, `recipe.CERT_PATH`.

- [ ] **Step 1: Write the failing test (import-safety + guards, no GPU needed)**

```python
# append to tests/test_capsule_swiss_landcover.py
import subprocess
import sys


def _load_recipe_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("swiss_capsule_recipe", CAPSULE / "recipe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recipe_imports_without_gpu_and_declares_contract():
    mod = _load_recipe_module()
    # Proven Swiss PT register, verbatim (spec: do not re-derive).
    assert (mod.GRID_MAX, mod.FRAME, mod.TILES, mod.MAX_FRAMES, mod.SEED) == (1536, 1120, 2, 2048, 7)
    assert mod.SNAPSHOT.name == "swiss_landcover.png"
    assert mod.CERT_PATH.name == "local.certificate.json"
    assert mod.EXIT_NO_GPU == 2 and mod.EXIT_NO_HITS == 3 and mod.EXIT_NO_CERT == 4


def test_recipe_has_no_pro_imports_and_no_telemetry():
    text = (CAPSULE / "recipe.py").read_text(encoding="utf-8")
    for banned in ("map_plate", "export_svg", "export_pdf", "add_buildings", "set_license_key",
                   "requests.", "urllib.request", "httpx"):
        assert banned not in text, f"banned reference in recipe.py: {banned}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -k recipe -v --tb=short`
Expected: FAIL (recipe.py does not exist).

- [ ] **Step 3: Write `recipe.py`**

Structure (complete except the two verbatim ports, whose exact source lines are named):

```python
#!/usr/bin/env python3
"""Swiss land cover — Verified Map Capsule recipe.

Reproduces the capsule's reference render on local hardware from the two
hash-pinned forge3d registry datasets, and emits a RenderCertificate for the
final path-traced tile. Verified against forge3d <PIN AT TASK 5>.

Pipeline and all lighting constants are the approved Swiss PT register
(examples/swiss_landcover_pt_3d.py); data prep uses the pre-baked registry
rasters instead of the network build. See VERIFY.md for the check procedure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

CAPSULE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CAPSULE_DIR.parents[1]          # examples/
sys.path.insert(0, str(EXAMPLES_DIR))

import swiss_terrain_landcover_viewer as swiss          # noqa: E402
import swiss_landcover_pt_3d as swisspt                 # noqa: E402
from forge3d.path_tracing import hybrid_render_terrain_reference  # noqa: E402
import forge3d as f3d                                   # noqa: E402

OUT_DIR = CAPSULE_DIR / "out"
SNAPSHOT = OUT_DIR / "swiss_landcover.png"
CERT_PATH = OUT_DIR / "local.certificate.json"

# Proven Swiss PT register — the swiss_landcover_pt_3d.py CLI defaults, frozen.
GRID_MAX = 1536
FRAME = 1120
TILES = 2
SPP = 1
MAX_FRAMES = 2048
MIN_FRAMES = 32
VARIANCE_THRESHOLD = 1e-3
SEED = 7
SNAPSHOT_SIZE = (1920, 1080)
SUPERSAMPLE = 2

EXIT_NO_GPU = 2
EXIT_NO_HITS = 3
EXIT_NO_CERT = 4


def _pt_args() -> argparse.Namespace:
    return argparse.Namespace(
        spp=SPP, max_frames=MAX_FRAMES, min_frames=MIN_FRAMES,
        variance_threshold=VARIANCE_THRESHOLD, seed=SEED, tiles=TILES,
    )


def _pt_light_field_certified(dem_grid: np.ndarray, frame: int, args) -> tuple:
    """Verbatim port of swiss_landcover_pt_3d._pt_light_field (lines 180-198)
    with ONE change: the final tile's hybrid_render_terrain_reference call
    passes certificate=str(CERT_PATH), so the certificate describes the last
    production pass of this run (engine/WGSL/adapter fields are common to all
    tiles). Port the loop body exactly; do not re-derive weights or stitching."""
    ...


def main() -> int:
    if not f3d.has_gpu():
        print("ERROR: no GPU adapter — this capsule requires the native PT path "
              "(no CPU fallback is offered).", file=sys.stderr)
        return EXIT_NO_GPU
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    swiss._configure_base()
    swiss.CLASS_ADJUST["atmosphere"] = {**swiss.CLASS_ADJUST["atmosphere"], "enabled": False}
    swiss.MAP_SUBJECT_ROTATION_DEG = 0.0
    import bosnia_terrain_landcover_viewer as base
    base.SNAPSHOT_SIZE = (SNAPSHOT_SIZE[0] * SUPERSAMPLE, SNAPSHOT_SIZE[1] * SUPERSAMPLE)

    print("== Fetching pinned datasets (sha256-checked by the registry) ==")
    dem_path = f3d.datasets.fetch_dem("swiss")
    lc_path = f3d.datasets.fetch("swiss-land-cover")

    print("== Building overlay + class mask from pre-baked rasters ==")
    terrain_path = swiss.build_aligned_dem(dem_path)
    classes_tif = swiss.build_landcover_classes(lc_path)
    with rasterio.open(terrain_path) as t:
        target_grid = {"crs": t.crs, "transform": t.transform,
                       "width": t.width, "height": t.height, "nodata": -1}
    classes = np.asarray(
        swiss.resample_raster_to_grid(classes_tif, target_grid,
                                      resampling="mode", dst_nodata=-1)["array"],
        dtype=np.int16,
    )
    classes = swiss.despeckle_landcover_classes(classes)
    overlay_path = OUT_DIR / "overlay.png"
    Image.fromarray(swiss.classes_to_rgba(classes), "RGBA").save(overlay_path)
    # `present` (legend classes) computed exactly as base._build_overlay does —
    # mirror its derivation from the class array (bosnia_terrain_landcover_viewer.py).
    present = ...

    print("== Path tracing the light field (PROMETHEUS, certified final tile) ==")
    dem_grid = swisspt._load_dem_grid(terrain_path, GRID_MAX)
    rgb_field, hit = _pt_light_field_certified(dem_grid, FRAME, _pt_args())
    if not np.any(hit):
        print("ERROR: path tracer produced no terrain hits.", file=sys.stderr)
        return EXIT_NO_HITS

    print("== Modulating overlay + composing plate ==")
    with Image.open(overlay_path) as probe:
        overlay_size = probe.size
    shade, tint = swisspt._shade_on_overlay_grid(rgb_field, hit, overlay_size)
    raw = swisspt._modulate_overlay(overlay_path, shade, tint)
    raw = swiss._apply_class_aware_adjustments(raw, classes)
    tmp = SNAPSHOT.parent / f".{SNAPSHOT.stem}_tmp.png"
    swiss.compose_snapshot(raw, tmp, present, sun_azimuth=315.0,
                           sun_elevation=float(swisspt.SUN_ELEVATION),
                           canvas_size=base.SNAPSHOT_SIZE)
    swiss._downsample_snapshot(tmp, SNAPSHOT_SIZE, SUPERSAMPLE)
    tmp.replace(SNAPSHOT)

    if not CERT_PATH.is_file():
        print("ERROR: render completed but no certificate was emitted.", file=sys.stderr)
        return EXIT_NO_CERT
    print(f"Success! Map: {SNAPSHOT}\nCertificate (development-signed): {CERT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The two `...` are filled by reading the named sources: (a) `_pt_light_field_certified` ports `swiss_landcover_pt_3d.py:180-198` adding `certificate=str(CERT_PATH)` to the last tile's call — the port must delete any pre-existing `CERT_PATH` first so a stale file can't satisfy the check; (b) `present` mirrors `base._build_overlay`'s derivation (open `bosnia_terrain_landcover_viewer.py`, find `_build_overlay`'s return, reproduce the `present` computation from the `classes` array — it is the per-class presence used by the legend).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short`
Expected: all pass (recipe import test runs module-level code only — it must not fetch or render at import time; keep everything inside `main()`).

- [ ] **Step 5: GPU smoke run at probe scale (manual, this machine)**

Run (temporarily overriding scale via env is NOT provided — instead run the real thing once at small scale by editing nothing: use a throwaway copy):

```bash
python - <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("r", Path("examples/capsules/swiss-landcover/recipe.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.GRID_MAX, m.FRAME, m.MAX_FRAMES = 512, 384, 128   # probe scale
sys.exit(m.main())
EOF
```

Expected: exit 0; `out/swiss_landcover.png` and `out/local.certificate.json` exist; console shows converged tiles. (Small-scale output will look coarse — that's fine; this validates wiring, not quality.)

- [ ] **Step 6: Commit**

```bash
git add examples/capsules/swiss-landcover/recipe.py tests/test_capsule_swiss_landcover.py
git commit -m "feat(capsule): certified swiss-landcover PT recipe from pinned datasets"
```

---

### Task 3: `verify.py` — SSIM + certificate comparison

**REV 4 AMENDMENTS (BINDING — override conflicting text below):** `verify.py` is the single acceptance command (spec §17.7). Beyond the compare() contract below it must, in `main()`: (a) validate `data-manifest.json` and `provenance.json` are well-formed with required keys; (b) recompute sha256 of both fetched dataset files (resolve via the same `forge3d.datasets` calls as the recipe), of `out/swiss_landcover.png`, of `recipe.py`, and of `out/local.certificate.json`, and cross-check them against `out/recipe_result.json` (written by the recipe; keys `png_sha256`, `inputs[].sha256`, `recipe_sha256`, `certificate_sha256`, `forge3d_version`, `created`) AND against the committed capsule-root `recipe_result.json` reference binding where present (inputs + recipe hashes must match; png hash matches `expected/reference.png`, not the local PNG); (c) compare `out/swiss_landcover.png` against `expected/reference.png` — the CANONICAL LOSSLESS reference — never against WebP/AVIF previews; (d) treat engine version, WGSL module hashes, and empty degradations as MANDATORY gates, and adapter/timings as informational-only (printed, never gating); (e) write machine-readable `out/verify_result.json` (all check results + pass/fail booleans + the two trust labels) and exit nonzero on any failure; (f) print the trust labels: reference certificate `development-signed — not production-signed (v1)`, local report `locally produced`. The certificate JSON is NEVER mutated by any tool. Add tests for: hash-mismatch fails, manifest-key-missing fails, verify_result.json written with correct booleans on both pass and fail paths (synthetic fixtures, no GPU).

**Files:**
- Create: `examples/capsules/swiss-landcover/verify.py`
- Create: `examples/capsules/swiss-landcover/verify_config.json` (placeholder thresholds; pinned in Task 4)
- Test: append to `tests/test_capsule_swiss_landcover.py`

**Interfaces:**
- Consumes: `expected/preview.webp` (reference image, Task 4), `expected/reference.certificate.json` (Task 4), `out/swiss_landcover.png` + `out/local.certificate.json` (Task 2).
- Produces: `python verify.py` exits 0 (pass) / 1 (fail) and prints a labeled report. Public function `compare(local_png, ref_img, local_cert, ref_cert, config) -> dict` with keys `ssim`, `mean_abs`, `ssim_ok`, `mean_abs_ok`, `engine_match`, `wgsl_match`, `degradations_empty`, `passed`. `verify.py` must be standalone (no imports from `tests/`); its SSIM must match `tests/_ssim.py` (test enforces).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capsule_swiss_landcover.py
import numpy as np


def _load_verify_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("swiss_capsule_verify", CAPSULE / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_ssim_matches_repo_reference_impl():
    from tests._ssim import ssim as repo_ssim
    v = _load_verify_module()
    rng = np.random.default_rng(11)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float64)
    b = np.clip(a + rng.normal(0, 6, a.shape), 0, 255)
    assert abs(v.ssim(a, b, data_range=255.0) - repo_ssim(a, b, data_range=255.0)) < 1e-9


def test_verify_compare_passes_and_fails_correctly():
    v = _load_verify_module()
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    cert = {"engine": {"version": "1.30.0", "wgsl_module_hashes": {"a": "h1"}}, "degradations": []}
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}
    ok = v.compare(img, img.copy(), cert, cert, cfg)
    assert ok["passed"] and ok["ssim"] == 1.0 and ok["engine_match"] and ok["wgsl_match"]
    bad_cert = {"engine": {"version": "1.30.0", "wgsl_module_hashes": {"a": "CHANGED"}}, "degradations": []}
    bad = v.compare(img, img.copy(), bad_cert, cert, cfg)
    assert not bad["passed"] and not bad["wgsl_match"]
    noisy = np.clip(img.astype(int) + 40, 0, 255).astype(np.uint8)
    assert not v.compare(noisy, img, cert, cert, cfg)["passed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -k verify -v --tb=short`
Expected: FAIL (verify.py does not exist).

- [ ] **Step 3: Write `verify.py`**

Copy the SSIM implementation from `tests/_ssim.py` into `verify.py` verbatim (cite the source file in a comment; the equality test keeps them in lock-step). Then:

```python
def compare(local_img, ref_img, local_cert, ref_cert, config):
    local = np.asarray(local_img, dtype=np.float64)[..., :3]
    ref = np.asarray(ref_img, dtype=np.float64)[..., :3]
    if local.shape != ref.shape:
        return {"passed": False, "error": f"shape {local.shape} != reference {ref.shape}"}
    score = ssim(local, ref, data_range=255.0)
    mean_abs = float(np.mean(np.abs(local - ref)))
    le, re_ = local_cert.get("engine", {}), ref_cert.get("engine", {})
    result = {
        "ssim": float(score),
        "mean_abs": mean_abs,
        "ssim_ok": score >= config["ssim_min"],
        "mean_abs_ok": mean_abs <= config["mean_abs_max"],
        "engine_match": le.get("version") == re_.get("version"),
        "wgsl_match": le.get("wgsl_module_hashes") == re_.get("wgsl_module_hashes"),
        "degradations_empty": not local_cert.get("degradations"),
    }
    result["passed"] = all(result[k] for k in
        ("ssim_ok", "mean_abs_ok", "engine_match", "wgsl_match", "degradations_empty"))
    return result
```

`main()`: load `verify_config.json`, `expected/preview.webp` (resized check is NOT allowed — sizes must match; render and reference are both 1920×1080), `expected/reference.certificate.json`, `out/*`; print each field with a PASS/FAIL marker plus the honesty labels: `reference certificate: development-signed (v1 — not production-signed)` and `local report: locally produced`. Exit 0 iff `passed`.

Initial `verify_config.json`: `{"ssim_min": null, "mean_abs_max": null, "note": "pinned by Task 4 calibration"}` — `main()` must refuse to run (exit 2, message "thresholds not yet calibrated") while values are null; `compare()` itself is null-safe only in tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add examples/capsules/swiss-landcover/verify.py examples/capsules/swiss-landcover/verify_config.json tests/test_capsule_swiss_landcover.py
git commit -m "feat(capsule): standalone verify tool (SSIM + certificate comparison)"
```

---

### Task 4: Reference render, threshold calibration, preview derivatives

**REV 4 AMENDMENTS (BINDING — override conflicting text below):** (a) The canonical comparison target is `expected/reference.png` — commit the chosen reference run's PNG LOSSLESS (this replaces Step 5's WebP-vs-PNG contingency; verify.py always compares PNG-to-PNG). (b) Do NOT add any field to the certificate JSON — commit `expected/reference.certificate.json` byte-identical to the run's `out/local.certificate.json`; the `development-signed` label lives in verify_config.json (`"trust": "development-signed — not production-signed (v1)"`), VERIFY.md, and verify.py output instead. The budget test asserting `cert["signing_label"]` is amended to assert the label string in `verify_config.json`. (c) Promote the reference run's `out/recipe_result.json` to the capsule root as the committed binding record (its `png_sha256` must equal the sha256 of `expected/reference.png`). (d) Derivatives are generated FROM `expected/reference.png`: `preview.avif`, `preview.webp` (1920px hero) and `thumb.webp` (640px) — display-only. (e) `verify_config.json`'s calibration block additionally records the GPU driver version and backend (both read from the certificate/adapter info). (f) The 1920×1080 white-margin concern from Task 2: before the calibration runs, render once and LOOK at the output; if large white margins dominate, switch SNAPSHOT_SIZE to the plate's natural aspect (square-ish, e.g. 2048×2048 — record the decision + regenerate constants test in a commit with rationale) — the reference and all three runs must use the FINAL size.

**Files:**
- Create: `examples/capsules/swiss-landcover/expected/preview.avif`, `expected/preview.webp`, `expected/thumb.webp`, `expected/reference.certificate.json`
- Create: `examples/capsules/swiss-landcover/make_derivatives.py`
- Modify: `examples/capsules/swiss-landcover/verify_config.json` (pin thresholds)
- Test: append derivative-budget test to `tests/test_capsule_swiss_landcover.py`

**Interfaces:**
- Consumes: `recipe.py` (full-scale runs on this machine's NVIDIA GPU).
- Produces: committed reference artifacts + pinned thresholds. `make_derivatives.py <src.png>` writes the three derivatives next to `expected/` and errors if budgets are exceeded.

- [ ] **Step 1: Full-scale reference render (long; NVIDIA)**

Run: `python examples/capsules/swiss-landcover/recipe.py`
Expected: exit 0; `out/swiss_landcover.png` (1920×1080) + `out/local.certificate.json`. If Windows TDR kills the device mid-run, re-run (Task 2's port has no per-cell cache — a rerun restarts the 2×2 field; acceptable at 4 tiles).

- [ ] **Step 2: Run-to-run variance calibration (2 more runs)**

```bash
for i in 1 2; do
  mv examples/capsules/swiss-landcover/out/swiss_landcover.png examples/capsules/swiss-landcover/out/run_$i.png
  python examples/capsules/swiss-landcover/recipe.py
done
python - <<'EOF'
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
cap = Path("examples/capsules/swiss-landcover")
spec = importlib.util.spec_from_file_location("v", cap / "verify.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
imgs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        for p in [cap/"out/run_1.png", cap/"out/run_2.png", cap/"out/swiss_landcover.png"]]
scores, mads = [], []
for i in range(3):
    for j in range(i+1, 3):
        scores.append(v.ssim(imgs[i], imgs[j], data_range=255.0))
        mads.append(float(np.mean(np.abs(imgs[i]-imgs[j]))))
print("pairwise SSIM:", [round(s, 6) for s in scores])
print("pairwise mean-abs:", [round(m, 4) for m in mads])
print("suggest ssim_min =", round(min(scores) - 3*(max(scores)-min(scores)) - 0.002, 4))
print("suggest mean_abs_max =", round(max(mads) * 2 + 0.5, 2))
EOF
```

Record the printed values. Pin `verify_config.json` with the suggested values plus `"calibration": {"vendor": "NVIDIA", "adapter": "<from certificate>", "runs": 3, "pairwise_ssim": [...], "date": "2026-08-07"}`. The single-vendor scope statement is REQUIRED in the file.

- [ ] **Step 3: Choose the reference run, write `make_derivatives.py`, generate derivatives**

Reference = the last full run's PNG; its certificate JSON is copied to `expected/reference.certificate.json` with one added top-level field `"signing_label": "development — not production-signed (v1)"`.

```python
#!/usr/bin/env python3
"""Generate web derivatives for the capsule; hard-fails on budget overrun."""
import sys
from pathlib import Path
from PIL import Image

BUDGETS = {"preview.avif": 350_000, "preview.webp": 500_000, "thumb.webp": 60_000}

def main(src: str) -> int:
    out = Path(__file__).resolve().parent / "expected"
    out.mkdir(exist_ok=True)
    img = Image.open(src).convert("RGB")
    hero = img.resize((1920, round(img.height * 1920 / img.width)), Image.Resampling.LANCZOS)
    thumb = img.resize((640, round(img.height * 640 / img.width)), Image.Resampling.LANCZOS)
    hero.save(out / "preview.avif", quality=60)
    hero.save(out / "preview.webp", quality=82, method=6)
    thumb.save(out / "thumb.webp", quality=80, method=6)
    ok = True
    for name, budget in BUDGETS.items():
        size = (out / name).stat().st_size
        print(f"{name}: {size/1024:.0f} KiB (budget {budget/1024:.0f} KiB)")
        if size > budget:
            ok = False
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
```

Run: `python examples/capsules/swiss-landcover/make_derivatives.py examples/capsules/swiss-landcover/out/swiss_landcover.png`
Expected: exit 0, three files within budget. If AVIF encoding is unavailable in the local Pillow, lower quality/use `pillow-avif-plugin` is NOT allowed (no new deps without asking) — instead drop `preview.avif` from BUDGETS, note it in the README, and record the decision in the commit message.

- [ ] **Step 4: Add the budget test + reference-presence test**

```python
def test_reference_artifacts_present_and_within_budget():
    exp = CAPSULE / "expected"
    cert = json.loads((exp / "reference.certificate.json").read_text(encoding="utf-8"))
    assert cert["signing_label"].startswith("development")
    budgets = {"preview.webp": 500_000, "thumb.webp": 60_000}
    for name, budget in budgets.items():
        assert (exp / name).stat().st_size <= budget


def test_verify_config_pinned_single_vendor():
    cfg = json.loads((CAPSULE / "verify_config.json").read_text(encoding="utf-8"))
    assert isinstance(cfg["ssim_min"], float) and isinstance(cfg["mean_abs_max"], float)
    assert cfg["calibration"]["vendor"] == "NVIDIA" and cfg["calibration"]["runs"] >= 3
```

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short` → all pass.

- [ ] **Step 5: End-to-end verify against the committed reference**

Run: `python examples/capsules/swiss-landcover/verify.py`
Expected: exit 0; report shows all PASS, both honesty labels printed. (Uses `expected/preview.webp` vs the current `out/swiss_landcover.png` — note verify compares the local PNG against the lossless-enough hero WebP; if WebP quantization alone breaks the calibrated threshold, commit `expected/reference.png` additionally and point verify at it — record which path was taken in the README.)

- [ ] **Step 6: Commit**

```bash
git add examples/capsules/swiss-landcover/expected examples/capsules/swiss-landcover/make_derivatives.py examples/capsules/swiss-landcover/verify_config.json tests/test_capsule_swiss_landcover.py
git commit -m "feat(capsule): NVIDIA-calibrated reference render, thresholds, web derivatives"
```

---

### Task 5: Human + agent documentation

**REV 4 AMENDMENTS (BINDING — override conflicting text below):** (a) Pinned install command everywhere it appears: `pip install "forge3d==1.34.0" rasterio pillow scipy numpy` (1.34.0 is the shipped version — NOT 1.30.0; record exact tested versions of the four libs in environment.lock.md from `pip show`). (b) VERIFY.md must document: the trust labels (reference = development-signed, not production-signed in v1; local = locally produced), that `recipe_result.json` is the hash binding between recipe/inputs/final image while the certificate covers the final PT tile pass only, and that non-convergence currently exits with an engine error (traceback) rather than a dedicated code. (c) README's Data section states provenance is "documented, with stated gaps" (spec §17.4) — never the word "verified" for provenance. (d) README's tested-matrix section lists NVIDIA calibration and each agent host as explicit rows marked `NOT_PROVEN`/`untested` until executed against the frozen capsule (Task 6 flips only the rows it actually runs).

**Files:**
- Create: `examples/capsules/swiss-landcover/README.md`
- Create: `examples/capsules/swiss-landcover/VERIFY.md`
- Create: `examples/capsules/swiss-landcover/AGENT_TASK.md`
- Create: `examples/capsules/swiss-landcover/environment.lock.md`
- Test: append doc-integrity test to `tests/test_capsule_swiss_landcover.py`

**Interfaces:**
- Consumes: everything above; `forge3d.__version__` (pin it — run `python -c "import forge3d; print(forge3d.__version__)"` and write the value).
- Produces: the four documents, sequence and content per spec §9/§10.

- [ ] **Step 1: Write the doc-integrity test**

```python
def test_readme_carries_attribution_and_sequence():
    text = (CAPSULE / "README.md").read_text(encoding="utf-8")
    assert "Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft" in text
    assert "Terrain Tiles" in text                      # elevation attribution
    assert "Tier: Free (open core)" in text
    assert "NVIDIA" in text                             # single-vendor scope disclosed
    for anchor in ("## See it", "## Reproduce it", "## Verify it", "## Data"):
        assert anchor in text
    assert "not production-signed" in (CAPSULE / "VERIFY.md").read_text(encoding="utf-8")


def test_agent_task_contract():
    text = (CAPSULE / "AGENT_TASK.md").read_text(encoding="utf-8")
    for required in ("python recipe.py", "python verify.py", "has_gpu",
                     "do not modify recipe parameters", "exit 0"):
        assert required.lower() in text.lower()
```

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -k "readme or agent_task" -v --tb=short` → FAIL (files missing).

- [ ] **Step 2: Write the four documents**

`README.md` sections in order — **See it** (embed `expected/preview.webp`; one paragraph: what the map is, capabilities shown: PROMETHEUS path-traced light field over real terrain, classified 10 m land cover drape, certificate-bearing offscreen render; `Tier: Free (open core)`; hardware: native GPU required, thresholds calibrated on NVIDIA only, expected full-render wall-clock range measured in Task 4 — write the actual number); **Reproduce it** (`pip install "forge3d==<pinned>"` then `python recipe.py`; expected outputs listed); **Verify it** (run `python verify.py`; link VERIFY.md); **Data** (table from `data-manifest.json` + both attribution lines verbatim from `provenance.json`); **Known limitations** (single-vendor calibration; Windows TDR note: long PT runs can lose the device — re-run the command; certificate is development-signed in v1).

`VERIFY.md` — the spec §7 walkthrough: run recipe → compare via verify.py → what each check means (SSIM, mean-abs, engine version, WGSL hashes, degradations) → the two labels verbatim: reference is `development-signed — not production-signed (v1)`, local report is `locally produced`.

`AGENT_TASK.md` — goal, success criteria (recipe exit 0; verify.py exit 0), non-goals (do not modify recipe parameters; do not substitute datasets; do not report success on fallback paths), preflight (`python -c "import forge3d; print(forge3d.has_gpu())"` must print True; `pip install "forge3d==<pinned>"`; ~2 GB free disk), ordered steps with exact commands and per-step failure diagnostics (exit 2 → no GPU adapter: check drivers; exit 3 → report as bug; verify fail → include the printed report), reporting format (platform, adapter from certificate, forge3d version, SSIM, wall-clock), version stamp (`capsule 0.1.0; tested matrix recorded in README`).

`environment.lock.md` — pinned forge3d version, Python floor 3.10, `numpy/rasterio/pillow/scipy` presence note (installed as forge3d example deps), OS/GPU of the calibration machine (from the reference certificate's adapter field).

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short` → all pass.

- [ ] **Step 4: Commit**

```bash
git add examples/capsules/swiss-landcover/README.md examples/capsules/swiss-landcover/VERIFY.md examples/capsules/swiss-landcover/AGENT_TASK.md examples/capsules/swiss-landcover/environment.lock.md tests/test_capsule_swiss_landcover.py
git commit -m "docs(capsule): README, VERIFY, AGENT_TASK, environment lock"
```

---

### Task 6: Acceptance — clean-venv keyless reproduction + agent dry run

**Files:**
- Modify: `examples/capsules/swiss-landcover/README.md` (record tested matrix + wall-clock)
- No new source files.

**Interfaces:**
- Consumes: the complete capsule.
- Produces: recorded acceptance evidence in README ("Tested matrix" section).

- [ ] **Step 1: Clean-venv keyless run (spec §12.2)**

```powershell
python -m venv .venv-capsule-accept
.venv-capsule-accept\Scripts\python -m pip install -U pip maturin
.venv-capsule-accept\Scripts\python -m pip install numpy rasterio pillow scipy
# in-repo build (external users get the pinned wheel instead):
.venv-capsule-accept\Scripts\python -m maturin develop --release
.venv-capsule-accept\Scripts\python examples/capsules/swiss-landcover/recipe.py
.venv-capsule-accept\Scripts\python examples/capsules/swiss-landcover/verify.py
```

Expected: both exit 0 with NO license key present in the environment. Record wall-clock of the recipe run.

- [ ] **Step 2: Full test suite + lint gates**

Run: `python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short` and `python -m pytest tests/test_api_contracts.py -v --tb=short` (contract lane untouched — must still pass). No Rust changes were made, so `cargo fmt --check`/clippy are not triggered; state this in the task report.

- [ ] **Step 3: Agent dry run (spec §12.5, recorded not claimed)**

Run the AGENT_TASK.md steps end-to-end via one Claude Code subagent session on this machine; record host + version in README's "Tested matrix" (`Claude Code <version>, Windows 11, NVIDIA <adapter>`). Cursor's slot in the matrix is recorded as `untested` until run there — the README must say `untested`, not omit it.

- [ ] **Step 4: Update README tested-matrix, run doc test, commit**

```bash
python -m pytest tests/test_capsule_swiss_landcover.py -v --tb=short
git add examples/capsules/swiss-landcover/README.md
git commit -m "docs(capsule): record acceptance evidence and tested matrix"
```

---

## Self-review notes (already applied)

- Spec §4.6 no-fallback → recipe exit codes 2/3/4 + banned-import test (Task 2).
- Spec §6 honesty labels → `signing_label` field + VERIFY.md/verify.py wording tests (Tasks 3-5).
- Spec §8 single-vendor → `verify_config.json` calibration block + test (Task 4).
- Spec §11 budgets → `make_derivatives.py` hard-fail + test (Task 4).
- Spec §13 no telemetry → banned-import test covers network libs (Task 2).
- Deliberate deviations from spec text: none. Open risk: WebP-vs-PNG reference comparison (Task 4 Step 5 records the resolution taken).
