# Audit Snapshot (Generated)

- Generated (UTC): 2026-02-19 08:44:09Z
- Commit: `988c9b525e262247db69aba966ad4df14f6ce56d`
- Branch: `main`
- Script: `scripts/generate_audit_snapshot.py`

## Section 1: Codebase Scale Overview

| Metric | Count |
|---|---:|
| Rust source files (`src/**/*.rs`) | 499 |
| Rust top-level modules (`pub mod` in `src/lib.rs`) | 40 |
| Python package files (`python/forge3d/**/*.py`) | 54 |
| PyO3 classes (`pyclass`) | 38 |
| PyO3 functions (`#[pyfunction]`) | 100 |
| Rust files with PyO3 annotations | 33 (6.6% of Rust files) |
| Test files (Rust + Python, naming/path heuristic) | 77 |
| Test functions (Rust `#[test]` + Python `def test_*`) | 1450 |

Architecture signals:
- Rust core + PyO3 bindings: detected
- Viewer IPC (TCP/NDJSON path): detected
- Rust top-level modules: accel, animation, bundle, camera, cli, colormap, converters, core, export, external_image, formats, geo, geometry, import, io, labels, lighting, loaders, math, mesh ...

## Section 3: Python Wrapper Classification

Classification rules:
- `Orchestration`: known workflow/entry modules (`__init__`, `render`, `terrain_demo`, `viewer`) unless overridden.
- `Pure Python`: no native-binding references/calls detected.
- `Thin wrapper`: native references present but limited call surface in a small module.
- `Thick wrapper`: larger module with non-trivial native integration.

| Category | Count | Files |
|---|---:|---|
| Pure Python | 24 | _validate, bench, bundle, colors, config, denoise, export, guiding, interactive, legend, map_plate, materials, mem, north_arrow, pointcloud, presets, scale_bar, style, style_expressions, terrain_params, terrain_pbr_pom, textures, tiles3d, viewer_ipc |
| Thick wrapper | 10 | buildings, cog, crs, geometry, io, lighting, mesh, path_tracing, sdf, vector |
| Thin wrapper | 4 | _gpu, _memory, _native, animation |
| Orchestration | 4 | __init__, render, terrain_demo, viewer |

Per-module details:

| File | Category | LOC | Native refs | Native calls | Reason |
|---|---|---:|---:|---:|---|
| `python/forge3d/__init__.py` | Orchestration | 460 | 2 | 0 | known workflow/entry module |
| `python/forge3d/_gpu.py` | Thin wrapper | 90 | 4 | 0 | small module with limited native call surface |
| `python/forge3d/_memory.py` | Thin wrapper | 127 | 4 | 0 | small module with limited native call surface |
| `python/forge3d/_native.py` | Thin wrapper | 38 | 7 | 0 | small module with limited native call surface |
| `python/forge3d/_validate.py` | Pure Python | 423 | 0 | 0 | no native binding references detected |
| `python/forge3d/animation.py` | Thin wrapper | 205 | 2 | 0 | small module with limited native call surface |
| `python/forge3d/bench.py` | Pure Python | 227 | 0 | 0 | no native binding references detected |
| `python/forge3d/buildings.py` | Thick wrapper | 574 | 1 | 5 | non-trivial module with native integration |
| `python/forge3d/bundle.py` | Pure Python | 361 | 0 | 0 | no native binding references detected |
| `python/forge3d/cog.py` | Thick wrapper | 378 | 3 | 0 | non-trivial module with native integration |
| `python/forge3d/colors.py` | Pure Python | 65 | 0 | 0 | no native binding references detected |
| `python/forge3d/config.py` | Pure Python | 726 | 0 | 0 | no native binding references detected |
| `python/forge3d/crs.py` | Thick wrapper | 354 | 2 | 0 | non-trivial module with native integration |
| `python/forge3d/denoise.py` | Pure Python | 148 | 0 | 0 | no native binding references detected |
| `python/forge3d/export.py` | Pure Python | 627 | 0 | 0 | no native binding references detected |
| `python/forge3d/geometry.py` | Thick wrapper | 577 | 26 | 21 | non-trivial module with native integration |
| `python/forge3d/guiding.py` | Pure Python | 47 | 0 | 0 | no native binding references detected |
| `python/forge3d/interactive.py` | Pure Python | 607 | 0 | 0 | no native binding references detected |
| `python/forge3d/io.py` | Thick wrapper | 728 | 10 | 6 | non-trivial module with native integration |
| `python/forge3d/legend.py` | Pure Python | 178 | 0 | 0 | no native binding references detected |
| `python/forge3d/lighting.py` | Thick wrapper | 976 | 33 | 6 | non-trivial module with native integration |
| `python/forge3d/map_plate.py` | Pure Python | 357 | 0 | 0 | no native binding references detected |
| `python/forge3d/materials.py` | Pure Python | 29 | 0 | 0 | no native binding references detected |
| `python/forge3d/mem.py` | Pure Python | 74 | 0 | 0 | no native binding references detected |
| `python/forge3d/mesh.py` | Thick wrapper | 677 | 5 | 2 | non-trivial module with native integration |
| `python/forge3d/north_arrow.py` | Pure Python | 188 | 0 | 0 | no native binding references detected |
| `python/forge3d/path_tracing.py` | Thick wrapper | 798 | 1 | 0 | non-trivial module with native integration |
| `python/forge3d/pointcloud.py` | Pure Python | 691 | 0 | 0 | no native binding references detected |
| `python/forge3d/presets.py` | Pure Python | 315 | 0 | 0 | no native binding references detected |
| `python/forge3d/render.py` | Orchestration | 2543 | 4 | 0 | known workflow/entry module |
| `python/forge3d/scale_bar.py` | Pure Python | 163 | 0 | 0 | no native binding references detected |
| `python/forge3d/sdf.py` | Thick wrapper | 555 | 7 | 1 | non-trivial module with native integration |
| `python/forge3d/style.py` | Pure Python | 992 | 0 | 0 | no native binding references detected |
| `python/forge3d/style_expressions.py` | Pure Python | 718 | 0 | 0 | no native binding references detected |
| `python/forge3d/terrain_demo.py` | Orchestration | 1123 | 0 | 0 | known workflow/entry module |
| `python/forge3d/terrain_params.py` | Pure Python | 1746 | 0 | 0 | no native binding references detected |
| `python/forge3d/terrain_pbr_pom.py` | Pure Python | 51 | 0 | 0 | no native binding references detected |
| `python/forge3d/textures.py` | Pure Python | 105 | 0 | 0 | no native binding references detected |
| `python/forge3d/tiles3d.py` | Pure Python | 405 | 0 | 0 | no native binding references detected |
| `python/forge3d/vector.py` | Thick wrapper | 448 | 1 | 0 | non-trivial module with native integration |
| `python/forge3d/viewer.py` | Orchestration | 573 | 4 | 0 | known workflow/entry module |
| `python/forge3d/viewer_ipc.py` | Pure Python | 942 | 0 | 0 | no native binding references detected |

## Section 5: Duplicate Implementations (Rust vs Python)

Note: `Likely route` is an inference from binding/export/call signals, not full runtime tracing.

| Feature | Rust present | Python present | Rust bound to Python | Public Python API | Native-call signal in Python | Likely route | Confidence |
|---|---|---|---|---|---|---|---|
| SVG export | Yes | Yes | No | Yes | No | Python path (inferred) | Medium |
| Mapbox style parser | Yes | Yes | No | Yes | No | Python path (inferred) | Medium |
| 3D Tiles parser | Yes | Yes | No | No | No | Unknown (trace call sites) | Low |
| Bundle format | Yes | Yes | No | Yes | No | Python path (inferred) | Medium |
| SDF raymarcher / hybrid render | Yes | Yes | Yes | No | Yes | Both (inferred) | Medium |
| Path tracer | Yes | Yes | Yes | Yes | Yes | Both (inferred) | Medium |
| Point cloud LOD/rendering | Yes | Yes | Yes | No | No | Rust-bound but not public API (inferred) | Low |
| Denoise | Yes | Yes | No | No | No | Unknown (trace call sites) | Low |

