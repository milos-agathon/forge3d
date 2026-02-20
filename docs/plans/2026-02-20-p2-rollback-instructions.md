# P2 Rollback Instructions

Date: 2026-02-20
Phase: P2 of API Consolidation Plan

Each section documents how to cleanly revert a P2 task.

## P2.1: Point Cloud GPU Rendering Path

**Commits**: `3dd51d1`, `ebe27c4`, `604e73f`

**Revert sequence** (reverse chronological):
```bash
git revert 604e73f  # Undo load_points dedup, elev_norm clamp
git revert ebe27c4  # Undo viewer buffer format, remove speculative bindings
git revert 3dd51d1  # Undo PyPointBuffer, create_gpu_buffer
```

**Files affected**:
- `src/pointcloud/renderer.rs` — Remove `create_gpu_buffer()`, `create_viewer_gpu_buffer()`, `gpu_byte_size()`, `MemoryReport`, constants `GPU_FLOATS_PER_VERTEX`/`VIEWER_FLOATS_PER_VERTEX`; restore original `load_copc_points`/`load_ept_points` methods
- `src/lib.rs` — Remove `PyPointBuffer` class and its `m.add_class` registration
- `python/forge3d/__init__.pyi` — Remove `PointBuffer` stub class
- `tests/test_api_contracts.py` — Remove Section 17 (`TestPointCloudBuffer`)

**Verification**:
```bash
maturin develop --release
python -m pytest tests/test_api_contracts.py -x -q
```

## P2.2: COPC LAZ Decompression

**Commit**: `0ae3431`

**Revert**:
```bash
git revert 0ae3431
```

**Files affected**:
- `src/pointcloud/copc_decode.rs` — Delete entire file
- `src/pointcloud/copc.rs` — Revert VLR scanning changes, restore `read_copc_vlr()`
- `src/pointcloud/mod.rs` — Remove `mod copc_decode;`
- `src/lib.rs` — Remove `read_laz_points_info_py` and `copc_laz_enabled_py` functions and their `wrap_pyfunction` registrations
- `Cargo.toml` — Remove `copc_laz = ["dep:laz"]` feature and `laz` optional dependency
- `pyproject.toml` — Remove `copc_laz` from maturin build features
- `python/forge3d/__init__.pyi` — Remove `read_laz_points_info` and `copc_laz_enabled` stubs
- `tests/test_api_contracts.py` — Remove Section 18 (`TestCopcLazDecompression`)

**Verification**:
```bash
maturin develop --release
python -m pytest tests/test_api_contracts.py -x -q
```

## P2.3: Labels Python Bindings

**Commits**: `d864844`, `78f02c2`

**Revert sequence**:
```bash
git revert 78f02c2  # Undo cfg gate fix
git revert d864844  # Undo PyLabelStyle/PyLabelFlags
```

**Files affected**:
- `src/labels/py_bindings.rs` — Delete entire file
- `src/labels/mod.rs` — Remove `pub mod py_bindings;` line
- `src/lib.rs` — Remove `m.add_class::<PyLabelStyle>()` and `m.add_class::<PyLabelFlags>()` registrations
- `python/forge3d/__init__.pyi` — Remove `LabelFlags` and `LabelStyle` stub classes
- `tests/test_api_contracts.py` — Remove Section 19 (`TestLabelBindings`)

**Verification**:
```bash
maturin develop --release
python -m pytest tests/test_api_contracts.py -x -q
```

## P2.4: Deprecation Policy

**Commit**: `bca1ab5`

**Revert**:
```bash
git revert bca1ab5
```

**Files affected**:
- `docs/plans/2026-02-20-deprecation-policy-decision.md` — Delete file

No code changes were made in P2.4. This is a documentation-only revert.

**Verification**: No build or test impact.
