# TV13 — Terrain Population LOD Pipeline Design

**Date:** 2026-03-22
**Epic:** TV13 from `docs/plans/2026-03-16-terrain-viz-epics.md`
**Scope:** Automatic mesh simplification, auto-generated LOD chains for scatter assets, and HLOD for dense distant populations.

---

## 1. Problem

Terrain scatter (TV3) works, but users must provide manual LOD meshes for every scatter asset. That does not scale for dense terrain scenes with many asset types. The missing pieces are:

1. **No mesh simplification** — there is no way to reduce triangle count programmatically.
2. **No auto-LOD** — every `TerrainScatterLevel` requires a user-authored mesh.
3. **No HLOD** — dense distant scatter still renders as thousands of individual instanced draws, wasting draw-call budget.

---

## 2. Current Architecture

### Rust side

- `src/geometry/mod.rs` — `MeshBuffers` (positions, normals, uvs, tangents, indices), primitives, validation, welding, subdivision, displacement. No simplification.
- `src/geometry/py_bindings.rs` — PyO3 bindings for geometry ops, registered via `src/py_module/functions/geometry.rs`.
- `src/terrain/scatter.rs` — `TerrainScatterBatch`, `TerrainScatterLevelSpec` (mesh + max_distance), CPU-side distance-threshold LOD selection in `prepare_draws()`, GPU instanced rendering via `MeshInstancedRenderer`.
- `src/terrain/renderer/scatter.rs` — render pass orchestration, `set_scatter_batches_native()`.

### Python side

- `python/forge3d/geometry.py` — `MeshBuffers` dataclass, `primitive_mesh()`, `validate_mesh()`, `weld_mesh()`, etc.
- `python/forge3d/terrain_scatter.py` — `TerrainScatterLevel`, `TerrainScatterBatch`, `TerrainScatterSource`, placement functions, serialization.

### Key constraint

The scatter renderer consumes only positions and normals from `MeshBuffers` (see `build_gpu_level()` in `scatter.rs:365-428`). UVs and tangents are preserved in `MeshBuffers` for other consumers but do not affect the current scatter path.

---

## 3. Design

### 3.1 TV13.1 — Automatic Mesh Simplification

#### Rust: `src/geometry/simplify.rs`

New module implementing QEM (Quadric Error Metrics) edge-collapse simplification.

**Core function:**

```rust
pub fn simplify_mesh(
    mesh: &MeshBuffers,
    target_ratio: f32,
) -> GeometryResult<MeshBuffers>
```

- `target_ratio` in `(0.0, 1.0]` — fraction of original triangle count to target.
- Returns a new `MeshBuffers` with reduced geometry.
- Normals are recomputed from face normals after collapse (area-weighted vertex normals).
- UVs and tangents: carried through on a best-effort basis. V1 does not promise seam-aware or attribute-weighted collapse. The algorithm assigns quadrics to vertex positions only; UV coordinates are copied from the surviving vertex of each collapse.
- Boundary edges: edges with only one adjacent triangle are penalized (higher collapse cost) to preserve mesh silhouette, but not locked.
- Returns error if `target_ratio` is out of range, mesh is empty, or mesh has no triangles.

**Algorithm outline:**

1. Build per-vertex quadric matrices from incident face planes.
2. Build edge-collapse priority queue (min-heap by quadric error).
3. Penalize boundary edges by scaling their error cost.
4. Collapse edges greedily until triangle count ≤ `target_ratio × original_count` or no more valid collapses exist.
5. Compact vertex/index buffers, recompute normals.

**PyO3 binding:**

```rust
#[pyfunction]
pub fn geometry_simplify_mesh_py(
    mesh: &Bound<'_, PyDict>,
    target_ratio: f32,
) -> PyResult<PyObject>
```

Registered in `src/py_module/functions/geometry.rs`. Added to `EXPECTED_FUNCTIONS` in `tests/test_api_contracts.py` as `"geometry_simplify_mesh_py"`.

#### Python: `python/forge3d/geometry.py`

```python
def simplify_mesh(mesh: MeshBuffers, target_ratio: float) -> MeshBuffers:
    """Simplify a mesh to approximately target_ratio of its original triangle count.

    Uses QEM edge-collapse. Normals are recomputed; UVs are best-effort.
    """
```

#### Python: `python/forge3d/geometry.py` — LOD chain generation

```python
def generate_lod_chain(
    mesh: MeshBuffers,
    ratios: list[float],
    *,
    min_triangles: int = 8,
) -> list[MeshBuffers]:
    """Generate a LOD chain from one high-detail mesh.

    Each level is simplified from the *original* mesh (not cascaded) to avoid
    compound quality loss. Ratios must be in descending order in (0.0, 1.0].

    If a ratio would produce fewer than min_triangles, that level and all
    coarser levels are dropped. Duplicate outputs (same triangle count as a
    prior level) are also dropped.
    """
```

- `ratios[0]` must be `1.0` — LOD 0 is always the original mesh. Raises `ValueError` otherwise.
- Simplifies from original each time, not cascaded.
- `min_triangles` floor prevents degenerate meshes on small assets.
- Deduplicates: if two ratios produce the same triangle count, the coarser one is dropped.

### 3.2 TV13.2 — Auto-Generated LOD Chains for Scatter

#### Python: `python/forge3d/terrain_scatter.py`

```python
def auto_lod_levels(
    mesh: MeshBuffers,
    *,
    lod_count: int = 3,
    lod_distances: list[float] | None = None,
    ratios: list[float] | None = None,
    draw_distance: float | None = None,
    min_triangles: int = 8,
) -> list[TerrainScatterLevel]:
    """Generate LOD levels from one high-detail mesh.

    Parameters
    ----------
    mesh
        The highest-detail mesh. Becomes LOD 0 (full detail).
    lod_count
        Number of LOD levels to generate (including LOD 0).
    lod_distances
        Explicit per-level max_distance values. Length must equal lod_count.
        The final level's distance may be None (open-ended).
        When omitted, distances are derived from draw_distance using
        geometric spacing.
    ratios
        Triangle-count ratios for each level. Length must equal lod_count.
        Default: geometric series from 1.0 down (e.g., [1.0, 0.25, 0.07]
        for 3 levels).
    draw_distance
        Used to derive lod_distances when lod_distances is not provided.
        When a TerrainScatterBatch already has max_draw_distance, pass it
        here so distance thresholds scale with the batch.
    min_triangles
        Minimum triangle count floor passed to generate_lod_chain.

    Returns
    -------
    list[TerrainScatterLevel]
        LOD levels ready for use in TerrainScatterBatch. The final level
        has max_distance=None (open-ended).
    """
```

**Distance derivation** when `lod_distances` is not provided:

- If `draw_distance` is given: `distances[i] = draw_distance × (i+1) / lod_count` with final level open-ended.
- If `draw_distance` is also absent: distances are `[50.0, 150.0, None]` as conservative defaults.

**Integration with existing API:**

- `auto_lod_levels()` returns `list[TerrainScatterLevel]` — the same type users pass to `TerrainScatterBatch` today.
- Manual LOD assets still work unchanged. If a user provides their own `TerrainScatterLevel` list, nothing changes.
- `auto_lod_levels()` is a convenience function, not a requirement.

### 3.3 TV13.3 — HLOD for Dense Distant Populations

#### Problem

Dense scatter scenes (10k+ instances) still render every instance individually. Beyond a certain distance, individual geometry is indistinguishable. HLOD merges distant clusters into cheaper aggregate draws.

#### Rust: `src/terrain/scatter.rs` — HLOD proxy representation

**New types:**

```rust
/// One HLOD cluster: a merged, simplified mesh representing many distant instances.
pub struct GpuHlodCluster {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    center: Vec3,           // cluster centroid in contract space
    radius: f32,            // bounding radius
    vertex_buffer_bytes: u64,
    index_buffer_bytes: u64,
    _vertex_handle: ResourceHandle,
    _index_handle: ResourceHandle,
}

/// HLOD configuration parsed from Python.
pub struct HlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}

/// Cached HLOD state, built once and stored alongside regular LOD levels.
pub struct HlodCache {
    clusters: Vec<GpuHlodCluster>,
    instance_to_cluster: Vec<Option<usize>>,  // indexed by instance, maps to cluster ID
    hlod_distance: f32,
    total_buffer_bytes: u64,
}
```

Note: `GpuHlodCluster` is a dedicated type, not a reuse of `GpuScatterLevel`. HLOD clusters are drawn as single non-instanced draw calls from their own vertex/index buffers. They have no `max_distance` for LOD selection — the HLOD distance threshold is on `HlodCache`, not per-cluster.

**Updated Rust `TerrainScatterBatch::new()` signature:**

```rust
pub fn new(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    levels: Vec<TerrainScatterLevelSpec>,
    transforms_rowmajor: &[[f32; 16]],
    color: [f32; 4],
    max_draw_distance: Option<f32>,
    name: Option<String>,
    hlod_config: Option<HlodConfig>,  // NEW — None preserves pre-TV13 behavior
) -> Result<Self>
```

When `hlod_config` is `Some`, the constructor builds the `HlodCache` (spatial clustering + mesh merge + simplification + GPU upload). When `None`, `hlod_cache` is `None` and all behavior is unchanged.

**Validation:** `hlod_distance` must be positive, finite, and less than `max_draw_distance` (when provided). `cluster_radius` must be positive and finite. `simplify_ratio` must be in `(0.0, 1.0]`.

**Key design decisions:**

1. **Computed once, cached** — HLOD is built in `TerrainScatterBatch::new()` and stored as `hlod_cache: Option<HlodCache>`. The `update_transforms()` method gains `device` and `queue` parameters to support HLOD rebuild:

    ```rust
    pub fn update_transforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        transforms_rowmajor: &[[f32; 16]],
    ) -> Result<()>
    ```

    When `hlod_cache` is `None`, the `device`/`queue` params are unused and the method remains as cheap as before. When HLOD is present, rebuild happens immediately (HLOD is a batch-level preprocessing step, not a per-frame operation). This is an internal Rust API change; the Python-side `update_transforms` call already goes through `py_api.rs` which has `device`/`queue` access.

2. **Separate draw representation** — HLOD clusters are *not* another `TerrainScatterLevelSpec`. Each cluster is a unique merged mesh (baked instance transforms into vertex positions). They use the same shader and vertex format (`VertexPN`) but are drawn as single non-instanced draw calls from their own vertex/index buffers.

3. **Clustering algorithm** — spatial grid clustering in contract space:
   - Partition instances into grid cells of size `cluster_radius`.
   - For each cell with ≥ 2 instances, merge instance geometry: transform each instance's mesh vertices by its transform, concatenate into one `MeshBuffers`, then simplify to `simplify_ratio × sum_of_triangle_counts`.
   - Cells with 1 instance are left as regular instanced draws (not assigned to any cluster).
   - Build `instance_to_cluster: Vec<Option<usize>>` mapping each instance index to its cluster ID (or `None` for unclustered instances).

4. **Runtime selection in `prepare_draws()`** — for each instance:
   - If `distance > hlod_distance` and `instance_to_cluster[i].is_some()` → skip individual draw (the cluster covers it).
   - If `distance > hlod_distance` and `instance_to_cluster[i].is_none()` → normal per-instance LOD selection (singleton beyond HLOD range).
   - If `distance ≤ hlod_distance` → normal per-instance LOD selection (unchanged).
   - After instance loop: draw each `GpuHlodCluster` whose centroid is within `max_draw_distance` as a single non-instanced draw call.

5. **Memory tracking** — `HlodCache.total_buffer_bytes` is included in `TerrainScatterMemoryReport.total_buffer_bytes()`. New fields:
   - `hlod_cluster_count: u32`
   - `hlod_buffer_bytes: u64`
   - `total_buffer_bytes()` is updated to return `vertex + index + instance + hlod_buffer_bytes`.

#### Python: `python/forge3d/terrain_scatter.py`

```python
@dataclass
class HLODPolicy:
    """Configuration for HLOD cluster generation.

    Attributes
    ----------
    hlod_distance : float
        Distance threshold beyond which instances are replaced by HLOD clusters.
    cluster_radius : float
        Spatial grid cell size for grouping instances.
    simplify_ratio : float
        Target triangle ratio for merged cluster meshes (0.0, 1.0].
        Default 0.1 (aggressive — distant clusters need far less detail).
    """
    hlod_distance: float
    cluster_radius: float
    simplify_ratio: float = 0.1
```

Integrated into `TerrainScatterBatch`:

```python
class TerrainScatterBatch:
    def __init__(
        self,
        levels: list[TerrainScatterLevel],
        transforms: np.ndarray,
        *,
        name: str | None = None,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        max_draw_distance: float | None = None,
        hlod: HLODPolicy | None = None,  # NEW
    ):
```

When `hlod` is provided, the native dict includes HLOD parameters:

```python
# In to_native_dict():
{
    ...,  # existing fields (name, color, max_draw_distance, transforms, levels)
    "hlod": {
        "hlod_distance": float,
        "cluster_radius": float,
        "simplify_ratio": float,
    }
}
```

The Rust `py_api.rs` parser extracts the `"hlod"` key and constructs `HlodConfig`. When absent, `hlod_config` is `None`.

When `hlod` is `None`, behavior is identical to pre-TV13 (no HLOD, no extra memory).

**Note on `auto_lod_levels` and HLOD interaction:** `auto_lod_levels()` is independent of HLOD policy. When combining both features, users should set `draw_distance` equal to `hlod_distance` so auto-generated LOD levels cover only the non-HLOD range. LOD levels beyond `hlod_distance` still work correctly (HLOD takes priority in `prepare_draws()`), but they waste simplification effort on meshes that will never be individually drawn at that distance.

**`__all__` updates:** `HLODPolicy` and `auto_lod_levels` are added to `terrain_scatter.py`'s `__all__` list. `simplify_mesh` and `generate_lod_chain` are added to `geometry.py` exports.

---

## 4. File Changes Summary

### New files

| File | Content |
|---|---|
| `src/geometry/simplify.rs` | QEM edge-collapse simplification algorithm |

### Modified files

| File | Changes |
|---|---|
| `src/geometry/mod.rs` | Add `mod simplify;` and `pub use simplify::simplify_mesh;` |
| `src/geometry/py_bindings.rs` | Add `geometry_simplify_mesh_py` binding |
| `src/py_module/functions/geometry.rs` | Register `geometry_simplify_mesh_py` |
| `python/forge3d/geometry.py` | Add `simplify_mesh()` and `generate_lod_chain()` |
| `python/forge3d/terrain_scatter.py` | Add `auto_lod_levels()`, `HLODPolicy`, update `TerrainScatterBatch` |
| `src/terrain/scatter.rs` | Add `HlodCluster`, `HlodCache`, HLOD build/draw logic, memory tracking |
| `src/terrain/renderer/scatter.rs` | HLOD cluster draw path in render pass |
| `src/terrain/renderer/py_api.rs` | Parse HLOD config from Python dict |
| `tests/test_api_contracts.py` | Add `"geometry_simplify_mesh_py"` to geometry contract list |

### New test files

| File | Content |
|---|---|
| `tests/test_terrain_tv13_lod_pipeline.py` | Full test coverage for TV13.1, TV13.2, TV13.3 |

### New example

| File | Content |
|---|---|
| `examples/terrain_tv13_lod_pipeline_demo.py` | Demo using real DEM with auto-LOD scatter and HLOD |

---

## 5. Test Plan

### TV13.1 — Mesh Simplification

- **Rust unit tests** in `src/geometry/simplify.rs`:
  - Simplify a box (12 tris) to ratio 0.5 → ≤ 6 triangles.
  - Simplify a sphere (high-res) → output has normals, vertex count < input.
  - Ratio 1.0 → output equals input.
  - Ratio too low on small mesh → returns minimum viable mesh (not empty).
  - Invalid inputs (empty mesh, ratio ≤ 0 or > 1) → error.
  - Boundary preservation: simplified open mesh retains boundary edges.

- **Python integration tests**:
  - `simplify_mesh(cone_mesh, 0.5)` returns valid `MeshBuffers` with fewer triangles.
  - Contract test: `geometry_simplify_mesh_py` exists in native module.

### TV13.2 — Auto LOD Chains

- **Python tests**:
  - `generate_lod_chain(sphere, [1.0, 0.25, 0.07])` returns 3 meshes with decreasing triangle counts.
  - `min_triangles` floor: tiny mesh with `[1.0, 0.01]` drops the second level.
  - Deduplication: two ratios producing the same count → only one level.
  - `generate_lod_chain` enforces `ratios[0] == 1.0` or documents that first output may be simplified.
  - `auto_lod_levels(mesh, lod_count=3, draw_distance=300.0)` returns 3 `TerrainScatterLevel` with distances derived from 300.0.
  - `auto_lod_levels` with explicit `lod_distances` parameter uses those distances directly.
  - `auto_lod_levels` with explicit `ratios` parameter uses those ratios.
  - Integration: `auto_lod_levels` output feeds directly into `TerrainScatterBatch` and renders.

### TV13.3 — HLOD

- **Rust unit tests**:
  - Spatial clustering groups nearby instances into cells.
  - HLOD cluster mesh has merged geometry with baked transforms.
  - `instance_to_cluster` mapping is correct for clustered and singleton instances.
  - Memory report includes HLOD fields; `total_buffer_bytes()` includes `hlod_buffer_bytes`.

- **Python integration tests**:
  - Batch with `HLODPolicy` renders without error.
  - HLOD reduces effective draw count at distance (visible in stats).
  - HLOD memory is tracked and queryable.
  - `hlod=None` preserves pre-TV13 behavior exactly.
  - `to_native_dict()` correctly serializes HLOD policy.
  - Invalid `HLODPolicy` (negative `hlod_distance`, `simplify_ratio` out of range, `hlod_distance ≥ max_draw_distance`) raises errors.

### End-to-end

- **Example demo** renders real DEM terrain with auto-LOD scatter and HLOD, producing PNG output.
- **Image output test**: rendered frame is non-empty, has expected dimensions, pixels differ from terrain-only baseline.

---

## 6. Backward Compatibility

- All changes are additive. No existing API is removed or renamed.
- `TerrainScatterBatch` without `hlod` parameter behaves identically to pre-TV13.
- Manual `TerrainScatterLevel` lists still work. `auto_lod_levels()` is optional.
- `generate_lod_chain()` and `simplify_mesh()` are new geometry functions with no effect on existing code.
- Scatter stats and memory reports gain new fields but existing fields are unchanged.
