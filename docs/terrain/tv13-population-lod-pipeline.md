# TV13: Terrain Population LOD Pipeline

Epic TV13 adds automatic mesh simplification, auto-generated LOD chains, and HLOD (Hierarchical Level of Detail) clustering to the terrain scatter system. Users no longer need to author manual LOD meshes for every scatter asset.

## What shipped

### TV13.1 — Automatic Mesh Simplification

- QEM (Quadric Error Metrics) edge-collapse simplification in Rust: `src/geometry/simplify.rs`
- Simplifies any `MeshBuffers` to a target fraction of its original triangle count
- Area-weighted vertex normals are recomputed after collapse
- UVs are carried through best-effort (surviving vertex)
- Boundary edges are penalized (10x cost) to preserve mesh silhouette
- Python wrapper: `forge3d.geometry.simplify_mesh()`
- LOD chain generator: `forge3d.geometry.generate_lod_chain()` — simplifies from the original mesh at each level (no cascaded quality loss)

### TV13.2 — Auto-Generated LOD Chains for Scatter

- `forge3d.terrain_scatter.auto_lod_levels()` generates `TerrainScatterLevel` lists from a single high-detail mesh
- Configurable LOD count, distance thresholds, and simplification ratios
- Geometric distance spacing derived from `draw_distance` when distances are not explicit
- Drop-in replacement for manually authored LOD level lists
- Manual LOD assets still work unchanged — `auto_lod_levels()` is a convenience, not a requirement

### TV13.3 — HLOD for Dense Distant Populations

- Spatial grid clustering merges distant scatter instances into cheaper aggregate representations
- Each HLOD cluster is a single non-instanced draw call from its own merged vertex/index buffers
- Clusters are built once at batch creation time (not per-frame)
- Runtime selection: clusters activate when the entire cluster (including mesh extents at per-instance scale) is beyond `hlod_distance`; near instances always render individually through normal LOD selection
- New stats fields: `hlod_cluster_draws`, `hlod_covered_instances`, `effective_draws`
- New memory fields: `hlod_buffer_bytes`, `hlod_cluster_count`
- Full plumbing through offscreen renderer, interactive viewer, and IPC paths
- `HLODPolicy` dataclass controls all HLOD behavior; `hlod=None` preserves pre-TV13 baseline exactly

## Public API

### Mesh Simplification

```python
from forge3d.geometry import simplify_mesh, generate_lod_chain, primitive_mesh

# Simplify a mesh to 25% of its original triangle count
sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
simplified = simplify_mesh(sphere, target_ratio=0.25)
print(f"{sphere.triangle_count} -> {simplified.triangle_count} triangles")

# Generate a 3-level LOD chain from one mesh
chain = generate_lod_chain(sphere, ratios=[1.0, 0.25, 0.07])
for i, mesh in enumerate(chain):
    print(f"LOD {i}: {mesh.triangle_count} triangles")
```

### Auto LOD Levels

```python
from forge3d import terrain_scatter as ts
from forge3d.geometry import primitive_mesh

tree = primitive_mesh("cone", radial_segments=24)

# Generate 3 LOD levels with automatic distance spacing
levels = ts.auto_lod_levels(tree, lod_count=3, draw_distance=300.0)
# Result: [TerrainScatterLevel(mesh=tree, max_distance=~33),
#          TerrainScatterLevel(mesh=simplified_25%, max_distance=~100),
#          TerrainScatterLevel(mesh=simplified_7%, max_distance=None)]

# Use with explicit distances
levels = ts.auto_lod_levels(
    tree,
    lod_count=3,
    lod_distances=[50.0, 150.0, None],
)

# Use with explicit ratios
levels = ts.auto_lod_levels(
    tree,
    lod_count=3,
    ratios=[1.0, 0.3, 0.05],
    draw_distance=200.0,
)
```

### HLOD Clustering

```python
from forge3d import terrain_scatter as ts

batch = ts.TerrainScatterBatch(
    name="dense_forest",
    levels=levels,
    transforms=transforms,
    max_draw_distance=1000.0,
    hlod=ts.HLODPolicy(
        hlod_distance=200.0,     # beyond this, use HLOD clusters
        cluster_radius=80.0,     # spatial grid cell size
        simplify_ratio=0.1,      # aggressive simplification for distant clusters
    ),
)
```

### Full Scatter Pipeline (Before and After TV13)

**Before TV13** — manual LOD meshes required:

```python
tree_hi = primitive_mesh("cone", radial_segments=24)
tree_lo = primitive_mesh("box")  # user must author this manually

batch = ts.TerrainScatterBatch(
    levels=[
        ts.TerrainScatterLevel(mesh=tree_hi, max_distance=100.0),
        ts.TerrainScatterLevel(mesh=tree_lo),
    ],
    transforms=transforms,
)
```

**After TV13** — one mesh, automatic LODs and HLOD:

```python
tree = primitive_mesh("cone", radial_segments=24)

batch = ts.TerrainScatterBatch(
    levels=ts.auto_lod_levels(tree, lod_count=3, draw_distance=300.0),
    transforms=transforms,
    max_draw_distance=1000.0,
    hlod=ts.HLODPolicy(
        hlod_distance=300.0,
        cluster_radius=80.0,
    ),
)
```

### Stats and Memory Reporting

```python
# After rendering
stats = renderer.get_scatter_stats()
memory = renderer.get_scatter_memory_report()

# New TV13 fields in stats:
stats["hlod_cluster_draws"]       # HLOD clusters drawn this frame
stats["hlod_covered_instances"]   # instances suppressed by active clusters
stats["effective_draws"]          # individual LOD draws + HLOD cluster draws

# New TV13 fields in memory report:
memory["hlod_cluster_count"]      # number of HLOD clusters
memory["hlod_buffer_bytes"]       # GPU memory used by HLOD cluster buffers
memory["total_buffer_bytes"]      # now includes HLOD buffer bytes
```

## Parameters Reference

### `simplify_mesh(mesh, target_ratio)`

| Parameter | Type | Description |
|---|---|---|
| `mesh` | `MeshBuffers` | Input mesh to simplify |
| `target_ratio` | `float` | Fraction of original triangles to target, in `(0.0, 1.0]` |
| **Returns** | `MeshBuffers` | Simplified mesh with recomputed normals |

### `generate_lod_chain(mesh, ratios, *, min_triangles=8)`

| Parameter | Type | Description |
|---|---|---|
| `mesh` | `MeshBuffers` | Source mesh (LOD 0) |
| `ratios` | `list[float]` | Triangle ratios, descending, starting with `1.0` |
| `min_triangles` | `int` | Floor — levels below this are dropped |
| **Returns** | `list[MeshBuffers]` | LOD chain (may be shorter than `ratios` due to floor/dedup) |

### `auto_lod_levels(mesh, *, lod_count=3, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mesh` | `MeshBuffers` | — | Highest-detail mesh (LOD 0) |
| `lod_count` | `int` | `3` | Number of LOD levels including LOD 0 |
| `lod_distances` | `list[float\|None]\|None` | `None` | Explicit per-level distances |
| `ratios` | `list[float]\|None` | `None` | Explicit per-level simplification ratios |
| `draw_distance` | `float\|None` | `None` | Used to derive distances via geometric spacing |
| `min_triangles` | `int` | `8` | Floor passed to `generate_lod_chain` |
| **Returns** | `list[TerrainScatterLevel]` | — | Ready for `TerrainScatterBatch` |

### `HLODPolicy`

| Field | Type | Default | Description |
|---|---|---|---|
| `hlod_distance` | `float` | — | Distance threshold for HLOD activation |
| `cluster_radius` | `float` | — | Spatial grid cell size for clustering |
| `simplify_ratio` | `float` | `0.1` | Simplification ratio for merged cluster meshes |

## Architecture

### Rust side

- `src/geometry/simplify.rs` — QEM algorithm: per-vertex quadrics, edge-collapse priority queue, boundary penalization, compact output with recomputed normals
- `src/geometry/py_bindings.rs` — `geometry_simplify_mesh_py` PyO3 binding
- `src/terrain/scatter.rs` — `HlodConfig`, `HlodCache`, `GpuHlodCluster` types; spatial grid clustering in `build_hlod_cache()`; three-pass `prepare_draws()` with cluster activation, instance skip, and cluster draw; extended `TerrainScatterBatchStats` and `TerrainScatterMemoryReport`
- `src/terrain/renderer/scatter.rs` — HLOD cluster draw path in `render_scatter_pass()`; `TerrainScatterUploadBatch` carries `hlod_config`
- `src/terrain/renderer/py_api.rs` — Parses `"hlod"` dict from Python batch config
- `src/viewer/terrain/scene/scatter.rs` — Viewer-side HLOD draw path
- `src/viewer/ipc/protocol/payloads.rs` — `hlod` field in `IpcTerrainScatterBatch`
- `src/viewer/ipc/protocol/translate/terrain.rs` — Maps IPC HLOD payload to viewer config

### Python side

- `python/forge3d/geometry.py` — `simplify_mesh()`, `generate_lod_chain()`
- `python/forge3d/terrain_scatter.py` — `HLODPolicy`, `auto_lod_levels()`, updated `TerrainScatterBatch`

## Backward Compatibility

All changes are additive. No existing API is removed or renamed.

- `TerrainScatterBatch` without `hlod` parameter behaves identically to pre-TV13
- Manual `TerrainScatterLevel` lists still work; `auto_lod_levels()` is optional
- `simplify_mesh()` and `generate_lod_chain()` are new geometry functions with no effect on existing code
- Scatter stats and memory reports gain new fields but existing fields are unchanged
- All new HLOD stats fields are `0` when `hlod=None`

## Test Coverage

- 8 Rust unit tests for QEM simplification (`src/geometry/simplify.rs`)
- 17 Rust tests for scatter and HLOD (scatter types, IPC parsing, backward compat)
- 23 Python tests in `tests/test_terrain_tv13_lod_pipeline.py`:
  - `TestSimplifyMesh` (3 tests): reduce triangles, preserve normals, ratio 1.0
  - `TestGenerateLodChain` (5 tests): decreasing counts, min_triangles floor, deduplication, validation
  - `TestAutoLodLevels` (4 tests): defaults, explicit distances, explicit ratios, batch integration
  - `TestHLODPolicy` (5 tests): creation, serialization, validation, viewer payload
  - `TestHLODRendering` (3 tests): baseline preservation, stats reporting, memory tracking
  - `TestEndToEndImageOutput` (1 test): auto-LOD scatter renders non-empty image

## Example

`examples/terrain_tv13_lod_pipeline_demo.py` — end-to-end demo using Mt. Fuji DEM:

```bash
python examples/terrain_tv13_lod_pipeline_demo.py
```

Demonstrates:
1. QEM simplification at multiple ratios (sphere: 1024 -> 511 -> 256 -> 102 triangles)
2. LOD chain generation from a single cone mesh
3. `auto_lod_levels()` producing scatter-ready LOD levels
4. Side-by-side rendering: baseline (no HLOD) vs HLOD-enabled
5. Stats comparison showing HLOD cluster draws, covered instances, and effective draws. Note: the HLOD benefit is most visible at high instance counts — the demo uses a moderate scene to keep runtime short, so `effective_draws` may increase slightly due to HLOD cluster overhead. In production-scale scenes (10k+ instances), HLOD reduces draw-call pressure significantly.
6. PNG output to `examples/out/terrain_tv13_lod_pipeline/`
