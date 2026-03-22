# TV13 — Terrain Population LOD Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic mesh simplification (QEM), auto-generated LOD chains, and HLOD clustering to the terrain scatter system so users no longer need to author manual LOD meshes.

**Architecture:** Rust-side QEM simplification in `src/geometry/simplify.rs` exposed via PyO3; Python-side LOD chain generation and scatter integration in `python/forge3d/geometry.py` and `python/forge3d/terrain_scatter.py`; Rust-side HLOD clustering and merged-mesh drawing in `src/terrain/scatter.rs` with plumbing through both the offscreen renderer and the interactive viewer paths.

**Tech Stack:** Rust, wgpu, PyO3/maturin, Python 3.10+, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-tv13-terrain-population-lod-pipeline-design.md`

---

## Prerequisites

**Build the native extension before running any Python tests:**

```bash
cd C:/Users/milos/forge3d
maturin develop --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz --profile release-lto
```

**Run Rust tests:**

```bash
cargo test --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz
```

**Run Python tests:**

```bash
python -m pytest tests/test_terrain_tv13_lod_pipeline.py -v
```

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `src/geometry/simplify.rs` | QEM edge-collapse mesh simplification algorithm |
| `tests/test_terrain_tv13_lod_pipeline.py` | All Python tests for TV13 (simplify, LOD chain, auto_lod_levels, HLOD) |
| `examples/terrain_tv13_lod_pipeline_demo.py` | End-to-end demo with real DEM, auto-LOD scatter, HLOD, PNG output |

### Modified files

| File | What changes |
|---|---|
| `src/geometry/mod.rs` | Add `mod simplify` + `pub use simplify::simplify_mesh` |
| `src/geometry/py_bindings.rs` | Add `geometry_simplify_mesh_py` PyO3 function |
| `src/py_module/functions/geometry.rs` | Register `geometry_simplify_mesh_py` |
| `python/forge3d/geometry.py` | Add `simplify_mesh()`, `generate_lod_chain()` |
| `python/forge3d/terrain_scatter.py` | Add `HLODPolicy`, `auto_lod_levels()`, update `TerrainScatterBatch` |
| `src/terrain/scatter.rs` | Add HLOD types, cluster build, prepare_draws HLOD path, stats, memory |
| `src/terrain/renderer/scatter.rs` | Add `hlod_config` to `TerrainScatterUploadBatch`, HLOD draw in render pass |
| `src/terrain/renderer/py_api.rs` | Parse `"hlod"` dict from Python batch config |
| `src/viewer/terrain/scene/scatter.rs` | Pass `hlod_config` through viewer scatter path, HLOD draw |
| `src/viewer/viewer_enums/config.rs` | Add `hlod_config` to `ViewerTerrainScatterBatchConfig` |
| `src/viewer/ipc/protocol/payloads.rs` | Add `hlod` to `IpcTerrainScatterBatch` |
| `src/viewer/ipc/protocol/translate/terrain.rs` | Map IPC HLOD to viewer config |
| `tests/test_api_contracts.py` | Add `"geometry_simplify_mesh_py"` to `GEOMETRY_FUNCTIONS` |

---

## Task 1: Create worktree and test scaffold

**Files:**
- Create: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Create epic-13 worktree**

```bash
cd C:/Users/milos/forge3d
git worktree add ../forge3d-epic-13 -b epic-13
cd ../forge3d-epic-13
```

- [ ] **Step 2: Create empty test file with skip guard**

Create `tests/test_terrain_tv13_lod_pipeline.py`:

```python
"""TV13 — Terrain Population LOD Pipeline tests.

Covers:
  - TV13.1: QEM mesh simplification (Rust) and Python wrapper
  - TV13.2: LOD chain generation and auto_lod_levels
  - TV13.3: HLOD clustering, rendering, stats, and memory tracking
"""
from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE

if not NATIVE_AVAILABLE:
    pytest.skip(
        "TV13 tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

ts = f3d.terrain_scatter
```

- [ ] **Step 3: Verify test file loads (should collect 0 tests)**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py -v`
Expected: `no tests ran` (0 collected, no errors)

- [ ] **Step 4: Commit**

```bash
git add tests/test_terrain_tv13_lod_pipeline.py
git commit -m "test(tv13): add empty test scaffold for terrain population LOD pipeline"
```

---

## Task 2: QEM mesh simplification — Rust core

**Files:**
- Create: `src/geometry/simplify.rs`
- Modify: `src/geometry/mod.rs`

- [ ] **Step 1: Add module declaration in mod.rs**

In `src/geometry/mod.rs`, after line `mod weld;`, add:

```rust
mod simplify;
```

And after the existing `pub use weld::{...};` line, add:

```rust
pub use simplify::simplify_mesh;
```

- [ ] **Step 2: Create `src/geometry/simplify.rs` with the QEM algorithm**

```rust
//! QEM (Quadric Error Metrics) edge-collapse mesh simplification.
//!
//! Simplifies a mesh to a target fraction of its original triangle count.
//! Normals are recomputed post-collapse. UVs are carried through best-effort
//! (copied from the surviving vertex, no seam-aware weighting).
//! Boundary edges are penalized but not locked.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use glam::Vec3;

use super::{GeometryError, GeometryResult, MeshBuffers};

/// Symmetric 4×4 matrix stored as 10 upper-triangular elements.
/// Used for quadric error computation.
#[derive(Debug, Clone, Copy)]
struct Quadric {
    // Upper triangle of symmetric 4x4: a00 a01 a02 a03 a11 a12 a13 a22 a23 a33
    data: [f64; 10],
}

impl Quadric {
    fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    fn from_plane(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        Self {
            data: [
                nx * nx, nx * ny, nx * nz, nx * d,
                ny * ny, ny * nz, ny * d,
                nz * nz, nz * d,
                d * d,
            ],
        }
    }

    fn add(&self, other: &Quadric) -> Quadric {
        let mut result = Quadric::zero();
        for i in 0..10 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    fn scale(&self, factor: f64) -> Quadric {
        let mut result = *self;
        for v in result.data.iter_mut() {
            *v *= factor;
        }
        result
    }

    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let d = &self.data;
        // v^T Q v where v = [x, y, z, 1]
        x * (d[0] * x + d[1] * y + d[2] * z + d[3])
            + y * (d[1] * x + d[4] * y + d[5] * z + d[6])
            + z * (d[2] * x + d[5] * y + d[7] * z + d[8])
            + (d[3] * x + d[6] * y + d[8] * z + d[9])
    }
}

/// An edge collapse candidate in the priority queue.
#[derive(Debug, Clone)]
struct CollapseCandidate {
    error: f64,
    v0: u32,
    v1: u32,
    target_pos: [f32; 3],
    generation: u32,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        other
            .error
            .partial_cmp(&self.error)
            .unwrap_or(Ordering::Equal)
    }
}

/// Simplify a mesh to approximately `target_ratio` of its original triangle count.
///
/// - `target_ratio` must be in `(0.0, 1.0]`.
/// - Returns a new `MeshBuffers` with reduced geometry.
/// - Normals are recomputed (area-weighted vertex normals).
/// - UVs are best-effort (copied from surviving vertex).
/// - Boundary edges are penalized (10× cost) but not locked.
pub fn simplify_mesh(mesh: &MeshBuffers, target_ratio: f32) -> GeometryResult<MeshBuffers> {
    if target_ratio <= 0.0 || target_ratio > 1.0 {
        return Err(GeometryError::new(
            "target_ratio must be in (0.0, 1.0]",
        ));
    }
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err(GeometryError::new("cannot simplify an empty mesh"));
    }
    if mesh.indices.len() % 3 != 0 {
        return Err(GeometryError::new(
            "indices length must be a multiple of 3",
        ));
    }

    let original_tri_count = mesh.indices.len() / 3;
    let target_tri_count = ((original_tri_count as f32 * target_ratio).ceil() as usize).max(1);

    // Ratio 1.0 or already at/below target → return clone
    if target_tri_count >= original_tri_count {
        return Ok(mesh.clone());
    }

    let vertex_count = mesh.positions.len();
    let has_uvs = mesh.uvs.len() == vertex_count;
    let has_tangents = mesh.tangents.len() == vertex_count;

    // --- Build adjacency ---
    // vertex_tris[v] = set of triangle indices that use vertex v
    let mut vertex_tris: Vec<HashSet<usize>> = vec![HashSet::new(); vertex_count];
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(original_tri_count);
    let mut tri_alive: Vec<bool> = Vec::with_capacity(original_tri_count);

    for tri_idx in 0..original_tri_count {
        let i0 = mesh.indices[tri_idx * 3];
        let i1 = mesh.indices[tri_idx * 3 + 1];
        let i2 = mesh.indices[tri_idx * 3 + 2];
        triangles.push([i0, i1, i2]);
        tri_alive.push(true);
        vertex_tris[i0 as usize].insert(tri_idx);
        vertex_tris[i1 as usize].insert(tri_idx);
        vertex_tris[i2 as usize].insert(tri_idx);
    }

    // --- Build per-vertex quadrics ---
    let mut quadrics: Vec<Quadric> = vec![Quadric::zero(); vertex_count];
    for tri_idx in 0..original_tri_count {
        let [i0, i1, i2] = triangles[tri_idx];
        let p0 = Vec3::from(mesh.positions[i0 as usize]);
        let p1 = Vec3::from(mesh.positions[i1 as usize]);
        let p2 = Vec3::from(mesh.positions[i2 as usize]);
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let normal = edge1.cross(edge2);
        let len = normal.length();
        if len < 1e-12 {
            continue; // degenerate triangle
        }
        let n = normal / len;
        let d = -n.dot(p0);
        let q = Quadric::from_plane(n.x as f64, n.y as f64, n.z as f64, d as f64);
        quadrics[i0 as usize] = quadrics[i0 as usize].add(&q);
        quadrics[i1 as usize] = quadrics[i1 as usize].add(&q);
        quadrics[i2 as usize] = quadrics[i2 as usize].add(&q);
    }

    // --- Identify boundary edges ---
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in &triangles {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let boundary_edges: HashSet<(u32, u32)> = edge_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&edge, _)| edge)
        .collect();
    let boundary_vertices: HashSet<u32> = boundary_edges
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();

    // --- Build positions (mutable copy) ---
    let mut positions: Vec<[f32; 3]> = mesh.positions.clone();
    let mut uvs: Vec<[f32; 2]> = if has_uvs {
        mesh.uvs.clone()
    } else {
        Vec::new()
    };
    let mut tangents_buf: Vec<[f32; 4]> = if has_tangents {
        mesh.tangents.clone()
    } else {
        Vec::new()
    };

    // --- Vertex remap (for collapse tracking) ---
    let mut vertex_remap: Vec<u32> = (0..vertex_count as u32).collect();
    let mut vertex_alive: Vec<bool> = vec![true; vertex_count];
    let mut generation: Vec<u32> = vec![0; vertex_count];

    fn find_root(remap: &[u32], mut v: u32) -> u32 {
        while remap[v as usize] != v {
            v = remap[v as usize];
        }
        v
    }

    // --- Build priority queue ---
    let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();

    // Collect unique edges
    let mut unique_edges: HashSet<(u32, u32)> = HashSet::new();
    for tri in &triangles {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            unique_edges.insert(key);
        }
    }

    let boundary_penalty = 10.0_f64;

    for &(v0, v1) in &unique_edges {
        let combined = quadrics[v0 as usize].add(&quadrics[v1 as usize]);
        let is_boundary = boundary_vertices.contains(&v0) || boundary_vertices.contains(&v1);
        let penalty = if is_boundary { boundary_penalty } else { 1.0 };

        // Use midpoint as target position
        let p0 = positions[v0 as usize];
        let p1 = positions[v1 as usize];
        let mid = [
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5,
        ];
        let error = combined.evaluate(mid[0] as f64, mid[1] as f64, mid[2] as f64) * penalty;

        heap.push(CollapseCandidate {
            error: error.max(0.0),
            v0,
            v1,
            target_pos: mid,
            generation: 0,
        });
    }

    // --- Collapse loop ---
    let mut alive_tri_count = original_tri_count;

    while alive_tri_count > target_tri_count {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let v0 = find_root(&vertex_remap, candidate.v0);
        let v1 = find_root(&vertex_remap, candidate.v1);

        // Skip stale or self-referencing candidates
        if v0 == v1 || !vertex_alive[v0 as usize] || !vertex_alive[v1 as usize] {
            continue;
        }

        // Skip stale generations
        if candidate.generation < generation[v0 as usize]
            || candidate.generation < generation[v1 as usize]
        {
            continue;
        }

        // Collapse v1 into v0
        // Move v0 to the target position
        positions[v0 as usize] = candidate.target_pos;
        if has_uvs {
            // Best-effort: keep v0's UVs (surviving vertex)
            // No interpolation in v1
        }

        // Update quadric
        quadrics[v0 as usize] = quadrics[v0 as usize].add(&quadrics[v1 as usize]);

        // Remap v1 -> v0
        vertex_remap[v1 as usize] = v0;
        vertex_alive[v1 as usize] = false;
        generation[v0 as usize] += 1;

        // Update triangles: replace v1 with v0, kill degenerate tris
        let tris_to_update: Vec<usize> = vertex_tris[v1 as usize].iter().copied().collect();
        for tri_idx in tris_to_update {
            if !tri_alive[tri_idx] {
                continue;
            }
            let tri = &mut triangles[tri_idx];
            for slot in tri.iter_mut() {
                if find_root(&vertex_remap, *slot) == v1 {
                    *slot = v0;
                } else {
                    *slot = find_root(&vertex_remap, *slot);
                }
            }

            // Check for degenerate (two or more identical vertices)
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                tri_alive[tri_idx] = false;
                alive_tri_count -= 1;
                // Remove from vertex_tris
                for &v in &[tri[0], tri[1], tri[2]] {
                    vertex_tris[v as usize].remove(&tri_idx);
                }
            } else {
                // Move triangle from v1 to v0
                vertex_tris[v0 as usize].insert(tri_idx);
            }
        }
        vertex_tris[v1 as usize].clear();

        // Re-enqueue edges from v0 to neighbors
        let neighbors: HashSet<u32> = vertex_tris[v0 as usize]
            .iter()
            .filter(|&&ti| tri_alive[ti])
            .flat_map(|&ti| triangles[ti].iter().copied())
            .filter(|&v| v != v0)
            .map(|v| find_root(&vertex_remap, v))
            .filter(|&v| v != v0 && vertex_alive[v as usize])
            .collect();

        let gen = generation[v0 as usize];
        for nb in neighbors {
            let combined = quadrics[v0 as usize].add(&quadrics[nb as usize]);
            let is_boundary =
                boundary_vertices.contains(&v0) || boundary_vertices.contains(&nb);
            let penalty = if is_boundary { boundary_penalty } else { 1.0 };

            let p0 = positions[v0 as usize];
            let p1 = positions[nb as usize];
            let mid = [
                (p0[0] + p1[0]) * 0.5,
                (p0[1] + p1[1]) * 0.5,
                (p0[2] + p1[2]) * 0.5,
            ];
            let error =
                combined.evaluate(mid[0] as f64, mid[1] as f64, mid[2] as f64) * penalty;

            heap.push(CollapseCandidate {
                error: error.max(0.0),
                v0,
                v1: nb,
                target_pos: mid,
                generation: gen,
            });
        }
    }

    // --- Compact output ---
    let mut new_positions: Vec<[f32; 3]> = Vec::new();
    let mut new_uvs: Vec<[f32; 2]> = Vec::new();
    let mut new_tangents: Vec<[f32; 4]> = Vec::new();
    let mut new_indices: Vec<u32> = Vec::new();
    let mut old_to_new: Vec<Option<u32>> = vec![None; vertex_count];

    for tri_idx in 0..triangles.len() {
        if !tri_alive[tri_idx] {
            continue;
        }
        let tri = triangles[tri_idx];
        for &old_v in &tri {
            let root = find_root(&vertex_remap, old_v);
            if old_to_new[root as usize].is_none() {
                let new_idx = new_positions.len() as u32;
                old_to_new[root as usize] = Some(new_idx);
                new_positions.push(positions[root as usize]);
                if has_uvs {
                    new_uvs.push(uvs[root as usize]);
                }
                if has_tangents {
                    new_tangents.push(tangents_buf[root as usize]);
                }
            }
            new_indices.push(old_to_new[root as usize].unwrap());
        }
    }

    // --- Recompute normals (area-weighted) ---
    let new_vertex_count = new_positions.len();
    let mut new_normals: Vec<Vec3> = vec![Vec3::ZERO; new_vertex_count];

    for tri_idx in (0..new_indices.len()).step_by(3) {
        let i0 = new_indices[tri_idx] as usize;
        let i1 = new_indices[tri_idx + 1] as usize;
        let i2 = new_indices[tri_idx + 2] as usize;
        let p0 = Vec3::from(new_positions[i0]);
        let p1 = Vec3::from(new_positions[i1]);
        let p2 = Vec3::from(new_positions[i2]);
        let face_normal = (p1 - p0).cross(p2 - p0); // area-weighted (not normalized)
        new_normals[i0] += face_normal;
        new_normals[i1] += face_normal;
        new_normals[i2] += face_normal;
    }

    let normals_out: Vec<[f32; 3]> = new_normals
        .iter()
        .map(|n| {
            let len = n.length();
            if len > 1e-12 {
                (*n / len).into()
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect();

    Ok(MeshBuffers {
        positions: new_positions,
        normals: normals_out,
        uvs: new_uvs,
        tangents: new_tangents,
        indices: new_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{generate_primitive, generate_unit_box, PrimitiveParams, PrimitiveType};

    #[test]
    fn simplify_box_reduces_triangles() {
        let mesh = generate_unit_box();
        assert_eq!(mesh.triangle_count(), 12);
        let simplified = simplify_mesh(&mesh, 0.5).unwrap();
        assert!(simplified.triangle_count() <= 6);
        assert!(simplified.triangle_count() >= 1);
        assert_eq!(simplified.normals.len(), simplified.positions.len());
    }

    #[test]
    fn simplify_sphere_reduces_geometry() {
        let mesh = generate_primitive(
            PrimitiveType::Sphere,
            PrimitiveParams {
                rings: 16,
                radial_segments: 32,
                ..Default::default()
            },
        );
        let original_tris = mesh.triangle_count();
        assert!(original_tris > 100);
        let simplified = simplify_mesh(&mesh, 0.25).unwrap();
        assert!(simplified.triangle_count() < original_tris);
        assert!(simplified.vertex_count() < mesh.vertex_count());
        assert_eq!(simplified.normals.len(), simplified.positions.len());
    }

    #[test]
    fn ratio_one_returns_clone() {
        let mesh = generate_unit_box();
        let result = simplify_mesh(&mesh, 1.0).unwrap();
        assert_eq!(result.triangle_count(), mesh.triangle_count());
        assert_eq!(result.vertex_count(), mesh.vertex_count());
    }

    #[test]
    fn very_low_ratio_on_small_mesh_returns_nonempty() {
        let mesh = generate_unit_box(); // 12 tris
        let result = simplify_mesh(&mesh, 0.01).unwrap();
        assert!(result.triangle_count() >= 1);
        assert!(!result.positions.is_empty());
    }

    #[test]
    fn rejects_empty_mesh() {
        let mesh = MeshBuffers::default();
        let err = simplify_mesh(&mesh, 0.5).unwrap_err();
        assert!(err.message().contains("empty"));
    }

    #[test]
    fn rejects_invalid_ratio() {
        let mesh = generate_unit_box();
        assert!(simplify_mesh(&mesh, 0.0).is_err());
        assert!(simplify_mesh(&mesh, -0.5).is_err());
        assert!(simplify_mesh(&mesh, 1.5).is_err());
    }

    #[test]
    fn preserves_uvs_when_present() {
        let mesh = generate_primitive(PrimitiveType::Sphere, PrimitiveParams::default());
        assert!(!mesh.uvs.is_empty());
        let simplified = simplify_mesh(&mesh, 0.5).unwrap();
        assert_eq!(simplified.uvs.len(), simplified.positions.len());
    }

    #[test]
    fn boundary_preservation_on_open_mesh() {
        // A plane is an open mesh (boundary edges on all four sides)
        let mesh = generate_primitive(
            PrimitiveType::Plane,
            PrimitiveParams {
                resolution: (8, 8),
                ..Default::default()
            },
        );
        let simplified = simplify_mesh(&mesh, 0.5).unwrap();
        // Should still produce a valid mesh with reduced tris
        assert!(simplified.triangle_count() < mesh.triangle_count());
        assert!(simplified.triangle_count() >= 1);
    }
}
```

- [ ] **Step 3: Verify Rust tests compile and pass**

Run: `cargo test --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz -p forge3d simplify`
Expected: All 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/geometry/simplify.rs src/geometry/mod.rs
git commit -m "feat(tv13.1): add QEM mesh simplification in src/geometry/simplify.rs"
```

---

## Task 3: PyO3 binding for simplify_mesh

**Files:**
- Modify: `src/geometry/py_bindings.rs`
- Modify: `src/py_module/functions/geometry.rs`
- Modify: `tests/test_api_contracts.py`

- [ ] **Step 1: Add `geometry_simplify_mesh_py` in `src/geometry/py_bindings.rs`**

Add at the end of the file (before the closing `}`), after the last `#[pyfunction]` block:

```rust
#[pyfunction]
pub fn geometry_simplify_mesh_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    target_ratio: f32,
) -> PyResult<PyObject> {
    let mesh_buffers = mesh_from_python(mesh)?;
    let simplified = map_geometry_err(super::simplify_mesh(&mesh_buffers, target_ratio))?;
    mesh_to_python(py, &simplified)
}
```

Also add `simplify_mesh` to the imports at the top of `py_bindings.rs`:

```rust
use super::{
    extrude_polygon_with_options, generate_primitive, simplify_mesh, transform, validate_mesh,
    weld_mesh, ExtrudeOptions, MeshBuffers, MeshValidationIssue, PrimitiveParams, PrimitiveType,
    WeldOptions,
};
```

- [ ] **Step 2: Register in `src/py_module/functions/geometry.rs`**

Add after the last `m.add_function(...)` line for geometry (before the instancing block):

```rust
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_simplify_mesh_py,
        m
    )?)?;
```

- [ ] **Step 3: Add re-export in `src/geometry/mod.rs`**

Add to the `#[cfg(feature = "extension-module")]` re-export block:

```rust
pub use py_bindings::geometry_simplify_mesh_py;
```

- [ ] **Step 4: Update contract test**

In `tests/test_api_contracts.py`, in the `TestGeometryFunctionContracts` class, add to the `GEOMETRY_FUNCTIONS` list:

```python
        "geometry_simplify_mesh_py",
```

- [ ] **Step 5: Build and run contract test**

```bash
maturin develop --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz --profile release-lto
python -m pytest tests/test_api_contracts.py::TestGeometryFunctionContracts -v
```

Expected: All geometry contract tests pass including the new `geometry_simplify_mesh_py`.

- [ ] **Step 6: Commit**

```bash
git add src/geometry/py_bindings.rs src/geometry/mod.rs src/py_module/functions/geometry.rs tests/test_api_contracts.py
git commit -m "feat(tv13.1): add PyO3 binding for geometry_simplify_mesh_py"
```

---

## Task 4: Python simplify_mesh and generate_lod_chain

**Files:**
- Modify: `python/forge3d/geometry.py`
- Modify: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Write failing tests in `tests/test_terrain_tv13_lod_pipeline.py`**

Add to the test file:

```python
from forge3d.geometry import MeshBuffers, primitive_mesh, simplify_mesh, generate_lod_chain


class TestSimplifyMesh:
    """TV13.1 — Python simplify_mesh wrapper."""

    def test_simplify_cone_reduces_triangles(self):
        cone = primitive_mesh("cone", radial_segments=32)
        simplified = simplify_mesh(cone, 0.5)
        assert simplified.triangle_count < cone.triangle_count
        assert simplified.vertex_count > 0

    def test_simplify_preserves_normals(self):
        sphere = primitive_mesh("sphere", rings=12, radial_segments=24)
        simplified = simplify_mesh(sphere, 0.5)
        assert simplified.normals.shape[0] == simplified.positions.shape[0]

    def test_simplify_ratio_one_unchanged(self):
        box_mesh = primitive_mesh("box")
        result = simplify_mesh(box_mesh, 1.0)
        assert result.triangle_count == box_mesh.triangle_count


class TestGenerateLodChain:
    """TV13.1 — LOD chain generation from a single mesh."""

    def test_three_level_chain_decreasing_triangles(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        chain = generate_lod_chain(sphere, [1.0, 0.25, 0.07])
        assert len(chain) == 3
        assert chain[0].triangle_count == sphere.triangle_count
        assert chain[1].triangle_count < chain[0].triangle_count
        assert chain[2].triangle_count < chain[1].triangle_count

    def test_min_triangles_floor_drops_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        # ratio 0.01 on 12 tris = 0.12 tris → below min_triangles=8
        chain = generate_lod_chain(box_mesh, [1.0, 0.01], min_triangles=8)
        assert len(chain) == 1  # only LOD 0 survives

    def test_deduplication_drops_identical_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        # 0.9 and 0.8 on a 12-tri mesh likely produce the same count
        chain = generate_lod_chain(box_mesh, [1.0, 0.9, 0.8])
        # Should deduplicate to fewer levels
        for i in range(1, len(chain)):
            assert chain[i].triangle_count < chain[i - 1].triangle_count

    def test_ratios_must_start_with_one(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="ratios.*1.0"):
            generate_lod_chain(sphere, [0.5, 0.25])

    def test_ratios_must_be_descending(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="descending"):
            generate_lod_chain(sphere, [1.0, 0.5, 0.7])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestSimplifyMesh -v`
Expected: FAIL — `ImportError: cannot import name 'simplify_mesh'`

- [ ] **Step 3: Implement in `python/forge3d/geometry.py`**

Add at the end of the file (before any `__all__` if present):

```python
def simplify_mesh(mesh: MeshBuffers, target_ratio: float) -> MeshBuffers:
    """Simplify a mesh to approximately target_ratio of its original triangle count.

    Uses QEM edge-collapse. Normals are recomputed; UVs are best-effort.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    result = _forge3d.geometry_simplify_mesh_py(payload, float(target_ratio))
    return _mesh_from_py(result)


def generate_lod_chain(
    mesh: MeshBuffers,
    ratios: list[float],
    *,
    min_triangles: int = 8,
) -> list[MeshBuffers]:
    """Generate a LOD chain from one high-detail mesh.

    Each level is simplified from the *original* mesh (not cascaded).
    ``ratios[0]`` must be 1.0. Ratios must be in descending order in (0.0, 1.0].

    If a ratio produces fewer than ``min_triangles``, that level and all coarser
    levels are dropped. Duplicate outputs (same triangle count as a prior level)
    are also dropped.
    """
    if not ratios:
        raise ValueError("ratios must be a non-empty list")
    if abs(ratios[0] - 1.0) > 1e-6:
        raise ValueError("ratios[0] must be 1.0 — LOD 0 is always the original mesh")
    for i in range(1, len(ratios)):
        if ratios[i] >= ratios[i - 1]:
            raise ValueError(
                f"ratios must be in strictly descending order, "
                f"but ratios[{i}]={ratios[i]} >= ratios[{i-1}]={ratios[i-1]}"
            )
        if ratios[i] <= 0.0 or ratios[i] > 1.0:
            raise ValueError(f"ratios[{i}]={ratios[i]} must be in (0.0, 1.0]")

    chain: list[MeshBuffers] = [mesh]  # LOD 0 is always the original
    prev_tri_count = mesh.triangle_count

    for ratio in ratios[1:]:
        simplified = simplify_mesh(mesh, ratio)  # always from original
        tri_count = simplified.triangle_count
        if tri_count < min_triangles:
            break  # stop — this and all coarser levels are too small
        if tri_count >= prev_tri_count:
            continue  # deduplicate — same count as previous level
        chain.append(simplified)
        prev_tri_count = tri_count

    return chain
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestSimplifyMesh tests/test_terrain_tv13_lod_pipeline.py::TestGenerateLodChain -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/geometry.py tests/test_terrain_tv13_lod_pipeline.py
git commit -m "feat(tv13.1): add Python simplify_mesh() and generate_lod_chain()"
```

---

## Task 5: auto_lod_levels for scatter

**Files:**
- Modify: `python/forge3d/terrain_scatter.py`
- Modify: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_terrain_tv13_lod_pipeline.py`:

```python
from forge3d.terrain_scatter import (
    TerrainScatterBatch,
    TerrainScatterLevel,
    auto_lod_levels,
)


class TestAutoLodLevels:
    """TV13.2 — auto_lod_levels generates scatter LOD levels from one mesh."""

    def test_default_three_levels(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(sphere, lod_count=3, draw_distance=300.0)
        assert len(levels) == 3
        assert all(isinstance(l, TerrainScatterLevel) for l in levels)
        # Last level should be open-ended
        assert levels[-1].max_distance is None
        # First levels should have distances
        assert levels[0].max_distance is not None
        assert levels[1].max_distance is not None
        assert levels[1].max_distance > levels[0].max_distance

    def test_explicit_distances(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            lod_distances=[50.0, 150.0, None],
        )
        assert levels[0].max_distance == 50.0
        assert levels[1].max_distance == 150.0
        assert levels[2].max_distance is None

    def test_explicit_ratios(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            ratios=[1.0, 0.3, 0.05],
            draw_distance=200.0,
        )
        # Triangle counts should decrease
        assert levels[0].mesh.triangle_count >= levels[1].mesh.triangle_count
        assert levels[1].mesh.triangle_count >= levels[2].mesh.triangle_count

    def test_feeds_into_scatter_batch(self):
        cone = primitive_mesh("cone", radial_segments=32)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=100.0)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        # Should not raise
        batch = TerrainScatterBatch(
            levels=levels,
            transforms=transforms,
            name="auto_lod_test",
        )
        assert batch.instance_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestAutoLodLevels -v`
Expected: FAIL — `ImportError: cannot import name 'auto_lod_levels'`

- [ ] **Step 3: Implement `auto_lod_levels` in `python/forge3d/terrain_scatter.py`**

Add before the `serialize_batches_for_native` function (around line 571):

```python
def auto_lod_levels(
    mesh: MeshBuffers,
    *,
    lod_count: int = 3,
    lod_distances: list[float | None] | None = None,
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
        Explicit per-level max_distance values (list[float | None]).
        Length must equal lod_count. The final level's distance may be
        None (open-ended).
    ratios
        Triangle-count ratios for each level. Length must equal lod_count.
        Default: geometric series from 1.0 down.
    draw_distance
        Used to derive lod_distances when lod_distances is not provided.
    min_triangles
        Minimum triangle count floor passed to generate_lod_chain.
    """
    from forge3d.geometry import generate_lod_chain

    if lod_count < 1:
        raise ValueError("lod_count must be >= 1")

    # --- Resolve ratios ---
    if ratios is None:
        ratios = [1.0]
        for i in range(1, lod_count):
            ratios.append(max(0.25 ** i, 0.01))
    if len(ratios) != lod_count:
        raise ValueError(f"ratios length ({len(ratios)}) must equal lod_count ({lod_count})")

    # --- Generate LOD chain ---
    chain = generate_lod_chain(mesh, ratios, min_triangles=min_triangles)
    actual_count = len(chain)

    # --- Resolve distances ---
    if lod_distances is not None:
        if len(lod_distances) != lod_count:
            raise ValueError(
                f"lod_distances length ({len(lod_distances)}) must equal lod_count ({lod_count})"
            )
        distances = list(lod_distances[:actual_count])
    elif draw_distance is not None:
        # Geometric spacing: ratio = 0.33
        geo_ratio = 0.33
        distances = []
        for i in range(actual_count):
            if i == actual_count - 1:
                distances.append(None)
            else:
                d = draw_distance * geo_ratio ** (actual_count - 1 - i)
                distances.append(float(d))
    else:
        # Conservative defaults
        default_dists = [30.0, 100.0, 300.0, 600.0, 1000.0]
        distances = []
        for i in range(actual_count):
            if i == actual_count - 1:
                distances.append(None)
            elif i < len(default_dists):
                distances.append(default_dists[i])
            else:
                distances.append(None)

    # Ensure final level is open-ended
    if distances and distances[-1] is not None:
        distances[-1] = None

    # --- Build TerrainScatterLevel list ---
    levels = []
    for i, lod_mesh in enumerate(chain):
        max_dist = distances[i] if i < len(distances) else None
        levels.append(TerrainScatterLevel(mesh=lod_mesh, max_distance=max_dist))

    return levels
```

Also add to `__all__` at the bottom:

```python
    "auto_lod_levels",
```

And add the import at the top of the file (with existing imports):

```python
from forge3d.geometry import MeshBuffers
```

Note: `MeshBuffers` is already imported transitively through `_mesh_to_py` usage. Check if it needs an explicit import. If the file already uses `MeshBuffers` via a local `_mesh_from_py` function, no additional import is needed — `auto_lod_levels` receives a `MeshBuffers` as parameter, which works without importing since Python uses duck typing at runtime.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestAutoLodLevels -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/terrain_scatter.py tests/test_terrain_tv13_lod_pipeline.py
git commit -m "feat(tv13.2): add auto_lod_levels() for automatic scatter LOD generation"
```

---

## Task 6: HLODPolicy and Python serialization

**Files:**
- Modify: `python/forge3d/terrain_scatter.py`
- Modify: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_terrain_tv13_lod_pipeline.py`:

```python
from forge3d.terrain_scatter import HLODPolicy


class TestHLODPolicy:
    """TV13.3 — HLODPolicy dataclass and serialization."""

    def test_hlod_policy_creation(self):
        policy = HLODPolicy(hlod_distance=200.0, cluster_radius=50.0)
        assert policy.hlod_distance == 200.0
        assert policy.cluster_radius == 50.0
        assert policy.simplify_ratio == 0.1  # default

    def test_hlod_in_native_dict(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        policy = HLODPolicy(hlod_distance=100.0, cluster_radius=30.0, simplify_ratio=0.2)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
            hlod=policy,
            max_draw_distance=500.0,
        )
        d = batch.to_native_dict()
        assert "hlod" in d
        assert d["hlod"]["hlod_distance"] == 100.0
        assert d["hlod"]["cluster_radius"] == 30.0
        assert d["hlod"]["simplify_ratio"] == 0.2

    def test_hlod_none_omitted_from_dict(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
        )
        d = batch.to_native_dict()
        assert d.get("hlod") is None

    def test_hlod_in_viewer_payload(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        policy = HLODPolicy(hlod_distance=100.0, cluster_radius=30.0)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
            hlod=policy,
            max_draw_distance=500.0,
        )
        payload = batch.to_viewer_payload()
        assert "hlod" in payload
        assert payload["hlod"]["hlod_distance"] == 100.0

    def test_hlod_validation_rejects_bad_params(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        # hlod_distance >= max_draw_distance
        with pytest.raises(ValueError, match="hlod_distance"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=500.0, cluster_radius=30.0),
                max_draw_distance=200.0,
            )

    def test_hlod_rejects_negative_cluster_radius(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        with pytest.raises(ValueError, match="cluster_radius"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=50.0, cluster_radius=-10.0),
                max_draw_distance=200.0,
            )

    def test_hlod_rejects_invalid_simplify_ratio(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        with pytest.raises(ValueError, match="simplify_ratio"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=50.0, cluster_radius=30.0, simplify_ratio=0.0),
                max_draw_distance=200.0,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestHLODPolicy -v`
Expected: FAIL — `ImportError: cannot import name 'HLODPolicy'`

- [ ] **Step 3: Implement HLODPolicy and update TerrainScatterBatch**

In `python/forge3d/terrain_scatter.py`:

Add the `HLODPolicy` dataclass before `TerrainScatterLevel`:

```python
@dataclass(frozen=True)
class HLODPolicy:
    """Configuration for HLOD cluster generation.

    Attributes
    ----------
    hlod_distance : float
        Distance beyond which instances are replaced by HLOD clusters.
    cluster_radius : float
        Spatial grid cell size for grouping instances.
    simplify_ratio : float
        Target triangle ratio for merged cluster meshes (0.0, 1.0].
    """
    hlod_distance: float
    cluster_radius: float
    simplify_ratio: float = 0.1
```

Update `TerrainScatterBatch` — add `hlod` field:

```python
@dataclass
class TerrainScatterBatch:
    levels: Sequence[TerrainScatterLevel]
    transforms: np.ndarray
    name: str | None = None
    color: Sequence[float] = (0.85, 0.85, 0.85, 1.0)
    max_draw_distance: float | None = None
    hlod: HLODPolicy | None = None
```

Update `__post_init__` — add HLOD validation after the existing checks:

```python
        if self.hlod is not None:
            if not isinstance(self.hlod, HLODPolicy):
                raise ValueError("hlod must be an HLODPolicy instance or None")
            if self.hlod.hlod_distance <= 0 or not np.isfinite(self.hlod.hlod_distance):
                raise ValueError("hlod_distance must be a positive finite float")
            if self.hlod.cluster_radius <= 0 or not np.isfinite(self.hlod.cluster_radius):
                raise ValueError("cluster_radius must be a positive finite float")
            if self.hlod.simplify_ratio <= 0 or self.hlod.simplify_ratio > 1.0:
                raise ValueError("simplify_ratio must be in (0.0, 1.0]")
            if self.max_draw_distance is not None and self.hlod.hlod_distance >= self.max_draw_distance:
                raise ValueError(
                    f"hlod_distance ({self.hlod.hlod_distance}) must be less than "
                    f"max_draw_distance ({self.max_draw_distance})"
                )
```

Update `to_native_dict` — add HLOD:

```python
    def to_native_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "color": tuple(self.color),
            "max_draw_distance": self.max_draw_distance,
            "transforms": self.transforms,
            "levels": [
                {
                    "mesh": _mesh_to_py(level.mesh),
                    "max_distance": level.max_distance,
                }
                for level in self.levels
            ],
        }
        if self.hlod is not None:
            d["hlod"] = {
                "hlod_distance": self.hlod.hlod_distance,
                "cluster_radius": self.hlod.cluster_radius,
                "simplify_ratio": self.hlod.simplify_ratio,
            }
        else:
            d["hlod"] = None
        return d
```

Update `to_viewer_payload` — add HLOD (at the end of the returned dict):

```python
        payload = {
            "name": self.name,
            "color": list(self.color),
            "max_draw_distance": self.max_draw_distance,
            "transforms": self.transforms.tolist(),
            "levels": levels,
        }
        if self.hlod is not None:
            payload["hlod"] = {
                "hlod_distance": self.hlod.hlod_distance,
                "cluster_radius": self.hlod.cluster_radius,
                "simplify_ratio": self.hlod.simplify_ratio,
            }
        return payload
```

Add to `__all__`:

```python
    "HLODPolicy",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestHLODPolicy -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/terrain_scatter.py tests/test_terrain_tv13_lod_pipeline.py
git commit -m "feat(tv13.3): add HLODPolicy dataclass and batch serialization"
```

---

## Task 7: HLOD Rust implementation — types, clustering, and build

**Files:**
- Modify: `src/terrain/scatter.rs`

This is the largest task. It adds HLOD types, spatial clustering, mesh merging, and the `HlodCache` build path to `TerrainScatterBatch`.

- [ ] **Step 1: Add HLOD types and HlodConfig**

At the top of `src/terrain/scatter.rs` (after existing imports), add:

```rust
use crate::geometry::simplify_mesh;
```

After `TerrainScatterMemoryReport`, add the new types:

```rust
#[derive(Debug, Clone)]
pub struct HlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}

struct GpuHlodCluster {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    center: Vec3,
    radius: f32,
    vertex_buffer_bytes: u64,
    index_buffer_bytes: u64,
    _vertex_handle: ResourceHandle,
    _index_handle: ResourceHandle,
}

struct HlodCache {
    clusters: Vec<GpuHlodCluster>,
    instance_to_cluster: Vec<Option<usize>>,
    hlod_distance: f32,
    total_buffer_bytes: u64,
}
```

- [ ] **Step 2: Add HLOD stats fields**

Update `TerrainScatterBatchStats`:

```rust
#[derive(Debug, Clone, Default)]
pub struct TerrainScatterBatchStats {
    pub total_instances: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub lod_instance_counts: Vec<u32>,
    pub hlod_cluster_draws: u32,
    pub hlod_covered_instances: u32,
    pub effective_draws: u32,
}
```

Update `TerrainScatterFrameStats`:

```rust
#[derive(Debug, Clone, Default)]
pub struct TerrainScatterFrameStats {
    pub batch_count: u32,
    pub rendered_batches: u32,
    pub total_instances: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub lod_instance_counts: Vec<u32>,
    pub hlod_cluster_draws: u32,
    pub hlod_covered_instances: u32,
    pub effective_draws: u32,
}
```

Add HLOD fields to `TerrainScatterMemoryReport`:

```rust
pub struct TerrainScatterMemoryReport {
    pub batch_count: u32,
    pub level_count: u32,
    pub total_instances: u32,
    pub vertex_buffer_bytes: u64,
    pub index_buffer_bytes: u64,
    pub instance_buffer_bytes: u64,
    pub hlod_cluster_count: u32,
    pub hlod_buffer_bytes: u64,
}

impl TerrainScatterMemoryReport {
    pub fn total_buffer_bytes(&self) -> u64 {
        self.vertex_buffer_bytes + self.index_buffer_bytes
            + self.instance_buffer_bytes + self.hlod_buffer_bytes
    }
}
```

Update `accumulate_frame_stats` to include HLOD fields:

```rust
    stats.hlod_cluster_draws += batch_stats.hlod_cluster_draws;
    stats.hlod_covered_instances += batch_stats.hlod_covered_instances;
    stats.effective_draws += batch_stats.effective_draws;
```

- [ ] **Step 3: Add HLOD build functions**

Add these functions before the `impl TerrainScatterBatch` block:

```rust
fn build_hlod_cache(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    levels: &[GpuScatterLevel],
    source_mesh: &MeshBuffers,
    transforms_rowmajor: &[[f32; 16]],
    positions: &[[f32; 3]],
    config: &HlodConfig,
) -> Result<HlodCache> {
    // Spatial grid clustering
    let inv_cell = 1.0 / config.cluster_radius;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (idx, pos) in positions.iter().enumerate() {
        let cx = (pos[0] * inv_cell).floor() as i32;
        let cy = (pos[1] * inv_cell).floor() as i32;
        let cz = (pos[2] * inv_cell).floor() as i32;
        grid.entry((cx, cy, cz)).or_default().push(idx);
    }

    let mut instance_to_cluster: Vec<Option<usize>> = vec![None; positions.len()];
    let mut clusters: Vec<GpuHlodCluster> = Vec::new();
    let mut total_buffer_bytes: u64 = 0;

    for (_cell, members) in &grid {
        if members.len() < 2 {
            continue; // singletons stay as individual instances
        }

        // Merge: bake instance transforms into source mesh vertices
        let mut merged_positions: Vec<[f32; 3]> = Vec::new();
        let mut merged_normals: Vec<[f32; 3]> = Vec::new();
        let mut merged_indices: Vec<u32> = Vec::new();

        for &inst_idx in members {
            let m = row_major_to_mat4(transforms_rowmajor[inst_idx]);
            let normal_mat = m.inverse().transpose();
            let base_vertex = merged_positions.len() as u32;

            for pos in &source_mesh.positions {
                let p = m.transform_point3(Vec3::from(*pos));
                merged_positions.push(p.into());
            }
            for norm in &source_mesh.normals {
                let n = normal_mat.transform_vector3(Vec3::from(*norm)).normalize_or_zero();
                merged_normals.push(n.into());
            }
            for idx in &source_mesh.indices {
                merged_indices.push(base_vertex + idx);
            }
        }

        let merged_mesh = MeshBuffers {
            positions: merged_positions,
            normals: merged_normals,
            uvs: Vec::new(),
            tangents: Vec::new(),
            indices: merged_indices,
        };

        // Simplify the merged mesh
        let simplified = match simplify_mesh(&merged_mesh, config.simplify_ratio) {
            Ok(m) => m,
            Err(_) => merged_mesh, // fallback: use unsimplified
        };

        // Compute cluster bounds
        let center = members.iter().fold(Vec3::ZERO, |acc, &i| {
            acc + Vec3::new(positions[i][0], positions[i][1], positions[i][2])
        }) / members.len() as f32;
        let radius = members.iter().fold(0.0_f32, |max_r, &i| {
            let p = Vec3::new(positions[i][0], positions[i][1], positions[i][2]);
            max_r.max(center.distance(p))
        });

        // Upload to GPU
        let gpu_cluster = build_gpu_hlod_cluster(device, queue, simplified, center, radius)?;
        let cluster_idx = clusters.len();
        total_buffer_bytes += gpu_cluster.vertex_buffer_bytes + gpu_cluster.index_buffer_bytes;

        for &inst_idx in members {
            instance_to_cluster[inst_idx] = Some(cluster_idx);
        }
        clusters.push(gpu_cluster);
    }

    Ok(HlodCache {
        clusters,
        instance_to_cluster,
        hlod_distance: config.hlod_distance,
        total_buffer_bytes,
    })
}

fn build_gpu_hlod_cluster(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: MeshBuffers,
    center: Vec3,
    radius: f32,
) -> Result<GpuHlodCluster> {
    use crate::render::mesh_instanced::VertexPN;

    let vertices: Vec<VertexPN> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, pos)| VertexPN {
            position: *pos,
            normal: mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
        })
        .collect();

    let vertex_buffer_bytes = (vertices.len() * std::mem::size_of::<VertexPN>()) as u64;
    let index_buffer_bytes = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;

    let vertex_handle = register_buffer(
        vertex_buffer_bytes,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );
    let index_handle = register_buffer(
        index_buffer_bytes,
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    );

    let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.hlod.vertex_buffer"),
        size: vertex_buffer_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.hlod.index_buffer"),
        size: index_buffer_bytes,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&vertices));
    queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&mesh.indices));

    Ok(GpuHlodCluster {
        vbuf,
        ibuf,
        index_count: mesh.indices.len() as u32,
        center,
        radius,
        vertex_buffer_bytes,
        index_buffer_bytes,
        _vertex_handle: vertex_handle,
        _index_handle: index_handle,
    })
}
```

Note: add `use std::collections::HashMap;` at the top of the file.

- [ ] **Step 4: Update `TerrainScatterBatch` to accept and build HLOD**

Add `hlod_cache: Option<HlodCache>` field to `TerrainScatterBatch`.

Update `new()` signature to accept `hlod_config: Option<HlodConfig>`.

After building `gpu_levels` in `new()`, add:

```rust
        let hlod_cache = if let Some(ref config) = hlod_config {
            // Use coarsest LOD level mesh as source
            let source_mesh = &levels_input.last().unwrap().mesh;
            // (levels_input is consumed by this point — need to keep a copy)
            Some(build_hlod_cache(
                device, queue, &gpu_levels, source_mesh,
                transforms_rowmajor, &positions, config,
            )?)
        } else {
            None
        };
```

Restructure `new()` to extract the source mesh **before** the `into_iter()` that consumes `levels`:

```rust
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        levels: Vec<TerrainScatterLevelSpec>,
        transforms_rowmajor: &[[f32; 16]],
        color: [f32; 4],
        max_draw_distance: Option<f32>,
        name: Option<String>,
        hlod_config: Option<HlodConfig>,
    ) -> Result<Self> {
        // ... existing validation unchanged ...

        // Extract HLOD source mesh (coarsest level) BEFORE consuming levels
        let hlod_source_mesh = if hlod_config.is_some() {
            Some(levels.last().unwrap().mesh.clone())
        } else {
            None
        };

        let gpu_levels = levels
            .into_iter()
            .map(|spec| build_gpu_level(device, queue, spec))
            .collect::<Result<Vec<_>>>()?;
        let level_count = gpu_levels.len();

        let positions = extract_positions(transforms_rowmajor);

        let hlod_cache = match (hlod_config, hlod_source_mesh) {
            (Some(ref config), Some(ref source)) => Some(build_hlod_cache(
                device, queue, &gpu_levels, source,
                transforms_rowmajor, &positions, config,
            )?),
            _ => None,
        };

        Ok(Self {
            name,
            color,
            max_draw_distance,
            levels: gpu_levels,
            transforms_rowmajor: transforms_rowmajor.to_vec(),
            positions,
            instance_buffers: std::iter::repeat_with(|| None).take(level_count).collect(),
            last_stats: TerrainScatterBatchStats::default(),
            hlod_cache,
        })
    }
```

- [ ] **Step 5: Rewrite `prepare_draws()` with complete HLOD cluster-level activation**

Replace the entire `prepare_draws` method body. The key change: before the per-instance loop, compute `cluster_active`, then check `instance_to_cluster` + `cluster_active` during the loop to skip covered instances.

```rust
    pub fn prepare_draws(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eye_contract: Vec3,
        render_from_contract: Mat4,
        instance_scale: f32,
    ) -> Result<(TerrainScatterBatchStats, Vec<PreparedScatterDraw>)> {
        let mut per_level = vec![Vec::<[f32; 16]>::new(); self.levels.len()];
        let mut stats = TerrainScatterBatchStats {
            total_instances: self.transforms_rowmajor.len() as u32,
            lod_instance_counts: vec![0; self.levels.len()],
            ..Default::default()
        };

        // --- HLOD: determine which clusters are active ---
        let cluster_active: Vec<bool> = if let Some(ref hlod) = self.hlod_cache {
            hlod.clusters.iter().map(|cluster| {
                let dist_to_center = eye_contract.distance(cluster.center);
                let effective_dist = dist_to_center - cluster.radius;
                effective_dist > hlod.hlod_distance && effective_dist < self.max_draw_distance
            }).collect()
        } else {
            Vec::new()
        };

        // --- Per-instance loop with HLOD skip ---
        for (i, (transform, position)) in self.transforms_rowmajor.iter().zip(self.positions.iter()).enumerate() {
            let dist = eye_contract.distance(Vec3::new(position[0], position[1], position[2]));
            if dist > self.max_draw_distance {
                continue;
            }

            // Check if this instance is covered by an active HLOD cluster
            if let Some(ref hlod) = self.hlod_cache {
                if let Some(cluster_idx) = hlod.instance_to_cluster[i] {
                    if cluster_active[cluster_idx] {
                        stats.hlod_covered_instances += 1;
                        continue; // skip — cluster draws this instance
                    }
                }
            }

            let level_index = select_level_index(&self.levels, dist);
            per_level[level_index].push(*transform);
            stats.visible_instances += 1;
            stats.lod_instance_counts[level_index] += 1;
        }

        stats.culled_instances = stats
            .total_instances
            .saturating_sub(stats.visible_instances + stats.hlod_covered_instances);

        // --- Upload instance buffers for per-instance draws ---
        let mut draws = Vec::new();
        for (level_index, transforms) in per_level.iter().enumerate() {
            if transforms.is_empty() {
                continue;
            }
            ensure_instance_capacity(
                device,
                &mut self.instance_buffers[level_index],
                transforms.len(),
            )?;
            let packed = pack_instance_transforms(transforms, render_from_contract, instance_scale);
            if let Some(instance_buffer) = self.instance_buffers[level_index].as_ref() {
                queue.write_buffer(&instance_buffer.buffer, 0, bytemuck::cast_slice(&packed));
            }
            draws.push(PreparedScatterDraw {
                level_index,
                instance_count: transforms.len() as u32,
            });
        }

        // --- Count active HLOD cluster draws ---
        stats.hlod_cluster_draws = cluster_active.iter().filter(|&&a| a).count() as u32;

        // effective_draws = individual LOD draws + HLOD cluster draws
        stats.effective_draws = draws.len() as u32 + stats.hlod_cluster_draws;

        self.last_stats = stats.clone();
        Ok((stats, draws))
    }
```

- [ ] **Step 6: Add HLOD cluster accessors for the render pass**

Add these public methods to `TerrainScatterBatch` so the render pass can draw HLOD clusters:

```rust
    /// Returns the list of active HLOD cluster indices for the current frame.
    /// Must be called after `prepare_draws()`.
    pub fn hlod_active_clusters(&self, eye_contract: Vec3) -> Vec<usize> {
        let Some(ref hlod) = self.hlod_cache else {
            return Vec::new();
        };
        hlod.clusters
            .iter()
            .enumerate()
            .filter_map(|(i, cluster)| {
                let dist = eye_contract.distance(cluster.center);
                let effective = dist - cluster.radius;
                if effective > hlod.hlod_distance && effective < self.max_draw_distance {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn hlod_cluster_vbuf(&self, cluster_index: usize) -> Option<&wgpu::Buffer> {
        self.hlod_cache.as_ref().map(|h| &h.clusters[cluster_index].vbuf)
    }

    pub fn hlod_cluster_ibuf(&self, cluster_index: usize) -> Option<&wgpu::Buffer> {
        self.hlod_cache.as_ref().map(|h| &h.clusters[cluster_index].ibuf)
    }

    pub fn hlod_cluster_index_count(&self, cluster_index: usize) -> u32 {
        self.hlod_cache.as_ref().map(|h| h.clusters[cluster_index].index_count).unwrap_or(0)
    }
```

- [ ] **Step 7: Update `memory_report()` to include HLOD**

```rust
        if let Some(ref hlod) = self.hlod_cache {
            report.hlod_cluster_count = hlod.clusters.len() as u32;
            report.hlod_buffer_bytes = hlod.total_buffer_bytes;
        }
```

- [ ] **Step 8: Update `summarize_memory()` to accumulate HLOD fields**

In the `summarize_memory` free function, add after existing accumulation:

```rust
        report.hlod_cluster_count += batch_report.hlod_cluster_count;
        report.hlod_buffer_bytes += batch_report.hlod_buffer_bytes;
```

- [ ] **Step 9: Update `update_transforms()` signature**

```rust
    pub fn update_transforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        transforms_rowmajor: &[[f32; 16]],
    ) -> Result<()> {
        if transforms_rowmajor.is_empty() {
            return Err(anyhow!("terrain scatter requires at least one transform"));
        }
        validate_transforms(transforms_rowmajor)?;

        self.transforms_rowmajor.clear();
        self.transforms_rowmajor.extend_from_slice(transforms_rowmajor);
        self.positions = extract_positions(transforms_rowmajor);
        self.last_stats = TerrainScatterBatchStats::default();

        // Rebuild HLOD if present
        if let Some(ref old_hlod) = self.hlod_cache {
            let config = HlodConfig {
                hlod_distance: old_hlod.hlod_distance,
                cluster_radius: old_hlod.clusters.first()
                    .map(|_| old_hlod.hlod_distance) // stored on cache
                    .unwrap_or(1.0),
                simplify_ratio: 0.1, // would need to store this
            };
            // Note: to properly rebuild, HlodCache should store the original config.
            // For v1, store config alongside cache. See hlod_config field below.
        }

        Ok(())
    }
```

**Implementation note:** Store the `HlodConfig` as a field on `TerrainScatterBatch` alongside `hlod_cache` so `update_transforms` can rebuild. Add `hlod_config: Option<HlodConfig>` to the struct fields. In `update_transforms`, when `hlod_config.is_some()`, extract the coarsest level mesh from `self.levels` (it's already a `GpuScatterLevel` — you'll need to keep the source `MeshBuffers` in `HlodCache` or as a batch field for rebuild). For v1, storing the source mesh in `HlodCache` is simplest.

- [ ] **Step 10: Add Rust HLOD unit tests**

Add to `src/terrain/scatter.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn hlod_stats_fields_default_to_zero() {
        let stats = TerrainScatterBatchStats::default();
        assert_eq!(stats.hlod_cluster_draws, 0);
        assert_eq!(stats.hlod_covered_instances, 0);
        assert_eq!(stats.effective_draws, 0);
    }

    #[test]
    fn memory_report_includes_hlod_in_total() {
        let report = TerrainScatterMemoryReport {
            vertex_buffer_bytes: 10,
            index_buffer_bytes: 20,
            instance_buffer_bytes: 30,
            hlod_buffer_bytes: 40,
            ..Default::default()
        };
        assert_eq!(report.total_buffer_bytes(), 100);
    }

    #[test]
    fn accumulate_frame_stats_includes_hlod() {
        let mut frame = TerrainScatterFrameStats::default();
        accumulate_frame_stats(
            &mut frame,
            &TerrainScatterBatchStats {
                total_instances: 10,
                visible_instances: 5,
                culled_instances: 2,
                lod_instance_counts: vec![3, 2],
                hlod_cluster_draws: 2,
                hlod_covered_instances: 3,
                effective_draws: 5,
            },
        );
        assert_eq!(frame.hlod_cluster_draws, 2);
        assert_eq!(frame.hlod_covered_instances, 3);
        assert_eq!(frame.effective_draws, 5);
    }
```

- [ ] **Step 11: Update existing call sites**

Update the `new()` call in `src/terrain/renderer/scatter.rs` (`set_scatter_batches_native`) and `src/viewer/terrain/scene/scatter.rs` (`set_scatter_batches_from_configs`) to pass `None` for `hlod_config` initially — the HLOD plumbing will be added in Task 8.

- [ ] **Step 12: Verify Rust tests compile and pass**

Run: `cargo test --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz`
Expected: All existing tests pass. New HLOD Rust unit tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/terrain/scatter.rs src/terrain/renderer/scatter.rs src/viewer/terrain/scene/scatter.rs
git commit -m "feat(tv13.3): add HLOD types, clustering, build, and prepare_draws integration"
```

---

## Task 8: HLOD plumbing — renderer, viewer, IPC

**Files:**
- Modify: `src/terrain/renderer/scatter.rs`
- Modify: `src/terrain/renderer/py_api.rs`
- Modify: `src/viewer/viewer_enums/config.rs`
- Modify: `src/viewer/ipc/protocol/payloads.rs`
- Modify: `src/viewer/ipc/protocol/translate/terrain.rs`
- Modify: `src/viewer/terrain/scene/scatter.rs`

- [ ] **Step 1: Add `hlod_config` to `TerrainScatterUploadBatch`**

In `src/terrain/renderer/scatter.rs`:

```rust
pub(super) struct TerrainScatterUploadBatch {
    pub(super) name: Option<String>,
    pub(super) color: [f32; 4],
    pub(super) max_draw_distance: Option<f32>,
    pub(super) transforms_rowmajor: Vec<[f32; 16]>,
    pub(super) levels: Vec<TerrainScatterLevelSpec>,
    pub(super) hlod_config: Option<HlodConfig>,  // NEW
}
```

Update `set_scatter_batches_native` to pass `batch.hlod_config`:

```rust
        gpu_batches.push(TerrainScatterBatch::new(
            self.device.as_ref(),
            self.queue.as_ref(),
            batch.levels,
            &batch.transforms_rowmajor,
            batch.color,
            batch.max_draw_distance,
            batch.name,
            batch.hlod_config,  // NEW
        )?);
```

- [ ] **Step 2: Add HLOD cluster draw in render pass**

In both `src/terrain/renderer/scatter.rs` (`render_scatter_pass`) and `src/viewer/terrain/scene/scatter.rs` (`render_scatter_batches`), after the existing per-batch instanced draw loop, add HLOD cluster draws.

Each HLOD cluster is drawn as a 1-instance draw call using an identity transform in a temporary instance buffer. Add a shared helper and use it in both render paths:

```rust
/// Draw active HLOD clusters for a batch. Called inside the render pass
/// after the per-instance LOD draws.
fn draw_hlod_clusters(
    batch: &TerrainScatterBatch,
    eye_contract: Vec3,
    render_from_contract: Mat4,
    instance_scale: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::RenderPass<'_>,
    renderer: &mut crate::render::mesh_instanced::MeshInstancedRenderer,
    view: Mat4,
    proj: Mat4,
    light_dir: [f32; 3],
    light_intensity: f32,
) {
    let active_clusters = batch.hlod_active_clusters(eye_contract);
    if active_clusters.is_empty() {
        return;
    }

    // HLOD cluster vertices are already in contract space (baked transforms).
    // We need a single identity instance that maps through render_from_contract.
    let identity_instance = Mat4::from_scale(Vec3::splat(instance_scale))
        * Mat4::from_translation(Vec3::ZERO); // no per-instance offset
    // Actually: HLOD verts are in contract space already. We need render_from_contract
    // applied. Use a single-instance buffer with render_from_contract * uniform_scale.
    let hlod_instance_mat = render_from_contract;
    let instance_data: [f32; 16] = hlod_instance_mat.to_cols_array();

    // Create or reuse a small 1-instance buffer
    let inst_bytes = 64u64; // 16 floats * 4 bytes
    let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.scatter.hlod.instance_temp"),
        size: inst_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&instance_data));

    for cluster_idx in active_clusters {
        let Some(vbuf) = batch.hlod_cluster_vbuf(cluster_idx) else { continue };
        let Some(ibuf) = batch.hlod_cluster_ibuf(cluster_idx) else { continue };
        let index_count = batch.hlod_cluster_index_count(cluster_idx);
        if index_count == 0 {
            continue;
        }

        renderer.draw_batch_params(
            device,
            pass,
            queue,
            view,
            proj,
            batch.color,
            light_dir,
            light_intensity,
            vbuf,
            ibuf,
            &inst_buf,
            index_count,
            1, // single instance
        );
    }
}
```

Then in `render_scatter_pass` (renderer path), after the existing `for draw in draws` loop:

```rust
        draw_hlod_clusters(
            batch, state.eye_contract, state.render_from_contract,
            state.instance_scale, device, queue, &mut pass, renderer,
            state.view, state.proj, state.light_dir, state.light_intensity,
        );
```

And similarly in `render_scatter_batches` (viewer path), after the existing draw loop:

```rust
        draw_hlod_clusters(
            batch, eye_contract, render_from_contract,
            instance_scale, device, queue, &mut pass, renderer,
            view, proj, light_dir, light_intensity,
        );
```

- [ ] **Step 3: Parse HLOD config in `py_api.rs`**

In `set_scatter_batches`, after parsing `levels`, add:

```rust
            let hlod_config = batch_dict
                .get_item("hlod")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| -> PyResult<HlodConfig> {
                    let hlod_dict = value.downcast::<PyDict>().map_err(|_| {
                        PyRuntimeError::new_err(format!("batch {batch_index}: 'hlod' must be a dict"))
                    })?;
                    let hlod_distance: f32 = hlod_dict
                        .get_item("hlod_distance")?
                        .ok_or_else(|| PyRuntimeError::new_err("hlod missing 'hlod_distance'"))?
                        .extract()?;
                    let cluster_radius: f32 = hlod_dict
                        .get_item("cluster_radius")?
                        .ok_or_else(|| PyRuntimeError::new_err("hlod missing 'cluster_radius'"))?
                        .extract()?;
                    let simplify_ratio: f32 = hlod_dict
                        .get_item("simplify_ratio")?
                        .ok_or_else(|| PyRuntimeError::new_err("hlod missing 'simplify_ratio'"))?
                        .extract()?;
                    Ok(HlodConfig { hlod_distance, cluster_radius, simplify_ratio })
                })
                .transpose()?;
```

And add `hlod_config` to the `TerrainScatterUploadBatch` construction.

Add import at top: `use crate::terrain::scatter::HlodConfig;`

- [ ] **Step 4: Add HLOD to viewer config**

In `src/viewer/viewer_enums/config.rs`, update `ViewerTerrainScatterBatchConfig`:

```rust
pub struct ViewerTerrainScatterBatchConfig {
    pub name: Option<String>,
    pub color: [f32; 4],
    pub max_draw_distance: Option<f32>,
    pub transforms: Vec<[f32; 16]>,
    pub levels: Vec<ViewerTerrainScatterLevelConfig>,
    pub hlod_config: Option<crate::terrain::scatter::HlodConfig>,  // NEW
}
```

- [ ] **Step 5: Add HLOD to IPC payloads**

In `src/viewer/ipc/protocol/payloads.rs`, update `IpcTerrainScatterBatch`:

```rust
#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterBatch {
    // ... existing fields ...
    #[serde(default)]
    pub hlod: Option<IpcHlodConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IpcHlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}
```

- [ ] **Step 6: Map IPC HLOD to viewer config**

In `src/viewer/ipc/protocol/translate/terrain.rs`, update `map_terrain_scatter_batch`:

```rust
fn map_terrain_scatter_batch(config: &IpcTerrainScatterBatch) -> ViewerTerrainScatterBatchConfig {
    ViewerTerrainScatterBatchConfig {
        // ... existing fields ...
        hlod_config: config.hlod.as_ref().map(|h| crate::terrain::scatter::HlodConfig {
            hlod_distance: h.hlod_distance,
            cluster_radius: h.cluster_radius,
            simplify_ratio: h.simplify_ratio,
        }),
    }
}
```

- [ ] **Step 7: Pass HLOD through viewer scatter path**

In `src/viewer/terrain/scene/scatter.rs`, update `set_scatter_batches_from_configs`:

```rust
            gpu_batches.push(TerrainScatterBatch::new(
                self.device.as_ref(),
                self.queue.as_ref(),
                levels,
                &batch.transforms,
                batch.color,
                batch.max_draw_distance,
                batch.name.clone(),
                batch.hlod_config.clone(),  // NEW
            )?);
```

- [ ] **Step 8: Update HLOD stats in py_api.rs scatter stats getter**

In `get_scatter_stats()`, add the new HLOD fields to the returned dict:

```rust
    dict.set_item("hlod_cluster_draws", stats.hlod_cluster_draws)?;
    dict.set_item("hlod_covered_instances", stats.hlod_covered_instances)?;
    dict.set_item("effective_draws", stats.effective_draws)?;
```

Also add HLOD fields to `get_scatter_memory_report()`:

```rust
    dict.set_item("hlod_cluster_count", report.hlod_cluster_count)?;
    dict.set_item("hlod_buffer_bytes", report.hlod_buffer_bytes)?;
```

- [ ] **Step 9: Add viewer IPC HLOD round-trip test**

In `tests/test_viewer_ipc.py`, add a test to `TestIpcPayloadShapes` that verifies the `set_terrain_scatter` command preserves HLOD config through JSON serialization (matching the existing `test_set_terrain_scatter_format` pattern):

```python
    def test_set_terrain_scatter_hlod_format(self):
        """set_terrain_scatter with hlod field round-trips through JSON."""
        cmd = {
            "cmd": "set_terrain_scatter",
            "batches": [
                {
                    "name": "trees_hlod",
                    "color": [0.2, 0.6, 0.3, 1.0],
                    "max_draw_distance": 300.0,
                    "transforms": [[1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1]],
                    "levels": [
                        {"positions": [[0, 0, 0]], "normals": [[0, 1, 0]], "indices": [0, 1, 2]},
                    ],
                    "hlod": {
                        "hlod_distance": 100.0,
                        "cluster_radius": 25.0,
                        "simplify_ratio": 0.1,
                    },
                }
            ],
        }
        parsed = json.loads(json.dumps(cmd))
        assert parsed["batches"][0]["hlod"]["hlod_distance"] == 100.0
        assert parsed["batches"][0]["hlod"]["cluster_radius"] == 25.0
        assert parsed["batches"][0]["hlod"]["simplify_ratio"] == 0.1

    def test_set_terrain_scatter_no_hlod_backward_compat(self):
        """set_terrain_scatter without hlod field still works (backward compat)."""
        cmd = {
            "cmd": "set_terrain_scatter",
            "batches": [
                {
                    "name": "trees",
                    "transforms": [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
                    "levels": [],
                }
            ],
        }
        parsed = json.loads(json.dumps(cmd))
        assert "hlod" not in parsed["batches"][0]
```

- [ ] **Step 10: Build and run all existing tests**

```bash
maturin develop --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz --profile release-lto
python -m pytest tests/test_terrain_scatter.py -v
python -m pytest tests/test_api_contracts.py -v
python -m pytest tests/test_viewer_ipc.py -v
cargo test --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz
```

Expected: All existing tests still pass (backward compatibility). New viewer IPC tests pass.

- [ ] **Step 11: Commit**

```bash
git add src/terrain/renderer/scatter.rs src/terrain/renderer/py_api.rs \
    src/viewer/viewer_enums/config.rs src/viewer/ipc/protocol/payloads.rs \
    src/viewer/ipc/protocol/translate/terrain.rs src/viewer/terrain/scene/scatter.rs \
    tests/test_viewer_ipc.py
git commit -m "feat(tv13.3): plumb HLOD config through renderer, viewer, and IPC paths"
```

---

## Task 9: HLOD integration tests

**Files:**
- Modify: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Write HLOD rendering integration tests**

Add to `tests/test_terrain_tv13_lod_pipeline.py`:

```python
class TestHLODRendering:
    """TV13.3 — HLOD rendering integration tests (require GPU)."""

    @pytest.fixture
    def gpu_session(self):
        if not f3d.has_gpu():
            pytest.skip("GPU not available")
        return f3d.Session(window=False)

    def _create_test_hdr(self, path):
        """Write a minimal HDR for IBL."""
        with open(path, "wb") as fh:
            fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            fh.write(bytes([128, 128, 180, 128] * 32))

    def _build_render_context(self, gpu_session):
        """Build renderer, IBL, params, and heightmap for HLOD tests."""
        import tempfile, os
        from forge3d.terrain_params import make_terrain_params_config

        heightmap = np.sin(np.mgrid[0:96, 0:96][1].astype(np.float32) / 7.0) * 8.0 \
            + np.cos(np.mgrid[0:96, 0:96][0].astype(np.float32) / 9.0) * 6.0 + 25.0
        heightmap = heightmap.astype(np.float32)

        renderer = f3d.TerrainRenderer(gpu_session)
        material_set = f3d.MaterialSet.terrain_default()

        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = tmp.name
        try:
            self._create_test_hdr(hdr_path)
            ibl = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
        finally:
            os.unlink(hdr_path)

        config = make_terrain_params_config(
            size_px=(256, 160),
            render_scale=1.0,
            terrain_span=180.0,
            msaa_samples=4,
            z_scale=1.4,
            exposure=1.0,
            domain=(float(np.min(heightmap)), float(np.max(heightmap))),
            cam_radius=220.0,
            cam_phi_deg=138.0,
            cam_theta_deg=57.0,
            fov_y_deg=48.0,
        )
        params = f3d.TerrainRenderParams(config)
        return renderer, material_set, ibl, params, heightmap

    def _build_dense_scatter(self, hlod=None):
        """Build a scatter batch with many instances for HLOD testing."""
        cone = primitive_mesh("cone", radial_segments=16)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=200.0)

        # 100 instances in a 10x10 grid
        transforms = []
        for x in range(10):
            for z in range(10):
                transforms.append([
                    1, 0, 0, float(x * 5),
                    0, 1, 0, 0,
                    0, 0, 1, float(z * 5),
                    0, 0, 0, 1,
                ])
        transforms = np.array(transforms, dtype=np.float32)

        return TerrainScatterBatch(
            levels=levels,
            transforms=transforms,
            name="dense_scatter",
            max_draw_distance=500.0,
            hlod=hlod,
        )

    def test_hlod_none_preserves_baseline(self, gpu_session):
        """hlod=None produces zero HLOD stats after rendering a frame."""
        renderer, material_set, ibl, params, heightmap = self._build_render_context(gpu_session)
        batch = self._build_dense_scatter(hlod=None)
        ts.apply_to_renderer(renderer, [batch])
        # Must render a frame — stats are populated during the render pass
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)
        stats = renderer.get_scatter_stats()
        assert stats["hlod_cluster_draws"] == 0
        assert stats["hlod_covered_instances"] == 0
        assert stats["effective_draws"] > 0  # individual draws still happen

    def test_hlod_renders_and_reports_stats(self, gpu_session):
        """Batch with HLOD policy renders successfully and reports HLOD stats."""
        renderer, material_set, ibl, params, heightmap = self._build_render_context(gpu_session)
        policy = HLODPolicy(hlod_distance=50.0, cluster_radius=15.0, simplify_ratio=0.1)
        batch = self._build_dense_scatter(hlod=policy)
        ts.apply_to_renderer(renderer, [batch])
        # Render a frame to populate stats
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)
        stats = renderer.get_scatter_stats()
        # At the test camera distance (220), most instances are beyond hlod_distance (50)
        assert stats["hlod_cluster_draws"] > 0
        assert stats["hlod_covered_instances"] > 0
        assert stats["effective_draws"] > 0

    def test_hlod_memory_tracked(self, gpu_session):
        """HLOD memory is reported and included in total."""
        renderer, _, _, _, _ = self._build_render_context(gpu_session)
        policy = HLODPolicy(hlod_distance=50.0, cluster_radius=15.0, simplify_ratio=0.1)
        batch = self._build_dense_scatter(hlod=policy)
        ts.apply_to_renderer(renderer, [batch])
        # Memory report is available after batch upload (no render needed)
        report = renderer.get_scatter_memory_report()
        assert report["hlod_cluster_count"] > 0
        assert report["hlod_buffer_bytes"] > 0
        total = report["total_buffer_bytes"]
        assert total >= report["hlod_buffer_bytes"]
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestHLODRendering -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_terrain_tv13_lod_pipeline.py
git commit -m "test(tv13.3): add HLOD rendering integration tests"
```

---

## Task 10: Example demo with real DEM

**Files:**
- Create: `examples/terrain_tv13_lod_pipeline_demo.py`

- [ ] **Step 1: Create the demo script**

Create `examples/terrain_tv13_lod_pipeline_demo.py` following the pattern in `terrain_tv3_scatter_demo.py`:

```python
"""TV13 — Terrain Population LOD Pipeline Demo.

Demonstrates automatic LOD chain generation and HLOD clustering on
a real DEM (Mount Fuji). Renders scatter with auto-LOD and HLOD
to PNG, prints stats and memory report.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        import forge3d as f3d
        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import
        ensure_repo_import()
        import forge3d as f3d
        return f3d


f3d = _import_forge3d()
ts = f3d.terrain_scatter

from forge3d.geometry import primitive_mesh, simplify_mesh, generate_lod_chain
from forge3d.terrain_scatter import (
    TerrainScatterBatch,
    TerrainScatterLevel,
    TerrainScatterSource,
    TerrainScatterFilters,
    HLODPolicy,
    auto_lod_levels,
    apply_to_renderer,
)
from forge3d.terrain_params import PomSettings, make_terrain_params_config

DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv13_lod_pipeline"


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def _load_dem(dem_path: Path, downsample: int = 2):
    heightmap = f3d.io.load_dem(str(dem_path))
    arr = np.flipud(heightmap.array)
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="TV13 LOD Pipeline Demo")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.dem.exists():
        print(f"DEM not found: {args.dem}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heightmap = _load_dem(args.dem)
    print(f"Heightmap: {heightmap.shape}, range [{heightmap.min():.0f}, {heightmap.max():.0f}]")

    # --- Setup renderer ---
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    hdr_path = Path(tempfile.mktemp(suffix=".hdr"))
    _write_preview_hdr(hdr_path)
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)

    h_min, h_max = float(heightmap.min()), float(heightmap.max())
    terrain_span = max(heightmap.shape[0], heightmap.shape[1])
    config = make_terrain_params_config(
        size_px=(512, 320),
        render_scale=1.0,
        terrain_span=float(terrain_span),
        msaa_samples=4,
        z_scale=1.8,
        exposure=1.0,
        domain=(h_min, h_max),
        camera_mode="mesh",
        light_azimuth_deg=135.0,
        light_elevation_deg=35.0,
        cam_phi_deg=180.0,
        cam_theta_deg=35.0,
        cam_radius=float(terrain_span) * 0.6,
    )
    params = f3d.TerrainRenderParams(config)

    # --- TV13.1: Simplify a mesh ---
    tree_mesh = primitive_mesh("cone", radial_segments=24)
    print(f"\nOriginal tree mesh: {tree_mesh.triangle_count} triangles")
    simplified = simplify_mesh(tree_mesh, 0.25)
    print(f"Simplified (0.25): {simplified.triangle_count} triangles")

    # --- TV13.1: Generate LOD chain ---
    chain = generate_lod_chain(tree_mesh, [1.0, 0.25, 0.07])
    print(f"\nLOD chain: {[m.triangle_count for m in chain]} triangles")

    # --- TV13.2: Auto LOD levels ---
    source = TerrainScatterSource(heightmap)
    draw_dist = ts.viewer_orbit_radius(source.terrain_width, scale=0.5)

    auto_levels = auto_lod_levels(tree_mesh, lod_count=3, draw_distance=draw_dist)
    print(f"\nAuto LOD levels:")
    for i, level in enumerate(auto_levels):
        print(f"  L{i}: {level.mesh.triangle_count} tris, max_distance={level.max_distance}")

    # --- Place scatter ---
    filters = TerrainScatterFilters(min_slope_deg=0, max_slope_deg=25)
    transforms = ts.grid_jitter_transforms(
        source, spacing=8.0, seed=42, jitter=0.3, filters=filters,
        yaw_range_deg=(0.0, 360.0), scale_range=(0.3, 0.8), edge_margin=0.05,
    )
    print(f"\nPlaced {transforms.shape[0]} instances")

    # --- Render without HLOD (baseline) ---
    batch_no_hlod = TerrainScatterBatch(
        levels=auto_levels,
        transforms=transforms,
        name="trees_no_hlod",
        color=(0.3, 0.55, 0.2, 1.0),
        max_draw_distance=draw_dist,
    )
    apply_to_renderer(renderer, [batch_no_hlod])
    frame_baseline = renderer.render_terrain_pbr_pom(
        material_set, ibl, params, heightmap,
    )
    baseline_path = args.output_dir / "baseline_auto_lod.png"
    frame_baseline.save(str(baseline_path))
    stats_baseline = renderer.get_scatter_stats()
    print(f"\nBaseline (no HLOD):")
    print(f"  Stats: {stats_baseline}")

    # --- TV13.3: Render with HLOD ---
    policy = HLODPolicy(
        hlod_distance=draw_dist * 0.4,
        cluster_radius=draw_dist * 0.1,
        simplify_ratio=0.1,
    )
    batch_hlod = TerrainScatterBatch(
        levels=auto_levels,
        transforms=transforms,
        name="trees_hlod",
        color=(0.3, 0.55, 0.2, 1.0),
        max_draw_distance=draw_dist,
        hlod=policy,
    )
    apply_to_renderer(renderer, [batch_hlod])
    frame_hlod = renderer.render_terrain_pbr_pom(
        material_set, ibl, params, heightmap,
    )
    hlod_path = args.output_dir / "hlod_enabled.png"
    frame_hlod.save(str(hlod_path))
    stats_hlod = renderer.get_scatter_stats()
    memory_report = renderer.get_scatter_memory_report()
    print(f"\nHLOD enabled:")
    print(f"  Stats: {stats_hlod}")
    print(f"  Memory: {memory_report}")

    # Cleanup
    hdr_path.unlink(missing_ok=True)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

```bash
cd C:/Users/milos/forge3d
python examples/terrain_tv13_lod_pipeline_demo.py
```

Expected: Prints stats, saves two PNGs to `examples/out/terrain_tv13_lod_pipeline/`.

- [ ] **Step 3: Verify PNG output**

Check that both `baseline_auto_lod.png` and `hlod_enabled.png` exist and are non-empty:

```bash
ls -la examples/out/terrain_tv13_lod_pipeline/
```

- [ ] **Step 4: Commit**

```bash
git add examples/terrain_tv13_lod_pipeline_demo.py
git commit -m "feat(tv13): add terrain_tv13_lod_pipeline_demo.py example with real DEM"
```

---

## Task 11: End-to-end image output test

**Files:**
- Modify: `tests/test_terrain_tv13_lod_pipeline.py`

- [ ] **Step 1: Write end-to-end test**

Add to `tests/test_terrain_tv13_lod_pipeline.py`:

```python
import os
import tempfile
from pathlib import Path


class TestEndToEndImageOutput:
    """TV13 end-to-end: auto-LOD scatter renders to PNG with real content."""

    @pytest.fixture
    def gpu_session(self):
        if not f3d.has_gpu():
            pytest.skip("GPU not available")
        return f3d.Session(window=False)

    def test_auto_lod_scatter_produces_nonempty_image(self, gpu_session):
        """Scatter with auto_lod_levels renders a non-black PNG."""
        from forge3d.terrain_params import make_terrain_params_config

        # Small synthetic heightmap
        heightmap = np.random.default_rng(42).uniform(0, 100, (64, 64)).astype(np.float32)

        renderer = f3d.TerrainRenderer(gpu_session)
        material_set = f3d.MaterialSet.terrain_default()

        # Minimal HDR for IBL
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = tmp.name
        with open(hdr_path, "wb") as fh:
            fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            fh.write(bytes([128, 128, 180, 128] * 32))
        try:
            ibl = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
        finally:
            os.unlink(hdr_path)

        config = make_terrain_params_config(
            size_px=(256, 160),
            render_scale=1.0,
            terrain_span=64.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(float(heightmap.min()), float(heightmap.max())),
            camera_mode="mesh",
            cam_radius=80.0,
        )
        params = f3d.TerrainRenderParams(config)

        # Auto-LOD scatter
        cone = primitive_mesh("cone", radial_segments=16)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=50.0)

        transforms = np.array([
            [1, 0, 0, 20, 0, 1, 0, 50, 0, 0, 1, 20, 0, 0, 0, 1],
            [1, 0, 0, 40, 0, 1, 0, 50, 0, 0, 1, 40, 0, 0, 0, 1],
        ], dtype=np.float32)

        batch = TerrainScatterBatch(
            levels=levels, transforms=transforms, name="e2e_test",
            color=(0.8, 0.2, 0.2, 1.0), max_draw_distance=100.0,
        )
        ts.apply_to_renderer(renderer, [batch])

        frame = renderer.render_terrain_pbr_pom(
            material_set, ibl, params, heightmap,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            frame.save(tmp.name)
            assert Path(tmp.name).stat().st_size > 100, "PNG should be non-trivial"
            Path(tmp.name).unlink()

        hdr_path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_terrain_tv13_lod_pipeline.py::TestEndToEndImageOutput -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/test_terrain_tv13_lod_pipeline.py -v
python -m pytest tests/test_terrain_scatter.py -v
python -m pytest tests/test_api_contracts.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_terrain_tv13_lod_pipeline.py
git commit -m "test(tv13): add end-to-end image output test for auto-LOD scatter"
```

---

## Task 12: Documentation

**Files:**
- Modify: `docs/plans/2026-03-16-terrain-viz-epics.md`

- [ ] **Step 1: Update the epics document**

In the **Effort Summary → Core build backlog** table, update the TV13 row to indicate it is implemented. Add a note in the **Implemented foundations** section:

```markdown
| TV13 - Terrain Population LOD Pipeline | Implemented |
```

- [ ] **Step 2: Commit**

```bash
git add docs/plans/2026-03-16-terrain-viz-epics.md
git commit -m "docs(tv13): mark TV13 as implemented in terrain viz epics"
```

---

## Task 13: Final verification

- [ ] **Step 1: Run full Rust test suite**

```bash
cargo test --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz
```

Expected: All tests pass.

- [ ] **Step 2: Rebuild native extension**

```bash
maturin develop --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz --profile release-lto
```

- [ ] **Step 3: Run full Python test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass including new TV13 tests.

- [ ] **Step 4: Run the demo one more time**

```bash
python examples/terrain_tv13_lod_pipeline_demo.py
```

Expected: Completes successfully, prints stats, saves PNGs.

- [ ] **Step 5: Verify image outputs**

Visually inspect or verify the generated PNGs are non-trivial (file size > 10KB, scatter is visible).
