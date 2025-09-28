# Geometry Utilities (Workstream F)

Forge3D exposes a set of geometry helpers for authoring, normalizing, and validating meshes. The
APIs mirror the Rust implementations contained under `src/geometry/` and are accessible through the
`forge3d.geometry` module.

## Mesh Transforms

`forge3d.geometry` provides high-level helpers that operate on the `MeshBuffers` container. Each
function returns a new `MeshBuffers` instance and, when relevant, additional metadata describing the
transform.

* `center_mesh(mesh, target=None) -> (MeshBuffers, np.ndarray)`
  * Recenters the mesh so that its axis-aligned bounding-box midpoint matches `target` (defaults to
    the origin). Returns the transformed mesh and the previous center.
* `scale_mesh(mesh, scale, pivot=None) -> (MeshBuffers, bool)`
  * Applies non-uniform scaling about an optional pivot. The boolean indicates whether triangle
    winding was flipped.
* `flip_mesh_axis(mesh, axis) -> (MeshBuffers, bool)`
  * Mirrors the mesh across the requested axis (`0 = X`, `1 = Y`, `2 = Z`). Returns the transformed
    mesh and a winding flip flag.
* `swap_mesh_axes(mesh, axis_a, axis_b) -> (MeshBuffers, bool)`
  * Exchanges two axes on both positions and normals. Swapping two axes is equivalent to an odd
    permutation, so winding flips as indicated by the boolean.
* `mesh_bounds(mesh) -> Optional[(np.ndarray, np.ndarray)]`
  * Returns the axis-aligned bounds as `(min, max)` vectors or `None` when the mesh is empty.

All helpers preserve UV channels when present and recompute normals using the appropriate inverse
transpose of the linear transform.

## Python Mesh Container

The `MeshBuffers` dataclass mirrors the Rust `MeshBuffers` structure and includes `positions`,
`normals`, `uvs`, and `indices`. Utility functions such as `extrude_polygon`, `primitive_mesh`,
`validate_mesh`, and `weld_mesh` return this container to facilitate chained transforms.

## Validation & Welding

* `validate_mesh(positions, indices)` runs topology checks (degenerate triangles, non-manifold
  edges, duplicate vertices) and returns a structured report.
* `weld_mesh(positions, indices, uvs=None, *, position_epsilon=1e-5, uv_epsilon=1e-4)` merges
  vertices using epsilon thresholds, recomputes normals, and returns both the welded mesh and a
  remap table.

## Related Rust Modules

* `src/geometry/transform.rs` – Core transform implementations.
* `src/geometry/extrude.rs` – Polygon extrusion utilities.
* `src/geometry/primitives.rs` – Parametric primitive generation.
* `src/geometry/validate.rs` – Mesh diagnostics and reporting.
* `src/geometry/weld.rs` – Vertex welding and smoothing.

These modules are re-exported through `src/geometry/mod.rs`, ensuring consistent access from both
Rust and Python entry points.
