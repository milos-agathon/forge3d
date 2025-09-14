# Triangle Mesh BVH API (A3)

This document describes the triangle mesh BVH construction and path tracing API implemented in Task A3.

## Overview

The mesh BVH system provides GPU-accelerated triangle mesh rendering through:
- CPU BVH construction with GPU-compatible layout
- GPU BVH traversal with watertight triangle intersection
- Python API for mesh creation, BVH building, and GPU upload
- Integration with existing path tracing functionality

## Python API

### Mesh Creation

```python
import numpy as np
from forge3d.mesh import make_mesh, validate_mesh_arrays

# Create mesh from vertex and index arrays
vertices = np.array([
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 0.0, 0.0],  # Vertex 1
    [0.5, 1.0, 0.0],  # Vertex 2
], dtype=np.float32)

indices = np.array([
    [0, 1, 2],  # Triangle using vertices 0, 1, 2
], dtype=np.uint32)

# Validate arrays (optional - make_mesh will validate automatically)
validate_mesh_arrays(vertices, indices)

# Create mesh object
mesh = make_mesh(vertices, indices)
print(f"Created mesh: {mesh['vertex_count']} vertices, {mesh['triangle_count']} triangles")
```

**Requirements:**
- `vertices`: NumPy array of shape `(N, 3)` with dtype `float32` or `float64`
- `indices`: NumPy array of shape `(M, 3)` with dtype `uint32` or compatible integer type
- Both arrays must be C-contiguous
- Triangle indices use counter-clockwise winding order

### BVH Construction

```python
from forge3d.mesh import build_bvh_cpu

# Build BVH acceleration structure
bvh = build_bvh_cpu(mesh, method="median")

print(f"BVH Statistics:")
print(f"  Nodes: {bvh['node_count']}")
print(f"  Max depth: {bvh['max_depth']}")
print(f"  Leaf count: {bvh['leaf_count']}")
print(f"  Build time: {bvh['build_time_ms']:.2f}ms")
print(f"  Memory usage: {bvh['memory_usage_bytes']} bytes")
```

**BVH Methods:**
- `"median"`: Median-split partitioning (currently the only supported method)

**BVH Output:**
The BVH dictionary contains:
- `method`: Construction method used
- `triangle_count`: Number of triangles
- `node_count`: Number of BVH nodes
- `max_depth`: Maximum tree depth
- `leaf_count`: Number of leaf nodes
- `avg_leaf_size`: Average triangles per leaf
- `build_time_ms`: Construction time in milliseconds
- `memory_usage_bytes`: Estimated memory usage
- `world_aabb_min`: World bounding box minimum
- `world_aabb_max`: World bounding box maximum

### GPU Upload

```python
from forge3d.mesh import upload_mesh, mesh_info

# Upload mesh and BVH to GPU
mesh_handle = upload_mesh(mesh, bvh)

# Get information about uploaded mesh
info = mesh_info(mesh_handle)
print(f"Uploaded mesh info: {info}")

# Access handle properties
print(f"Triangle count: {mesh_handle.triangle_count}")
print(f"Vertex count: {mesh_handle.vertex_count}")
print(f"BVH nodes: {mesh_handle.node_count}")

# Get build statistics
stats = mesh_handle.build_stats
print(f"Build stats: {stats}")

# Get world bounding box
aabb_min, aabb_max = mesh_handle.world_aabb
print(f"World AABB: {aabb_min} to {aabb_max}")
```

### Path Tracing Integration

```python
from forge3d.path_tracing import render_rgba, render_aovs

# Create camera
camera = {
    'origin': (2, 2, 2),
    'look_at': (0.5, 0.5, 0.5),
    'up': (0, 1, 0),
    'fov_y': 45.0,
    'aspect': 1.0,
    'exposure': 1.0
}

# Render with mesh (and optional spheres)
scene = [{"center": (0, 0, -1), "radius": 0.3, "albedo": (1.0, 0.0, 0.0)}]
img = render_rgba(256, 256, scene, camera, seed=42, frames=1, mesh=mesh_handle)

# Render AOVs with mesh
aovs = render_aovs(256, 256, scene, camera,
                   aovs=("albedo", "normal", "depth", "visibility"),
                   seed=42, frames=1, mesh=mesh_handle)
```

## Convenience Functions

### Built-in Test Meshes

```python
from forge3d.mesh import create_triangle_mesh, create_cube_mesh, create_quad_mesh

# Simple triangle (3 vertices, 1 triangle)
vertices, indices = create_triangle_mesh()

# Unit cube (8 vertices, 12 triangles)
vertices, indices = create_cube_mesh()

# Quad (4 vertices, 2 triangles)
vertices, indices = create_quad_mesh()

# Use with mesh creation
mesh = make_mesh(vertices, indices)
```

## GPU Implementation Details

### BVH Node Layout

The BVH uses a GPU-compatible node layout optimized for traversal:

```c
struct BvhNode {
    vec3 aabb_min;     // AABB minimum bounds
    u32  left;         // Internal: left child index; Leaf: first triangle index
    vec3 aabb_max;     // AABB maximum bounds
    u32  right;        // Internal: right child index; Leaf: triangle count
    u32  flags;        // Bit 0: leaf flag (1 = leaf, 0 = internal)
    u32  _pad;         // Padding for alignment
};
```

- **Size**: 32 bytes (8 × 4-byte values)
- **Alignment**: 4-byte aligned for GPU compatibility
- **Layout**: Matches WGSL struct exactly for direct GPU upload

### WGSL Integration

The system provides WGSL functions for GPU traversal:

- `bvh_intersect()`: Closest-hit BVH traversal
- `bvh_intersect_any()`: Any-hit traversal for shadow rays
- `ray_triangle_intersect()`: Watertight Möller-Trumbore intersection
- `ray_aabb_intersect()`: AABB intersection testing

**Bind Group Layout:**
- Group 0: Uniforms (camera, frame parameters)
- Group 1: Scene data (spheres, vertices, indices, BVH nodes)
- Group 2: Output buffers (accumulation, textures)

## Performance Considerations

### Memory Usage

- **BVH nodes**: ~32 bytes per node
- **Triangle indices**: 4 bytes per triangle
- **Vertices**: 16 bytes per vertex (with padding)
- **Total**: Approximately 50-100 bytes per triangle

### GPU Memory Budget

The system respects the ≤512 MiB host-visible heap constraint:
- Large meshes may need tiling or level-of-detail
- BVH construction is performed on CPU to minimize GPU memory pressure
- Temporary build data is not retained after GPU upload

### Build Performance

Typical BVH construction performance:
- **Small meshes** (< 1K triangles): < 1ms
- **Medium meshes** (1K-100K triangles): 1-100ms
- **Large meshes** (> 100K triangles): 100ms-1s

GPU traversal performance scales with scene complexity and depth complexity.

## Error Handling

### Mesh Validation Errors

```python
# Common validation errors and solutions:

# Wrong vertex shape
vertices = np.array([[0, 0]], dtype=np.float32)  # Missing Z coordinate
# Solution: vertices must be (N, 3) shape

# Non-contiguous arrays
vertices = vertices[::2]  # Creates non-contiguous array
# Solution: use np.ascontiguousarray(vertices)

# Invalid indices
indices = np.array([[0, 1, 5]], dtype=np.uint32)  # Index 5 > vertex count
# Solution: ensure all indices are < vertex_count

# Wrong dtype
vertices = vertices.astype(np.int32)  # Wrong data type
# Solution: use float32 or float64 for vertices, uint32 for indices
```

### BVH Construction Errors

```python
# Empty mesh
mesh = make_mesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32))
bvh = build_bvh_cpu(mesh)  # Raises ValueError
# Solution: ensure mesh has at least one triangle

# Unsupported method
bvh = build_bvh_cpu(mesh, method="unsupported")  # Raises ValueError
# Solution: use method="median"
```

### Upload Errors

```python
# Mismatched triangle counts
mesh1 = make_mesh(vertices1, indices1)  # 10 triangles
bvh2 = build_bvh_cpu(mesh2)  # Built from different mesh with 5 triangles
handle = upload_mesh(mesh1, bvh2)  # Raises ValueError
# Solution: ensure mesh and BVH are built from same triangle data
```

## Limitations

### Current Limitations

1. **BVH Method**: Only median-split is currently supported (SAH planned for future)
2. **GPU Memory**: Large meshes may exceed host-visible heap limits
3. **Material System**: Uses default material properties (PBR integration planned)
4. **Animation**: Static meshes only (no deformation or animation support yet)

### Planned Features

- SAH (Surface Area Heuristic) BVH construction
- GPU BVH construction for very large meshes
- Material attribute support (normals, UVs, materials)
- Instancing support for repeated geometry
- Mesh deformation and animation

## Examples

See `tests/test_mesh_tracing_gpu.py` for comprehensive usage examples and `examples/` directory for complete working examples.

## Integration Notes

The mesh BVH system is designed to:
- Work alongside existing sphere path tracing
- Fall back gracefully to CPU when GPU is unavailable
- Provide deterministic results with fixed seeds
- Scale from simple test cases to complex production meshes
- Maintain compatibility with existing path tracing workflows