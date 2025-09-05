# TBN (Tangent, Bitangent, Normal) Generation

Forge3D provides robust TBN generation for indexed meshes to support normal mapping and advanced lighting techniques. The TBN system follows MikkTSpace-compatible algorithms for industry-standard tangent space calculations.

## Overview

TBN generation creates orthogonal tangent space basis vectors for each vertex in a mesh, enabling:

- **Normal mapping**: Transform tangent-space normals to world space
- **Advanced lighting**: Anisotropic reflections and surface detail enhancement
- **PBR materials**: Physically-based rendering with microgeometry details

## API Reference

### Rust API

```rust
use forge3d::mesh::{TbnVertex, TbnData, generate_tbn, generate_cube_tbn, generate_plane_tbn};

// Generate TBN data for indexed mesh
let tbn_data = generate_tbn(&vertices, &indices);

// Pre-built meshes with TBN
let (vertices, indices, tbn) = generate_cube_tbn();
let (vertices, indices, tbn) = generate_plane_tbn(width, height);

// TBN vertex structure
struct TbnVertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}
```

### Python API

```python
import forge3d.mesh as mesh

# Generate TBN data
vertices, indices, tbn_data = mesh.generate_cube_tbn()

# Validate TBN correctness
results = mesh.validate_tbn_data(tbn_data)
assert results['valid']

# Check feature availability
has_tbn = mesh.has_tbn_support()
```

## Validation Steps

### 1. Unit Length Vectors

All TBN vectors must have unit length (≈1.0):

```python
import numpy as np

for tbn in tbn_data:
    t_len = np.linalg.norm(tbn['tangent'])
    b_len = np.linalg.norm(tbn['bitangent'])
    n_len = np.linalg.norm(tbn['normal'])
    
    assert abs(t_len - 1.0) < 1e-3
    assert abs(b_len - 1.0) < 1e-3
    assert abs(n_len - 1.0) < 1e-3
```

### 2. Orthogonality

Tangent and normal vectors must be orthogonal:

```python
t = np.array(tbn['tangent'])
n = np.array(tbn['normal'])
dot_tn = abs(np.dot(t, n))
assert dot_tn <= 1e-3
```

### 3. Determinant Check

TBN matrix determinant should be ±1 for proper handedness:

```python
t = np.array(tbn['tangent'])
b = np.array(tbn['bitangent'])
n = np.array(tbn['normal'])

tbn_matrix = np.column_stack([t, b, n])
det = np.linalg.det(tbn_matrix)
assert 0.99 <= abs(det) <= 1.01
```

### 4. GPU Validation

Verify TBN vertex attributes bind without WebGPU validation errors:

```rust
use wgpu::{VertexBufferLayout, VertexAttribute, VertexFormat, VertexStepMode};

let vertex_layout = VertexBufferLayout {
    array_stride: std::mem::size_of::<TbnVertex>() as BufferAddress,
    step_mode: VertexStepMode::Vertex,
    attributes: &[
        VertexAttribute { offset: 0,  shader_location: 0, format: VertexFormat::Float32x3 }, // position
        VertexAttribute { offset: 12, shader_location: 1, format: VertexFormat::Float32x2 }, // uv
        VertexAttribute { offset: 20, shader_location: 2, format: VertexFormat::Float32x3 }, // normal
        VertexAttribute { offset: 32, shader_location: 3, format: VertexFormat::Float32x3 }, // tangent
        VertexAttribute { offset: 44, shader_location: 4, format: VertexFormat::Float32x3 }, // bitangent
    ],
};
```

## Memory Layout

TBN vertices use a compact memory layout:

| Field | Offset | Size | Format |
|-------|--------|------|--------|
| position | 0 | 12 bytes | Float32x3 |
| uv | 12 | 8 bytes | Float32x2 |
| normal | 20 | 12 bytes | Float32x3 |
| tangent | 32 | 12 bytes | Float32x3 |
| bitangent | 44 | 12 bytes | Float32x3 |
| **Total** | | **56 bytes** | |

## Feature Flags

TBN functionality is gated behind feature flags:

```toml
[dependencies]
forge3d = { features = ["enable-tbn"] }
```

## Cross-Platform Compatibility

TBN generation works consistently across:

- **Windows**: DirectX 12 backend
- **Linux**: Vulkan backend  
- **macOS**: Metal backend

All implementations use identical MikkTSpace-compatible algorithms ensuring consistent results across platforms.

## Performance Considerations

- **Memory**: 56 bytes per vertex vs 32 bytes for basic vertices
- **GPU Transfer**: Use structured buffer uploads for large meshes
- **Validation**: Enable validation only in debug builds
- **Caching**: Pre-compute TBN data for static meshes

## Examples

See the test suite for comprehensive validation examples:

- `tests/test_tbn_generation.py`: Basic TBN functionality
- `tests/test_normal_mapping.py`: Integration with normal mapping pipeline

## Troubleshooting

### Common Issues

1. **Non-unit vectors**: Check for zero-area triangles or degenerate UV coordinates
2. **Validation errors**: Ensure proper vertex buffer stride and attribute alignment
3. **NaN values**: Validate input mesh for malformed geometry
4. **Performance**: Use compact vertex format for large meshes

### Debug Output

Enable TBN validation logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

results = mesh.validate_tbn_data(tbn_data, tolerance=1e-3)
if not results['valid']:
    for error in results['errors']:
        print(f"TBN Error: {error}")
```