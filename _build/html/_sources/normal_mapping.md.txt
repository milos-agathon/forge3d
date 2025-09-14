# Normal Mapping

Normal mapping adds surface detail to 3D models without increasing geometric complexity. Forge3D provides a complete normal mapping pipeline that integrates with the TBN generation system to transform tangent-space normal maps into world-space surface details.

## Overview

Normal mapping works by:

1. **Storing surface detail** in tangent-space normal map textures
2. **Computing TBN basis vectors** for each vertex using the mesh TBN system  
3. **Transforming normals** from tangent space to world space during rendering
4. **Applying enhanced lighting** using the perturbed surface normals

This technique enables:
- **Detailed surfaces** without high-poly geometry
- **Realistic lighting** with fine surface variations  
- **Efficient rendering** with minimal performance cost
- **Cross-platform compatibility** using WebGPU/WGSL shaders

## API Reference

### Python API

```python
import forge3d.normalmap as normalmap

# Create test normal maps
normal_map = normalmap.create_checkerboard_normal_map(256)

# Validate normal map data
results = normalmap.validate_normal_map(normal_map)
assert results['valid']

# Encode/decode normal vectors
encoded = normalmap.encode_normal_vector([0.2, 0.3, 0.9])
decoded = normalmap.decode_normal_vector(encoded)

# Load/save normal maps
normal_map = normalmap.load_normal_map("assets/brick_normal.png")
normalmap.save_normal_map(normal_map, "output/normal.png")

# Compute luminance differences for validation
diff = normalmap.compute_luminance_difference(image1, image2)
```

### Rust API

```rust
use forge3d::pipeline::{NormalMappingPipeline, NormalMappingUniforms};

// Create normal mapping pipeline
let pipeline = NormalMappingPipeline::new(&device, surface_format);

// Upload mesh data
pipeline.upload_mesh(&device, &vertices, &indices);

// Update uniforms
let uniforms = NormalMappingUniforms {
    model_matrix: model.to_cols_array_2d(),
    view_matrix: view.to_cols_array_2d(),
    projection_matrix: proj.to_cols_array_2d(),
    normal_matrix: compute_normal_matrix(model).to_cols_array_2d(),
    light_direction: [0.5, -1.0, 0.3, 1.0],
    normal_strength: 1.0,
    ..Default::default()
};

pipeline.update_uniforms(&queue, &uniforms);

// Render with normal mapping
pipeline.render(&mut render_pass, &uniforms_bind_group, &texture_bind_group);
```

## Usage Guide

### 1. Prepare Mesh with TBN Data

Normal mapping requires meshes with tangent-space basis vectors:

```python
import forge3d.mesh as mesh

# Generate mesh with TBN data
vertices, indices, tbn_data = mesh.generate_cube_tbn()

# Validate TBN for normal mapping
validation = mesh.validate_tbn_data(tbn_data)
assert validation['valid']
```

### 2. Create or Load Normal Maps

Normal maps store tangent-space surface details:

```python
# Option 1: Create test patterns
checkerboard = normalmap.create_checkerboard_normal_map(128)

# Option 2: Load from file
normal_map = normalmap.load_normal_map("assets/surface_normal.png")

# Validate normal map
validation = normalmap.validate_normal_map(normal_map)
if not validation['valid']:
    print("Issues:", validation['errors'])
```

### 3. Set Up GPU Pipeline

Configure the rendering pipeline for normal mapping:

```rust
// Vertex attributes must include tangent/bitangent
let vertex_layout = VertexBufferLayout {
    array_stride: std::mem::size_of::<TbnVertex>() as BufferAddress,
    attributes: &[
        VertexAttribute { offset: 0,  shader_location: 0, format: Float32x3 }, // position
        VertexAttribute { offset: 12, shader_location: 1, format: Float32x2 }, // uv  
        VertexAttribute { offset: 20, shader_location: 2, format: Float32x3 }, // normal
        VertexAttribute { offset: 32, shader_location: 3, format: Float32x3 }, // tangent
        VertexAttribute { offset: 44, shader_location: 4, format: Float32x3 }, // bitangent
    ],
};
```

### 4. Configure Textures and Samplers

Normal maps require proper texture setup:

```rust
// Create normal map texture
let normal_texture = device.create_texture(&TextureDescriptor {
    size: Extent3d { width, height, depth_or_array_layers: 1 },
    format: TextureFormat::Rgba8Unorm,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    // ... other settings
});

// Create linear filtering sampler
let sampler = device.create_sampler(&SamplerDescriptor {
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    // ... other settings
});
```

## Shader Integration  

### WGSL Vertex Shader

Transform vertex attributes to world space:

```wgsl
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Transform TBN vectors to world space
    let normal_mat = mat3x3<f32>(
        uniforms.normal_matrix[0].xyz,
        uniforms.normal_matrix[1].xyz, 
        uniforms.normal_matrix[2].xyz
    );
    
    output.world_normal = normalize(normal_mat * input.normal);
    output.world_tangent = normalize(normal_mat * input.tangent);
    output.world_bitangent = normalize(normal_mat * input.bitangent);
    
    // ... other transformations
}
```

### WGSL Fragment Shader

Sample and apply normal mapping:

```wgsl
@fragment  
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample normal map
    let normal_sample = textureSample(normal_texture, normal_sampler, input.uv);
    
    // Apply normal mapping
    let final_normal = sample_normal_map(
        normal_sample,
        uniforms.normal_strength,
        input.world_tangent,
        input.world_bitangent,
        input.world_normal
    );
    
    // Lighting calculations with perturbed normal
    let light_dir = normalize(-uniforms.light_direction.xyz);
    let ndotl = max(0.0, dot(final_normal, light_dir));
    
    return vec4<f32>(vec3<f32>(ndotl), 1.0);
}
```

## Normal Map Formats

### Standard Encoding

Normal maps use RGB channels to encode XYZ normal components:

- **Red Channel**: Tangent X component [-1, 1] → [0, 255]
- **Green Channel**: Tangent Y component [-1, 1] → [0, 255]  
- **Blue Channel**: Tangent Z component [0, 1] → [128, 255]
- **Alpha Channel**: Optional (unused or material masks)

### Coordinate Systems

Forge3D uses **right-handed tangent space** with:

- **X-axis**: Tangent direction (U texture axis)
- **Y-axis**: Bitangent direction (V texture axis)
- **Z-axis**: Surface normal (pointing outward)

## Validation and Testing

### Luminance Difference Test

Validate normal mapping effectiveness:

```python
# Render with and without normal mapping
flat_image = render_with_flat_normals(mesh)
normal_mapped_image = render_with_normal_mapping(mesh, normal_map)

# Compute luminance difference  
diff = normalmap.compute_luminance_difference(normal_mapped_image, flat_image)

# AC requirement: ≥10% difference
assert diff >= 10.0, f"Insufficient effect: {diff:.1f}% < 10%"
```

### TBN Matrix Validation

Ensure proper tangent space setup:

```python
results = mesh.validate_tbn_data(tbn_data)
assert results['valid'], f"TBN errors: {results['errors']}"
assert results['unit_length_ok']
assert results['orthogonal_ok'] 
assert results['handedness_ok']
```

### GPU Validation

Verify shader binding without errors:

```python
# Test creates GPU pipeline with TBN vertex layout
# Logs should show no wgpu validation errors
pytest tests/test_tbn_gpu_validation.py -v -s
```

See `tests/test_normal_mapping.py` for automated validation testing of luminance differences and NaN detection.

## Performance Considerations

### Memory Usage

- **TBN vertices**: 56 bytes vs 32 bytes (75% increase)
- **Normal maps**: RGBA8 format, mipmap chains recommended
- **GPU memory**: Budget for vertex buffers + texture atlases

### Rendering Performance

- **Vertex processing**: Additional TBN transform calculations
- **Fragment processing**: Extra texture samples and matrix operations
- **Bandwidth**: Higher vertex data throughput requirements

### Optimization Tips

1. **Use texture atlases** to reduce draw calls
2. **Enable mipmapping** for distant surfaces
3. **Pack normals** in compressed formats when possible
4. **Precompute TBN data** for static meshes
5. **Validate in debug only** to avoid runtime overhead

## Examples

### Basic Normal Mapping

```bash
python examples/normal_mapping_demo.py --headless --out out/normal_map.png
```

This example:
- Creates a test mesh with TBN data
- Generates a checkerboard normal map
- Renders with flat normals vs normal mapping
- Computes and validates ≥10% luminance difference
- Saves side-by-side comparison image

### Custom Normal Map Loading

```python
import forge3d.normalmap as normalmap

# Load custom normal map
normal_map = normalmap.load_normal_map("assets/brick_normal.png")

# Validate format
validation = normalmap.validate_normal_map(normal_map)
if not validation['valid']:
    print("Fixing issues:", validation['errors'])

# Apply to rendering pipeline
# ... pipeline setup code
```

## Troubleshooting

### Common Issues

1. **No visual effect**: Check TBN generation and normal map validity
2. **Inverted normals**: Verify coordinate system handedness
3. **Artifacts at edges**: Ensure proper UV unwrapping and TBN continuity
4. **Performance drops**: Profile vertex/fragment shader complexity
5. **Memory errors**: Check GPU memory budget and texture sizes

### Debug Techniques

```python
# Enable validation logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check TBN orthogonality
for tbn in tbn_data:
    t = np.array(tbn['tangent'])
    n = np.array(tbn['normal'])
    dot_product = abs(np.dot(t, n))
    if dot_product > 1e-3:
        print(f"Non-orthogonal TBN: {dot_product}")

# Validate normal map encoding
decoded = normalmap.decode_normal_vector(normal_map)
lengths = np.linalg.norm(decoded, axis=2)
invalid_pixels = np.sum(np.abs(lengths - 1.0) > 0.1)
print(f"Invalid normal pixels: {invalid_pixels}")
```

## Feature Flags

Normal mapping requires feature flags:

```toml
[dependencies]
forge3d = { features = ["enable-tbn", "enable-normal-mapping"] }
```

## Cross-Platform Support

Normal mapping works consistently across:

- **Windows**: DirectX 12 + HLSL compilation
- **Linux**: Vulkan + SPIR-V compilation
- **macOS**: Metal + MSL compilation

All platforms use identical WGSL shaders ensuring consistent visual results.