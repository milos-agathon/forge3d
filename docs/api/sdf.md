# SDF (Signed Distance Functions) Documentation

## Overview

The forge3d SDF module provides analytic signed distance function primitives and constructive solid geometry (CSG) operations for procedural geometry generation and rendering. This system combines traditional mesh rendering with SDF raymarching for hybrid geometry representation.

## Features

- **Analytic SDF Primitives**: Sphere, box, cylinder, plane, torus, capsule
- **CSG Operations**: Union, intersection, subtraction with smooth variants
- **Hybrid Rendering**: Combines SDF raymarching with BVH mesh traversal
- **GPU Acceleration**: WGSL compute shaders for high-performance rendering
- **Python API**: High-level Python interface for scene construction
- **Performance Optimization**: Early termination, bounds checking, soft shadows

## Core Concepts

### Signed Distance Functions

A Signed Distance Function (SDF) returns the distance from any point in 3D space to the surface of an object. The sign indicates whether the point is inside (negative) or outside (positive) the object.

```rust
// Example: Sphere SDF
fn sdf_sphere(point: Vec3, center: Vec3, radius: f32) -> f32 {
    (point - center).length() - radius
}
```

### Constructive Solid Geometry (CSG)

CSG operations combine multiple SDF primitives using boolean operations:

- **Union**: Combines shapes (minimum distance)
- **Intersection**: Creates overlap region (maximum distance)
- **Subtraction**: Removes one shape from another
- **Smooth variants**: Create organic blending between shapes

### Hybrid Traversal

The hybrid system combines:
- **SDF Raymarching**: Sphere tracing for analytic geometry
- **BVH Traversal**: Traditional ray-triangle intersection for meshes
- **Unified Pipeline**: Single rendering path for both geometry types

## Rust API

### Basic Usage

```rust
use forge3d::sdf::*;
use glam::Vec3;

// Create SDF primitives
let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
let box_prim = SdfPrimitive::box_primitive(Vec3::new(2.0, 0.0, 0.0), Vec3::ONE, 2);

// Build scene using builder pattern
let (builder, sphere_idx) = SdfSceneBuilder::new()
    .add_sphere(Vec3::new(-1.0, 0.0, 0.0), 0.8, 1);

let (builder, box_idx) = builder
    .add_box(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.7, 0.7, 0.7), 2);

let (builder, union_idx) = builder
    .union(sphere_idx, box_idx, 0);

let scene = builder.build();

// Evaluate SDF at a point
let result = scene.evaluate(Vec3::ZERO);
println!("Distance: {}, Material: {}", result.distance, result.material_id);
```

### Hybrid Scene Construction

```rust
// Create hybrid scene combining SDF and mesh data
let mut hybrid_scene = HybridScene::new();

// Set SDF geometry
hybrid_scene.set_sdf_scene(sdf_scene);

// Add mesh geometry (vertices, indices, BVH)
hybrid_scene.add_mesh(vertices, indices, bvh)?;

// Prepare GPU resources
hybrid_scene.prepare_gpu_resources()?;

// Ray intersection test
let ray = HybridRay {
    origin: Vec3::ZERO,
    direction: Vec3::new(0.0, 0.0, -1.0),
    tmin: 0.001,
    tmax: 100.0,
};

let hit_result = hybrid_scene.intersect(ray);
if hit_result.hit {
    println!("Hit at distance: {}", hit_result.t);
    println!("Hit type: {}", if hit_result.hit_type == 1 { "SDF" } else { "Mesh" });
}
```

### SDF Primitives

#### Sphere
```rust
let sphere = SdfPrimitive::sphere(
    Vec3::new(0.0, 0.0, 0.0), // center
    1.0,                      // radius
    1                         // material_id
);
```

#### Box
```rust
let box_prim = SdfPrimitive::box_primitive(
    Vec3::new(0.0, 0.0, 0.0),     // center
    Vec3::new(1.0, 1.0, 1.0),     // extents (half-size)
    2                             // material_id
);
```

#### Cylinder
```rust
let cylinder = SdfPrimitive::cylinder(
    Vec3::new(0.0, 0.0, 0.0), // center
    1.0,                      // radius
    2.0,                      // height
    3                         // material_id
);
```

#### Plane
```rust
let plane = SdfPrimitive::plane(
    Vec3::new(0.0, 1.0, 0.0), // normal
    0.0,                      // distance from origin
    4                         // material_id
);
```

#### Torus
```rust
let torus = SdfPrimitive::torus(
    Vec3::new(0.0, 0.0, 0.0), // center
    2.0,                      // major radius
    0.5,                      // minor radius
    5                         // material_id
);
```

#### Capsule
```rust
let capsule = SdfPrimitive::capsule(
    Vec3::new(-1.0, 0.0, 0.0), // point A
    Vec3::new(1.0, 0.0, 0.0),  // point B
    0.5,                       // radius
    6                          // material_id
);
```

### CSG Operations

#### Basic Operations
```rust
// Union: Combines shapes
let (builder, union_node) = builder.union(left_node, right_node, material_id);

// Intersection: Keeps only overlap
let (builder, intersect_node) = builder.intersect(left_node, right_node, material_id);

// Subtraction: Removes right from left
let (builder, subtract_node) = builder.subtract(left_node, right_node, material_id);
```

#### Smooth Operations
```rust
// Smooth union with blending parameter
let (builder, smooth_union) = builder.smooth_union(
    left_node,
    right_node,
    0.2,        // smoothing amount
    material_id
);

// Smooth subtraction
let (builder, smooth_subtract) = builder.smooth_subtract(
    left_node,
    right_node,
    0.1,        // smoothing amount
    material_id
);
```

## Python API

### Basic Usage

```python
import forge3d
from forge3d.sdf import *

# Create SDF primitives
sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)
box = SdfPrimitive.box((2, 0, 0), (1, 1, 1), 2)

# Build scene
builder = SdfSceneBuilder()
builder, sphere_idx = builder.add_sphere((0, 0, 0), 1.0, 1)
builder, box_idx = builder.add_box((2, 0, 0), (0.8, 0.8, 0.8), 2)
builder, union_idx = builder.union(sphere_idx, box_idx, 0)

scene = builder.build()

# Evaluate SDF
distance, material = scene.evaluate((0, 0, 0))
print(f"Distance: {distance}, Material: {material}")
```

### Rendering

```python
# Create renderer
renderer = HybridRenderer(512, 512)

# Set camera
renderer.set_camera(
    origin=(0, 0, 5),
    target=(0, 0, 0),
    up=(0, 1, 0),
    fov_degrees=45
)

# Set traversal mode
renderer.set_traversal_mode(TraversalMode.HYBRID)  # SDF + mesh
# renderer.set_traversal_mode(TraversalMode.SDF_ONLY)  # SDF only
# renderer.set_traversal_mode(TraversalMode.MESH_ONLY)  # Mesh only

# Render scene
image = renderer.render_sdf_scene(scene)  # Returns numpy array (H, W, 4)
```

### Convenience Functions

```python
# Quick primitive creation
sphere = create_sphere((0, 0, 0), 1.0, 1)
box = create_box((0, 0, 0), (1, 1, 1), 2)

# Quick scene creation and rendering
scene = create_simple_scene()
image = render_simple_scene(width=256, height=256)

# Save image (if using additional libraries)
from PIL import Image
Image.fromarray(image).save('output.png')
```

## WGSL Shader Interface

The SDF system provides WGSL compute shaders for GPU acceleration:

### Include Structure
```wgsl
#include "sdf_primitives.wgsl"    // Basic SDF evaluation functions
#include "sdf_operations.wgsl"    // CSG operations and domain ops
#include "hybrid_traversal.wgsl"  // Hybrid ray intersection
```

### Key Functions

#### SDF Evaluation
```wgsl
// Evaluate specific primitive types
fn sdf_sphere(point: vec3f, sphere: SdfSphere) -> f32;
fn sdf_box(point: vec3f, box: SdfBox) -> f32;
fn sdf_cylinder(point: vec3f, cylinder: SdfCylinder) -> f32;

// Generic primitive evaluation
fn evaluate_sdf_primitive(point: vec3f, primitive: SdfPrimitive) -> f32;

// Normal calculation using finite differences
fn sdf_normal(point: vec3f, primitive: SdfPrimitive) -> vec3f;
```

#### CSG Operations
```wgsl
// Basic CSG operations
fn csg_union(a: CsgResult, b: CsgResult) -> CsgResult;
fn csg_intersection(a: CsgResult, b: CsgResult) -> CsgResult;
fn csg_subtraction(a: CsgResult, b: CsgResult) -> CsgResult;

// Smooth CSG operations
fn csg_smooth_union(a: CsgResult, b: CsgResult, k: f32) -> CsgResult;
fn csg_smooth_intersection(a: CsgResult, b: CsgResult, k: f32) -> CsgResult;
fn csg_smooth_subtraction(a: CsgResult, b: CsgResult, k: f32) -> CsgResult;

// Apply operation by type
fn apply_csg_operation(operation: u32, a: CsgResult, b: CsgResult, smoothing: f32) -> CsgResult;
```

#### Hybrid Traversal
```wgsl
// Main hybrid intersection function
fn intersect_hybrid(ray: Ray) -> HybridHitResult;

// Optimized version with early termination
fn intersect_hybrid_optimized(ray: Ray, early_exit_distance: f32) -> HybridHitResult;

// SDF raymarching
fn raymarch_sdf(ray: Ray) -> HybridHitResult;

// Mesh intersection (BVH traversal)
fn intersect_mesh(ray: Ray) -> HybridHitResult;

// Surface properties
fn get_surface_properties(hit: HybridHitResult) -> vec3f;

// Shadow testing
fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool;
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32;
```

### Data Structures

#### Primitives
```wgsl
struct SdfPrimitive {
    primitive_type: u32,
    material_id: u32,
    _pad: vec2u,
    params: array<f32, 16>,  // Primitive-specific parameters
}

struct SdfSphere {
    center: vec3f,
    radius: f32,
}

struct SdfBox {
    center: vec3f,
    _pad1: f32,
    extents: vec3f,
    _pad2: f32,
}
```

#### CSG Tree
```wgsl
struct CsgNode {
    operation: u32,      // Operation type
    left_child: u32,     // Left child index
    right_child: u32,    // Right child index
    smoothing: f32,      // Smoothing parameter
    material_id: u32,    // Material ID
    is_leaf: u32,        // 1 if leaf node, 0 if operation
    _pad: vec2u,
}

struct CsgResult {
    distance: f32,       // Signed distance
    material_id: u32,    // Material identifier
}
```

#### Hybrid Results
```wgsl
struct HybridHitResult {
    t: f32,              // Ray parameter
    point: vec3f,        // Hit point in world space
    normal: vec3f,       // Surface normal
    material_id: u32,    // Material identifier
    hit_type: u32,       // 0 = mesh, 1 = SDF
    hit: u32,            // 0 = no hit, 1 = hit
    _pad: vec2u,
}
```

## Performance Considerations

### Optimization Techniques

1. **Early Ray Termination**: Stop raymarching when close enough to surface
2. **Bounding Volumes**: Use scene bounds to cull rays early
3. **Adaptive Step Sizes**: Larger steps in empty space, smaller near surfaces
4. **LOD System**: Use different detail levels based on distance
5. **Hybrid Rendering**: Combine fast BVH traversal with flexible SDF raymarching

### Performance Metrics

The system tracks several performance metrics:

```rust
struct HybridMetrics {
    sdf_steps: u32,           // Total raymarching steps
    bvh_nodes_visited: u32,   // BVH nodes traversed
    triangle_tests: u32,      // Triangle intersection tests
    total_rays: u32,          // Total rays cast
    sdf_hits: u32,            // Rays hitting SDF geometry
    mesh_hits: u32,           // Rays hitting mesh geometry
}

// Calculate performance overhead vs mesh-only rendering
let overhead = metrics.performance_overhead();  // Returns ratio
```

### Tuning Parameters

```python
# Performance tuning
renderer = HybridRenderer(512, 512)
renderer.set_performance_params(
    early_exit_distance=0.01,  # Ray termination threshold
    shadow_softness=4.0        # Soft shadow quality vs speed
)
```

## Common Usage Patterns

### Creating Complex Geometry

```python
def create_complex_shape():
    builder = SdfSceneBuilder()

    # Main body: union of two spheres
    builder, sphere1 = builder.add_sphere((-0.5, 0, 0), 0.8, 1)
    builder, sphere2 = builder.add_sphere((0.5, 0, 0), 0.8, 1)
    builder, body = builder.smooth_union(sphere1, sphere2, 0.2, 1)

    # Subtract a hole through the middle
    builder, hole = builder.add_cylinder((0, 0, 0), 0.3, 2.0, 2)
    builder, result = builder.subtract(body, hole, 1)

    return builder.build()
```

### Material-Based Rendering

```python
def render_with_materials(scene):
    renderer = HybridRenderer(512, 512)

    # Material properties are handled in the shader
    # Material IDs can be used to index into material arrays
    image = renderer.render_sdf_scene(scene)
    return image
```

### Animation and Deformation

```rust
// Animated scene (requires rebuilding per frame)
fn create_animated_scene(time: f32) -> SdfScene {
    let offset = Vec3::new(time.sin() * 2.0, 0.0, 0.0);
    let (builder, sphere_idx) = SdfSceneBuilder::new()
        .add_sphere(offset, 1.0, 1);

    builder.build()
}
```

## Error Handling

### Rust Error Types

```rust
use forge3d::error::RenderError;

match hybrid_scene.prepare_gpu_resources() {
    Ok(()) => println!("GPU resources ready"),
    Err(RenderError::Upload(msg)) => eprintln!("Upload failed: {}", msg),
    Err(RenderError::Device(msg)) => eprintln!("Device error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Python Error Handling

```python
try:
    image = renderer.render_sdf_scene(scene)
except RuntimeError as e:
    print(f"Rendering failed: {e}")
    # Fall back to CPU rendering or default scene
    image = render_fallback_scene()
```

## Debugging and Visualization

### SDF Visualization

```python
def visualize_sdf_distances(scene, bounds=(-2, 2), resolution=100):
    """Create a 2D slice visualization of SDF distances"""
    import numpy as np

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    distances = np.zeros((resolution, resolution))

    for i, px in enumerate(x):
        for j, py in enumerate(y):
            distance, _ = scene.evaluate((px, py, 0))
            distances[j, i] = distance

    return distances
```

### Performance Profiling

```rust
use std::time::Instant;

let start = Instant::now();
let result = hybrid_scene.intersect(ray);
let duration = start.elapsed();

println!("Ray intersection took: {:?}", duration);
```

## Advanced Features

### Domain Operations

The system supports domain transformation operations for creating procedural patterns:

```wgsl
// Infinite repetition
fn domain_repeat_infinite(point: vec3f, spacing: vec3f) -> vec3f;

// Limited repetition
fn domain_repeat_limited(point: vec3f, spacing: vec3f, limit: vec3f) -> vec3f;

// Twist transformation
fn domain_twist(point: vec3f, twist_amount: f32) -> vec3f;

// Bend transformation
fn domain_bend(point: vec3f, bend_amount: f32) -> vec3f;
```

### Soft Shadows

SDF geometry enables high-quality soft shadows:

```wgsl
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32 {
    // Returns 0.0 (full shadow) to 1.0 (no shadow)
    // Uses SDF field to compute penumbra
}
```

### Ambient Occlusion

SDFs can provide efficient ambient occlusion calculation:

```rust
fn calculate_ambient_occlusion(point: Vec3, normal: Vec3, scene: &SdfScene) -> f32 {
    let mut occlusion = 0.0;
    let step_size = 0.1;
    let max_distance = 1.0;

    for i in 1..=5 {
        let sample_point = point + normal * (step_size * i as f32);
        let distance = scene.evaluate(sample_point).distance;
        occlusion += (step_size * i as f32 - distance) / (1.0 + i as f32 * i as f32);
    }

    (1.0 - occlusion).max(0.0).min(1.0)
}
```

## Integration Examples

### With Existing Mesh Pipeline

```rust
// Combine SDF objects with traditional meshes
let mut hybrid_scene = HybridScene::new();

// Add SDF geometry
hybrid_scene.set_sdf_scene(procedural_sdf_scene);

// Add mesh geometry from file
let (vertices, indices) = load_obj_file("model.obj")?;
let bvh = build_bvh(&vertices, &indices)?;
hybrid_scene.add_mesh(vertices, indices, bvh)?;

// Render both together
let image = render_hybrid_scene(&hybrid_scene);
```

### With Path Tracing

```rust
use forge3d::path_tracing::hybrid_compute::*;

let hybrid_tracer = HybridPathTracer::new()?;

let params = HybridTracerParams {
    base_uniforms: create_camera_uniforms(width, height, camera),
    traversal_mode: TraversalMode::Hybrid,
    early_exit_distance: 0.01,
    shadow_softness: 4.0,
};

let image = hybrid_tracer.render(width, height, &spheres, &hybrid_scene, params)?;
```

## Future Extensions

The SDF system is designed to be extensible:

1. **Additional Primitives**: Easy to add new SDF shapes
2. **Custom Operations**: Implement domain-specific CSG operations
3. **Volumetric Rendering**: Extend to volumetric SDFs for clouds, smoke, etc.
4. **GPU Optimization**: Further WGSL optimizations for specific hardware
5. **AI Integration**: Use neural SDFs for learned geometry representation

## References

- [Inigo Quilez - SDF Functions](https://iquilezles.org/articles/distfunctions/)
- [Shadertoy - SDF Examples](https://www.shadertoy.com/results?query=tag%3Dsdf)
- [Raymarching Workshop](https://github.com/electricsquare/raymarching-workshop)
- [Real-Time Rendering of Signed Distance Functions](http://www.cs.columbia.edu/cg/pdfs/17_sdf.pdf)