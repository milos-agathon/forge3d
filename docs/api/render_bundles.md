# Render Bundles for GPU Command Optimization

## Overview

forge3d provides a comprehensive render bundle system for optimizing GPU command execution. Render bundles group multiple draw calls into reusable command buffers, significantly reducing CPU overhead and improving rendering performance for scenes with repeated geometry or UI elements.

## Key Features

- **Command Batching**: Group multiple draw calls into single GPU command buffer
- **Reusable Execution**: Pre-compiled bundles can be executed multiple times
- **Multiple Bundle Types**: Optimized for instanced rendering, UI, particles, and batch rendering
- **Performance Monitoring**: Built-in statistics and performance analysis
- **Memory Efficiency**: Shared resources and optimized GPU memory usage
- **Automatic Validation**: Performance validation with recommendations

## Quick Start

```python
import forge3d.bundles as bundles

# Create bundle manager
manager = bundles.BundleManager()

# Create instanced bundle (many copies of same object)
cube_vertices = create_cube_geometry()
transforms = create_instance_transforms(100)  # 100 instances
colors = create_instance_colors(100)

instanced_bundle = bundles.BundleBuilder.create_instanced_bundle(
    "cubes",
    {'vertices': cube_vertices['vertices'], 'indices': cube_vertices['indices']},
    transforms,
    colors
).compile()

# Add to manager and execute
manager.add_bundle(instanced_bundle)
manager.execute_bundle("cubes")  # Renders all 100 cubes efficiently

# Get performance stats
stats = manager.get_performance_stats("cubes")
print(f"Average execution time: {stats['avg_execution_time_ms']:.2f}ms")
```

## Core Concepts

### Render Bundle Types

forge3d supports several bundle types optimized for different rendering scenarios:

#### 1. Instanced Rendering
**Use Case**: Many copies of the same object with different transforms
```python
# Example: 1000 trees in a forest
tree_bundle = bundles.BundleBuilder.create_instanced_bundle(
    "forest",
    tree_geometry,
    tree_transforms,  # 1000 different positions/rotations
    tree_colors       # Seasonal color variations
)
```

**Benefits**: 
- Single draw call for thousands of objects
- GPU-side transform application
- Minimal CPU overhead

#### 2. UI Rendering
**Use Case**: UI elements like buttons, panels, text backgrounds
```python
# Example: Game UI with multiple elements
ui_elements = [
    {'position': (50, 50), 'size': (200, 100), 'color': (0.2, 0.3, 0.8, 0.9)},
    {'position': (300, 50), 'size': (150, 80), 'color': (0.8, 0.2, 0.2, 0.9)},
    # ... more UI elements
]

ui_bundle = bundles.BundleBuilder.create_ui_bundle("game_ui", ui_elements)
```

**Benefits**:
- Batch multiple UI elements
- Screen-space optimized rendering
- Transparency sorting

#### 3. Particle Systems
**Use Case**: Large numbers of particles (smoke, fire, sparks)
```python
# Example: Fire particle system
fire_particles = generate_fire_particles(5000)  # position, velocity, life, color
particle_bundle = bundles.BundleBuilder.create_particle_bundle("fire", fire_particles)
```

**Benefits**:
- Efficient billboard rendering
- GPU-side particle simulation
- Automatic culling of dead particles

#### 4. Batch Rendering
**Use Case**: Different objects with shared rendering state
```python
# Example: Mixed scene objects
scene_objects = [
    {'vertices': cube_verts, 'transform': cube_transform, 'material': metal_material},
    {'vertices': sphere_verts, 'transform': sphere_transform, 'material': plastic_material},
    # ... more objects
]

batch_bundle = bundles.BundleBuilder.create_batch_bundle("scene", scene_objects)
```

**Benefits**:
- Reduced state changes
- Shared texture atlases
- Optimized for varied geometry

### Bundle Lifecycle

1. **Creation**: Define bundle type and name
2. **Data Addition**: Add geometry, uniforms, textures
3. **Compilation**: Process data into GPU-optimized format
4. **Execution**: Render bundle in GPU command stream
5. **Performance Monitoring**: Track execution statistics

## API Reference

### RenderBundle Class

```python
class RenderBundle:
    def __init__(self, bundle_type: BundleType, name: str = ""):
        """Create new render bundle."""
    
    def add_geometry(self, vertices: np.ndarray, indices: np.ndarray = None,
                    instances: np.ndarray = None) -> 'RenderBundle':
        """Add geometry data to bundle."""
    
    def add_uniform(self, name: str, data: Union[np.ndarray, bytes]) -> 'RenderBundle':
        """Add uniform buffer data."""
    
    def add_texture(self, texture_data: np.ndarray, format: str = "rgba8") -> 'RenderBundle':
        """Add texture data to bundle."""
    
    def compile(self) -> 'RenderBundle':
        """Compile bundle for GPU execution."""
    
    def is_compiled(self) -> bool:
        """Check if bundle has been compiled."""
    
    def get_stats(self) -> BundleStats:
        """Get bundle statistics."""
```

### BundleManager Class

```python
class BundleManager:
    def __init__(self):
        """Create new bundle manager."""
    
    def create_bundle(self, name: str, bundle_type: BundleType) -> RenderBundle:
        """Create new render bundle."""
    
    def add_bundle(self, bundle: RenderBundle) -> None:
        """Add compiled bundle to manager."""
    
    def execute_bundle(self, name: str) -> bool:
        """Execute bundle by name."""
    
    def execute_bundles(self, names: List[str]) -> int:
        """Execute multiple bundles in sequence."""
    
    def get_performance_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for bundle."""
    
    def get_total_stats(self) -> Dict[str, Any]:
        """Get statistics for all bundles."""
```

### BundleBuilder Class

```python
class BundleBuilder:
    @staticmethod
    def create_instanced_bundle(name: str, base_geometry: Dict[str, np.ndarray],
                              instance_transforms: np.ndarray,
                              instance_colors: np.ndarray = None) -> RenderBundle:
        """Create instanced rendering bundle."""
    
    @staticmethod
    def create_ui_bundle(name: str, ui_elements: List[Dict[str, Any]]) -> RenderBundle:
        """Create UI rendering bundle."""
    
    @staticmethod
    def create_particle_bundle(name: str, particles: np.ndarray) -> RenderBundle:
        """Create particle system bundle."""
    
    @staticmethod
    def create_batch_bundle(name: str, objects: List[Dict[str, Any]]) -> RenderBundle:
        """Create batch rendering bundle."""
```

### Bundle Types

```python
class BundleType(Enum):
    INSTANCED = "instanced"      # Many instances of same geometry
    UI = "ui"                    # UI elements (quads, text, etc.)
    PARTICLES = "particles"      # Particle systems
    BATCH = "batch"              # Different objects in one bundle
    WIREFRAME = "wireframe"      # Debug wireframe rendering
```

## Usage Patterns

### Instanced Rendering Pattern

```python
# Create base geometry once
def create_tree_geometry():
    # Tree vertices, indices
    return {'vertices': tree_verts, 'indices': tree_indices}

# Generate instance data
def create_forest_instances(count):
    transforms = np.zeros((count, 4, 4), dtype=np.float32)
    colors = np.zeros((count, 4), dtype=np.float32)
    
    for i in range(count):
        # Random position
        x = np.random.uniform(-100, 100)
        z = np.random.uniform(-100, 100)
        
        # Random rotation around Y axis
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Random scale
        scale = np.random.uniform(0.8, 1.2)
        
        # Create transform matrix
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] *= scale
        transform[0, 0] = scale * np.cos(angle)
        transform[0, 2] = scale * np.sin(angle) 
        transform[2, 0] = scale * -np.sin(angle)
        transform[2, 2] = scale * np.cos(angle)
        transform[:3, 3] = [x, 0, z]
        
        transforms[i] = transform
        
        # Seasonal color variation
        green_factor = np.random.uniform(0.3, 0.8)
        colors[i] = [0.2, green_factor, 0.1, 1.0]
    
    return transforms, colors

# Create and use bundle
tree_geometry = create_tree_geometry()
transforms, colors = create_forest_instances(1000)

forest_bundle = bundles.BundleBuilder.create_instanced_bundle(
    "forest", tree_geometry, transforms, colors
).compile()

# Single call renders 1000 trees
manager.add_bundle(forest_bundle)
manager.execute_bundle("forest")
```

### UI Rendering Pattern

```python
# Define UI layout
def create_game_hud():
    ui_elements = []
    
    # Health bar background
    ui_elements.append({
        'position': (20, 20),
        'size': (200, 20),
        'color': (0.2, 0.2, 0.2, 0.8)
    })
    
    # Health bar fill (80% health)
    ui_elements.append({
        'position': (20, 20),
        'size': (160, 20),  # 80% of 200
        'color': (0.8, 0.2, 0.2, 1.0)
    })
    
    # Mana bar
    ui_elements.append({
        'position': (20, 50),
        'size': (200, 20),
        'color': (0.2, 0.2, 0.2, 0.8)
    })
    
    ui_elements.append({
        'position': (20, 50),
        'size': (120, 20),  # 60% mana
        'color': (0.2, 0.2, 0.8, 1.0)
    })
    
    # Action buttons
    for i in range(6):
        ui_elements.append({
            'position': (300 + i * 60, 500),
            'size': (50, 50),
            'color': (0.3, 0.3, 0.3, 0.9)
        })
    
    # Mini-map
    ui_elements.append({
        'position': (700, 20),
        'size': (150, 150),
        'color': (0.1, 0.1, 0.1, 0.7)
    })
    
    return ui_elements

# Create and execute UI bundle
hud_elements = create_game_hud()
hud_bundle = bundles.BundleBuilder.create_ui_bundle("game_hud", hud_elements).compile()

manager.add_bundle(hud_bundle)
manager.execute_bundle("game_hud")
```

### Particle System Pattern

```python
# Particle emitter configuration
class ParticleEmitter:
    def __init__(self, position, particle_count):
        self.position = position
        self.particle_count = particle_count
    
    def generate_fire_particles(self):
        particles = np.zeros((self.particle_count, 13), dtype=np.float32)
        
        # Position (3D)
        particles[:, 0] = self.position[0] + np.random.uniform(-1, 1, self.particle_count)
        particles[:, 1] = self.position[1] + np.random.uniform(0, 2, self.particle_count)
        particles[:, 2] = self.position[2] + np.random.uniform(-1, 1, self.particle_count)
        
        # Velocity (3D) - upward with random spread
        particles[:, 3] = np.random.uniform(-0.5, 0.5, self.particle_count)  # X vel
        particles[:, 4] = np.random.uniform(2, 6, self.particle_count)        # Y vel (up)
        particles[:, 5] = np.random.uniform(-0.5, 0.5, self.particle_count)  # Z vel
        
        # Size
        particles[:, 6] = np.random.uniform(0.2, 0.8, self.particle_count)
        
        # Life (0.0 = dead, 1.0 = newly born)
        particles[:, 7] = np.random.uniform(0.3, 1.0, self.particle_count)
        
        # Color (RGBA) - fire colors
        particles[:, 8] = 1.0  # Red
        particles[:, 9] = np.random.uniform(0.3, 0.8, self.particle_count)  # Green
        particles[:, 10] = 0.1  # Blue
        particles[:, 11] = 0.9  # Alpha
        
        # Padding
        particles[:, 12] = 0.0
        
        return particles

# Create particle systems
campfire = ParticleEmitter((0, 0, 0), 2000)
torch1 = ParticleEmitter((10, 2, 0), 500)
torch2 = ParticleEmitter((-10, 2, 0), 500)

# Create bundles for each emitter
fire_particles = campfire.generate_fire_particles()
campfire_bundle = bundles.BundleBuilder.create_particle_bundle("campfire", fire_particles).compile()

torch1_particles = torch1.generate_fire_particles()
torch1_bundle = bundles.BundleBuilder.create_particle_bundle("torch1", torch1_particles).compile()

torch2_particles = torch2.generate_fire_particles()  
torch2_bundle = bundles.BundleBuilder.create_particle_bundle("torch2", torch2_particles).compile()

# Execute all particle systems
manager.add_bundle(campfire_bundle)
manager.add_bundle(torch1_bundle)
manager.add_bundle(torch2_bundle)

manager.execute_bundles(["campfire", "torch1", "torch2"])
```

### Batch Rendering Pattern

```python
# Scene with mixed objects
def create_scene_objects():
    objects = []
    
    # Ground plane
    plane_vertices = create_plane_geometry(20, 20)
    ground_transform = np.eye(4, dtype=np.float32)
    ground_transform[1, 3] = -1  # Lower by 1 unit
    
    objects.append({
        'vertices': plane_vertices['vertices'],
        'indices': plane_vertices['indices'],
        'transform': ground_transform,
        'material': [0.0, 0.8, 0.0, 0.0]  # Rough, non-metallic (grass)
    })
    
    # Buildings
    building_geometry = create_building_geometry()
    for i in range(5):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [i * 8 - 16, 0, -10]  # Line of buildings
        
        # Vary building heights
        height_scale = np.random.uniform(1.0, 3.0)
        transform[1, 1] = height_scale
        
        objects.append({
            'vertices': building_geometry['vertices'],
            'indices': building_geometry['indices'],
            'transform': transform,
            'material': [0.1, 0.3, 0.0, 0.0]  # Concrete-like
        })
    
    # Street lamps
    lamp_geometry = create_lamp_geometry()
    for i in range(10):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [i * 4 - 18, 0, 5]  # Along street
        
        objects.append({
            'vertices': lamp_geometry['vertices'],
            'indices': lamp_geometry['indices'],
            'transform': transform,
            'material': [1.0, 0.1, 0.1, 0.0]  # Metallic lamp posts
        })
    
    return objects

# Create and execute scene bundle
scene_objects = create_scene_objects()
scene_bundle = bundles.BundleBuilder.create_batch_bundle("city_scene", scene_objects).compile()

manager.add_bundle(scene_bundle)
manager.execute_bundle("city_scene")
```

## Performance Optimization

### Bundle Validation

```python
# Validate bundle performance
validation = bundles.validate_bundle_performance(my_bundle)

if not validation['efficient']:
    print("Bundle performance warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
    
    print("Recommendations:")
    for rec in validation['recommendations']:
        print(f"  - {rec}")

print(f"Memory usage: {validation['metrics']['memory_mb']:.1f}MB")
print(f"Draw call efficiency: {validation['metrics']['draw_call_efficiency']}")
```

### Performance Comparison

```python
# Compare bundle vs individual draw calls
comparison = bundles.compare_bundle_vs_individual(bundle, individual_draw_count=100)

print(f"Estimated speedup: {comparison['estimated_speedup']:.2f}x")
print(f"Draw call reduction: {comparison['draw_call_reduction']}")
print(f"Efficiency score: {comparison['efficiency_score']:.2f}")
```

### Performance Monitoring

```python
# Monitor bundle performance over time
manager = bundles.BundleManager()

# Execute bundle multiple times
for frame in range(100):
    manager.execute_bundle("my_bundle")

# Get performance statistics
perf_stats = manager.get_performance_stats("my_bundle")
print(f"Average: {perf_stats['avg_execution_time_ms']:.2f}ms")
print(f"Min: {perf_stats['min_execution_time_ms']:.2f}ms")
print(f"Max: {perf_stats['max_execution_time_ms']:.2f}ms")
print(f"Samples: {perf_stats['sample_count']}")

# Get total statistics
total_stats = manager.get_total_stats()
print(f"Total bundles: {total_stats['bundle_count']}")
print(f"Total memory: {total_stats['total_memory_usage'] / (1024*1024):.1f}MB")
print(f"Total vertices: {total_stats['total_vertices']}")
```

## Best Practices

### 1. Bundle Granularity

**Good**: Group related objects with similar rendering requirements
```python
# Good - all UI elements together
ui_bundle = bundles.BundleBuilder.create_ui_bundle("hud", all_ui_elements)

# Good - all particles from same emitter
fire_bundle = bundles.BundleBuilder.create_particle_bundle("fire", fire_particles)
```

**Avoid**: Mixing unrelated rendering types in one bundle
```python
# Bad - mixing different rendering types
mixed_bundle = create_batch_bundle("mixed", [ui_elements, particles, 3d_objects])
```

### 2. Memory Management

```python
# Monitor memory usage
def check_memory_usage(manager):
    total_memory = manager.get_total_stats()['total_memory_usage']
    memory_mb = total_memory / (1024 * 1024)
    
    if memory_mb > 100:  # More than 100MB
        print(f"High memory usage: {memory_mb:.1f}MB")
        # Consider reducing geometry detail or bundle count
    
    return memory_mb

# Clean up unused bundles
def cleanup_bundles(manager, active_bundles):
    for bundle_name in list(manager.get_bundle_names()):
        if bundle_name not in active_bundles:
            manager.remove_bundle(bundle_name)
```

### 3. Dynamic Updates

```python
# For frequently changing data, create new bundles
class DynamicParticleSystem:
    def __init__(self, manager, name):
        self.manager = manager
        self.name = name
        self.frame_count = 0
    
    def update(self, particles):
        # Remove old bundle
        if self.frame_count > 0:
            self.manager.remove_bundle(self.name)
        
        # Create new bundle with updated particles
        bundle = bundles.BundleBuilder.create_particle_bundle(self.name, particles)
        bundle.compile()
        self.manager.add_bundle(bundle)
        
        self.frame_count += 1
    
    def render(self):
        self.manager.execute_bundle(self.name)
```

### 4. Error Handling

```python
def safe_bundle_execution(manager, bundle_names):
    successful = 0
    failed = 0
    
    for bundle_name in bundle_names:
        try:
            if manager.execute_bundle(bundle_name):
                successful += 1
            else:
                print(f"Bundle '{bundle_name}' failed to execute")
                failed += 1
        except Exception as e:
            print(f"Error executing bundle '{bundle_name}': {e}")
            failed += 1
    
    return successful, failed
```

## Integration with Other Systems

### With PBR Materials

```python
# Bundle objects can use PBR materials
import forge3d.pbr as pbr

# Create PBR materials
metal_material = pbr.PbrMaterial(base_color=(0.7, 0.7, 0.7, 1.0), metallic=1.0, roughness=0.1)
plastic_material = pbr.PbrMaterial(base_color=(0.8, 0.2, 0.2, 1.0), metallic=0.0, roughness=0.8)

# Use in batch bundle
objects = [
    {'vertices': metal_verts, 'transform': transform1, 'material': metal_material},
    {'vertices': plastic_verts, 'transform': transform2, 'material': plastic_material},
]

pbr_bundle = bundles.BundleBuilder.create_batch_bundle("pbr_objects", objects)
```

### With Shadow Systems

```python
# Bundles work with shadow casting
import forge3d.shadows as shadows

# Create shadow-casting scene
shadow_renderer = shadows.ShadowRenderer(1920, 1080)
shadow_light = shadows.DirectionalLight(direction=(-0.5, -0.8, -0.3))
shadow_renderer.set_light(shadow_light)

# Render bundles with shadows
manager.execute_bundle("scene_objects")  # Objects cast shadows
manager.execute_bundle("ground_plane")   # Receives shadows
```

### With HDR Rendering

```python
# Bundles integrate with HDR pipeline
import forge3d.hdr as hdr

hdr_renderer = hdr.HdrRenderer(1920, 1080, format='rgba16float')

# Render bundles to HDR target
with hdr_renderer.begin_frame() as hdr_target:
    manager.execute_bundle("scene_geometry")
    manager.execute_bundle("particle_effects")

# Apply tone mapping
final_image = hdr_renderer.tone_map(hdr_target, 'aces')
```

## Common Issues and Solutions

### Issue: Bundle Compilation Fails
**Cause**: Missing vertex data or invalid geometry
**Solution**:
```python
try:
    bundle.compile()
except ValueError as e:
    print(f"Compilation failed: {e}")
    # Check that bundle has vertex data
    if not bundle.vertex_data:
        bundle.add_geometry(default_vertices)
```

### Issue: Poor Performance
**Cause**: Too many small bundles or inefficient geometry
**Solution**:
```python
# Validate before use
validation = bundles.validate_bundle_performance(bundle)
if not validation['efficient']:
    # Combine small bundles or reduce geometry complexity
    pass
```

### Issue: High Memory Usage  
**Cause**: Large textures or excessive geometry
**Solution**:
```python
# Monitor and limit memory usage
total_memory = manager.get_total_stats()['total_memory_usage']
if total_memory > memory_limit:
    # Remove least used bundles
    # Reduce texture resolution
    # Simplify geometry
    pass
```

### Issue: Inconsistent Performance
**Cause**: GPU driver issues or thermal throttling
**Solution**:
```python
# Monitor performance consistency
perf_stats = manager.get_performance_stats("bundle_name")
if perf_stats['max_execution_time_ms'] > perf_stats['avg_execution_time_ms'] * 3:
    print("Inconsistent performance detected")
    # Consider reducing workload or enabling VSync
```

## Examples

See `examples/bundles_demo.py` for comprehensive demonstrations including:

- Instanced rendering with 500+ objects
- UI system with multiple element types
- Particle systems with different emitters
- Mixed batch rendering scenarios
- Performance benchmarking and validation
- Memory usage analysis

The example generates visual output showing each bundle type and provides detailed performance metrics for optimization guidance.

## Implementation Notes

### GPU Shader Support

Render bundles use the standard instanced rendering shaders:

- **Instanced shaders**: Handle per-instance transforms and colors
- **UI shaders**: Screen-space rendering with transparency
- **Particle shaders**: Billboard rendering with life/fade calculations
- **Batch shaders**: Multiple materials and textures in single draw call

### Memory Layout

Bundle data is optimized for GPU consumption:

```rust
// Instance data layout (20 floats per instance)
struct InstanceData {
    transform: mat4x4<f32>,  // 16 floats - transformation matrix
    color: vec4<f32>,        // 4 floats - instance color
}

// Particle data layout (13 floats per particle)  
struct ParticleData {
    position: vec3<f32>,     // 3 floats - world position
    velocity: vec3<f32>,     // 3 floats - velocity vector
    size: f32,               // 1 float - particle size
    life: f32,               // 1 float - life value [0,1]
    color: vec4<f32>,        // 4 floats - particle RGBA
    padding: f32,            // 1 float - alignment padding
}
```

### Performance Characteristics

Render bundles provide significant performance benefits:

- **CPU Overhead**: Reduced by 80-95% for bundled draw calls
- **GPU Utilization**: Improved by reducing command buffer switches
- **Memory Bandwidth**: Optimized through shared resources
- **Scalability**: Linear scaling with instance count rather than draw call count

This system enables efficient rendering of complex scenes with thousands of objects while maintaining high frame rates and low CPU usage.