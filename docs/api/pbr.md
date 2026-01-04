# Physically-Based Rendering (PBR) Materials

## Overview

forge3d provides a comprehensive PBR (Physically-Based Rendering) materials system implementing the metallic-roughness workflow. This system enables realistic material representation for metals, dielectrics, and complex surfaces with proper energy conservation and physically plausible light interaction.

## Key Features

- **Metallic-Roughness Workflow**: Industry-standard Disney/Unreal Engine material model
- **BRDF Implementation**: Cook-Torrance microfacet BRDF with GGX distribution
- **Texture Support**: Base color, metallic-roughness, normal, occlusion, and emissive maps
- **Material Validation**: Comprehensive validation with error reporting and statistics
- **CPU/GPU Integration**: Seamless integration between Python API and GPU shaders

## Quick Start

```python
import forge3d.pbr as pbr

# Create basic PBR material
material = pbr.PbrMaterial(
    base_color=(0.8, 0.2, 0.2, 1.0),  # Red plastic-like
    metallic=0.0,                      # Non-metallic (dielectric)
    roughness=0.7                      # Fairly rough surface
)

# Validate material properties
validation = pbr.validate_pbr_material(material)
print(f"Valid: {validation['valid']}")
print(f"Material type: {'metallic' if validation['statistics']['is_metallic'] else 'dielectric'}")

# Create material library
materials = pbr.create_test_materials()
gold_material = materials['metal_gold']  # Pre-configured gold material
```

## Core Concepts

### Material Properties

PBR materials in forge3d use the following core properties:

| Property | Range | Description |
|----------|-------|-------------|
| **base_color** | [0.0, 1.0] × 4 | Albedo color (RGB) + alpha channel |
| **metallic** | [0.0, 1.0] | Blend factor between dielectric (0) and metallic (1) |
| **roughness** | [0.04, 1.0] | Surface microsurface roughness (0 = mirror, 1 = rough) |
| **normal_scale** | [0.0, ∞) | Normal map intensity multiplier |
| **occlusion_strength** | [0.0, 1.0] | Ambient occlusion effect strength |
| **emissive** | [0.0, ∞) × 3 | Self-emitted light color (RGB) |
| **alpha_cutoff** | [0.0, 1.0] | Alpha testing threshold for transparency |

### Metallic vs Dielectric Materials

The PBR system distinguishes between two fundamental material types:

**Dielectric Materials** (metallic = 0.0):
- Plastics, ceramics, glass, wood, skin
- Reflect ~4% of light as specular reflection
- Base color determines diffuse reflection
- Fresnel reflectance varies with viewing angle

**Metallic Materials** (metallic = 1.0):
- Metals like gold, silver, iron, copper
- No diffuse reflection (all absorbed or reflected)
- Base color determines reflectance tint
- High reflectance (60-95% depending on metal)

**Hybrid Materials** (0.0 < metallic < 1.0):
- Oxidized metals, painted metals, layered materials
- Linear blend between dielectric and metallic behavior

### BRDF Implementation

The system implements the Cook-Torrance microfacet BRDF:

```
f(l,v) = D(h) × G(l,v,h) × F(v,h) / (4 × (n·l) × (n·v))
```

Where:
- **D(h)**: GGX/Trowbridge-Reitz normal distribution function
- **G(l,v,h)**: Smith geometry function with height-correlated masking
- **F(v,h)**: Fresnel-Schlick approximation
- **l**: Light direction, **v**: View direction, **h**: Half-vector, **n**: Surface normal

## API Reference

### PbrMaterial Class

```python
class PbrMaterial:
    def __init__(self, 
                 base_color=(1.0, 1.0, 1.0, 1.0),
                 metallic=0.0,
                 roughness=1.0,
                 normal_scale=1.0,
                 occlusion_strength=1.0,
                 emissive=(0.0, 0.0, 0.0),
                 alpha_cutoff=0.5):
        """Create PBR material with specified properties."""
    
    def set_base_color_texture(self, texture):
        """Assign base color texture (RGB or RGBA)."""
    
    def set_metallic_roughness_texture(self, texture):
        """Assign metallic-roughness texture (RG format: R=metallic, G=roughness)."""
    
    def set_normal_texture(self, texture):
        """Assign normal map texture (RGB tangent-space normals)."""
    
    def set_occlusion_texture(self, texture):
        """Assign ambient occlusion texture (single channel)."""
    
    def set_emissive_texture(self, texture):
        """Assign emissive texture (RGB)."""
```

### Material Validation

```python
def validate_pbr_material(material):
    """
    Validate PBR material properties and compute statistics.
    
    Returns:
        dict: {
            'valid': bool,           # Overall validity
            'errors': [str],         # Error messages
            'warnings': [str],       # Warning messages
            'statistics': {          # Material statistics
                'is_metallic': bool,     # metallic >= 0.5
                'is_dielectric': bool,   # metallic < 0.5
                'is_rough': bool,        # roughness >= 0.5
                'is_smooth': bool,       # roughness < 0.5
                'is_emissive': bool,     # any emissive > 0
                'has_textures': bool,    # texture_flags != 0
                'luminance': float       # Perceived brightness
            }
        }
    """
```

### BRDF Evaluation

```python
class PbrRenderer:
    def __init__(self):
        """Create PBR renderer for BRDF evaluation."""
    
    def set_lighting(self, lighting):
        """Configure lighting parameters."""
    
    def evaluate_brdf(self, material, light_dir, view_dir, normal):
        """
        Evaluate BRDF for given material and vectors.
        
        Args:
            material: PbrMaterial instance
            light_dir: Light direction vector (3D, normalized)
            view_dir: View direction vector (3D, normalized) 
            normal: Surface normal vector (3D, normalized)
        
        Returns:
            np.array: RGB reflectance values [0.0, ∞)
        """

class PbrLighting:
    def __init__(self, 
                 light_direction=(0.0, -1.0, 0.0),
                 light_color=(1.0, 1.0, 1.0),
                 light_intensity=1.0,
                 camera_position=(0.0, 0.0, 0.0)):
        """Configure lighting environment for PBR evaluation."""
```

### Material and Texture Libraries

```python
def create_test_materials():
    """
    Create library of pre-configured test materials.
    
    Returns:
        dict: Material name -> PbrMaterial mappings including:
            - 'plastic_red', 'plastic_green', 'plastic_blue': Colored plastics
            - 'metal_gold', 'metal_silver', 'metal_iron': Common metals
            - 'glass_clear': Transparent dielectric
            - 'emissive_warm': Self-illuminated material
    """

def create_test_textures():
    """
    Create library of procedural test textures.
    
    Returns:
        dict: Texture name -> numpy array mappings including:
            - 'checker_base_color': Checkered base color pattern
            - 'metallic_roughness': Varying metallic/roughness values
            - 'normal': Normal map with surface details
            - 'noise_occlusion': Procedural occlusion texture
    """
```

## Common Usage Patterns

### Creating Material Variants

```python
# Base material
base_material = pbr.PbrMaterial(
    base_color=(0.7, 0.4, 0.2, 1.0),  # Copper color
    roughness=0.3
)

# Dielectric variant (painted copper)
dielectric_copper = pbr.PbrMaterial(
    base_color=base_material.base_color,
    metallic=0.0,  # Non-metallic
    roughness=base_material.roughness
)

# Metallic variant (raw copper)
metallic_copper = pbr.PbrMaterial(
    base_color=base_material.base_color,
    metallic=1.0,  # Fully metallic
    roughness=base_material.roughness
)
```

### Texture Assignment Workflow

```python
# Create base material
material = pbr.PbrMaterial(
    base_color=(1.0, 1.0, 1.0, 1.0),  # White base (will be modulated by texture)
    metallic=0.0,
    roughness=0.5
)

# Load or create textures
textures = pbr.create_test_textures()

# Assign textures
material.set_base_color_texture(textures['checker_base_color'])
material.set_metallic_roughness_texture(textures['metallic_roughness'])
material.set_normal_texture(textures['normal'])

# Verify texture assignment
print(f"Texture flags: {bin(material.texture_flags)}")
print(f"Has base color: {bool(material.texture_flags & 1)}")
print(f"Has metallic-roughness: {bool(material.texture_flags & 2)}")
print(f"Has normal map: {bool(material.texture_flags & 4)}")
```

### BRDF Comparison Studies

```python
# Compare different materials under same lighting
materials = {
    'plastic': pbr.PbrMaterial(base_color=(0.8, 0.2, 0.2, 1.0), metallic=0.0, roughness=0.8),
    'metal': pbr.PbrMaterial(base_color=(0.8, 0.6, 0.4, 1.0), metallic=1.0, roughness=0.2),
    'hybrid': pbr.PbrMaterial(base_color=(0.6, 0.4, 0.3, 1.0), metallic=0.5, roughness=0.4)
}

renderer = pbr.PbrRenderer()
lighting = pbr.PbrLighting(
    light_direction=(0.0, -1.0, 0.3),
    light_intensity=2.0
)
renderer.set_lighting(lighting)

# Standard test vectors
light_dir = np.array([0.0, -1.0, 0.3])
light_dir = light_dir / np.linalg.norm(light_dir)
view_dir = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])

# Evaluate BRDF for each material
results = {}
for name, material in materials.items():
    brdf = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
    luminance = 0.299 * brdf[0] + 0.587 * brdf[1] + 0.114 * brdf[2]
    results[name] = {'brdf': brdf, 'luminance': luminance}
    print(f"{name:8s}: BRDF={brdf}, luminance={luminance:.3f}")
```

## Best Practices

### Material Authoring

1. **Roughness Guidelines**:
   - Mirror surfaces: 0.04-0.1
   - Polished metals: 0.1-0.3
   - Worn metals: 0.3-0.7
   - Rough surfaces: 0.7-1.0

2. **Base Color Guidelines**:
   - **Dielectrics**: Use measured albedo values (usually 0.04-0.9)
   - **Metals**: Use measured reflectance (F0 values)
   - **Avoid extremes**: Pure black (0,0,0) or pure white (1,1,1) rarely occur in nature

3. **Metallic Guidelines**:
   - Use binary values (0.0 or 1.0) for most materials
   - Use intermediate values only for layered/oxidized surfaces
   - Painted metals should be metallic=0.0 (paint is dielectric)

### Performance Considerations

1. **Texture Sizes**: Use power-of-2 dimensions for optimal GPU performance
2. **Texture Formats**: 
   - Base color: sRGB for color textures
   - Metallic/Roughness: Linear space
   - Normal maps: Tangent space, normalized
3. **Material Validation**: Validate materials during development to catch authoring errors early

### Integration with Other Systems

The PBR system integrates with other forge3d components:

- **Normal Mapping**: Combine with tangent-space normal maps for surface detail
- **Environment Mapping**: Use with IBL for realistic environmental lighting
- **HDR Rendering**: PBR materials work best with HDR lighting and tone mapping
- **Shadow Mapping**: Combine with CSM shadows for realistic light attenuation

## Troubleshooting

### Common Issues

**1. Materials appear too dark/bright**
- Check lighting configuration (intensity, direction)
- Verify base color values are in proper range
- Ensure HDR rendering and tone mapping are configured

**2. Metallic materials look wrong**
- Verify metallic=1.0 for pure metals
- Check base color represents F0 reflectance, not surface color
- Ensure no diffuse component for pure metals

**3. Rough surfaces appear smooth**
- Check roughness values are > 0.04
- Verify normal maps are assigned and scaled properly
- Ensure sufficient geometric tessellation for detail

**4. Texture assignment fails**
- Verify texture dimensions are powers of 2
- Check texture format compatibility
- Ensure textures are C-contiguous numpy arrays

### Validation Errors

The material validation system reports common authoring errors:

- **Color out of range**: Base color components outside [0,1] range
- **Metallic out of range**: Metallic values outside [0,1] range  
- **Roughness too low**: Roughness below minimum threshold (0.04)
- **Invalid alpha**: Alpha channel outside [0,1] range
- **Extreme values**: Unusual parameter combinations that may cause artifacts

### Performance Issues

- **Large textures**: Reduce texture resolution if GPU memory is limited
- **Complex materials**: Simplify material complexity if experiencing frame rate drops
- **Batch similar materials**: Group objects with similar materials for better GPU utilization

## Examples

See `examples/pbr_materials.py` for a comprehensive demonstration of PBR materials functionality, including:

- Material gallery creation and validation
- BRDF evaluation and comparison
- Texture assignment workflows
- Material sphere rendering
- Metallic vs dielectric comparisons

The example generates visual output showing different material types and their light interaction properties.

## Implementation Notes

### GPU Shader Integration

PBR materials are processed by GPU shaders in `src/shaders/pbr.wgsl`:

- **Vertex stage**: Transforms vertices and prepares interpolated values
- **Fragment stage**: Evaluates BRDF and combines with lighting
- **Texture sampling**: Handles all PBR texture types with proper filtering
- **Normal mapping**: Tangent-space to world-space normal transformation

### Memory Layout

Materials are stored in GPU-compatible format:

```rust
#[repr(C)]
pub struct PbrMaterial {
    pub base_color: [f32; 4],      // 16 bytes
    pub metallic: f32,             // 4 bytes  
    pub roughness: f32,            // 4 bytes
    pub normal_scale: f32,         // 4 bytes
    pub occlusion_strength: f32,   // 4 bytes
    pub emissive: [f32; 3],        // 12 bytes
    pub alpha_cutoff: f32,         // 4 bytes
    pub texture_flags: u32,        // 4 bytes
    pub _padding: [f32; 3],        // 12 bytes (64-byte alignment)
}
```

### Feature Flags

PBR functionality is controlled by Rust feature flags:

- Built-in by default in forge3d
- CPU-side evaluation always available
- GPU rendering requires compatible adapter
- Texture support requires image processing capabilities

This system provides a solid foundation for physically-based material authoring and rendering in forge3d applications.