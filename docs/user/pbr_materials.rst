PBR Materials
=============

forge3d provides a comprehensive Physically-Based Rendering (PBR) materials system following the metallic-roughness workflow for realistic material rendering.

.. note::
   This functionality is available via the ``forge3d.pbr`` module. Import it explicitly for access to PBR features.

Quick Start
-----------

Basic PBR material creation:

.. code-block:: python

    import forge3d.pbr as pbr
    
    # Create a gold material
    gold = pbr.PbrMaterial(
        base_color=(1.0, 0.86, 0.57, 1.0),  # Gold color
        metallic=1.0,                        # Fully metallic
        roughness=0.1                        # Smooth surface
    )
    
    # Create a red plastic material  
    plastic = pbr.PbrMaterial(
        base_color=(0.8, 0.2, 0.2, 1.0),   # Red color
        metallic=0.0,                       # Non-metallic
        roughness=0.7                       # Somewhat rough
    )

API Reference
-------------

**PbrMaterial Class**

The main class for defining PBR materials:

.. code-block:: python

    material = pbr.PbrMaterial(
        base_color=(1.0, 1.0, 1.0, 1.0),    # RGBA base color [0,1]
        metallic=0.0,                        # Metallic factor [0,1]
        roughness=1.0,                       # Roughness factor [0,1]
        normal_scale=1.0,                    # Normal map intensity
        occlusion_strength=1.0,              # AO strength [0,1]
        emissive=(0.0, 0.0, 0.0),           # RGB emissive color
        alpha_cutoff=0.5                     # Alpha testing threshold
    )

**Material Properties**

- **base_color**: Surface albedo in linear space (RGBA)
- **metallic**: 0.0 = dielectric, 1.0 = metallic conductor
- **roughness**: 0.0 = mirror-smooth, 1.0 = completely rough (minimum 0.04 enforced)
- **normal_scale**: Multiplier for normal map intensity
- **occlusion_strength**: How much ambient occlusion affects the surface
- **emissive**: Self-illumination color (can exceed [0,1] for HDR)
- **alpha_cutoff**: Threshold for alpha testing transparency

Texture Support
---------------

PBR materials support various texture maps:

**Base Color Texture**

.. code-block:: python

    import numpy as np
    
    # Create or load base color texture (RGBA)
    base_texture = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
    material.set_base_color_texture(base_texture)

**Metallic-Roughness Texture**

.. code-block:: python

    # Metallic-roughness packed texture
    # Green channel = roughness, Blue channel = metallic
    mr_texture = np.zeros((256, 256, 4), dtype=np.uint8)
    mr_texture[:, :, 1] = 128  # 50% roughness
    mr_texture[:, :, 2] = 255  # 100% metallic
    mr_texture[:, :, 3] = 255  # Full alpha
    
    material.set_metallic_roughness_texture(mr_texture)

**Normal Map**

.. code-block:: python

    # Tangent-space normal map (RGB)
    normal_texture = np.zeros((256, 256, 3), dtype=np.uint8)
    # Encode normals from [-1,1] to [0,255]
    normal_texture[:, :, 0] = 127  # X = 0 (neutral)
    normal_texture[:, :, 1] = 127  # Y = 0 (neutral)  
    normal_texture[:, :, 2] = 255  # Z = 1 (pointing up)
    
    material.set_normal_texture(normal_texture)

**Additional Maps**

.. code-block:: python

    # Ambient occlusion
    ao_texture = np.random.randint(128, 256, (256, 256, 3), dtype=np.uint8)
    material.set_occlusion_texture(ao_texture)
    
    # Emissive map
    emissive_texture = np.zeros((256, 256, 3), dtype=np.uint8)
    material.set_emissive_texture(emissive_texture)

BRDF Evaluation
---------------

The PBR system includes CPU-side BRDF evaluation:

.. code-block:: python

    # Create renderer for BRDF evaluation
    renderer = pbr.PbrRenderer()
    renderer.add_material("gold", gold)
    
    # Set up lighting
    lighting = pbr.PbrLighting(
        light_direction=(0.0, -1.0, 0.3),      # Sun direction
        light_color=(1.0, 1.0, 1.0),           # White light
        light_intensity=3.0,                    # Brightness
        camera_position=(0.0, 0.0, 5.0)        # View position
    )
    renderer.set_lighting(lighting)
    
    # Evaluate BRDF at a surface point
    light_dir = np.array([0.0, -1.0, 0.3])
    view_dir = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 0.0, 1.0])
    
    color = renderer.evaluate_brdf(gold, light_dir, view_dir, normal)
    print(f"BRDF result: {color}")

Predefined Materials
--------------------

The system includes common material presets:

.. code-block:: python

    # Get predefined materials
    materials = pbr.create_test_materials()
    
    gold = materials['gold']
    silver = materials['silver'] 
    copper = materials['copper']
    plastic_red = materials['plastic_red']
    rubber_black = materials['rubber_black']
    wood = materials['wood']
    glass = materials['glass']
    emissive = materials['emissive']

Material Validation
-------------------

Validate material properties:

.. code-block:: python

    # Validate material
    validation = pbr.validate_pbr_material(material)
    
    if validation['valid']:
        print("✓ Material is valid")
    else:
        print("✗ Material has errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠ Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Material statistics
    stats = validation['statistics']
    print(f"Material type: {'Metallic' if stats['is_metallic'] else 'Dielectric'}")
    print(f"Surface: {'Rough' if stats['is_rough'] else 'Smooth'}")
    print(f"Textures: {stats['texture_count']}")

GPU Integration
---------------

Materials can be uploaded to GPU for rendering:

.. code-block:: python

    # Get material data for GPU upload
    material_data = material.get_material_data()
    print(f"Material data shape: {material_data.shape}")
    print(f"Material data: {material_data}")
    
    # Data is laid out for GPU uniform buffer:
    # - base_color (vec4)
    # - metallic, roughness, normal_scale, occlusion_strength (vec4)
    # - emissive (vec3) + alpha_cutoff (float)
    # - texture_flags + padding (vec4)

Lighting Configuration
----------------------

Set up realistic lighting for PBR materials:

.. code-block:: python

    lighting = pbr.PbrLighting(
        light_direction=(-0.5, -0.7, -0.5),    # Sun angle
        light_color=(1.0, 0.95, 0.8),          # Warm sunlight
        light_intensity=3.2,                    # Brightness
        camera_position=(0.0, 0.0, 5.0),       # Eye position
        ibl_intensity=1.0,                      # Environment contribution
        ibl_rotation=0.0,                       # Environment rotation
        exposure=1.0,                           # Exposure adjustment
        gamma=2.2                               # Gamma correction
    )
    
    # Get lighting data for GPU
    lighting_data = lighting.get_lighting_data()

Feature Detection
-----------------

Check PBR support availability:

.. code-block:: python

    if pbr.has_pbr_support():
        print("✓ PBR materials are supported")
        
        # Use PBR features
        material = pbr.PbrMaterial()
        renderer = pbr.PbrRenderer()
    else:
        print("✗ PBR materials not available")
        # Use fallback rendering

Best Practices
--------------

**Material Authoring**

1. **Physically Plausible Values**
   - Dielectric F0 should be around 0.04 (4% reflection)
   - Metallic materials use base_color as F0
   - Avoid pure black (0.0) or pure white (1.0) albedo

2. **Roughness Guidelines**
   - Minimum roughness of 0.04 to avoid singularities
   - Very smooth surfaces: 0.04-0.1
   - Typical surfaces: 0.2-0.8
   - Rough surfaces: 0.8-1.0

3. **Energy Conservation**
   - Brighter base colors should be less metallic
   - Rougher surfaces reflect less light at grazing angles
   - Emissive materials can exceed [0,1] range

**Performance Optimization**

- Reuse materials where possible to reduce GPU memory
- Use texture atlases to minimize texture switches
- Consider LOD for distant materials
- Validate materials during development to catch issues early

**Texture Authoring**

- Use linear color space for base color textures
- Normal maps should be in tangent space
- Pack metallic and roughness into single texture (green=roughness, blue=metallic)
- Ensure proper sRGB/linear handling

Legacy Compatibility
--------------------

The ``forge3d.materials`` module provides backward compatibility:

.. code-block:: python

    # Legacy import (still supported)
    import forge3d.materials as mat
    material = mat.PbrMaterial()  # Same as pbr.PbrMaterial()
    
    # Recommended import (clearer)
    import forge3d.pbr as pbr
    material = pbr.PbrMaterial()

Both approaches provide identical functionality, but ``forge3d.pbr`` is recommended for new code.