Uniform Buffer Layout
=====================

This document describes the terrain uniform buffer layout used by the WebGPU/WGSL terrain shader.

Overview
--------

The terrain shader uses a single uniform buffer (``@group(0) @binding(0)``) with a std140-compatible layout totaling **176 bytes**. This buffer contains view/projection matrices and terrain rendering parameters.

Layout Specification
---------------------

The uniform buffer contains exactly 44 floats (176 bytes) with the following layout:

.. code-block:: wgsl

   struct Globals {
     view: mat4x4<f32>,                    // 64 B
     proj: mat4x4<f32>,                    // 64 B
     sun_exposure: vec4<f32>,              // xyz = sun_dir (normalized), w = exposure (16 B)
     spacing_h_exag_pad: vec4<f32>,        // x=dx, y=dy, z=height_exaggeration, w=unused (16 B)
     _pad_tail: vec4<f32>,                 // tail padding to keep total size multiple-of-16 (16 B)
   };

Field Breakdown
---------------

=================== ======= ========== =====================================================
Field               Size    Offset     Description
=================== ======= ========== =====================================================
``view``            64 B    0-63       View matrix (camera transform)
``proj``            64 B    64-127     Projection matrix  
``sun_exposure``    16 B    128-143    Sun direction (xyz) + exposure (w)
``spacing_h_exag_pad`` 16 B  144-159    Terrain spacing (x,y) + height exaggeration (z) + padding (w=0)
``_pad_tail``       16 B    160-175    Tail padding (all zeros)
=================== ======= ========== =====================================================

**Total**: 176 bytes (44 × 4-byte floats)

Float Array Layout
------------------

When accessed as a 44-float array, the layout is:

* **[0..15]**: View matrix (16 floats, column-major)
* **[16..31]**: Projection matrix (16 floats, column-major)  
* **[32..35]**: Sun exposure: ``[sun_dir.x, sun_dir.y, sun_dir.z, exposure]``
* **[36..39]**: Spacing/height: ``[dx, dy, height_exaggeration, 0.0]``
* **[40..43]**: Tail padding: ``[0.0, 0.0, 0.0, 0.0]``

Implementation Notes
--------------------

Rust Structure
~~~~~~~~~~~~~~

The corresponding Rust structure is:

.. code-block:: rust

   #[repr(C, align(16))]
   pub struct TerrainUniforms {
       pub view: [[f32; 4]; 4],          // 64 B
       pub proj: [[f32; 4]; 4],          // 64 B
       pub sun_exposure: [f32; 4],       // 16 B
       pub spacing_h_exag_pad: [f32; 4], // 16 B  
       pub _pad_tail: [f32; 4],          // 16 B
   }

   // Compile-time assertions
   const _: () = assert!(::std::mem::size_of::<TerrainUniforms>() == 176);
   const _: () = assert!(::std::mem::align_of::<TerrainUniforms>() == 16);

Buffer Allocation
~~~~~~~~~~~~~~~~~

The uniform buffer is allocated using the struct size:

.. code-block:: rust

   let uniform_size = std::mem::size_of::<TerrainUniforms>() as u64; // 176 bytes
   let ubo = device.create_buffer_init(&BufferInitDescriptor {
       label: Some("terrain-ubo"),
       contents: bytemuck::cast_slice(&[uniforms]),
       usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
   });

Validation
----------

The layout is validated by:

1. **Compile-time assertions** in Rust ensuring 176-byte size and 16-byte alignment
2. **Runtime debug assertions** checking buffer size matches expectations  
3. **Unit tests** verifying the 44-float debug snapshot has correct values in expected slots
4. **Integration tests** ensuring WGPU validation passes during rendering

Historical Context
------------------

This 176-byte layout replaced a previous 656-byte layout that included complex lighting data. The simplified layout removes:

* Point light arrays
* Spot light arrays  
* Light count fields
* View world position
* Normal transformation matrix
* Additional padding

The reduction from 656 to 176 bytes resolves WGPU validation errors where "Buffer is bound with size X where the shader expects Y".