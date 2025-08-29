Camera Uniforms & viewWorldPosition Policy
==========================================

This document describes the camera uniform system in forge3d, including the viewWorldPosition field and recommended policies for near/far/FOV parameters.

Overview
--------

The camera system provides both perspective and orthographic projections with consistent uniform buffer layout. Camera uniforms are automatically computed and passed to shaders for lighting calculations.

Uniform Buffer Layout
---------------------

The camera-related uniforms are stored in the ``Globals`` structure with the following layout:

.. code-block:: rust

   struct Globals {
       view: Mat4,                    // View matrix (world → camera)
       proj: Mat4,                    // Projection matrix (camera → NDC)  
       sun_exposure: Vec4,            // (sun_dir.xyz, exposure)
       spacing: Vec4,                 // (spacing, h_range, exaggeration, 0)
       light_counts: Vec4,            // (num_point_lights, num_spot_lights, 0, 0)
       view_world_position: Vec4,     // (camera_world_pos.xyz, 0) - NEW
       // ... lighting data arrays follow
   }

viewWorldPosition Field
-----------------------

The ``view_world_position`` field contains the camera's world-space position, which is essential for:

- Specular lighting calculations
- Distance-based effects  
- View-dependent shading

**Computation**: The world position is extracted by inverting the view matrix and taking the translation component:

.. code-block:: rust

   let inv_view = view_matrix.inverse();
   let camera_world_pos = inv_view.w_axis.xyz();

**WGSL Access**: In shaders, access the camera position via:

.. code-block:: wgsl

   let camera_pos = globals.view_world_position.xyz;

Near/Far/FOV Policy & Recommendations
--------------------------------------

Default Values
~~~~~~~~~~~~~~

The following defaults provide good balance between visual quality and numerical precision:

============== =============== ==========================================
Parameter      Default Value   Rationale
============== =============== ==========================================
FOV            45.0°           Good balance, not too wide or narrow
Near Plane     0.1             Close enough for detail, avoids z-fighting
Far Plane      100.0           Sufficient range, preserves depth precision  
Aspect Ratio   16:9            Modern widescreen standard
============== =============== ==========================================

FOV Guidelines
~~~~~~~~~~~~~~

- **30-60°**: Recommended range for most applications
- **45°**: Standard default, matches human vision comfortably
- **90°**: Maximum before noticeable fisheye distortion
- **> 120°**: Avoid unless specifically needed (causes severe distortion)

Near/Far Ratio Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

The near/far ratio affects depth buffer precision. Follow these guidelines:

- **Ratio < 1000:1**: Excellent precision
- **Ratio 1000-10000:1**: Good precision (recommended maximum)
- **Ratio > 10000:1**: Poor precision, avoid if possible

**Examples**:

.. code-block:: python

   # Good ratios
   proj = camera_perspective(45.0, 16/9, 0.1, 100.0)   # 1000:1 ratio
   proj = camera_perspective(45.0, 16/9, 1.0, 1000.0)  # 1000:1 ratio
   
   # Acceptable ratios  
   proj = camera_perspective(45.0, 16/9, 0.01, 100.0)  # 10000:1 ratio
   
   # Avoid
   proj = camera_perspective(45.0, 16/9, 0.001, 100.0) # 100000:1 ratio (bad precision)

Orthographic Projection
~~~~~~~~~~~~~~~~~~~~~~~

For 2D rendering and pixel-perfect alignment:

.. code-block:: python

   # Pixel-aligned 2D rendering
   ortho = camera_orthographic(0, width, 0, height, 0.1, 10.0)
   
   # Centered coordinate system
   ortho = camera_orthographic(-width/2, width/2, -height/2, height/2, 0.1, 10.0)

Coordinate System Conventions
-----------------------------

forge3d uses **right-handed, Y-up** coordinates:

- **+X**: Right
- **+Y**: Up  
- **+Z**: Toward viewer (out of screen)
- **View direction**: -Z

This matches OpenGL conventions and is converted automatically for WGPU/Vulkan backends.

Clip Space
----------

The system supports both GL and WGPU clip spaces:

- **GL clip space**: Z ∈ [-1, 1]
- **WGPU clip space**: Z ∈ [0, 1] (default)

The conversion is handled automatically via the ``clip_space`` parameter.

Usage Examples
--------------

Basic Camera Setup
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import forge3d as f3d
   
   # Create view matrix
   view = f3d.camera_look_at(
       eye=(0, 0, 5),      # Camera position
       target=(0, 0, 0),   # Look at origin  
       up=(0, 1, 0)        # Y-up
   )
   
   # Create projection matrix
   proj = f3d.camera_perspective(
       fovy_deg=45.0,      # Field of view
       aspect=16.0/9.0,    # Aspect ratio
       znear=0.1,          # Near plane
       zfar=100.0          # Far plane
   )
   
   # Or combined
   view_proj = f3d.camera_view_proj(
       eye=(0, 0, 5), target=(0, 0, 0), up=(0, 1, 0),
       fovy_deg=45.0, aspect=16.0/9.0, znear=0.1, zfar=100.0
   )

Orthographic for UI
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pixel-perfect UI rendering
   ui_proj = f3d.camera_orthographic(
       left=0, right=1920,     # Screen width
       bottom=0, top=1080,     # Screen height  
       znear=0.1, zfar=10.0    # Minimal depth range
   )

Camera Animation
~~~~~~~~~~~~~~~~

.. code-block:: python

   import math
   
   # Orbit camera around origin
   t = time  # animation parameter
   radius = 10.0
   eye = (radius * math.cos(t), 5.0, radius * math.sin(t))
   
   view = f3d.camera_look_at(eye, (0, 0, 0), (0, 1, 0))

Testing & Validation
--------------------

The camera system includes comprehensive tests:

- **Matrix properties**: Invertibility, determinant checks
- **Coordinate consistency**: View-projection pipeline correctness  
- **Uniform integration**: Proper data flow to shaders
- **Edge cases**: Extreme parameters, degenerate inputs

Run the test suite:

.. code-block:: bash

   pytest tests/test_d6_camera_uniforms.py -v

Version History
---------------

- **v1.0**: Initial camera math implementation (D1)
- **v1.1**: Sun direction and tonemap integration (D2)  
- **v1.2**: Lighting system integration (D3)
- **v1.3**: Orthographic projection support (D5)
- **v1.4**: viewWorldPosition uniform field (D6) - **Current**

See Also
--------

- :doc:`../api/camera` - Camera API reference
- :doc:`../shaders/uniforms` - Shader uniform conventions
- :doc:`../rendering/coordinate_systems` - Coordinate system details