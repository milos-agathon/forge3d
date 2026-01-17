Point Cloud Viewer
==================

The forge3d point cloud viewer provides interactive visualization of LAZ/LAS point cloud data with orbit camera controls and multiple color modes.

Overview
--------

The point cloud viewer supports:

* **LAZ/LAS file loading** with automatic decompression
* **Interactive orbit camera** with mouse and keyboard controls
* **Multiple color modes**: Elevation, RGB, and Intensity
* **Smart fallback rendering** when data attributes are missing
* **IPC-based runtime control** for dynamic parameter updates
* **Snapshot capability** for high-resolution image export

Quick Start
-----------

Basic usage with the interactive viewer:

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py \
       --input assets/lidar/MtStHelens.laz \
       --color-mode elevation

This launches an interactive window with the point cloud loaded and ready for exploration.

Color Modes
-----------

The viewer supports three color visualization modes:

Elevation Mode
~~~~~~~~~~~~~~

Colors points based on their elevation using a terrain colormap (blue → green → brown → white).
This is the default mode and always works regardless of data availability.

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py \
       --input mydata.laz \
       --color-mode elevation

RGB Mode
~~~~~~~~

Uses the RGB color values stored in the point cloud file. When RGB data is not available,
the viewer falls back to **red-tinted elevation** to provide visual distinction and indicate
missing data.

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py \
       --input mydata.laz \
       --color-mode rgb

Intensity Mode
~~~~~~~~~~~~~~

Displays points in grayscale based on their intensity values (normalized to 0-1 range).
When intensity data is not available or has zero range, the viewer falls back to
**green-tinted elevation**.

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py \
       --input mydata.laz \
       --color-mode intensity

Interactive Controls
--------------------

Mouse Controls
~~~~~~~~~~~~~~

* **Drag**: Orbit camera around the point cloud
* **Scroll wheel**: Zoom in/out

Keyboard Controls
~~~~~~~~~~~~~~~~~

* **W / ↑**: Tilt camera up
* **S / ↓**: Tilt camera down
* **A / ←**: Rotate camera left
* **D / →**: Rotate camera right
* **Q**: Zoom out
* **E**: Zoom in

Terminal Commands
~~~~~~~~~~~~~~~~~

While the viewer is running, you can enter commands in the terminal:

.. code-block:: text

   # Set camera position
   set phi=45 theta=30 radius=2.0

   # Adjust point size
   set point_size=3.0

   # Change color mode
   set color_mode=elevation
   set color_mode=rgb
   set color_mode=intensity

   # Show current parameters
   params

   # Reload point cloud from file
   reload

   # Clear point cloud
   clear

   # Exit viewer
   quit

Taking Snapshots
----------------

Capture high-resolution images from the viewer:

.. code-block:: text

   # From terminal while viewer is running
   snap output.png 1920x1080

Or specify output path and size via command:

.. code-block:: bash

   # In viewer terminal
   > snap st-helens.png 3840x2160

The snapshot will be saved to the specified path with the requested dimensions.

Command-Line Options
--------------------

.. code-block:: text

   usage: pointcloud_viewer_interactive.py [-h] --input INPUT
                                           [--color-mode {elevation,rgb,intensity}]
                                           [--port PORT]
                                           [--width WIDTH]
                                           [--height HEIGHT]

   optional arguments:
     -h, --help            show this help message and exit
     --input INPUT         Path to LAZ/LAS point cloud file
     --color-mode {elevation,rgb,intensity}
                           Color visualization mode (default: elevation)
     --port PORT           IPC port for viewer communication (default: 9876)
     --width WIDTH         Window width in pixels (default: 1200)
     --height HEIGHT       Window height in pixels (default: 800)

Technical Details
-----------------

File Format Support
~~~~~~~~~~~~~~~~~~~

The viewer uses the ``las`` crate for native LAZ/LAS decompression:

* **LAZ** (compressed LAS): Automatic decompression
* **LAS**: Direct reading
* **Point formats**: All standard LAS point formats
* **Coordinate systems**: Automatic conversion from LAS (X=easting, Y=northing, Z=elevation) to 3D viewer space (X=easting, Y=elevation, Z=northing)

Data Processing
~~~~~~~~~~~~~~~

On load, the viewer performs:

1. **Bounds calculation**: Computes min/max for X, Y, Z
2. **Centering**: Translates all points so the centroid is at origin (0,0,0)
3. **Intensity normalization**: Maps intensity values from their raw range to 0-1
4. **RGB extraction**: Reads 16-bit RGB values when available, normalizes to 0-1
5. **Subsampling**: Optional stride-based sampling for large files (configurable via ``max_points`` parameter)

The centering ensures the orbit camera can properly frame the point cloud regardless of its absolute coordinates.

Rendering Pipeline
~~~~~~~~~~~~~~~~~~

The viewer uses GPU-based point sprite rendering:

* **Instance buffer**: Stores position, elevation, RGB, and intensity for each point
* **WGSL shader**: Vertex shader expands each point to a quad, fragment shader applies circular soft edges
* **Uniforms**: Camera view-projection matrix, point size, color mode flags, data availability flags
* **Color selection**: Performed in vertex shader based on mode and data availability

Smart Fallback System
~~~~~~~~~~~~~~~~~~~~~~

When color data is missing from the LAZ/LAS file:

* **RGB mode**: Falls back to ``elevation_color * vec3(1.0, 0.5, 0.5)`` (red tint)
* **Intensity mode**: Falls back to ``elevation_color * vec3(0.5, 1.0, 0.5)`` (green tint)

This tinted approach ensures:

1. Visual distinction from standard elevation mode
2. Geometry remains visible and useful
3. Clear indication that requested data is unavailable

Performance Considerations
--------------------------

Point Count Limits
~~~~~~~~~~~~~~~~~~

The default viewer loads up to 5 million points. For larger datasets, configure subsampling:

.. code-block:: python

   # Via IPC command
   {"cmd": "load_point_cloud", "path": "/path/to/data.laz", "max_points": 1000000}

Memory Usage
~~~~~~~~~~~~

Each point requires approximately 48 bytes in the instance buffer:

* Position: 12 bytes (3 × f32)
* Elevation norm: 4 bytes (f32)
* RGB: 12 bytes (3 × f32)
* Intensity: 4 bytes (f32)
* Size: 4 bytes (f32)
* Padding: 12 bytes (alignment)

A 1 million point cloud uses ~48 MB GPU memory for the instance buffer.

Known Limitations
-----------------

Current implementation limitations:

* **No streaming LOD**: All points loaded into memory at once
* **No spatial indexing**: Cannot pick/query individual points
* **No classification filtering**: Displays all point classes
* **Single file only**: Cannot load multiple point clouds simultaneously
* **No Gaussian splats**: Traditional point sprite rendering only

Future enhancements planned:

* EPT/COPC octree streaming for large datasets
* Point picking and attribute queries
* Classification-based filtering and coloring
* Gaussian splat rendering mode
* Multi-cloud support with separate transforms

Troubleshooting
---------------

White or tinted terrain when expecting RGB/Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: The LAZ/LAS file does not contain RGB or intensity data.

**Solution**: Verify your data format:

.. code-block:: bash

   # Check if file has RGB/intensity using lasinfo
   lasinfo mydata.laz

If the file lacks this data, the viewer will show tinted elevation (red for RGB mode, green for Intensity mode) to indicate the fallback behavior.

All modes look identical
~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: File may have RGB/intensity data but with zero range (all same value).

**Solution**: The viewer normalizes data to 0-1 range. If all intensity values are identical, normalization produces 0.5 grayscale. Check your data processing pipeline.

Points not visible
~~~~~~~~~~~~~~~~~~

**Cause**: Camera may be inside the point cloud or too far away.

**Solution**: Reset camera position:

.. code-block:: text

   # In viewer terminal
   set phi=45 theta=30 radius=2.0

Or reload the point cloud to reset camera to default framing.

Examples
--------

Mount St. Helens Lidar
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py \
       --input assets/lidar/MtStHelens.laz \
       --color-mode elevation \
       --width 1920 --height 1080

Custom color mode and snapshot workflow:

.. code-block:: bash

   python examples/pointcloud_viewer_interactive.py --input mydata.laz
   # In viewer terminal:
   > set color_mode=intensity
   > set point_size=2.0
   > set phi=60 theta=45 radius=1.5
   > snap output.png 2560x1440
   > quit

See Also
--------

* :doc:`rendering_options` - General rendering configuration
* :doc:`troubleshooting_visuals` - Visual debugging techniques
* :doc:`presets_overview` - Preset system for reproducible renders
