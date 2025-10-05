Interactive Viewer
==================

.. note::
   **Workstream I1** - Interactive Viewer implementation for windowed exploration with orbit and FPS camera modes.

The forge3d interactive viewer provides real-time windowed rendering with interactive camera controls.

Overview
--------

The interactive viewer is a native Rust application that provides:

- **Windowed rendering** using winit and wgpu
- **Orbit camera mode** for rotating around a target point
- **FPS camera mode** for free movement
- **60+ FPS performance** on simple scenes
- **DPI scaling support** for high-resolution displays
- **Real-time FPS counter** in window title

Architecture
------------

The viewer module (``src/viewer/mod.rs``) consists of:

- **ViewerConfig**: Configuration for window size, title, VSync, FOV, and clipping planes
- **Viewer**: Main viewer struct managing the window, surface, device, and camera
- **CameraController**: Handles orbit and FPS camera modes with smooth controls
- **FpsCounter**: Tracks and reports frame rate

Camera Modes
------------

Orbit Camera
~~~~~~~~~~~~

The orbit camera rotates around a target point:

- **Mouse drag** - Rotate camera around target
- **Mouse scroll** - Zoom in/out
- **Shift + Mouse drag** - Pan the target point

FPS Camera
~~~~~~~~~~

The FPS camera provides free movement:

- **W/A/S/D** - Move forward/left/backward/right
- **Q/E** - Move down/up
- **Mouse** - Look around (hold left button)
- **Shift** - Move faster (2x speed multiplier)

Keyboard Controls
-----------------

- **Tab** - Toggle between Orbit and FPS camera modes
- **Esc** - Exit the viewer

Usage (Python)
--------------

The interactive viewer can be opened directly from Python using the ``open_viewer()`` function:

.. code-block:: python

   import forge3d as f3d

   # Basic usage with defaults
   f3d.open_viewer()

   # Custom configuration
   f3d.open_viewer(
       width=1280,
       height=720,
       title="My Scene Viewer",
       vsync=True,
       fov_deg=60.0,
       znear=0.1,
       zfar=1000.0
   )

   # High performance mode (VSync off)
   f3d.open_viewer(
       width=1920,
       height=1080,
       title="High FPS Viewer",
       vsync=False
   )

Parameters
~~~~~~~~~~

- **width** (int): Window width in pixels (default: 1024)
- **height** (int): Window height in pixels (default: 768)
- **title** (str): Window title (default: "forge3d Interactive Viewer")
- **vsync** (bool): Enable VSync for smoother rendering (default: True)
- **fov_deg** (float): Field of view in degrees (default: 45.0)
- **znear** (float): Near clipping plane distance (default: 0.1)
- **zfar** (float): Far clipping plane distance (default: 1000.0)

Usage (Rust)
------------

.. code-block:: rust

   use forge3d::viewer::{run_viewer, ViewerConfig};

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       let config = ViewerConfig {
           width: 1280,
           height: 720,
           title: "My Viewer".to_string(),
           vsync: true,
           fov_deg: 60.0,
           znear: 0.1,
           zfar: 1000.0,
       };

       run_viewer(config)
   }

Running the Examples
--------------------

Python Example
~~~~~~~~~~~~~~

To run the Python interactive viewer demo:

.. code-block:: bash

   # Basic mode (default settings)
   python examples/interactive_viewer_demo.py

   # Custom resolution
   python examples/interactive_viewer_demo.py --mode custom --width 1920 --height 1080

   # High FPS mode (VSync disabled)
   python examples/interactive_viewer_demo.py --mode no-vsync

Rust Example
~~~~~~~~~~~~

To run the Rust interactive viewer example:

.. code-block:: bash

   cargo run --example interactive_viewer

Performance
-----------

The viewer is designed to run at 60 FPS or higher on simple scenes. Performance characteristics:

- **VSync enabled** - Locks to display refresh rate (typically 60 Hz)
- **VSync disabled** - Runs at maximum GPU frame rate
- **FPS counter** - Displayed in window title, updated every second
- **DPI aware** - Handles high-resolution displays correctly

Acceptance Criteria (I1)
-------------------------

✓ Windowed exploration with winit
✓ Orbit camera controls
✓ FPS camera controls
✓ DPI scaling support
✓ 60 FPS on simple scenes

Integration Notes
-----------------

The interactive viewer is available in both Python and Rust:

**Python**:
- Use ``f3d.open_viewer()`` for direct windowed interaction
- Blocking call - runs until window is closed
- Full camera control support

**Offscreen Rendering** (alternative for headless environments):
- Use ``render_offscreen_rgba`` or ``save_png_with_exif`` for headless rendering
- Integrate with **Python GUI frameworks** (tkinter, PyQt, pygame)
- Display results using **matplotlib** or **PIL/Pillow**

See ``examples/screenshot_demo.py`` for offscreen rendering examples.

Technical Details
-----------------

Dependencies
~~~~~~~~~~~~

- **winit 0.29** - Cross-platform window creation and event handling
- **wgpu 0.19** - GPU abstraction layer
- **pollster** - Block on async operations in the event loop

Surface Configuration
~~~~~~~~~~~~~~~~~~~~~

- **Surface format** - Prefers sRGB color space
- **Present mode** - AutoVsync or AutoNoVsync based on config
- **Alpha mode** - Uses first supported alpha mode from capabilities

Limitations
~~~~~~~~~~~

- Currently renders a blue-gray clear color (scene rendering TODO)
- Requires a GPU backend (Vulkan, Metal, DX12, or OpenGL)
- Blocking call - runs on main thread (required by winit on some platforms)

Future Enhancements
-------------------

Planned improvements for the interactive viewer:

- Scene rendering integration (terrain, meshes, vector graphics)
- Screenshot capture (S key)
- Video recording (R key)
- UI overlay for camera info
- Multiple viewport support
- Grid/axes helpers
- Lighting controls
- Scene loading from files

See Also
--------

- :doc:`examples_guide` - Other forge3d examples
- :doc:`installation` - Building forge3d
- Camera module: ``src/camera.rs``
- Camera controller: ``src/viewer/camera_controller.rs``
