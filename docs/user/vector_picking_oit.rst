Vector Picking & OIT
====================

The current top-level ``forge3d`` package exposes scene-level OIT controls, but
it does **not** currently re-export the low-level weighted-OIT vector demo
helpers that older docs referenced.

Public OIT Controls
-------------------

* ``Scene.enable_oit()``
* ``Scene.disable_oit()``
* ``Scene.is_oit_enabled()``
* ``Scene.get_oit_mode()``
* ``forge3d.viewer_ipc.set_oit_enabled(...)``

Experimental Native-Only Helpers
--------------------------------

If your build includes the relevant native feature gates, the compiled
extension may expose helpers such as ``vector_oit_and_pick_demo`` and
``vector_render_oit_and_pick_py``. Access them explicitly through the native
module:

.. code-block:: python

   from forge3d._native import get_native_module

   native = get_native_module()
   if native is not None and hasattr(native, "vector_oit_and_pick_demo"):
       rgba, pick_id = native.vector_oit_and_pick_demo(640, 360)
       print("center pick id:", pick_id)

Status
------

Treat these native-only vector OIT helpers as internal/experimental. For the
maintained public workflow, use scene OIT controls, viewer IPC, and the export
or vector helper modules as appropriate.
