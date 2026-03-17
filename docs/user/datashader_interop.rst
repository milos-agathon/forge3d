Datashader Integration
======================

There is no current public ``forge3d.adapters`` module for Datashader
conversion.

## Practical interop today

Use Datashader to produce an RGBA image, then feed that image into forge3d
through standard Python image/array workflows:

* save the Datashader image and load it as a terrain overlay in the viewer
* display it with Matplotlib next to forge3d output
* package it into a downstream cartography/export workflow

The current integration point is image exchange, not a dedicated zero-copy
adapter API inside the forge3d package.
