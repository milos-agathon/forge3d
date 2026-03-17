# Converters

There is no public `forge3d.converters` Python module in the current package.

## What exists today

- `forge3d.geometry.extrude_polygon()` for prism-style extrusion of planar polygons
- `forge3d.io.import_osm_buildings_extrude()` for merged building extrusion from footprint features
- `forge3d.buildings.add_buildings()`, `add_buildings_cityjson()`, and `add_buildings_3dtiles()` for the higher-level buildings workflow

## Native-only helper

The compiled extension also exposes `converters_multipolygonz_to_obj_py` on the native module, but it is not wrapped as a stable public Python API. If you need that exact conversion path today, use `forge3d._native.get_native_module()` explicitly and treat it as internal.

## Recommended path

For public Python code, prefer the wrapped geometry and IO modules above instead of relying on undocumented converter entry points.
