# Raster registration

Area samples describe cells. Point samples describe nodes. The GeoTIFF
file anchor is a cell corner for Area and the first sample for Point.
Forge3D carries the registration separately from a corner-normalized
affine, as specified by [GDAL RFC 33](https://gdal.org/en/stable/development/rfc/rfc33_gtiff_pixelispoint.html).
A missing GeoTIFF registration key uses its specified Area default.
Legacy in-memory preparation grids use Area by default. Declare
`registration="area"` or `registration="point"` on a preparation
target to select registered rendering for an in-memory scene. Existing
schema-1 recipes with an implicit Area target retain their screen placement.

For a corner affine A, sample (column, row) is at
A(column + 0.5, row + 0.5). In an image of size N along an axis, sample
i has texture UV (i + 0.5) / N. Geometry whose extent is the outer sample
positions uses node UV i / (N - 1). The conversion from node UV to texture
UV is (node_uv * (N - 1) + 0.5) / N. A singleton axis always samples its
sole texel. Preserve the sample locations when converting Area to Point by
adding half of each pixel axis to the native file origin. Subtract half of
each axis in the reverse direction. The normalized affine stays unchanged
for that conversion. A warp to a different grid evaluates the destination
samples under the destination affine.

| Stage | Before #84 | Registration and coordinates |
| --- | --- | --- |
| Native GIS TIFF reader and window | Used raw tiepoint or matrix as a corner affine | Reads the GeoTIFF raster type and shifts Point by −0.5 of both pixel axes before taking bounds or cutting a window |
| Streamed COG georeferencing | Used raw GeoTIFF anchors for pixel coordinates | Carries the raster type through the IFD reader and normalizes coordinates in both directions |
| GDAL and Rasterio DEM loader | Corner-normalized affine with the raster type dropped | Carries Area or Point with the affine and applies no second shift |
| Preparation and alignment | Warped normalized corners and marked preparation output Area | Checks registration before combining rasters and records an explicitly chosen conversion. Warp samples use both normalized affines |
| Viewer terrain mesh and shadow | Spread vertices across the corner footprint while indexing heights with texel units | Georeferenced mesh vertices land at source samples. Raster overlays retain edge-normalized extent and draped vectors evaluate height on the same sample lattice |
| MapScene warp and raster drape | Baked samples on the height grid, then sampled the drape over outer texel edges | Bakes on the registered grid and converts its geometry node UV to texel-centre UV in the material shader |
| MapScene vectors and labels | Camera-projected world anchors already used outer samples. Prepared screen vectors used outer edges | World and prepared screen anchors use the outer samples as the terrain extent |
| Hybrid traced terrain and albedo | Node-centred heights and albedo with caller-supplied spacing | The registered wrapper derives spacing and the local origin from its affine. Drape and height arrays must share the same lattice. World mesh vertices use the same origin |
| Physical-water texture | Explicit node-UV bilinear loads across first and last samples | Derived terrain maps continue to use the same node domain as the height field |
| Landscape export | First pixel centre became vertex 0,0 | Carries source registration and identifies the exported vertices as Point, with the first vertex's world coordinates |
| TIFF and orthographic trace output | TIFF writer emitted Area. Traced output carried pixels alone | Data TIFFs emit the actual registration. Orthographic traced pixels form an Area grid independently of the input registration |

The MapScene material change uses the existing overlay uniform, with
the registered drape flag in its formerly reserved `params7.z` lane.
The drape texture and sampler remain at group 6, bindings 17 and 15.
The viewer shadow change uses the existing 128-byte depth uniform,
with the registered grid flag at `grid_params.y`. The height input
remains a read-only R32Float texture at group 0, binding 1.


To trace a registered north-up map, pass the DEM samples and their
`PreparationGrid` to `forge3d.path_tracing.render_registered_terrain`.
Supply a ground pixel size in metres. A drape must share the terrain's
grid. Use `forge3d.export.render_to_geotiff` to write the resulting
beauty image or one of its raster layers.
