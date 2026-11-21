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
The viewer retains the legacy mesh layout for unlabelled Cartesian Area
terrain. A GeoTIFF with a CRS and transform, or an explicit Point key,
selects its sample lattice.

For a corner affine A, sample (column, row) is at
A(column + 0.5, row + 0.5). In an image of size N along an axis, sample
i has texture UV (i + 0.5) / N. Geometry whose extent is the outer sample
positions uses node UV i / (N - 1). The conversion from node UV to texture
UV is (node_uv * (N - 1) + 0.5) / N. A singleton axis always samples its
sole texel. Preserve the sample locations when converting Area to Point by
adding half of each pixel axis to the native file origin. Subtract half of
each axis in the reverse direction. The normalized affine stays unchanged
for that conversion. Set `registration_conversion="preserve_sample_locations"`
on a preparation source, an alignment call, or MapScene drape metadata to
declare it. A warp to a different grid evaluates the destination samples
under the destination affine.

| Stage | Before #84 | Registration and coordinates |
| --- | --- | --- |
| Native GIS TIFF reader and window | Used raw tiepoint or matrix as a corner affine | Reads the GeoTIFF raster type and shifts Point by −0.5 of both pixel axes before taking bounds or cutting a window |
| Streamed COG georeferencing | Used raw GeoTIFF anchors for pixel coordinates | Carries the raster type through the IFD reader and normalizes coordinates in both directions |
| GDAL and Rasterio DEM loader | Corner-normalized affine with the raster type dropped | Carries Area or Point with the affine and applies no second shift |
| Preparation and alignment | Warped normalized corners and marked preparation output Area | Checks registration before combining rasters and records an explicitly chosen conversion. Warp samples use both normalized affines |
| Viewer terrain mesh and shadow | Spread vertices across the corner footprint while indexing heights with texel units | Georeferenced mesh vertices land at source samples. Raster overlays retain edge-normalized extent and draped vectors evaluate height on the same sample lattice |
| Viewer raster overlays | Loaded TIFFs as ordinary images and assigned a normalized extent | A registered GeoTIFF must match the terrain's CRS, transform, dimensions and registration. Prepare any different lattice before loading. Ordinary images use their explicitly normalized extent, defaulting to full terrain coverage |
| MapScene warp and raster drape | Baked samples on the height grid, then sampled the drape over outer texel edges | Bakes on the registered grid and converts geometry node UV to texel-centre UV in the material shader. The registered fullscreen path reads both height and drape north-up |
| MapScene thematic overlays and RasterRecipe | Read thematic textures across their outer texel edges. RasterRecipe accepted only Area | Thematic GeoTIFFs are aligned before classification. Resident recipes require exact grid agreement for operands and terrain. The shader uses the same node-to-texel rule for classes, drapes and water reflections |
| MapScene vectors and labels | Camera-projected world anchors already used outer samples. Prepared screen vectors used outer edges | World and prepared screen anchors use the outer samples as the terrain extent |
| Subject story | Validated the stated bounds against the outer DEM sample centres | Continues to use outer sample centres from the normalized affine for its mesh and feature anchors |
| Hybrid traced terrain and albedo | Node-centred heights and albedo with caller-supplied spacing | The registered wrapper derives spacing and the local origin from its affine. Drape and height arrays must share the same lattice. World mesh vertices use the same origin |
| Physical-water texture | Heights and normals used node-UV bilinear loads while coverage used edge-normalized sampling | Registered coverage, height and normals use the same bilinear loads on the height-field nodes |
| Landscape export | First pixel centre became vertex 0,0 | Carries source registration and identifies the exported vertices as Point, with the first vertex's world coordinates |
| Data GeoTIFF and orthographic render output | TIFF writer emitted Area. Rendered images carried pixels alone | Data GeoTIFFs emit their actual registration. Viewer and MapScene map the outer terrain samples to viewport edges. Tracing uses the requested sensor centre and ground pixel size. All three write Area render pixels using the resulting output affine |

The MapScene material change uses the existing overlay uniform, with
the registered raster flag in its formerly reserved `params7.z` lane.
The drape texture and sampler remain at group 6, bindings 17 and 15.
The viewer shadow change uses the existing 128-byte depth uniform,
with the registered grid flag at `grid_params.y`. The height input
remains a read-only R32Float texture at group 0, binding 1.


To trace a registered north-up map, pass the DEM samples and their
`PreparationGrid` to `forge3d.path_tracing.render_registered_terrain`.
Supply a ground pixel size in metres. A drape must share the terrain's
grid. Use `forge3d.export.render_to_geotiff` to write the resulting
beauty image or one of its raster layers.

For a registered north-up MapScene, set `OutputSpec(format="geotiff", …)`.
Use `metadata={"coordinate_space": "world"}` for vector or label layers
authored in the terrain CRS. The viewer provides
`ViewerHandle.snapshot_geotiff(path, width, height)` for a georeferenced
terrain. Both output grids cover the interval between the first and last
source samples, using the complete cell on a singleton axis.
