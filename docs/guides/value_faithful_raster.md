# Value-faithful raster relief

`ValueFaithfulRasterBinding` links one prepared count field and an exact grid
window to a native statistical surface, fixed colour classes, coverage text
and a native north-up inset of the same cells. The example uses a square
8 km Amsterdam window in the original CBS RD New grid (EPSG:28992). The
window is a map extent and has no municipal boundary.

## Run the Amsterdam example

Use a fresh shell in the worktree with the matching Forge3D Python source and
native extension, the existing MapPlate Pro entitlement, real rasterio,
pyogrio, pyproj, Pillow and ffmpeg. Pass the corrected v2 archive explicitly.

```powershell
python examples/c02_amsterdam_value_faithful.py D:/forge3d/data/cbs_vk100/2026-cbs_vk100_2024_v2.zip D:/forge3d/artifacts/c02-value-faithful/publication
```

Use `--plate-only` for the plate and four revised plates. The full run writes a
1600 × 1200 plate, a five-second 24 fps lossless RGB MP4, three matching PNG
stills, the native frame sequence, per-page JSON and `evidence.json`. The
example retains the source archive and member hashes, the affine and field
tags, and the prepared raster hashes. Its SQL check independently reads each
cell and denominator from the source GeoPackage and compares every inset cell
centre against the displayed PNG. The example uses the [CBS grid
specification](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/2-technische-gegevens)
and [publication rules](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers).

## Bind another prepared field

Prepare a single-band GeoTIFF with a north-up square-cell grid in a projected
metre CRS. Include `measurement_field` and `measurement_units` dataset tags.
The former must match the field in `PublicationSource.provenance`. Set the
file nodata tag to a negative integer and store the distinct negative
withheld code as a regular sample. Use nonnegative integer samples for known
counts, including known zeros. Set `observation_year` to the year in the
source declaration when it is available. Supply an exact square window of
whole cells with at least two rows and columns. The renderer uses a square
512 × 512 vertex mesh in this mode. The binding checks that the mesh vertices
around each source cell centre sample its own height. Grids too fine for this
registration raise before rendering. With the current mesh, sizes 2 through
94 pass, as do 96, 97, 100, 102, 103, 128 and 170 cells per side. These sizes
follow from sample alignment. The output layout must also fit the inset.

The Amsterdam window contains no published zeros. The zero key is part of
the shared domain and the regression fixture exercises a known zero.

```python
from forge3d import OrbitCamera, ValueFaithfulRasterBinding

# source and extent are PublicationSource and PublicationExtent objects.
binding = ValueFaithfulRasterBinding(
    source, extent,
    measurement_type="count", height_scale=1.1,
    colour_edges=(0, 50, 150, 300, 600, 2500),
    palette=palette, missing_code=-9999, withheld_code=-99997,
)
binding.render_series(
    {"plate": OrbitCamera(target=(0, 0, 310), distance=9000,
                          azimuth_deg=-90, elevation_deg=24, fov_deg=76)},
    "publication",
)
```

The palette has one opaque RGBA key for zero, one for each positive interval
between successive edges, one for withheld and one for missing. The positive
intervals start above zero, include internal lower edges and include the final
upper edge. Values outside the declared domain fail before rendering. A change
in camera or crop leaves the domain and palette fixed. The camera target uses
east and north offsets from the extent centre and statistical metres above the
zero baseline. With the native Z-up camera, zero degrees looks straight down.
The camera must frame the full extent and declared height range inside its
viewport and clip planes. `near` and `far` can override the native defaults
for a statistical surface. Invalid or empty views raise before writing a page.

`count` displays the published count per cell. `density` divides that count
by the full projected cell area in square kilometres, including any land or
water within the cell. Declare the domain and height factor in the displayed
unit. The summary always adds original known counts and reports contributors,
unknown cells and represented area. Counts can be rounded by the publisher.
Neither the summary nor density extrapolates unknown cells.

`RasterHeightSurfaceStyle` supplies value × factor heights in statistical
metres. Native mesh interpolation connects adjacent samples and smooths cell
edges. Nearest-cell colours refer to the original cells. Use the inset to
read their class ranges and distinguish withheld from missing where the
perspective occludes them. A withheld CBS code is ambiguous between 0–4,
confidential and absent. A grid position without a published record can also
be suppressed. Both are holes in the perspective. Statistical relief has no
geographic elevation datum.

The height key gives statistical heights at fixed values from the legend
domain. The north arrow shows projected grid north at the centre of the
relief's zero plane and rotates with the camera. The inset keeps grid north up.

Use `dataclasses.replace` to revise the source field, extent, or height
factor. Use `to_dict()` and `from_dict()` to serialize the declaration. It
references the prepared GeoTIFF. The page receipts record the exact file hash
actually consumed. The example records the authoring interventions for each
revision.
