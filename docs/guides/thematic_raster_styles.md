# Population and thematic rasters

`RasterHeightSurfaceStyle`, `CategoricalRasterStyle` and `BivariateRasterStyle`
are exported from `forge3d` and `forge3d.style`. They use the existing native
GIS normalization/classification API. Their `apply()` results include a true-valid
mask, display-sRGB RGBA8 cells and legend data. Population results also contain
float32 heights; thematic results contain class indices (0 means nodata).

```python
import numpy as np
import forge3d as f3d

population = np.array([[0, 10, 20], [20, 10, 0]], dtype=np.float32)
height_style = f3d.RasterHeightSurfaceStyle(
    height_scale=2, source_units="people/km²", height_units="m", nodata=-9999,
)
terrain = f3d.TerrainSource(
    data=population, style=height_style, crs="EPSG:3857",
    metadata={"source_id": "local-population"},
)
scene = f3d.MapScene(
    terrain=terrain, lighting=f3d.LightingPreset(name="daylight"),
    output=f3d.OutputSpec(width=192, height=192),
)
scene.render("population.png")
```

Height is **source value × height_scale**. The scale is in declared height units
per source unit; use metres for metric terrain coordinates. Values must be
nonnegative outside nodata, and the output must fit finite float32. Units are
labels carried into parameters and legends, never inferred or converted. Shade
interpolates the low/high colors with `gis.normalize_raster` minmax values; a
constant raster uses the low color. NaN/nonfinite source cells, declared nodata,
and false `valid_mask` cells are excluded. An all-invalid input diagnoses
`empty_raster`. Invalid scales/domains diagnose `invalid_argument`.

For land cover, supply a `CategoricalRasterStyle(palette={class_id: rgba},
labels={class_id: label}, nodata=...)` to `RasterOverlay(style=..., data=...)`.
Class IDs must be integers and every valid source ID must appear in the palette.
Zero is a valid source ID unless explicitly declared nodata. RGBA colors are
four integer bytes; class legend rows retain original IDs in ascending order.
Native classifier output indices are numbered from 1; 0 is reserved for nodata.

For the starting bivariate slice, supply **two increasing finite boundaries per
axis** and a **3×3 palette** to `BivariateRasterStyle(x_bins, y_bins, palette,
x_label, y_label, ...)`. Use `RasterOverlay(data=x, secondary_data=y, style=...)`.
`palette[y_bin][x_bin]` has low-to-high y rows and x columns. `right=False`
preserves classifier intervals `[-∞, b0)`, `[b0, b1)`, `[b1, ∞)`;
`right=True` gives `(-∞, b0]`, `(b0, b1]`, `(b1, ∞)`.
The two masks intersect. Results expose both axis indices, combined indices
1–9, palette cells and labelled axis legends with units and boundary closure.

MapScene accepts inline arrays or local `.npy`/GeoTIFF sources (`path` and
`secondary_path`). GeoTIFF nodata masks are retained. Styled sources must share
the terrain grid and CRS; pre-align them using the existing GIS/alignment APIs.
The native terrain pass has one thematic color source. Multiple styled overlays
diagnose explicitly. A thematic overlay can color a population height surface;
it then supplies the color palette while the population style supplies height.
Ordinary unstyled raster overlays keep their existing compositor and 3D block.

Nearest-cell native sampling preserves class colors without palette blending.
Cells with zero alpha, including nodata, discard terrain fragments. Other alpha values and
overlay opacity blend the palette into terrain albedo before native lighting.
Lighting and tone mapping can change displayed RGB; `apply().rgba` is the exact
palette reference. Legends are structured data, ready for a consumer; this slice
does not compose a legend into a map plate.

Style parameters and inline source cells survive `MapScene.save_bundle()` /
`load_bundle()` without a schema-version change. Nonfinite source cells encode
as null holes and negative zero normalizes to zero. Invalid nonfinite style
parameters are rejected. Existing unstyled recipe serialization is unchanged.
Styled inline terrain with shape `(1, H, W)` normalizes to `(H, W)` at construction,
so the data summary and stored values use the same single-band grid.
Local file paths remain references; keep those sources available for replay.

Run the complete public example on a physical terrain-capable GPU:

```console
python examples/mapscene_thematic_rasters.py --output artifacts/thematic
```

It renders all three styles, saves/reloads each bundle, compares PNG bytes and
writes render metadata and hashes. The scope excludes spikes, contours, data
fetching, alignment redesign and plate composition. No cross-backend or hosted
acceptance is established by a single local adapter run.
