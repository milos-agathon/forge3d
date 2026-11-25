# Measured terrain transects

A measured section follows a declared WGS 84 geodesic across a terrain raster.
Its CSV and JSON retain each requested position, cumulative ground distance,
original signed elevation, raster row and column, and validity. The linked
still places the same sample IDs on the map and profile.

```python
from forge3d.measured_sections import Transect, render_section, sample_transect

raster = "surface.tif"
transect = Transect(start=(-121.74, 36.38), end=(-122.37, 36.65), spacing_m=375)
section = sample_transect(raster, transect, source_id="NOAA ETOPO 2022 v1",
                         vertical_crs="EPSG:3855")
render_section(section, raster, "monterey-original", title="From ridge to canyon")
```

Use a self-contained, single-band, north-up GeoTIFF on WGS 84. Compound
GeoTIFFs are accepted when their horizontal CRS is WGS 84 and the vertical
CRS agrees with the explicit declaration. A horizontal-only GeoTIFF uses
the caller's documented vertical CRS. Keep the GeoTIFF unchanged during sampling and rendering. Elevations must be stored directly
in metres, positive up. Supply a recognized EPSG vertical height CRS.
Projected or other horizontal CRSs, antimeridian crossings, rotated grids,
ellipsoidal heights, and datum or unit conversions are outside this route.
Sampling uses the existing `rasterio` and `pyproj` raster dependencies. The
still uses Pillow, the native renderer, and the C06 `MapPlate` Pro feature.

Sample spacing is a maximum. The interval count is the ceiling of geodesic
length divided by maximum spacing. Intervals are equal, with both endpoints
included. The caller sets spacing according to the source grid and purpose.
An interval wider than a cell can miss narrow terrain features or gaps
between samples. Sampling uses the containing cell without interpolation
across cells. Under the binary64 GDAL pixel-edge affine, west and north
edges belong to a cell. East and south edges belong to the next cell or fall
outside the raster. The original coordinates stay in the table. Masked,
nodata and nonfinite pixels have an empty CSV elevation and JSON `null`.
Out-of-bounds rows also have empty row and column fields. Negative and zero
heights are valid when the source mask and nodata declaration allow them.

The output directory must be new. `map.png` is the base raster frame,
`section.png` is the full linked plate, and `layout.json` records panel
geometry, linked sample IDs, vertical exaggeration, renderer and file
hashes. `samples.json` adds CRS, units, pixel geometry, source SHA-256 and
sampling conventions to the same rows as `samples.csv`. The map trace is the
requested path. Only runs of valid samples are connected in the profile,
where gray spans mark missing samples. The distance and elevation axes
keep their physical units. Exaggeration is the ratio of vertical to
horizontal display pixels per metre. The map uses fixed elevation bands
through the C02 categorical style. The land/water context is the grid's
zero-height color boundary and follows the model's vertical reference.

The optional `cache=` is the existing validated MapScene cache for the base
raster frame. The trace, profile and sample table are regenerated. The
optional `certificate=` applies to the native map render under MapScene's
existing signing contract. The composed still records its file hash.

## Reproduce the Monterey comparison

From the isolated checkout, build and install the wheel into a clean
environment using the checkout's Python and native sources together. The
local validation used CPython 3.13.14 and maturin 1.16.0. Set `$r` to an
output root where `subset`, `original`, `revised` and the QGIS result
directories do not yet exist.

```powershell
$r = 'artifacts/c08-measured-sections'
$env:CARGO_TARGET_DIR = Join-Path $PWD "$r/native-target"
uv venv "$r/venv" --python 3.13
$py = Join-Path $PWD "$r/venv/Scripts/python.exe"
maturin build --release --locked --interpreter $py --out "$r/wheels"
uv pip install --python $py (Get-ChildItem "$r/wheels/*.whl").FullName rasterio==1.5.0 pyproj==3.7.2 pillow==12.1.1 pytest cryptography
& $py examples/measured_sections/monterey.py prepare $r
```

Use the CLI in a licensed environment. Run each version in a fresh process.

```powershell
& $py examples/measured_sections/monterey.py render "$r/subset" examples/measured_sections/original.json "$r/original"
& $py examples/measured_sections/monterey.py render "$r/subset" examples/measured_sections/revised.json "$r/revised"
```

Local test reproduction uses the repository's existing Pro fixture. It runs
the same CLI body and keeps the usual license check active.

```powershell
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:FORGE3D_TEST_INSTALLED_WHEEL = '1'
$env:WGPU_BACKEND = 'vulkan'
$env:WGPU_BACKENDS = 'vulkan'
$env:FORGE3D_MEASURED_SECTIONS_DEMO = $r
foreach ($variant in 'original', 'revised') {
    $env:FORGE3D_MEASURED_SECTIONS_RECIPE = $variant
    & $py -m pytest examples/measured_sections/demo_pytest.py -q
}
```

Use QGIS 3.44's Python with PyQGIS, Qt and GDAL on the same subset and
recipe. The comparison runs headlessly and renders its own PNGs and CSV.

```powershell
$qgis = 'D:/forge3d/artifacts/c05-real-data/qgis-runtime/extracted/QGIS 3.44.0/bin/python-qgis.bat'
$env:QT_QPA_PLATFORM = 'offscreen'
foreach ($variant in 'original', 'revised') {
    & $qgis examples/measured_sections/qgis_compare.py "$r/subset" "examples/measured_sections/$variant.json" "$r/qgis-$variant"
}
& $py examples/measured_sections/verify.py $r
```

The evidence and commands for the local qualification are recorded in the
[Monterey acceptance record](measured_sections_evidence.md).
