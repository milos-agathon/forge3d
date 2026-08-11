# HELIOS WhiteboxTools viewshed reference

`whitebox_switzerland_64.png` is the visibility band (`value > 0`) from
WhiteboxTools 2.4.0 `Viewshed`, generated offline from the committed
`assets/tif/switzerland_dem.tif`
(SHA-256 `d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478`).

- Source crop: EPSG:4326 bounds `(7.0, 46.4, 8.0, 47.2)`, bilinear-resampled
  to 64×64. The crop contains no nodata and spans about 77×89 km.
- Vertical datum: the source orthometric EGM96 heights are converted with
  `forge3d.dem_orthometric_to_ellipsoidal` before both implementations.
- Reference CRS: EPSG:2056 (CH1903+ / LV95). Rasterio 1.5.0
  `calculate_default_transform` and `reproject(..., bilinear)` produced a
  59×69 DEM with 1294.94 m pixels and origin
  `(2566265.3736383095, 1227824.77029089)`.
- Station: raster cell `(row=55, column=49)`, cell-centre WGS84
  `(lat=46.50625, lon=7.7734375)`, projected
  `(x=2625697.861772111, y=1150603.6002684815)`, 8000 m AGL.
- Whitebox output is a flat projected reference. HELIOS is compared using
  `Ellipsoid + EffectiveRadius(k=0.13)`; the separate low-observer ablation
  test proves that curvature/refraction remain load-bearing.

Install the offline-only generator dependency `pyshp==3.1.6`, then regenerate
the projected DEM, station shapefile, Whitebox raster, reprojection, and PNG
with:

```text
python tests/golden/viewshed/generate_reference.py --whitebox-tools "C:\tools\WhiteboxTools\whitebox_tools.exe"
```

The generator prints and executes WhiteboxTools with `--run=Viewshed`,
`--dem=dem.tif`, `--stations=station.shp`, `--output=whitebox.tif`,
`--height=8000.0`, and `--compress_rasters=False` inside its temporary
directory. The result is reprojected to the original 64×64 EPSG:4326 grid
with nearest resampling, then values greater than zero are mapped to opaque
white and all other values to opaque black. The source SHA-256 is locked by
the acceptance test.
