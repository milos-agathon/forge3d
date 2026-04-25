# Khumbu Sentinel Timelapse Design

Date: 2026-04-25

## Goal

Create a forge3d example that renders a polished Sentinel-2 time-series animation of the Khumbu Icefall near Mount Everest. The output should echo the CESBIO "Khumbu icefall in 4D" idea while using forge3d rendering, live public Sentinel data, local caching, and a no-login public DEM at the crispest practical resolution.

Reference source: https://www.cesbio.cnrs.fr/multitemp/khumbu-icefall-in-4d/

## User-Approved Decisions

- Data access: live download plus local cache, so the example is reproducible.
- Imagery: real Sentinel-2 L2A time series, not a single image with synthetic time variation.
- DEM: no-login public DEM only, using the crispest available source. Do not silently fall back to a softer DEM.
- Visual direction: "Forge3d Polished", not a strict rayshader clone.

## Example Surface

Add `examples/khumbu_icefall_sentinel_timelapse.py`.

Default outputs:

- `examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4`
- `examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_preview.png`
- `examples/out/khumbu_icefall_sentinel_timelapse/frames/frame_0000.png`, etc.

Default cache:

- `examples/.cache/khumbu_icefall_sentinel_timelapse/`

Core CLI:

- `--start-date`, default `2018-11-01`
- `--end-date`, default `2020-12-31`
- `--max-scenes`, default `24`
- `--cloud-cover`, default `35`
- `--fps`, default `10`
- `--duration`, optional; if omitted, frame count follows selected scenes
- `--size`, default `1600 1000`
- `--preview-only`
- `--frames-only`
- `--force`

## Location And AOI

Use a fixed WGS84 AOI centered on the Khumbu Icefall:

- Center: `27.98806, 86.87972`
- Default bbox: `(86.835, 27.965, 86.905, 28.015)` as `(west, south, east, north)`
- Target projected CRS: `EPSG:32645` (UTM zone 45N)

The bbox covers the icefall, lower Western Cwm, Everest Base Camp side, and surrounding high relief needed for a compelling 3D view. Keep the constants explicit in the script so users can audit and override them later if needed.

## Data Pipeline

### Sentinel-2

Use Microsoft Planetary Computer STAC:

- Endpoint: `https://planetarycomputer.microsoft.com/api/stac/v1/search`
- Collection: `sentinel-2-l2a`
- Bands/assets: prefer `visual` for true-color rendering. If needed, fall back to `B04`, `B03`, `B02`.
- Search filter: AOI bbox, date range, `eo:cloud_cover <= --cloud-cover`.
- Sort candidates chronologically.
- Select up to `--max-scenes`, with light de-duplication by acquisition date so adjacent MGRS tile duplicates do not double-count one overpass.
- Sign asset URLs through the Planetary Computer SAS signing API before reading COG assets.

The script should download or window-read only the AOI, not entire Sentinel tiles. Cached outputs should be small clipped/reprojected GeoTIFF or PNG artifacts keyed by scene id, date, bbox, and target grid.

### DEM

Use Copernicus DEM GLO-30 Public COGs from AWS Open Data:

- Registry: https://registry.opendata.aws/copernicus-dem/
- Prefer 30 m public tiles only.
- Construct the 1-degree tile URL(s) intersecting the AOI using the Copernicus naming pattern, for example `Copernicus_DSM_COG_10_N27_00_E086_00_DEM` and `Copernicus_DSM_COG_10_N28_00_E086_00_DEM` for the default bbox.
- Read the AOI through rasterio windowed IO.
- Reproject/resample to the render grid with cubic/bilinear interpolation.
- If a required 30 m public tile is unavailable, stop with a clear error explaining that the no-login crisp DEM requirement cannot be met for the AOI.

Do not use NSIDC HMA 8 m because it may require Earthdata/NSIDC access and violates the no-login requirement.

## Rendering Pipeline

1. Build a terrain GeoTIFF from the DEM crop in `EPSG:32645`, preserving metric shape.
2. For each selected Sentinel scene, crop/reproject the RGB data to the DEM/render grid.
3. Normalize RGB robustly:
   - Use `visual` asset values as display RGB when available.
   - For band fallback, percentile stretch each channel and apply gentle gamma/saturation.
   - Preserve snow/ice highlights without clipping large areas to flat white.
4. Save each scene overlay as RGBA PNG.
5. Open forge3d terrain through `open_viewer_async()`.
6. Load each overlay, set camera/sun, snapshot a frame, then post-process:
   - Add date label.
   - Apply subtle contrast/clarity pass if needed.
   - Keep composition cinematic and readable.
7. Encode MP4 with `ffmpeg` when available; otherwise leave frames and preview.

The camera should orbit gently across the time series, similar in spirit to the article's rotating view, but tuned for forge3d:

- High oblique terrain view.
- Strong relief and shadowing.
- Clean background.
- Stable target near the icefall center.
- Enough camera motion to reveal the 3D terrain, not so much that time-series comparison becomes hard.

## Error Handling

- Missing optional Python dependencies: exit with a precise install hint.
- No Sentinel scenes found: report bbox, date range, and cloud threshold.
- Planetary Computer signing/read failure: report scene id and asset key.
- DEM tile unavailable: report tile URL and no-login DEM constraint.
- Missing `ffmpeg`: keep frame sequence and preview, print that MP4 encoding was skipped.
- Missing native forge3d viewer: exit with the same style as existing viewer examples.

## Tests

Add `tests/test_khumbu_icefall_sentinel_timelapse.py`.

Focus on pure-Python behavior:

- CLI defaults match the approved scope.
- STAC payload contains collection, bbox, date range, and cloud-cover filter.
- Candidate scene selection sorts chronologically and de-duplicates same-date tile duplicates.
- Copernicus DEM tile names/URLs are built correctly for the AOI.
- RGB normalization returns `uint8` RGBA with alpha filled for valid pixels.
- Frame names are deterministic.
- `--preview-only` computes one frame without requiring MP4 encoding.

Do not require GPU rendering or live network in unit tests. Live data and viewer rendering remain manual/integration behavior.

## Documentation

Update `docs/examples/index.md` with a row under "Animation And Camera Automation" or "Interactive Terrain And Cartography", whichever fits the final implementation best.

Mention data sources in the script docstring and final console output:

- Sentinel-2 L2A via Microsoft Planetary Computer.
- Copernicus DEM GLO-30 Public via AWS Open Data.
- Concept reference: CESBIO Khumbu Icefall in 4D.

## Out Of Scope

- Earthdata/NSIDC credential flow.
- Reproducing Venµs 5 m data.
- A strict pixel-for-pixel clone of the article video.
- Cloud masking beyond scene-level cloud cover and optional simple SCL filtering.
- New forge3d native APIs.
