# Humanity Globe forge3d Video Design

Date: 2026-06-08

## Goal

Create a forge3d example that recreates the reference video at `C:\Users\milos\Downloads\T9yLmHMjuuvFJuKB.mp4`: a 720x720 rotating globe showing GPW-v4 2020 world population density as stepped, turbo-colored shells over a grey sphere.

The output should match the reference structure closely:

- 720x720 MP4.
- 25 fps.
- 28.8 seconds.
- 720 frames.
- Black background.
- Grey Earth sphere.
- Full 360-degree orbit.
- Title, stepped legend, source credit, and creation/reference credit.

The implementation target is an offline forge3d example, not a new native globe runtime.

## Reference Sources

- Local reference video: `C:\Users\milos\Downloads\T9yLmHMjuuvFJuKB.mp4`
- Original published recipe: https://gist.github.com/tylermorganwall/3ee1c6e2a5dff19aca7836c05cbbf9ac
- Original data family: GPW-v4 population density, revision 11, 2020.
- GPW-v4 source collection: https://sedac.ciesin.columbia.edu/data/collection/gpw-v4
- Public metadata/data mirror used during design research: https://pacific-data.sprep.org/resource/gridded-population-world-version-4-gpwv4-population-density-2020

The original recipe uses `gpw_v4_population_density_rev11_2020_15_min.tif` and thresholds the raster into seven positive population-density layers plus a zero/background layer.

## User-Approved Decisions

- Build a faithful forge3d example script rather than adding native textured-globe support.
- Use the same GPW-v4 population density source family as the reference.
- Match the visible output contract first: size, duration, frame count, globe style, orbit, stepped legend, title, and credits.
- Keep the design honest about current forge3d capabilities: forge3d has terrain, scene, colormap, and image utilities, but no finished native globe renderer. The globe rasterization can live in the example while using forge3d helpers where the current API supports them.

## Example Surface

Add `examples/humanity_globe_video.py`.

Default outputs:

- `examples/out/humanity_globe/humanity_globe_forge3d.mp4`
- `examples/out/humanity_globe/humanity_globe_preview.png`
- `examples/out/humanity_globe/frames/frame_0000.png`, etc. when frames are kept.

Default cache:

- `examples/.cache/humanity_globe/`

Core CLI:

- `--gpw-tif PATH`: use an explicit GPW-v4 2020 population-density GeoTIFF.
- `--output PATH`: default MP4 path above.
- `--preview PATH`: default preview path above.
- `--cache-dir PATH`: default cache path above.
- `--size INT`: default `720`.
- `--fps INT`: default `25`.
- `--duration FLOAT`: default `28.8`.
- `--frames INT`: optional explicit override; default derives to `720`.
- `--preview-only`: render one representative frame and skip MP4.
- `--frames-only`: render all frames but skip MP4 encoding.
- `--keep-frames`: retain frame PNGs after MP4 encoding.
- `--force`: regenerate cached data and frames.

## Data Strategy

The script should prefer an exact 15-minute GPW-v4 population-density raster when available:

1. `--gpw-tif`.
2. `data/gpw_v4_population_density_rev11_2020_15_min.tif`.
3. Cached 15-minute raster in `examples/.cache/humanity_globe/`.

If the 15-minute raster is absent, the script derives it from a public GPW-v4 2020 30-arc-second population-density raster by aggregating to 15-minute cells. This preserves the same data source family and produces the resolution used by the reference. The derivation must be explicit in console output and cache metadata. If the public 30-arc-second raster cannot be downloaded or read, the script stops with the same clear `--gpw-tif` guidance.

Do not silently use unrelated population sources, country subsets, synthetic data, or the existing Poland/Turkey WorldPop rasters. If neither exact nor derivable GPW-v4 data is available, stop with a clear error explaining how to provide `--gpw-tif`.

### Raster Shape And Orientation

Expected 15-minute grid:

- Longitude span: -180 to 180.
- Latitude span: -90 to 90.
- Dimensions: 1440 columns by 720 rows.
- Equirectangular cell centers.
- Top row is northern latitude in normal GeoTIFF orientation.

The in-memory matrix used for rendering should match the original visual orientation: North America visible in the first frame and a west-to-east globe orbit over 720 frames.

## Classification And Color

Classify density into the same stepped layers as the original recipe:

- `0`: grey/base sphere.
- `>1`
- `>5`
- `>10`
- `>50`
- `>100`
- `>500`
- `>1000`

Use the seven-color turbo palette for the positive classes. Prefer Matplotlib `turbo` when available because the local environment has Matplotlib. Include a deterministic fallback table with seven sRGB colors so tests and rendering do not depend on optional provider behavior.

The legend labels should match the reference:

- `People per 30km^2`
- `0`, `1>`, `5>`, `10>`, `50>`, `100>`, `500>`, `1000>`

Keep the label text exactly as the reference uses it, even though the underlying grid is technically 15 arc-minute, roughly 30 km at the equator rather than a constant 30 square kilometer area.

## Rendering Pipeline

Use a deterministic offline rasterization pipeline inside the example:

1. Load or derive the 15-minute GPW-v4 density matrix.
2. Build boolean masks for each threshold layer.
3. For each output frame:
   - Compute orbit longitude from frame index.
   - Analytically project an orthographic sphere into the 720x720 frame.
   - Map visible sphere pixels to latitude/longitude.
   - Sample the density class texture with nearest-neighbor lookup so stepped pixels remain crisp.
   - Shade the base sphere grey with soft Lambert/specular lighting similar to the reference.
   - Draw positive population classes as slightly raised layers or point/shell pixels over the base sphere.
   - Preserve edge visibility so population on the limb remains visible like the reference.
   - Composite title, legend, data credit, and forge3d/reference credit.
   - Save frame PNG using `forge3d.numpy_to_png`.
4. Encode frames with `ffmpeg` to H.264 `yuv420p`, `+faststart`, default CRF around 18.
5. Save frame 0 as the preview.

The renderer should be vectorized with NumPy. Per-frame loops over all pixels are acceptable at 720x720 only if the full render remains practical, but pixel math should be array-based where straightforward.

## forge3d Usage Boundary

The example should import forge3d through the standard `_import_shim` path and use current public forge3d utilities for:

- package/example integration,
- PNG writing through `forge3d.numpy_to_png`,
- optional colormap/color helpers when they match the approved palette path,
- consistent example output conventions.

The script should not pretend that `Scene`, `TerrainRenderer`, or the viewer can render a textured globe today. If future native globe support is added, this example can become an integration target for it, but that is out of scope for this task.

## Layout And Credits

Default composition:

- Title near top-left: `The Humanity Globe: World Population Density, 30km^2 Grid`
- Legend along lower-left/middle, matching reference proportions.
- Bottom-left credit:
  - `Data: 2020 GPW-v4`
  - `Created with forge3d`
- Bottom-right credit:
  - `Reference: @tylermorganwall`

This keeps the reference author visible while making the generated artifact honest about the forge3d recreation.

Use Windows Segoe UI fonts when available, matching existing examples. Fall back to PIL default fonts only when system fonts are missing.

## Error Handling

- Missing `numpy`, `PIL`, or `rasterio`: fail with a precise install hint.
- Missing Matplotlib: use the fallback turbo table, not a hard failure.
- Missing GPW data: explain expected file name and `--gpw-tif`.
- Failed public data download or aggregation: report the URL/path and leave any partial file as `.download` or remove it.
- Missing `ffmpeg`: keep frames and preview, print that MP4 encoding was skipped.
- Invalid raster dimensions or CRS: report expected global equirectangular density raster assumptions.
- Empty or all-nodata raster: stop before frame rendering.

## Tests

Add `tests/test_humanity_globe_video.py`.

Focus on pure-Python behavior without live network, GPU, or full MP4 render:

- CLI defaults produce 720x720, 25 fps, 28.8 seconds, and 720 frames.
- Density classification maps exact thresholds correctly.
- 15-minute grid dimension validation accepts 720x1440 and rejects incompatible shapes.
- Longitude/latitude sampling is deterministic for known sphere pixels.
- Orbit longitude changes smoothly and completes one 360-degree rotation over the frame count.
- Turbo fallback palette has seven positive class colors.
- Legend labels match the reference text.
- Frame filename generation is deterministic.
- `ffmpeg` command uses H.264, `yuv420p`, and the requested fps.
- Preview render can run on a tiny synthetic global raster without network or forge3d GPU access.

Manual verification:

- `python examples/humanity_globe_video.py --preview-only`
- `python examples/humanity_globe_video.py --frames 25 --frames-only --size 360`
- Full default render when data and time budget are available.
- `ffprobe` confirms 720x720, 25 fps, and 28.8 seconds for the MP4.

## Documentation

Update `docs/examples/index.md` with the new example under animation/camera or data visualization examples after implementation.

The script docstring and console output should cite:

- GPW-v4 population density, revision 11, 2020.
- Tyler Morgan-Wall's original Humanity Globe gist as the concept/reference.
- forge3d as the recreation workflow.

## Out Of Scope

- Adding a native Rust/WebGPU globe renderer.
- Cesium-grade or browser globe streaming.
- Exact path-traced parity with rayshader/rayrender.
- Reproducing every anti-aliasing/noise characteristic of the downloaded MP4.
- Using unrelated population datasets.
- Changing existing Poland, Turkey, or terrain examples.
- Making claims that forge3d already supports general live globe rendering.
