# docs/ — Memory for examples, notebooks, and video helper

The goal is **fast reproducibility** with **zero external data fetches**.

## Examples (MVP)
1) **Terrain hillshade with overlays** — generate a synthetic DEM (e.g., Gaussian hills), add synthetic polygons/lines/points; render PNG and metrics JSON.
2) **City basemap** — synthetic blocks/roads/POIs; demonstrate polygon fills + strokes, line caps/joins, and point sprites.
3) **Graph snapshot** — synthetic positions (grid or fixed layout) for 50k nodes and ~100k edges; demonstrate batching and metrics.

All examples must:
- Be **deterministic** (fixed seeds).
- Save `env.json` from `report_environment()` and `metrics.json` from `render_metrics()` alongside outputs.
- Print adapter/backend to stdout.

## Notebooks
- Use the same synthetic data generators as scripts; avoid heavy deps.
- Execute in CI with `nbconvert --execute` (or run the equivalent `.py` scripts if nbconvert is too slow).
- Include small inline thumbnails; keep cell output minimal.

## Video helper
- `vulkan_forge.video.turntable(renderer, orbit, frames, fps, out_path)` renders frames then shells out to **ffmpeg**.
- Detect missing ffmpeg and raise an actionable error (don’t download ffmpeg).

## Troubleshooting
- If images differ across OSes, check color policy (linear→sRGB), row‑padding removal, and LOD usage in shaders.
- For CI flakiness: ensure `prefer_software` is set; add small sleeps if the driver needs settling time after device creation (rare).
