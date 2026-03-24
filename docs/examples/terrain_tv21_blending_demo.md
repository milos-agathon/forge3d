# `terrain_tv21_blending_demo.py`

`examples/terrain_tv21_blending_demo.py` renders three TV21 grounding cases from crops of a real DEM in the repository.

## What it demonstrates

- Baseline terrain scatter rendering with no terrain-aware grounding.
- TV21 seam fading through `TerrainMeshBlendSettings`.
- TV21 contact darkening through `TerrainContactSettings`.
- A diff panel per case so the image-space effect is visible immediately.

The three cases are:

- `rock_cluster`
- `road_edge`
- `building_foundation`

Each crop keeps the real DEM shape but normalizes its local relief before rendering. That keeps the close-up objects and the terrain grounding effect in a readable scale range without changing the underlying terrain form.

## Default dataset

The script defaults to:

- `assets/tif/dem_rainier.tif`

You can point it at any other DEM asset in `assets/tif/` with `--dem`.

## Usage

```bash
python examples/terrain_tv21_blending_demo.py
python examples/terrain_tv21_blending_demo.py --dem assets/tif/luxembourg_dem.tif --width 960 --height 640
```

## Outputs

The default output directory is:

- `examples/out/terrain_tv21_blending_demo/`

The script writes:

- `terrain_tv21_<case>_baseline.png`
- `terrain_tv21_<case>_tv21.png`
- `terrain_tv21_<case>_diff.png`
- `terrain_tv21_contact_sheet.png`
- `terrain_tv21_summary.json`

The contact sheet is arranged per row as:

- baseline
- TV21 enabled
- diff

`terrain_tv21_summary.json` records the DEM path, effective DEM size, crop origins, changed-pixel counts, and per-case output paths.
