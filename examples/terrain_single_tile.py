from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


def main() -> int:
    f3d = _import_forge3d()

    dem = np.asarray(f3d.mini_dem(), dtype=np.float32)
    dem_min = float(dem.min())
    dem_max = float(dem.max())
    span = max(dem_max - dem_min, 1e-6)

    normalized = ((dem - dem_min) / span * 255.0).clip(0.0, 255.0).astype(np.uint8)
    rgba = np.stack([normalized, normalized, normalized, np.full_like(normalized, 255)], axis=-1)

    output = Path("terrain_single_tile.png")
    f3d.numpy_to_png(output, rgba)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
