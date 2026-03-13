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

    src = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255]],
            [[0, 0, 255, 255], [255, 255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    output = Path("png_numpy_roundtrip.png")
    f3d.numpy_to_png(output, src)
    loaded = f3d.png_to_numpy(output)

    if not np.array_equal(src, loaded):
        raise RuntimeError("PNG roundtrip mismatch")

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
