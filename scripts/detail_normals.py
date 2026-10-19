#!/usr/bin/env python
"""Source-tree CLI for the packaged DEM detail-normal encoder."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from forge3d._detail_normals import (
    compute_detail_residual,
    compute_file_hash,
    detail_to_normal_map,
    gaussian_blur,
    load_dem,
    main as _main,
    save_normal_map,
)


def main(argv: list[str] | None = None) -> int:
    return _main(argv, base_dir=PROJECT_ROOT)


if __name__ == "__main__":
    sys.exit(main())
