#!/usr/bin/env python3
"""Switzerland hydrologic poster using the approved HydroSHEDS PT recipe."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import bosnia_terrain_landcover_viewer as dem_viewer
import germany_hydrologic_pt_3d as hydro

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "examples" / ".cache" / "switzerland_hydrologic_pt"
DEFAULT_OUT_DIR = ROOT / "examples" / "out" / "switzerland_hydrologic_pt"


def main() -> int:
    # Reuse the shared Natural Earth/Terrarium builder, selecting Switzerland
    # before the template asks for its boundary or constructs its DEM cache key.
    dem_viewer.COUNTRY_A3 = "CHE"
    dem_viewer.COUNTRY_NAME = "Switzerland"
    dem_viewer.TARGET_CRS = "EPSG:3035"

    hydro.CACHE_DIR = DEFAULT_CACHE_DIR
    hydro.OUT_DIR = DEFAULT_OUT_DIR
    hydro.LAKES_SHP = Path("D:/hydroshed/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp")
    hydro.RELIEF_WORLD = 3.2
    hydro.TITLE_LINE_2 = "SWITZERLAND"
    hydro.CAPTION_LINES = (
        "©2026 Milos Popovic (https://milospopovic.net)",
        "Data: HydroSHEDS HydroRIVERS & HydroLAKES, AWS Terrarium DEM",
    )

    # The shared template keeps its country in the filename; redirect only the
    # final composed artifact so the Swiss script has Swiss-named deliverables.
    compose = hydro._compose

    def compose_switzerland(plate, out_path, canvas_height):
        swiss_path = out_path.with_name(out_path.name.replace("germany_hydrologic", "switzerland_hydrologic"))
        return compose(plate, swiss_path, canvas_height)

    hydro._compose = compose_switzerland
    return hydro.main()


if __name__ == "__main__":
    raise SystemExit(main())
