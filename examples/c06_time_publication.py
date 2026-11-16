"""Publish real CBS 2023/2024 C04-selected pages from the prepared C06 example.

Usage: python examples/c06_time_publication.py PREPARED_C06_DIRECTORY OUTPUT
The prepared directory is an executed c06_population_publication.py delivery.
No test entitlement is enabled here; ordinary MapPlate product gates apply.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import rasterio  # Import before the repository's partial test stub.
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _import_shim  # noqa: E402,F401
from forge3d.map_plate import BBox, MapPlateConfig
from forge3d.observation_time import ObservationRecord, ObservationTrack
from forge3d.publication import (PublicationSource, PublicationExtent, PublicationRecipe,
                                PublicationLayout, PublicationInset)


def run(prepared: Path, output: Path):
    receipts = {year: json.loads((prepared / directory / "amsterdam.json").read_text())
                for year, directory in [(2024, "base"), (2023, "year-2023")]}
    source_fields = {f.name for f in fields(PublicationSource)}
    sources = {year: PublicationSource(**{k: v for k, v in r["source"].items() if k in source_fields})
               for year, r in receipts.items()}
    for year, source in sources.items():
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != receipts[year]["source"]["sha256"]:
            raise ValueError(f"Prepared CBS {year} source has changed; rerun preparation")
        if source.year != str(year):
            raise ValueError("Prepared source year does not match its delivery")
    base = receipts[2024]
    bounds = BBox(**base["source"]["grid"])
    pages = []
    for city in ("amsterdam", "rotterdam"):
        receipt = json.loads((prepared / "base" / f"{city}.json").read_text())
        points = np.asarray(receipt["footprint"]["coordinates"])
        pages.append(PublicationExtent(city, receipt["title"], BBox(
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()), bounds.crs)))
    breaks = [row["right"] for row in base["classification"][1:] if row["right"] is not None]
    palette = [entry["rgba"] for entry in base["legend"]["entries"][1:]]
    layout = PublicationLayout(MapPlateConfig(width=2000, height=1400, margin=(100, 460, 240, 450)),
                               [PublicationInset("amsterdam", "Amsterdam reference", pages[0].bounds,
                                                 (20, 120, 420, 550))])
    recipe = PublicationRecipe(sources[2024], breaks, palette, bounds, layout=layout,
                               missing_color=base["legend"]["entries"][0]["rgba"])
    # CBS's population reference date is 1 January. Midnight Netherlands time
    # and annual validity below are declared authoring conventions, not archive
    # release/download times or claims of continuous observation during the year.
    records = [ObservationRecord(str(year), f"{year}-01-01T00:00:00+01:00",
                                f"{year}-01-01T00:00:00+01:00", f"{year+1}-01-01T00:00:00+01:00",
                                {"source": sources[year].description()}) for year in (2023, 2024)]
    times = ["2022-12-31T00:00:00+01:00", "2023-01-01T00:00:00+01:00",
             "2024-01-01T00:00:00+01:00", "2025-01-01T00:00:00+01:00"]
    held = recipe.render_times(ObservationTrack(records, "hold"), times, pages, output / "hold")
    missing = recipe.render_times(ObservationTrack(records, "no-data"), [times[-1]], pages, output / "no-data")
    for index, year in [(1, "2023"), (2, "2024"), (3, "2024")]:
        for row in held[index * len(pages):(index + 1) * len(pages)]:
            assert row["source"]["year"] == year
            assert row["source"]["sha256"] == receipts[int(year)]["source"]["sha256"]
            assert row["selection"]["record"]["payload"]["source"]["year"] == year
            assert row["selection"]["caption"] in row["caption"]
    assert all(row["source"] is None and not row["map_rendered"] for row in held[:len(pages)] + missing)
    assert all(row["selection"]["state"] == "held" for row in held[-len(pages):])
    # Values are actual separate observations, not a relabelled raster.
    assert receipts[2023]["source"]["sha256"] != receipts[2024]["source"]["sha256"]
    result = {"prepared": str(prepared.resolve()), "sources": {y: r["source"] for y, r in receipts.items()},
              "time_convention": "CBS 1 January reference date; author-declared midnight Europe/Amsterdam and annual validity",
              "source_reference": "https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers",
              "pages": held + missing, "final_size_inspection": "not performed by this script"}
    (output / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prepared, args.output)
