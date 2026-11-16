"""Prepare exact C07 MRQ inputs, including a Python-authored shorter Ostegg hold."""

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from prepare_unreal_job import digest, prepare, read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_kit(kit):
    manifest = read(kit / "comparison.json")
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or digest(kit / filename) != expected:
            raise ValueError(f"comparison input differs from its manifest: {filename}")
    return manifest


def revised_kit(base, kit, manifest):
    from c07_grindelwald_eiger import compile_shot

    # Keep the three-second reveal and shorten the hold from two seconds to one.
    shot_meta = manifest["shots"]["ostegg"]
    started = perf_counter()
    shot, _ = compile_shot(kit, shot_meta["subject_id"], shot_meta["size"], hold_seconds=1)
    reference = read(kit / "forge3d-ostegg-shot.json")
    count = shot.frame_count
    if json.loads(json.dumps(shot.frame_checks)) != reference["frame_checks"][:count]:
        raise ValueError("shortened hold changed the previously evaluated camera or terrain")
    base.mkdir()
    shot_meta.update(frame_count=count, key_frames=shot.key_frames)
    for key in ("forge3d_movie_sha256", "chronos_manifest_sha256", "frame_selection",
                "source_shot_receipt_sha256"):
        shot_meta.pop(key)
    hashes = {}
    for filename in ("ostegg-camera.csv", "ostegg-landmarks.csv"):
        with (kit / filename).open(encoding="utf-8", newline="") as source:
            table = csv.DictReader(source)
            with (base / filename).open("x", encoding="utf-8", newline="") as target:
                out = csv.DictWriter(target, fieldnames=table.fieldnames)
                out.writeheader()
                out.writerows(row for row in table if int(row["frame"]) < count)
        hashes[filename] = digest(base / filename)
    manifest["files_sha256"] = hashes
    write(base / "comparison.json", manifest)
    write(base / "python-revision.json", {
        "parent_comparison_sha256": digest(kit / "comparison.json"),
        "parent_camera_sha256": digest(kit / "ostegg-camera.csv"),
        "subject_id": shot_meta["subject_id"], "reveal_frames": shot.key_frames["reveal"],
        "hold_seconds": 1, "frame_count": count, "fps": shot.fps,
        "compiler_seconds": perf_counter() - started, "frame_checks": shot.frame_checks,
        "source_terrain_sha256": digest(kit / "terrain.npy"),
        "source_features_sha256": digest(kit / "features.json"),
    })


def main(base, kit, asset_folder, scene_from=None, neutral=False, lifecycle=False):
    meta = verify_kit(kit)
    dem = np.load(kit / "terrain.npy", allow_pickle=False)
    if not np.array_equal(dem, np.loadtxt(kit / "terrain-north-first.csv", delimiter=",", dtype=dem.dtype)):
        raise ValueError("Unreal DEM table does not round-trip to the reviewed heights")
    revision = base / "short-ostegg-kit"
    revised_kit(revision, kit, read(kit / "comparison.json"))
    registration = {
        "horizontal_crs": meta["horizontal_crs"], "vertical_datum": meta["vertical_datum"],
        "project_id": "C07Acceptance", "level_id": f"/Game/{asset_folder}/SwissTerrain",
        "preset_id": "C07PerspectiveFrameCenter", "origin_enu_m": meta["mesh_origin_e_n_z_m"],
        "translation_cm": [0, 0, 0], "rotation_neu_to_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "sensor_height_mm": 24,
    }
    write(base / "registration.json", registration)
    jobs = {}
    for name, shot_kit, shot in (("eiger", kit, "eiger"), ("ostegg", kit, "ostegg"),
                                 ("ostegg_short", revision, "ostegg")):
        job = prepare(shot_kit, shot, registration)
        job["terrain_sha256"] = meta["files_sha256"]["terrain.npy"]
        job["terrain_csv_sha256"] = meta["files_sha256"]["terrain-north-first.csv"]
        job["feature_sha256"] = meta["files_sha256"]["features.json"]
        target = base / f"{name}-job.json"
        write(target, job)
        jobs[name] = str(target)
        print(name, job["frame_count"], job["max_export_projection_error_px"])
    runs = [("eiger", "eiger", "render"), ("ostegg", "ostegg", "render"),
            ("ostegg-short", "ostegg_short", "render")]
    if lifecycle:
        runs.extend((("eiger-repeat", "eiger", "render"),
                     ("eiger-cancel", "eiger", "cancel"), ("eiger-error", "eiger", "error")))
    request = {"kit": str(kit), "jobs": jobs, "neutral": neutral,
               "runs": [{"name": name, "shot": shot, "mode": mode} for name, shot, mode in runs]}
    if scene_from is not None:
        request["scene_from"] = str(scene_from.resolve(strict=True))
    return request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("kit", type=Path)
    parser.add_argument("--asset-folder", default="C07")
    parser.add_argument("--scene-from", type=Path)
    parser.add_argument("--neutral", action="store_true", help="disable the filmic tone curve")
    parser.add_argument("--lifecycle", action="store_true", help="also run repeat, cancellation and failure checks")
    args = parser.parse_args()
    if not args.asset_folder.startswith("C07") or not args.asset_folder.replace("_", "").isalnum():
        parser.error("use a C07 asset folder name containing letters, digits or underscores")
    base = args.attempt.resolve()
    base.mkdir(parents=True)
    pending = base / "request-pending.json"
    write(pending, main(base, args.kit.resolve(strict=True), args.asset_folder,
                        args.scene_from, args.neutral, args.lifecycle))
    pending.rename(base / "request.json")
