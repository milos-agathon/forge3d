"""Render and encode a feature-bound reveal over Grindelwald."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from _chronos_video import encode_video
from _import_shim import ensure_repo_import

ensure_repo_import()

from forge3d import LightingPreset, OutputSpec, TerrainSource
from forge3d.camera_rigs import TerrainClearance, TerrainOrbitRig
from forge3d.subject_story import StoryAnnotation, SubjectFeature, compile_subject_shot

EIGER = "{5BE294B0-910F-4DA3-BECD-201FB42320E3}"
OSTEGG = "{B056353F-275D-47FA-AB13-DE1E0E7826B5}"
GRINDELWALD = "{2E3C1D00-71F1-4F81-8A16-5D5D17507F13}"
RAIL_APPROACH = "{B4A69DCA-D9A3-4F12-9817-C9671B1B46D0}"
RAIL_VERTEX = 34
FPS = 24
SEED = 7


def compile_shot(source_dir, subject_id, size, *, hold_seconds=2):
    source = json.loads((source_dir / "sources.json").read_text(encoding="utf-8"))
    features_path = source_dir / "features.json"
    terrain_path = source_dir / "terrain.npy"
    for path, key in ((features_path, "features_sha256"), (terrain_path, "terrain_sha256")):
        if sha256(path.read_bytes()).hexdigest() != source[key]:
            raise ValueError(f"Swiss input differs from recorded source: {path}")
    layers = json.loads(features_path.read_text(encoding="utf-8"))
    records = {f["properties"]["UUID"]: f for features in layers.values() for f in features}
    # These are editorial picks. Selection uses original swissTLM3D UUIDs,
    # including the original vertex of the named rail approach segment.
    names = {EIGER: "Eiger", OSTEGG: "Ostegg",
             GRINDELWALD: "Grindelwald", RAIL_APPROACH: "Grindelwald Grund rail approach"}
    selection = {}
    for key, name in names.items():
        record = records[key]
        coordinates = record["geometry"]["coordinates"]
        if key == RAIL_APPROACH:
            coordinates = coordinates[RAIL_VERTEX]
        selection[key] = {
            "name": name, "source_name": record["properties"]["NAME"],
            "easting_northing_height_m": coordinates,
        }
    features = [
        SubjectFeature(key, row["name"], *row["easting_northing_height_m"][:2])
        for key, row in selection.items()
    ]
    dx, x_skew, west, y_skew, dy, south = source["texture_affine"]
    shot = compile_subject_shot(
        TerrainSource(
            path=terrain_path, crs=source["crs"],
            metadata={"geotransform": (west, dx, x_skew, south, y_skew, dy)},
        ),
        features, subject_id=subject_id, feature_crs=source["crs"],
        bounds_wsen=source["mesh_bounds_wsen"],
        orbit=lambda target: TerrainOrbitRig(
            target, duration=3, radius=10500, radius_end=4700,
            phi_start_deg=78, phi_end_deg=103, theta_start_deg=17,
            theta_end_deg=18, fov_start_deg=68, fov_end_deg=50, target_height_offset=-600,
            clearance=TerrainClearance(minimum_height=300),
        ),
        fps=FPS, hold_seconds=hold_seconds, output=OutputSpec(width=size[0], height=size[1]),
        lighting=LightingPreset(sun_direction=(-0.65, 0.64, 0.4), intensity=1.25),
        framing=(0.04, 0.04, 0.96, 0.96), reserved_text=(0.035, 0.035, 0.88, 0.22),
        reveal_annotation=StoryAnnotation("Grindelwald to {subject}", (0.06, 0.135), 30),
        hold_annotation=StoryAnnotation("{subject} above the valley", (0.06, 0.135), 30),
        establish_must_see=(GRINDELWALD, RAIL_APPROACH),
    )
    return shot, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--subject-id", choices=[EIGER, OSTEGG], default=EIGER)
    parser.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--hold-seconds", type=float, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if any(side % 2 for side in args.size):
        parser.error("the yuv420p encoder requires even dimensions")
    args.out.mkdir(parents=True)
    started = perf_counter()
    try:
        shot, selection = compile_shot(args.source, args.subject_id, args.size, hold_seconds=args.hold_seconds)
    except ValueError as exc:
        (args.out / "failure.json").write_text(json.dumps({
            "subject_id": args.subject_id, "size": args.size, "diagnostic": str(exc),
            "elapsed_s": perf_counter() - started,
        }, indent=2))
        raise
    receipt = {
        "subject_id": args.subject_id, "features": selection, "size": args.size,
        "source_receipt": str((args.source / "sources.json").resolve()),
        "compile_seconds": perf_counter() - started, "fps": shot.fps, "frame_count": shot.frame_count,
        "key_frames": shot.key_frames, "frame_checks": shot.frame_checks,
    }
    receipt_path = args.out / "shot.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.plan_only:
        return
    started = perf_counter()
    manifest = shot.render(args.out / "frames", base_seed=SEED)
    receipt["render_seconds"] = perf_counter() - started
    started = perf_counter()
    movie = args.out / "grindelwald.mp4"
    encode_video(args.out / "frames", movie, shot.fps, transfer="iec61966-2-1", frame_count=shot.frame_count)
    receipt["encode_seconds"] = perf_counter() - started
    receipt["movie_sha256"] = sha256(movie.read_bytes()).hexdigest()
    receipt["chronos_manifest"] = "frames/flythrough_manifest.json"
    receipt["chronos_manifest_sha256"] = sha256(
        (args.out / receipt["chronos_manifest"]).read_bytes()
    ).hexdigest()
    receipt["stills"] = {}
    for name, frame in shot.key_frames.items():
        row = manifest.frame_record(frame)
        receipt["stills"][name] = {
            "frame": frame, "image": "frames/" + row["image"],
            "provenance": "frames/" + row["provenance"], "pixel_hash": row["pixel_hash"],
        }
    # Recompile and render one selected key view through CHRONOS replay.
    key = shot.key_frames["reveal"]
    started = perf_counter()
    replay = manifest.replay_frame(shot.frame_scene(key), key, args.out / "replayed-reveal.png")
    receipt["replay"] = {"frame": key, "pixel_hash": replay["pixel_hash"], "seconds": perf_counter() - started}
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
