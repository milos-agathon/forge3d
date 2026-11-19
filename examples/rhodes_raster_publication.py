"""Compose and measure the matching original/revised QGIS and Forge3D publications."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image

from rhodes_raster_recipe import publication_plate


def encode(directory):
    start = time.perf_counter()
    movie = directory / "rhodes-camera.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "12",
            "-i",
            str(directory / "plate-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(movie),
        ],
        check=True,
    )
    report = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(movie),
        ],
        text=True,
    )
    (stream,) = json.loads(report)["streams"]
    return time.perf_counter() - start, stream


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def qgis_legend(project):
    with ZipFile(project) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".qgs"))
        root = ElementTree.fromstring(archive.read(name))
    entries = []
    for row in root.findall(".//colorPalette/paletteEntry"):
        color = bytes.fromhex(row.attrib["color"].removeprefix("#"))
        entries.append(
            {
                "class_id": int(row.attrib["value"]),
                "label": row.attrib["label"],
                "rgba": [*color, int(row.attrib["alpha"])],
            }
        )
    return {"entries": sorted(entries, key=lambda entry: entry["class_id"])}


def compare(root, camera_dir):
    qgis = json.loads((root / "qgis" / "metrics.json").read_text())
    adaptation = json.loads((root / "qgis-3d" / "setup-cost.json").read_text())
    launches = json.loads(
        (root / "qgis-3d" / "launch-cost.json").read_text(encoding="utf-8-sig")
    )
    assert all(result["exit_code"] == 0 for result in launches.values())
    stages = []
    for stem, publication in (
        ("2023-08-07-0", "publication"),
        ("2023-08-17-2", "publication-revised"),
    ):
        forge = root / publication
        qgis_capture = root / "qgis-3d" / stem / camera_dir
        qgis_output = qgis_capture / "publication"
        qgis_output.mkdir(exist_ok=True)
        forge_frames = json.loads((forge / "frames.json").read_text())
        qgis_frames = json.loads((qgis_capture / "evidence.json").read_text())
        qgis_case = next(
            row for row in qgis["runs"] if f'{row["date"]}-{row["revision"]}' == stem
        )

        start = time.perf_counter()
        classes = np.load(root / "qgis" / f"{stem}-classes.npy")
        legend = qgis_legend(root / "qgis" / f"{stem}.qgz")
        for qgis_entry, forge_entry in zip(
            legend["entries"], forge_frames["legend"]["entries"], strict=True
        ):
            assert qgis_entry == (
                forge_entry | {"label": forge_entry["label"].replace("≥", ">=")}
            )
        palette = np.asarray(
            [(25, 25, 38, 255), *[entry["rgba"] for entry in legend["entries"]]],
            np.uint8,
        )
        inset = Image.fromarray(palette[classes]).resize(
            (280, 280), Image.Resampling.NEAREST
        )
        inset.save(qgis_output / "reference-inset.png")
        setup_seconds = time.perf_counter() - start
        layout_seconds = 0.0
        for index, (frame, pose) in enumerate(
            zip(forge_frames["frames"], qgis_frames["camera_poses"], strict=True)
        ):
            assert (
                pose["pitch"] == frame["tilt"]
                and pose["heading"] == frame["azimuth"] + 90
            )
            assert pose["distance"] == 21000 and pose["fov"] == 58
            native = qgis_capture / "frames" / f"frame_{index:05d}.png"
            assert digest(native) == qgis_frames["frame_sha256"][index]
            with (
                Image.open(native) as image,
                Image.open(forge / frame["frame"]) as other,
            ):
                assert image.size == other.size == (1024, 768)
                assert digest(forge / frame["frame"]) == frame["sha256"]
                start = time.perf_counter()
                path = qgis_output / f"plate-{index:03d}.png"
                publication_plate(image, inset, stem[:10], legend).save(path)
                layout_seconds += time.perf_counter() - start
            with (
                Image.open(path) as a,
                Image.open(forge / f"plate-{index:03d}.png") as b,
            ):
                assert a.size == b.size == (1340, 870)
                # The inset and shaded native pixels legitimately differ.
                for box in ((0, 0, 1340, 74), (1040, 410, 1340, 870)):
                    np.testing.assert_array_equal(
                        np.asarray(a.crop(box)), np.asarray(b.crop(box))
                    )
        count = len(forge_frames["frames"])
        forge_encode, forge_media = encode(forge)
        qgis_encode, qgis_media = encode(qgis_output)
        assert (
            forge_media
            == qgis_media
            == {"width": 1340, "height": 870, "nb_frames": str(count)}
        )
        stages.append(
            {
                "case": stem,
                "frames": count,
                "dimensions": [1340, 870],
                "forge3d_seconds": forge_frames["timings"]
                | {"encode_seconds": forge_encode},
                "qgis_seconds": {
                    "calculator_style_project_and_plan": qgis_case["seconds"],
                    "prepare_capture_project_and_rgba_drape": adaptation[stem],
                    "process_wall_including_native_frame_export": launches[stem][
                        "seconds"
                    ],
                    "native_frame_export_within_process_wall": qgis_frames[
                        "render_seconds"
                    ],
                    "prepare_publication_inset": setup_seconds,
                    "layout": layout_seconds,
                    "encode": qgis_encode,
                },
                "manual_interventions_during_timed_runs": {"forge3d": [], "qgis": []},
                "manual_work_outside_timing": [
                    "Authored both scripts and chose dates, thresholds, and camera path.",
                    "Calibrated the QGIS heading against map coordinates and staged its profile.",
                    "Used the same Python plate compositor and FFmpeg encoder for both tools.",
                ],
                "files": {
                    "forge3d_movie_sha256": digest(forge / "rhodes-camera.mp4"),
                    "qgis_movie_sha256": digest(qgis_output / "rhodes-camera.mp4"),
                },
            }
        )
    report = {
        "scope": "Repeatable scripted authoring and export from prepared inputs",
        "unmeasured": "One-off implementation time for both tools",
        "cases": stages,
    }
    (root / "publication-comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--camera-directory", required=True)
    args = parser.parse_args()
    compare(args.directory.resolve(), args.camera_directory)
