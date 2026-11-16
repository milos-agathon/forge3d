"""Finish verified C07 MRQ frames with their frozen CHRONOS caption and subject leader."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from prepare_c07_mrq import write
from prepare_unreal_job import digest, read
from verify_c07_mrq import check_camera
from verify_unreal_return import verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _chronos_video import encode_video
from c07_grindelwald_eiger import compile_shot
from forge3d import CompiledFrame, compile_frame
from forge3d.chronos import FlythroughManifest
from forge3d.map_scene import _chronos_label_plans, _composite_native_label_layers


def draw_leader(rgb, start, end):
    """Blend a one-pixel line and its one-pixel halo at subpixel positions."""
    origin, target = np.asarray(start), np.asarray(end)
    margin = 4  # The three-pixel endpoint radius plus its antialias footprint.
    lo = np.maximum(0, np.floor(np.minimum(origin, target)).astype(int) - margin)
    hi = np.minimum(rgb.shape[1::-1], np.ceil(np.maximum(origin, target)).astype(int) + margin)
    y, x = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
    pixels = np.stack((x + .5, y + .5), axis=-1)
    segment = target - origin
    t = np.clip(np.sum((pixels - origin) * segment, axis=-1) / (segment @ segment), 0, 1)
    line = np.linalg.norm(pixels - (origin + t[..., None] * segment), axis=-1)
    tip = np.linalg.norm(pixels - target, axis=-1)
    outer = np.maximum(np.clip(2 - line, 0, 1), np.clip(3.5 - tip, 0, 1))[..., None]
    inner = np.maximum(np.clip(1 - line, 0, 1), np.clip(2.5 - tip, 0, 1))[..., None]
    crop = rgb[lo[1]:hi[1], lo[0]:hi[0]]
    crop[:] = np.rint(crop * (1 - outer) + 24 * (outer - inner) + 255 * inner).astype(np.uint8)


class Publication:
    """Use the source frame's caption plan and an independently verified landmark."""

    def __init__(self, job, kit, chronos):
        self.job = job
        self.chronos = chronos
        self.meta = read(kit / "comparison.json")
        self.manifest = FlythroughManifest.load(chronos / "flythrough_manifest.json")
        self.manifest_sha256 = digest(chronos / "flythrough_manifest.json")
        shot_meta = self.meta["shots"][job["shot"]]
        if (self.manifest_sha256 != shot_meta["chronos_manifest_sha256"]
                or job["size"] != shot_meta["size"] or job["subject_id"] != shot_meta["subject_id"]
                or job["fps"] != shot_meta["fps"] or job["frame_count"] > shot_meta["frame_count"]):
            raise ValueError("CHRONOS and the external job identify different shots")
        self.shot, _ = compile_shot(kit, job["subject_id"], job["size"])
        self.checks = self.shot.frame_checks
        self.landmarks = {p["frame"]: p["expected_viewport_xy"] for p in job["landmarks"]
                          if p["feature_id"] == job["subject_id"]}
        self.scene = None
        self.caption_plates = {}
        if set(self.landmarks) != set(range(job["frame_count"])):
            raise ValueError("every publication frame needs a projected subject landmark")

    def _label_scene(self, cls, width, height):
        if self.scene is None:
            self.scene = cls(width, height)
        self.scene.clear_native_text()
        return self.scene

    def _caption_plate(self, compiled, recipe, plans, width, height):
        # Extract a screen-space glyph/halo plate using two opaque mattes.
        # This leaves the external renderer's own pixels outside the glyphs intact.
        key = json.dumps(json.loads(compiled.to_json())["labels"], sort_keys=True)
        if key not in self.caption_plates:
            mattes = []
            for shade in (0, 255):
                base = np.full((height, width, 4), shade, dtype=np.uint8)
                base[:, :, 3] = 255
                drawn, applied = _composite_native_label_layers(
                    base, recipe, plans, scene_factory=self._label_scene,
                )
                if not applied:
                    raise RuntimeError("the native compositor did not draw the frozen publication caption")
                mattes.append(drawn[:, :, :3].astype(np.int16))
            black, white = mattes
            weight = white - black
            mask = np.any((black != 0) | (weight != 255), axis=2)
            yy, xx = np.where(mask)
            if not len(xx) or np.any(weight < 0):
                raise RuntimeError("the native caption did not yield a valid glyph plate")
            reserve = self.meta["caption"]["reserve_left_top_right_bottom"]
            if not (reserve[0] * width <= xx.min() <= xx.max() < reserve[2] * width
                    and reserve[1] * height <= yy.min() <= yy.max() < reserve[3] * height):
                raise ValueError("drawn caption escaped the reserved area")
            self.caption_plates[key] = (yy, xx, black[mask], weight[mask])
        return self.caption_plates[key]

    def compose(self, index, source, recipe):
        record = self.manifest.frame_record(index)
        compiled = CompiledFrame.from_json(record["compiled_frame"])
        camera = json.loads(record["compiled_frame"])["camera"]
        reference = compile_frame(index, self.manifest.base_seed, self.manifest.samples,
                                  json.dumps(recipe.camera.to_dict()), "{}")
        if camera != json.loads(reference.to_json())["camera"]:
            raise ValueError(f"frame {index}: publication camera differs from CHRONOS")
        plans = _chronos_label_plans(compiled)
        caption, = plans["subject-story"].accepted
        text = self.job["frames"][index]["caption"]
        if caption.text != text:
            raise ValueError(f"frame {index}: job caption differs from CHRONOS")

        width, height = source.size
        if [width, height] != self.job["size"]:
            raise ValueError(f"frame {index}: publication image size differs from job")
        projected = self.landmarks[index]
        if projected != list(self.checks[index]["features"][self.job["subject_id"]]["viewport_xy"]):
            raise ValueError(f"frame {index}: landmark differs from the checked shot")
        x, y = (a * b for a, b in zip(projected, (width, height), strict=True))
        yy, xx, ink, weight = self._caption_plate(compiled, recipe, plans, width, height)
        start = (int(xx.min()) + .5, int(yy.max()) + 3.5)
        if not (start[1] < y < height and 0 <= x < width):
            raise ValueError(f"frame {index}: subject has no space below the caption for a leader")
        finished = np.array(source.convert("RGB"))
        draw_leader(finished, start, (x, y))
        finished[yy, xx] = ink + (weight.astype(np.int32) * finished[yy, xx] + 127) // 255
        return Image.fromarray(finished), {
            "frame": index, "time_s": self.job["frames"][index]["time_s"],
            "caption": text, "caption_pixel_bounds": [int(v) for v in (xx.min(), yy.min(), xx.max(), yy.max())],
            "caption_pixels": len(xx), "leader_from_px": start, "subject_px": [x, y],
            "chronos_compiled_sha256": sha256(record["compiled_frame"].encode()).hexdigest(),
            "chronos_source_pixel_hash": record["pixel_hash"],
            "chronos_source_sidecar_sha256": digest(self.chronos / record["provenance"]),
        }


def package(frames, movie, fps, size, count):
    started = perf_counter()
    encode_video(frames, movie, fps, transfer="iec61966-2-1", frame_count=count)
    elapsed = perf_counter() - started
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt,color_range,color_space,"
        "color_transfer,color_primaries:frame=pix_fmt,color_range,color_space,color_transfer,color_primaries",
        "-of", "json", str(movie),
    ]))
    stream, = metadata["streams"]
    expected = {"pix_fmt": "yuv420p", "color_range": "tv", "color_space": "bt709",
                "color_transfer": "iec61966-2-1", "color_primaries": "bt709"}
    if (len(metadata["frames"]) != count or stream["codec_name"] != "h264"
            or [stream["width"], stream["height"]] != size or stream["r_frame_rate"] != f"{fps}/1"
            or any(any(row.get(k) != v for k, v in expected.items())
                   for row in [stream, *metadata["frames"]])):
        raise ValueError("encoded frames differ from declared count, format or colour metadata")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"], check=True)
    return {"file": movie.name, "sha256": digest(movie), "encode_seconds": elapsed, "stream": stream,
            "decoded_frames": count, "checked_colour_metadata_per_frame": True}


def finish(job_path, raw, kit, chronos, out):
    verified = verify(job_path, raw)
    returned = read(raw / "result.json")
    if not verified["usable_movie_frames"] or returned["renderer"]["id"] != "unreal-mrq-external":
        raise ValueError("publication requires a complete verified external MRQ result")
    job = read(job_path)
    scene_path, samples_path = raw.parent / "scene.json", raw / "samples.json"
    if (returned["scene_sha256"] != digest(scene_path)
            or returned["evaluated_samples_sha256"] != digest(samples_path)):
        raise ValueError("MRQ scene or evaluated samples changed")
    sequence = returned["sequence"].rsplit(".", 1)[-1]
    check_camera(job, read(samples_path), read(scene_path), sequence)
    out.mkdir(parents=True)
    frames = out / "frames"
    frames.mkdir()
    started = perf_counter()
    publication = Publication(job, kit, chronos)
    compiled_seconds = perf_counter() - started
    started = perf_counter()
    records = []
    for i in range(job["frame_count"]):
        name = f"frame_{i:08d}.png"
        recipe = publication.shot.frame_scene(i).recipe
        with Image.open(raw / name) as source:
            frame, record = publication.compose(i, source, recipe)
        target = frames / name
        frame.save(target)
        records.append({**record, "file": name, "sha256": digest(target), "mrq_input_sha256": digest(raw / name)})
    composite_seconds = perf_counter() - started
    movie = package(frames, out / "publication.mp4", job["fps"], job["size"], job["frame_count"])
    if any(digest(frames / record["file"]) != record["sha256"]
           or digest(raw / record["file"]) != record["mrq_input_sha256"]
           or digest(chronos / publication.manifest.frame_record(record["frame"])["provenance"])
           != record["chronos_source_sidecar_sha256"] for record in records):
        raise RuntimeError("encoding changed a publication frame, MRQ input frame or CHRONOS sidecar")
    stills = {key: records[i] for key, i in publication.shot.key_frames.items() if i < len(records)}
    stills["hold"] = records[-1]
    write(out / "publication.json", {
        "renderer": returned["renderer"], "source_mrq_result_sha256": digest(raw / "result.json"),
        "mrq_scene_sha256": returned["scene_sha256"],
        "evaluated_samples_sha256": returned["evaluated_samples_sha256"],
        "job_sha256": digest(job_path), "annotation_chronos_sha256": publication.manifest_sha256,
        "compile_seconds": compiled_seconds, "composite_seconds": composite_seconds,
        "movie": movie, "stills": stills, "frames": records,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "raw", "kit", "chronos", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    finish(args.job, args.raw.resolve(), args.kit.resolve(), args.chronos.resolve(), args.out)
