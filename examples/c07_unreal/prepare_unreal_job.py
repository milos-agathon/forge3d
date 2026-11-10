"""Prepare a perspective camera job for an explicitly registered Unreal level."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def vec(values, length):
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"expected {length} finite coordinates")
    return result


def unit(value):
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length == 0:
        raise ValueError("camera axes are degenerate")
    return value / length


def prepare(kit, name, registration):
    meta_path = kit / "comparison.json"
    camera_path = kit / f"{name}-camera.csv"
    landmarks_path = kit / f"{name}-landmarks.csv"
    meta = read(meta_path)
    shot = meta["shots"][name]
    if (shot.get("projection", "perspective") != "perspective"
            or shot.get("temporal_samples", 1) != 1
            or shot.get("off_axis") or shot.get("animated_layers")):
        raise ValueError("only centred perspective, one temporal sample and static scene layers are supported")
    if registration["horizontal_crs"] != meta["horizontal_crs"] or registration["vertical_datum"] != meta["vertical_datum"]:
        raise ValueError("registration CRS and datum must match the comparison source")
    if not all(registration[key].strip() for key in ("project_id", "level_id", "preset_id")):
        raise ValueError("project, level and render preset identifiers are required")
    reference = vec(registration["origin_enu_m"], 3)
    translation = vec(registration["translation_cm"], 3)
    rotation = np.asarray(registration["rotation_neu_to_world"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(np.linalg.det(rotation), 1)):
        raise ValueError("registration rotation must be a proper orthonormal 3D rotation")
    height_mm = float(registration["sensor_height_mm"])
    if not np.isfinite(height_mm) or height_mm <= 0:
        raise ValueError("sensor height must be finite and positive")
    width, height = shot["size"]

    def world(enu):
        return translation + rotation @ ((enu - reference)[[1, 0, 2]] * 100)

    directions = {}
    frames = []
    for row in rows(camera_path):
        index = int(row["frame"])
        if index != len(frames):
            raise ValueError("camera frames must be contiguous and zero-based")
        time_s = index / shot["fps"]
        if float(row["time_s"]) != time_s or float(row["roll_deg"]) != 0:
            raise ValueError("supported camera requires frame-aligned times and zero roll")
        eye = world(vec([row[key] for key in ("eye_e_m", "eye_n_m", "eye_z_m")], 3))
        target = world(vec([row[key] for key in ("target_e_m", "target_n_m", "target_z_m")], 3))
        forward = unit(target - eye)
        right = unit(np.cross(rotation[:, 2], forward))
        up = np.cross(forward, right)
        fov_y = float(row["fov_y_deg"])
        near, far = (float(row[key]) * 100 for key in ("near_m", "far_m"))
        if not 0 < fov_y < 180 or not 0 < near < far < float("inf"):
            raise ValueError(f"frame {index}: invalid perspective FOV or clip range")
        if frames and near != frames[0]["reference_clip_cm"][0]:
            raise ValueError("the registered camera requires a fixed near plane")
        half_y = np.tan(np.deg2rad(fov_y) / 2)
        directions[index] = (eye, forward, right, up, half_y)
        frames.append({
            "frame": index, "time_s": time_s, "eye_cm": eye.tolist(), "target_cm": target.tolist(),
            "forward": forward.tolist(), "right": right.tolist(), "up": up.tolist(),
            "focal_length_mm": height_mm / (2 * half_y), "caption": row["caption"],
            "reference_clip_cm": [near, far],
        })
    if len(frames) != shot["frame_count"]:
        raise ValueError("camera count does not match shot manifest")
    residuals = []
    for row in rows(landmarks_path):
        index = int(row["frame"])
        eye, forward, right, up, half_y = directions[index]
        point = world(vec([row["e_m"], row["n_m"], row["terrain_z_m"]], 3))
        offset = point - eye
        depth = offset @ forward
        if depth <= 0:
            raise ValueError(f"frame {index}: transformed landmark is behind the camera")
        viewport = np.array([0.5 + (offset @ right) / (2 * depth * half_y * width / height),
                             0.5 - (offset @ up) / (2 * depth * half_y)])
        reference_xy = vec([row["x_from_left"], row["y_from_top"]], 2)
        residuals.append({
            "frame": index, "feature_id": row["feature_id"],
            "world_cm": point.tolist(), "expected_viewport_xy": reference_xy.tolist(),
            "export_projection_error_px": ((viewport - reference_xy) * [width, height]).tolist(),
        })
    peak = max(max(abs(value) for value in p["export_projection_error_px"]) for p in residuals)
    if peak > 0.5:
        raise ValueError(f"exported camera or registration differs by {peak:g} pixels")
    return {
        "schema": "c07.unreal-perspective-job/1", "registration": registration,
        "shot": name, "subject_id": shot["subject_id"], "size": shot["size"], "fps": shot["fps"],
        "projection": "perspective", "frame_count": len(frames), "temporal_samples": 1,
        "sampling": "one frame-centred sample per PNG; record actual MRQ shutter and time mapping",
        "position_interpolation": "linear between frame keys",
        "filmback_mm": [height_mm * width / height, height_mm],
        "comparison_sha256": digest(meta_path), "camera_sha256": digest(camera_path),
        "landmarks_sha256": digest(landmarks_path), "frames": frames,
        "landmarks": residuals, "max_export_projection_error_px": peak,
        "execution_state": "prepared",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit", type=Path)
    parser.add_argument("shot", choices=["eiger", "ostegg"])
    parser.add_argument("registration", type=Path)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = prepare(args.kit, args.shot, read(args.registration))
    with args.job.open("x", encoding="utf-8") as output:
        json.dump(job, output, indent=2)
        output.write("\n")
    print(f'{job["frame_count"]} poses; max export residual {job["max_export_projection_error_px"]:.6g} px')
