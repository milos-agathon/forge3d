"""AUTOCHIRON shot grammar and canonical, independently validated GPU plans.

Coordinates use the rigs' local Y-up terrain contract. Geographic subjects
require an explicit caller-supplied transform into that same contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ._native import get_native_module
from .animation import CameraAnimation, CameraKeyframe
from .camera_rigs import TerrainClearance, _unwrap_keyframes
from .terrain_scatter import TerrainScatterSource

__all__ = ["Subject", "FramingSpec", "Shot", "ShotPlan", "CinemaInfeasible", "CinemaNativeUnavailable", "plan"]


def _canonical(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cinema artifacts require finite numbers")
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    return value


def _json(value) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _numbers(value, length, name):
    if len(value) != length or any(isinstance(x, bool) for x in value):
        raise ValueError(f"{name} must contain {length} numbers")
    result = tuple(float(x) for x in value)
    _canonical(result)
    return result


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    _canonical(result)
    return result


class CinemaInfeasible(ValueError):
    """A binding constraint prevents a validated shot from being emitted."""
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.binding_constraint = diagnostic.get("binding_constraint", "spec")
        super().__init__(_json(diagnostic))


class CinemaNativeUnavailable(RuntimeError):
    """The planning/physical GPU validation backend is unavailable."""


@dataclass(frozen=True)
class Subject:
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self):
        if len(self.bounds) != 2:
            raise ValueError("subject bounds must contain minimum and maximum")
        bounds = tuple(_numbers(p, 3, "subject corner") for p in self.bounds)
        if any(bounds[1][i] <= bounds[0][i] for i in range(3)):
            raise ValueError("subject bounds need positive extent in all three axes")
        object.__setattr__(self, "bounds", bounds)

    @classmethod
    def point(cls, position, *, diameter: float):
        p = _numbers(position, 3, "subject point")
        diameter = _number(diameter, "diameter")
        if not math.isfinite(diameter) or diameter <= 0:
            raise ValueError("point subject diameter must be positive and finite")
        return cls((tuple(v - diameter / 2 for v in p), tuple(v + diameter / 2 for v in p)))

    @classmethod
    def from_lon_lat_height(cls, position, *, to_world: Callable, diameter: float):
        """Transform a declared geographic point; diameter is in world units."""
        p = _numbers(position, 3, "lon/lat/height")
        if not -180 <= p[0] <= 180 or not -90 <= p[1] <= 90:
            raise ValueError("longitude/latitude out of range")
        return cls.point(to_world(*p), diameter=diameter)

    @classmethod
    def from_lon_lat_bounds(cls, bounds, *, to_world: Callable):
        """Enclose all eight transformed geographic AABB corners in world space."""
        if len(bounds) != 2:
            raise ValueError("geographic bounds require minimum and maximum")
        lower, upper = (_numbers(p, 3, "geographic corner") for p in bounds)
        if any(a >= b for a, b in zip(lower, upper)):
            raise ValueError("geographic bounds must be ordered with positive extent")
        if not (-180 <= lower[0] < upper[0] <= 180 and -90 <= lower[1] < upper[1] <= 90):
            raise ValueError("longitude/latitude out of range")
        corners = [_numbers(to_world(*(upper[k] if i & (1 << k) else lower[k] for k in range(3))), 3,
                            "transformed corner") for i in range(8)]
        return cls((tuple(min(p[k] for p in corners) for k in range(3)),
                    tuple(max(p[k] for p in corners) for k in range(3))))


@dataclass(frozen=True)
class FramingSpec:
    region: tuple[float, float, float, float]
    size_frac_band: tuple[float, float]
    headroom: float
    fov_deg: float
    aspect: float

    def __post_init__(self):
        region = _numbers(self.region, 4, "screen region")
        band = _numbers(self.size_frac_band, 2, "size fraction band")
        if not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
            raise ValueError("screen region must be an ordered unit rectangle")
        if not 0 < band[0] < band[1] <= 1:
            raise ValueError("size fraction band must be inside (0,1]")
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "size_frac_band", band)
        for key in ("headroom", "fov_deg", "aspect"):
            object.__setattr__(self, key, _number(getattr(self, key), key))
        if not (0 <= self.headroom < 1 and 0 < self.fov_deg < 180 and self.aspect > 0):
            raise ValueError("invalid headroom, FOV or aspect")
        _json(asdict(self))


@dataclass(frozen=True)
class Shot:
    kind: str
    subject: Subject
    duration_s: float
    framing: FramingSpec
    clearance: float
    max_speed: float
    max_jerk: float
    bearing_deg: tuple[float, float]
    elevation_deg: tuple[float, float]
    transition: tuple[float, float] = (0.0, 0.0)
    start_eye: tuple[float, float, float] | None = None
    end_eye: tuple[float, float, float] | None = None
    max_refine_passes: int = TerrainClearance().max_refine_passes

    def __post_init__(self):
        if self.kind not in ("reveal", "orbit", "flyover", "dolly"):
            raise ValueError("unknown shot kind")
        if not isinstance(self.subject, Subject) or not isinstance(self.framing, FramingSpec):
            raise TypeError("shot requires Subject and FramingSpec")
        for key in ("duration_s", "clearance", "max_speed", "max_jerk"):
            v = _number(getattr(self, key), key)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be positive and finite")
            object.__setattr__(self, key, v)
        for key in ("bearing_deg", "elevation_deg", "transition"):
            object.__setattr__(self, key, _numbers(getattr(self, key), 2, key))
        if min(self.transition) < 0 or sum(self.transition) > self.duration_s:
            raise ValueError("transition windows exceed shot duration")
        if isinstance(self.max_refine_passes, bool) or not isinstance(self.max_refine_passes, int) or self.max_refine_passes < 0:
            raise ValueError("max_refine_passes must be a non-negative integer")
        for key in ("start_eye", "end_eye"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, _numbers(getattr(self, key), 3, key))


def _native():
    native = get_native_module()
    if native is None or not all(hasattr(native, n) for n in ("cinema_solve", "cinema_validate")):
        raise CinemaNativeUnavailable("AUTOCHIRON requires rebuilt cinema_solve/cinema_validate bindings")
    return native


def _terrain(source):
    if not isinstance(source, TerrainScatterSource):
        raise TypeError("source must be TerrainScatterSource")
    values = np.array(source._scaled_heights, dtype="<f4", order="C", copy=True)
    values[values == 0] = 0.0
    digest = hashlib.sha256(values.tobytes() + _json([source.width, source.height, source.terrain_width]).encode()).hexdigest()
    return (values.ravel().tolist(), source.width, source.height, source.terrain_width), digest


def _call(function, *args):
    try:
        return json.loads(function(*args))
    except ValueError as error:
        try:
            diagnostic = json.loads(str(error))
        except (ValueError, TypeError):
            raise
        if diagnostic.get("binding_constraint") in ("gpu_trace", "validation_resolution"):
            raise CinemaNativeUnavailable(_json(diagnostic)) from error
        raise CinemaInfeasible(diagnostic) from error


@dataclass(frozen=True)
class ShotPlan:
    shots: tuple[Shot, ...]
    _artifact: str | None = None

    def __post_init__(self):
        shots = tuple(self.shots)
        if not shots or not all(isinstance(s, Shot) for s in shots):
            raise ValueError("ShotPlan requires at least one Shot")
        object.__setattr__(self, "shots", shots)

    @property
    def validation(self):
        return [] if self._artifact is None else [s["validation"] for s in json.loads(self._artifact)["compiled"]]

    @property
    def orbit_adjustments(self):
        """Requested and effective orbit inputs, including exact joined starts."""
        return [] if self._artifact is None else [s.get("orbit_adjustments") for s in json.loads(self._artifact)["compiled"]]

    @property
    def keyframes_json(self):
        if self._artifact is None:
            raise CinemaNativeUnavailable("plan has not been solved and validated")
        return _json(json.loads(self._artifact)["keyframes"])

    def bake(self) -> CameraAnimation:
        _native()
        records = json.loads(self.keyframes_json)
        animation = CameraAnimation()
        animation.replace_keyframes([CameraKeyframe(k["time"], k["phi"], k["theta"], k["radius"], k["fov"], target=tuple(k["target"])) for k in records])
        return animation

    def to_json(self):
        return self._artifact or _json({"schema": "forge3d.shotplan/1", "shots": [asdict(s) for s in self.shots], "compiled": None, "keyframes": None, "terrain_sha256": None, "certificate_links": []})

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid {value}")))
        if set(data) != {"schema", "shots", "compiled", "keyframes", "terrain_sha256", "certificate_links"} or data["schema"] != "forge3d.shotplan/1":
            raise ValueError("unsupported shot plan schema")
        shots = []
        for item in data["shots"]:
            item = dict(item)
            item["subject"] = Subject(**item["subject"])
            item["framing"] = FramingSpec(**item["framing"])
            shots.append(Shot(**item))
        if data["compiled"] is None:
            if data["keyframes"] is not None or data["terrain_sha256"] is not None or data["certificate_links"]:
                raise ValueError("unsolved plan cannot carry compiled data")
            return cls(tuple(shots))
        if len(data["compiled"]) != len(shots) or not data["keyframes"]:
            raise ValueError("incomplete compiled artifact")
        for k in data["keyframes"]:
            if set(k) != {"time", "phi", "theta", "radius", "fov", "target"}:
                raise ValueError("invalid keyframe record")
            k["target"] = _numbers(k["target"], 3, "keyframe target")
            for field in ("time", "phi", "theta", "radius", "fov"):
                k[field] = _number(k[field], f"keyframe {field}")
            if k["time"] < 0 or k["radius"] <= 0 or not 0 < k["fov"] < 180:
                raise ValueError("invalid keyframe time, radius or FOV")
        if any(b["time"] <= a["time"] for a, b in zip(data["keyframes"], data["keyframes"][1:])):
            raise ValueError("keyframes must be ordered")
        data["shots"] = [asdict(s) for s in shots]
        return cls(tuple(shots), _json(data))

    def validate(self, source: TerrainScatterSource):
        if self._artifact is None:
            raise CinemaNativeUnavailable("unsolved shot plan")
        args, digest = _terrain(source)
        data = json.loads(self._artifact)
        if digest != data["terrain_sha256"]:
            raise ValueError("terrain differs from the compiled plan")
        offset = 0.0
        reports = []
        for shot, compiled in zip(self.shots, data["compiled"]):
            reports.append(_call(_native().cinema_validate, _json(asdict(shot)), _json(compiled["solution"]), self.keyframes_json, *args, offset))
            offset += shot.duration_s
        return reports

    def link_certificate(self, path, *, pubkey, frame_time: float):
        """Link an existing verified render certificate; never signs or rewrites it."""
        from .certificate import verify, payload_sha256
        frame_time = _number(frame_time, "frame_time")
        if self._artifact is None or not 0 <= frame_time <= sum(s.duration_s for s in self.shots):
            raise ValueError("certificate requires a solved plan and an in-range frame time")
        if not verify(path, pubkey):
            raise ValueError("render certificate verification failed")
        certificate = json.loads(Path(path).read_text(encoding="utf-8"))
        data = json.loads(self._artifact)
        # This is an artifact association, not a signature over this shot plan.
        data["certificate_links"].append({"payload_sha256": payload_sha256(certificate), "frame_time": float(frame_time), "plan_keyframes_sha256": hashlib.sha256(self.keyframes_json.encode()).hexdigest()})
        return replace(self, _artifact=_json(data))


def plan(shots: Sequence[Shot] | ShotPlan, source: TerrainScatterSource) -> ShotPlan:
    requested = shots if isinstance(shots, ShotPlan) else ShotPlan(tuple(shots))
    args, digest = _terrain(source)
    native = _native()
    compiled, records, effective = [], [], []
    offset = 0.0
    last_eye = None
    for shot in requested.shots:
        requested_bearings = shot.bearing_deg
        if last_eye is not None:
            if shot.start_eye is None:
                shot = replace(shot, start_eye=last_eye)
            elif tuple(shot.start_eye) != tuple(last_eye):
                raise CinemaInfeasible({"status": "infeasible", "binding_constraint": "shot_join", "detail": "adjacent shots must share an eye position"})
            if shot.kind == "orbit":
                center = np.mean(shot.subject.bounds, axis=0)
                principal = math.degrees(math.atan2(last_eye[2] - center[2], last_eye[0] - center[0]))
                bearing = principal + 360.0 * round((requested_bearings[0] - principal) / 360.0)
                shot = replace(shot, bearing_deg=(bearing, bearing + requested_bearings[1] - requested_bearings[0]))
        result = _call(native.cinema_solve, _json(asdict(shot)), *args)
        if shot.kind == "orbit":
            result["orbit_adjustments"]["requested_bearing_deg"] = requested_bearings
            result["orbit_adjustments"]["joined_start"] = last_eye is not None
        if records:
            first, prior = result["keyframes"][0], records[-1]
            candidates = []
            for sign, turn in ((1., 0.), (-1., 180.)):
                phi_offset = turn + 360. * round((prior["phi"] - first["phi"] - turn) / 360.)
                theta_offset = 360. * round((prior["theta"] - sign * first["theta"]) / 360.)
                distance = (first["phi"] + phi_offset - prior["phi"]) ** 2 + (sign * first["theta"] + theta_offset - prior["theta"]) ** 2
                candidates.append((distance, sign, phi_offset, theta_offset))
            _, sign, phi_offset, theta_offset = min(candidates, key=lambda entry: entry[0])
            # Align the whole shot's equivalent chart, retaining its signed
            # azimuth sweep and the exact joined eye. Revalidate the real cubic.
            for key in result["keyframes"]:
                key["phi"] += phi_offset
                key["theta"] = sign * key["theta"] + theta_offset
        for i, key in enumerate(result["keyframes"]):
            if records and i == 0:
                if key["target"] != records[-1]["target"]:
                    raise CinemaInfeasible({"status": "infeasible", "binding_constraint": "shot_join", "detail": "adjacent look targets differ"})
                continue
            records.append(dict(key, time=float(np.float32(key["time"] + offset))))
        last_eye = tuple(result["solution"]["segments"][-1]["eye1"])
        compiled.append(result)
        effective.append(shot)
        offset += shot.duration_s
    # Reuse the rig's angle-unwrapping machinery before validating the joined cubic.
    normalized = _unwrap_keyframes([CameraKeyframe(k["time"], k["phi"], k["theta"], k["radius"], k["fov"], target=tuple(k["target"])) for k in records])
    records = [{"time": k.time, "phi": k.phi_deg, "theta": k.theta_deg, "radius": k.radius, "fov": k.fov_deg, "target": k.target} for k in normalized]
    data = {"schema": "forge3d.shotplan/1", "shots": [asdict(s) for s in effective], "compiled": compiled, "keyframes": records, "terrain_sha256": digest, "certificate_links": []}
    result = ShotPlan(tuple(effective), _json(data))
    reports = result.validate(source)
    for entry, report in zip(compiled, reports):
        entry["validation"] = report
    return replace(result, _artifact=_json(data))
