"""Conservative bounds for the baked AUTOCHIRON curve consumed by C07."""

import math

import numpy as np

from ._subject_geometry import mesh_height, obstacle_at_eye


def _split(control, t):
    first = (1 - t) * control[:-1] + t * control[1:]
    second = (1 - t) * first[:-1] + t * first[1:]
    middle = (1 - t) * second[0] + t * second[1]
    return (np.stack((control[0], first[0], second[0], middle)),
            np.stack((middle, second[1], first[2], control[3])))


def _restrict(control, start, end):
    if end < 1:
        control, _ = _split(control, end)
    if start > 0:
        _, control = _split(control, start / end)
    return control


def _trig(lo, hi, phase):
    # cos(angle - phase) reaches alternating extrema at k*pi.
    a, b = math.radians(lo) - phase, math.radians(hi) - phase
    ends = [math.cos(a), math.cos(b)]
    first, last = math.ceil(a / math.pi), math.floor(b / math.pi)
    if first < last:
        return np.array([-1., 1.])
    if first == last:
        ends.append(1. if first % 2 == 0 else -1.)
    return np.array([min(ends), max(ends)])


def _multiply(*intervals):
    out = np.array([1., 1.])
    for interval in intervals:
        products = out[:, None] * interval
        out = np.array([products.min(), products.max()])
    return out


def _gamma(dtype, operations):
    unit_roundoff = np.finfo(dtype).eps / 2
    return operations * unit_roundoff / (1 - operations * unit_roundoff)


def _eye_bounds(control, padding, shift):
    bounds = np.stack((control.min(axis=0), control.max(axis=0)), axis=1)
    bounds[:, 0] -= padding
    bounds[:, 1] += padding
    target, radius, phi, theta = bounds[:3] + shift[:, None], bounds[3], bounds[4], bounds[5]
    sin_theta = _trig(*theta, math.pi / 2)
    offset = np.stack((
        _multiply(radius, sin_theta, _trig(*phi, 0)),
        _multiply(radius, sin_theta, _trig(*phi, math.pi / 2)),
        _multiply(radius, _trig(*theta, 0)),
    ))
    # f32 target/radius casts, degree conversion, sin/cos, products and
    # addition in upload.rs. Conversion error scales with angle magnitude.
    angle = math.radians(float(np.abs(bounds[4:]).max()))
    rounding = _gamma(np.float32, 12) * (
        np.abs(target).max(axis=1) + np.abs(radius).max() * (1 + 2 * angle)
    )
    box = target + offset
    return box[:, 0] - rounding, box[:, 1] + rounding


class BakedClearance:
    def __init__(self, animation, mesh, obstacles, minimum, shift):
        self.mesh = mesh
        self.obstacles = obstacles
        self.minimum = minimum
        self.shift = np.asarray(shift)
        keys = animation.get_keyframes()
        self.times = np.array([key.time for key in keys], dtype=np.float32)
        if len(keys) < 2 or not np.all(np.diff(self.times) > 0) or any(key.target is None for key in keys):
            raise ValueError("continuous clearance requires a strictly timed, targeted baked rig")
        values = np.array([
            (*np.asarray(key.target)[[0, 2, 1]], key.radius, key.phi_deg, key.theta_deg)
            for key in keys
        ])
        self.segments = []
        for i, (a, b) in enumerate(zip(values[:-1], values[1:])):
            before, after = values[max(0, i - 1)], values[min(len(values) - 1, i + 2)]
            # The native Catmull-Rom tangents are half the adjacent difference.
            # Bezier form provides a convex-hull bound; it supplies no poses.
            control = np.stack((a, a + (b - before) / 6, b - (after - a) / 6, b))
            magnitude = np.maximum(1., np.abs([before, a, b, after]).max(axis=0))
            # Hermite polynomial/basis intermediate terms are bounded by
            # 12 times the largest key value on [0,1]. Both native expressions
            # take fewer than 32 rounded operations per component.
            padding = 12 * magnitude * np.array([_gamma(np.float64, 32)] * 4
                                                + [_gamma(np.float32, 32)] * 2)
            self.segments.append((control, padding))

    def _lower_bound(self, lo, hi):
        if np.any(lo[:2] < -self.mesh.span / 2) or np.any(hi[:2] > self.mesh.span / 2):
            return -math.inf, "terrain footprint"
        margin = lo[2] - self.mesh.max_height(lo[:2], hi[:2])
        if margin <= 0 or margin < self.minimum:
            return margin, "terrain"
        for obstacle, lower, upper in self.obstacles:
            if (np.all(hi[:2] >= lower[:2]) and np.all(lo[:2] <= upper[:2])
                    and hi[2] >= lower[2] and lo[2] <= upper[2] + self.minimum):
                return -math.inf, obstacle
        return margin, None

    def check(self, start, end, eye_at, fail):
        """Prove a lower bound across the existing rig time interval."""
        start, end = np.clip(np.asarray((start, end), dtype=np.float32), self.times[0], self.times[-1])
        if start == end:
            return None
        first = max(0, np.searchsorted(self.times, start, side="right") - 1)
        last = min(len(self.segments), np.searchsorted(self.times, end, side="left"))
        proven = math.inf
        for i in range(first, last):
            origin, limit = self.times[i:i + 2]
            control, padding = self.segments[i]
            pending = [(max(start, origin), min(end, limit))]
            while pending:
                a, b = pending.pop()
                # Native animation computes local t using these f32 operations.
                sub = _restrict(control, float((a - origin) / (limit - origin)),
                                float((b - origin) / (limit - origin)))
                margin, uncertain = self._lower_bound(*_eye_bounds(sub, padding, self.shift))
                if uncertain is None:
                    proven = min(proven, margin)
                    continue
                t = np.float32((float(a) + float(b)) / 2)
                eye = eye_at(float(t))
                xy = self.mesh.grid(eye[:2])
                if np.any(xy < 0) or np.any(xy > self.mesh.heights.shape[0] - 1):
                    fail(f"consumer eye left the terrain footprint at rig time {t:g} s")
                clearance = eye[2] - mesh_height(self.mesh.heights, xy[None])[0]
                if clearance <= 0 or clearance < self.minimum:
                    fail(f"consumer terrain clearance {clearance:.3f} m at rig time {t:g} s violates {self.minimum:g} m")
                obstacle = obstacle_at_eye(eye, self.obstacles, self.minimum)
                if obstacle is not None:
                    fail(f"consumer camera violates clearance of {obstacle} at rig time {t:g} s")
                if t == a or t == b:
                    fail(f"could not prove clearance from {uncertain} at native animation time precision near {t:g} s")
                pending.extend(((t, b), (a, t)))
        return proven
