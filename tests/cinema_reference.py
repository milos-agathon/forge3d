"""Input-only analytic budget proposal. Does not import or call the solver."""
import hashlib
import json
import math
from pathlib import Path




def reference(data):
    shot = data["shot"]
    center = [sum(pair) / 2 for pair in zip(*shot["subject"]["bounds"])]
    start, end = shot["start_eye"], shot["end_eye"]
    # Independently declared reference: rise, sweep, settle; equal phase time.
    # The top plane comes only from declared terrain, subject and clearance.
    top = max(max(map(max, data["heightmap"])) + shot["clearance"],
              shot["subject"]["bounds"][1][1])
    points = [start, [start[0], top, start[2]], [end[0], top, end[2]], end]
    duration = shot["duration_s"] / 3

    def spherical(point):
        delta = [point[i] - center[i] for i in range(3)]
        radius = math.sqrt(sum(x * x for x in delta))
        return math.atan2(delta[2], delta[0]), math.acos(delta[1] / radius), radius

    phases = []
    for first, last in zip(points, points[1:]):
        p0, t0, r0 = spherical(first)
        p1, t1, r1 = spherical(last)
        dp = abs((p1 - p0 + math.pi) % (2 * math.pi) - math.pi)
        if shot["kind"] == "orbit" and len(phases) == 1:
            dp = abs(math.radians(shot["bearing_deg"][1] - shot["bearing_deg"][0]))
        angle = dp + abs(t1 - t0)
        radial = abs(r1 - r0)
        # Exact q extrema for q=10u^3-15u^4+6u^5, with chain-rule triangle bounds.
        v = (15 / 8) * angle / duration
        a = (10 / math.sqrt(3)) * angle / duration ** 2
        j = 60 * angle / duration ** 3
        vr = (15 / 8) * radial / duration
        ar = (10 / math.sqrt(3)) * radial / duration ** 2
        jr = 60 * radial / duration ** 3
        bound = jr + 3 * ar * v + 3 * vr * (a + v * v) + max(r0, r1) * (j + 3 * v * a + v ** 3)
        phases.append({"first_eye": first, "last_eye": last, "duration_s": duration,
                       "analytic_jerk_bound": bound})
    return {"reference_top": top, "phases": phases,
            "proposed_max_jerk": max(p["analytic_jerk_bound"] for p in phases)}
