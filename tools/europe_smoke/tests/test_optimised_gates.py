"""Every production gate in this package must survive ``python -O``.

MERGE NOTE. Two of the seven fixes independently hit the same trap: a gate
written with the ``assert`` STATEMENT is compiled out entirely by ``python -O``,
so the check exists in the source and not in the run. The gate11 fix fixed its
own ``assert_static_bound`` and the hillshade fix fixed its own
``assert_hillshade_patch_ran``; the hillshade fix explicitly declined to fix
``basemap.assert_frame_rect`` on ownership grounds, and nobody was checking
``cams.assert_grid``, ``cams.assert_axis`` or ``cams.merge_arms`` at all. This
module is the integrator's sweep: it drives every remaining gate in a real
``-O`` subprocess and requires each to raise.

The repo has hit this before -- 246353b8 "fix(examples): France sea-clamp gate
survives python -O".
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[3])

# (gate name, source that must raise, substring the message must carry)
CASES = [
    (
        "cams.assert_grid",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams, config
        lon, lat = config.grid_axes()
        ds = xr.Dataset(coords={"latitude": lat, "longitude": lon + 0.13})
        cams.assert_grid(ds)
        """,
        "lattice",
    ),
    (
        "cams.assert_grid (shape)",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams, config
        lon, lat = config.grid_axes()
        ds = xr.Dataset(coords={"latitude": lat[:-1], "longitude": lon})
        cams.assert_grid(ds)
        """,
        "grid",
    ),
    (
        "cams.assert_axis",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams
        t = np.array(["2026-08-04T00"], dtype="datetime64[ns]")
        ds = xr.Dataset({"omaod550": (("time",), np.zeros(1))}, coords={"time": t})
        cams.assert_axis(ds, np.array(["2026-08-03T00"], dtype="datetime64[ns]"))
        """,
        "delivered axis != requested axis",
    ),
    (
        "cams.merge_arms lead-0 identity",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams
        def arm(times, value):
            t = np.array(times, dtype="datetime64[ns]")
            lat = 78.0 - 0.4 * np.arange(4)
            lon = -38.0 + 0.4 * np.arange(5)
            d = np.full((t.size, lat.size, lon.size), float(value), "float32")
            return xr.Dataset({"omaod550": (("time","latitude","longitude"), d)},
                              coords={"time": t, "latitude": lat, "longitude": lon})
        an = arm(["2026-08-04T00"], 1.0)
        fc = arm(["2026-08-04T00", "2026-08-04T03"], 9.0)
        cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))
        """,
        "lead 0",
    ),
    (
        "cams.merge_arms overlap window",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams
        def arm(times, value):
            t = np.array(times, dtype="datetime64[ns]")
            lat = 78.0 - 0.4 * np.arange(3)
            lon = -38.0 + 0.4 * np.arange(4)
            d = np.full((t.size, lat.size, lon.size), float(value), "float32")
            return xr.Dataset({"omaod550": (("time","latitude","longitude"), d)},
                              coords={"time": t, "latitude": lat, "longitude": lon})
        cams.merge_arms(arm(["2026-08-01T00"], 1.0), arm(["2026-08-01T00"], 2.0),
                        np.datetime64("2026-08-04T00"), np.datetime64("2026-08-03T12"))
        """,
        "overlap",
    ),
    (
        "cams.merge_arms all-non-finite lead 0",
        """
        import numpy as np, xarray as xr
        from tools.europe_smoke import cams
        def arm(times, field):
            t = np.array(times, dtype="datetime64[ns]")
            d = np.stack([field] * t.size).astype("float32")
            return xr.Dataset({"omaod550": (("time","latitude","longitude"), d)},
                              coords={"time": t,
                                      "latitude": 78.0 - 0.4*np.arange(d.shape[1]),
                                      "longitude": -38.0 + 0.4*np.arange(d.shape[2])})
        nan = np.full((4, 5), np.nan, "float32")
        real = np.full((4, 5), 0.3, "float32")
        cams.merge_arms(arm(["2026-08-04T00"], nan),
                        arm(["2026-08-04T00", "2026-08-04T03"], real),
                        np.datetime64("2026-08-04T00"), np.datetime64("2026-08-04T00"))
        """,
        "no finite cells",
    ),
    (
        "basemap.assert_frame_rect",
        """
        from tools.europe_smoke import basemap
        class P:
            frame_rect = (0.0, 0.0, 1.04, 1.0)
        basemap.assert_frame_rect(P())
        """,
        "frame_rect",
    ),
    (
        "advection.assert_static_bound",
        """
        from tools.europe_smoke import advection
        advection.assert_static_bound(area=(72.2, -25.2, 29.8, 45.2))
        """,
        "gate 11(a) FAILED",
    ),
    (
        "basemap.assert_hillshade_patch_ran",
        """
        from tools.europe_smoke import basemap
        state = {"shading_calls": 3, "bounds": None, "shape": None,
                 "hillshade": None, "engine_hillshade": None}
        basemap.assert_hillshade_patch_ran(state, 3)
        """,
        "wrapper did not run",
    ),
]


def _run_under_O(body: str) -> tuple[str, str]:
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, sys.argv[1])
        if __debug__:
            raise SystemExit("the subprocess is not running under -O")
        try:
{textwrap.indent(textwrap.dedent(body).strip(), " " * 12)}
        except AssertionError as exc:
            print("RAISED")
            print(str(exc))
        else:
            print("SILENT")
    """)
    out = subprocess.run([sys.executable, "-O", "-c", src, ROOT],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    return lines[0], "\n".join(lines[1:])


@pytest.mark.parametrize("name, body, needle",
                         CASES, ids=[c[0] for c in CASES])
def test_the_gate_still_raises_under_python_dash_O(name, body, needle):
    verdict, message = _run_under_O(body)
    assert verdict == "RAISED", (
        f"{name} is a no-op under -O; it must raise explicitly rather than "
        f"using the `assert` statement")
    assert needle in message, (name, needle, message)
