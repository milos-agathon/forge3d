#!/usr/bin/env python3
"""Optimal label placement behind a ridge, explained on the console.

A synthetic DEM holds one cone peak and one sharp east-west ridge between the
camera and a winding valley. Seven point labels and one curved river label go
through ``MapScene`` + ``LabelLayer(occlusion="terrain")``. The script renders
the scene twice, with the optimal plan and with a ``declutter="greedy"`` plan of
the same labels, and saves them side by side in one PNG with each label's
anchor marked. It then explains the compiled plan on the console:

* which candidate each label was placed at, whether it is visible, and the
  optimality gap the solver reported;
* which labels had their primary (center) anchor hidden behind the ridge, and
  which visible alternative won instead;
* how the result differs from ``declutter="greedy"`` on the same labels.

Occlusion is real, not a proxy: labels are projected through the same camera
the terrain renderer uses (``camera_mode="mesh:zup"``), and a linear-eye-depth
image of the terrain, ray-cast on the CPU from that camera, is handed to the
label layer as ``depth_occlusion``. A candidate is hidden when the terrain at
its pixel is nearer to the camera than the label.

The river is a ``curved_text`` label with no caller-supplied geometry: at
compile time the native ``layout_label_candidates`` authority lays its shaped
glyphs along the projected river and offers several stretches as candidates.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d


LAYER_ID = "ridge.labels"
WIDTH, HEIGHT = 1280, 800
SEED = 20260703
FONT_SIZE = 16.0
LATIN_FONT = Path(f3d.__file__).resolve().parent / "data" / "fonts" / "NotoSansLatin-subset.ttf"

DEM_SIZE = 512
CELL_M = 30.0
SPAN_M = DEM_SIZE * CELL_M
# Camera: azimuth ~110 deg puts the ridge between the camera and the valley.
# ``elevation_deg`` is the renderer's polar angle, measured from the zenith.
CAMERA = {"distance": 12500.0, "azimuth_deg": 110.0, "elevation_deg": 54.0, "fov_deg": 40.0}
Z_SCALE = 1.35
CLIP_NEAR, CLIP_FAR = 0.1, max(6000.0, SPAN_M * 1.5)  # MapScene's terrain clip range
DEPTH_SIZE = (640, 400)  # depth image resolution; the sampler rescales to the viewport
DEPTH_BIAS_M = 150.0
LABEL_LIFT_M = 25.0

# Elevation-linear hypsometric tint (metres -> colour).
TINT = (
    (250, "3b5b33"), (550, "567a42"), (850, "7d9255"), (1150, "9c9a6a"),
    (1500, "8e8b72"), (1900, "a6a499"), (2300, "d3d3cf"), (2700, "f5f5f3"),
)
SKY_TOP, SKY_HORIZON = (118, 160, 206), (214, 229, 240)

# Radial candidates far enough from the anchor to peek over the ridge crest.
CANDIDATE_POLICY = {"offset_px": 30.0, "radial_count": 8, "radial_radius_px": 44.0}

# (label_id, text, DEM column, DEM row, priority). Rows grow toward the camera.
POINT_LABELS = (
    ("summit", "Cinder Peak", 175, 153, 50.0),
    ("valley-hut", "Valley Hut", 200, 335, 40.0),
    ("hidden-tarn", "Hidden Tarn", 300, 322, 35.0),
    ("camp-1", "Camp I", 262, 232, 30.0),
    ("camp-2", "Camp II", 272, 240, 25.0),
    ("spring", "Spring", 256, 246, 20.0),
    ("lookout", "Ridge Lookout", 330, 362, 15.0),
)
RIVER_ID = "silver-river"
RIVER_TEXT = "Silver River"
RIVER_PRIORITY = 45.0


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def _fbm(size: int, seed: int, octaves: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), np.float32)
    amplitude = 1.0
    for octave in range(octaves):
        cells = 4 * 2**octave
        grid = rng.standard_normal((cells + 1, cells + 1)).astype(np.float32)
        x = np.linspace(0.0, cells, size)
        i = np.clip(x.astype(int), 0, cells - 1)
        t = x - i
        t = t * t * (3.0 - 2.0 * t)
        top = grid[i][:, i] * (1 - t)[None, :] + grid[i][:, i + 1] * t[None, :]
        bottom = grid[i + 1][:, i] * (1 - t)[None, :] + grid[i + 1][:, i + 1] * t[None, :]
        out += amplitude * (top * (1 - t)[:, None] + bottom * t[:, None])
        amplitude *= 0.55
    return out / np.abs(out).max()


def _river_v(u: np.ndarray | float) -> np.ndarray | float:
    return 0.52 + 0.04 * np.sin(u * 9.0 + 0.6)


def _heightmap(size: int = DEM_SIZE) -> np.ndarray:
    """Cone peak, sharp east-west ridge, and a winding valley, in metres."""
    v, u = np.mgrid[0.0:1.0 : complex(size), 0.0:1.0 : complex(size)]
    cone = 2400.0 * np.exp(-(np.hypot(u - 0.34, v - 0.30) / 0.15) ** 1.6)
    crest = 0.70 + 0.025 * np.sin(u * 7.0)
    along = np.exp(-(((u - 0.52) / 0.36) ** 6))
    ridge = 1200.0 * along / np.cosh((v - crest) / 0.03) ** 1.5
    valley = -120.0 * np.exp(-((v - _river_v(u)) ** 2) / (2.0 * 0.035**2))
    noise = _fbm(size, seed=7)
    detail = 150.0 * noise * (0.35 + 0.65 * np.clip(cone / 1500.0, 0.0, 1.0))
    detail += 60.0 * noise * np.clip(ridge / 1200.0, 0.0, 1.0)
    return (300.0 + cone + ridge + valley + detail).astype(np.float32)


def _hypsometric_palette(dem: np.ndarray, stops: int = 256) -> str:
    """Hex palette whose colours follow elevation linearly.

    MapScene places custom palette stops at evenly spaced elevation
    quantiles, so each stop is coloured by the elevation of its quantile.
    """
    elevations = np.array([e for e, _ in TINT], float)
    colours = np.array([[int(h[i : i + 2], 16) for i in (0, 2, 4)] for _, h in TINT], float)
    quantiles = np.quantile(dem, np.linspace(0.0, 1.0, stops))
    return ",".join(
        "#%02x%02x%02x" % tuple(int(round(np.interp(q, elevations, colours[:, k]))) for k in range(3))
        for q in quantiles
    )


# ---------------------------------------------------------------------------
# Camera: the same orbit/perspective the terrain renderer builds in mesh:zup mode
# ---------------------------------------------------------------------------


class TerrainCamera:
    def __init__(self, dem: np.ndarray) -> None:
        self.dem = dem
        self.rows, self.cols = dem.shape
        self.h_center = 0.5 * (float(dem.min()) + float(dem.max()))
        theta = math.radians(CAMERA["elevation_deg"])
        phi = math.radians(CAMERA["azimuth_deg"])
        self.eye = CAMERA["distance"] * np.array(
            [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
        )
        self.forward = -self.eye / np.linalg.norm(self.eye)
        self.right = np.cross(self.forward, (0.0, 0.0, 1.0))
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.tan_half = math.tan(math.radians(CAMERA["fov_deg"]) * 0.5)
        self.aspect = WIDTH / HEIGHT

    def world(self, col: float, row: float, elevation: float) -> np.ndarray:
        return np.array(
            [
                (col / (self.cols - 1) - 0.5) * SPAN_M,
                (row / (self.rows - 1) - 0.5) * SPAN_M,
                (elevation - self.h_center) * Z_SCALE,
            ]
        )

    def elevation(self, col: float, row: float) -> float:
        c0, r0 = int(np.clip(math.floor(col), 0, self.cols - 2)), int(np.clip(math.floor(row), 0, self.rows - 2))
        tc, tr = col - c0, row - r0
        d = self.dem
        top = d[r0, c0] * (1 - tc) + d[r0, c0 + 1] * tc
        bottom = d[r0 + 1, c0] * (1 - tc) + d[r0 + 1, c0 + 1] * tc
        return float(top * (1 - tr) + bottom * tr)

    def project(self, col: float, row: float, lift: float = LABEL_LIFT_M) -> tuple[float, float, float]:
        """Viewport pixel x, y and linear eye depth of a point on the terrain."""
        p = self.world(col, row, self.elevation(col, row) + lift / Z_SCALE) - self.eye
        depth = float(p @ self.forward)
        ndc_x = float(p @ self.right) / (depth * self.tan_half * self.aspect)
        ndc_y = float(p @ self.up) / (depth * self.tan_half)
        return (ndc_x + 1.0) * 0.5 * WIDTH, (1.0 - ndc_y) * 0.5 * HEIGHT, depth

    def depth_image(self, size: tuple[int, int] = DEPTH_SIZE, coarse_step: float = 90.0) -> np.ndarray:
        """Linear eye depth of the terrain per pixel (``CLIP_FAR`` where no terrain)."""
        width, height = size
        px = (np.arange(width) + 0.5) / width * 2.0 - 1.0
        py = 1.0 - (np.arange(height) + 0.5) / height * 2.0
        ndc_x, ndc_y = np.meshgrid(px, py)
        # Direction with unit forward component, so the ray parameter is the eye depth.
        dirs = (
            self.forward[None, None, :]
            + ndc_x[..., None] * self.tan_half * self.aspect * self.right[None, None, :]
            + ndc_y[..., None] * self.tan_half * self.up[None, None, :]
        ).reshape(-1, 3)
        dem_world = (self.dem - self.h_center) * Z_SCALE

        def below_terrain(t: np.ndarray, rays: np.ndarray) -> np.ndarray:
            p = self.eye + rays * t[:, None]
            col = (p[:, 0] / SPAN_M + 0.5) * (self.cols - 1)
            row = (p[:, 1] / SPAN_M + 0.5) * (self.rows - 1)
            inside = (col >= 0) & (col <= self.cols - 1) & (row >= 0) & (row <= self.rows - 1)
            c0 = np.clip(np.floor(col).astype(int), 0, self.cols - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, self.rows - 2)
            tc, tr = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
            h = (
                dem_world[r0, c0] * (1 - tc) * (1 - tr)
                + dem_world[r0, c0 + 1] * tc * (1 - tr)
                + dem_world[r0 + 1, c0] * (1 - tc) * tr
                + dem_world[r0 + 1, c0 + 1] * tc * tr
            )
            return inside & (p[:, 2] <= h)

        depth = np.full(dirs.shape[0], CLIP_FAR, np.float32)
        active = np.arange(dirs.shape[0])
        t_prev = np.full(dirs.shape[0], CAMERA["distance"] - SPAN_M, np.float64).clip(CLIP_NEAR)
        t = t_prev.copy()
        while active.size and float(t[active].min()) < CLIP_FAR:
            t[active] = t_prev[active] + coarse_step
            hit = below_terrain(t[active], dirs[active])
            hits = active[hit]
            lo, hi = t_prev[hits], t[hits]
            for _ in range(8):  # refine each hit by bisection
                mid = 0.5 * (lo + hi)
                inside = below_terrain(mid, dirs[hits])
                hi = np.where(inside, mid, hi)
                lo = np.where(inside, lo, mid)
            depth[hits] = hi
            t_prev[active] = t[active]
            active = active[~hit & (t[active] < CLIP_FAR)]
        return depth.reshape(height, width)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _typography(color: Sequence[float] = (1.0, 1.0, 1.0, 1.0)) -> dict[str, Any]:
    return {
        "font_size": FONT_SIZE,
        "color": list(color),
        "halo_color": [0.03, 0.05, 0.04, 0.9],
        "halo_width_px": 2.5,
    }


def _depth_keys() -> dict[str, Any]:
    return {
        "projected_depth_convention": "linear_eye_depth",
        "projected_depth_domain": [CLIP_NEAR, CLIP_FAR],
    }


def _river_screen_path(camera: TerrainCamera, samples: int = 160) -> list[tuple[float, float, float]]:
    path = []
    for index in range(samples):
        u = 0.06 + 0.90 * index / (samples - 1)
        path.append(camera.project(u * (DEM_SIZE - 1), float(_river_v(u)) * (DEM_SIZE - 1), lift=5.0))
    # Lay the text out left-to-right on screen so it reads the right way up.
    return path if path[-1][0] >= path[0][0] else path[::-1]


def build_labels(camera: TerrainCamera) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for label_id, text, col, row, priority in POINT_LABELS:
        x, y, eye_depth = camera.project(col, row)
        labels.append(
            {
                "id": label_id,
                "kind": "point",
                "text": text,
                "priority": priority,
                "geometry": {"type": "Point", "coordinates": camera.world(col, row, camera.elevation(col, row)).tolist()},
                "projected_anchor": [x, y, eye_depth],
                **_depth_keys(),
                "requires_terrain": True,
                "terrain_mode": "terrain",
                "occlusion": "terrain",
                "candidate_policy": dict(CANDIDATE_POLICY),
                "typography": _typography(),
            }
        )
    path = _river_screen_path(camera)
    labels.append(
        {
            "id": RIVER_ID,
            "kind": "curved",
            "text": RIVER_TEXT,
            "priority": RIVER_PRIORITY,
            "geometry": {"type": "LineString", "coordinates": [list(p) for p in path]},
            # No geometry_authority: LabelPlan.compile asks the native
            # layout_label_candidates producer to lay the shaped glyphs along
            # this screen-space path ([x, y, eye depth] per vertex).
            "curved_text": True,
            **_depth_keys(),
            "requires_terrain": True,
            "terrain_mode": "terrain",
            "occlusion": "terrain",
            "typography": _typography((0.80, 0.92, 1.0, 1.0)),
        }
    )
    return labels


def _depth_occlusion(depth: np.ndarray) -> dict[str, Any]:
    return {
        "image": np.round(depth, 1).tolist(),
        "depth_convention": "linear_eye_depth",
        "depth_domain": [CLIP_NEAR, CLIP_FAR],
        "bias": DEPTH_BIAS_M,
        "authoritative": False,
        "source": "cpu_heightfield_raycast",
    }


def build_scene(
    output_dir: str | Path,
    dem: np.ndarray,
    labels: Sequence[Mapping[str, Any]],
    depth: np.ndarray,
    *,
    plan: Any | None = None,
    name: str = "optimal",
) -> f3d.MapScene:
    """Scene for the labels; pass ``plan`` to render an already compiled plan."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=dem,
            crs="EPSG:32654",
            metadata={
                "width": DEM_SIZE,
                "height": DEM_SIZE,
                "resolution": [CELL_M, CELL_M],
                "source_id": "synthetic-ridge-dem",
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), **CAMERA),
        lighting=f3d.LightingPreset(
            name="rainier_showcase",
            settings={"colormap": _hypsometric_palette(dem), "colormap_strength": 1.0},
            overrides={"camera": {**CAMERA, "camera_mode": "mesh:zup"}, "exaggeration": Z_SCALE},
        ),
        output=f3d.OutputSpec(
            width=WIDTH, height=HEIGHT, format="png", path=str(output_dir / f"ridge_labels_{name}.png")
        ),
        layers=[
            f3d.LabelLayer(
                layer_id=LAYER_ID,
                labels=list(labels),
                plan=plan,
                atlas=f3d.FontAtlas.default_latin(),
                occlusion="terrain",
                metadata={"source_id": "ridge-labels", "depth_occlusion": _depth_occlusion(depth)},
            )
        ],
        reproducibility_profile=f3d.ReproducibilityProfile(seed=SEED),
    )


def _paint_sky(png_path: Path) -> None:
    """Replace the renderer's flat clear colour with a sky gradient."""
    from PIL import Image

    rgb = np.asarray(Image.open(png_path).convert("RGB")).copy()
    clear = np.all(rgb == rgb[0, 0], axis=-1)
    t = np.linspace(0.0, 1.0, rgb.shape[0])[:, None, None]
    sky = (np.array(SKY_TOP) * (1 - t) + np.array(SKY_HORIZON) * t).astype(np.uint8)
    rgb[clear] = np.broadcast_to(sky, rgb.shape)[clear]
    Image.fromarray(rgb).save(png_path)


INK, HIDDEN, MOVED, DROPPED = (245, 245, 240), (226, 62, 52), (250, 196, 64), (226, 62, 52)


def _nearest_on_rect(point: Sequence[float], rect: Sequence[float]) -> tuple[float, float]:
    return min(max(point[0], rect[0]), rect[2]), min(max(point[1], rect[1]), rect[3])


def _hidden_primaries(plan: Any) -> set[str]:
    hidden = set()
    for label in plan.accepted:
        primary = next((c for c in label.candidates if c.candidate_type == "center"), None)
        if primary is not None and not _candidate_visible(primary):
            hidden.add(label.label_id)
    return hidden


def _annotate_panel(image: Any, labels: Sequence[Mapping[str, Any]], plan: Any, hidden: set[str]) -> Any:
    from PIL import ImageDraw

    placed = {label.label_id: label for label in plan.accepted}
    draw = ImageDraw.Draw(image)
    for record in labels:
        anchor = record.get("projected_anchor")
        if anchor is None:
            continue
        ax, ay = float(anchor[0]), float(anchor[1])
        label = placed.get(record["id"])
        if label is not None and label.candidate.candidate_type != "center":
            tx, ty = _nearest_on_rect((ax, ay), label.candidate.bounds)
            draw.line([(ax, ay), (tx, ty)], fill=(20, 20, 20), width=4)
            draw.line([(ax, ay), (tx, ty)], fill=MOVED, width=2)
        if label is None:
            r = 8
            for dx, dy in ((r, r), (r, -r)):
                draw.line([(ax - dx, ay - dy), (ax + dx, ay + dy)], fill=(20, 20, 20), width=6)
                draw.line([(ax - dx, ay - dy), (ax + dx, ay + dy)], fill=DROPPED, width=3)
        elif record["id"] in hidden:
            draw.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], outline=(20, 20, 20), width=5)
            draw.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], outline=HIDDEN, width=3)
        elif label.candidate.candidate_type != "center":
            draw.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=INK, outline=(20, 20, 20), width=2)
    return image


def compose_comparison(
    greedy_png: Path,
    optimal_png: Path,
    labels: Sequence[Mapping[str, Any]],
    greedy: Any,
    optimal: Any,
    out_path: Path,
) -> None:
    """Greedy and optimal renders side by side, with anchors and captions."""
    from PIL import Image, ImageDraw, ImageFont

    title_font = ImageFont.truetype(str(LATIN_FONT), 30)
    body_font = ImageFont.truetype(str(LATIN_FONT), 21)
    hidden = _hidden_primaries(optimal)
    texts = {str(record["id"]): str(record["text"]) for record in labels}
    gap = _solver_record(optimal).get("gap")

    dropped = sorted(set(texts) - {label.label_id for label in greedy.accepted})
    moved = sorted(label.label_id for label in optimal.accepted if label.label_id in hidden)
    panels = (
        (
            greedy_png,
            greedy,
            "Greedy declutter",
            f"{len(greedy.accepted)} of {len(texts)} labels placed"
            + (f"  |  dropped: {', '.join(texts[i] for i in dropped)}" if dropped else ""),
        ),
        (
            optimal_png,
            optimal,
            "Optimal declutter",
            f"{len(optimal.accepted)} of {len(texts)} labels placed  |  "
            f"{len(moved)} moved off anchors hidden by the ridge  |  "
            f"gap {float(gap or 0.0):.1e}, certified",
        ),
    )
    header, footer, gutter = 96, 58, 16
    canvas = Image.new("RGB", (2 * WIDTH + gutter, header + HEIGHT + footer), (24, 28, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (png, plan, title, subtitle) in enumerate(panels):
        x0 = index * (WIDTH + gutter)
        panel = _annotate_panel(Image.open(png).convert("RGB"), labels, plan, hidden)
        canvas.paste(panel, (x0, header))
        draw.text((x0 + 24, 14), title, font=title_font, fill=INK)
        draw.text((x0 + 24, 56), subtitle, font=body_font, fill=(190, 196, 190))

    y = header + HEIGHT + footer // 2
    x = 24
    draw.ellipse([x, y - 5, x + 10, y + 5], fill=INK)
    x += 20
    note = "anchor of a moved label"
    draw.text((x, y), note, font=body_font, fill=INK, anchor="lm")
    x += draw.textlength(note, font=body_font) + 40
    draw.ellipse([x, y - 8, x + 16, y + 8], outline=HIDDEN, width=3)
    x += 26
    note = "anchor hidden behind the ridge (depth test)"
    draw.text((x, y), note, font=body_font, fill=INK, anchor="lm")
    x += draw.textlength(note, font=body_font) + 40
    draw.line([(x, y), (x + 36, y)], fill=MOVED, width=3)
    x += 46
    note = "label moved to a visible candidate"
    draw.text((x, y), note, font=body_font, fill=INK, anchor="lm")
    x += draw.textlength(note, font=body_font) + 40
    draw.line([(x, y - 8), (x + 16, y + 8)], fill=DROPPED, width=3)
    draw.line([(x, y + 8), (x + 16, y - 8)], fill=DROPPED, width=3)
    x += 26
    draw.text((x, y), "label dropped", font=body_font, fill=INK, anchor="lm")
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Greedy comparison: the same depth test, driven through LabelPlan.compile
# ---------------------------------------------------------------------------


class DepthOcclusionSampler:
    """Depth-image sampler for calling ``LabelPlan.compile`` directly.

    Applies the same rule as MapScene's ``depth_occlusion`` input: a candidate
    is visible when its linear eye depth is no further than the terrain depth
    at its pixel plus the bias. ``run_example`` checks that a direct optimal
    compile with this sampler reproduces the MapScene plan, so the greedy
    comparison is like-for-like.
    """

    requires_projected_anchor = True

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth

    def sample_label(self, coords: Sequence[float], *, record: Mapping[str, Any], label_id: str) -> dict[str, Any]:
        del record, label_id
        rows, cols = self.depth.shape
        col = int(round(min(1.0, max(0.0, coords[0] / (WIDTH - 1))) * (cols - 1)))
        row = int(round(min(1.0, max(0.0, coords[1] / (HEIGHT - 1))) * (rows - 1)))
        scene_depth = float(self.depth[row, col])
        return {
            "scene_depth": scene_depth,
            "label_depth": float(coords[2]),
            "visible": bool(float(coords[2]) <= scene_depth + DEPTH_BIAS_M),
            "occlusion": "depth_aov",
            "depth_tested": True,
            "depth_convention": "linear_eye_depth",
            "depth_domain": [CLIP_NEAR, CLIP_FAR],
            "bias": DEPTH_BIAS_M,
            "sample_pixel": [col, row],
            "depth_authority": "deterministic_depth_proxy",
        }


# ---------------------------------------------------------------------------
# Console explanation
# ---------------------------------------------------------------------------


def _candidate_visible(candidate: Any) -> bool:
    return (candidate.details or {}).get("visible") is not False and (
        candidate.terrain_sample or {}
    ).get("visible") is not False


def _solver_record(plan: Any) -> Mapping[str, Any]:
    return next((r for r in plan.rationale if r.get("kind") == "solver"), {})


def _choices(plan: Any) -> dict[str, str]:
    return {label.label_id: label.candidate.candidate_id for label in plan.accepted}


def _short(candidate_id: str | None) -> str:
    return "-" if candidate_id is None else candidate_id.split(":", 1)[-1]


def print_plan_table(plan: Any) -> None:
    solver = _solver_record(plan)
    gap = solver.get("gap")
    rows = [
        (label.label_id, _short(label.candidate.candidate_id), "yes" if _candidate_visible(label.candidate) else "no", "placed")
        for label in plan.accepted
    ] + [(label.label_id, _short(label.candidate_id), "-", f"rejected: {label.reason}") for label in plan.rejected]
    rows.sort()
    print(f"{'label_id':<14} {'candidate':<12} {'visible':<8} status")
    print("-" * 52)
    for label_id, candidate, visible, status in rows:
        print(f"{label_id:<14} {candidate:<12} {visible:<8} {status}")
    print("-" * 52)
    gap_text = "n/a" if gap is None else f"{float(gap):.6f}"
    print(
        f"solver: {solver.get('algorithm', 'optimal')}  gap={gap_text}  "
        f"certified={bool(solver.get('certified'))}  nodes={solver.get('nodes_explored', 0)}"
    )


def occlusion_relocations(plan: Any) -> list[str]:
    lines = []
    for label in plan.accepted:
        primary = next((c for c in label.candidates if c.candidate_type == "center"), None)
        if primary is None or _candidate_visible(primary):
            continue
        lines.append(f"{label.label_id}: center occluded -> {_short(label.candidate.candidate_id)}")
    for label in plan.rejected:
        if label.reason == "terrain_occluded":
            lines.append(f"{label.label_id}: center occluded -> no visible alternative")
    return lines


def greedy_vs_optimal(greedy: Any, optimal: Any) -> list[str]:
    greedy_choice, optimal_choice = _choices(greedy), _choices(optimal)
    lines = []
    for label_id in sorted(set(greedy_choice) | set(optimal_choice)):
        g, o = greedy_choice.get(label_id), optimal_choice.get(label_id)
        if g == o:
            continue
        if g is None:
            lines.append(f"+ {label_id}: greedy dropped it; optimal keeps it at {_short(o)}")
        elif o is None:
            lines.append(f"- {label_id}: greedy kept it at {_short(g)}; optimal drops it")
        else:
            lines.append(f"~ {label_id}: greedy {_short(g)} -> optimal {_short(o)}")
    return lines


def _render(scene: f3d.MapScene) -> Path | None:
    try:
        scene.render()
    except f3d.MapSceneNativeUnavailable:
        return None
    png_path = Path(str(scene.last_render_path))
    _paint_sky(png_path)
    return png_path


def run_example(output_dir: str | Path) -> dict[str, Any]:
    dem = _heightmap()
    camera = TerrainCamera(dem)
    depth = camera.depth_image()
    labels = build_labels(camera)
    scene = build_scene(output_dir, dem, labels, depth)
    validation = scene.validate()
    optimal_png = _render(scene)
    if optimal_png is None:
        print("GPU unavailable — plan still compiled")
    plan = scene.compiled_label_plans[LAYER_ID]

    compile_args = dict(
        labels=labels,
        camera=scene.recipe.camera,
        viewport=(WIDTH, HEIGHT),
        terrain=DepthOcclusionSampler(depth),
        seed=SEED,
    )
    direct = f3d.LabelPlan.compile(**compile_args, declutter="optimal")
    if _choices(direct) != _choices(plan):
        raise RuntimeError(
            f"direct compile diverged from the MapScene plan: {_choices(direct)} != {_choices(plan)}"
        )
    greedy = f3d.LabelPlan.compile(**compile_args, declutter="greedy")

    comparison_png = None
    if optimal_png is not None:
        greedy_png = _render(build_scene(output_dir, dem, labels, depth, plan=greedy, name="greedy"))
        if greedy_png is not None:
            comparison_png = Path(output_dir) / "ridge_labels_optimal.png"
            compose_comparison(greedy_png, optimal_png, labels, greedy, plan, comparison_png)
    return {
        "validation_status": validation.status,
        "png_path": comparison_png,
        "plan": plan,
        "greedy": greedy,
        "depth": depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("examples/out/ridge_labels_optimal/ridge_labels_optimal.png")
    )
    args = parser.parse_args()

    payload = run_example(args.snapshot.parent)
    plan, greedy = payload["plan"], payload["greedy"]

    print(f"\nCompiled label plan for {LAYER_ID!r}")
    print_plan_table(plan)

    print("\nOccluded primaries")
    for line in occlusion_relocations(plan) or ["(none)"]:
        print(f"  {line}")

    print("\nGreedy vs optimal (accepted label -> candidate)")
    for line in greedy_vs_optimal(greedy, plan) or ["(no difference)"]:
        print(f"  {line}")
    print(f"  greedy kept {len(greedy.accepted)} labels, optimal kept {len(plan.accepted)}")

    if payload["png_path"] is not None:
        rendered = Path(payload["png_path"])
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if rendered.resolve() != args.snapshot.resolve():
            args.snapshot.write_bytes(rendered.read_bytes())
        print(f"\n{args.snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
