# tests/test_flythrough_popping.py
# TESSELLA win 5: "Zero popping, zero cracks."
#
# Spec gate (docs/prompts/fable5-moonshots/19-tessella.md:79):
#   over a committed 600-frame flythrough, max dE2000 between consecutive frames
#   < 1.0 (no tile pop, no LOD snap, no fallback flash), and a
#   depth-discontinuity crack detector reports 0 cracks across clipmap ring
#   boundaries.
#
# What changed here, and why
# --------------------------
# The previous version of this file rendered 600 frames at 96x64 of a 64x64
# downsample of a band-limited analytic surface, with the camera one metre above
# a terrain whose total relief was 1.2 m on a 100 km span, moving 200 m in total
# -- less than the 195.3 m the clipmap needs before it regenerates at all
# (src/terrain/clipmap/level.rs:168-171). The clipmap therefore rebuilt at most
# once across the whole run, no ring boundary was ever on screen, and
# ``seam_stats()`` was sampled exactly once, after the loop. Both gates were
# close to tautologies.
#
# This version:
#   * renders at 1280x720 -- 150x the pixel count (see WALL CLOCK below);
#   * drives a non-separable 6-octave fBm height field (``fractal_dem``) whose
#     finest octave sits exactly at ring 1's Nyquist -- 8 base cells, against
#     ``_steep_dem``'s ~51 -- instead of one separable sinusoid pair;
#   * frames three clipmap regions (centre block, ring 0, ring 1) with two ring
#     boundaries permanently on screen, and *computes* that from the Rust ring
#     formulas rather than asserting it in prose;
#   * moves the clipmap/streaming centre 353.6 m every frame -- 1.81x the
#     ``base_cell_size * 0.5`` regeneration threshold -- so the mesh, the seam
#     analysis and the LOD tiling are rebuilt on every one of the 600 frames;
#   * asserts the crack metric, the seam-gap headroom and a threshold-free hole
#     count on EVERY frame, not once at the end;
#   * warms the height mosaic to asserted full fine-tile residency before the
#     measured run, so a fallback during the 600 frames is a defect rather than
#     the streamer's start-up transient;
#   * proves the crack metric is a measurement and not a constant (a control at
#     the API's maximum relief must make the same detector fire), and proves the
#     dE2000 gate discriminates at this resolution and DEM (a control translation
#     must exceed it).
#
# The relief control was DEAD until 2026-07-29: it asked for z_scale 1200
# against a 0.1-50.0 API ceiling, so it raised ValueError before rendering and
# no run had ever shown the crack detector firing. At the ceiling (z_scale 50)
# it now reports 302 cracks and max_gap 2.209 against a 0.1 threshold, versus
# 0 cracks / max_gap 0.053 at Z_SCALE.
#
# WALL CLOCK
# ----------
# ci.yml:1209 gives ``test-tessella-gpu`` 90 minutes shared by six gate files,
# the wheel install, the adapter probe and the 1,000-camera Rust differential.
# This file is budgeted 600 s. Per-frame model on the self-hosted RTX 3070:
#   render + readback at 1280x720 .................... ~30 ms  -> 18 s
#   dE2000 over 921,600 px (tests/_deltae.py) ........ ~220 ms -> 132 s
#   coarse-AO recompute over the 256^2 heightmap,
#     which is uncached and runs on every render
#     (src/terrain/renderer/resources/ao.rs:17-111,
#      called unconditionally from draw/setup/context.rs:122);
#     256^2 * 288 iterations with a sqrt and a conditional
#     atan each ................................... ~90 ms  -> 54 s
#   clipmap rebuild + CPU visibility oracle .......... ~20 ms  -> 12 s
#   hole mask + seam bookkeeping ..................... ~12 ms  -> 7 s
#                                                              ~= 225 s
# plus 5 control renders. MEASURED 2026-07-29 on the reference RTX 3070: 793 ms
# per frame end to end, i.e. ~476 s for 600 frames -- 2.1x the model above, not
# 225 s. It still fits the 600 s budget, but with 1.26x headroom rather than
# 2.5x. ``wall_clock_s`` is recorded in the evidence; if the measured value
# exceeds 400 s the knob to turn is DEM_SIZE (which is what the AO term scales
# with), never the render resolution and never a gate.
#
# THE POP GATE IS RED, AND WHY (measured 2026-07-29, reference RTX 3070)
# ---------------------------------------------------------------------
# ``test_600_frame_streaming_flythrough_has_no_pop_or_crack`` fails at
# transition 1 with max dE2000 = 36.0149. That number is NOT a tile pop. Two
# independent mechanisms produce it, and both are outside this file:
#
# 1. The camera is tied to the clipmap centre (``cam_target = center + DX``),
#    so every frame translates the whole image by CENTER_STEP_LENGTH_M /
#    GROUND_PIXEL_M = 353.6 / 27.61 = 12.8 px. A max-pixel dE2000 over a
#    translating image measures translation, not popping. The fixture is in
#    fact self-contradictory as committed: the pop control asserts a 16 px
#    translation must score >= 1.0 while the gate asserts the flythrough's own
#    12.8 px translation must score < 1.0.
#    The clipmap mesh centre follows the STREAMING centre, not the camera
#    (``renderer/streaming.rs:636``, consumed at ``renderer/geometry.rs:601``),
#    so the confound is removable: hold cam_target fixed and sweep only
#    ``stream_height_tiles``. Measured that way over 40 frames the mesh still
#    rebuilds on every frame (39/39 seam-signature changes) and max dE2000 is
#    2.0554 -- a real, previously invisible pop.
#
# 2. That residual 2.0554 is itself an artifact of the fixture's colour ramp.
#    ``_build_overlay``'s 4-stop terrain colormap renders a HARD colour step:
#    adjacent pixels differ by up to 35/255 (green [110,144,110] against pink
#    [145,111,111], with the green channel moving OPPOSITE to the ramp's own
#    direction), while a 4-stop GREY colormap at the identical stop positions
#    is smooth (max adjacent delta 2) and a 2-stop terrain colormap is smooth
#    (3). Through that step, the underlying geometric pop -- max |dRGB| = 1,
#    i.e. at the 8-bit quantization floor -- is amplified to 5 levels and
#    dE 2.0554. With a continuous ramp the same 40 frames score 0.3127 (grey)
#    and 0.8275 (2-stop warm).
#
# So the clipmap/geomorph machinery is stable to ~1 LSB frame to frame; the gate
# is red because of a camera/centre coupling in this file and a colormap-LUT
# discontinuity in `src/colormap/` + the terrain overlay sampling path. Fixing
# either in isolation is not enough and neither belongs to this file alone, so
# the number is reported rather than papered over. Reproducers live in the
# TESSELLA handoff, not in the tree.
#
# RELIEF CEILING (real, and load-bearing for the numbers below)
# -------------------------------------------------------------
# ``analyze_depth_discontinuities`` measures the world-space disagreement
# between the fine surface and the coarse edge it T-junctions into, scaled by
# ``z_scale`` (geomorph.rs:305,326), and compares it against
# ``(terrain_span * 1e-6).max(0.001)`` (geometry.rs:631). At a 100 km span that
# is a demand for 0.1 m agreement. Nothing in the shipped geomorph makes the two
# surfaces agree: the centre block's vertices are emitted with morph weight 0
# (ring.rs:27-32) so the centre/ring-0 boundary is never blended at all, and for
# ring-to-ring boundaries the coarse grid the shader snaps to is
# ``2**(ring+1)`` height texels (terrain_pbr_pom.wgsl:4579), which for a texel
# the size of ``base_cell`` is the ring's OWN vertex spacing rather than its
# coarser neighbour's. The gap is therefore just the height field's deviation
# from linearity over the coarse spacing, and it scales linearly with both
# roughness and relief -- so the gate as shipped caps relief at roughly 1e-5 of
# the terrain span, which is why ``Z_SCALE`` stays at the 1.2 m the gate was
# already green with rather than becoming a realistic exaggeration.
#
# Rather than guess where that cliff is, the fixture is budgeted against the one
# empirical data point available: the ``_steep_dem`` surface this gate is already
# green with. ``seam_roughness`` measures, on the real arrays, exactly the
# quantity the detector accumulates, and the new fixture must come in at most
# 0.75x the legacy value at every ring boundary -- while carrying six octaves
# instead of one. ``seam_gap_headroom_ratio`` in the evidence then makes the
# achieved runtime margin auditable instead of invisible.

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _deltae import delta_e_2000, srgb_to_lab
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from _terrain_flythrough import (
    LEGACY_GATE_Z_SCALE,
    background_pixel_count,
    build_overlay,
    clipmap_base_cell_size,
    clipmap_boundary_half_spacings,
    clipmap_covered_positive_x_limit,
    clipmap_region_outer_radii,
    clipmap_region_vertex_spacing,
    flythrough_params,
    fractal_dem,
    fractal_dem_octave_amplitudes,
    legacy_gate_dem,
    render_rgba,
    seam_gap_threshold,
    seam_roughness,
    seam_signature,
    triangle_wave,
)
from forge3d.diagnostics import (
    render_certificate,
    seam_stats,
    visibility_stats,
    vt_stats,
)

requires_terrain = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)

# --- committed flythrough configuration ------------------------------------
MODE = "clipmap:4:32:32:10:0.3"
RING_COUNT = 4
RING_RESOLUTION = 32
CENTER_RESOLUTION = 32
SPAN = 100_000.0
SIZE = (1280, 720)
FRAMES = 600
FOV_Y_DEG = 45.0

DEM_SIZE = 256
DEM_OCTAVES = 6
DEM_GAIN = 0.45
DEM_SEED = 1904
Z_SCALE = 1.2

# Exact nadir for the Y-up orbit clipmap modes use (`is_zup_camera_mode` is
# false for "clipmap:*", so upload.rs:362-370 applies: theta=90/phi=90 puts the
# eye at target + (0, 0, cam_radius) with up = +Y). A nadir camera makes the
# ground footprint an exactly computable axis-aligned rectangle, which is what
# lets the zero-hole assertion be a gate instead of a coin flip, and removes
# self-occlusion silhouettes as a confound in the dE2000 metric.
CAM_THETA_DEG = 90.0
CAM_PHI_DEG = 90.0
CAM_RADIUS = 24_000.0
# The clipmap rings do not cover the +x corners beyond the centre block's
# half-extent (see clipmap_covered_positive_x_limit); offsetting the look-at
# point keeps the whole frame inside the covered region.
CAM_TARGET_DX = -12_000.0
CLIP = (18_000.0, 40_000.0)

CENTER_STEP_M = 250.0
CENTER_X_AMPLITUDE_STEPS = 19
CENTER_Y_AMPLITUDE_STEPS = 23
MIN_DISTINCT_CENTERS = 500
STREAM_ALTITUDE_M = 3_000.0

STREAM_LOD = 3
STREAM_TILE_RESOLUTION = 32
STREAM_MAX_RESIDENT_BYTES = 1024 * 1024
STREAM_TOTAL_TILES = (2**STREAM_LOD) ** 2  # 64 tiles x 32^2 texels = the 256^2 DEM
MAX_WARMUP_STEPS = 400

# Controls.
# ``TerrainRenderParams`` rejects z_scale outside 0.1-50.0
# (python/forge3d/terrain_params.py:2037), so the relief control has to live
# inside that range. The previous value (1000.0) put z_scale at 1200 and the
# control raised ValueError before it rendered anything -- it had never
# executed, so nothing had ever demonstrated that the crack detector CAN fire.
# 50.0 is the largest relief the public API admits -- 41.7x Z_SCALE, and 33x
# the gap the detector already measures there. It is stated as an absolute
# z_scale rather than a factor so no float product can drift past the ceiling.
RELIEF_CONTROL_Z_SCALE = 50.0
POP_CONTROL_SHIFT_PX = (1.0, 4.0, 16.0)
POP_CONTROL_ASSERTED_PX = 16.0

# Non-vacuity minimums.
MIN_REGIONS_ON_SCREEN = 3
MIN_REBUILD_SIGNATURE_CHANGES = 540  # >= 0.9 * (FRAMES - 1)
# The new fixture must stay under this fraction of the roughness the gate is
# already green with (see legacy_gate_dem), measured in the exact quantity the
# crack detector accumulates. It buys margin without softening any gate.
SEAM_ROUGHNESS_BUDGET_FRACTION = 0.75

# --- derived framing (pure arithmetic on the constants above) --------------
HALF_HEIGHT_M = CAM_RADIUS * math.tan(math.radians(FOV_Y_DEG) * 0.5)
HALF_WIDTH_M = HALF_HEIGHT_M * (SIZE[0] / SIZE[1])
GROUND_PIXEL_M = 2.0 * HALF_WIDTH_M / SIZE[0]
FRAME_MIN_X = CAM_TARGET_DX - HALF_WIDTH_M
FRAME_MAX_X = CAM_TARGET_DX + HALF_WIDTH_M
FRAME_MAX_ABS_Y = HALF_HEIGHT_M
BASE_CELL_M = clipmap_base_cell_size(SPAN, CENTER_RESOLUTION)
REGENERATION_THRESHOLD_M = BASE_CELL_M * 0.5
CENTER_STEP_LENGTH_M = CENTER_STEP_M * math.sqrt(2.0)
SEAM_THRESHOLD = seam_gap_threshold(SPAN)


def _region_outer_radii() -> list[float]:
    return clipmap_region_outer_radii(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )


def _regions_on_screen() -> int:
    """Clipmap regions the committed frame intersects.

    The frame is an axis-aligned rectangle around the look-at point and the
    regions are concentric squares around the clipmap centre, so a region
    boundary is on screen exactly when its radius is inside the frame's
    Chebyshev reach.
    """
    reach = max(abs(FRAME_MIN_X), abs(FRAME_MAX_X), FRAME_MAX_ABS_Y)
    boundaries = _region_outer_radii()[:-1]
    return 1 + sum(1 for radius in boundaries if radius < reach)


def _center_at(index: int) -> tuple[float, float]:
    """Clipmap/streaming centre for frame ``index``.

    Two integer triangle waves of period 76 and 92 frames, which visit 511
    distinct centres over the run while keeping the path inside a
    +/-4750 x +/-5750 m box -- small enough that the framed ground never leaves
    the DEM footprint, large enough that every single step is exactly
    ``CENTER_STEP_LENGTH_M = 353.6 m``, i.e. 1.81x the clipmap's
    ``base_cell_size * 0.5 = 195.3 m`` regeneration threshold
    (``src/terrain/clipmap/level.rs:168-171``). No two consecutive frames ever
    share a centre, so the clipmap geometry cache key
    (``geometry.rs:603-615``) misses on every frame of the run.
    """
    return (
        triangle_wave(index, CENTER_X_AMPLITUDE_STEPS) * CENTER_STEP_M,
        triangle_wave(index, CENTER_Y_AMPLITUDE_STEPS) * CENTER_STEP_M,
    )


def _params(center: tuple[float, float], *, z_scale: float, overlay, shading="forward"):
    return flythrough_params(
        size_px=SIZE,
        terrain_span=SPAN,
        camera_mode=MODE,
        cam_radius=CAM_RADIUS,
        cam_target=(center[0] + CAM_TARGET_DX, center[1], 0.0),
        theta_deg=CAM_THETA_DEG,
        phi_deg=CAM_PHI_DEG,
        fov_y_deg=FOV_Y_DEG,
        z_scale=z_scale,
        clip=CLIP,
        overlay=overlay,
        shading=shading,
    )


def _enable_streaming(renderer, dem: np.ndarray) -> None:
    renderer.enable_height_streaming(
        terrain_extent_m=SPAN,
        ring_count=RING_COUNT,
        ring_resolution=RING_RESOLUTION,
        lod=STREAM_LOD,
        tile_resolution=STREAM_TILE_RESOLUTION,
        max_in_flight=32,
        pool_size=4,
        dem=dem,
        coarse_prefill=True,
        max_resident_bytes=STREAM_MAX_RESIDENT_BYTES,
    )


def _warm_streaming_to_full_residency(renderer, center: tuple[float, float]) -> int:
    """Drive the streamer to full fine-tile residency before a measured run.

    The coarse-prefill -> fine-tile transition is a start-up transient of the
    streamer, not something a flythrough does: with the whole 256 KiB working set
    inside ``STREAM_MAX_RESIDENT_BYTES`` nothing can be evicted once resident, so
    a fallback during the measured frames would be a genuine defect while one
    during warm-up is just the mosaic filling in. Residency is asserted here, not
    assumed, and the frames the gate measures all start from it.
    """
    steps = 0
    stream = renderer.stream_height_tiles(
        (center[0], STREAM_ALTITUDE_M, center[1]), max_uploads=STREAM_TOTAL_TILES
    )
    while (
        stream["resident_fine_tiles"] < stream["total_tiles"]
        or stream["loader_pending"] > 0
    ) and steps < MAX_WARMUP_STEPS:
        stream = renderer.stream_height_tiles(
            (center[0], STREAM_ALTITUDE_M, center[1]),
            max_uploads=STREAM_TOTAL_TILES,
        )
        steps += 1
    assert stream["total_tiles"] == STREAM_TOTAL_TILES, stream
    assert stream["resident_fine_tiles"] == stream["total_tiles"], stream
    assert stream["loader_pending"] == 0, stream
    return steps


def _assert_seams_clean(seams: dict, where: str) -> None:
    """The spec's crack gate, applied to one published geometry build."""
    assert seams["depth_sample_count"] > 0, (where, seams)
    assert seams["crack_count"] == 0, (where, seams)
    assert seams["seams_valid"] is True, (where, seams)
    assert seams["max_gap"] <= SEAM_THRESHOLD, (where, seams, SEAM_THRESHOLD)


# ---------------------------------------------------------------------------
# CPU-only: the fixture's design budget, asserted rather than commented
# ---------------------------------------------------------------------------


def test_fractal_dem_is_deterministic_and_inside_the_seam_budget():
    first = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    second = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.shape == (DEM_SIZE, DEM_SIZE)
    assert float(first.min()) == 0.0
    assert float(first.max()) == 1.0

    # Six octaves, each materially present: a surface that would quietly
    # collapse back to the old single-sinusoid case if a later edit dropped the
    # fine ones.
    amplitudes = fractal_dem_octave_amplitudes(
        DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED
    )
    assert len(amplitudes) == DEM_OCTAVES
    assert min(amplitudes) > 0.005, amplitudes
    assert amplitudes[0] > amplitudes[-1], amplitudes

    # One DEM texel is one clipmap base cell, so the height field carries no
    # content the finest mesh cannot represent...
    assert math.isclose(SPAN / (DEM_SIZE - 1), BASE_CELL_M, rel_tol=0.01), (
        SPAN / (DEM_SIZE - 1),
        BASE_CELL_M,
    )
    # ...and its finest octave sits exactly at ring 1's Nyquist, which is the
    # scale a LOD step destroys and geomorph is supposed to hide. `_steep_dem`'s
    # finest content is ~51 base cells wide; this is 8.
    finest_wavelength_m = SPAN / float(2 ** (DEM_OCTAVES - 1))
    ring1_spacing = clipmap_region_vertex_spacing(SPAN, CENTER_RESOLUTION, 2)
    assert finest_wavelength_m == pytest.approx(2.0 * ring1_spacing)
    assert finest_wavelength_m <= 8.0 * BASE_CELL_M

    # Roughness budget: the shipped crack detector demands the fine and coarse
    # surfaces agree to within one part per million of the terrain span, and
    # nothing in the geomorph makes them agree (see the module docstring), so
    # what it really bounds is the height field's deviation from linearity over
    # the coarse spacing. Rather than guess where that cliff is, stay measurably
    # under the fixture the gate is ALREADY green with.
    half_spacings = clipmap_boundary_half_spacings(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )
    assert half_spacings == pytest.approx([390.625, 781.25, 1562.5, 3125.0])
    measured = seam_roughness(first, SPAN, half_spacings, Z_SCALE)
    budget = seam_roughness(
        legacy_gate_dem(), SPAN, half_spacings, LEGACY_GATE_Z_SCALE
    )
    assert measured <= budget * SEAM_ROUGHNESS_BUDGET_FRACTION, {
        "boundary_half_spacings_m": half_spacings,
        "seam_roughness": measured,
        "legacy_seam_roughness": budget,
        "fraction_of_legacy": measured / budget,
        "seam_gap_threshold": SEAM_THRESHOLD,
        "note": (
            "raising DEM_OCTAVES/DEM_GAIN/Z_SCALE past this point takes the "
            "fixture past the roughness the shipped seam machinery is known to "
            "hold; fix the geomorph (centre-block morph weight, coarse-grid "
            "power of two) before raising them"
        ),
    }


def test_committed_camera_frames_multiple_clipmap_regions():
    radii = _region_outer_radii()
    assert radii[:3] == pytest.approx([6250.0, 18750.0, 43750.0])

    # Every pixel is terrain: the frame's right edge stays inside the +x limit
    # that make_ring's short strips actually cover.
    covered = clipmap_covered_positive_x_limit(SPAN, CENTER_RESOLUTION)
    assert FRAME_MAX_X < covered, (FRAME_MAX_X, covered)

    # Two ring boundaries permanently on screen -> three regions.
    assert _regions_on_screen() >= MIN_REGIONS_ON_SCREEN, {
        "frame_x": (FRAME_MIN_X, FRAME_MAX_X),
        "frame_abs_y": FRAME_MAX_ABS_Y,
        "region_outer_radii": radii,
    }
    assert FRAME_MAX_ABS_Y > radii[0], (FRAME_MAX_ABS_Y, radii[0])
    assert FRAME_MIN_X < -radii[1], (FRAME_MIN_X, radii[1])

    # Every frame regenerates the clipmap: the streaming centre step clears
    # ClipmapLevel::update_center's half-cell threshold with margin.
    assert CENTER_STEP_LENGTH_M > REGENERATION_THRESHOLD_M, (
        CENTER_STEP_LENGTH_M,
        REGENERATION_THRESHOLD_M,
    )

    # The framed ground never leaves the DEM footprint, so no ring boundary on
    # screen is silently flattened by the UV clamp.
    max_center_x = CENTER_X_AMPLITUDE_STEPS * CENTER_STEP_M
    max_center_y = CENTER_Y_AMPLITUDE_STEPS * CENTER_STEP_M
    assert max_center_x + abs(FRAME_MIN_X) < SPAN * 0.5
    assert max_center_y + FRAME_MAX_ABS_Y < SPAN * 0.5

    # The relief control must be constructible. It previously was not: the
    # control asked for z_scale 1200 against an API ceiling of 50.0, so it died
    # in ValueError before rendering and the crack detector had never been shown
    # to fire at all. Asserting it here makes that a fast CPU failure instead of
    # something only the GPU lane can notice.
    assert 0.1 <= RELIEF_CONTROL_Z_SCALE <= 50.0, RELIEF_CONTROL_Z_SCALE
    assert RELIEF_CONTROL_Z_SCALE > Z_SCALE * 10.0, (
        RELIEF_CONTROL_Z_SCALE,
        Z_SCALE,
    )

    # Consecutive centres always differ by the full step, and the run covers a
    # large set of distinct positions rather than oscillating between a few.
    steps = [
        math.dist(_center_at(index), _center_at(index + 1))
        for index in range(FRAMES - 1)
    ]
    assert min(steps) == pytest.approx(CENTER_STEP_LENGTH_M)
    assert max(steps) == pytest.approx(CENTER_STEP_LENGTH_M)
    centers = {_center_at(index) for index in range(FRAMES)}
    assert len(centers) >= MIN_DISTINCT_CENTERS, len(centers)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.gpu_lane
@requires_terrain
def test_600_frame_streaming_flythrough_has_no_pop_or_crack():
    started = time.perf_counter()
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    half_spacings = clipmap_boundary_half_spacings(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )

    max_delta_e = 0.0
    max_seam_gap = 0.0
    max_crack_count = 0
    total_crack_count = 0
    min_depth_samples = None
    hole_pixels_total = 0
    signature_changes = 0
    previous_signature = None
    previous_lab = None

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        warmup_steps = _warm_streaming_to_full_residency(renderer, _center_at(0))

        for index in range(FRAMES):
            center = _center_at(index)
            renderer.stream_height_tiles(
                (center[0], STREAM_ALTITUDE_M, center[1]), max_uploads=8
            )
            frame = render_rgba(
                renderer,
                _params(center, z_scale=Z_SCALE, overlay=overlay),
                dem,
                ibl,
                material_set,
            )

            # Per-frame crack gate on the geometry build this frame drew.
            seams = seam_stats()
            _assert_seams_clean(seams, f"frame {index}")
            max_seam_gap = max(max_seam_gap, float(seams["max_gap"]))
            max_crack_count = max(max_crack_count, int(seams["crack_count"]))
            total_crack_count += int(seams["crack_count"])
            samples = int(seams["depth_sample_count"])
            min_depth_samples = (
                samples if min_depth_samples is None else min(min_depth_samples, samples)
            )
            signature = seam_signature(seams)
            if previous_signature is not None and signature != previous_signature:
                signature_changes += 1
            previous_signature = signature

            # Per-frame threshold-free hole gate on the shipped image.
            holes = background_pixel_count(frame)
            assert holes == 0, {"frame": index, "background_pixels": holes}
            hole_pixels_total += holes

            # Per-transition pop gate.
            lab = srgb_to_lab(frame[..., :3])
            if previous_lab is not None:
                delta = float(delta_e_2000(previous_lab, lab).max())
                assert delta < 1.0, {"transition": index, "max_delta_e_2000": delta}
                max_delta_e = max(max_delta_e, delta)
            previous_lab = lab

        height_vt = vt_stats()
        certificate = render_certificate(sign=False)

    # Run-level non-vacuity: the mesh really was rebuilt on (nearly) every frame
    # with new geometry, so "0 cracks, 600 times" is 600 measurements and not one
    # cached answer replayed.
    assert signature_changes >= MIN_REBUILD_SIGNATURE_CHANGES, {
        "seam_signature_changes": signature_changes,
        "transitions": FRAMES - 1,
    }
    assert height_vt["resident_tiles_height"] > 0, height_vt
    assert height_vt["height_pending_requests"] == 0, height_vt

    degraded = {
        str(entry.get("name")) for entry in certificate.get("degradations", []) or []
    }
    assert "terrain_visibility_buffer" not in degraded, degraded
    assert "terrain_hzb_two_phase" not in degraded, degraded

    wall_clock_s = time.perf_counter() - started
    record_tessella_result(
        "flythrough_popping",
        {
            # Field names the win-5 row of scripts/tessella_evidence_report.py
            # renders; everything after them is extra context.
            "frames": FRAMES,
            "width": SIZE[0],
            "height": SIZE[1],
            "worst_frame_crack_count": max_crack_count,
            "crack_count": total_crack_count,
            "depth_sample_count": int(min_depth_samples or 0),
            "frames_crack_checked": FRAMES,
            "dem_size": DEM_SIZE,
            "dem_octaves": DEM_OCTAVES,
            "terrain_span_m": SPAN,
            "z_scale": Z_SCALE,
            "clipmap_center_step_m": CENTER_STEP_LENGTH_M,
            "clipmap_regeneration_threshold_m": REGENERATION_THRESHOLD_M,
            "clipmap_center_path_m": CENTER_STEP_LENGTH_M * (FRAMES - 1),
            "distinct_clipmap_centers": len(
                {_center_at(index) for index in range(FRAMES)}
            ),
            "regions_on_screen": _regions_on_screen(),
            "region_outer_radii_m": _region_outer_radii(),
            "ground_pixel_m": GROUND_PIXEL_M,
            "max_delta_e_2000": max_delta_e,
            "max_seam_gap": max_seam_gap,
            "seam_gap_threshold": SEAM_THRESHOLD,
            "seam_gap_headroom_ratio": SEAM_THRESHOLD / max(max_seam_gap, 1e-12),
            "seam_roughness": seam_roughness(dem, SPAN, half_spacings, Z_SCALE),
            "legacy_seam_roughness": seam_roughness(
                legacy_gate_dem(), SPAN, half_spacings, LEGACY_GATE_Z_SCALE
            ),
            "seam_signature_changes": signature_changes,
            "hole_pixels_total": hole_pixels_total,
            "resident_height_tiles": int(height_vt["resident_tiles_height"]),
            "height_pending_requests": int(height_vt["height_pending_requests"]),
            "streaming_warmup_steps": warmup_steps,
            "streaming_total_tiles": STREAM_TOTAL_TILES,
            "wall_clock_s": wall_clock_s,
        },
    )


# ---------------------------------------------------------------------------
# Negative controls: both gates must be able to fail
# ---------------------------------------------------------------------------


@pytest.mark.gpu_lane
@requires_terrain
def test_crack_detector_fires_when_the_seams_actually_separate():
    """The crack metric is a measurement, not a constant.

    Same camera, DEM, resolution and clipmap mode; only the vertical
    exaggeration changes. ``analyze_depth_discontinuities`` scales the fine/coarse
    disagreement by ``z_scale`` (geomorph.rs:305), so a relief the shipped
    geomorph cannot hold must make the identical detector report cracks. A
    detector that cannot fail proves nothing about the frames where it stays
    silent.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        _warm_streaming_to_full_residency(renderer, center)

        render_rgba(
            renderer,
            _params(center, z_scale=Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )
        clean = seam_stats()

        render_rgba(
            renderer,
            _params(center, z_scale=RELIEF_CONTROL_Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )
        separated = seam_stats()

    _assert_seams_clean(clean, "relief control baseline")
    assert separated["crack_count"] > 0, separated
    assert separated["max_gap"] > SEAM_THRESHOLD, (separated, SEAM_THRESHOLD)
    assert separated["seams_valid"] is False, separated
    assert separated["max_gap"] > clean["max_gap"] * 10.0, (clean, separated)

    record_tessella_result(
        "flythrough_crack_detector_control",
        {
            "z_scale": Z_SCALE,
            "control_z_scale": RELIEF_CONTROL_Z_SCALE,
            "seam_gap_threshold": SEAM_THRESHOLD,
            "max_gap": float(clean["max_gap"]),
            "control_max_gap": float(separated["max_gap"]),
            "crack_count": int(clean["crack_count"]),
            "control_crack_count": int(separated["crack_count"]),
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_pop_gate_discriminates_at_this_resolution_and_dem():
    """The dE2000 < 1.0 threshold is discriminating for this configuration.

    One JND at 1280x720 over this DEM is a small quantity; this control measures
    what a known image change of 1, 4 and 16 ground pixels actually scores, and
    asserts the largest of them clears the gate. Without it, "max dE2000 < 1.0"
    could just mean the scene never changed.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)
    measured: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        _warm_streaming_to_full_residency(renderer, center)

        reference_lab = srgb_to_lab(
            render_rgba(
                renderer,
                _params(center, z_scale=Z_SCALE, overlay=overlay),
                dem,
                ibl,
                material_set,
            )[..., :3]
        )

        for shift_px in POP_CONTROL_SHIFT_PX:
            shifted_center = (center[0] + shift_px * GROUND_PIXEL_M, center[1])
            shifted = render_rgba(
                renderer,
                flythrough_params(
                    size_px=SIZE,
                    terrain_span=SPAN,
                    camera_mode=MODE,
                    cam_radius=CAM_RADIUS,
                    # Move only the look-at point: the clipmap centre stays put,
                    # so this is a pure image translation, not a re-tessellation.
                    cam_target=(shifted_center[0] + CAM_TARGET_DX, center[1], 0.0),
                    theta_deg=CAM_THETA_DEG,
                    phi_deg=CAM_PHI_DEG,
                    fov_y_deg=FOV_Y_DEG,
                    z_scale=Z_SCALE,
                    clip=CLIP,
                    overlay=overlay,
                ),
                dem,
                ibl,
                material_set,
            )
            measured[f"{shift_px:g}px"] = float(
                delta_e_2000(reference_lab, srgb_to_lab(shifted[..., :3])).max()
            )

    asserted = measured[f"{POP_CONTROL_ASSERTED_PX:g}px"]
    assert asserted >= 1.0, measured

    record_tessella_result(
        "flythrough_pop_gate_control",
        {
            "ground_pixel_m": GROUND_PIXEL_M,
            "control_max_delta_e_2000": measured,
            "asserted_shift_px": POP_CONTROL_ASSERTED_PX,
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_visibility_shading_is_identical_and_hole_free_at_flythrough_settings():
    """GPU-side confirmation of the image-side hole gate.

    ``visibility_stats()`` counts the resolve pass's background pixels on the
    shipped draw path. Win 3 only proves visibility == forward at 640x360 over
    ``_steep_dem(96)`` and without height streaming; the flythrough's zero-hole
    claim is only as good as that equivalence at ITS settings.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()

        forward_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _enable_streaming(forward_renderer, dem)
        # Both renderers must reach identical mosaics or the bitwise comparison
        # would be measuring streaming luck rather than shading equivalence.
        _warm_streaming_to_full_residency(forward_renderer, center)
        forward = render_rgba(
            forward_renderer,
            _params(center, z_scale=Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )

        visibility_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _enable_streaming(visibility_renderer, dem)
        _warm_streaming_to_full_residency(visibility_renderer, center)
        visibility = render_rgba(
            visibility_renderer,
            _params(center, z_scale=Z_SCALE, overlay=overlay, shading="visibility"),
            dem,
            ibl,
            material_set,
        )
        stats = visibility_stats()

    np.testing.assert_array_equal(visibility, forward)
    covered = stats["visible_pixels"] + stats["background_pixels"]
    assert covered == SIZE[0] * SIZE[1], stats
    assert stats["background_pixels"] == 0, stats
    assert background_pixel_count(forward) == 0

    record_tessella_result(
        "flythrough_visibility_coverage",
        {
            "render_size_px": list(SIZE),
            "visible_pixels": int(stats["visible_pixels"]),
            "background_pixels": int(stats["background_pixels"]),
            "image_background_pixels": int(background_pixel_count(forward)),
            "bitwise_identical_to_forward": True,
        },
    )
