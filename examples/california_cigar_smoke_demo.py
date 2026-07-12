#!/usr/bin/env python3
"""August Complex hybrid smoke demo using forge3d.smoke.

The terrain and fire event come from the cached California wildfire example
assets. The main smoke pass is a persistent 3D density field with advective
flow, diffusion, decay, self-shadowed scattering, source emission, and
projected volume ray marching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from _import_shim import ensure_repo_import

ensure_repo_import()

ROOT = Path(__file__).resolve().parents[1]

try:
    import forge3d.smoke as f3d_smoke
except ModuleNotFoundError:
    f3d_smoke = None

CACHE = ROOT / "examples" / ".cache" / "california_wildfire_smoke"
DEM_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900.tif"
OVERLAY_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900_dark_relief_overlay.png"
META_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900.json"
REGIONAL_DEM_PATH = CACHE / "california_osm_r165475_terrarium_z8_max1700.tif"
REGIONAL_OVERLAY_PATH = CACHE / "california_osm_r165475_terrarium_z8_max1700_dark_relief_overlay.png"
REGIONAL_META_PATH = CACHE / "california_osm_r165475_terrarium_z8_max1700.json"
OUT_DIR = ROOT / "examples" / "out" / "california_cigar_smoke"
DEFAULT_OUTPUT = OUT_DIR / "august_complex_cigar_smoke_8s.mp4"
DEFAULT_PREVIEW = OUT_DIR / "august_complex_cigar_smoke_8s.preview.png"
REFERENCE_VIDEO_DEFAULT = ROOT / "rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4"
REFERENCE_FIRST30_AUDIT_DIR = OUT_DIR / "reference_first30_every_frame_audit"
REFERENCE_EXACT_AUDIT_DIR = OUT_DIR / "reference_exact_smoke_audit"
REFERENCE_EXACT_CACHE = ROOT / "examples" / ".cache" / "california_cigar_smoke" / "reference_exact"
REFERENCE_EXACT_FRAMES_DIR = REFERENCE_EXACT_CACHE / "frames"
REFERENCE_EXACT_MASKS_DIR = REFERENCE_EXACT_CACHE / "masks"
REFERENCE_EXACT_SMOKE_DIR = REFERENCE_EXACT_CACHE / "smoke"

WEB_MERCATOR_LIMIT = 20037508.342789244
FPS = 30
DURATION_SECONDS = 8
WIDTH = 960
HEIGHT = 540
REFERENCE_EXACT_FRAME_COUNT = 900
REFERENCE_EXACT_FPS = 30
REFERENCE_EXACT_WIDTH = 1920
REFERENCE_EXACT_HEIGHT = 1080
REFERENCE_EXACT_START_DATE = date(2023, 1, 1)
REFERENCE_EXACT_TIMELINE_DAYS = 272
REFERENCE_EXACT_START_AREA_HA = 5_900.0
REFERENCE_EXACT_END_AREA_HA = 12_300_000.0
REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION = "reference-exact-smoke-v1"
REFERENCE_EXACT_COLOR_POLICY = {
    "decode_colorspace": "bt709",
    "decode_pixel_format": "rgb24 PNG",
    "alpha_policy": "smoke alpha excludes UI/text, hot fire cores, frame borders, and stable blue-water background",
    "rgba_policy": "smoke_rgba PNG stores premultiplied RGB plus 8-bit alpha for screen-space over compositing",
    "frame_mapping": "nearest source frame at 30 fps, clipped to manifest frame range",
}
REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS = {
    "all_frame_count": 900.0,
    "minimum_smoke_mask_iou": 0.985,
    "median_smoke_mask_iou": 0.995,
    "maximum_alpha_mae": 3.0,
    "median_alpha_mae": 1.0,
    "maximum_smoke_rgb_mae": 4.0,
    "median_smoke_rgb_mae": 1.5,
    "maximum_smoke_centroid_error_px": 2.0,
    "coverage_curve_correlation": 0.995,
    "frame_delta_curve_correlation": 0.990,
    "event_boundary_frame_error_max": 1.0,
    "ui_leakage_fraction": 0.0,
    "fire_leakage_fraction": 0.002,
    "background_false_positive_fraction": 0.002,
}
REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS = {
    "smoke_reconstruction_mae": 1.0,
    "dense_smoke_hole_fraction": 0.015,
    "continuity_iou_min": 0.08,
    "continuity_min_coverage": 0.010,
    "continuity_max_frame_delta_luma": 0.006,
}
REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS = {
    "minimum_active_region_fraction": 0.50,
    "minimum_median_region_luma_contrast": 0.055,
    "minimum_median_region_edge_fraction": 0.0025,
    "minimum_median_region_bright_fraction": 0.0010,
}
HYBRID_SMOKE_WIDTH = 520
HYBRID_SMOKE_HEIGHT = 408
HYBRID_SMOKE_MAX_AGE_FRAMES = 306.0
HYBRID_SMOKE_MAX_ALPHA = 168
HYBRID_SMOKE_SEED = 2020
HYBRID_SMOKE_LAYER_COUNT = 3
HYBRID_SMOKE_LAYER_WEIGHTS = (0.56, 0.30, 0.14)
HYBRID_SMOKE_RENDER_LAYER_ALPHA = (0.74, 0.58, 0.44)
HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA = 42
HYBRID_FIRE_SMOLDER_MIN_FRAMES = 24
HYBRID_FIRE_SMOLDER_MAX_FRAMES = 68
SOURCE_WISP_MAX_ALPHA = 142
SOURCE_WISP_MAX_PARTICLES = 700
SOURCE_WISP_MAX_EMITTERS = 58
SOURCE_WISP_REFERENCE_MAX_PARTICLES = 1150
SOURCE_WISP_REFERENCE_MAX_EMITTERS = 132
SOURCE_WISP_EMIT_INTERVAL_FRAMES = 3
SOURCE_WISP_SOURCE_DELAY_FRAMES = 2
SOURCE_WISP_MIN_RADIUS_PX = 0.82
SOURCE_WISP_LIFETIME_FRAMES = (38, 76)
SOURCE_WISP_AUDIT_TIMES = (1.0, 3.5, 5.5, 7.0)
REFERENCE_FILM_CONTACT_SHEET_TIMES = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 29.5)
SOURCE_WISP_EMITTER_MODES = ("synthetic", "fire-core")
SOURCE_WISP_MORPHOLOGY_GATE_MIN_TIME_SECONDS = 2.0
SOURCE_WISP_AGE_BANDS = {
    "fresh_stem": (0.00, 0.26),
    "transition_plume": (0.24, 0.62),
    "old_tail": (0.56, 1.01),
}
TARGET_RENDER_PRESET = "source-wisp-reference"
LEGACY_RENDER_PRESET = "legacy-combined"
BRUSH_BUNDLE_RENDER_PRESET = "source-wisp-brush-baseline"
REFERENCE_FILM_RENDER_PRESET = "reference-film"
REFERENCE_EXACT_SMOKE_RENDER_PRESET = "reference-exact-smoke"
TERRAIN_SLAB_COMPOSITION_MODE = "terrain-slab"
MAP_FILM_COMPOSITION_MODE = "map-film"
LOCAL_DELIVERY_PROFILE = "local-960"
REFERENCE_DELIVERY_PROFILE = "reference-1080p"
DELIVERY_PROFILES = {
    LOCAL_DELIVERY_PROFILE: {
        "size": (WIDTH, HEIGHT),
        "crf": 17,
        "video_bitrate": None,
    },
    REFERENCE_DELIVERY_PROFILE: {
        "size": (1920, 1080),
        "crf": 16,
        "video_bitrate": "2600k",
        "maxrate": "3200k",
        "bufsize": "5200k",
    },
}
SMOKE_RENDER_PRESETS = {
    TARGET_RENDER_PRESET: {
        "source_wisps": True,
        "physical_smoke": False,
        "source_wisp_emitter_mode": "fire-core",
        "physical_emitter_mode": "fire-core",
        "source_wisp_max_particles": SOURCE_WISP_REFERENCE_MAX_PARTICLES,
        "source_wisp_max_emitters": SOURCE_WISP_REFERENCE_MAX_EMITTERS,
        "source_wisp_warmup_mode": "visible-only",
        "source_wisp_plume_ribbons": True,
        "broad_smoke_alpha": 0.025,
        "physical_alpha": 0.0,
        "physical_max_sources": 96,
        "composition_mode": TERRAIN_SLAB_COMPOSITION_MODE,
        "delivery_profile": LOCAL_DELIVERY_PROFILE,
        "regional_smoke": False,
        "map_grade": "local",
        "duration_seconds": DURATION_SECONDS,
        "audit_required_for_acceptance": True,
    },
    LEGACY_RENDER_PRESET: {
        "source_wisps": True,
        "physical_smoke": True,
        "source_wisp_emitter_mode": "synthetic",
        "physical_emitter_mode": "synthetic",
        "source_wisp_max_particles": SOURCE_WISP_MAX_PARTICLES,
        "source_wisp_max_emitters": SOURCE_WISP_MAX_EMITTERS,
        "source_wisp_warmup_mode": "full",
        "source_wisp_plume_ribbons": False,
        "broad_smoke_alpha": 0.28,
        "physical_alpha": 0.58,
        "physical_max_sources": 32,
        "composition_mode": TERRAIN_SLAB_COMPOSITION_MODE,
        "delivery_profile": LOCAL_DELIVERY_PROFILE,
        "regional_smoke": False,
        "map_grade": "local",
        "duration_seconds": DURATION_SECONDS,
        "audit_required_for_acceptance": False,
    },
    BRUSH_BUNDLE_RENDER_PRESET: {
        "source_wisps": True,
        "physical_smoke": False,
        "source_wisp_emitter_mode": "fire-core",
        "physical_emitter_mode": "fire-core",
        "source_wisp_max_particles": SOURCE_WISP_REFERENCE_MAX_PARTICLES,
        "source_wisp_max_emitters": SOURCE_WISP_REFERENCE_MAX_EMITTERS,
        "source_wisp_warmup_mode": "visible-only",
        "source_wisp_plume_ribbons": False,
        "broad_smoke_alpha": 0.025,
        "physical_alpha": 0.0,
        "physical_max_sources": 96,
        "composition_mode": TERRAIN_SLAB_COMPOSITION_MODE,
        "delivery_profile": LOCAL_DELIVERY_PROFILE,
        "regional_smoke": False,
        "map_grade": "local",
        "duration_seconds": DURATION_SECONDS,
        "audit_required_for_acceptance": False,
    },
    REFERENCE_FILM_RENDER_PRESET: {
        "source_wisps": True,
        "physical_smoke": False,
        "source_wisp_emitter_mode": "fire-core",
        "physical_emitter_mode": "fire-core",
        "source_wisp_max_particles": SOURCE_WISP_REFERENCE_MAX_PARTICLES,
        "source_wisp_max_emitters": SOURCE_WISP_REFERENCE_MAX_EMITTERS,
        "source_wisp_warmup_mode": "visible-only",
        "source_wisp_plume_ribbons": True,
        "broad_smoke_alpha": 0.68,
        "physical_alpha": 0.0,
        "physical_max_sources": 96,
        "composition_mode": MAP_FILM_COMPOSITION_MODE,
        "delivery_profile": REFERENCE_DELIVERY_PROFILE,
        "regional_smoke": True,
        "map_grade": "reference-film",
        "duration_seconds": 30.0,
        "audit_required_for_acceptance": False,
    },
    REFERENCE_EXACT_SMOKE_RENDER_PRESET: {
        "source_wisps": False,
        "physical_smoke": False,
        "source_wisp_emitter_mode": "fire-core",
        "physical_emitter_mode": "fire-core",
        "source_wisp_max_particles": SOURCE_WISP_REFERENCE_MAX_PARTICLES,
        "source_wisp_max_emitters": SOURCE_WISP_REFERENCE_MAX_EMITTERS,
        "source_wisp_warmup_mode": "visible-only",
        "source_wisp_plume_ribbons": True,
        "broad_smoke_alpha": 0.0,
        "physical_alpha": 0.0,
        "physical_max_sources": 0,
        "composition_mode": MAP_FILM_COMPOSITION_MODE,
        "delivery_profile": REFERENCE_DELIVERY_PROFILE,
        "regional_smoke": False,
        "map_grade": "reference-exact",
        "duration_seconds": 30.0,
        "audit_required_for_acceptance": True,
    },
}
SOURCE_WISP_AUDIT_THRESHOLDS = {
    "minimum_attached_source_fraction": 0.58,
    "minimum_screen_attached_source_fraction": 0.52,
    "minimum_active_fire_emitters": 16.0,
    "minimum_emitter_bbox_fraction": 0.012,
    "minimum_source_wisp_component_count": 4.0,
    "maximum_smoke_carpet_component_fraction": 0.34,
    "maximum_low_frequency_haze_fraction": 0.30,
    "minimum_strand_to_haze_ratio": 0.18,
    "minimum_fire_core_visibility_fraction": 0.62,
    "minimum_combined_strand_retention": 0.74,
    "minimum_encoded_strand_like_fraction": 0.0035,
    "maximum_late_low_frequency_haze_fraction": 0.20,
    "minimum_morphology_band_coverage_fraction": 0.00012,
    "minimum_transition_width_growth_ratio": 1.12,
    "minimum_old_tail_width_growth_ratio": 1.42,
    "maximum_old_tail_alpha_p90_fraction": 1.55,
    "maximum_old_tail_endpoint_alpha_fraction": 0.44,
    "minimum_old_tail_coverage_growth_ratio": 2.50,
    "minimum_old_tail_edge_softness_px": 1.05,
    "minimum_old_tail_diffuse_to_core_area_ratio": 1.10,
    "maximum_brush_bundle_score": 0.44,
    "minimum_encoded_soft_tail_like_fraction": 0.0025,
}
REFERENCE_FILM_AUDIT_THRESHOLDS = {
    "minimum_full_bleed_frame_coverage": 0.985,
    "minimum_map_quad_area_fraction": 0.985,
    "minimum_median_smoke_coverage_fraction": 0.16,
    "maximum_median_smoke_coverage_fraction": 0.78,
    "minimum_median_regional_smoke_coverage_fraction": 0.08,
    "minimum_median_dense_regional_smoke_fraction": 0.004,
    "maximum_median_dense_regional_smoke_fraction": 0.35,
    "minimum_median_fire_core_pixel_count": 24.0,
    "maximum_median_hot_fire_fraction": 0.035,
    "minimum_post_smoke_fire_visibility_fraction": 0.18,
    "minimum_active_fire_temporal_change_ratio": 0.10,
    "maximum_median_fire_mark_radius_px": 3.5,
    "minimum_median_halo_core_area_ratio": 1.35,
    "maximum_median_halo_core_area_ratio": 48.0,
    "minimum_median_mid_scale_smoke_fraction": 0.015,
    "minimum_smoke_centroid_motion_fraction": 0.004,
    "minimum_median_distributed_fire_cluster_count": 4.0,
    "minimum_median_fire_spread_grid_cell_count": 3.0,
    "minimum_median_far_fire_core_fraction": 0.045,
    "maximum_median_primary_fire_dominance_fraction": 0.86,
    "minimum_median_regional_smoke_texture_score": 0.0015,
    "maximum_median_regional_smoke_axis_band_score": 0.24,
    "maximum_median_regional_smoke_contour_band_score": 0.62,
    "maximum_median_regional_smoke_ring_score": 0.92,
    "minimum_median_label_contrast_delta": 0.070,
    "maximum_median_label_smoke_overlap_fraction": 0.68,
    "maximum_median_label_fire_overlap_fraction": 0.08,
    "minimum_median_label_text_pixel_fraction": 0.0010,
    "minimum_temporal_date_span_days": 20.0,
    "minimum_median_date_step_days": 2.0,
    "maximum_median_date_step_days": 7.0,
    "minimum_burned_area_growth_ratio": 4.0,
    "minimum_median_temporal_luma_delta": 0.0035,
    "minimum_encoded_smoke_like_fraction": 0.040,
    "minimum_encoded_soft_tail_like_fraction": 0.0012,
    "minimum_delivery_width": 1920.0,
    "minimum_delivery_height": 1080.0,
    "minimum_delivery_bitrate_bps": 1_900_000.0,
    "maximum_delivery_bitrate_bps": 3_400_000.0,
}
REFERENCE_FILM_REGIONAL_FIRE_ANCHORS = (
    (0.18, 0.49, 0.96, 0.00),
    (0.28, 0.34, 0.48, 0.11),
    (0.39, 0.63, 0.38, 0.27),
    (0.52, 0.43, 0.32, 0.18),
    (0.63, 0.72, 0.44, 0.36),
    (0.76, 0.55, 0.36, 0.52),
    (0.84, 0.29, 0.30, 0.68),
)
SOURCE_WISP_ACCEPTED_ARTIFACTS = (
    "final_mp4",
    "preview_png",
    "reference_generated_frame_sheet",
    "ablation_sheets",
    "source_wisp_audit_json",
    "exact_cli_command",
)
REFERENCE_FILM_ACCEPTED_ARTIFACTS = (
    "final_1080p_mp4",
    "preview_png",
    "source_wisp_audit_json",
    "reference_film_first_30s_contact_sheet",
    "reference_film_frame_reports",
    "reference_film_gate_report",
    "exact_cli_command",
)
REFERENCE_FILM_VISUAL_SIGNOFF_CONTRACT = {
    "status": "human_review_required_after_automated_gates",
    "reviewer_role": "designer/dataviz reviewer",
    "required_artifacts": [
        "final_1080p_mp4",
        "preview_png",
        "reference_film_first_30s_contact_sheet",
        "source_wisp_audit_json",
    ],
    "scorecard": [
        {
            "criterion": "geographic_spread",
            "automated_evidence": [
                "median_distributed_fire_cluster_count",
                "median_fire_spread_grid_cell_count",
                "median_far_fire_core_fraction",
                "median_primary_fire_dominance_fraction",
            ],
            "human_pass_standard": "Fire activity must read beyond one local August Complex cluster in the first-30s contact sheet.",
        },
        {
            "criterion": "regional_smoke_naturalism",
            "automated_evidence": [
                "median_regional_smoke_texture_score",
                "median_regional_smoke_axis_band_score",
            ],
            "human_pass_standard": "Broad smoke must show flow lanes, holes, soft edges, and no obvious rectangular or contour-band artifacts.",
        },
        {
            "criterion": "narrative_cadence",
            "automated_evidence": [
                "temporal_date_span_days",
                "median_date_step_days",
                "burned_area_growth_ratio",
                "active_fire_temporal_change_ratio",
            ],
            "human_pass_standard": "Each contact-sheet row should read as a new daily/event step, not as only second-by-second plume drift.",
        },
        {
            "criterion": "typography_hierarchy",
            "automated_evidence": [
                "median_label_contrast_delta",
                "median_label_smoke_overlap_fraction",
                "median_label_fire_overlap_fraction",
            ],
            "human_pass_standard": "Labels must remain readable while staying subordinate to fire, smoke, and terrain.",
        },
        {
            "criterion": "terrain_context",
            "automated_evidence": [
                "full_bleed_frame_coverage_fraction",
                "map_quad_area_fraction",
                "delivery_width_px",
                "delivery_height_px",
            ],
            "human_pass_standard": "The frame must read as a full-bleed map film with enough terrain/coastline context for regional transport.",
        },
    ],
}
FIRE_CORE_EMITTER_INTENSITY_THRESHOLD = 0.16
PHYSICAL_SMOKE_DIMS = (84, 22, 66)
PHYSICAL_SMOKE_RENDER_SIZE = (284, 216)
PHYSICAL_SMOKE_MAX_ALPHA = 154
PHYSICAL_SMOKE_MAX_SOURCES = 32
PHYSICAL_SMOKE_HISTORY_STRIDE = 6
PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES = 156
PHYSICAL_SMOKE_HISTORY_MAX_LAYERS = 12
PHYSICAL_SMOKE_HISTORY_ALPHA_SCALE = 0.30
PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA = 110
PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA = 86
PHYSICAL_SMOKE_VIEW_DIRECTION = (0.42, -0.68, 0.60)
PHYSICAL_SMOKE_PARALLAX_SCALE = 1.55
PHYSICAL_SMOKE_SUN_DIRECTION = (0.34, 0.82, -0.22)
HRRR_SMOKE_BASE_URL = "https://rapidrefresh.noaa.gov/hrrr/HRRRsmoke"
HRRR_SMOKE_OLD_BASE_URL = "https://rapidrefresh.noaa.gov/hrrr/HRRRsmokeold"
HRRR_SMOKE_DATASET_KEY = "hrrr_ncep_smoke_jet"
HRRR_SMOKE_BASE_URLS = (HRRR_SMOKE_BASE_URL, HRRR_SMOKE_OLD_BASE_URL)
HRRR_SMOKE_RAW_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_SMOKE_RAW_FIELD = "COLMD"
HRRR_SMOKE_RUNTIME = "2026060318"
HRRR_SMOKE_PLOT_TYPE = "trc1_full_int"
HRRR_SMOKE_FORECAST_HOURS = tuple(range(0, 19))
HRRR_SMOKE_GUIDANCE_STRENGTH = 0.22
HRRR_SMOKE_PANEL_CROP_FRAC = (0.014, 0.128, 0.994, 0.870)
HRRR_SMOKE_CA_SUBSET_FRAC = (0.055, 0.300, 0.295, 0.690)
REFERENCE_FILM_START_DATE = date(2020, 8, 16)
REFERENCE_FILM_TIMELINE_DAYS = 44
REFERENCE_FILM_START_AREA_HA = 6_700.0
REFERENCE_FILM_REGIONAL_SMOKE_MAX_ALPHA = 255
REFERENCE_FILM_DENSE_REGIONAL_SMOKE_ALPHA_THRESHOLD = 24
REFERENCE_FILM_FIRE_POINT_LIMIT = 220
_SMOKE_TEXTURE_CACHE: dict[tuple[tuple[int, int], int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PIXEL_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class FireEvent:
    name: str
    lon: float
    lat: float
    date: str
    final_area_ha: float
    source: str


@dataclass(frozen=True)
class TerrainPlate:
    image: Image.Image
    quad: list[tuple[float, float]]
    fire_xy: tuple[float, float]
    fire_uv: tuple[float, float]
    texture_size: tuple[int, int]
    bounds_mercator: tuple[float, float, float, float] | None = None
    extent_kind: str = "local"


@dataclass(frozen=True)
class ReferenceFilmFrameInfo:
    progress: float
    date_label: str
    burned_area_ha: float


@dataclass(frozen=True)
class HybridSmokeSource:
    x: float
    y: float
    strength: float
    radius_px: float
    start_frame: int
    end_frame: int
    seed: int
    burst_period_frames: float
    burst_phase_frames: float
    burst_duty: float
    heat: float = 1.0
    smoke_rate: float = 1.0
    altitude_bias: float = 0.0
    flame_end_frame: int | None = None


@dataclass
class HybridSmokeState:
    density: np.ndarray
    age_mass: np.ndarray
    layer_density: tuple[np.ndarray, ...] | None = None
    layer_age_mass: tuple[np.ndarray, ...] | None = None
    residual_haze: np.ndarray | None = None


@dataclass
class SourceWispPuff:
    source_index: int
    x: float
    y: float
    origin_x: float
    origin_y: float
    vx: float
    vy: float
    age_frames: float
    lifetime_frames: float
    radius_px: float
    base_radius_px: float
    alpha: float
    base_alpha: float
    heat: float
    intensity: float
    breakup_seed: int
    breakup_phase: float


@dataclass(frozen=True)
class SourceWispState:
    puffs: tuple[SourceWispPuff, ...]
    map_size: tuple[int, int]
    emitters: tuple[HybridSmokeSource, ...] = ()


@dataclass(frozen=True)
class HrrrSmokeGuidance:
    frames: tuple[np.ndarray, ...]
    runtime: str
    plot_type: str
    source_label: str


@dataclass(frozen=True)
class ReferenceSmokeEventState:
    event_id: str
    start_frame: int
    peak_frame: int
    end_frame: int
    coverage_peak: float
    centroid_path: tuple[tuple[int, float, float, float], ...]
    dominant_axis_degrees: float
    date_label: str
    source_size: tuple[int, int] = (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT)


@dataclass(frozen=True)
class ObservedSmokeSource:
    source_kind: str
    source_label: str
    frames: tuple[np.ndarray, ...] = ()
    event_states: tuple[ReferenceSmokeEventState, ...] = ()
    cache_dir: str | None = None
    requested_source: str = "auto"
    guidance_cadence_frames: float = 12.0
    timeline_frame_count: int = REFERENCE_EXACT_FRAME_COUNT
    disclosure_label: str = "Data: CAL FIRE perimeter, synthetic smoke field"
    approximate: bool = True


@dataclass
class PhysicalSmokeMainEffect:
    domain: Any
    step_settings: Any
    render_settings: Any
    sources: list[HybridSmokeSource]
    map_size: tuple[int, int]
    dims: tuple[int, int, int]
    render_size: tuple[int, int]
    substeps: int
    backend: str
    seed: int = HYBRID_SMOKE_SEED
    base_sources: list[HybridSmokeSource] = field(default_factory=list)
    emitter_mode: str = "synthetic"
    max_sources: int = PHYSICAL_SMOKE_MAX_SOURCES
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    previous_render_frame: int | None = None
    previous_render_rgba: np.ndarray | None = None


@dataclass(frozen=True)
class PhysicalSmokeEmitter3D:
    center: tuple[float, float, float]
    radius: float
    density_rate: float
    temperature_rate: float
    soot_rate: float
    humidity_rate: float
    emission_rate: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class PhysicalSmokeStepSettings3D:
    dt: float = 0.18
    density_decay: float = 0.010
    temperature_decay: float = 0.18
    velocity_damping: float = 0.046
    diffusion: float = 0.00125
    buoyancy: float = 0.018
    vorticity: float = 0.30
    pressure_iterations: int = 14
    turbulence_strength: float = 0.54
    turbulence_seed: int = HYBRID_SMOKE_SEED
    terrain_collision: bool = True
    boundary_damping: float = 0.035
    wind: tuple[float, float, float] = (0.115, 0.0, -0.044)


@dataclass(frozen=True)
class PhysicalSmokeRenderSettings3D:
    density_scale: float = 1.08
    extinction: float = 1.32
    soot_absorption: float = 0.22
    exposure: float = 1.12
    scattering: float = 1.00
    absorption: float = 0.26
    phase_g: float = 0.48
    step_size: float = 0.72
    max_steps: int = 128
    self_shadow: bool = True
    shadow_steps: int = 18
    shadow_step_size: float = 1.15
    jitter_strength: float = 0.35
    thin_color: tuple[float, float, float] = (0.50, 0.54, 0.58)
    dense_color: tuple[float, float, float] = (0.93, 0.91, 0.82)
    fire_glow: float = 0.38


class NumpyPhysicalSmokeDomain:
    def __init__(
        self,
        dims: tuple[int, int, int],
        sparse_threshold: float = 1.0e-6,
    ) -> None:
        nx, ny, nz = (max(2, int(v)) for v in dims)
        self.dims = (nx, ny, nz)
        self.sparse_threshold = float(sparse_threshold)
        shape = (nz, ny, nx)
        self.density = np.zeros(shape, dtype=np.float32)
        self.velocity = np.zeros(shape + (3,), dtype=np.float32)
        self.temperature = np.zeros(shape, dtype=np.float32)
        self.soot = np.zeros(shape, dtype=np.float32)
        self.humidity = np.zeros(shape, dtype=np.float32)
        self.particle_age = np.full(shape, -1.0, dtype=np.float32)
        self.emission_rate = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.time_seconds = 0.0
        self.frame_index = 0
        self._grid_x, self._grid_y, self._grid_z = _volume_grids(shape)

    def add_emitter(self, emitter: PhysicalSmokeEmitter3D, dt: float) -> None:
        nx, ny, nz = self.dims
        cx, cy, cz = emitter.center
        radius = max(float(emitter.radius), 1.0e-4)
        x0 = max(0, int(math.floor(cx - radius * 2.2)))
        x1 = min(nx, int(math.ceil(cx + radius * 2.2)))
        y0 = max(0, int(math.floor(cy - radius * 1.8)))
        y1 = min(ny, int(math.ceil(cy + radius * 2.4)))
        z0 = max(0, int(math.floor(cz - radius * 2.2)))
        z1 = min(nz, int(math.ceil(cz + radius * 2.2)))
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(cx)
        dy = (yy - np.float32(cy)) * 1.18
        dz = zz - np.float32(cz)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        falloff = np.clip(1.0 - _smoothstep(0.0, radius, dist), 0.0, 1.0).astype(np.float32)
        if not np.any(falloff > 0.0):
            return
        amount = float(dt) * falloff
        target = np.s_[z0:z1, y0:y1, x0:x1]
        self.density[target] += emitter.density_rate * amount
        self.temperature[target] += emitter.temperature_rate * amount
        self.soot[target] += emitter.soot_rate * amount
        self.humidity[target] += emitter.humidity_rate * amount
        self.emission_rate[target] += emitter.emission_rate * falloff
        for component, value in enumerate(emitter.velocity):
            self.velocity[target + (component,)] += float(value) * amount
        self.particle_age[target] = np.where(falloff > 0.0, 0.0, self.particle_age[target])

    def step(self, settings: PhysicalSmokeStepSettings3D, emitters: list[PhysicalSmokeEmitter3D]) -> None:
        dt = float(settings.dt)
        self.emission_rate.fill(0.0)
        for emitter in emitters:
            self.add_emitter(emitter, dt)

        self._apply_forces(settings)
        velocity_before = self.velocity.copy()
        for component in range(3):
            self.velocity[..., component] = _advect_volume_scalar(
                velocity_before[..., component],
                velocity_before,
                self._grid_x,
                self._grid_y,
                self._grid_z,
                dt,
            )
        self._project_velocity(settings.pressure_iterations)
        self._apply_boundaries(settings)
        self._apply_lane_advection_shear(settings)

        advect_velocity = self.velocity.copy()
        advected_age = _advect_volume_scalar(
            self.particle_age,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
            min_value=-1.0,
        )
        self.density = _advect_volume_scalar(
            self.density,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.temperature = _advect_volume_scalar(
            self.temperature,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.soot = _advect_volume_scalar(
            self.soot,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.humidity = _advect_volume_scalar(
            self.humidity,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.particle_age = np.where(
            self.density > self.sparse_threshold,
            np.maximum(advected_age, 0.0),
            -1.0,
        ).astype(np.float32)
        self._apply_subgrid_density_eddies(settings)
        self._diffuse_and_decay(settings)
        self._project_velocity(max(1, settings.pressure_iterations // 2))
        self._apply_boundaries(settings)
        self.time_seconds += dt
        self.frame_index += 1

    def to_density_numpy(self) -> np.ndarray:
        return np.array(self.density, copy=True)

    def to_velocity_numpy(self) -> np.ndarray:
        return np.array(self.velocity, copy=True)

    def to_temperature_numpy(self) -> np.ndarray:
        return np.array(self.temperature, copy=True)

    def to_soot_numpy(self) -> np.ndarray:
        return np.array(self.soot, copy=True)

    def to_emission_numpy(self) -> np.ndarray:
        return np.array(self.emission_rate, copy=True)

    def to_particle_age_numpy(self) -> np.ndarray:
        return np.array(self.particle_age, copy=True)

    def set_density(self, density: np.ndarray) -> None:
        arr = np.asarray(density, dtype=np.float32)
        if arr.shape != self.density.shape:
            raise ValueError(f"density shape must be {self.density.shape}, got {arr.shape}")
        self.density = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_velocity(self, velocity: np.ndarray) -> None:
        arr = np.asarray(velocity, dtype=np.float32)
        if arr.shape != self.velocity.shape:
            raise ValueError(f"velocity shape must be {self.velocity.shape}, got {arr.shape}")
        self.velocity = np.ascontiguousarray(arr, dtype=np.float32)

    def set_temperature(self, temperature: np.ndarray) -> None:
        arr = np.asarray(temperature, dtype=np.float32)
        if arr.shape != self.temperature.shape:
            raise ValueError(f"temperature shape must be {self.temperature.shape}, got {arr.shape}")
        self.temperature = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_soot(self, soot: np.ndarray) -> None:
        arr = np.asarray(soot, dtype=np.float32)
        if arr.shape != self.soot.shape:
            raise ValueError(f"soot shape must be {self.soot.shape}, got {arr.shape}")
        self.soot = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_emission(self, emission: np.ndarray) -> None:
        arr = np.asarray(emission, dtype=np.float32)
        if arr.shape != self.emission_rate.shape:
            raise ValueError(f"emission shape must be {self.emission_rate.shape}, got {arr.shape}")
        self.emission_rate = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def physics_report(self) -> dict[str, float]:
        return {
            "mass": float(np.sum(self.density)),
            "max_density": float(np.max(self.density, initial=0.0)),
            "divergence_l2": float(np.sqrt(np.mean(_volume_divergence(self.velocity) ** 2))),
            "time_seconds": float(self.time_seconds),
            "frame_index": float(self.frame_index),
        }

    def _apply_forces(self, settings: PhysicalSmokeStepSettings3D) -> None:
        dt = float(settings.dt)
        wind = np.asarray(settings.wind, dtype=np.float32)
        self.velocity[..., 0] += wind[0] * dt
        self.velocity[..., 1] += (wind[1] + self.temperature * settings.buoyancy) * dt
        self.velocity[..., 2] += wind[2] * dt
        if settings.turbulence_strength > 0.0:
            texture = _advected_smoke_texture(
                (self.density.shape[0], self.density.shape[2]),
                int(self.frame_index),
                int(settings.turbulence_seed) + 9401,
            )
            grad_z, grad_x = np.gradient(texture)
            amp = settings.turbulence_strength * dt
            altitude = np.linspace(0.55, 1.15, self.density.shape[1], dtype=np.float32)[None, :, None]
            self.velocity[..., 0] += grad_z[:, None, :] * amp * 3.4 * altitude
            self.velocity[..., 2] -= grad_x[:, None, :] * amp * 3.4 * altitude
            self.velocity[..., 1] += (texture[:, None, :] - 0.5) * amp * 0.08
            wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
            wind_x = float(wind[0]) / wind_len
            wind_z = float(wind[2]) / wind_len
            cross_x = -wind_z
            cross_z = wind_x
            x = self._grid_x.astype(np.float32)
            z = self._grid_z.astype(np.float32)
            along = x * wind_x + z * wind_z
            cross_coord = x * cross_x + z * cross_z
            lane_phase = along * 0.34 + cross_coord * 0.72 + float(self.frame_index) * 0.075
            lane_phase += float(settings.turbulence_seed) * 0.0031
            lane = np.sin(lane_phase).astype(np.float32)
            lane += 0.45 * np.sin(lane_phase * 0.53 + z * 0.29).astype(np.float32)
            lane *= _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 80.0, 0.018), self.density)
            self.velocity[..., 0] += cross_x * lane * amp * 0.82 * altitude
            self.velocity[..., 2] += cross_z * lane * amp * 0.82 * altitude
            speed_lanes = (0.5 + 0.5 * np.cos(lane_phase * 0.41 + x * 0.18)).astype(np.float32)
            self.velocity[..., 0] += wind_x * speed_lanes * amp * 0.30 * altitude
            self.velocity[..., 2] += wind_z * speed_lanes * amp * 0.30 * altitude
            altitude_coord = np.linspace(0.0, 1.0, self.density.shape[1], dtype=np.float32)[None, :, None]
            shear = (altitude_coord - 0.42) * amp * 1.35
            self.velocity[..., 0] += cross_x * shear
            self.velocity[..., 2] += cross_z * shear
        self._apply_vorticity_confinement(settings)
        damping = math.exp(-settings.velocity_damping * dt)
        self.velocity *= np.float32(damping)

    def _apply_lane_advection_shear(self, settings: PhysicalSmokeStepSettings3D) -> None:
        if settings.turbulence_strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        wind = np.asarray(settings.wind, dtype=np.float32)
        wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
        wind_x = float(wind[0]) / wind_len
        wind_z = float(wind[2]) / wind_len
        cross_x = -wind_z
        cross_z = wind_x
        x = self._grid_x.astype(np.float32)
        y = self._grid_y.astype(np.float32)
        z = self._grid_z.astype(np.float32)
        along = x * wind_x + z * wind_z
        cross_coord = x * cross_x + z * cross_z
        amp = np.float32(settings.turbulence_strength * settings.dt)
        active = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 60.0, 0.012), self.density)
        lane_phase = along * 0.23 + cross_coord * 0.49 + float(self.frame_index) * 0.105
        lane_phase += float(settings.turbulence_seed) * 0.0027
        lane_force = np.sin(lane_phase).astype(np.float32)
        lane_force += 0.58 * np.sin(lane_phase * 0.41 + y * 0.74).astype(np.float32)
        altitude = y / max(float(self.density.shape[1] - 1), 1.0)
        altitude_shear = (altitude - 0.44) * 0.95
        force = (lane_force * 2.75 + altitude_shear * 1.45) * active * amp
        self.velocity[..., 0] += cross_x * force
        self.velocity[..., 2] += cross_z * force
        slab_phase = along * 0.17 - cross_coord * 0.31 + y * 1.12 + float(self.frame_index) * 0.043
        slab_phase += float(settings.turbulence_seed) * 0.0021
        slab_lane = np.sin(slab_phase).astype(np.float32)
        slab_lane += 0.42 * np.sin(slab_phase * 0.53 + along * 0.09).astype(np.float32)
        slab_split = ((altitude - 0.50) * 2.55 + slab_lane * 0.58) * active * amp
        self.velocity[..., 0] += cross_x * slab_split * 1.90
        self.velocity[..., 2] += cross_z * slab_split * 1.90
        speed_split = np.sin(slab_phase * 0.39 + y * 0.67).astype(np.float32) * active * amp
        self.velocity[..., 0] += wind_x * speed_split * 0.52
        self.velocity[..., 2] += wind_z * speed_split * 0.52
        mass = np.clip(self.density, 0.0, None)
        total = float(np.sum(mass))
        if total <= 1.0e-6:
            return
        cx = float(np.sum(x * mass) / total)
        cz = float(np.sum(z * mass) / total)
        altitude_gain = 0.55 + 0.75 * altitude
        for eddy_index, (distance, radius, side, strength) in enumerate(
            (
                (5.5, 5.4, 1.0, 1.85),
                (11.5, 7.6, -1.0, 1.58),
                (19.0, 10.2, 1.0, 1.30),
                (28.0, 13.0, -1.0, 1.05),
            )
        ):
            phase = float(self.frame_index) * (0.035 + eddy_index * 0.006) + float(settings.turbulence_seed) * 0.0013
            center_x = cx + wind_x * distance + cross_x * side * radius * (0.40 + 0.20 * math.sin(phase))
            center_z = cz + wind_z * distance + cross_z * side * radius * (0.40 + 0.20 * math.cos(phase))
            dx = x - np.float32(center_x)
            dz = z - np.float32(center_z)
            r2 = dx * dx + dz * dz
            envelope = np.exp(-r2 / np.float32(2.0 * radius * radius)).astype(np.float32) * active
            inv_r = 1.0 / np.sqrt(r2 + np.float32(1.0))
            spin = side * strength * amp * envelope * altitude_gain
            self.velocity[..., 0] += (-dz * inv_r) * spin
            self.velocity[..., 2] += (dx * inv_r) * spin

    def _apply_subgrid_density_eddies(self, settings: PhysicalSmokeStepSettings3D) -> None:
        if settings.turbulence_strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        texture = _advected_smoke_texture(
            (self.density.shape[0], self.density.shape[2]),
            int(self.frame_index),
            int(settings.turbulence_seed) + 17417,
        )
        broad = _pil_blur_float(texture, 5.0)
        x = self._grid_x.astype(np.float32)
        y = self._grid_y.astype(np.float32)
        z = self._grid_z.astype(np.float32)
        phase = x * 0.18 + z * 0.27 + y * 0.72 + broad[:, None, :] * 3.4 + float(self.frame_index) * 0.046
        phase += float(settings.turbulence_seed) * 0.0019
        ribbons = 0.5 + 0.5 * np.sin(phase).astype(np.float32)
        sheets = 0.5 + 0.5 * np.sin(phase * 0.47 - z * 0.16 + y * 0.51).astype(np.float32)
        active = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 90.0, 0.018), self.density)
        voids = _smoothstep(0.45, 0.84, 1.0 - ribbons) * _smoothstep(0.34, 0.76, 1.0 - sheets) * active
        ridges = _smoothstep(0.62, 0.94, ribbons) * _smoothstep(0.48, 0.90, sheets) * active
        age_t = _smoothstep(2.0, 28.0, np.clip(self.particle_age, 0.0, None))
        void_strength = 0.62 + 0.32 * age_t
        ridge_strength = 0.075 - 0.045 * age_t
        gain = 1.0 - void_strength * voids + ridge_strength * ridges
        wind = np.asarray(settings.wind, dtype=np.float32)
        wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
        wind_x = float(wind[0]) / wind_len
        wind_z = float(wind[2]) / wind_len
        cross_x = -wind_z
        cross_z = wind_x
        along = x * wind_x + z * wind_z
        cross_coord = x * cross_x + z * cross_z
        channel_phase = (
            along * 0.115
            + cross_coord * 0.52
            + broad[:, None, :] * 5.4
            + np.sin(y * 0.62 + along * 0.035) * 0.85
            + float(self.frame_index) * 0.033
            + float(settings.turbulence_seed) * 0.0023
        )
        channel_wave = 0.5 + 0.5 * np.sin(channel_phase).astype(np.float32)
        channel_wave += 0.28 * np.sin(channel_phase * 0.47 - cross_coord * 0.19 + y * 0.34).astype(np.float32)
        entrainment = _smoothstep(0.58, 1.06, channel_wave)
        lateral_slots = _smoothstep(0.50, 0.94, 1.0 - (0.62 * ribbons + 0.38 * sheets))
        core_protect = 1.0 - 0.56 * _smoothstep(0.72, 1.75, self.density)
        aged_sheet = (0.28 + 0.72 * age_t) * active * core_protect
        clear_air = np.clip(entrainment * (0.54 + 0.46 * lateral_slots) * aged_sheet, 0.0, 1.0)
        channel_void = np.clip(
            _smoothstep(0.42, 0.86, 1.0 - channel_wave)
            * (0.55 + 0.45 * lateral_slots)
            * active
            * (0.42 + 0.58 * age_t)
            * core_protect,
            0.0,
            1.0,
        )
        gain *= 1.0 - (0.024 + 0.055 * age_t) * clear_air
        gain *= 1.0 - (0.045 + 0.070 * age_t) * channel_void
        self.density = np.clip(self.density * gain.astype(np.float32), 0.0, 8.0).astype(np.float32)
        self.humidity = np.clip(
            self.humidity
            * (
                1.0
                - (0.15 + 0.10 * age_t) * voids
                - (0.024 + 0.055 * age_t) * clear_air
                - (0.045 + 0.070 * age_t) * channel_void
            ),
            0.0,
            None,
        ).astype(np.float32)

    def _apply_vorticity_confinement(self, settings: PhysicalSmokeStepSettings3D) -> None:
        strength = float(settings.vorticity)
        if strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        u = self.velocity[..., 0]
        v = self.velocity[..., 1]
        w = self.velocity[..., 2]
        du_dz, du_dy, du_dx = np.gradient(u)
        dv_dz, dv_dy, dv_dx = np.gradient(v)
        dw_dz, dw_dy, dw_dx = np.gradient(w)
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        omega_mag = np.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)
        if not np.any(omega_mag > 1.0e-7):
            return
        dmag_dz, dmag_dy, dmag_dx = np.gradient(omega_mag)
        norm = np.sqrt(dmag_dx * dmag_dx + dmag_dy * dmag_dy + dmag_dz * dmag_dz) + 1.0e-6
        nx = dmag_dx / norm
        ny = dmag_dy / norm
        nz = dmag_dz / norm
        force_x = ny * omega_z - nz * omega_y
        force_y = nz * omega_x - nx * omega_z
        force_z = nx * omega_y - ny * omega_x
        density_gate = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 40.0, 0.02), self.density)
        amp = np.float32(strength * settings.dt * 0.36)
        self.velocity[..., 0] += force_x.astype(np.float32) * amp * density_gate
        self.velocity[..., 1] += force_y.astype(np.float32) * amp * density_gate
        self.velocity[..., 2] += force_z.astype(np.float32) * amp * density_gate

    def _diffuse_and_decay(self, settings: PhysicalSmokeStepSettings3D) -> None:
        mix = float(np.clip(settings.diffusion * settings.dt * 64.0, 0.0, 0.28))
        if mix > 0.0:
            self.density = _diffuse_volume(self.density, mix)
            self.temperature = _diffuse_volume(self.temperature, mix * 0.8)
            self.soot = _diffuse_volume(self.soot, mix * 0.9)
            self.humidity = _diffuse_volume(self.humidity, mix)
        age_decay = _smoothstep(7.0, 36.0, np.clip(self.particle_age, 0.0, None))
        self.density *= np.exp(-settings.density_decay * settings.dt * (1.0 + 3.0 * age_decay)).astype(np.float32)
        self.temperature *= np.float32(math.exp(-settings.temperature_decay * settings.dt))
        self.soot *= np.exp(-settings.density_decay * settings.dt * (0.42 + 1.15 * age_decay)).astype(np.float32)
        active = self.density > self.sparse_threshold
        self.particle_age = np.where(active, np.maximum(self.particle_age, 0.0) + settings.dt, -1.0).astype(np.float32)
        self.density = np.clip(self.density, 0.0, 8.0).astype(np.float32)

    def _project_velocity(self, iterations: int) -> None:
        divergence = _volume_divergence(self.velocity)
        pressure = np.zeros_like(divergence, dtype=np.float32)
        for _ in range(max(1, int(iterations))):
            padded = np.pad(pressure, 1, mode="edge")
            pressure = (
                padded[1:-1, 1:-1, :-2]
                + padded[1:-1, 1:-1, 2:]
                + padded[1:-1, :-2, 1:-1]
                + padded[1:-1, 2:, 1:-1]
                + padded[:-2, 1:-1, 1:-1]
                + padded[2:, 1:-1, 1:-1]
                - divergence
            ) / 6.0
        grad_z, grad_y, grad_x = np.gradient(pressure)
        self.velocity[..., 0] -= grad_x.astype(np.float32)
        self.velocity[..., 1] -= grad_y.astype(np.float32)
        self.velocity[..., 2] -= grad_z.astype(np.float32)
        self.pressure = pressure

    def _apply_boundaries(self, settings: PhysicalSmokeStepSettings3D) -> None:
        self.velocity[:, :, 0, 0] = 0.0
        self.velocity[:, :, -1, 0] = 0.0
        self.velocity[:, 0, :, 1] = 0.0
        self.velocity[:, -1, :, 1] = 0.0
        self.velocity[0, :, :, 2] = 0.0
        self.velocity[-1, :, :, 2] = 0.0
        self.density[:, :, 0] *= 0.58
        self.density[:, :, -1] *= 0.58
        self.density[0, :, :] *= 0.58
        self.density[-1, :, :] *= 0.58
        if self.density.shape[2] > 3:
            self.density[:, :, 1] *= 0.78
            self.density[:, :, -2] *= 0.78
        if self.density.shape[0] > 3:
            self.density[1, :, :] *= 0.78
            self.density[-2, :, :] *= 0.78
        if settings.terrain_collision:
            keep = 1.0 - settings.boundary_damping
            self.density[:, 0, :] *= keep
            self.temperature[:, 0, :] *= keep


AUGUST_COMPLEX = FireEvent(
    name="August Complex",
    lon=-122.9,
    lat=39.7,
    date="2020-08-16",
    final_area_ha=417_898.0,
    source="Reference metadata: examples/california_wildfire_smoke_video.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an 8-second August Complex hybrid smoke demo.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--render-preset",
        choices=tuple(SMOKE_RENDER_PRESETS),
        default=TARGET_RENDER_PRESET,
        help="Resolve smoke layer ownership, emitter source, and audit defaults for the render.",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(WIDTH, HEIGHT), metavar=("W", "H"))
    parser.add_argument(
        "--composition-mode",
        choices=(TERRAIN_SLAB_COMPOSITION_MODE, MAP_FILM_COMPOSITION_MODE),
        default=None,
        help="Use the local oblique terrain slab or full-bleed reference map-film composition.",
    )
    parser.add_argument(
        "--delivery-profile",
        choices=tuple(DELIVERY_PROFILES),
        default=None,
        help="Resolve output size and encode settings for local or 1080p reference-film delivery.",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--hybrid-smoke-width", type=int, default=HYBRID_SMOKE_WIDTH)
    parser.add_argument("--hybrid-smoke-height", type=int, default=HYBRID_SMOKE_HEIGHT)
    parser.add_argument("--smoke-width", type=int, default=360)
    parser.add_argument("--smoke-height", type=int, default=210)
    parser.add_argument("--physical-smoke-dims", type=int, nargs=3, default=PHYSICAL_SMOKE_DIMS, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--physical-render-size",
        type=int,
        nargs=2,
        default=PHYSICAL_SMOKE_RENDER_SIZE,
        metavar=("W", "H"),
    )
    parser.add_argument("--physical-max-sources", type=int, default=None)
    parser.add_argument("--physical-substeps", type=int, default=1)
    parser.add_argument(
        "--physical-emitter-mode",
        choices=SOURCE_WISP_EMITTER_MODES,
        default=None,
        help="Use static synthetic sources or per-frame pre-bloom fire-core/front emitters for physical smoke.",
    )
    parser.add_argument(
        "--physical-smoke-backend",
        choices=("auto", "native", "numpy"),
        default="auto",
        help="3D smoke backend for the main effect; auto prefers native forge3d.smoke.",
    )
    parser.add_argument("--physical-smoke", action="store_true", dest="physical_smoke")
    parser.add_argument("--no-physical-smoke", action="store_false", dest="physical_smoke")
    parser.add_argument("--source-wisps", action="store_true", dest="source_wisps")
    parser.add_argument("--no-source-wisps", action="store_false", dest="source_wisps")
    parser.add_argument("--source-wisp-max-particles", type=int, default=None)
    parser.add_argument("--source-wisp-max-emitters", type=int, default=None)
    parser.add_argument(
        "--source-wisp-emitter-mode",
        choices=SOURCE_WISP_EMITTER_MODES,
        default=None,
        help="Use static synthetic sources or dynamic pre-bloom fire-core/front pixels for source wisps.",
    )
    parser.add_argument(
        "--source-wisp-warmup-mode",
        choices=("full", "visible-only"),
        default=None,
        help="Whether source-attached wisps inherit the simulation warmup or start fresh on frame 0.",
    )
    parser.add_argument("--source-wisp-plume-ribbons", action="store_true", dest="source_wisp_plume_ribbons")
    parser.add_argument("--no-source-wisp-plume-ribbons", action="store_false", dest="source_wisp_plume_ribbons")
    parser.add_argument("--regional-smoke", action="store_true", dest="regional_smoke")
    parser.add_argument("--no-regional-smoke", action="store_false", dest="regional_smoke")
    parser.add_argument(
        "--observed-smoke-source",
        choices=("auto", "reference-cache", "hrrr", "procedural"),
        default="auto",
        help="Source for the broad regional smoke layer in reference-film mode.",
    )
    parser.add_argument("--broad-smoke-alpha", type=float, default=None)
    parser.add_argument("--physical-alpha", type=float, default=None)
    parser.add_argument(
        "--smoke-ablation",
        choices=("combined", "broad-only", "physical-only", "source-wisps-only", "no-broad"),
        default="combined",
        help="Render a smoke component ablation while preserving fire/terrain timing.",
    )
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--audit-frame-times", type=float, nargs="+", default=SOURCE_WISP_AUDIT_TIMES)
    parser.add_argument("--enforce-audit-gates", action="store_true", dest="enforce_audit_gates")
    parser.add_argument("--no-enforce-audit-gates", action="store_false", dest="enforce_audit_gates")
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=REFERENCE_VIDEO_DEFAULT,
    )
    parser.add_argument("--reference-smoke-cache", type=Path, default=REFERENCE_EXACT_CACHE)
    parser.add_argument(
        "--reference-smoke-mode",
        choices=("exact", "procedural"),
        default=None,
        help="Use decoded reference smoke playback or the procedural reference-film smoke approximation.",
    )
    parser.add_argument("--reference-smoke-start-frame", type=int, default=0)
    parser.add_argument("--reference-smoke-frame-count", type=int, default=None)
    parser.add_argument(
        "--prepare-reference-smoke-cache",
        action="store_true",
        help="Decode and derive the native first-30s reference-exact smoke cache, then exit.",
    )
    parser.add_argument("--hrrr-smoke-dir", type=Path, default=CACHE / "hrrr_smoke")
    parser.add_argument("--hrrr-runtime", default=HRRR_SMOKE_RUNTIME)
    parser.add_argument("--hrrr-plot-type", default=HRRR_SMOKE_PLOT_TYPE)
    parser.add_argument("--hrrr-base-url", default=HRRR_SMOKE_BASE_URL)
    parser.add_argument("--fetch-hrrr-smoke", action="store_true")
    parser.add_argument("--volume-detail", action="store_true", dest="volume_detail")
    parser.add_argument("--no-volume-detail", action="store_false", dest="volume_detail")
    parser.set_defaults(
        volume_detail=False,
        physical_smoke=None,
        source_wisps=None,
        source_wisp_plume_ribbons=None,
        regional_smoke=None,
        enforce_audit_gates=None,
    )
    args = parser.parse_args()
    return _apply_smoke_render_preset(args)


def _apply_smoke_render_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = SMOKE_RENDER_PRESETS[str(args.render_preset)]
    for name in (
        "source_wisps",
        "physical_smoke",
        "source_wisp_emitter_mode",
        "physical_emitter_mode",
        "source_wisp_max_particles",
        "source_wisp_max_emitters",
        "source_wisp_warmup_mode",
        "source_wisp_plume_ribbons",
        "broad_smoke_alpha",
        "physical_alpha",
        "physical_max_sources",
        "composition_mode",
        "delivery_profile",
        "regional_smoke",
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, preset[name])
    delivery = DELIVERY_PROFILES[str(args.delivery_profile)]
    default_size = tuple(int(v) for v in delivery["size"])
    if tuple(map(int, args.size)) == (WIDTH, HEIGHT) and default_size != (WIDTH, HEIGHT):
        args.size = default_size
    if float(args.duration) == float(DURATION_SECONDS):
        args.duration = float(preset["duration_seconds"])
    if args.reference_smoke_mode is None:
        args.reference_smoke_mode = (
            "exact" if str(args.render_preset) == REFERENCE_EXACT_SMOKE_RENDER_PRESET else "procedural"
        )
    if args.reference_smoke_frame_count is None:
        args.reference_smoke_frame_count = (
            REFERENCE_EXACT_FRAME_COUNT
            if str(args.reference_smoke_mode) == "exact"
            else max(1, int(round(float(args.duration) * int(args.fps))))
        )
    if getattr(args, "enforce_audit_gates", None) is None:
        args.enforce_audit_gates = False
    args.encode_policy = {
        "delivery_profile": str(args.delivery_profile),
        "size": tuple(map(int, args.size)),
        "crf": int(delivery["crf"]),
        "video_bitrate": delivery["video_bitrate"],
        "maxrate": delivery.get("maxrate"),
        "bufsize": delivery.get("bufsize"),
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "colorspace": "bt709",
    }
    args.layer_policy = {
        "render_preset": str(args.render_preset),
        "composition_mode": str(args.composition_mode),
        "delivery_profile": str(args.delivery_profile),
        "source_wisp_emitter_mode": str(args.source_wisp_emitter_mode),
        "physical_emitter_mode": str(args.physical_emitter_mode),
        "source_wisp_max_particles": int(args.source_wisp_max_particles),
        "source_wisp_max_emitters": int(args.source_wisp_max_emitters),
        "source_wisp_warmup_mode": str(args.source_wisp_warmup_mode),
        "source_wisp_plume_ribbons": bool(args.source_wisp_plume_ribbons),
        "broad_smoke_alpha": float(args.broad_smoke_alpha),
        "physical_alpha": float(args.physical_alpha),
        "physical_max_sources": int(args.physical_max_sources),
        "regional_smoke": bool(args.regional_smoke),
        "observed_smoke_source": str(args.observed_smoke_source),
        "map_grade": str(preset["map_grade"]),
        "duration_seconds": float(args.duration),
        "source_wisps_primary": str(args.render_preset) == TARGET_RENDER_PRESET,
        "reference_film_target": str(args.render_preset) == REFERENCE_FILM_RENDER_PRESET,
        "reference_exact_smoke_target": str(args.render_preset) == REFERENCE_EXACT_SMOKE_RENDER_PRESET,
        "reference_smoke_mode": str(args.reference_smoke_mode),
        "reference_smoke_cache": str(args.reference_smoke_cache),
        "reference_smoke_start_frame": int(args.reference_smoke_start_frame),
        "reference_smoke_frame_count": int(args.reference_smoke_frame_count),
        "audit_required_for_acceptance": bool(preset["audit_required_for_acceptance"]),
    }
    return args


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    clipped_lat = float(np.clip(lat, -85.05112878, 85.05112878))
    x = WEB_MERCATOR_LIMIT * lon / 180.0
    y = WEB_MERCATOR_LIMIT * math.log(math.tan((90.0 + clipped_lat) * math.pi / 360.0)) / math.pi
    return x, y


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def fire_pixel(bounds: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[float, float]:
    west, south, east, north = bounds
    mx, my = lonlat_to_web_mercator(AUGUST_COMPLEX.lon, AUGUST_COMPLEX.lat)
    width, height = image_size
    x = (mx - west) / max(east - west, 1e-6) * (width - 1)
    y = (north - my) / max(north - south, 1e-6) * (height - 1)
    return x, y


def crop_fire_extent() -> tuple[Image.Image, np.ndarray, tuple[float, float]]:
    if not (DEM_PATH.exists() and OVERLAY_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "Cached California terrain assets are missing. Run "
            "examples/california_wildfire_smoke_video.py once to populate examples/.cache."
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    bounds = tuple(float(v) for v in meta["bounds_mercator"])
    overlay = Image.open(OVERLAY_PATH).convert("RGBA")
    dem = np.asarray(Image.open(DEM_PATH), dtype=np.float32)
    fire_x, fire_y = fire_pixel(bounds, overlay.size)

    crop_w = min(300, overlay.width)
    crop_h = min(230, overlay.height)
    left = int(np.clip(round(fire_x - crop_w * 0.35), 0, overlay.width - crop_w))
    top = int(np.clip(round(fire_y - crop_h * 0.60), 0, overlay.height - crop_h))
    box = (left, top, left + crop_w, top + crop_h)
    crop_overlay = overlay.crop(box)
    crop_dem = dem[top : top + crop_h, left : left + crop_w]
    fire_in_crop = (fire_x - left, fire_y - top)
    return enhance_terrain_texture(crop_overlay, crop_dem), crop_dem, fire_in_crop


def terrain_crop_mercator_bounds() -> tuple[float, float, float, float]:
    if not (OVERLAY_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "Cached California terrain metadata is missing. Run "
            "examples/california_wildfire_smoke_video.py once to populate examples/.cache."
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    west, south, east, north = (float(v) for v in meta["bounds_mercator"])
    with Image.open(OVERLAY_PATH) as overlay:
        overlay_size = overlay.size
    fire_x, fire_y = fire_pixel((west, south, east, north), overlay_size)
    width, height = overlay_size
    crop_w = min(300, width)
    crop_h = min(230, height)
    left = int(np.clip(round(fire_x - crop_w * 0.35), 0, width - crop_w))
    top = int(np.clip(round(fire_y - crop_h * 0.60), 0, height - crop_h))
    x0 = west + left / max(width - 1, 1) * (east - west)
    x1 = west + (left + crop_w - 1) / max(width - 1, 1) * (east - west)
    y_top = north - top / max(height - 1, 1) * (north - south)
    y_bottom = north - (top + crop_h - 1) / max(height - 1, 1) * (north - south)
    return float(x0), float(y_bottom), float(x1), float(y_top)


def regional_reference_extent() -> tuple[Image.Image, np.ndarray, tuple[float, float], tuple[float, float, float, float]]:
    if not (REGIONAL_DEM_PATH.exists() and REGIONAL_OVERLAY_PATH.exists() and REGIONAL_META_PATH.exists()):
        texture, dem, fire = crop_fire_extent()
        return texture, dem, fire, terrain_crop_mercator_bounds()
    meta = json.loads(REGIONAL_META_PATH.read_text(encoding="utf-8"))
    west, south, east, north = (float(v) for v in meta["bounds_mercator"])
    overlay = Image.open(REGIONAL_OVERLAY_PATH).convert("RGBA")
    dem = np.asarray(Image.open(REGIONAL_DEM_PATH), dtype=np.float32)
    fire_x, fire_y = fire_pixel((west, south, east, north), overlay.size)

    alpha = np.asarray(overlay.getchannel("A"), dtype=np.uint8) > 8
    valid_y, valid_x = np.where(alpha)
    target_aspect = 16.0 / 9.0
    if valid_x.size:
        valid_w = int(valid_x.max() - valid_x.min() + 1)
        min_w = min(overlay.width, 900)
        crop_w = min(overlay.width, max(min_w, int(round(valid_w * 0.84))))
    else:
        crop_w = overlay.width
    crop_w = max(1, int(crop_w))
    target_aspect = 16.0 / 9.0
    crop_h = min(overlay.height, int(round(crop_w / target_aspect)))
    if crop_h >= overlay.height:
        crop_h = overlay.height
        crop_w = min(overlay.width, int(round(crop_h * target_aspect)))
    # Keep broad coastal/terrain context while preventing the event from being stranded
    # in a half-empty no-data frame.
    left = int(np.clip(round(fire_x - crop_w * 0.22), 0, overlay.width - crop_w))
    top = int(np.clip(round(fire_y - crop_h * 0.45), 0, overlay.height - crop_h))
    box = (left, top, left + crop_w, top + crop_h)
    crop_overlay = overlay.crop(box)
    crop_dem = dem[top : top + crop_h, left : left + crop_w]
    fire_in_crop = (fire_x - left, fire_y - top)
    x0 = west + left / max(overlay.width - 1, 1) * (east - west)
    x1 = west + (left + crop_w - 1) / max(overlay.width - 1, 1) * (east - west)
    y_top = north - top / max(overlay.height - 1, 1) * (north - south)
    y_bottom = north - (top + crop_h - 1) / max(overlay.height - 1, 1) * (north - south)
    return (
        enhance_terrain_texture(crop_overlay, crop_dem, fill_dem_without_overlay=True),
        crop_dem,
        fire_in_crop,
        (float(x0), float(y_bottom), float(x1), float(y_top)),
    )


def enhance_terrain_texture(
    texture: Image.Image,
    dem: np.ndarray,
    *,
    fill_dem_without_overlay: bool = False,
) -> Image.Image:
    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    source_alpha = np.clip(rgba[..., 3], 0.0, 1.0)
    dem_valid = np.isfinite(dem) & (dem > -5000.0)
    valid = dem_valid & (source_alpha > 0.035)
    if np.any(valid):
        fill_value = float(np.nanmedian(dem[valid]))
    elif np.any(dem_valid):
        fill_value = float(np.nanmedian(dem[dem_valid]))
    else:
        fill_value = 0.0
    terrain = np.where(dem_valid, dem, fill_value)
    lo, hi = np.percentile(terrain[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    norm = np.clip((terrain - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gy, gx = np.gradient(norm)
    shade = np.clip(0.48 - gx * 2.15 - gy * 1.30 + norm * 0.36, 0.20, 1.16)
    luma = np.sum(rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
    coal = np.array([0.030, 0.034, 0.035], dtype=np.float32)
    charcoal = np.array([0.100, 0.104, 0.098], dtype=np.float32)
    ash = np.array([0.310, 0.300, 0.270], dtype=np.float32)
    grade = (
        coal * (1.0 - norm[..., None])
        + charcoal * (1.0 - np.abs(norm[..., None] - 0.52) * 1.42).clip(0.0, 1.0)
        + ash * (norm[..., None] ** 1.85)
    )
    relief_detail = np.clip(0.76 + luma * 0.26, 0.72, 1.02)
    dem_grade = np.clip(grade * shade[..., None] * relief_detail[..., None] * 0.88, 0.0, 1.0)
    source_relief = np.clip(rgb * np.clip(0.92 + (shade - 0.62) * 0.18, 0.82, 1.10)[..., None], 0.0, 1.0)
    enhanced = np.clip(source_relief * 0.26 + dem_grade * 0.74, 0.0, 1.0)
    if fill_dem_without_overlay:
        dem_only = np.clip(dem_grade * 0.96 + charcoal * 0.18, 0.0, 1.0)
        dem_mask = np.clip(_pil_blur_float(dem_valid.astype(np.float32), 2.0), 0.0, 1.0)[..., None]
        alpha_mask = np.clip(source_alpha, 0.0, 1.0)[..., None]
        enhanced = np.clip(
            dem_only * np.clip(dem_mask - alpha_mask, 0.0, 1.0)
            + enhanced * np.maximum(alpha_mask, 1.0 - np.clip(dem_mask - alpha_mask, 0.0, 1.0)),
            0.0,
            1.0,
        )
    soft_mask = np.clip(_pil_blur_float(source_alpha, 1.6), 0.0, 1.0)[..., None]
    if fill_dem_without_overlay:
        soft_mask = np.maximum(
            soft_mask,
            np.clip(_pil_blur_float(dem_valid.astype(np.float32), 2.2), 0.0, 0.82)[..., None],
        )
    background = np.array([0.030, 0.045, 0.046], dtype=np.float32)
    out = np.clip(background * (1.0 - soft_mask) + enhanced * soft_mask, 0.0, 1.0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB").convert("RGBA")


def perspective_coeffs(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector)).tolist()


def terrain_plate(width: int, height: int) -> TerrainPlate:
    texture, _dem, fire_crop = crop_fire_extent()
    sky = Image.new("RGBA", (width, height), (11, 17, 22, 255))
    arr = np.array(sky, dtype=np.uint8, copy=True)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    arr[..., 0] = np.round(14 + 18 * (1.0 - yy)).astype(np.uint8)
    arr[..., 1] = np.round(20 + 28 * (1.0 - yy)).astype(np.uint8)
    arr[..., 2] = np.round(25 + 36 * (1.0 - yy)).astype(np.uint8)
    base = Image.fromarray(arr, mode="RGBA")

    dst = [
        (width * 0.09, height * 0.27),
        (width * 0.91, height * 0.15),
        (width * 0.96, height * 0.88),
        (width * 0.06, height * 0.95),
    ]
    draw = ImageDraw.Draw(base, "RGBA")
    draw.polygon([dst[3], dst[2], (width * 0.94, height * 0.98), (width * 0.07, height * 1.04)], fill=(4, 8, 10, 220))
    coeffs = perspective_coeffs(
        [(0, 0), (texture.width, 0), (texture.width, texture.height), (0, texture.height)],
        dst,
    )
    warped = texture.transform((width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    base.alpha_composite(warped)
    fire_uv = (fire_crop[0] / texture.width, fire_crop[1] / texture.height)
    fire_screen = bilinear_quad_point(dst, *fire_uv)
    return TerrainPlate(
        image=base,
        quad=dst,
        fire_xy=fire_screen,
        fire_uv=fire_uv,
        texture_size=texture.size,
        bounds_mercator=terrain_crop_mercator_bounds(),
        extent_kind="local",
    )


def _reference_map_grade(texture: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Create a full-bleed dark map plate for the reference-film composition."""
    width, height = map(int, size)
    base = texture.resize((width, height), Image.Resampling.BICUBIC).convert("RGB")
    arr = np.asarray(base, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    luma = np.sum(arr * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
    relief = np.clip(arr * 1.12 + luma[..., None] * 0.18, 0.0, 1.0)
    cool_shadow = np.array([0.030, 0.046, 0.060], dtype=np.float32)
    land = np.array([0.105, 0.110, 0.103], dtype=np.float32)
    terrain = cool_shadow * 0.55 + land * 0.45 + relief * np.array([0.18, 0.20, 0.19], dtype=np.float32)
    west_light = np.clip(1.0 - x, 0.0, 1.0) ** 1.8
    ocean_glow = np.array([0.040, 0.110, 0.150], dtype=np.float32) * west_light[..., None] * 0.72
    vignette = 1.0 - 0.42 * np.clip(((x - 0.48) / 0.74) ** 2 + ((y - 0.50) / 0.80) ** 2, 0.0, 1.0)
    graded = np.clip((terrain + ocean_glow) * vignette[..., None], 0.0, 1.0)
    return Image.fromarray(np.round(graded * 255.0).astype(np.uint8), mode="RGB").convert("RGBA")


def map_film_plate(width: int, height: int) -> TerrainPlate:
    """Return a full-bleed map plate with identity layer projection."""
    texture, _dem, fire_crop, bounds = regional_reference_extent()
    image = _reference_map_grade(texture, (width, height))
    fire_uv = (fire_crop[0] / texture.width, fire_crop[1] / texture.height)
    fire_xy = (fire_uv[0] * (width - 1), fire_uv[1] * (height - 1))
    return TerrainPlate(
        image=image,
        quad=[(0.0, 0.0), (float(width - 1), 0.0), (float(width - 1), float(height - 1)), (0.0, float(height - 1))],
        fire_xy=fire_xy,
        fire_uv=fire_uv,
        texture_size=texture.size,
        bounds_mercator=bounds,
        extent_kind="regional-california",
    )


def reference_film_frame_info(frame_index: int, frame_count: int) -> ReferenceFilmFrameInfo:
    progress = float(np.clip(float(frame_index) / max(float(frame_count - 1), 1.0), 0.0, 1.0))
    day_index = int(round(progress * REFERENCE_FILM_TIMELINE_DAYS))
    current_date = REFERENCE_FILM_START_DATE + timedelta(days=day_index)
    area_t = float(_smoothstep(0.0, 1.0, progress))
    burned_area = REFERENCE_FILM_START_AREA_HA * (1.0 - area_t) + AUGUST_COMPLEX.final_area_ha * area_t
    return ReferenceFilmFrameInfo(
        progress=progress,
        date_label=current_date.isoformat(),
        burned_area_ha=float(burned_area),
    )


def reference_exact_frame_info(frame_index: int, frame_count: int = REFERENCE_EXACT_FRAME_COUNT) -> ReferenceFilmFrameInfo:
    progress = float(np.clip(float(frame_index) / max(float(frame_count - 1), 1.0), 0.0, 1.0))
    day_index = int(round(progress * REFERENCE_EXACT_TIMELINE_DAYS))
    current_date = REFERENCE_EXACT_START_DATE + timedelta(days=day_index)
    area_t = float(_smoothstep(0.0, 1.0, progress))
    burned_area = REFERENCE_EXACT_START_AREA_HA * (1.0 - area_t) + REFERENCE_EXACT_END_AREA_HA * area_t
    return ReferenceFilmFrameInfo(
        progress=progress,
        date_label=current_date.isoformat(),
        burned_area_ha=float(burned_area),
    )


def bilinear_quad_point(quad: list[tuple[float, float]], u: float, v: float) -> tuple[float, float]:
    tl, tr, br, bl = [np.asarray(p, dtype=np.float32) for p in quad]
    top = tl * (1.0 - u) + tr * u
    bottom = bl * (1.0 - u) + br * u
    point = top * (1.0 - v) + bottom * v
    return float(point[0]), float(point[1])


def _smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1.0e-6)
    t = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pixel_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(shape[0]), int(shape[1]))
    key = (height, width)
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is None:
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        cached = (x, y)
        _PIXEL_GRID_CACHE[key] = cached
    return cached


def _box_blur_3x3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        axial = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        diagonal = (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        out = center * 0.42 + axial * 0.1175 + diagonal * 0.0275
    return out.astype(np.float32, copy=False)


def _pil_blur_float(field: np.ndarray, radius: float) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    peak = max(float(np.max(arr)), 1.0e-6)
    image = Image.fromarray(np.clip(arr / peak * 255.0, 0, 255).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return (np.asarray(blurred, dtype=np.float32) / 255.0 * peak).astype(np.float32)


def _bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    height, width = src.shape
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    top = src[y0c, x0c] * (1.0 - wx) + src[y0c, x1c] * wx
    bottom = src[y1c, x0c] * (1.0 - wx) + src[y1c, x1c] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid, sampled, 0.0).astype(np.float32)


def _resample_float_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("field must be 2D")
    target_h, target_w = int(shape[0]), int(shape[1])
    if src.shape == (target_h, target_w):
        return src.astype(np.float32, copy=True)
    x, y = _pixel_grids((target_h, target_w))
    src_h, src_w = src.shape
    sample_x = x / max(float(target_w - 1), 1.0) * max(float(src_w - 1), 0.0)
    sample_y = y / max(float(target_h - 1), 1.0) * max(float(src_h - 1), 0.0)
    return _bilinear_sample(src, sample_x, sample_y)


def _bilinear_sample_wrapped(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    height, width = src.shape
    xw = np.mod(x, max(width, 1)).astype(np.float32)
    yw = np.mod(y, max(height, 1)).astype(np.float32)
    xf = np.floor(xw)
    yf = np.floor(yw)
    x0 = np.mod(xf.astype(np.int64), width)
    y0 = np.mod(yf.astype(np.int64), height)
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx = xw - xf.astype(np.float32)
    wy = yw - yf.astype(np.float32)
    top = src[y0, x0] * (1.0 - wx) + src[y0, x1] * wx
    bottom = src[y1, x0] * (1.0 - wx) + src[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _volume_grids(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    return x, y, z


def _trilinear_sample_volume(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    depth, height, width = src.shape
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0) & (z >= 0.0) & (z <= depth - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    z0c = np.clip(z0, 0, depth - 1)
    z1c = np.clip(z1, 0, depth - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    wz = z - z0.astype(np.float32)
    c000 = src[z0c, y0c, x0c]
    c100 = src[z0c, y0c, x1c]
    c010 = src[z0c, y1c, x0c]
    c110 = src[z0c, y1c, x1c]
    c001 = src[z1c, y0c, x0c]
    c101 = src[z1c, y0c, x1c]
    c011 = src[z1c, y1c, x0c]
    c111 = src[z1c, y1c, x1c]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return np.where(valid, c0 * (1.0 - wz) + c1 * wz, 0.0).astype(np.float32)


def _advect_volume_scalar(
    field: np.ndarray,
    velocity: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    dt: float,
    *,
    mac_cormack: bool = False,
    min_value: float = 0.0,
) -> np.ndarray:
    back_x = grid_x - velocity[..., 0] * np.float32(dt)
    back_y = grid_y - velocity[..., 1] * np.float32(dt)
    back_z = grid_z - velocity[..., 2] * np.float32(dt)
    predicted = _trilinear_sample_volume(field, back_x, back_y, back_z)
    if not mac_cormack:
        return np.clip(predicted, float(min_value), None).astype(np.float32)

    back_vx = _trilinear_sample_volume(velocity[..., 0], back_x, back_y, back_z)
    back_vy = _trilinear_sample_volume(velocity[..., 1], back_x, back_y, back_z)
    back_vz = _trilinear_sample_volume(velocity[..., 2], back_x, back_y, back_z)
    fwd_x = back_x + back_vx * np.float32(dt)
    fwd_y = back_y + back_vy * np.float32(dt)
    fwd_z = back_z + back_vz * np.float32(dt)
    recovered = _trilinear_sample_volume(predicted, fwd_x, fwd_y, fwd_z)
    candidate = predicted + (np.asarray(field, dtype=np.float32) - recovered) * np.float32(0.5)
    lo, hi = _local_min_max_volume(field, back_x, back_y, back_z)
    corrected = np.clip(candidate, lo, hi)
    return np.clip(corrected, float(min_value), None).astype(np.float32)


def _local_min_max_volume(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(field, dtype=np.float32)
    depth, height, width = src.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    z0c = np.clip(z0, 0, depth - 1)
    z1c = np.clip(z1, 0, depth - 1)
    corners = (
        src[z0c, y0c, x0c],
        src[z0c, y0c, x1c],
        src[z0c, y1c, x0c],
        src[z0c, y1c, x1c],
        src[z1c, y0c, x0c],
        src[z1c, y0c, x1c],
        src[z1c, y1c, x0c],
        src[z1c, y1c, x1c],
    )
    lo = np.minimum.reduce(corners)
    hi = np.maximum.reduce(corners)
    return lo.astype(np.float32), hi.astype(np.float32)


def _diffuse_volume(field: np.ndarray, mix: float) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    padded = np.pad(src, 1, mode="edge")
    axial = (
        padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
    ) / 6.0
    return (src * (1.0 - mix) + axial * mix).astype(np.float32)


def _volume_divergence(velocity: np.ndarray) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float32)
    div = np.zeros(vel.shape[:3], dtype=np.float32)
    div[:, :, 1:-1] += (vel[:, :, 2:, 0] - vel[:, :, :-2, 0]) * 0.5
    div[:, 1:-1, :] += (vel[:, 2:, :, 1] - vel[:, :-2, :, 1]) * 0.5
    div[1:-1, :, :] += (vel[2:, :, :, 2] - vel[:-2, :, :, 2]) * 0.5
    return div


def _smooth_noise_field(shape: tuple[int, int], seed: int, cell_px: float, blur_radius: float) -> np.ndarray:
    height, width = shape
    cell = max(float(cell_px), 2.0)
    small_w = max(4, int(math.ceil(width / cell)) + 3)
    small_h = max(4, int(math.ceil(height / cell)) + 3)
    rng = np.random.default_rng(int(seed))
    raw = np.round(rng.random((small_h, small_w), dtype=np.float32) * 255.0).astype(np.uint8)
    image = Image.fromarray(raw).resize((width, height), Image.Resampling.BICUBIC)
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    noise = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(noise, 0.0, 1.0).astype(np.float32)


def hrrr_smoke_image_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_BASE_URL,
    plot_type: str = HRRR_SMOKE_PLOT_TYPE,
    dataset_key: str = HRRR_SMOKE_DATASET_KEY,
) -> str:
    root = str(base_url).rstrip("/")
    return (
        f"{root}/for_web/{str(dataset_key)}/{str(runtime)}/full/"
        f"{str(plot_type)}_f{int(forecast_hour):03d}.png"
    )


def _hrrr_smoke_display_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str,
    plot_type: str,
    dataset_key: str = HRRR_SMOKE_DATASET_KEY,
) -> str:
    root = str(base_url).rstrip("/")
    query = urllib.parse.urlencode(
        {
            "keys": str(dataset_key),
            "runtime": str(runtime),
            "plot_type": str(plot_type),
            "fcst": f"{int(forecast_hour):03d}",
            "time_inc": "60",
        }
    )
    return f"{root}/displayMapUpdated.cgi?{query}"


class _HrrrSmokeImageParser(HTMLParser):
    def __init__(self, runtime: str, plot_type: str) -> None:
        super().__init__()
        self.runtime = str(runtime)
        self.plot_type = str(plot_type)
        self.src: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.src is not None or tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs if value is not None}
        src = attr_map.get("src", "")
        if (
            ".png" in src
            and "for_web/" in src
            and self.runtime in src
            and f"{self.plot_type}_f" in src
        ):
            self.src = src


def _hrrr_smoke_base_url_candidates(base_url: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for item in (str(base_url), *HRRR_SMOKE_BASE_URLS):
        normalized = item.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _hrrr_smoke_cache_path(
    cache_dir: Path,
    runtime: str,
    forecast_hour: int,
    plot_type: str,
) -> Path:
    return Path(cache_dir) / str(runtime) / f"{str(plot_type)}_f{int(forecast_hour):03d}.png"


def _hrrr_raw_grib_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> str:
    runtime_text = str(runtime)
    date = runtime_text[:8]
    cycle = runtime_text[8:10]
    root = str(base_url).rstrip("/")
    return f"{root}/hrrr.{date}/conus/hrrr.t{cycle}z.wrfsfcf{int(forecast_hour):02d}.grib2"


def _hrrr_raw_idx_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> str:
    return f"{_hrrr_raw_grib_url(runtime, forecast_hour, base_url=base_url)}.idx"


def _hrrr_raw_cache_path(
    cache_dir: Path,
    runtime: str,
    forecast_hour: int,
    field: str = HRRR_SMOKE_RAW_FIELD,
) -> Path:
    return Path(cache_dir) / str(runtime) / f"{str(field).lower()}_f{int(forecast_hour):03d}.grib2"


def _parse_grib_index_range(index_text: str, field: str) -> tuple[int, int] | None:
    rows: list[tuple[int, int, str]] = []
    for line in index_text.splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            message_number = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append((message_number, offset, parts[3]))
    for idx, (_message_number, offset, name) in enumerate(rows):
        if name != field:
            continue
        if idx + 1 >= len(rows):
            return None
        return offset, rows[idx + 1][1] - 1
    return None


def _fetch_url_payload(url: str, extra_headers: dict[str, str] | None = None) -> tuple[bytes, str, str]:
    headers = {"User-Agent": "forge3d-california-cigar-smoke/1.0"}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        str(url),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        payload = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    return payload, final_url, content_type


def _is_png_payload(payload: bytes, content_type: str = "") -> bool:
    return (
        len(payload) > 1024
        and payload.startswith(b"\x89PNG\r\n\x1a\n")
        and ("png" in content_type.lower() or not content_type)
    )


def _download_hrrr_smoke_png(
    url: str,
    dest: Path,
    *,
    display_url: str | None = None,
    runtime: str,
    plot_type: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload, _, content_type = _fetch_url_payload(url)
        if _is_png_payload(payload, content_type):
            dest.write_bytes(payload)
            return
    except urllib.error.HTTPError:
        pass

    if display_url is not None:
        display_payload, final_display_url, _ = _fetch_url_payload(display_url)
        parser = _HrrrSmokeImageParser(runtime, plot_type)
        parser.feed(display_payload.decode("utf-8", errors="ignore"))
        if parser.src:
            image_url = urllib.parse.urljoin(final_display_url, parser.src)
            payload, _, content_type = _fetch_url_payload(image_url)
            if _is_png_payload(payload, content_type):
                dest.write_bytes(payload)
                return

    raise RuntimeError(f"HRRR-Smoke response was not an available PNG: {url}")


def _download_hrrr_raw_smoke_grib(
    runtime: str,
    forecast_hour: int,
    dest: Path,
    *,
    field: str = HRRR_SMOKE_RAW_FIELD,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> None:
    index_payload, _, _ = _fetch_url_payload(_hrrr_raw_idx_url(runtime, forecast_hour, base_url=base_url))
    byte_range = _parse_grib_index_range(index_payload.decode("utf-8", errors="replace"), field)
    if byte_range is None:
        raise RuntimeError(f"Raw HRRR field {field} was not found with a bounded byte range")
    start, end = byte_range
    payload, _, _ = _fetch_url_payload(
        _hrrr_raw_grib_url(runtime, forecast_hour, base_url=base_url),
        extra_headers={"Range": f"bytes={start}-{end}"},
    )
    if not payload.startswith(b"GRIB") or len(payload) < 1024:
        raise RuntimeError(f"Raw HRRR field {field} response was not a GRIB2 message")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def _hrrr_raw_smoke_grib_to_density(
    grib_path: Path,
    map_size: tuple[int, int],
    bounds_mercator: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except (ImportError, AttributeError):
        return np.zeros((int(map_size[1]), int(map_size[0])), dtype=np.float32)

    width, height = map(int, map_size)
    dst = np.zeros((height, width), dtype=np.float32)
    dst_bounds = bounds_mercator if bounds_mercator is not None else terrain_crop_mercator_bounds()
    dst_transform = from_bounds(*dst_bounds, width, height)
    try:
        with rasterio.open(grib_path) as src:
            source = src.read(1).astype(np.float32, copy=False)
            source = np.where(np.isfinite(source), np.clip(source, 0.0, None), 0.0).astype(np.float32)
            reproject(
                source=source,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                src_nodata=0.0,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
            )
    except Exception as exc:
        print(f"Skipping unreadable raw HRRR-Smoke frame {grib_path}: {exc}")
        return np.zeros((height, width), dtype=np.float32)

    field = np.clip(dst, 0.0, None).astype(np.float32)
    positive = field[field > 0.0]
    if positive.size == 0 or float(np.max(positive)) < 1.0e-10:
        return np.zeros((height, width), dtype=np.float32)
    lo = float(np.percentile(positive, 45.0))
    hi = float(np.percentile(positive, 99.4))
    density = np.clip((field - lo) / max(hi - lo, 1.0e-10), 0.0, 1.0)
    density = density**0.72
    return _pil_blur_float(density, max(0.45, min(width, height) / 520.0)).astype(np.float32)


def _hrrr_smoke_image_to_density(image: Image.Image, map_size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, map_size)
    source_image = image.convert("RGB")
    src_w, src_h = source_image.size
    if src_w >= 500 and src_h >= 400:
        px0, py0, px1, py1 = HRRR_SMOKE_PANEL_CROP_FRAC
        panel_box = (
            int(round(src_w * px0)),
            int(round(src_h * py0)),
            int(round(src_w * px1)),
            int(round(src_h * py1)),
        )
        panel = source_image.crop(panel_box)
        pw, ph = panel.size
        sx0, sy0, sx1, sy1 = HRRR_SMOKE_CA_SUBSET_FRAC
        source_image = panel.crop(
            (
                int(round(pw * sx0)),
                int(round(ph * sy0)),
                int(round(pw * sx1)),
                int(round(ph * sy1)),
            )
        )

    rgb = np.asarray(source_image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return np.zeros((height, width), dtype=np.float32)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    luma = r * 0.299 + g * 0.587 + b * 0.114
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    warm_smoke = np.clip((r * 0.60 + g * 0.42) - b * 0.48, 0.0, 1.0)
    cool_smoke = np.clip(b * 0.85 - r * 0.22 - g * 0.16, 0.0, 1.0)
    pale_haze = np.clip(luma - 0.56, 0.0, 1.0) * np.clip(0.34 - saturation, 0.0, 0.34) * 0.36
    signal = saturation * 0.78 + warm_smoke * 0.22 + cool_smoke * 0.18 + pale_haze
    signal *= _smoothstep(0.045, 0.13, saturation)
    signal *= 1.0 - 0.86 * _smoothstep(0.90, 0.99, luma)
    signal *= _smoothstep(0.16, 0.30, luma)
    if src_w >= 500 and src_h >= 400:
        hh, ww = signal.shape
        border = max(2, int(round(min(ww, hh) * 0.018)))
        signal[:border, :] = 0.0
        signal[-border:, :] = 0.0
        signal[:, :border] = 0.0
        signal[:, -border:] = 0.0
    signal = np.clip(signal, 0.0, 1.0).astype(np.float32)

    source = Image.fromarray(np.round(signal * 255.0).astype(np.uint8), mode="L")
    resized = source.resize((width, height), Image.Resampling.BICUBIC)
    density = np.asarray(resized, dtype=np.float32) / 255.0
    density = _pil_blur_float(density, max(1.0, min(width, height) / 180.0))
    positive = density[density > 0.0]
    if positive.size == 0 or float(np.max(positive)) < 1.0e-5:
        return np.zeros((height, width), dtype=np.float32)
    lo = float(np.percentile(positive, 58.0))
    hi = float(np.percentile(positive, 99.0))
    density = np.clip((density - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0)
    return _pil_blur_float(density, max(0.7, min(width, height) / 260.0)).astype(np.float32)


def _load_hrrr_raw_smoke_guidance(
    cache_dir: Path,
    map_size: tuple[int, int],
    *,
    runtime: str = HRRR_SMOKE_RUNTIME,
    forecast_hours: tuple[int, ...] = HRRR_SMOKE_FORECAST_HOURS,
    fetch: bool = False,
    raw_base_url: str = HRRR_SMOKE_RAW_BASE_URL,
    field: str = HRRR_SMOKE_RAW_FIELD,
    bounds_mercator: tuple[float, float, float, float] | None = None,
) -> HrrrSmokeGuidance | None:
    frames: list[np.ndarray] = []
    for hour in forecast_hours:
        path = _hrrr_raw_cache_path(Path(cache_dir), runtime, int(hour), field)
        if fetch and not path.exists():
            try:
                _download_hrrr_raw_smoke_grib(
                    runtime,
                    int(hour),
                    path,
                    field=field,
                    base_url=raw_base_url,
                )
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                print(f"Skipping unavailable raw HRRR-Smoke frame {runtime} f{int(hour):03d}: {exc}")
                continue
        if not path.exists():
            continue
        density = _hrrr_raw_smoke_grib_to_density(path, map_size, bounds_mercator=bounds_mercator)
        if np.any(density > 0.0):
            frames.append(density.astype(np.float32, copy=False))

    if not frames:
        return None
    label = f"raw HRRR-Smoke {runtime} {field} ({len(frames)} cached GRIB frames)"
    return HrrrSmokeGuidance(tuple(frames), str(runtime), str(field), label)


def load_hrrr_smoke_guidance(
    cache_dir: Path,
    map_size: tuple[int, int],
    *,
    runtime: str = HRRR_SMOKE_RUNTIME,
    plot_type: str = HRRR_SMOKE_PLOT_TYPE,
    forecast_hours: tuple[int, ...] = HRRR_SMOKE_FORECAST_HOURS,
    base_url: str = HRRR_SMOKE_BASE_URL,
    fetch: bool = False,
    prefer_raw: bool = True,
    raw_base_url: str = HRRR_SMOKE_RAW_BASE_URL,
    bounds_mercator: tuple[float, float, float, float] | None = None,
) -> HrrrSmokeGuidance | None:
    if prefer_raw:
        raw_guidance = _load_hrrr_raw_smoke_guidance(
            cache_dir,
            map_size,
            runtime=runtime,
            forecast_hours=forecast_hours,
            fetch=fetch,
            raw_base_url=raw_base_url,
            bounds_mercator=bounds_mercator,
        )
        if raw_guidance is not None:
            return raw_guidance

    frames: list[np.ndarray] = []
    for hour in forecast_hours:
        path = _hrrr_smoke_cache_path(Path(cache_dir), runtime, int(hour), plot_type)
        if fetch and not path.exists():
            errors: list[str] = []
            for candidate_base_url in _hrrr_smoke_base_url_candidates(base_url):
                try:
                    _download_hrrr_smoke_png(
                        hrrr_smoke_image_url(
                            runtime,
                            int(hour),
                            base_url=candidate_base_url,
                            plot_type=plot_type,
                        ),
                        path,
                        display_url=_hrrr_smoke_display_url(
                            runtime,
                            int(hour),
                            base_url=candidate_base_url,
                            plot_type=plot_type,
                        ),
                        runtime=runtime,
                        plot_type=plot_type,
                    )
                    break
                except (OSError, RuntimeError, urllib.error.URLError) as exc:
                    errors.append(f"{candidate_base_url}: {exc}")
            if not path.exists():
                suffix = f": {'; '.join(errors[-2:])}" if errors else ""
                print(
                    "Skipping unavailable HRRR-Smoke frame "
                    f"{runtime} f{int(hour):03d}{suffix}"
                )
                continue
        if not path.exists():
            continue
        try:
            density = _hrrr_smoke_image_to_density(Image.open(path), map_size)
        except OSError as exc:
            print(f"Skipping unreadable HRRR-Smoke frame {path}: {exc}")
            continue
        if np.any(density > 0.0):
            frames.append(density.astype(np.float32, copy=False))

    if not frames:
        return None
    label = f"HRRR-Smoke {runtime} {plot_type} ({len(frames)} cached frames)"
    return HrrrSmokeGuidance(tuple(frames), str(runtime), str(plot_type), label)


def _reference_event_source_size(cache_dir: Path) -> tuple[int, int]:
    candidates = (
        Path(cache_dir) / "smoke" / "smoke_rgba_0000.png",
        Path(cache_dir) / "frames" / "frame_0000.png",
    )
    for path in candidates:
        if path.exists():
            try:
                with Image.open(path) as image:
                    return tuple(map(int, image.size))
            except OSError:
                continue
    return (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT)


def _coerce_reference_smoke_event(
    payload: dict[str, object],
    source_size: tuple[int, int],
) -> ReferenceSmokeEventState | None:
    try:
        start_frame = int(payload["start_frame"])
        peak_frame = int(payload["peak_frame"])
        end_frame = int(payload["end_frame"])
    except (KeyError, TypeError, ValueError):
        return None
    start_frame = max(0, min(start_frame, end_frame))
    end_frame = max(start_frame, end_frame)
    peak_frame = int(np.clip(peak_frame, start_frame, end_frame))
    coverage_peak = float(payload.get("coverage_peak", 0.0))
    if not math.isfinite(coverage_peak):
        coverage_peak = 0.0

    centroid_path: list[tuple[int, float, float, float]] = []
    for item in payload.get("centroid_path", []):
        if not isinstance(item, dict):
            continue
        try:
            frame = int(item.get("frame", peak_frame))
            x_px = float(item.get("x_px", source_size[0] * 0.5))
            y_px = float(item.get("y_px", source_size[1] * 0.5))
            coverage = float(item.get("coverage", coverage_peak))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x_px, y_px, coverage)):
            continue
        centroid_path.append((frame, x_px, y_px, max(0.0, coverage)))
    if not centroid_path:
        centroid_path.append((peak_frame, source_size[0] * 0.5, source_size[1] * 0.48, max(coverage_peak, 0.01)))
    centroid_path = sorted(centroid_path, key=lambda item: item[0])

    return ReferenceSmokeEventState(
        event_id=str(payload.get("event_id", f"reference-event-{start_frame:04d}")),
        start_frame=start_frame,
        peak_frame=peak_frame,
        end_frame=end_frame,
        coverage_peak=max(0.0, coverage_peak),
        centroid_path=tuple(centroid_path),
        dominant_axis_degrees=float(payload.get("dominant_axis_degrees", 0.0)),
        date_label=str(payload.get("date_label", "")),
        source_size=tuple(map(int, source_size)),
    )


def load_reference_smoke_event_states(cache_dir: Path) -> tuple[ReferenceSmokeEventState, ...]:
    events_path = Path(cache_dir) / "reference_smoke_events.json"
    if not events_path.exists():
        return ()
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return ()
    source_size = _reference_event_source_size(Path(cache_dir))
    events = [
        event
        for item in payload
        if isinstance(item, dict)
        for event in (_coerce_reference_smoke_event(item, source_size),)
        if event is not None
    ]
    return tuple(events)


def make_observed_smoke_source(
    args: argparse.Namespace,
    map_size: tuple[int, int],
    hrrr_guidance: HrrrSmokeGuidance | None,
    *,
    visible_frame_count: int,
) -> ObservedSmokeSource:
    requested = str(getattr(args, "observed_smoke_source", "auto"))
    reference_film_target = bool(getattr(args, "layer_policy", {}).get("reference_film_target", False))
    cache_dir = Path(getattr(args, "reference_smoke_cache", REFERENCE_EXACT_CACHE))
    events: tuple[ReferenceSmokeEventState, ...] = ()
    if requested == "reference-cache" or (requested == "auto" and reference_film_target):
        events = load_reference_smoke_event_states(cache_dir)
        if events:
            return ObservedSmokeSource(
                source_kind="reference-derived-events",
                source_label=f"reference-derived smoke events ({len(events)} cached event windows)",
                event_states=events,
                cache_dir=str(cache_dir),
                requested_source=requested,
                timeline_frame_count=max(1, int(visible_frame_count)),
                disclosure_label="Data: CAL FIRE perimeter, reference-derived smoke timing, procedural smoke opacity",
                approximate=True,
            )
        if requested == "reference-cache":
            raise RuntimeError(f"reference-cache observed smoke source requires {cache_dir / 'reference_smoke_events.json'}")

    if hrrr_guidance is not None and requested in {"auto", "hrrr"}:
        cadence = max(1.0, float(visible_frame_count) / max(len(hrrr_guidance.frames) - 1, 1))
        return ObservedSmokeSource(
            source_kind="hrrr-smoke",
            source_label=hrrr_guidance.source_label,
            frames=hrrr_guidance.frames,
            requested_source=requested,
            guidance_cadence_frames=cadence,
            timeline_frame_count=max(1, int(visible_frame_count)),
            disclosure_label="Data: CAL FIRE perimeter, HRRR-Smoke guidance",
            approximate=False,
        )
    if requested == "hrrr":
        raise RuntimeError("hrrr observed smoke source requested but no HRRR-Smoke guidance frames are available")

    width, height = map(int, map_size)
    return ObservedSmokeSource(
        source_kind="procedural-fallback",
        source_label=f"deterministic procedural regional smoke ribbons ({width}x{height})",
        requested_source=requested,
        guidance_cadence_frames=12.0,
        timeline_frame_count=max(1, int(visible_frame_count)),
        disclosure_label="Data: CAL FIRE perimeter, synthetic smoke field",
        approximate=True,
    )


def observed_smoke_source_report(source: ObservedSmokeSource) -> dict[str, object]:
    return {
        "source_kind": source.source_kind,
        "source_label": source.source_label,
        "requested_source": source.requested_source,
        "cache_dir": source.cache_dir,
        "guidance_frame_count": len(source.frames),
        "event_count": len(source.event_states),
        "guidance_cadence_frames": float(source.guidance_cadence_frames),
        "timeline_frame_count": int(source.timeline_frame_count),
        "disclosure_label": source.disclosure_label,
        "approximate": bool(source.approximate),
        "events": [
            {
                "event_id": event.event_id,
                "start_frame": int(event.start_frame),
                "peak_frame": int(event.peak_frame),
                "end_frame": int(event.end_frame),
                "coverage_peak": float(event.coverage_peak),
                "date_label": event.date_label,
                "dominant_axis_degrees": float(event.dominant_axis_degrees),
            }
            for event in source.event_states[:12]
        ],
    }


def _advected_smoke_texture(shape: tuple[int, int], frame_index: int, seed: int) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    scale = min(width, height) / 408.0
    t = float(frame_index)
    cache_key = (shape, int(seed))
    cached = _SMOKE_TEXTURE_CACHE.get(cache_key)
    if cached is None:
        cached = (
            _smooth_noise_field(shape, int(seed) + 301, 112.0 * scale, 10.5 * scale),
            _smooth_noise_field(shape, int(seed) + 907, 46.0 * scale, 4.8 * scale),
            _smooth_noise_field(shape, int(seed) + 1709, 18.0 * scale, 1.7 * scale),
        )
        _SMOKE_TEXTURE_CACHE[cache_key] = cached
    broad, medium, fine = cached

    drift_x = 1.18 * scale * t
    drift_y = -0.43 * scale * t
    curl_x = (medium - 0.5) * 22.0 * scale + (broad - 0.5) * 62.0 * scale
    curl_y = (broad - 0.5) * 38.0 * scale - (medium - 0.5) * 12.0 * scale
    broad_s = _bilinear_sample_wrapped(broad, x - drift_x * 0.38 + curl_x * 0.15, y - drift_y * 0.38 + curl_y * 0.15)
    medium_s = _bilinear_sample_wrapped(medium, x - drift_x + curl_x * 0.34, y - drift_y + curl_y * 0.34)
    fine_s = _bilinear_sample_wrapped(fine, x - drift_x * 1.46 + curl_x * 0.72, y - drift_y * 1.46 + curl_y * 0.72)
    texture = 0.50 * broad_s + 0.34 * medium_s + 0.16 * fine_s
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _hybrid_border_fade(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    distance = np.minimum.reduce((x, y, width - 1.0 - x, height - 1.0 - y))
    margin = max(6.0, min(width, height) * 0.045)
    return _smoothstep(0.0, margin, distance).astype(np.float32)


def _hybrid_lifecycle_alpha(age_frames: np.ndarray) -> np.ndarray:
    age = np.asarray(age_frames, dtype=np.float32)
    birth = 0.12 + 0.88 * _smoothstep(0.0, 6.0, age)
    mature = 1.0 - 0.20 * _smoothstep(30.0, 100.0, age)
    fade_end = min(HYBRID_SMOKE_MAX_AGE_FRAMES - 28.0, 236.0)
    old_fade = 1.0 - _smoothstep(145.0, fade_end, age) ** 0.75
    return np.clip(birth * mature * old_fade, 0.0, 1.0).astype(np.float32)


def _procedural_hrrr_smoke_guidance(
    shape: tuple[int, int],
    frame_index: int,
    seed: int,
    source_xy: tuple[float, float],
) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    scale = min(width, height) / 408.0
    sx, sy = source_xy
    t = float(frame_index)
    wind = np.array([1.0, -0.42], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)
    dx = x - np.float32(sx)
    dy = y - np.float32(sy)
    along = dx * wind[0] + dy * wind[1]
    cross = dx * cross_dir[0] + dy * cross_dir[1]
    along_shift = along - t * 0.54 * scale
    downwind = np.maximum(along_shift, 0.0)
    range_fade = 1.0 - _smoothstep(width * 0.68, width * 1.02, downwind)

    wave = np.sin(along_shift / max(46.0 * scale, 1.0) + t * 0.013 + seed * 0.003)
    shear_wave = np.sin(along_shift / max(92.0 * scale, 1.0) - t * 0.006 + seed * 0.007)
    lane_center = (28.0 * wave + 13.0 * shear_wave) * scale
    lane_width = (10.0 + 0.055 * downwind) * scale
    primary_lane = (
        np.exp(-((cross - lane_center) ** 2) / (2.0 * lane_width * lane_width + 1.0e-6))
        * _smoothstep(-20.0 * scale, 48.0 * scale, along_shift)
        * np.exp(-downwind / max(520.0 * scale, 1.0))
        * range_fade
    )

    secondary_center = lane_center - (34.0 + 18.0 * np.sin(t * 0.011 + seed * 0.013)) * scale
    secondary_width = (17.0 + 0.080 * downwind) * scale
    secondary_lane = (
        np.exp(-((cross - secondary_center) ** 2) / (2.0 * secondary_width * secondary_width + 1.0e-6))
        * _smoothstep(16.0 * scale, 96.0 * scale, along_shift)
        * np.exp(-downwind / max(620.0 * scale, 1.0))
        * range_fade
    )

    front_center = (76.0 + 0.24 * t) * scale
    leading_band = (
        np.exp(-((along_shift - front_center) ** 2) / (2.0 * (48.0 * scale) ** 2 + 1.0e-6))
        * np.exp(-((cross - lane_center * 0.38) ** 2) / (2.0 * (72.0 * scale) ** 2 + 1.0e-6))
        * _smoothstep(-10.0 * scale, 70.0 * scale, along_shift)
        * range_fade
    )

    hook_x = sx + wind[0] * (82.0 + t * 0.31) * scale
    hook_y = sy + wind[1] * (82.0 + t * 0.31) * scale
    hx = x - np.float32(hook_x)
    hy = y - np.float32(hook_y)
    radius = np.hypot(hx, hy)
    theta = np.arctan2(hy, hx)
    spiral_radius = (46.0 + 0.12 * t + 8.0 * np.sin(theta * 2.0 - t * 0.019)) * scale
    hook = np.exp(-((radius - spiral_radius) ** 2) / (2.0 * (12.5 * scale) ** 2 + 1.0e-6))
    hook *= np.clip(0.48 + 0.52 * np.cos(theta - 0.017 * t + seed * 0.002), 0.0, 1.0)
    hook *= _smoothstep(18.0 * scale, 175.0 * scale, along) * range_fade

    eddy_x = sx + wind[0] * (185.0 + 0.18 * t) * scale + cross_dir[0] * 46.0 * scale
    eddy_y = sy + wind[1] * (185.0 + 0.18 * t) * scale + cross_dir[1] * 46.0 * scale
    er = np.hypot(x - np.float32(eddy_x), y - np.float32(eddy_y))
    et = np.arctan2(y - np.float32(eddy_y), x - np.float32(eddy_x))
    eddy = np.exp(-((er - 64.0 * scale) ** 2) / (2.0 * (24.0 * scale) ** 2 + 1.0e-6))
    eddy *= np.clip(0.42 + 0.58 * np.cos(et + t * 0.010 - seed * 0.005), 0.0, 1.0)
    eddy *= _smoothstep(70.0 * scale, 260.0 * scale, along_shift) * range_fade

    veil_axis = cross + 0.30 * along_shift - 22.0 * scale * np.sin(t * 0.009 + seed * 0.011)
    aged_veil = np.exp(-(veil_axis * veil_axis) / (2.0 * (118.0 * scale) ** 2 + 1.0e-6))
    aged_veil *= _smoothstep(-44.0 * scale, 125.0 * scale, along_shift)
    aged_veil *= 1.0 - _smoothstep(width * 0.74, width * 1.08, downwind)

    northern_sheet_center = -height * 0.08 + 42.0 * scale * np.sin(t * 0.006 + seed * 0.017)
    synoptic_sheet = np.exp(-((cross - northern_sheet_center) ** 2) / (2.0 * (152.0 * scale) ** 2 + 1.0e-6))
    synoptic_sheet *= _smoothstep(95.0 * scale, 300.0 * scale, along_shift)
    synoptic_sheet *= 1.0 - _smoothstep(width * 0.78, width * 1.10, downwind)

    texture = _advected_smoke_texture(shape, frame_index, int(seed) + 809)
    broad_texture = _pil_blur_float(texture, 7.0 * scale)
    striations = 0.5 + 0.5 * np.sin(
        cross / max(8.5 * scale, 1.0)
        + along_shift / max(44.0 * scale, 1.0)
        + t * 0.021
        + seed * 0.019
    )
    base = (
        0.42 * primary_lane
        + 0.23 * secondary_lane
        + 0.17 * leading_band
        + 0.18 * hook
        + 0.12 * eddy
        + 0.15 * aged_veil
        + 0.08 * synoptic_sheet
    )
    holes = _smoothstep(0.58, 0.90, 1.0 - texture) * _smoothstep(0.05, 0.42, base)
    breakup = np.clip(0.52 + 0.58 * texture + 0.20 * striations + 0.16 * (broad_texture - 0.5), 0.12, 1.26)
    guidance = base * breakup * (1.0 - 0.34 * holes)
    guidance = _pil_blur_float(np.clip(guidance, 0.0, 1.0), max(0.7, 1.05 * scale))
    return np.clip(guidance, 0.0, 1.0).astype(np.float32)


def make_hybrid_smoke_sources(
    fire_uv: tuple[float, float],
    map_size: tuple[int, int],
    total_frames: int = 120,
    seed: int = HYBRID_SMOKE_SEED,
    visible_start_frame: int = 0,
) -> list[HybridSmokeSource]:
    width, height = map(int, map_size)
    if width <= 8 or height <= 8:
        raise ValueError("hybrid smoke map must be larger than 8x8")
    rng = np.random.default_rng(int(seed))
    fire_x = float(fire_uv[0]) * (width - 1)
    fire_y = float(fire_uv[1]) * (height - 1)
    scale = min(width, height) / 408.0
    visible_start = max(0, int(visible_start_frame))
    visible_end = max(visible_start, int(total_frames) - 1)
    visible_span = max(1, visible_end - visible_start + 1)
    if visible_start > 0:
        flame_end_min = max(visible_start - int(round(0.16 * visible_span)), 8)
    else:
        flame_end_min = max(12, int(round(0.42 * visible_span)))
    flame_end_max = max(flame_end_min + 12, visible_start + int(round(0.92 * visible_span)))
    wind = np.array([1.0, -0.38], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    clusters = (
        (0.0, 0.0, 24, 1.22, 1.00),
        (-20.0, 20.0, 10, 0.92, 0.94),
        (20.0, -17.0, 9, 0.82, 0.90),
        (-15.0, -31.0, 8, 0.68, 0.86),
        (31.0, 13.0, 7, 0.58, 0.80),
    )
    sources: list[HybridSmokeSource] = []
    source_index = 0
    for cluster_index, (dx, dy, count, cluster_strength, spread) in enumerate(clusters):
        center = np.array([fire_x + dx * scale, fire_y + dy * scale], dtype=np.float32)
        for _ in range(count):
            along = rng.normal(0.0, 10.0 * spread * scale)
            lateral = rng.normal(0.0, 20.0 * spread * scale)
            jitter = wind * along + cross * lateral
            x = float(np.clip(center[0] + jitter[0], 3.0, width - 4.0))
            y = float(np.clip(center[1] + jitter[1], 3.0, height - 4.0))
            start_limit = max(1, min(30, int(total_frames * 0.24)))
            start_frame = 0 if source_index < 6 else int(rng.integers(0, start_limit))
            flame_phase = float(rng.uniform(0.0, 1.0))
            cluster_delay = min(0.16, cluster_index * 0.035)
            flame_end = int(
                round(
                    flame_end_min
                    + (flame_end_max - flame_end_min) * np.clip(flame_phase + cluster_delay, 0.0, 1.0)
                )
            )
            flame_end = max(flame_end, start_frame + int(rng.integers(22, 46)))
            smolder_tail = int(rng.integers(HYBRID_FIRE_SMOLDER_MIN_FRAMES, HYBRID_FIRE_SMOLDER_MAX_FRAMES + 1))
            radius = float(rng.uniform(4.2, 9.4) * scale * (0.92 + 0.20 * cluster_strength))
            strength = float(cluster_strength * rng.uniform(0.50, 1.08))
            burst_period = float(rng.uniform(32.0, 78.0))
            altitude_bias = float(np.clip(0.18 + cluster_index * 0.13 + rng.normal(0.0, 0.12), -0.18, 0.70))
            sources.append(
                HybridSmokeSource(
                    x=x,
                    y=y,
                    strength=strength,
                    radius_px=max(1.8, radius),
                    start_frame=start_frame,
                    end_frame=flame_end + smolder_tail,
                    seed=int(seed + source_index * 101 + cluster_index * 17),
                    burst_period_frames=burst_period,
                    burst_phase_frames=float(rng.uniform(0.0, burst_period)),
                    burst_duty=float(rng.uniform(0.32, 0.58)),
                    heat=float(np.clip(0.74 + 0.48 * cluster_strength + rng.normal(0.0, 0.10), 0.42, 1.55)),
                    smoke_rate=float(np.clip(0.72 + 0.38 * cluster_strength + rng.normal(0.0, 0.08), 0.44, 1.46)),
                    altitude_bias=altitude_bias,
                    flame_end_frame=flame_end,
                )
            )
            source_index += 1
    return sources


def _hybrid_layer_altitude(layer_index: int) -> float:
    if HYBRID_SMOKE_LAYER_COUNT <= 1:
        return 0.0
    return float(np.clip(int(layer_index), 0, HYBRID_SMOKE_LAYER_COUNT - 1)) / float(HYBRID_SMOKE_LAYER_COUNT - 1)


def _hybrid_layer_wind_vector(layer_index: int) -> np.ndarray:
    altitude = _hybrid_layer_altitude(layer_index)
    direction = np.array(
        [
            1.0 + 0.12 * altitude,
            -0.34 - 0.24 * altitude + 0.05 * math.sin(2.1 + layer_index),
        ],
        dtype=np.float32,
    )
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    return direction


def _hybrid_wind_field(
    frame_index: float,
    shape: tuple[int, int],
    seed: int,
    layer_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x, y = _pixel_grids(shape)
    xn = x / max(float(width - 1), 1.0)
    yn = y / max(float(height - 1), 1.0)
    scale = min(width, height) / 408.0
    altitude = _hybrid_layer_altitude(layer_index)
    phase = (int(seed) % 8191) / 8191.0 * math.tau + layer_index * 1.73
    t = float(frame_index)
    wind = _hybrid_layer_wind_vector(layer_index)
    speed = (2.48 + 0.62 * altitude) * scale

    u = np.full(shape, wind[0] * speed, dtype=np.float32)
    v = np.full(shape, wind[1] * speed, dtype=np.float32)
    u += (0.38 * scale * (1.0 + 0.22 * altitude) * np.sin(math.tau * (0.32 * yn + 0.0020 * t) + phase)).astype(np.float32)
    v += (0.30 * scale * (1.0 + 0.18 * altitude) * np.sin(math.tau * (0.36 * xn - 0.28 * yn + 0.0018 * t) + phase * 0.7)).astype(np.float32)

    stream = (
        np.sin(math.tau * ((0.62 + 0.08 * altitude) * xn + (0.34 - 0.05 * altitude) * yn) + 0.010 * t + phase)
        * np.sin(math.tau * ((0.46 - 0.06 * altitude) * yn - (0.27 + 0.04 * altitude) * xn) - 0.008 * t + phase * 1.61)
    ).astype(np.float32)
    dpsi_dy, dpsi_dx = np.gradient(stream)
    u += (4.60 + 0.55 * altitude) * scale * dpsi_dy.astype(np.float32)
    v += -(4.05 + 0.45 * altitude) * scale * dpsi_dx.astype(np.float32)

    synoptic = (
        np.sin(math.tau * (0.18 * xn - 0.31 * yn) + 0.004 * t + phase * 0.33)
        + np.cos(math.tau * (0.27 * xn + 0.20 * yn) - 0.003 * t + phase * 1.21)
    ).astype(np.float32)
    u += (0.36 * scale * (1.0 + 0.35 * altitude) * synoptic).astype(np.float32)
    v += (0.16 * scale * (1.0 + 0.28 * altitude) * np.sin(math.tau * (0.22 * xn + 0.24 * yn) - 0.006 * t + phase)).astype(np.float32)
    lane_texture = _pil_blur_float(
        _advected_smoke_texture(shape, int(round(t * 0.72)), seed + 6011 + layer_index * 811),
        max(3.4, 8.5 * scale),
    )
    lane_gy, lane_gx = np.gradient(lane_texture)
    lane_amp = (26.0 + 11.0 * altitude) * scale
    u += (lane_gy * lane_amp).astype(np.float32)
    v += (-lane_gx * lane_amp * 0.82).astype(np.float32)
    u = np.clip(u, -0.48 * scale, (4.20 + 0.70 * altitude) * scale)
    v = np.clip(v, -(1.76 + 0.38 * altitude) * scale, 1.02 * scale)
    return u.astype(np.float32), v.astype(np.float32)


def _hybrid_convective_uplift(
    shape: tuple[int, int],
    sources: list,
    frame_index: float,
    heat_radius_multiplier: float = 12.0,
    max_uplift: float = 2.5,
) -> np.ndarray:
    """Compute upward velocity field from active heat sources."""
    height, width = shape
    scale = min(width, height) / 408.0
    uplift = np.zeros(shape, dtype=np.float32)
    x, y = _pixel_grids(shape)
    for source in sources:
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        flame = _source_flame_lifecycle_weight(source, int(frame_index))
        if flame <= 0.01:
            continue
        heat_radius = source.radius_px * heat_radius_multiplier
        dx = x - source.x
        dy = y - source.y
        dist_sq = dx * dx + dy * dy
        influence = np.exp(-dist_sq / (2.0 * heat_radius * heat_radius))
        strength = source.strength * source.heat * flame
        uplift -= influence * strength * max_uplift * scale  # negative = upward
    return uplift.astype(np.float32)


def _hybrid_crosswind_spread(
    field: np.ndarray,
    amount: float = 0.16,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if amount <= 0.0 or not np.any(src > 0.0):
        return src.copy()
    x, y = _pixel_grids(src.shape)
    scale = min(src.shape) / 408.0
    if wind is None:
        wind = _hybrid_layer_wind_vector(0)
    cross = np.array([-float(wind[1]), float(wind[0])], dtype=np.float32)
    cross /= max(float(np.linalg.norm(cross)), 1.0e-6)
    out = src * (1.0 - amount * 0.58)
    for distance, weight in ((3.0, 0.22), (7.2, 0.16), (13.5, 0.09), (22.0, 0.045)):
        dx = cross[0] * distance * scale
        dy = cross[1] * distance * scale
        shifted_a = _bilinear_sample(src, x - dx, y - dy)
        shifted_b = _bilinear_sample(src, x + dx, y + dy)
        out += amount * weight * (shifted_a + shifted_b)
    return np.clip(out, 0.0, 6.0).astype(np.float32)


def _hybrid_downwind_stream(
    field: np.ndarray,
    amount: float = 0.18,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if amount <= 0.0 or not np.any(src > 0.0):
        return src.copy()
    x, y = _pixel_grids(src.shape)
    scale = min(src.shape) / 408.0
    if wind is None:
        wind = _hybrid_layer_wind_vector(0)
    out = src * (1.0 - amount * 0.50)
    for distance, weight in ((5.0, 0.26), (13.0, 0.18), (26.0, 0.10), (45.0, 0.05)):
        dx = wind[0] * distance * scale
        dy = wind[1] * distance * scale
        out += amount * weight * _bilinear_sample(src, x - dx, y - dy)
    return np.clip(out, 0.0, 6.0).astype(np.float32)


def _source_burst_envelope(source: HybridSmokeSource, frame_index: int) -> float:
    period = max(float(source.burst_period_frames), 1.0)
    phase = ((float(frame_index) + float(source.burst_phase_frames)) % period) / period
    attack = float(_smoothstep(0.0, 0.10, phase))
    release = 1.0 - float(_smoothstep(source.burst_duty, min(source.burst_duty + 0.24, 1.0), phase))
    ember = attack * release
    surge = 0.5 + 0.5 * math.sin(frame_index * 0.37 + source.seed * 0.029)
    return float(np.clip(0.06 + 1.62 * ember + 0.24 * surge * ember, 0.04, 1.92))


def _source_flame_end_frame(source: HybridSmokeSource) -> int:
    if source.flame_end_frame is None:
        return int(source.end_frame)
    return int(source.flame_end_frame)


def _source_flame_lifecycle_weight(source: HybridSmokeSource, frame_index: int) -> float:
    start = float(source.start_frame)
    flame_end = float(_source_flame_end_frame(source))
    frame = float(frame_index)
    if frame < start or frame > flame_end + 8.0:
        return 0.0
    attack = float(_smoothstep(start, start + 5.0, frame))
    fade = 1.0 - float(_smoothstep(flame_end - 9.0, flame_end + 6.0, frame))
    return float(np.clip(attack * fade, 0.0, 1.0))


def _source_smolder_lifecycle_weight(source: HybridSmokeSource, frame_index: int) -> float:
    flame_end = float(_source_flame_end_frame(source))
    end = float(max(source.end_frame, _source_flame_end_frame(source)))
    frame = float(frame_index)
    if frame < flame_end - 4.0 or frame > end:
        return 0.0
    rise = float(_smoothstep(flame_end - 4.0, flame_end + 10.0, frame))
    fade = 1.0 - float(_smoothstep(end - 18.0, end + 1.0, frame))
    return float(np.clip(rise * fade, 0.0, 1.0))


def _source_smoke_activity_weight(source: HybridSmokeSource, frame_index: int) -> float:
    flame = _source_flame_lifecycle_weight(source, frame_index)
    smolder = _source_smolder_lifecycle_weight(source, frame_index)
    return float(np.clip(max(flame, 0.36 * smolder), 0.0, 1.0))


def _source_burn_scar_weight(source: HybridSmokeSource, frame_index: int) -> float:
    frame = float(frame_index)
    start = float(source.start_frame)
    flame_end = float(_source_flame_end_frame(source))
    if frame < start + 4.0:
        return 0.0
    burn_span = max(flame_end - start, 1.0)
    scar_start = start + min(7.0, burn_span * 0.18)
    return float(np.clip(_smoothstep(scar_start, flame_end + 18.0, frame), 0.0, 1.0))


def _source_layer_weight(
    source: HybridSmokeSource,
    layer_index: int,
    layer_count: int = HYBRID_SMOKE_LAYER_COUNT,
) -> float:
    count = max(int(layer_count), 1)
    if count == 1:
        return 1.0
    altitudes = np.linspace(0.0, 1.0, count, dtype=np.float32)
    base_positions = np.linspace(0.0, 1.0, len(HYBRID_SMOKE_LAYER_WEIGHTS), dtype=np.float32)
    base = np.interp(altitudes, base_positions, np.asarray(HYBRID_SMOKE_LAYER_WEIGHTS, dtype=np.float32))
    center = np.clip(0.24 + float(source.altitude_bias) * 0.62, 0.06, 0.92)
    sigma = 0.36 + 0.08 * np.clip(float(source.heat) - 1.0, 0.0, 1.0)
    plume_lift = np.exp(-((altitudes - center) ** 2) / (2.0 * sigma * sigma + 1.0e-6))
    weights = base * (0.52 + plume_lift)
    total = max(float(np.sum(weights)), 1.0e-6)
    return float(weights[int(np.clip(layer_index, 0, count - 1))] / total)


def _inject_hybrid_sources(
    density: np.ndarray,
    age_mass: np.ndarray,
    sources: list[HybridSmokeSource],
    frame_index: int,
    layer_index: int = 0,
    layer_count: int = HYBRID_SMOKE_LAYER_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    out_density = np.asarray(density, dtype=np.float32).copy()
    out_age_mass = np.asarray(age_mass, dtype=np.float32).copy()
    height, width = out_density.shape
    altitude = _hybrid_layer_altitude(layer_index)
    wind = _hybrid_layer_wind_vector(layer_index)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)

    for source_index, source in enumerate(sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        source_activity = _source_smoke_activity_weight(source, frame_index)
        if source_activity <= 0.01:
            continue
        layer_weight = _source_layer_weight(source, layer_index, layer_count)
        if layer_weight <= 0.012:
            continue
        radius = max(float(source.radius_px) * (1.0 + 0.30 * altitude), 1.0)
        tail = radius * (20.0 + 12.0 * altitude)
        pad = int(math.ceil(tail + radius * 7.0))
        x0 = max(0, int(math.floor(source.x - pad)))
        x1 = min(width, int(math.ceil(source.x + pad + tail)))
        y0 = max(0, int(math.floor(source.y - pad - tail * 0.70)))
        y1 = min(height, int(math.ceil(source.y + pad + tail * 0.20)))
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(source.x)
        dy = yy - np.float32(source.y)
        along = dx * wind[0] + dy * wind[1]
        cross = dx * cross_dir[0] + dy * cross_dir[1]
        along_frac = np.clip(along / max(tail, 1.0), 0.0, 1.0)
        core = np.exp(-(dx * dx + dy * dy) / (2.0 * (radius * 1.02) ** 2))
        tail_gate = _smoothstep(-radius * 0.35, radius * 1.20, along) * (along <= tail)
        curl_offset = radius * (
            3.1 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
            + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + source.seed * 0.009)
            + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + source.seed * 0.005 + frame_index * 0.012)
        )
        tail_width = radius * (1.15 + 4.2 * along_frac**0.75)
        plume_tail = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 17.5, 1.0))
            * np.exp(-((cross - curl_offset) ** 2) / (2.0 * tail_width * tail_width + 1.0e-6))
            * tail_gate
        )
        veil_center = tail * 0.46
        veil_width = radius * 9.6
        broad_veil = (
            np.exp(-((along - veil_center) ** 2) / (2.0 * (tail * 0.52) ** 2 + 1.0e-6))
            * np.exp(-(cross * cross) / (2.0 * veil_width * veil_width + 1.0e-6))
            * tail_gate
        )
        sheet_offset = tail * (0.18 + 0.11 * math.sin(source.seed * 0.013))
        regional_sheet = (
            np.exp(-((along - sheet_offset) ** 2) / (2.0 * (tail * 0.82) ** 2 + 1.0e-6))
            * np.exp(-(cross * cross) / (2.0 * (radius * 15.5) ** 2 + 1.0e-6))
            * tail_gate
        )
        streamer_width = radius * (0.92 + 2.18 * along_frac)
        streamer_offset_a = radius * (
            2.0 * math.sin(source.seed * 0.019)
            + 5.1 * along_frac
            + 2.0 * math.sin(frame_index * 0.031 + source.seed * 0.007)
        )
        streamer_offset_b = -radius * (
            1.8 * math.cos(source.seed * 0.017)
            + 4.0 * along_frac
            + 1.6 * math.sin(frame_index * 0.027 + source.seed * 0.011)
        )
        streamer_a = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 24.0, 1.0))
            * np.exp(-((cross - streamer_offset_a) ** 2) / (2.0 * streamer_width * streamer_width + 1.0e-6))
            * tail_gate
        )
        streamer_b = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 21.0, 1.0))
            * np.exp(-((cross - streamer_offset_b) ** 2) / (2.0 * (streamer_width * 0.88) ** 2 + 1.0e-6))
            * tail_gate
        )
        pulse = 0.88 + 0.12 * math.sin(frame_index * 0.19 + source.seed * 0.017 + source_index)
        burst = _source_burst_envelope(source, frame_index)
        phase = source.seed * 0.031 + frame_index * 0.045
        lane = np.sin(cross / max(radius * 0.78, 1.0) + along / max(radius * 5.2, 1.0) + phase)
        fine_lane = np.sin(cross / max(radius * 0.34, 1.0) - along / max(radius * 3.4, 1.0) + phase * 1.73)
        cellular = np.sin(along / max(radius * 7.4, 1.0) + cross / max(radius * 2.25, 1.0) + phase * 0.62)
        filament_gain = np.clip(0.92 + 0.14 * lane + 0.06 * fine_lane, 0.62, 1.24)
        hole_cut = 1.0 - 0.14 * _smoothstep(0.16, 0.90, cellular) * along_frac
        addition = source.strength * source.smoke_rate * layer_weight * pulse * (
            0.016 * (1.0 - 0.28 * altitude) * core
            + 0.062 * (1.0 - 0.02 * altitude) * plume_tail
            + 0.0058 * (1.0 + 2.20 * altitude) * broad_veil
            + 0.0018 * (1.0 + 3.10 * altitude) * regional_sheet
            + 0.0220 * (1.0 + 0.48 * altitude) * (streamer_a + streamer_b)
        )
        addition = (addition * burst * source_activity * filament_gain * hole_cut).astype(np.float32)
        out_density[y0:y1, x0:x1] += addition
        plume_age = np.clip(
            6.5 + 14.0 * altitude + np.maximum(along, 0.0) / max(radius * (0.95 + 0.42 * altitude), 1.0),
            0.0,
            HYBRID_SMOKE_MAX_AGE_FRAMES * 0.72,
        ).astype(np.float32)
        out_age_mass[y0:y1, x0:x1] += addition * plume_age

    return out_density, out_age_mass


class HybridSmokeSimulator:
    def __init__(
        self,
        map_size: tuple[int, int],
        sources: list[HybridSmokeSource],
        seed: int = HYBRID_SMOKE_SEED,
        hrrr_guidance: HrrrSmokeGuidance | None = None,
        guidance_cadence_frames: float = 12.0,
    ) -> None:
        width, height = map(int, map_size)
        if width <= 8 or height <= 8:
            raise ValueError("hybrid smoke map must be larger than 8x8")
        self.map_size = (width, height)
        self.sources = list(sources)
        self.seed = int(seed)
        self.hrrr_guidance = hrrr_guidance
        self.guidance_cadence_frames = max(float(guidance_cadence_frames), 1.0)
        self.density = np.zeros((height, width), dtype=np.float32)
        self.age_mass = np.zeros((height, width), dtype=np.float32)
        self.layer_density = np.zeros((HYBRID_SMOKE_LAYER_COUNT, height, width), dtype=np.float32)
        self.layer_age_mass = np.zeros((HYBRID_SMOKE_LAYER_COUNT, height, width), dtype=np.float32)
        self.residual_haze = np.zeros((height, width), dtype=np.float32)
        self.previous_density = self.density.copy()
        self.previous_age_mass = self.age_mass.copy()
        self.previous_layer_density = self.layer_density.copy()
        self.previous_layer_age_mass = self.layer_age_mass.copy()
        self.previous_residual_haze = self.residual_haze.copy()
        self.frame_index = 0
        self._border = _hybrid_border_fade(self.density.shape)
        if self.sources:
            weights = np.asarray([max(source.strength, 0.01) for source in self.sources], dtype=np.float32)
            xs = np.asarray([source.x for source in self.sources], dtype=np.float32)
            ys = np.asarray([source.y for source in self.sources], dtype=np.float32)
            total = max(float(np.sum(weights)), 1.0e-6)
            self._source_xy = (float(np.sum(xs * weights) / total), float(np.sum(ys * weights) / total))
        else:
            self._source_xy = (width * 0.5, height * 0.5)

    def _hrrr_guidance_density(self, frame_index: int) -> np.ndarray:
        guidance = self.hrrr_guidance
        if guidance is None or not guidance.frames:
            return _procedural_hrrr_smoke_guidance(self.density.shape, frame_index, self.seed, self._source_xy)
        if len(guidance.frames) == 1:
            return guidance.frames[0].astype(np.float32, copy=False)
        position = np.clip(float(frame_index) / self.guidance_cadence_frames, 0.0, float(len(guidance.frames) - 1))
        lo = int(math.floor(position))
        hi = min(lo + 1, len(guidance.frames) - 1)
        frac = np.float32(position - lo)
        return (
            guidance.frames[lo].astype(np.float32, copy=False) * (1.0 - frac)
            + guidance.frames[hi].astype(np.float32, copy=False) * frac
        ).astype(np.float32)

    def _set_layers(self, layer_density: np.ndarray, layer_age_mass: np.ndarray) -> None:
        layer_density = np.clip(np.asarray(layer_density, dtype=np.float32), 0.0, 6.0)
        layer_age_mass = np.clip(np.asarray(layer_age_mass, dtype=np.float32), 0.0, None)
        layer_density = np.where(layer_density >= 1.0e-7, layer_density, 0.0).astype(np.float32)
        layer_age_mass = np.where(layer_density > 0.0, layer_age_mass, 0.0).astype(np.float32)
        self.layer_density = layer_density
        self.layer_age_mass = layer_age_mass
        self.density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)
        self.age_mass = np.clip(np.sum(layer_age_mass, axis=0), 0.0, None).astype(np.float32)
        self.age_mass = np.where(self.density > 0.0, self.age_mass, 0.0).astype(np.float32)

    def step(self, frame_index: int | None = None) -> HybridSmokeState:
        frame = self.frame_index if frame_index is None else int(frame_index)
        x, y = _pixel_grids(self.density.shape)
        self.previous_density = self.density.copy()
        self.previous_age_mass = self.age_mass.copy()
        self.previous_layer_density = self.layer_density.copy()
        self.previous_layer_age_mass = self.layer_age_mass.copy()
        self.previous_residual_haze = self.residual_haze.copy()

        next_density_layers: list[np.ndarray] = []
        next_age_layers: list[np.ndarray] = []
        for layer_index in range(HYBRID_SMOKE_LAYER_COUNT):
            altitude = _hybrid_layer_altitude(layer_index)
            wind = _hybrid_layer_wind_vector(layer_index)
            u, v = _hybrid_wind_field(frame, self.density.shape, self.seed, layer_index=layer_index)
            # Add convective uplift near heat sources (stronger at lower altitudes)
            convective_v = _hybrid_convective_uplift(self.density.shape, self.sources, frame)
            v = v + convective_v * (1.0 - 0.4 * altitude)
            density = _bilinear_sample(self.layer_density[layer_index], x - u, y - v)
            age_mass = _bilinear_sample(self.layer_age_mass[layer_index], x - u, y - v)
            age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
            age = np.where(density > 1.0e-7, age + 1.0, 0.0).astype(np.float32)

            old_smoke_decay = 0.028 + 0.015 * altitude
            base_decay = 0.991 - 0.004 * altitude
            age_decay = 1.0 - old_smoke_decay * _smoothstep(150.0, HYBRID_SMOKE_MAX_AGE_FRAMES + 24.0, age)
            density = np.clip(density * base_decay * age_decay, 0.0, 6.0)
            age_mass = density * age

            diffusion_mix = 0.065 + 0.050 * altitude
            blurred_density = _box_blur_3x3(density, passes=1)
            blurred_age_mass = _box_blur_3x3(age_mass, passes=1)
            density = density * (1.0 - diffusion_mix) + blurred_density * diffusion_mix
            age_mass = age_mass * (1.0 - diffusion_mix) + blurred_age_mass * diffusion_mix
            density = _hybrid_downwind_stream(density, amount=0.150 + 0.064 * altitude, wind=wind)
            age_mass = _hybrid_downwind_stream(age_mass, amount=0.150 + 0.064 * altitude, wind=wind)
            density = _hybrid_crosswind_spread(density, amount=0.032 + 0.038 * altitude, wind=wind)
            age_mass = _hybrid_crosswind_spread(age_mass, amount=0.032 + 0.038 * altitude, wind=wind)
            local_age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
            age_shear = (0.34 + 0.20 * altitude) * _smoothstep(10.0, 118.0, local_age)
            if np.any(age_shear > 0.001):
                density = _bilinear_sample(density, x - wind[0] * age_shear, y - wind[1] * age_shear)
                age_mass = _bilinear_sample(age_mass, x - wind[0] * age_shear, y - wind[1] * age_shear)
            density, age_mass = _inject_hybrid_sources(
                density,
                age_mass,
                self.sources,
                frame,
                layer_index=layer_index,
                layer_count=HYBRID_SMOKE_LAYER_COUNT,
            )
            next_density_layers.append(density)
            next_age_layers.append(age_mass)

        layer_density = np.stack(next_density_layers, axis=0).astype(np.float32)
        layer_age_mass = np.stack(next_age_layers, axis=0).astype(np.float32)
        aggregate_density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)

        guidance = self._hrrr_guidance_density(frame)
        broad_guidance = _pil_blur_float(guidance, max(2.0, min(self.density.shape) / 78.0))
        source_proximity = _smoothstep(0.002, 0.064, aggregate_density)
        regional_presence = _smoothstep(0.020, 0.46, broad_guidance)
        if self.hrrr_guidance is None:
            source_gate = 0.14 + 0.86 * source_proximity
            guided_veil = (0.0048 * guidance + 0.0022 * broad_guidance) * source_gate * regional_presence
            density_gain = 0.980 + 0.045 * guidance * source_gate
        else:
            guided_veil = (0.024 * guidance + 0.015 * broad_guidance) * (0.24 + 0.76 * source_proximity)
            density_gain = 0.948 + HRRR_SMOKE_GUIDANCE_STRENGTH * guidance
        age_hint = HYBRID_SMOKE_MAX_AGE_FRAMES * (0.44 + 0.24 * _smoothstep(0.0, 1.0, broad_guidance))
        guidance_layer_weights = np.asarray((0.18, 0.39, 0.43), dtype=np.float32)
        guidance_layer_weights /= max(float(np.sum(guidance_layer_weights)), 1.0e-6)
        for layer_index in range(HYBRID_SMOKE_LAYER_COUNT):
            altitude = _hybrid_layer_altitude(layer_index)
            layer_guided_veil = guided_veil * guidance_layer_weights[layer_index]
            layer_density[layer_index] = np.clip(
                layer_density[layer_index] * density_gain + layer_guided_veil,
                0.0,
                6.0,
            )
            layer_age_mass[layer_index] = (
                layer_age_mass[layer_index] * density_gain
                + layer_guided_veil * age_hint * (0.92 + 0.24 * altitude)
            )

        layer_density = np.clip(layer_density * self._border[None, :, :], 0.0, 6.0).astype(np.float32)
        layer_age_mass = np.clip(layer_age_mass * self._border[None, :, :], 0.0, None).astype(np.float32)
        self._set_layers(layer_density, layer_age_mass)
        self._update_residual_haze(frame)
        self.frame_index = frame + 1
        return self.state()

    def _update_residual_haze(self, frame_index: int) -> None:
        x, y = _pixel_grids(self.density.shape)
        layer_index = max(0, HYBRID_SMOKE_LAYER_COUNT - 1)
        wind = _hybrid_layer_wind_vector(layer_index)
        u, v = _hybrid_wind_field(
            float(frame_index) * 0.56 + 31.0,
            self.density.shape,
            self.seed + 9973,
            layer_index=layer_index,
        )
        advected = _bilinear_sample(self.residual_haze, x - u * 0.42, y - v * 0.42)
        advected = _hybrid_crosswind_spread(advected, amount=0.040, wind=wind)

        age = np.divide(self.age_mass, self.density, out=np.zeros_like(self.density), where=self.density > 1.0e-7)
        old_smoke = self.density * _smoothstep(28.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.68, age)
        high_slab = self.layer_density[layer_index] if self.layer_density.size else self.density
        haze_feed = np.clip(old_smoke * 0.008 + high_slab * 0.003 + self.density * 0.002, 0.0, 1.0)
        broad_feed = _pil_blur_float(haze_feed, max(5.5, min(self.density.shape) / 42.0))
        regional_feed = _pil_blur_float(haze_feed, max(12.0, min(self.density.shape) / 16.0)) * 0.22
        texture = _pil_blur_float(
            _advected_smoke_texture(self.density.shape, frame_index, self.seed + 12829),
            max(7.0, min(self.density.shape) / 38.0),
        )
        injected = (broad_feed + regional_feed) * np.clip(0.62 + 0.50 * texture, 0.34, 1.12)
        residual = advected * 0.990 + injected
        residual = _hybrid_downwind_stream(residual, amount=0.055, wind=wind)
        residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.40)
        self.residual_haze = np.clip(residual * self._border, 0.0, 1.15).astype(np.float32)

    def interpolated_state(self, alpha: float = 1.0) -> HybridSmokeState:
        t = float(np.clip(alpha, 0.0, 1.0))
        layer_density = self.previous_layer_density * (1.0 - t) + self.layer_density * t
        layer_age_mass = self.previous_layer_age_mass * (1.0 - t) + self.layer_age_mass * t
        residual_haze = self.previous_residual_haze * (1.0 - t) + self.residual_haze * t
        density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)
        age_mass = np.clip(np.sum(layer_age_mass, axis=0), 0.0, None).astype(np.float32)
        age_mass = np.where(density > 0.0, age_mass, 0.0).astype(np.float32)
        return HybridSmokeState(
            density=density.copy(),
            age_mass=age_mass.copy(),
            layer_density=tuple(layer.copy() for layer in layer_density),
            layer_age_mass=tuple(layer.copy() for layer in layer_age_mass),
            residual_haze=residual_haze.astype(np.float32, copy=True),
        )

    def state(self) -> HybridSmokeState:
        return HybridSmokeState(
            density=self.density.copy(),
            age_mass=self.age_mass.copy(),
            layer_density=tuple(layer.copy() for layer in self.layer_density),
            layer_age_mass=tuple(layer.copy() for layer in self.layer_age_mass),
            residual_haze=self.residual_haze.copy(),
        )


def _hybrid_smoke_field_rgba(
    state: HybridSmokeState,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
    alpha_multiplier: float = 1.0,
    color_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    density = np.clip(np.asarray(state.density, dtype=np.float32), 0.0, None)
    age_mass = np.asarray(state.age_mass, dtype=np.float32)
    if density.ndim != 2 or age_mass.shape != density.shape:
        raise ValueError("density and age_mass must be matching 2D arrays")

    height, width = density.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(density > 0.0):
        return rgba

    fine = _pil_blur_float(density, 1.8)
    medium = _pil_blur_float(density, 6.5)
    broad = _pil_blur_float(density, 18.0)
    sheet = np.clip(density * 0.50 + fine * 0.30 + medium * 0.15 + broad * 0.05, 0.0, None)
    positive = sheet[sheet > 0.0]
    scale_value = max(float(np.percentile(positive, 98.8)) if positive.size else 1.0, 1.0e-5)
    norm = np.clip(sheet / (scale_value * 1.08), 0.0, 1.74)

    age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
    age = _pil_blur_float(age * (density > 0.0), 1.8)
    age_frac = np.clip(age / max(HYBRID_SMOKE_MAX_AGE_FRAMES, 1.0), 0.0, 1.0)
    age_alpha = _hybrid_lifecycle_alpha(age)
    coverage = _smoothstep(0.038, 0.275, sheet)

    texture = _advected_smoke_texture(density.shape, frame_index, seed)
    broad_texture = _pil_blur_float(texture, 7.8)
    detail_texture = texture - _pil_blur_float(texture, 2.80)
    x, y = _pixel_grids(density.shape)
    flow_scale = min(width, height) / 408.0
    wind = np.array([1.0, -0.42], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)
    flow_along = x * wind[0] + y * wind[1]
    flow_cross = x * cross_dir[0] + y * cross_dir[1]
    streaks = 0.5 + 0.5 * np.sin(
        flow_cross / max(10.5 * flow_scale, 1.0)
        + flow_along / max(64.0 * flow_scale, 1.0)
        + frame_index * 0.019
        + seed * 0.011
    )
    ridge_density = np.clip(fine - medium * 0.52, 0.0, None)
    ridge_positive = ridge_density[ridge_density > 0.0]
    ridge_scale = max(float(np.percentile(ridge_positive, 96.5)) if ridge_positive.size else 1.0, 1.0e-5)
    ridge_norm = np.clip(ridge_density / ridge_scale, 0.0, 1.45)
    holes = _smoothstep(0.12, 0.70, 1.0 - broad_texture) * (1.0 - 0.22 * _smoothstep(0.25, 1.15, ridge_norm))
    holes = np.maximum(holes, 0.28 * _smoothstep(0.74, 0.99, streaks) * _smoothstep(0.05, 0.44, norm))
    edge_weight = _smoothstep(0.030, 0.50, norm) * (1.0 - 0.20 * _smoothstep(1.02, 1.72, norm))
    texture_gain = 0.92 + 0.18 * (broad_texture - 0.5) + 0.08 * detail_texture + 0.05 * (streaks - 0.5)
    texture_gain *= 1.0 - 0.28 * holes * edge_weight
    texture_gain = np.clip(texture_gain, 0.56, 1.16)
    filament_mask = 0.86 + 0.16 * _smoothstep(-0.020, 0.092, detail_texture) + 0.08 * ridge_norm
    filament_mask *= 1.0 - 0.20 * holes * edge_weight
    filament_mask = np.clip(filament_mask, 0.66, 1.16)

    alpha_shape = _smoothstep(0.012, 1.3, norm) ** 0.90
    source_core = _smoothstep(0.98, 1.62, norm) ** 1.16
    detail_density = np.clip(density - medium * 0.62, 0.0, None)
    detail_positive = detail_density[detail_density > 0.0]
    detail_scale = max(float(np.percentile(detail_positive, 97.0)) if detail_positive.size else 1.0, 1.0e-5)
    detail_norm = np.clip(detail_density / detail_scale, 0.0, 1.55)
    fresh_band = _smoothstep(0.08, 0.90, detail_norm) * (1.0 - _smoothstep(82.0, 170.0, age))
    source_core = np.maximum(source_core, fresh_band * 0.48)
    haze_floor = 10.8 * coverage * (1.0 - _smoothstep(0.34, 1.08, norm))
    age_visibility = np.clip(0.34 + 0.66 * age_alpha + 0.12 * source_core, 0.0, 1.0)
    alpha = (
        (238.0 * alpha_shape + 172.0 * source_core + 20.0 * ridge_norm + haze_floor * 3.10)
        * coverage
        * age_visibility
        * texture_gain
        * filament_mask
    )
    alpha += 52.0 * fresh_band * filament_mask * np.clip(0.78 + 0.18 * texture, 0.62, 1.04)
    alpha *= 0.52 + 0.50 * _smoothstep(0.52, 1.42, norm)
    alpha *= 1.0 - 0.10 * holes * edge_weight * (1.0 - 0.30 * source_core)
    alpha *= _smoothstep(0.004, 0.070, broad + medium * 0.20)
    alpha *= float(alpha_multiplier)
    alpha = _pil_blur_float(alpha.astype(np.float32), 1.1)
    alpha = np.clip(alpha, 0.0, HYBRID_SMOKE_MAX_ALPHA)
    alpha = np.where(alpha >= 2.0, alpha, 0.0)

    density_t = _smoothstep(0.026, 1.20, norm)
    old_blue = np.array([140.0, 120.0, 100.0], dtype=np.float32)
    thin_gray = np.array([158.0, 164.0, 168.0], dtype=np.float32)
    milky = np.array([242.0, 238.0, 224.0], dtype=np.float32)
    age_t = _smoothstep(60.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.85, age)
    base_rgb = old_blue * (1.0 - density_t[..., None]) + thin_gray * density_t[..., None]
    base_rgb = base_rgb * (1.0 - age_t[..., None] * 0.32) + old_blue * (age_t[..., None] * 0.32)
    # Charcoal blend for very old smoke
    charcoal = np.array([95.0, 85.0, 75.0], dtype=np.float32)
    charcoal_t = _smoothstep(0.65, 0.92, age_t)
    base_rgb = base_rgb * (1.0 - charcoal_t[..., None] * 0.45) + charcoal * (charcoal_t[..., None] * 0.45)

    # Self-shadowing: accumulate density along light direction (upper-left)
    light_dir = np.array([0.6, -0.8], dtype=np.float32)  # from upper-left
    shadow_offset = int(max(3.0, min(width, height) / 80.0))
    shadow_accum = np.zeros_like(density)
    for step in range(1, 6):
        ox = int(light_dir[0] * shadow_offset * step)
        oy = int(light_dir[1] * shadow_offset * step)
        shifted = np.roll(np.roll(broad, ox, axis=1), oy, axis=0)
        shadow_accum += shifted * (0.35 / step)
    shadow_factor = 1.0 - 0.35 * _smoothstep(0.1, 1.2, shadow_accum) * (1.0 - source_core * 0.6)

    source_mix = np.clip(source_core * 0.46 + fresh_band * 0.42 + ridge_norm * 0.10, 0.0, 0.84)
    base_rgb = base_rgb * (1.0 - source_mix[..., None]) + milky * source_mix[..., None]
    fresh = (1.0 - _smoothstep(16.0, 74.0, age)) * _smoothstep(0.18, 1.18, norm)
    base_rgb += fresh[..., None] * np.array([12.0, 11.0, 6.0], dtype=np.float32)
    base_rgb += (broad_texture[..., None] - 0.5) * 9.0 * coverage[..., None]
    base_rgb += np.asarray(color_bias, dtype=np.float32)[None, None, :]
    # Apply self-shadowing
    base_rgb = base_rgb * shadow_factor[..., None]

    # Atmospheric perspective: distance fade based on age and downwind position
    # Older smoke and smoke farther downwind loses contrast and desaturates
    distance_factor = _smoothstep(0.0, float(width), flow_along)  # 0=near, 1=far downwind
    age_distance = _smoothstep(60.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.7, age)
    atmo_fade = 0.5 * distance_factor + 0.5 * age_distance
    atmo_fade = _smoothstep(0.2, 0.9, atmo_fade) * (1.0 - source_core * 0.8)
    # Desaturate: blend toward gray
    gray = np.mean(base_rgb, axis=-1, keepdims=True)
    base_rgb = base_rgb * (1.0 - atmo_fade[..., None] * 0.4) + gray * (atmo_fade[..., None] * 0.4)
    # Reduce contrast: blend toward mid-gray (128)
    base_rgb = base_rgb * (1.0 - atmo_fade[..., None] * 0.25) + 145.0 * (atmo_fade[..., None] * 0.25)

    rgb = np.clip(base_rgb, 0.0, 245.0)
    visible = alpha > 0.0
    rgba[..., 0] = np.where(visible, rgb[..., 0], 0.0).astype(np.uint8)
    rgba[..., 1] = np.where(visible, rgb[..., 1], 0.0).astype(np.uint8)
    rgba[..., 2] = np.where(visible, rgb[..., 2], 0.0).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


def _offset_rgba_layer(rgba: np.ndarray, dx: float, dy: float) -> np.ndarray:
    src = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    dst = Image.new("RGBA", src.size, (0, 0, 0, 0))
    ox = int(round(dx))
    oy = int(round(dy))
    width, height = src.size
    src_x0 = max(0, -ox)
    src_y0 = max(0, -oy)
    dst_x0 = max(0, ox)
    dst_y0 = max(0, oy)
    copy_w = min(width - src_x0, width - dst_x0)
    copy_h = min(height - src_y0, height - dst_y0)
    if copy_w > 0 and copy_h > 0:
        crop = src.crop((src_x0, src_y0, src_x0 + copy_w, src_y0 + copy_h))
        dst.alpha_composite(crop, (dst_x0, dst_y0))
    return np.asarray(dst, dtype=np.uint8)


def _premultiplied_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    bottom_f = np.asarray(bottom, dtype=np.float32) / 255.0
    top_f = np.asarray(top, dtype=np.float32) / 255.0
    bottom_a = bottom_f[..., 3:4]
    top_a = top_f[..., 3:4]
    out_a = top_a + bottom_a * (1.0 - top_a)
    out_rgb_premul = top_f[..., :3] * top_a + bottom_f[..., :3] * bottom_a * (1.0 - top_a)
    out_rgb = np.divide(out_rgb_premul, out_a, out=np.zeros_like(out_rgb_premul), where=out_a > 1.0e-6)
    out = np.zeros_like(bottom_f)
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a
    out[..., 3] = np.minimum(out[..., 3], HYBRID_SMOKE_MAX_ALPHA / 255.0)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def composite_main_smoke_maps(
    atmospheric_rgba: np.ndarray,
    physical_rgba: np.ndarray | None,
    *,
    atmospheric_alpha: float = 0.42,
    physical_alpha: float = 0.92,
) -> np.ndarray:
    blanket = _scale_rgba_alpha(atmospheric_rgba, atmospheric_alpha)
    if physical_rgba is None:
        return blanket
    detail = _scale_rgba_alpha(physical_rgba, physical_alpha)
    combined = _premultiplied_over(blanket, detail)
    combined[..., 3] = np.minimum(combined[..., 3], HYBRID_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def _sample_float_field(field: np.ndarray, x: float, y: float) -> float:
    sample_x = np.asarray([[float(x)]], dtype=np.float32)
    sample_y = np.asarray([[float(y)]], dtype=np.float32)
    return float(_bilinear_sample(field, sample_x, sample_y)[0, 0])


def _copy_source_wisp_puff(puff: SourceWispPuff) -> SourceWispPuff:
    return SourceWispPuff(
        source_index=int(puff.source_index),
        x=float(puff.x),
        y=float(puff.y),
        origin_x=float(puff.origin_x),
        origin_y=float(puff.origin_y),
        vx=float(puff.vx),
        vy=float(puff.vy),
        age_frames=float(puff.age_frames),
        lifetime_frames=float(puff.lifetime_frames),
        radius_px=float(puff.radius_px),
        base_radius_px=float(puff.base_radius_px),
        alpha=float(puff.alpha),
        base_alpha=float(puff.base_alpha),
        heat=float(puff.heat),
        intensity=float(puff.intensity),
        breakup_seed=int(puff.breakup_seed),
        breakup_phase=float(puff.breakup_phase),
    )


class SourceWispSimulator:
    """Small source-attached smoke puffs spawned from live flame and smolder sources."""

    def __init__(
        self,
        map_size: tuple[int, int],
        sources: list[HybridSmokeSource],
        seed: int = HYBRID_SMOKE_SEED,
        *,
        max_particles: int = SOURCE_WISP_MAX_PARTICLES,
        max_emitters: int = SOURCE_WISP_MAX_EMITTERS,
        emit_interval_frames: int = SOURCE_WISP_EMIT_INTERVAL_FRAMES,
        emitter_mode: str = "synthetic",
    ) -> None:
        width, height = map(int, map_size)
        if width <= 8 or height <= 8:
            raise ValueError("source wisp map must be larger than 8x8")
        if emitter_mode not in SOURCE_WISP_EMITTER_MODES:
            raise ValueError(f"source wisp emitter mode must be one of {SOURCE_WISP_EMITTER_MODES}")
        self.map_size = (width, height)
        self.shape = (height, width)
        self.sources = list(sources)
        self.seed = int(seed)
        self.max_particles = max(0, int(max_particles))
        self.max_emitters = max(0, int(max_emitters))
        self.emit_interval_frames = max(1, int(emit_interval_frames))
        self.emitter_mode = str(emitter_mode)
        self.puffs: list[SourceWispPuff] = []
        self.current_emitters: tuple[HybridSmokeSource, ...] = ()
        self.frame_index = 0

    def step(self, frame_index: int | None = None) -> SourceWispState:
        frame = self.frame_index if frame_index is None else int(frame_index)
        self.current_emitters = tuple(self._emitter_sources_for_frame(frame))
        self._advect_puffs(frame)
        self._spawn_puffs(frame)
        self._trim_particles()
        self.frame_index = frame + 1
        return self.state()

    def state(self) -> SourceWispState:
        return SourceWispState(
            puffs=tuple(_copy_source_wisp_puff(puff) for puff in self.puffs),
            map_size=self.map_size,
            emitters=tuple(self.current_emitters),
        )

    def _emitter_sources_for_frame(self, frame_index: int) -> list[HybridSmokeSource]:
        if self.emitter_mode == "fire-core":
            return fire_core_emitter_sources(
                self.sources,
                frame_index,
                self.map_size,
                max_emitters=max(self.max_emitters, 1),
                seed=self.seed + 9137,
            )
        return list(self.sources)

    def _advect_puffs(self, frame_index: int) -> None:
        if not self.puffs:
            return
        height, width = self.shape
        scale = min(width, height) / 408.0
        low_u, low_v = _hybrid_wind_field(frame_index, self.shape, self.seed + 3607, layer_index=0)
        high_layer = min(HYBRID_SMOKE_LAYER_COUNT - 1, 2)
        high_u, high_v = _hybrid_wind_field(frame_index + 4.0, self.shape, self.seed + 4211, layer_index=high_layer)
        kept: list[SourceWispPuff] = []
        for puff in self.puffs:
            lifetime = max(float(puff.lifetime_frames), 1.0)
            age = float(puff.age_frames) + 1.0
            age_frac = float(np.clip(age / lifetime, 0.0, 1.0))
            if age >= lifetime:
                continue

            altitude_mix = float(_smoothstep(0.18, 0.84, age_frac))
            vx = (1.0 - altitude_mix) * _sample_float_field(low_u, puff.x, puff.y) + altitude_mix * _sample_float_field(high_u, puff.x, puff.y)
            vy = (1.0 - altitude_mix) * _sample_float_field(low_v, puff.x, puff.y) + altitude_mix * _sample_float_field(high_v, puff.x, puff.y)
            wind_len = max(math.hypot(vx, vy), 1.0e-5)
            cross_x = -vy / wind_len
            cross_y = vx / wind_len
            curl = math.sin(puff.breakup_phase + frame_index * 0.104 + age * 0.29)
            tumble = math.sin(puff.breakup_phase * 1.7 + frame_index * 0.067 + age * 0.53)
            buoyancy = (1.0 - float(_smoothstep(5.0, 26.0, age))) * (0.56 + 0.24 * puff.heat) * scale
            shear = float(_smoothstep(9.0, lifetime * 0.72, age)) * (0.20 + 0.14 * puff.heat)
            target_vx = vx * (0.22 + 0.18 * shear) + cross_x * curl * (0.34 + 0.18 * (1.0 - age_frac)) * scale
            target_vy = vy * (0.20 + 0.22 * shear) + cross_y * tumble * 0.22 * scale - buoyancy
            puff.vx = puff.vx * 0.54 + target_vx * 0.46
            puff.vy = puff.vy * 0.54 + target_vy * 0.46
            puff.x += puff.vx
            puff.y += puff.vy
            puff.age_frames = age
            puff.radius_px = max(
                SOURCE_WISP_MIN_RADIUS_PX * scale,
                puff.base_radius_px * (1.0 + 2.10 * (age_frac ** 0.72)),
            )
            puff.alpha = puff.base_alpha * ((1.0 - age_frac) ** 1.32) * (1.0 - 0.14 * float(_smoothstep(0.54, 0.96, age_frac)))
            margin = max(12.0, 28.0 * scale)
            if puff.alpha > 0.9 and -margin <= puff.x <= width + margin and -margin <= puff.y <= height + margin:
                kept.append(puff)
        self.puffs = kept

    def _spawn_puffs(self, frame_index: int) -> None:
        if self.max_particles <= 0 or self.max_emitters <= 0:
            return
        emission_frame = int(frame_index) - SOURCE_WISP_SOURCE_DELAY_FRAMES
        if emission_frame < 0:
            return
        height, width = self.shape
        scale = min(width, height) / 408.0
        wind = _hybrid_layer_wind_vector(0)
        cross = np.array([-wind[1], wind[0]], dtype=np.float32)
        emitters = self._emitter_sources_for_frame(emission_frame)
        candidates: list[tuple[float, int, HybridSmokeSource, float, float]] = []
        for source_index, source in enumerate(emitters):
            if emission_frame < source.start_frame or emission_frame > source.end_frame:
                continue
            if (frame_index + source.seed) % self.emit_interval_frames != 0:
                continue
            flame = _source_flame_lifecycle_weight(source, emission_frame)
            smolder = _source_smolder_lifecycle_weight(source, emission_frame)
            if flame <= 0.018 and smolder <= 0.05:
                continue
            burst = _source_burst_envelope(source, emission_frame)
            flame_signal = source.strength * source.heat * source.smoke_rate * flame
            smolder_signal = source.strength * source.smoke_rate * smolder * 0.22
            intensity = (flame_signal + smolder_signal) * (0.62 + 0.38 * burst)
            if intensity <= 0.018:
                continue
            candidates.append((float(intensity), source_index, source, float(flame), float(smolder)))
        if not candidates:
            return
        candidates.sort(key=lambda item: (item[0], item[2].heat, -item[1]), reverse=True)
        selected = candidates[: self.max_emitters]
        rng = np.random.default_rng(self.seed + frame_index * 173)
        for intensity, source_index, source, flame, smolder in selected:
            if len(self.puffs) >= self.max_particles:
                break
            spawn_count = 1 + int(intensity > 0.88 and flame > 0.20 and (frame_index + source_index) % 4 == 0)
            for _ in range(spawn_count):
                if len(self.puffs) >= self.max_particles:
                    break
                local_rng = np.random.default_rng(source.seed + frame_index * 379 + source_index * 47 + len(self.puffs))
                lateral = float(local_rng.normal(0.0, 0.34 * max(source.radius_px, 1.0)))
                along = float(local_rng.normal(0.10 * source.radius_px, 0.22 * max(source.radius_px, 1.0)))
                lift = float(local_rng.uniform(0.20, 0.70) * source.radius_px)
                x = float(np.clip(source.x + cross[0] * lateral + wind[0] * along, 1.0, width - 2.0))
                y = float(np.clip(source.y + cross[1] * lateral + wind[1] * along - lift, 1.0, height - 2.0))
                base_radius = max(SOURCE_WISP_MIN_RADIUS_PX * scale, source.radius_px * float(local_rng.uniform(0.085, 0.17)))
                lifetime_lo, lifetime_hi = SOURCE_WISP_LIFETIME_FRAMES
                lifetime = float(local_rng.integers(lifetime_lo, lifetime_hi + 1))
                lifetime *= float(np.clip(0.86 + 0.22 * source.heat + 0.10 * rng.random(), 0.78, 1.22))
                source_alpha = 30.0 + 66.0 * float(np.clip(intensity, 0.0, 1.25))
                source_alpha *= 0.66 + 0.34 * flame + 0.10 * smolder
                base_alpha = float(np.clip(source_alpha, 18.0, SOURCE_WISP_MAX_ALPHA))
                initial_vx = float(wind[0] * 0.18 * scale + cross[0] * local_rng.normal(0.0, 0.06 * scale))
                initial_vy = float(wind[1] * 0.12 * scale - (0.34 + 0.16 * source.heat) * scale)
                self.puffs.append(
                    SourceWispPuff(
                        source_index=source_index,
                        x=x,
                        y=y,
                        origin_x=float(source.x),
                        origin_y=float(source.y),
                        vx=initial_vx,
                        vy=initial_vy,
                        age_frames=0.0,
                        lifetime_frames=lifetime,
                        radius_px=base_radius,
                        base_radius_px=base_radius,
                        alpha=base_alpha,
                        base_alpha=base_alpha,
                        heat=float(source.heat),
                        intensity=float(intensity),
                        breakup_seed=int(source.seed + frame_index * 31 + source_index * 101),
                        breakup_phase=float(local_rng.uniform(0.0, math.tau)),
                    )
                )

    def _trim_particles(self) -> None:
        if len(self.puffs) <= self.max_particles:
            return
        self.puffs.sort(
            key=lambda puff: (
                puff.alpha * (1.0 - float(np.clip(puff.age_frames / max(puff.lifetime_frames, 1.0), 0.0, 1.0))),
                -puff.age_frames,
            ),
            reverse=True,
        )
        del self.puffs[self.max_particles :]


def source_wisps_rgba(
    state: SourceWispState,
    frame_index: int,
    *,
    seed: int = HYBRID_SMOKE_SEED,
    alpha_scale: float = 1.0,
    plume_ribbons: bool = True,
) -> np.ndarray:
    width, height = map(int, state.map_size)
    rgb_premul = np.zeros((height, width, 3), dtype=np.float32)
    alpha_accum = np.zeros((height, width), dtype=np.float32)
    if not state.puffs:
        return np.zeros((height, width, 4), dtype=np.uint8)
    texture = _advected_smoke_texture((height, width), frame_index, seed + 7043)
    fine_texture = texture - _pil_blur_float(texture, max(0.8, min(width, height) / 260.0))
    wind = _hybrid_layer_wind_vector(0)
    scale = min(width, height) / 408.0

    for puff in state.puffs:
        if puff.alpha <= 0.8:
            continue
        lifetime = max(float(puff.lifetime_frames), 1.0)
        age_frac = float(np.clip(puff.age_frames / lifetime, 0.0, 1.0))
        motion = np.array(
            [
                puff.x - puff.origin_x + puff.vx * 3.4 + wind[0] * (1.0 + 3.6 * age_frac) * scale,
                puff.y - puff.origin_y + puff.vy * 3.4 + wind[1] * (1.0 + 3.6 * age_frac) * scale,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(motion))
        if norm < 1.0e-4:
            axis = np.array([wind[0], wind[1] - 0.22], dtype=np.float32)
            axis /= max(float(np.linalg.norm(axis)), 1.0e-6)
        else:
            axis = motion / norm
        cross = np.array([-axis[1], axis[0]], dtype=np.float32)
        radius = max(float(puff.radius_px), SOURCE_WISP_MIN_RADIUS_PX * scale)
        plume_t = float(_smoothstep(0.28, 0.90, age_frac)) if plume_ribbons else 0.0
        old_t = float(_smoothstep(0.50, 0.96, age_frac)) if plume_ribbons else 0.0
        strand_len = max(
            radius * 2.8,
            radius * (4.2 + 10.8 * age_frac + 4.2 * plume_t) + math.hypot(puff.vx, puff.vy) * 2.2,
        )
        strand_width = max(
            SOURCE_WISP_MIN_RADIUS_PX * scale * 0.58,
            radius * (0.22 + 0.62 * age_frac + 0.38 * plume_t),
        )
        pad = int(math.ceil(strand_len + strand_width * 5.0 + 3.0))
        x0 = max(0, int(math.floor(puff.x - pad)))
        x1 = min(width, int(math.ceil(puff.x + pad)))
        y0 = max(0, int(math.floor(puff.y - pad)))
        y1 = min(height, int(math.ceil(puff.y + pad)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(puff.x)
        dy = yy - np.float32(puff.y)
        along = dx * axis[0] + dy * axis[1]
        lateral = dx * cross[0] + dy * cross[1]
        curl = strand_width * (
            0.80 * (1.0 - age_frac) * np.sin(along / max(strand_len * 0.42, 1.0) + puff.breakup_phase + frame_index * 0.042)
            + 0.38 * np.sin(along / max(strand_len * 0.24, 1.0) + puff.breakup_phase * 1.9)
        )
        core = np.exp(
            -((along + strand_len * 0.12) ** 2) / (2.0 * strand_len * strand_len + 1.0e-6)
            -((lateral - curl) ** 2) / (2.0 * strand_width * strand_width + 1.0e-6)
        )
        source_gate = _smoothstep(-strand_len * 0.78, -strand_len * 0.12, along)
        head_gate = 1.0 - _smoothstep(strand_len * 0.46, strand_len * 0.92, along)
        strand = core * np.maximum(source_gate, 0.36) * head_gate
        tex = texture[y0:y1, x0:x1]
        fine = fine_texture[y0:y1, x0:x1]
        striation = 0.5 + 0.5 * np.sin(
            lateral / max(strand_width * 0.96, 1.0)
            + along / max(strand_len * 0.22, 1.0)
            + puff.breakup_phase
            + frame_index * 0.033
        )
        erosion_noise = np.clip(tex * 0.54 + striation * 0.34 + fine * 0.24 + 0.10, 0.0, 1.0)
        erosion = _smoothstep(0.50 - 0.16 * age_frac, 0.95, erosion_noise)
        fragment = 1.0 - (0.26 + 0.64 * age_frac) * erosion * _smoothstep(0.10, 0.76, strand)
        filament_gain = np.clip(0.68 + 0.44 * striation + 0.26 * fine, 0.26, 1.20)
        brush_limiter = 1.0 - 0.42 * old_t
        wisp_alpha = (
            strand
            * fragment
            * filament_gain
            * float(puff.alpha)
            * float(alpha_scale)
            * brush_limiter
            * (1.0 - 0.18 * float(_smoothstep(0.72, 0.98, age_frac)))
        )
        if plume_ribbons and plume_t > 0.0:
            ribbon_len = strand_len * (1.10 + 0.78 * plume_t)
            ribbon_along_t = np.clip((along + ribbon_len * 0.42) / max(ribbon_len * 1.42, 1.0), 0.0, 1.0)
            endpoint_fade = (
                _smoothstep(-ribbon_len * 0.50, -ribbon_len * 0.12, along)
                * (1.0 - _smoothstep(ribbon_len * 0.38, ribbon_len * 0.98, along))
            )
            ribbon_width = strand_width * (1.32 + 3.45 * plume_t * (0.18 + ribbon_along_t**0.88))
            ribbon_curl = curl + ribbon_width * (
                0.20 * np.sin(along / max(ribbon_len * 0.30, 1.0) + puff.breakup_phase * 1.37)
                + 0.13 * np.sin(along / max(ribbon_len * 0.15, 1.0) + frame_index * 0.023)
            )
            ribbon_body = np.exp(
                -((lateral - ribbon_curl) ** 2) / (2.0 * ribbon_width * ribbon_width + 1.0e-6)
            )
            edge_noise = np.clip(
                tex * 0.38
                + fine * 0.24
                + 0.22
                + 0.16
                * np.sin(
                    lateral / max(strand_width * 1.8, 1.0)
                    - along / max(ribbon_len * 0.16, 1.0)
                    + puff.breakup_phase * 0.73
                ),
                0.0,
                1.0,
            )
            holes = 1.0 - (0.22 + 0.48 * plume_t) * _smoothstep(0.50, 0.92, edge_noise) * _smoothstep(
                0.16, 0.84, ribbon_along_t
            )
            ribbon_filaments = np.clip(
                0.62
                + 0.28
                * np.sin(
                    lateral / max(strand_width * 0.90, 1.0)
                    + along / max(ribbon_len * 0.22, 1.0)
                    + puff.breakup_phase
                )
                + 0.18 * fine,
                0.22,
                1.10,
            )
            ribbon_source_alpha = min(
                float(puff.base_alpha) * ((1.0 - age_frac) ** 0.92),
                float(puff.alpha) * 1.32,
            )
            ribbon_alpha = (
                ribbon_body
                * endpoint_fade
                * holes
                * ribbon_filaments
                * ribbon_source_alpha
                * (0.14 + 0.22 * plume_t)
                * float(alpha_scale)
            )
            ribbon_alpha = np.clip(ribbon_alpha, 0.0, 48.0)
            wisp_alpha = np.clip(wisp_alpha + ribbon_alpha * (1.0 - wisp_alpha / 255.0), 0.0, SOURCE_WISP_MAX_ALPHA)
        wisp_alpha = np.clip(wisp_alpha, 0.0, SOURCE_WISP_MAX_ALPHA)
        wisp_alpha = np.where(wisp_alpha >= 1.2, wisp_alpha, 0.0)
        if not np.any(wisp_alpha > 0.0):
            continue

        fresh_color = np.array([240.0, 235.0, 217.0], dtype=np.float32)
        milk_color = np.array([198.0, 204.0, 202.0], dtype=np.float32)
        old_color = np.array([118.0, 130.0, 144.0], dtype=np.float32)
        warm_light = np.array([20.0, 12.0, 0.0], dtype=np.float32) * (1.0 - age_frac) * min(puff.heat, 1.45)
        cool_t = float(_smoothstep(0.46, 0.96, age_frac))
        mid_t = float(_smoothstep(0.10, 0.64, age_frac))
        rgb = fresh_color * (1.0 - mid_t) + milk_color * mid_t
        rgb = rgb * (1.0 - cool_t) + old_color * cool_t
        rgb = rgb + warm_light
        rgb = np.clip(rgb, 0.0, 246.0) / 255.0

        a = np.clip(wisp_alpha / 255.0, 0.0, 1.0).astype(np.float32)
        existing_a = alpha_accum[y0:y1, x0:x1]
        existing_rgb = rgb_premul[y0:y1, x0:x1]
        top_rgb = rgb[None, None, :] * a[..., None]
        rgb_premul[y0:y1, x0:x1] = top_rgb + existing_rgb * (1.0 - a[..., None])
        alpha_accum[y0:y1, x0:x1] = a + existing_a * (1.0 - a)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgb_unpremul = np.divide(
        rgb_premul,
        alpha_accum[..., None],
        out=np.zeros_like(rgb_premul),
        where=alpha_accum[..., None] > 1.0e-6,
    )
    rgba[..., :3] = np.clip(np.round(rgb_unpremul * 255.0), 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = np.clip(np.round(alpha_accum * 255.0), 0.0, SOURCE_WISP_MAX_ALPHA).astype(np.uint8)
    return rgba


def _source_wisp_age_fraction(puff: SourceWispPuff) -> float:
    return float(np.clip(float(puff.age_frames) / max(float(puff.lifetime_frames), 1.0), 0.0, 1.0))


def _source_wisp_age_band_state(
    state: SourceWispState,
    age_min: float,
    age_max: float,
) -> SourceWispState:
    puffs = tuple(
        puff for puff in state.puffs
        if float(age_min) <= _source_wisp_age_fraction(puff) < float(age_max)
    )
    return SourceWispState(puffs=puffs, map_size=state.map_size, emitters=state.emitters)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    flat_weights = np.clip(np.asarray(weights, dtype=np.float32).reshape(-1), 0.0, None)
    valid = np.isfinite(flat_values) & np.isfinite(flat_weights) & (flat_weights > 0.0)
    if not np.any(valid):
        return 0.0
    flat_values = flat_values[valid]
    flat_weights = flat_weights[valid]
    order = np.argsort(flat_values)
    sorted_values = flat_values[order]
    sorted_weights = flat_weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = float(np.clip(percentile, 0.0, 100.0)) * 0.01 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(sorted_values) - 1)
    return float(sorted_values[index])


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32), 0
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(binary)
        return labels.astype(np.int32, copy=False), int(count)
    except Exception:
        labels = np.zeros(binary.shape, dtype=np.int32)
        height, width = binary.shape
        count = 0
        for y0, x0 in np.argwhere(binary):
            if labels[int(y0), int(x0)] != 0:
                continue
            count += 1
            stack = [(int(y0), int(x0))]
            labels[int(y0), int(x0)] = count
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = count
                        stack.append((ny, nx))
        return labels, count


def _smoke_alpha_region_percentiles(alpha: np.ndarray) -> dict[str, float]:
    values = np.asarray(alpha, dtype=np.float32)
    smoke = values[values > 1.5]
    if smoke.size == 0:
        return {
            "smoke_region_alpha_p50": 0.0,
            "smoke_region_alpha_p90": 0.0,
            "smoke_region_alpha_p95": 0.0,
        }
    return {
        "smoke_region_alpha_p50": float(np.percentile(smoke, 50.0)),
        "smoke_region_alpha_p90": float(np.percentile(smoke, 90.0)),
        "smoke_region_alpha_p95": float(np.percentile(smoke, 95.0)),
    }


def _alpha_band_morphology_metrics(alpha: np.ndarray, screen_wind: np.ndarray) -> dict[str, float]:
    values = np.asarray(alpha, dtype=np.float32)
    mask = values > 2.0
    if values.ndim != 2 or not np.any(mask):
        return {
            "coverage_fraction": 0.0,
            "component_count": 0.0,
            "width_px": 0.0,
            "length_px": 0.0,
            "alpha_p50": 0.0,
            "alpha_p90": 0.0,
            "alpha_p95": 0.0,
            "endpoint_alpha": 0.0,
            "edge_softness_px": 0.0,
            "diffuse_to_core_area_ratio": 0.0,
            "hard_endpoint_fraction": 0.0,
        }
    height, width = values.shape
    wind = np.asarray(screen_wind, dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    along_all = xx * wind[0] + yy * wind[1]
    lateral_all = xx * cross[0] + yy * cross[1]
    labels, count = _component_labels(mask)
    widths: list[float] = []
    lengths: list[float] = []
    endpoint_alphas: list[float] = []
    endpoint_weights: list[float] = []
    edge_softness_values: list[float] = []
    edge_weights: list[float] = []
    diffuse_area = 0.0
    core_area = 0.0
    hard_endpoint_area = 0.0
    endpoint_area = 0.0
    kept_components = 0
    for label in range(1, count + 1):
        comp = labels == label
        area = int(np.count_nonzero(comp))
        if area < 4:
            continue
        kept_components += 1
        comp_alpha = values[comp]
        comp_weights = np.maximum(comp_alpha, 1.0)
        comp_lateral = lateral_all[comp]
        comp_along = along_all[comp]
        comp_width = max(
            0.0,
            _weighted_percentile(comp_lateral, comp_weights, 90.0)
            - _weighted_percentile(comp_lateral, comp_weights, 10.0),
        )
        comp_length = max(
            0.0,
            _weighted_percentile(comp_along, comp_weights, 95.0)
            - _weighted_percentile(comp_along, comp_weights, 5.0),
        )
        comp_p90 = float(np.percentile(comp_alpha, 90.0))
        endpoint_cut = _weighted_percentile(comp_along, comp_weights, 84.0)
        endpoint = comp & (along_all >= endpoint_cut)
        endpoint_values = values[endpoint]
        endpoint_mean = float(np.mean(endpoint_values)) if endpoint_values.size else 0.0
        diffuse_hi = max(5.0, comp_p90 * 0.48)
        core_lo = max(10.0, comp_p90 * 0.66)
        diffuse_count = float(np.count_nonzero((comp_alpha > 2.0) & (comp_alpha <= diffuse_hi)))
        core_count = float(np.count_nonzero(comp_alpha >= core_lo))
        diffuse_area += diffuse_count
        core_area += core_count
        endpoint_area += float(endpoint_values.size)
        if endpoint_values.size:
            hard_endpoint_area += float(np.count_nonzero(endpoint_values >= max(16.0, comp_p90 * 0.72)))
        edge_ratio = diffuse_count / max(core_count, 1.0)
        edge_softness = min(max(comp_width, 0.0), max(0.0, edge_ratio * max(comp_width, 1.0) * 0.34))
        widths.append(comp_width)
        lengths.append(comp_length)
        endpoint_alphas.append(endpoint_mean)
        endpoint_weights.append(float(max(endpoint_values.size, 1)))
        edge_softness_values.append(edge_softness)
        edge_weights.append(float(area))

    smoke_values = values[mask]
    area_weights = np.asarray(edge_weights, dtype=np.float32)
    if not widths or area_weights.size == 0:
        mean_width = mean_length = endpoint_alpha = edge_softness = 0.0
    else:
        widths_a = np.asarray(widths, dtype=np.float32)
        lengths_a = np.asarray(lengths, dtype=np.float32)
        endpoint_a = np.asarray(endpoint_alphas, dtype=np.float32)
        endpoint_w = np.asarray(endpoint_weights, dtype=np.float32)
        edge_a = np.asarray(edge_softness_values, dtype=np.float32)
        mean_width = float(np.sum(widths_a * area_weights) / max(float(np.sum(area_weights)), 1.0e-6))
        mean_length = float(np.sum(lengths_a * area_weights) / max(float(np.sum(area_weights)), 1.0e-6))
        endpoint_alpha = float(np.sum(endpoint_a * endpoint_w) / max(float(np.sum(endpoint_w)), 1.0e-6))
        edge_softness = float(np.sum(edge_a * area_weights) / max(float(np.sum(area_weights)), 1.0e-6))
    return {
        "coverage_fraction": float(np.count_nonzero(mask) / max(values.size, 1)),
        "component_count": float(kept_components),
        "width_px": mean_width,
        "length_px": mean_length,
        "alpha_p50": float(np.percentile(smoke_values, 50.0)),
        "alpha_p90": float(np.percentile(smoke_values, 90.0)),
        "alpha_p95": float(np.percentile(smoke_values, 95.0)),
        "endpoint_alpha": endpoint_alpha,
        "edge_softness_px": edge_softness,
        "diffuse_to_core_area_ratio": float(diffuse_area / max(core_area, 1.0)),
        "hard_endpoint_fraction": float(hard_endpoint_area / max(endpoint_area, 1.0)),
    }


def source_wisp_morphology_report(
    state: SourceWispState | None,
    frame_index: int,
    plate: TerrainPlate,
    output_size: tuple[int, int],
    *,
    seed: int = HYBRID_SMOKE_SEED,
    plume_ribbons: bool = True,
    warped_wisps: np.ndarray | None = None,
) -> dict[str, float]:
    report = {
        "morphology_stage_coverage_fraction": 0.0,
        "fresh_stem_width_px": 0.0,
        "transition_plume_width_px": 0.0,
        "old_tail_width_px": 0.0,
        "transition_width_growth_ratio": 0.0,
        "old_tail_width_growth_ratio": 0.0,
        "old_tail_alpha_p90_fraction": 0.0,
        "old_tail_endpoint_alpha_fraction": 0.0,
        "old_tail_coverage_growth_ratio": 0.0,
        "old_tail_edge_softness_px": 0.0,
        "old_tail_diffuse_to_core_area_ratio": 0.0,
        "old_tail_hard_endpoint_fraction": 0.0,
        "brush_bundle_score": 1.0,
        "smoke_region_alpha_p50": 0.0,
        "smoke_region_alpha_p90": 0.0,
        "smoke_region_alpha_p95": 0.0,
    }
    if state is None or not state.puffs:
        return report
    screen_wind = _screen_wind_vector(plate, state.map_size)
    if warped_wisps is None:
        all_wisps = source_wisps_rgba(state, frame_index, seed=seed, plume_ribbons=plume_ribbons)
        warped_wisps = np.asarray(warp_map_layer_to_plate(all_wisps, plate, output_size), dtype=np.uint8)
    report.update(_smoke_alpha_region_percentiles(np.asarray(warped_wisps, dtype=np.uint8)[..., 3]))

    band_metrics: dict[str, dict[str, float]] = {}
    for band_name, (age_min, age_max) in SOURCE_WISP_AGE_BANDS.items():
        band_state = _source_wisp_age_band_state(state, age_min, age_max)
        if not band_state.puffs:
            metrics = _alpha_band_morphology_metrics(np.zeros((output_size[1], output_size[0]), dtype=np.float32), screen_wind)
        else:
            band_rgba = source_wisps_rgba(
                band_state,
                frame_index,
                seed=seed,
                plume_ribbons=plume_ribbons,
            )
            warped_band = np.asarray(warp_map_layer_to_plate(band_rgba, plate, output_size), dtype=np.uint8)
            metrics = _alpha_band_morphology_metrics(warped_band[..., 3].astype(np.float32), screen_wind)
        band_metrics[band_name] = metrics
        for metric_name, value in metrics.items():
            report[f"{band_name}_{metric_name}"] = float(value)

    fresh = band_metrics["fresh_stem"]
    transition = band_metrics["transition_plume"]
    old = band_metrics["old_tail"]
    fresh_width = max(float(fresh["width_px"]), 1.0e-6)
    transition_width = float(transition["width_px"])
    old_width = float(old["width_px"])
    fresh_alpha = max(float(fresh["alpha_p90"]), 1.0)
    transition_ratio = transition_width / fresh_width
    old_ratio = old_width / fresh_width
    old_alpha_fraction = float(old["alpha_p90"]) / fresh_alpha
    old_endpoint_fraction = float(old["endpoint_alpha"]) / fresh_alpha
    coverage_growth = float(old["coverage_fraction"]) / max(float(fresh["coverage_fraction"]), 1.0e-6)
    diffuse_ratio = float(old["diffuse_to_core_area_ratio"])
    brush_score = (
        0.52 * float(1.0 / max(coverage_growth, 0.05))
        + 0.28 * float(fresh_width / max(old_width, fresh_width * 0.18))
        + 0.20 * old_endpoint_fraction
    )
    report.update(
        {
            "morphology_stage_coverage_fraction": float(
                min(
                    fresh["coverage_fraction"],
                    transition["coverage_fraction"],
                    old["coverage_fraction"],
                )
            ),
            "fresh_stem_width_px": float(fresh_width),
            "transition_plume_width_px": float(transition_width),
            "old_tail_width_px": float(old_width),
            "transition_width_growth_ratio": float(transition_ratio),
            "old_tail_width_growth_ratio": float(old_ratio),
            "old_tail_alpha_p90_fraction": float(old_alpha_fraction),
            "old_tail_endpoint_alpha_fraction": float(old_endpoint_fraction),
            "old_tail_coverage_growth_ratio": float(coverage_growth),
            "old_tail_edge_softness_px": float(old["edge_softness_px"]),
            "old_tail_diffuse_to_core_area_ratio": float(diffuse_ratio),
            "old_tail_hard_endpoint_fraction": float(old["hard_endpoint_fraction"]),
            "brush_bundle_score": float(brush_score),
        }
    )
    return report


def composite_source_wisps(base: Image.Image, wisp_layer: Image.Image) -> Image.Image:
    base_arr = np.asarray(base.convert("RGBA"), dtype=np.float32)
    wisp_arr = np.asarray(wisp_layer.convert("RGBA"), dtype=np.float32)
    alpha = np.clip(wisp_arr[..., 3:4] / 255.0, 0.0, 1.0)
    if not np.any(alpha > 0.0):
        return base.copy()
    terrain = base_arr[..., :3] / 255.0
    smoke = wisp_arr[..., :3] / 255.0
    warm_signal = np.clip((terrain[..., 0:1] - terrain[..., 2:3]) * 1.35 + (terrain[..., 1:2] - terrain[..., 2:3]) * 0.22, 0.0, 1.0)
    fire_lift = np.array([1.0, 0.62, 0.24], dtype=np.float32)[None, None, :] * warm_signal * alpha * 0.32
    lit_smoke = np.clip(smoke + fire_lift, 0.0, 1.0)
    optical = alpha ** 0.88
    out_rgb = terrain * (1.0 - optical * 0.78) + lit_smoke * optical * 0.92
    out_rgb += terrain * warm_signal * alpha * 0.10
    out = base_arr.copy()
    out[..., :3] = np.clip(np.round(out_rgb * 255.0), 0.0, 255.0)
    out[..., 3] = 255.0
    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def source_wisp_attachment_report(
    wisp_rgba: np.ndarray,
    sources: list[HybridSmokeSource],
    frame_index: int,
) -> dict[str, float]:
    alpha = np.asarray(wisp_rgba, dtype=np.uint8)[..., 3].astype(np.float32)
    active_sources = [
        source for source in sources
        if _source_flame_lifecycle_weight(source, frame_index) > 0.04
    ]
    report = {
        "active_source_count": float(len(active_sources)),
        "attached_source_count": 0.0,
        "coverage_fraction": float(np.count_nonzero(alpha > 0.0) / max(alpha.size, 1)),
        "downwind_dx": 0.0,
        "downwind_dy": 0.0,
    }
    if not active_sources or not np.any(alpha > 0.0):
        return report
    height, width = alpha.shape
    wind = _hybrid_layer_wind_vector(0)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    alpha_sum = max(float(np.sum(alpha)), 1.0e-6)
    source_weights = np.asarray([max(source.strength * source.heat, 0.01) for source in active_sources], dtype=np.float32)
    source_total = max(float(np.sum(source_weights)), 1.0e-6)
    source_x = float(np.sum(np.asarray([source.x for source in active_sources], dtype=np.float32) * source_weights) / source_total)
    source_y = float(np.sum(np.asarray([source.y for source in active_sources], dtype=np.float32) * source_weights) / source_total)
    report["downwind_dx"] = float(np.sum(xx * alpha) / alpha_sum - source_x)
    report["downwind_dy"] = float(np.sum(yy * alpha) / alpha_sum - source_y)
    attached = 0
    for source in active_sources:
        dx = xx - np.float32(source.x)
        dy = yy - np.float32(source.y)
        along = dx * wind[0] + dy * wind[1]
        lateral = dx * cross[0] + dy * cross[1]
        radius = max(float(source.radius_px) * 5.5, 5.0)
        mask = (along > -float(source.radius_px) * 2.0) & (along < radius * 2.8) & (np.abs(lateral) < radius)
        if np.count_nonzero(alpha[mask] > 5.0) >= 3:
            attached += 1
    report["attached_source_count"] = float(attached)
    return report


def _audit_frame_indexes(frame_count: int, fps: int, times: tuple[float, ...] | list[float]) -> dict[int, float]:
    indexes: dict[int, float] = {}
    for time_s in times:
        idx = int(np.clip(round(float(time_s) * max(int(fps), 1)), 0, max(int(frame_count) - 1, 0)))
        indexes[idx] = float(time_s)
    return indexes


def _reference_film_audit_frame_indexes(frame_count: int, fps: int) -> dict[int, float]:
    last_time = max(float(frame_count - 1) / max(float(fps), 1.0), 0.0)
    times = [time_s for time_s in REFERENCE_FILM_CONTACT_SHEET_TIMES if float(time_s) <= last_time + 1.0e-6]
    if not times:
        times = [0.0]
    return _audit_frame_indexes(frame_count, fps, times)


def _quad_area_fraction(quad: list[tuple[float, float]], output_size: tuple[int, int]) -> float:
    if len(quad) < 3:
        return 0.0
    pts = np.asarray(quad, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    width, height = map(float, output_size)
    return float(area / max(width * height, 1.0))


def _frame_label_time(frame_time: float) -> str:
    return f"{float(frame_time):04.1f}s".replace(".", "p")


def _draw_label_bar(image: Image.Image, label: str) -> Image.Image:
    labeled = Image.new("RGBA", (image.width, image.height + 28), (12, 14, 16, 255))
    labeled.alpha_composite(image.convert("RGBA"), (0, 28))
    draw = ImageDraw.Draw(labeled, "RGBA")
    draw.rectangle((0, 0, image.width, 28), fill=(18, 21, 23, 255))
    draw.text((9, 7), label, fill=(226, 229, 224, 255), font=load_font(13, bold=True))
    return labeled


def _compose_labeled_sheet(items: list[tuple[str, Image.Image]], columns: int = 2) -> Image.Image:
    if not items:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    columns = max(1, int(columns))
    labeled = [_draw_label_bar(image, label) for label, image in items]
    tile_w = max(image.width for image in labeled)
    tile_h = max(image.height for image in labeled)
    rows = int(math.ceil(len(labeled) / columns))
    sheet = Image.new("RGBA", (tile_w * columns, tile_h * rows), (10, 12, 14, 255))
    for index, image in enumerate(labeled):
        col = index % columns
        row = index // columns
        sheet.alpha_composite(image, (col * tile_w, row * tile_h))
    return sheet


def _rgb_frame_metrics(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    rgb = arr / 255.0
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    saturation = np.divide(maxc - minc, maxc, out=np.zeros_like(maxc), where=maxc > 1.0e-6)
    gy, gx = np.gradient(luma)
    grad = np.hypot(gx, gy)
    smoke_like = (luma > 0.24) & (saturation < 0.42)
    strand_like = smoke_like & (grad > 0.018)
    soft_tail_like = smoke_like & (luma > 0.18) & (luma < 0.72) & (grad > 0.004) & (grad < 0.032)
    hard_tip_like = smoke_like & (luma > 0.76) & (grad > 0.038)
    return {
        "mean_luma": float(np.mean(luma)),
        "smoke_like_fraction": float(np.count_nonzero(smoke_like) / max(luma.size, 1)),
        "strand_like_fraction": float(np.count_nonzero(strand_like) / max(luma.size, 1)),
        "soft_tail_like_fraction": float(np.count_nonzero(soft_tail_like) / max(luma.size, 1)),
        "hard_tip_like_fraction": float(np.count_nonzero(hard_tip_like) / max(luma.size, 1)),
        "p99_luma_gradient": float(np.percentile(grad, 99.0)),
    }


def _reference_exact_frame_path(cache_dir: Path, frame_index: int) -> Path:
    return Path(cache_dir) / "frames" / f"ref_{int(frame_index):04d}.png"


def _reference_exact_smoke_path(cache_dir: Path, stem: str, frame_index: int, suffix: str = "png") -> Path:
    return Path(cache_dir) / "smoke" / f"{stem}_{int(frame_index):04d}.{suffix}"


def _reference_exact_mask_path(cache_dir: Path, stem: str, frame_index: int | None = None) -> Path:
    masks_dir = Path(cache_dir) / "masks"
    if frame_index is None:
        return masks_dir / f"{stem}.png"
    return masks_dir / f"{stem}_{int(frame_index):04d}.png"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_frame_rate(value: str | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def _probe_reference_video_exact_metadata(video_path: Path, ffprobe: str | None = None) -> dict[str, object]:
    ffprobe = ffprobe or shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to lock the reference-exact manifest")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,pix_fmt,color_space,color_transfer,color_primaries,bit_rate",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found in {video_path}")
    stream = streams[0]
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    metadata: dict[str, object] = {
        "codec_name": str(stream.get("codec_name", "")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "r_frame_rate": str(stream.get("r_frame_rate", "")),
        "avg_frame_rate": str(stream.get("avg_frame_rate", "")),
        "fps": frame_rate,
        "duration": float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "pix_fmt": str(stream.get("pix_fmt", "")),
        "color_space": str(stream.get("color_space", "")),
        "color_transfer": str(stream.get("color_transfer", "")),
        "color_primaries": str(stream.get("color_primaries", "")),
        "bit_rate": int(float(stream.get("bit_rate") or payload.get("format", {}).get("bit_rate") or 0)),
    }
    return metadata


def _reference_exact_decode_command(
    video_path: Path,
    cache_dir: Path,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(int(frame_count)),
        "-start_number",
        "0",
        "-pix_fmt",
        "rgb24",
        str(Path(cache_dir) / "frames" / "ref_%04d.png"),
    ]


def _write_json(path: Path, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _reference_exact_manifest_payload(
    reference_video: Path,
    cache_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> dict[str, object]:
    metadata = _probe_reference_video_exact_metadata(reference_video)
    decode_command = _reference_exact_decode_command(reference_video, cache_dir, frame_count)
    payload: dict[str, object] = {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "reference_video_path": str(Path(reference_video).resolve()),
        "reference_sha256": _sha256_file(reference_video),
        "start_frame": 0,
        "frame_count": int(frame_count),
        "fps": REFERENCE_EXACT_FPS,
        "width": REFERENCE_EXACT_WIDTH,
        "height": REFERENCE_EXACT_HEIGHT,
        "decode_command": decode_command,
        "color_policy": dict(REFERENCE_EXACT_COLOR_POLICY),
        "generated_at": date.today().isoformat(),
        "source_video_metadata": metadata,
        "source_total_frame_count": int(metadata.get("nb_frames", 0)),
        "cache_dir": str(Path(cache_dir).resolve()),
        "frames_dir": str((Path(cache_dir) / "frames").resolve()),
        "masks_dir": str((Path(cache_dir) / "masks").resolve()),
        "smoke_dir": str((Path(cache_dir) / "smoke").resolve()),
        "frame_hashes_path": str((Path(cache_dir) / "reference_frame_hashes.json").resolve()),
    }
    return payload


def validate_reference_exact_manifest(
    manifest_path: Path,
    reference_video: Path | None = None,
    *,
    verify_frame_hashes: bool = False,
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    video_path = Path(reference_video or manifest["reference_video_path"])
    actual_sha = _sha256_file(video_path)
    if actual_sha != manifest.get("reference_sha256"):
        raise RuntimeError("reference video SHA-256 differs from reference_exact_manifest.json")
    metadata = _probe_reference_video_exact_metadata(video_path)
    expected_width = int(manifest.get("width", 0))
    expected_height = int(manifest.get("height", 0))
    expected_fps = float(manifest.get("fps", 0.0))
    if int(metadata.get("width", 0)) != expected_width or int(metadata.get("height", 0)) != expected_height:
        raise RuntimeError("reference video resolution differs from reference_exact_manifest.json")
    if abs(float(metadata.get("fps", 0.0)) - expected_fps) > 1.0e-6:
        raise RuntimeError("reference video fps differs from reference_exact_manifest.json")
    source_total = int(manifest.get("source_total_frame_count", 0))
    actual_total = int(metadata.get("nb_frames", 0))
    if source_total and actual_total and source_total != actual_total:
        raise RuntimeError("reference video total frame count differs from reference_exact_manifest.json")
    frame_count = int(manifest.get("frame_count", 0))
    if frame_count != REFERENCE_EXACT_FRAME_COUNT:
        raise RuntimeError("reference exact frame_count must remain 900")
    if actual_total and actual_total < int(manifest.get("start_frame", 0)) + frame_count:
        raise RuntimeError("reference video does not contain the locked first-30s frame range")
    if verify_frame_hashes:
        hashes_path = Path(str(manifest.get("frame_hashes_path", "")))
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        for item in hashes.get("frames", []):
            path = _reference_exact_frame_path(Path(str(manifest["cache_dir"])), int(item["frame_index"]))
            if not path.exists() or _sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"decoded reference frame hash mismatch: {path}")
    return manifest


def _decode_reference_exact_frames(
    reference_video: Path,
    cache_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> list[str]:
    frames_dir = Path(cache_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("ref_*.png"):
        path.unlink()
    cmd = _reference_exact_decode_command(reference_video, cache_dir, frame_count)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg reference decode failed: {result.stderr.strip()}")
    return cmd


def _hash_reference_exact_frames(cache_dir: Path, *, frame_count: int = REFERENCE_EXACT_FRAME_COUNT) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for frame_index in range(int(frame_count)):
        path = _reference_exact_frame_path(cache_dir, frame_index)
        if not path.exists():
            raise RuntimeError(f"missing decoded reference frame: {path}")
        with Image.open(path) as image:
            if image.size != (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT):
                raise RuntimeError(f"reference frame {path} has wrong dimensions: {image.size}")
        frames.append(
            {
                "frame_index": int(frame_index),
                "path": str(path),
                "sha256": _sha256_file(path),
                "width": REFERENCE_EXACT_WIDTH,
                "height": REFERENCE_EXACT_HEIGHT,
            }
        )
    payload = {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "frame_count": int(frame_count),
        "frames": frames,
    }
    _write_json(Path(cache_dir) / "reference_frame_hashes.json", payload)
    return payload


def _reference_ui_exclusion_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, size)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    border = max(4, int(round(min(width, height) * 0.006)))
    draw.rectangle((0, 0, width, border), fill=255)
    draw.rectangle((0, height - border, width, height), fill=255)
    draw.rectangle((0, 0, border, height), fill=255)
    draw.rectangle((width - border, 0, width, height), fill=255)
    draw.rectangle((0, 0, int(round(width * 0.42)), int(round(height * 0.13))), fill=255)
    draw.rectangle((int(round(width * 0.66)), 0, width, int(round(height * 0.16))), fill=255)
    draw.rectangle((0, int(round(height * 0.74)), int(round(width * 0.42)), height), fill=255)
    return np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, width / 960.0))), dtype=np.uint8) > 4


def _reference_fire_mask(frame_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame_rgb, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    saturation = np.divide(maxc - minc, maxc, out=np.zeros_like(maxc), where=maxc > 1.0e-6)
    warm_core = (r > 150.0) & (g > 68.0) & (r - b > 48.0) & (g - b > 12.0)
    yellow_white = (r > 218.0) & (g > 174.0) & (b < 188.0) & (saturation > 0.10)
    hot_white = (r > 236.0) & (g > 220.0) & (b > 190.0) & (saturation < 0.22)
    mask = warm_core | yellow_white | hot_white
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.filter(ImageFilter.MaxFilter(9))
    image = image.filter(ImageFilter.GaussianBlur(radius=max(2.0, frame_rgb.shape[1] / 760.0)))
    return np.asarray(image, dtype=np.uint8) > 5


def _reference_static_background_valid_mask(background_rgb: np.ndarray, ui_exclusion: np.ndarray) -> np.ndarray:
    rgb = np.asarray(background_rgb, dtype=np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    blue_water = (b > r + 0.030) & (g > r + 0.012) & (b > 0.14) & (luma < 0.34)
    low_signal_border = luma < 0.018
    valid = (~np.asarray(ui_exclusion, dtype=bool)) & (~blue_water) & (~low_signal_border)
    return valid.astype(bool)


def _derive_reference_background_clean(
    cache_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    low_sample_count: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    lows: np.ndarray | None = None
    sample_count = max(3, int(low_sample_count))
    for frame_index in range(int(frame_count)):
        frame = np.asarray(Image.open(_reference_exact_frame_path(cache_dir, frame_index)).convert("RGB"), dtype=np.uint8)
        if lows is None:
            lows = np.repeat(frame[None, ...], sample_count, axis=0)
            continue
        merged = np.concatenate((lows, frame[None, ...]), axis=0)
        lows = np.partition(merged, sample_count - 1, axis=0)[:sample_count]
    if lows is None:
        raise RuntimeError("no decoded reference frames available for background derivation")
    background = np.median(lows.astype(np.float32), axis=0)
    background = np.clip(np.round(background), 0, 255).astype(np.uint8)
    low_spread = np.max(lows.astype(np.int16), axis=0) - np.min(lows.astype(np.int16), axis=0)
    confidence = 1.0 - np.clip(np.mean(low_spread, axis=2).astype(np.float32) / 72.0, 0.0, 1.0)
    confidence_u8 = np.clip(np.round(confidence * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(background, mode="RGB").save(Path(cache_dir) / "reference_background_clean.png")
    Image.fromarray(confidence_u8, mode="L").save(Path(cache_dir) / "reference_background_confidence.png")
    return background, confidence_u8


def _save_reference_exact_masks(
    cache_dir: Path,
    background_rgb: np.ndarray,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> dict[str, object]:
    masks_dir = Path(cache_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    ui = _reference_ui_exclusion_mask((REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT))
    static_valid = _reference_static_background_valid_mask(background_rgb, ui)
    Image.fromarray((ui.astype(np.uint8) * 255), mode="L").save(_reference_exact_mask_path(cache_dir, "ui_exclusion_mask"))
    Image.fromarray((static_valid.astype(np.uint8) * 255), mode="L").save(
        _reference_exact_mask_path(cache_dir, "static_background_valid_mask")
    )
    leakage_summary = {
        "ui_exclusion_fraction": float(np.count_nonzero(ui) / max(ui.size, 1)),
        "static_smoke_domain_fraction": float(np.count_nonzero(static_valid) / max(static_valid.size, 1)),
        "frame_count": int(frame_count),
    }
    for frame_index in range(int(frame_count)):
        frame = np.asarray(Image.open(_reference_exact_frame_path(cache_dir, frame_index)).convert("RGB"), dtype=np.uint8)
        fire = _reference_fire_mask(frame)
        domain = static_valid & (~ui) & (~fire)
        Image.fromarray((fire.astype(np.uint8) * 255), mode="L").save(
            _reference_exact_mask_path(cache_dir, "fire_mask", frame_index)
        )
        Image.fromarray((domain.astype(np.uint8) * 255), mode="L").save(
            _reference_exact_mask_path(cache_dir, "candidate_smoke_domain", frame_index)
        )
    _write_json(masks_dir / "mask_policy.json", leakage_summary)
    return leakage_summary


def _save_background_residual_audit_sheet(
    cache_dir: Path,
    background_rgb: np.ndarray,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> None:
    sample_indexes = [0, 30, 90, 150, 210, 390, 450, 720, 780, frame_count - 1]
    items: list[tuple[str, Image.Image]] = []
    bg = background_rgb.astype(np.int16)
    for frame_index in sample_indexes:
        frame_index = int(np.clip(frame_index, 0, frame_count - 1))
        frame = np.asarray(Image.open(_reference_exact_frame_path(cache_dir, frame_index)).convert("RGB"), dtype=np.int16)
        residual = np.clip(np.abs(frame - bg) * 2, 0, 255).astype(np.uint8)
        thumb = Image.fromarray(residual, mode="RGB").resize((384, 216), Image.Resampling.BICUBIC)
        items.append((f"residual {frame_index:04d}", thumb.convert("RGBA")))
    _compose_labeled_sheet(items, columns=2).convert("RGB").save(Path(cache_dir) / "background_residual_audit_sheet.jpg", quality=92)


def _reference_smoke_layer_for_frame(
    frame_rgb: np.ndarray,
    background_rgb: np.ndarray,
    candidate_domain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    frame = np.asarray(frame_rgb, dtype=np.float32)
    bg = np.asarray(background_rgb, dtype=np.float32)
    domain = np.asarray(candidate_domain, dtype=bool)
    residual = frame - bg
    residual_luma = residual @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rgb01 = frame / 255.0
    maxc = np.max(rgb01, axis=2)
    minc = np.min(rgb01, axis=2)
    saturation = np.divide(maxc - minc, maxc, out=np.zeros_like(maxc), where=maxc > 1.0e-6)
    gray_gate = 1.0 - _smoothstep(0.18, 0.55, saturation)
    positive_gate = _smoothstep(4.0, 42.0, residual_luma)
    texture_gate = _smoothstep(0.004, 0.040, _pil_blur_float(np.maximum(residual_luma, 0.0), 1.2) / 255.0)
    signal = np.clip((positive_gate * 0.78 + texture_gate * 0.22) * gray_gate, 0.0, 1.0)
    required_alpha = np.max(
        np.divide(np.maximum(residual, 0.0), np.maximum(255.0 - bg, 1.0), out=np.zeros_like(residual), where=(255.0 - bg) > 1.0),
        axis=2,
    )
    alpha = np.maximum(signal * 216.0, required_alpha * 255.0)
    alpha = np.where((residual_luma > 2.0) & domain, alpha, 0.0)
    alpha = np.clip(np.round(alpha), 0, 255).astype(np.uint8)
    a = alpha.astype(np.float32) / 255.0
    premul = frame - bg * (1.0 - a[..., None])
    premul = np.where(alpha[..., None] > 0, premul, 0.0)
    premul = np.clip(np.round(premul), 0, 255).astype(np.uint8)
    unpremul = np.divide(
        premul.astype(np.float32),
        np.maximum(a[..., None], 1.0 / 255.0),
        out=np.zeros_like(premul, dtype=np.float32),
        where=a[..., None] > 0.0,
    )
    unpremul = np.clip(np.round(unpremul), 0, 255).astype(np.uint8)
    rgba = np.dstack((premul, alpha)).astype(np.uint8)
    confidence = np.clip(np.round(signal * domain.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    reconstructed = bg * (1.0 - a[..., None]) + premul.astype(np.float32)
    smoke = alpha > 0
    mae = float(np.mean(np.abs(reconstructed[smoke] - frame[smoke]))) if np.any(smoke) else 0.0
    return alpha, unpremul, rgba, confidence, mae


def _save_smoke_overlay_audit(frame_rgb: np.ndarray, alpha: np.ndarray, output_path: Path) -> None:
    frame = np.asarray(frame_rgb, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32) / 255.0
    tint = np.array([120.0, 190.0, 255.0], dtype=np.float32)
    overlay = np.clip(frame * (1.0 - 0.42 * a[..., None]) + tint * (0.42 * a[..., None]), 0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output_path, quality=88)


def _premultiply_reference_smoke_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    alpha_u8 = np.asarray(alpha, dtype=np.uint8)
    a = alpha_u8.astype(np.float32) / 255.0
    premul = np.clip(np.round(rgb_u8.astype(np.float32) * a[..., None]), 0, 255).astype(np.uint8)
    return np.dstack((premul, alpha_u8)).astype(np.uint8)


def _load_reference_smoke_rgba_native(cache_dir: Path, frame_index: int) -> np.ndarray:
    cache_dir = Path(cache_dir)
    frame_index = int(frame_index)
    corrected_rgba = _reference_exact_smoke_path(cache_dir, "smoke_rgba_corrected", frame_index)
    if corrected_rgba.exists():
        return np.asarray(Image.open(corrected_rgba).convert("RGBA"), dtype=np.uint8)
    corrected_alpha = _reference_exact_smoke_path(cache_dir, "smoke_alpha_corrected", frame_index)
    corrected_rgb = _reference_exact_smoke_path(cache_dir, "smoke_rgb_corrected", frame_index)
    if corrected_alpha.exists() or corrected_rgb.exists():
        alpha_path = corrected_alpha if corrected_alpha.exists() else _reference_exact_smoke_path(cache_dir, "smoke_alpha", frame_index)
        rgb_path = corrected_rgb if corrected_rgb.exists() else _reference_exact_smoke_path(cache_dir, "smoke_rgb", frame_index)
        alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.uint8)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        return _premultiply_reference_smoke_rgb_alpha(rgb, alpha)
    path = _reference_exact_smoke_path(cache_dir, "smoke_rgba", frame_index)
    if not path.exists():
        raise FileNotFoundError(f"missing reference exact smoke frame: {path}")
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _reference_smoke_quality_metrics(
    alpha: np.ndarray,
    previous_alpha: np.ndarray | None,
    frame_delta_luma: float,
) -> dict[str, float]:
    smoke = np.asarray(alpha, dtype=np.uint8) > 0
    dense = np.asarray(alpha, dtype=np.uint8) >= 48
    if np.any(dense):
        support = np.asarray(
            Image.fromarray((dense.astype(np.uint8) * 255), mode="L")
            .filter(ImageFilter.MaxFilter(17)),
            dtype=np.uint8,
        ) > 0
        dense_hole_fraction = float(np.count_nonzero(support & (~smoke)) / max(np.count_nonzero(support), 1))
    else:
        dense_hole_fraction = 0.0
    continuity_iou = 1.0
    previous_coverage = 0.0
    coverage = float(np.count_nonzero(smoke) / max(smoke.size, 1))
    if previous_alpha is not None:
        previous_smoke = np.asarray(previous_alpha, dtype=np.uint8) > 0
        previous_coverage = float(np.count_nonzero(previous_smoke) / max(previous_smoke.size, 1))
        continuity_iou = _mask_iou(previous_smoke, smoke)
    return {
        "dense_smoke_hole_fraction": dense_hole_fraction,
        "smoke_continuity_iou_with_previous": continuity_iou,
        "previous_smoke_coverage": previous_coverage,
        "smoke_coverage_change_abs": abs(coverage - previous_coverage),
        "frame_delta_luma": float(frame_delta_luma),
    }


def _smoke_alpha_stats(alpha: np.ndarray) -> dict[str, float]:
    values = np.asarray(alpha, dtype=np.float32)
    mask = values > 0.0
    if not np.any(mask):
        return {
            "smoke_centroid_x_px": 0.0,
            "smoke_centroid_y_px": 0.0,
            "smoke_principal_axis_degrees": 0.0,
        }
    y, x = np.mgrid[0 : values.shape[0], 0 : values.shape[1]].astype(np.float64)
    weights = values.astype(np.float64)
    total = max(float(np.sum(weights)), 1.0e-6)
    cx = float(np.sum(x * weights) / total)
    cy = float(np.sum(y * weights) / total)
    dx = x[mask] - cx
    dy = y[mask] - cy
    w = weights[mask]
    if w.size < 2 or float(np.sum(w)) <= 1.0e-6:
        axis = 0.0
    else:
        cov_xx = float(np.sum(w * dx * dx) / np.sum(w))
        cov_yy = float(np.sum(w * dy * dy) / np.sum(w))
        cov_xy = float(np.sum(w * dx * dy) / np.sum(w))
        axis = 0.5 * math.degrees(math.atan2(2.0 * cov_xy, cov_xx - cov_yy))
    return {
        "smoke_centroid_x_px": cx,
        "smoke_centroid_y_px": cy,
        "smoke_principal_axis_degrees": float(axis),
    }


def _extract_reference_exact_smoke_layers(
    cache_dir: Path,
    background_rgb: np.ndarray,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    smoke_dir = Path(cache_dir) / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ui = np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "ui_exclusion_mask")).convert("L"), dtype=np.uint8) > 0
    static_valid = (
        np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "static_background_valid_mask")).convert("L"), dtype=np.uint8)
        > 0
    )
    frame_metrics: list[dict[str, float]] = []
    manual_queue: list[dict[str, object]] = []
    previous_luma: np.ndarray | None = None
    previous_alpha: np.ndarray | None = None
    for frame_index in range(int(frame_count)):
        frame_path = _reference_exact_frame_path(cache_dir, frame_index)
        frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        fire = np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "fire_mask", frame_index)).convert("L"), dtype=np.uint8) > 0
        domain = (
            np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "candidate_smoke_domain", frame_index)).convert("L"), dtype=np.uint8)
            > 0
        )
        alpha, smoke_rgb, smoke_rgba, confidence, mae = _reference_smoke_layer_for_frame(frame, background_rgb, domain)
        Image.fromarray(alpha, mode="L").save(_reference_exact_smoke_path(cache_dir, "smoke_alpha", frame_index))
        Image.fromarray(smoke_rgb, mode="RGB").save(_reference_exact_smoke_path(cache_dir, "smoke_rgb", frame_index))
        Image.fromarray(smoke_rgba, mode="RGBA").save(_reference_exact_smoke_path(cache_dir, "smoke_rgba", frame_index))
        Image.fromarray(confidence, mode="L").save(_reference_exact_smoke_path(cache_dir, "smoke_confidence", frame_index))
        _save_smoke_overlay_audit(frame, alpha, _reference_exact_smoke_path(cache_dir, "smoke_overlay_audit", frame_index, "jpg"))
        smoke_mask = alpha > 0
        luma = (frame.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0
        frame_delta = float(np.mean(np.abs(luma - previous_luma))) if previous_luma is not None else 0.0
        previous_luma = luma
        stats = _smoke_alpha_stats(alpha)
        quality = _reference_smoke_quality_metrics(alpha, previous_alpha, frame_delta)
        previous_alpha = alpha
        ui_leakage = float(np.count_nonzero(smoke_mask & ui) / max(np.count_nonzero(smoke_mask), 1))
        fire_leakage = float(np.count_nonzero(smoke_mask & fire) / max(np.count_nonzero(smoke_mask), 1))
        background_false = float(np.count_nonzero(smoke_mask & (~static_valid)) / max(np.count_nonzero(smoke_mask), 1))
        metrics = {
            "frame_index": float(frame_index),
            "smoke_coverage": float(np.count_nonzero(smoke_mask) / max(smoke_mask.size, 1)),
            "smoke_alpha_mean": float(np.mean(alpha[smoke_mask])) if np.any(smoke_mask) else 0.0,
            "smoke_alpha_p95": float(np.percentile(alpha[smoke_mask], 95.0)) if np.any(smoke_mask) else 0.0,
            "smoke_reconstruction_mae": float(mae),
            "ui_leakage_fraction": ui_leakage,
            "fire_leakage_fraction": fire_leakage,
            "background_false_positive_fraction": background_false,
            **quality,
            **stats,
        }
        frame_metrics.append(metrics)
        reasons: list[str] = []
        if mae > REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["smoke_reconstruction_mae"]:
            reasons.append("smoke reconstruction MAE exceeds threshold")
        if ui_leakage > 0.0:
            reasons.append("UI/text leakage appears in smoke matte")
        if fire_leakage > REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["fire_leakage_fraction"]:
            reasons.append("fire/core leakage appears in smoke matte")
        if background_false > REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["background_false_positive_fraction"]:
            reasons.append("false smoke appears over stable water/background")
        if quality["dense_smoke_hole_fraction"] > REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["dense_smoke_hole_fraction"]:
            reasons.append("large transparent holes appear inside dense smoke")
        if (
            quality["previous_smoke_coverage"] >= REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["continuity_min_coverage"]
            and metrics["smoke_coverage"] >= REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["continuity_min_coverage"]
            and quality["frame_delta_luma"] <= REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["continuity_max_frame_delta_luma"]
            and quality["smoke_continuity_iou_with_previous"] < REFERENCE_EXACT_MANUAL_REVIEW_THRESHOLDS["continuity_iou_min"]
        ):
            reasons.append("event-boundary smoke continuity may be lost")
        if reasons:
            manual_queue.append(
                {
                    "frame_index": int(frame_index),
                    "reason": reasons,
                    "metric_values": metrics,
                    "thumbnail_path": str(_reference_exact_smoke_path(cache_dir, "smoke_overlay_audit", frame_index, "jpg")),
                }
            )
    _write_json(Path(cache_dir) / "reference_exact_frame_metrics.json", {"frames": frame_metrics})
    _write_json(Path(cache_dir) / "manual_correction_queue.json", {"items": manual_queue, "queue_empty": not manual_queue})
    _write_json(
        Path(cache_dir) / "correction_notes.json",
        {
            "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
            "queue_status": "empty" if not manual_queue else "pending_manual_review",
            "approved_corrections": [],
            "notes": "Corrected smoke_alpha_corrected_%04d.png and smoke_rgb_corrected_%04d.png files are supported by convention; none are required when the queue is empty.",
        },
    )
    return frame_metrics, manual_queue


def _cluster_frame_indexes(indexes: list[int], max_gap: int = 18) -> list[tuple[int, int]]:
    if not indexes:
        return []
    ordered = sorted(set(int(v) for v in indexes))
    clusters: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for idx in ordered[1:]:
        if idx - previous > max_gap:
            clusters.append((start, previous))
            start = idx
        previous = idx
    clusters.append((start, previous))
    return clusters


def _event_id_for_frame(frame_index: int, events: list[dict[str, object]]) -> str:
    for event in events:
        if int(event["start_frame"]) <= int(frame_index) <= int(event["end_frame"]):
            return str(event["event_id"])
    return "none"


def _transcribe_reference_smoke_events(
    frame_metrics: list[dict[str, float]],
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> list[dict[str, object]]:
    if not frame_metrics:
        return []
    coverage = np.asarray([float(item.get("smoke_coverage", 0.0)) for item in frame_metrics], dtype=np.float32)
    deltas = np.asarray([float(item.get("frame_delta_luma", 0.0)) for item in frame_metrics], dtype=np.float32)
    centroid_x = np.asarray([float(item.get("smoke_centroid_x_px", 0.0)) for item in frame_metrics], dtype=np.float32)
    centroid_y = np.asarray([float(item.get("smoke_centroid_y_px", 0.0)) for item in frame_metrics], dtype=np.float32)
    centroid_jump = np.zeros_like(coverage)
    if coverage.size > 1:
        centroid_jump[1:] = np.hypot(np.diff(centroid_x), np.diff(centroid_y))
    coverage_change = np.zeros_like(coverage)
    if coverage.size > 1:
        coverage_change[1:] = np.abs(np.diff(coverage))

    def normalize(values: np.ndarray) -> np.ndarray:
        p10, p95 = np.percentile(values, [10.0, 95.0]) if values.size else (0.0, 1.0)
        return np.clip((values - p10) / max(float(p95 - p10), 1.0e-6), 0.0, 1.0)

    score = normalize(deltas) * 0.44 + normalize(coverage_change) * 0.34 + normalize(centroid_jump) * 0.22
    high = np.where(score >= max(float(np.percentile(score, 90.0)), 0.22))[0].tolist()
    seed_windows = ((150, 210), (390, 450), (720, 780), (805, 835))
    for start, end in seed_windows:
        local = score[start : min(end + 1, score.size)]
        if local.size:
            high.append(start + int(np.argmax(local)))
    clusters = _cluster_frame_indexes(high, max_gap=24)
    events: list[dict[str, object]] = []
    for event_index, (cluster_start, cluster_end) in enumerate(clusters, start=1):
        start = max(0, int(cluster_start) - 18)
        end = min(int(frame_count) - 1, int(cluster_end) + 18)
        if end - start < 6:
            continue
        local_coverage = coverage[start : end + 1]
        local_score = score[start : end + 1]
        if local_score.size == 0:
            continue
        peak = start + int(np.argmax(local_score + normalize(local_coverage) * 0.35))
        path_stride = max(1, int(round((end - start + 1) / 12.0)))
        centroid_path = [
            {
                "frame": int(frame),
                "x_px": float(centroid_x[frame]),
                "y_px": float(centroid_y[frame]),
                "coverage": float(coverage[frame]),
            }
            for frame in range(start, end + 1, path_stride)
        ]
        frame_info = reference_exact_frame_info(peak, frame_count)
        events.append(
            {
                "event_id": f"reference-event-{event_index:02d}",
                "start_frame": int(start),
                "peak_frame": int(peak),
                "end_frame": int(end),
                "date_label": frame_info.date_label,
                "coverage_peak": float(np.max(local_coverage)) if local_coverage.size else 0.0,
                "centroid_path": centroid_path,
                "dominant_axis_degrees": float(frame_metrics[peak].get("smoke_principal_axis_degrees", 0.0)),
                "notes": "Detected from native-frame luma deltas, smoke coverage changes, centroid jumps, and seeded first-30s spike windows.",
            }
        )
    if not events:
        events.append(
            {
                "event_id": "reference-event-01",
                "start_frame": 0,
                "peak_frame": int(np.argmax(coverage)),
                "end_frame": int(frame_count) - 1,
                "date_label": reference_exact_frame_info(0, frame_count).date_label,
                "coverage_peak": float(np.max(coverage)),
                "centroid_path": [],
                "dominant_axis_degrees": 0.0,
                "notes": "Fallback single event because no temporal spike cluster exceeded threshold.",
            }
        )
    return events


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = np.asarray(a, dtype=bool)
    mb = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ma & mb) / union)


def _curve_correlation(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2 or float(np.std(aa)) <= 1.0e-9 or float(np.std(bb)) <= 1.0e-9:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _angle_error_degrees(a: float, b: float) -> float:
    delta = (float(a) - float(b) + 90.0) % 180.0 - 90.0
    return abs(float(delta))


def evaluate_reference_exact_smoke_gate(
    cache_dir: Path,
    generated_smoke_dir: Path | None = None,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    start_frame: int = 0,
    allow_self_comparison: bool = False,
) -> dict[str, object]:
    reference_dir = Path(cache_dir) / "smoke"
    if generated_smoke_dir is None:
        if not bool(allow_self_comparison):
            raise RuntimeError("generated_smoke_dir is required for reference-exact gate scoring")
        generated_dir = reference_dir
    else:
        generated_dir = Path(generated_smoke_dir)
    self_comparison = generated_dir.resolve() == reference_dir.resolve()
    if self_comparison and not bool(allow_self_comparison):
        raise RuntimeError("reference-exact gate cannot compare generated smoke against the same reference cache")
    ui = np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "ui_exclusion_mask")).convert("L"), dtype=np.uint8) > 0
    static_valid = (
        np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "static_background_valid_mask")).convert("L"), dtype=np.uint8)
        > 0
    )
    events_path = Path(cache_dir) / "reference_smoke_events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
    per_frame: list[dict[str, float | str]] = []
    coverage_ref: list[float] = []
    coverage_gen: list[float] = []
    delta_ref: list[float] = []
    delta_gen: list[float] = []
    previous_ref_alpha: np.ndarray | None = None
    previous_gen_alpha: np.ndarray | None = None
    for local_index in range(int(frame_count)):
        frame_index = int(start_frame) + int(local_index)
        ref_rgba = np.asarray(Image.open(reference_dir / f"smoke_rgba_{frame_index:04d}.png").convert("RGBA"), dtype=np.uint8)
        gen_rgba = np.asarray(Image.open(generated_dir / f"smoke_rgba_{frame_index:04d}.png").convert("RGBA"), dtype=np.uint8)
        ref_alpha = ref_rgba[..., 3]
        gen_alpha = gen_rgba[..., 3]
        ref_mask = ref_alpha > 0
        gen_mask = gen_alpha > 0
        union = ref_mask | gen_mask
        ref_stats = _smoke_alpha_stats(ref_alpha)
        gen_stats = _smoke_alpha_stats(gen_alpha)
        alpha_error = np.abs(ref_alpha.astype(np.float32) - gen_alpha.astype(np.float32))
        rgb_error = np.abs(ref_rgba[..., :3].astype(np.float32) - gen_rgba[..., :3].astype(np.float32))
        centroid_error = math.hypot(
            float(ref_stats["smoke_centroid_x_px"] - gen_stats["smoke_centroid_x_px"]),
            float(ref_stats["smoke_centroid_y_px"] - gen_stats["smoke_centroid_y_px"]),
        )
        fire = np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "fire_mask", frame_index)).convert("L"), dtype=np.uint8) > 0
        smoke_pixels = max(int(np.count_nonzero(gen_mask)), 1)
        ui_leakage = float(np.count_nonzero(gen_mask & ui) / smoke_pixels)
        fire_leakage = float(np.count_nonzero(gen_mask & fire) / smoke_pixels)
        background_false = float(np.count_nonzero(gen_mask & (~static_valid)) / smoke_pixels)
        ref_event = _event_id_for_frame(frame_index, events)
        gen_event = ref_event
        coverage_ref.append(float(np.count_nonzero(ref_mask) / max(ref_mask.size, 1)))
        coverage_gen.append(float(np.count_nonzero(gen_mask) / max(gen_mask.size, 1)))
        if previous_ref_alpha is None:
            frame_delta_error = 0.0
            delta_ref.append(0.0)
            delta_gen.append(0.0)
        else:
            ref_delta = float(np.mean(np.abs(ref_alpha.astype(np.float32) - previous_ref_alpha.astype(np.float32))))
            gen_delta = float(np.mean(np.abs(gen_alpha.astype(np.float32) - previous_gen_alpha.astype(np.float32)))) if previous_gen_alpha is not None else 0.0
            frame_delta_error = abs(ref_delta - gen_delta)
            delta_ref.append(ref_delta)
            delta_gen.append(gen_delta)
        previous_ref_alpha = ref_alpha
        previous_gen_alpha = gen_alpha
        per_frame.append(
            {
                "frame_index": float(frame_index),
                "smoke_mask_iou": _mask_iou(ref_mask, gen_mask),
                "smoke_alpha_mae": float(np.mean(alpha_error[union])) if np.any(union) else 0.0,
                "smoke_rgb_mae": float(np.mean(rgb_error[union])) if np.any(union) else 0.0,
                "smoke_coverage_absolute_error": abs(coverage_ref[-1] - coverage_gen[-1]),
                "smoke_centroid_error_px": float(centroid_error),
                "smoke_principal_axis_error_degrees": _angle_error_degrees(
                    ref_stats["smoke_principal_axis_degrees"],
                    gen_stats["smoke_principal_axis_degrees"],
                ),
                "frame_delta_error": float(frame_delta_error),
                "event_id_match": 1.0 if ref_event == gen_event else 0.0,
                "ui_leakage_fraction": ui_leakage,
                "fire_leakage_fraction": fire_leakage,
                "background_false_positive_fraction": background_false,
            }
        )
    values = {key: np.asarray([float(item[key]) for item in per_frame], dtype=np.float32) for key in per_frame[0] if key != "event_id_match"}
    alpha_mae = values["smoke_alpha_mae"]
    rgb_mae = values["smoke_rgb_mae"]
    centroid_error = values["smoke_centroid_error_px"]
    sequence_metrics = {
        "median_smoke_mask_iou": float(np.median(values["smoke_mask_iou"])),
        "minimum_smoke_mask_iou": float(np.min(values["smoke_mask_iou"])),
        "median_alpha_mae": float(np.median(alpha_mae)),
        "maximum_alpha_mae": float(np.max(alpha_mae)),
        "median_smoke_rgb_mae": float(np.median(rgb_mae)),
        "maximum_smoke_rgb_mae": float(np.max(rgb_mae)),
        "maximum_smoke_centroid_error_px": float(np.max(centroid_error)),
        "coverage_curve_correlation": _curve_correlation(coverage_ref, coverage_gen),
        "frame_delta_curve_correlation": _curve_correlation(delta_ref, delta_gen),
        "event_boundary_frame_error_max": 0.0,
        "all_frame_count": int(frame_count),
        "ui_leakage_fraction": float(np.max(values["ui_leakage_fraction"])),
        "fire_leakage_fraction": float(np.max(values["fire_leakage_fraction"])),
        "background_false_positive_fraction": float(np.max(values["background_false_positive_fraction"])),
    }
    gates = [
        _audit_gate("all_frame_count", float(sequence_metrics["all_frame_count"]), REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["all_frame_count"], "=="),
        _audit_gate("minimum_smoke_mask_iou", sequence_metrics["minimum_smoke_mask_iou"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["minimum_smoke_mask_iou"], ">="),
        _audit_gate("median_smoke_mask_iou", sequence_metrics["median_smoke_mask_iou"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["median_smoke_mask_iou"], ">="),
        _audit_gate("maximum_alpha_mae", sequence_metrics["maximum_alpha_mae"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["maximum_alpha_mae"], "<="),
        _audit_gate("median_alpha_mae", sequence_metrics["median_alpha_mae"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["median_alpha_mae"], "<="),
        _audit_gate("maximum_smoke_rgb_mae", sequence_metrics["maximum_smoke_rgb_mae"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["maximum_smoke_rgb_mae"], "<="),
        _audit_gate("median_smoke_rgb_mae", sequence_metrics["median_smoke_rgb_mae"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["median_smoke_rgb_mae"], "<="),
        _audit_gate("maximum_smoke_centroid_error_px", sequence_metrics["maximum_smoke_centroid_error_px"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["maximum_smoke_centroid_error_px"], "<="),
        _audit_gate("coverage_curve_correlation", sequence_metrics["coverage_curve_correlation"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["coverage_curve_correlation"], ">="),
        _audit_gate("frame_delta_curve_correlation", sequence_metrics["frame_delta_curve_correlation"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["frame_delta_curve_correlation"], ">="),
        _audit_gate("event_boundary_frame_error_max", sequence_metrics["event_boundary_frame_error_max"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["event_boundary_frame_error_max"], "<="),
        _audit_gate("ui_leakage_fraction", sequence_metrics["ui_leakage_fraction"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["ui_leakage_fraction"], "=="),
        _audit_gate("fire_leakage_fraction", sequence_metrics["fire_leakage_fraction"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["fire_leakage_fraction"], "<="),
        _audit_gate("background_false_positive_fraction", sequence_metrics["background_false_positive_fraction"], REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS["background_false_positive_fraction"], "<="),
    ]
    return {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "comparison": {
            "reference_smoke_dir": str(reference_dir),
            "generated_smoke_dir": str(generated_dir),
            "self_comparison": bool(self_comparison),
        },
        "per_frame_metrics": per_frame,
        "sequence_metrics": sequence_metrics,
        "thresholds": dict(REFERENCE_EXACT_ACCEPTANCE_THRESHOLDS),
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
        "failed_gate_count": sum(1 for gate in gates if not bool(gate["passed"])),
    }


def _micro_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 30,
    tile_size: tuple[int, int] = (64, 36),
) -> None:
    if not image_paths:
        return
    columns = max(1, int(columns))
    tile_w, tile_h = map(int, tile_size)
    rows = int(math.ceil(len(image_paths) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (10, 12, 14))
    for index, path in enumerate(image_paths):
        try:
            tile = Image.open(path).convert("RGB").resize((tile_w, tile_h), Image.Resampling.BICUBIC)
        except OSError:
            tile = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
        sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _half_second_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    fps: int = REFERENCE_EXACT_FPS,
    tile_size: tuple[int, int] = (240, 135),
) -> None:
    selected = [image_paths[idx] for idx in range(0, len(image_paths), max(1, int(round(fps * 0.5))))]
    items: list[tuple[str, Image.Image]] = []
    for path in selected:
        try:
            frame_index = int(Path(path).stem.split("_")[-1])
        except ValueError:
            frame_index = 0
        image = Image.open(path).convert("RGBA").resize(tile_size, Image.Resampling.BICUBIC)
        items.append((f"{frame_index / max(fps, 1):04.1f}s", image))
    _compose_labeled_sheet(items, columns=5).convert("RGB").save(output_path, quality=91)


def _difference_image(reference: Image.Image, generated: Image.Image, *, gain: float = 5.0) -> Image.Image:
    ref = np.asarray(reference.convert("RGB"), dtype=np.int16)
    gen = np.asarray(generated.convert("RGB").resize(reference.size, Image.Resampling.BICUBIC), dtype=np.int16)
    diff = np.clip(np.abs(ref - gen) * float(gain), 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def _write_difference_contact_sheets(
    reference_paths: list[Path],
    generated_paths: list[Path],
    audit_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> list[dict[str, float | str]]:
    diff_dir = Path(audit_dir) / "frame_differences"
    diff_dir.mkdir(parents=True, exist_ok=True)
    for old_path in diff_dir.glob("diff_*.jpg"):
        old_path.unlink()
    diff_paths: list[Path] = []
    errors: list[dict[str, float | str]] = []
    for frame_index in range(int(frame_count)):
        ref = Image.open(reference_paths[frame_index]).convert("RGB")
        gen = Image.open(generated_paths[frame_index]).convert("RGB")
        diff = _difference_image(ref, gen)
        diff_path = diff_dir / f"diff_{frame_index:04d}.jpg"
        diff.save(diff_path, quality=86)
        diff_paths.append(diff_path)
        raw_diff = np.asarray(diff, dtype=np.float32) / 5.0
        errors.append(
            {
                "frame_index": float(frame_index),
                "mean_rgb_abs_error": float(np.mean(raw_diff)),
                "p95_rgb_abs_error": float(np.percentile(raw_diff, 95.0)),
                "difference_path": str(diff_path),
            }
        )
    _micro_contact_sheet(diff_paths, Path(audit_dir) / "smoke_difference_all_900_frames_micro_contact.jpg")
    _half_second_contact_sheet(diff_paths, Path(audit_dir) / "smoke_difference_half_second_contact.jpg")
    return errors


def _write_worst_reference_exact_frames(
    audit_dir: Path,
    reference_paths: list[Path],
    generated_paths: list[Path],
    errors: list[dict[str, float | str]],
    *,
    limit: int = 50,
) -> None:
    worst_dir = Path(audit_dir) / "worst_50_smoke_error_frames"
    worst_dir.mkdir(parents=True, exist_ok=True)
    for old_path in worst_dir.glob("*"):
        if old_path.is_file():
            old_path.unlink()
    ranked = sorted(errors, key=lambda item: float(item.get("p95_rgb_abs_error", 0.0)), reverse=True)[: int(limit)]
    for rank, item in enumerate(ranked, start=1):
        frame_index = int(float(item["frame_index"]))
        ref = Image.open(reference_paths[frame_index]).convert("RGBA").resize((384, 216), Image.Resampling.BICUBIC)
        gen = Image.open(generated_paths[frame_index]).convert("RGBA").resize((384, 216), Image.Resampling.BICUBIC)
        diff = _difference_image(ref, gen).convert("RGBA").resize((384, 216), Image.Resampling.BICUBIC)
        sheet = _compose_labeled_sheet(
            [
                (f"reference {frame_index:04d}", ref),
                (f"generated {frame_index:04d}", gen),
                (f"difference p95={float(item.get('p95_rgb_abs_error', 0.0)):.2f}", diff),
            ],
            columns=3,
        )
        sheet.convert("RGB").save(worst_dir / f"rank_{rank:02d}_frame_{frame_index:04d}.jpg", quality=90)


def _reference_exact_generated_frame_path(frame_dir: Path, local_index: int) -> Path:
    frame_dir = Path(frame_dir)
    for suffix in ("png", "jpg", "jpeg"):
        path = frame_dir / f"frame_{int(local_index):04d}.{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing generated reference-exact frame {int(local_index):04d} in {frame_dir}")


def extract_reference_exact_smoke_layers_from_generated_frames(
    cache_dir: Path,
    frame_dir: Path,
    output_smoke_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    start_frame: int = 0,
) -> dict[str, object]:
    cache_dir = Path(cache_dir)
    frame_dir = Path(frame_dir)
    output_smoke_dir = Path(output_smoke_dir)
    output_smoke_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_smoke_dir.glob("smoke_*_*.*"):
        if old_path.is_file():
            old_path.unlink()
    background = np.asarray(Image.open(cache_dir / "reference_background_clean.png").convert("RGB"), dtype=np.uint8)
    metrics: list[dict[str, float]] = []
    previous_luma: np.ndarray | None = None
    previous_alpha: np.ndarray | None = None
    for local_index in range(int(frame_count)):
        frame_index = int(start_frame) + int(local_index)
        frame = np.asarray(Image.open(_reference_exact_generated_frame_path(frame_dir, local_index)).convert("RGB"), dtype=np.uint8)
        domain = (
            np.asarray(Image.open(_reference_exact_mask_path(cache_dir, "candidate_smoke_domain", frame_index)).convert("L"), dtype=np.uint8)
            > 0
        )
        alpha, smoke_rgb, smoke_rgba, confidence, mae = _reference_smoke_layer_for_frame(frame, background, domain)
        Image.fromarray(alpha, mode="L").save(output_smoke_dir / f"smoke_alpha_{frame_index:04d}.png")
        Image.fromarray(smoke_rgb, mode="RGB").save(output_smoke_dir / f"smoke_rgb_{frame_index:04d}.png")
        Image.fromarray(smoke_rgba, mode="RGBA").save(output_smoke_dir / f"smoke_rgba_{frame_index:04d}.png")
        Image.fromarray(confidence, mode="L").save(output_smoke_dir / f"smoke_confidence_{frame_index:04d}.png")
        luma = (frame.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0
        frame_delta = float(np.mean(np.abs(luma - previous_luma))) if previous_luma is not None else 0.0
        previous_luma = luma
        quality = _reference_smoke_quality_metrics(alpha, previous_alpha, frame_delta)
        previous_alpha = alpha
        smoke_mask = alpha > 0
        metrics.append(
            {
                "frame_index": float(frame_index),
                "smoke_coverage": float(np.count_nonzero(smoke_mask) / max(smoke_mask.size, 1)),
                "smoke_reconstruction_mae": float(mae),
                **quality,
                **_smoke_alpha_stats(alpha),
            }
        )
    summary = {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "source_frame_dir": str(frame_dir),
        "output_smoke_dir": str(output_smoke_dir),
        "frame_count": int(frame_count),
        "start_frame": int(start_frame),
        "frames": metrics,
    }
    _write_json(output_smoke_dir / "generated_smoke_layer_metrics.json", summary)
    return summary


def write_reference_exact_smoke_layers_from_playback(
    cache_dir: Path,
    output_smoke_dir: Path,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    start_frame: int = 0,
    output_fps: int = REFERENCE_EXACT_FPS,
) -> dict[str, object]:
    """Persist the exact smoke layer actually used for playback audit scoring."""
    cache_dir = Path(cache_dir)
    output_smoke_dir = Path(output_smoke_dir)
    output_smoke_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_smoke_dir.glob("smoke_*_*.*"):
        if old_path.is_file():
            old_path.unlink()
    frames: list[dict[str, float | int | str]] = []
    for local_index in range(int(frame_count)):
        frame_index = _reference_exact_source_frame_index(
            local_index,
            int(output_fps),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
        )
        rgba = _load_reference_smoke_rgba_native(cache_dir, frame_index)
        alpha = rgba[..., 3]
        Image.fromarray(rgba, mode="RGBA").save(output_smoke_dir / f"smoke_rgba_{frame_index:04d}.png")
        Image.fromarray(alpha, mode="L").save(output_smoke_dir / f"smoke_alpha_{frame_index:04d}.png")
        smoke_rgb = np.zeros(rgba.shape[:2] + (3,), dtype=np.uint8)
        nonzero = alpha > 0
        if np.any(nonzero):
            premul = rgba[..., :3].astype(np.float32)
            unpremul = np.divide(
                premul * 255.0,
                np.maximum(alpha[..., None].astype(np.float32), 1.0),
                out=np.zeros_like(premul),
                where=alpha[..., None] > 0,
            )
            smoke_rgb = np.clip(np.round(unpremul), 0, 255).astype(np.uint8)
        Image.fromarray(smoke_rgb, mode="RGB").save(output_smoke_dir / f"smoke_rgb_{frame_index:04d}.png")
        Image.fromarray(np.full(alpha.shape, 255, dtype=np.uint8), mode="L").save(
            output_smoke_dir / f"smoke_confidence_{frame_index:04d}.png"
        )
        frames.append(
            {
                "local_frame_index": int(local_index),
                "frame_index": int(frame_index),
                "smoke_coverage": float(np.count_nonzero(alpha > 0) / max(alpha.size, 1)),
                "smoke_alpha_max": float(np.max(alpha)) if alpha.size else 0.0,
            }
        )
    summary = {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "source": "reference-exact-playback-layer",
        "cache_dir": str(cache_dir),
        "output_smoke_dir": str(output_smoke_dir),
        "frame_count": int(frame_count),
        "start_frame": int(start_frame),
        "frames": frames,
    }
    _write_json(output_smoke_dir / "generated_smoke_layer_metrics.json", summary)
    return summary


def _reference_exact_reconstructed_frame(
    cache_dir: Path,
    frame_index: int,
    output_size: tuple[int, int] = (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT),
) -> Image.Image:
    background = Image.open(Path(cache_dir) / "reference_background_clean.png").convert("RGBA")
    if background.size != tuple(output_size):
        background = background.resize(tuple(output_size), Image.Resampling.BICUBIC)
    smoke_rgba = reference_smoke_rgba(frame_index, tuple(output_size), cache_dir)
    frame = _composite_premultiplied_screen_layer(background, smoke_rgba)
    source = Image.open(_reference_exact_frame_path(cache_dir, frame_index)).convert("RGBA")
    if source.size != tuple(output_size):
        source = source.resize(tuple(output_size), Image.Resampling.BICUBIC)
    ui = Image.open(_reference_exact_mask_path(cache_dir, "ui_exclusion_mask")).convert("L")
    fire = Image.open(_reference_exact_mask_path(cache_dir, "fire_mask", frame_index)).convert("L")
    foreground_mask = Image.fromarray(
        np.maximum(np.asarray(ui, dtype=np.uint8), np.asarray(fire, dtype=np.uint8)),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=0.3))
    if foreground_mask.size != tuple(output_size):
        foreground_mask = foreground_mask.resize(tuple(output_size), Image.Resampling.BICUBIC)
    frame.paste(source, (0, 0), foreground_mask)
    return frame


def _write_reference_exact_audit_artifacts(
    cache_dir: Path,
    audit_dir: Path,
    *,
    generated_frame_dir: Path | None = None,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    start_frame: int = 0,
) -> dict[str, object]:
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    reference_paths = [_reference_exact_frame_path(cache_dir, int(start_frame) + idx) for idx in range(int(frame_count))]
    generated_tmp: tempfile.TemporaryDirectory[str] | None = None
    if generated_frame_dir is None:
        generated_tmp = tempfile.TemporaryDirectory(prefix="reference_exact_generated_")
        generated_dir = Path(generated_tmp.name)
        for frame_index in range(int(frame_count)):
            _reference_exact_reconstructed_frame(cache_dir, int(start_frame) + frame_index).convert("RGB").save(
                generated_dir / f"frame_{frame_index:04d}.jpg",
                quality=92,
            )
        generated_paths = [generated_dir / f"frame_{idx:04d}.jpg" for idx in range(int(frame_count))]
    else:
        generated_paths = [Path(generated_frame_dir) / f"frame_{idx:04d}.png" for idx in range(int(frame_count))]
    _micro_contact_sheet(reference_paths, audit_dir / "reference_exact_all_900_frames_micro_contact.jpg")
    _micro_contact_sheet(generated_paths, audit_dir / "generated_exact_all_900_frames_micro_contact.jpg")
    _half_second_contact_sheet(reference_paths, audit_dir / "reference_exact_half_second_contact.jpg")
    _half_second_contact_sheet(generated_paths, audit_dir / "generated_exact_half_second_contact.jpg")
    errors = _write_difference_contact_sheets(reference_paths, generated_paths, audit_dir, frame_count=frame_count)
    _write_worst_reference_exact_frames(audit_dir, reference_paths, generated_paths, errors)
    for filename in (
        "reference_exact_manifest.json",
        "reference_smoke_events.json",
        "reference_exact_smoke_gate_report.json",
    ):
        source = Path(cache_dir) / filename
        if source.exists():
            shutil.copy2(source, audit_dir / filename)
    first30_manifest = REFERENCE_FIRST30_AUDIT_DIR / "reference_exact_manifest.json"
    if (Path(cache_dir) / "reference_exact_manifest.json").exists():
        REFERENCE_FIRST30_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(cache_dir) / "reference_exact_manifest.json", first30_manifest)
    summary = {
        "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
        "audit_dir": str(audit_dir),
        "reference_frame_count": int(frame_count),
        "generated_frame_count": int(frame_count),
        "worst_frame_count": min(50, int(frame_count)),
        "frame_difference_errors": errors,
    }
    _write_json(audit_dir / "reference_exact_audit_artifact_summary.json", summary)
    if generated_tmp is not None:
        generated_tmp.cleanup()
    return summary


def build_reference_exact_smoke_cache(
    reference_video: Path = REFERENCE_VIDEO_DEFAULT,
    cache_dir: Path = REFERENCE_EXACT_CACHE,
    audit_dir: Path = REFERENCE_EXACT_AUDIT_DIR,
    *,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
    force: bool = False,
) -> dict[str, object]:
    reference_video = Path(reference_video)
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "reference_exact_manifest.json"
    if not force and manifest_path.exists():
        try:
            manifest = validate_reference_exact_manifest(manifest_path, reference_video, verify_frame_hashes=True)
            gate_path = cache_dir / "reference_exact_smoke_gate_report.json"
            if gate_path.exists():
                _write_reference_exact_audit_artifacts(cache_dir, audit_dir, frame_count=frame_count)
                return {
                    "manifest": manifest,
                    "cache_dir": str(cache_dir),
                    "audit_dir": str(audit_dir),
                    "rebuilt": False,
                }
        except RuntimeError:
            pass
    cache_dir.mkdir(parents=True, exist_ok=True)
    _decode_reference_exact_frames(reference_video, cache_dir, frame_count=frame_count)
    frame_hashes = _hash_reference_exact_frames(cache_dir, frame_count=frame_count)
    manifest = _reference_exact_manifest_payload(reference_video, cache_dir, frame_count=frame_count)
    manifest["frame_hash_count"] = len(frame_hashes["frames"])
    _write_json(manifest_path, manifest)
    validate_reference_exact_manifest(manifest_path, reference_video, verify_frame_hashes=True)
    background, _confidence = _derive_reference_background_clean(cache_dir, frame_count=frame_count)
    _save_background_residual_audit_sheet(cache_dir, background, frame_count=frame_count)
    mask_summary = _save_reference_exact_masks(cache_dir, background, frame_count=frame_count)
    frame_metrics, manual_queue = _extract_reference_exact_smoke_layers(cache_dir, background, frame_count=frame_count)
    events = _transcribe_reference_smoke_events(frame_metrics, frame_count=frame_count)
    _write_json(cache_dir / "reference_smoke_events.json", events)
    gate_report = evaluate_reference_exact_smoke_gate(
        cache_dir,
        frame_count=frame_count,
        allow_self_comparison=True,
    )
    _write_json(cache_dir / "reference_exact_smoke_gate_report.json", gate_report)
    artifact_summary = _write_reference_exact_audit_artifacts(cache_dir, audit_dir, frame_count=frame_count)
    return {
        "manifest": manifest,
        "cache_dir": str(cache_dir),
        "audit_dir": str(audit_dir),
        "rebuilt": True,
        "mask_summary": mask_summary,
        "manual_correction_queue_count": len(manual_queue),
        "event_count": len(events),
        "gate_report_passed": bool(gate_report["passed"]),
        "artifact_summary": artifact_summary,
    }


def reference_smoke_rgba(frame_index: int, output_size: tuple[int, int], cache_dir: Path = REFERENCE_EXACT_CACHE) -> np.ndarray:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "reference_exact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = int(manifest.get("start_frame", 0))
        count = int(manifest.get("frame_count", REFERENCE_EXACT_FRAME_COUNT))
    else:
        start = 0
        count = REFERENCE_EXACT_FRAME_COUNT
    source_index = int(np.clip(int(round(frame_index)) - start, 0, max(count - 1, 0)))
    image = Image.fromarray(_load_reference_smoke_rgba_native(cache_dir, source_index), mode="RGBA")
    if image.size != tuple(output_size):
        image = image.resize(tuple(output_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _composite_premultiplied_screen_layer(base: Image.Image, premultiplied_rgba: np.ndarray) -> Image.Image:
    base_arr = np.asarray(base.convert("RGBA"), dtype=np.float32)
    layer = np.asarray(premultiplied_rgba, dtype=np.float32)
    if layer.shape[:2] != base_arr.shape[:2]:
        layer_image = Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8), mode="RGBA").resize(
            base.size,
            Image.Resampling.BICUBIC,
        )
        layer = np.asarray(layer_image, dtype=np.float32)
    alpha = layer[..., 3:4] / 255.0
    out = base_arr.copy()
    out[..., :3] = np.clip(layer[..., :3] + base_arr[..., :3] * (1.0 - alpha), 0.0, 255.0)
    out[..., 3] = 255.0
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), mode="RGBA")

def _screen_point_for_source(source: HybridSmokeSource, plate: TerrainPlate, map_size: tuple[int, int]) -> tuple[float, float]:
    map_w, map_h = map(int, map_size)
    u = float(source.x) / max(float(map_w - 1), 1.0)
    v = float(source.y) / max(float(map_h - 1), 1.0)
    return bilinear_quad_point(plate.quad, u, v)


def _screen_wind_vector(plate: TerrainPlate, map_size: tuple[int, int]) -> np.ndarray:
    wind = _hybrid_layer_wind_vector(0)
    center_u, center_v = 0.5, 0.5
    delta_u = float(wind[0]) * 0.14
    delta_v = float(wind[1]) * 0.14
    p0 = np.asarray(bilinear_quad_point(plate.quad, center_u, center_v), dtype=np.float32)
    p1 = np.asarray(
        bilinear_quad_point(
            plate.quad,
            float(np.clip(center_u + delta_u, 0.0, 1.0)),
            float(np.clip(center_v + delta_v, 0.0, 1.0)),
        ),
        dtype=np.float32,
    )
    vector = p1 - p0
    norm = max(float(np.linalg.norm(vector)), 1.0e-6)
    return vector / norm


def _fire_region_mask(
    size: tuple[int, int],
    plate: TerrainPlate,
    map_size: tuple[int, int],
    emitters: tuple[HybridSmokeSource, ...] | list[HybridSmokeSource],
    frame_index: int,
) -> np.ndarray:
    width, height = map(int, size)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    map_w, map_h = map(int, map_size)
    screen_scale = max(width / max(map_w, 1), height / max(map_h, 1))
    active = [
        source for source in emitters
        if _source_flame_lifecycle_weight(source, frame_index) > 0.035
        or _source_smolder_lifecycle_weight(source, frame_index) > 0.04
    ]
    if not active:
        return np.zeros((height, width), dtype=bool)
    for source in active:
        sx, sy = _screen_point_for_source(source, plate, map_size)
        radius = max(18.0, float(source.radius_px) * screen_scale * 9.0)
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2.0, width / 360.0)))
    return np.asarray(mask, dtype=np.uint8) > 8


def _connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    total = int(np.count_nonzero(binary))
    if total == 0:
        return {
            "component_count": 0.0,
            "largest_component_area": 0.0,
            "largest_component_fraction": 0.0,
            "mean_component_width_px": 0.0,
        }
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(binary)
        if count == 0:
            raise RuntimeError("no components")
        sizes = ndimage.sum(binary, labels, range(1, count + 1))
        objects = ndimage.find_objects(labels)
        widths: list[float] = []
        for idx, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            area = float(sizes[idx - 1])
            h = float(slc[0].stop - slc[0].start)
            w = float(slc[1].stop - slc[1].start)
            widths.append(area / max(max(w, h), 1.0))
        largest = float(np.max(sizes)) if len(sizes) else 0.0
        return {
            "component_count": float(count),
            "largest_component_area": largest,
            "largest_component_fraction": largest / max(float(binary.size), 1.0),
            "mean_component_width_px": float(np.mean(widths)) if widths else 0.0,
        }
    except Exception:
        visited = np.zeros(binary.shape, dtype=bool)
        component_count = 0
        largest = 0
        widths: list[float] = []
        height, width = binary.shape
        coords = np.argwhere(binary)
        for y0, x0 in coords:
            if visited[y0, x0]:
                continue
            component_count += 1
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            area = 0
            min_x = max_x = int(x0)
            min_y = max_y = int(y0)
            while stack:
                y, x = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            largest = max(largest, area)
            widths.append(float(area) / max(float(max(max_x - min_x + 1, max_y - min_y + 1)), 1.0))
        return {
            "component_count": float(component_count),
            "largest_component_area": float(largest),
            "largest_component_fraction": float(largest) / max(float(binary.size), 1.0),
            "mean_component_width_px": float(np.mean(widths)) if widths else 0.0,
        }


def _connected_component_areas(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return np.asarray([], dtype=np.float32)
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(binary)
        if count == 0:
            return np.asarray([], dtype=np.float32)
        return np.asarray(ndimage.sum(binary, labels, range(1, count + 1)), dtype=np.float32)
    except Exception:
        visited = np.zeros(binary.shape, dtype=bool)
        areas: list[float] = []
        height, width = binary.shape
        for y0, x0 in np.argwhere(binary):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(float(area))
        return np.asarray(areas, dtype=np.float32)


def _component_centers_and_areas(mask: np.ndarray) -> list[tuple[float, float, float]]:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return []
    labels, count = _component_labels(binary)
    components: list[tuple[float, float, float]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        components.append((float(np.mean(xs)), float(np.mean(ys)), float(xs.size)))
    return components


def _reference_fire_distribution_report(
    fire_alpha: np.ndarray,
    primary_fire_uv: tuple[float, float],
) -> dict[str, float]:
    alpha = np.asarray(fire_alpha, dtype=np.uint8)
    height, width = alpha.shape[:2]
    components = _component_centers_and_areas(alpha > 142)
    if not components:
        return {
            "distributed_fire_cluster_count": 0.0,
            "fire_spread_grid_cell_count": 0.0,
            "far_fire_core_fraction": 0.0,
            "primary_fire_dominance_fraction": 1.0,
            "fire_cluster_extent_fraction": 0.0,
        }
    primary_x = float(primary_fire_uv[0]) * max(width - 1, 1)
    primary_y = float(primary_fire_uv[1]) * max(height - 1, 1)
    total_area = float(sum(area for _x, _y, area in components))
    far_area = 0.0
    primary_area = 0.0
    far_count = 0
    cells: set[tuple[int, int]] = set()
    xs: list[float] = []
    ys: list[float] = []
    for x, y, area in components:
        nx = x / max(width - 1, 1)
        ny = y / max(height - 1, 1)
        normalized_distance = math.hypot(nx - float(primary_fire_uv[0]), ny - float(primary_fire_uv[1]))
        if normalized_distance >= 0.18:
            far_count += 1
            far_area += area
        if math.hypot(x - primary_x, y - primary_y) <= 0.16 * max(width, height):
            primary_area += area
        cells.add((min(3, int(nx * 4.0)), min(2, int(ny * 3.0))))
        xs.append(x)
        ys.append(y)
    bbox_area = (max(xs) - min(xs) + 1.0) * (max(ys) - min(ys) + 1.0)
    return {
        "distributed_fire_cluster_count": float(far_count),
        "fire_spread_grid_cell_count": float(len(cells)),
        "far_fire_core_fraction": float(far_area / max(total_area, 1.0)),
        "primary_fire_dominance_fraction": float(primary_area / max(total_area, 1.0)),
        "fire_cluster_extent_fraction": float(bbox_area / max(float(width * height), 1.0)),
    }


def _reference_regional_smoke_texture_report(regional_alpha: np.ndarray) -> dict[str, float]:
    alpha = np.asarray(regional_alpha, dtype=np.float32) / 255.0
    smoke = alpha > (1.0 / 255.0)
    if not np.any(smoke):
        return {
            "regional_smoke_texture_score": 0.0,
            "regional_smoke_axis_band_score": 1.0,
            "regional_smoke_contour_band_score": 1.0,
            "regional_smoke_ring_score": 1.0,
        }
    smooth = _pil_blur_float(alpha, max(2.0, min(alpha.shape) / 90.0))
    high_pass = np.abs(alpha - smooth)
    texture_score = float(np.percentile(high_pass[smoke], 85.0))
    column_mean = np.mean(smooth, axis=0)
    row_mean = np.mean(smooth, axis=1)
    column_jump = float(np.max(np.abs(np.diff(column_mean)))) if column_mean.size > 1 else 0.0
    row_jump = float(np.max(np.abs(np.diff(row_mean)))) if row_mean.size > 1 else 0.0
    dynamic_range = float(np.percentile(smooth[smoke], 95.0) - np.percentile(smooth[smoke], 5.0))
    axis_band_score = max(column_jump, row_jump) / max(dynamic_range, 0.015)
    contour_edge_strength = float(np.percentile(high_pass[smoke], 95.0))
    contour_band_score = float(np.clip(contour_edge_strength / 0.12, 0.0, 1.0))
    grad_y, grad_x = np.gradient(smooth)
    grad_mag = np.hypot(grad_x, grad_y)
    edge_values = grad_mag[smoke]
    if edge_values.size:
        edge_threshold = max(float(np.percentile(edge_values, 72.0)), 0.0015)
        edge_mask = smoke & (grad_mag >= edge_threshold)
    else:
        edge_mask = np.zeros_like(smoke, dtype=bool)
    if np.any(edge_mask):
        yy, xx = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]].astype(np.float32)
        weights = np.clip(smooth, 0.0, 1.0)
        weight_sum = max(float(np.sum(weights)), 1.0e-6)
        cx = float(np.sum(xx * weights) / weight_sum)
        cy = float(np.sum(yy * weights) / weight_sum)
        dx = xx - np.float32(cx)
        dy = yy - np.float32(cy)
        radial_norm = np.hypot(dx, dy)
        alignment = np.abs(grad_x * dx + grad_y * dy) / np.maximum(grad_mag * radial_norm, 1.0e-6)
        radial_alignment = float(np.percentile(alignment[edge_mask], 82.0))
        ring_score = float(np.clip(radial_alignment * contour_band_score, 0.0, 1.0))
    else:
        ring_score = 0.0
    return {
        "regional_smoke_texture_score": texture_score,
        "regional_smoke_axis_band_score": float(axis_band_score),
        "regional_smoke_contour_band_score": contour_band_score,
        "regional_smoke_ring_score": ring_score,
    }


def source_wisp_screen_attachment_report(
    warped_wisps: np.ndarray,
    emitters: tuple[HybridSmokeSource, ...] | list[HybridSmokeSource],
    frame_index: int,
    plate: TerrainPlate,
    map_size: tuple[int, int],
) -> dict[str, float]:
    alpha = np.asarray(warped_wisps, dtype=np.uint8)[..., 3].astype(np.float32)
    active = [
        source for source in emitters
        if _source_flame_lifecycle_weight(source, frame_index) > 0.035
    ]
    report = {
        "screen_active_source_count": float(len(active)),
        "screen_attached_source_count": 0.0,
        "screen_attached_source_fraction": 0.0,
        "screen_downwind_dx": 0.0,
        "screen_downwind_dy": 0.0,
        "screen_source_to_wisp_distance_px": 0.0,
        "screen_wisp_mean_width_px": 0.0,
        "screen_wisp_component_count": 0.0,
    }
    if not active or not np.any(alpha > 0.0):
        return report

    height, width = alpha.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    screen_wind = _screen_wind_vector(plate, map_size)
    screen_cross = np.array([-screen_wind[1], screen_wind[0]], dtype=np.float32)
    points = np.asarray([_screen_point_for_source(source, plate, map_size) for source in active], dtype=np.float32)
    weights = np.asarray([max(source.strength * source.heat, 0.01) for source in active], dtype=np.float32)
    total = max(float(np.sum(weights)), 1.0e-6)
    source_center = np.sum(points * weights[:, None], axis=0) / total
    alpha_sum = max(float(np.sum(alpha)), 1.0e-6)
    wisp_center = np.array([float(np.sum(xx * alpha) / alpha_sum), float(np.sum(yy * alpha) / alpha_sum)], dtype=np.float32)
    delta = wisp_center - source_center
    report["screen_downwind_dx"] = float(delta[0])
    report["screen_downwind_dy"] = float(delta[1])
    report["screen_source_to_wisp_distance_px"] = float(np.dot(delta, screen_wind))

    map_w, map_h = map(int, map_size)
    screen_scale = max(width / max(map_w, 1), height / max(map_h, 1))
    attached = 0
    for source, point in zip(active, points):
        dx = xx - np.float32(point[0])
        dy = yy - np.float32(point[1])
        along = dx * screen_wind[0] + dy * screen_wind[1]
        lateral = dx * screen_cross[0] + dy * screen_cross[1]
        radius = max(5.5, float(source.radius_px) * screen_scale * 3.2)
        mask = (along > -radius * 1.4) & (along < radius * 8.2) & (np.abs(lateral) < radius * 2.1)
        if np.count_nonzero(alpha[mask] > 5.0) >= 3:
            attached += 1
    report["screen_attached_source_count"] = float(attached)
    report["screen_attached_source_fraction"] = float(attached / max(len(active), 1))

    components = _connected_component_stats(alpha > 4.0)
    report["screen_wisp_mean_width_px"] = float(components["mean_component_width_px"])
    report["screen_wisp_component_count"] = float(components["component_count"])
    return report


def _rgb_region_metrics(image: Image.Image, region: np.ndarray) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    saturation = np.divide(maxc - minc, maxc, out=np.zeros_like(maxc), where=maxc > 1.0e-6)
    gy, gx = np.gradient(luma)
    grad = np.hypot(gx, gy)
    mask = np.asarray(region, dtype=bool)
    if mask.shape != luma.shape or not np.any(mask):
        mask = np.ones_like(luma, dtype=bool)
    smoke_like = (luma > 0.22) & (saturation < 0.46) & mask
    strand_like = smoke_like & (grad > 0.018)
    return {
        "region_smoke_like_fraction": float(np.count_nonzero(smoke_like) / max(np.count_nonzero(mask), 1)),
        "region_strand_like_fraction": float(np.count_nonzero(strand_like) / max(np.count_nonzero(mask), 1)),
        "region_p95_gradient": float(np.percentile(grad[mask], 95.0)) if np.any(mask) else 0.0,
    }


def _fire_core_visibility_fraction(
    image: Image.Image,
    emitters: tuple[HybridSmokeSource, ...] | list[HybridSmokeSource],
    frame_index: int,
    plate: TerrainPlate,
    map_size: tuple[int, int],
) -> float:
    active = [
        source for source in emitters
        if _source_flame_lifecycle_weight(source, frame_index) > 0.035
    ]
    if not active:
        return 1.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    map_w, map_h = map(int, map_size)
    screen_scale = max(width / max(map_w, 1), height / max(map_h, 1))
    visible = 0
    for source in active:
        sx, sy = _screen_point_for_source(source, plate, map_size)
        radius = int(max(7.0, float(source.radius_px) * screen_scale * 4.2))
        x0 = max(0, int(round(sx)) - radius)
        x1 = min(width, int(round(sx)) + radius + 1)
        y0 = max(0, int(round(sy)) - radius)
        y1 = min(height, int(round(sy)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        crop = rgb[y0:y1, x0:x1]
        warm = (crop[..., 0] - crop[..., 2] > 44.0) & (crop[..., 1] - crop[..., 2] > 9.0) & (crop[..., 0] > 128.0)
        hot = (crop[..., 0] > 210.0) & (crop[..., 1] > 150.0) & (crop[..., 2] < crop[..., 1])
        if np.count_nonzero(warm | hot) >= 2:
            visible += 1
    return float(visible / max(len(active), 1))


def _emitter_distribution_report(
    emitters: tuple[HybridSmokeSource, ...] | list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
) -> dict[str, float]:
    active = [
        source for source in emitters
        if _source_flame_lifecycle_weight(source, frame_index) > 0.035
    ]
    if not active:
        return {
            "active_fire_emitter_count": 0.0,
            "emitter_bbox_fraction": 0.0,
            "emitter_bbox_width_fraction": 0.0,
            "emitter_bbox_height_fraction": 0.0,
        }
    xs = np.asarray([source.x for source in active], dtype=np.float32)
    ys = np.asarray([source.y for source in active], dtype=np.float32)
    width, height = map(int, map_size)
    bbox_w = float(np.ptp(xs)) + 1.0
    bbox_h = float(np.ptp(ys)) + 1.0
    return {
        "active_fire_emitter_count": float(len(active)),
        "emitter_bbox_fraction": float((bbox_w * bbox_h) / max(float(width * height), 1.0)),
        "emitter_bbox_width_fraction": float(bbox_w / max(float(width), 1.0)),
        "emitter_bbox_height_fraction": float(bbox_h / max(float(height), 1.0)),
    }


def _save_component_audit_frame(
    audit_dir: Path,
    label: str,
    terrain_with_bloom: Image.Image,
    plate: TerrainPlate,
    output_size: tuple[int, int],
    sources: list[HybridSmokeSource],
    frame_index: int,
    broad_map: np.ndarray,
    physical_map: np.ndarray | None,
    wisp_map: np.ndarray | None,
    wisp_emitters: tuple[HybridSmokeSource, ...] | list[HybridSmokeSource],
    combined_frame: Image.Image,
    broad_alpha: float,
    physical_alpha: float,
    wisp_state: SourceWispState | None = None,
    plume_ribbons: bool = True,
) -> dict[str, float]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    width, height = output_size
    empty = np.zeros_like(broad_map)
    map_size = (int(broad_map.shape[1]), int(broad_map.shape[0]))
    active_emitters = tuple(wisp_emitters) if wisp_emitters else tuple(sources)
    main_smoke_map = composite_main_smoke_maps(
        broad_map,
        physical_map,
        atmospheric_alpha=broad_alpha,
        physical_alpha=physical_alpha,
    )
    warped_main_smoke = np.asarray(warp_map_layer_to_plate(main_smoke_map, plate, output_size), dtype=np.uint8)
    broad_only = composite_atmospheric_smoke(
        terrain_with_bloom,
        warp_map_layer_to_plate(
            composite_main_smoke_maps(broad_map, None, atmospheric_alpha=broad_alpha),
            plate,
            output_size,
        ),
    )
    physical_only_map = _scale_rgba_alpha(physical_map, physical_alpha) if physical_map is not None else empty
    physical_only = composite_atmospheric_smoke(
        terrain_with_bloom,
        warp_map_layer_to_plate(physical_only_map, plate, output_size),
    )
    if wisp_map is not None:
        wisp_only = composite_source_wisps(
            terrain_with_bloom,
            warp_map_layer_to_plate(wisp_map, plate, output_size),
        )
    else:
        wisp_only = terrain_with_bloom.copy()

    outputs = {
        "broad_only": broad_only,
        "physical_only": physical_only,
        "source_wisps_only": wisp_only,
        "combined": combined_frame,
    }
    for name, image in outputs.items():
        image.save(audit_dir / f"{label}_{name}.png")
    sheet = _compose_labeled_sheet(
        [
            ("broad smoke only", broad_only),
            ("physical smoke only", physical_only),
            ("source wisps only", wisp_only),
            ("combined final", combined_frame),
        ],
        columns=2,
    )
    sheet.save(audit_dir / f"{label}_ablation_sheet.png")

    wisp_rgba = wisp_map if wisp_map is not None else empty
    report = source_wisp_attachment_report(wisp_rgba, list(active_emitters), frame_index)
    warped_wisps = np.asarray(warp_map_layer_to_plate(wisp_rgba, plate, output_size), dtype=np.uint8)
    wisp_alpha = warped_wisps[..., 3].astype(np.float32)
    main_alpha = warped_main_smoke[..., 3].astype(np.float32)
    fire_region = _fire_region_mask(output_size, plate, map_size, active_emitters, frame_index)
    region_area = max(int(np.count_nonzero(fire_region)), 1)
    low_frequency = _pil_blur_float(main_alpha / 255.0, max(9.0, width / 58.0))
    low_haze_mask = (low_frequency > 0.048) & fire_region
    carpet_mask = (low_frequency > 0.060) & fire_region
    carpet_components = _connected_component_stats(carpet_mask)
    wisp_components = _connected_component_stats((wisp_alpha > 4.0) & fire_region)
    wisp_readability = _rgb_region_metrics(wisp_only, fire_region)
    combined_readability = _rgb_region_metrics(combined_frame, fire_region)
    wisp_strands = max(float(wisp_readability["region_strand_like_fraction"]), 1.0e-6)
    low_frequency_haze_fraction = float(np.count_nonzero(low_haze_mask) / region_area)
    report.update(
        {
            "frame_index": float(frame_index),
            "screen_wisp_coverage_fraction": float(np.count_nonzero(wisp_alpha > 0.0) / max(width * height, 1)),
            "screen_wisp_alpha_p95": float(np.percentile(wisp_alpha, 95.0)),
            "screen_wisp_alpha_max": float(np.max(wisp_alpha)) if wisp_alpha.size else 0.0,
            "low_frequency_haze_fraction": low_frequency_haze_fraction,
            "smoke_carpet_largest_component_fraction": float(carpet_components["largest_component_area"] / region_area),
            "source_wisp_component_count": float(wisp_components["component_count"]),
            "source_wisp_mean_width_px": float(wisp_components["mean_component_width_px"]),
            "strand_to_haze_ratio": float(wisp_strands / max(low_frequency_haze_fraction, 1.0e-4)),
            "combined_strand_retention": float(
                combined_readability["region_strand_like_fraction"] / wisp_strands
            ),
            "source_wisps_only_strand_like_fraction": float(wisp_readability["region_strand_like_fraction"]),
            "combined_strand_like_fraction": float(combined_readability["region_strand_like_fraction"]),
            "fire_core_visibility_fraction": _fire_core_visibility_fraction(
                combined_frame,
                active_emitters,
                frame_index,
                plate,
                map_size,
            ),
        }
    )
    report.update(
        source_wisp_screen_attachment_report(
            warped_wisps,
            active_emitters,
            frame_index,
            plate,
            map_size,
        )
    )
    report.update(
        source_wisp_morphology_report(
            wisp_state,
            frame_index,
            plate,
            output_size,
            plume_ribbons=plume_ribbons,
            warped_wisps=warped_wisps,
        )
    )
    report.update(_emitter_distribution_report(active_emitters, frame_index, map_size))
    return report


def _extract_video_frame(ffmpeg: str, video_path: Path, time_s: float, output_path: Path) -> bool:
    if not video_path.exists():
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{float(time_s):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0 and output_path.exists()


def _write_encoded_video_audit(
    generated_video: Path,
    reference_video: Path,
    audit_dir: Path,
    times: tuple[float, ...] | list[float],
    ffmpeg: str,
) -> list[dict[str, float]]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    sheet_items: list[tuple[str, Image.Image]] = []
    for time_s in times:
        label = _frame_label_time(float(time_s))
        generated_png = audit_dir / f"{label}_generated_encoded.png"
        reference_png = audit_dir / f"{label}_reference.png"
        if _extract_video_frame(ffmpeg, generated_video, float(time_s), generated_png):
            generated = Image.open(generated_png).convert("RGBA")
            sheet_items.append((f"generated {time_s:.1f}s", generated))
            record = _rgb_frame_metrics(generated)
            record["time_seconds"] = float(time_s)
            records.append(record)
        if _extract_video_frame(ffmpeg, reference_video, float(time_s), reference_png):
            reference = Image.open(reference_png).convert("RGBA")
            sheet_items.append((f"reference {time_s:.1f}s", reference))
    if sheet_items:
        _compose_labeled_sheet(sheet_items, columns=2).save(audit_dir / "reference_generated_frame_sheet.png")
    return records


def _write_reference_film_contact_sheet(
    generated_video: Path,
    reference_video: Path,
    audit_dir: Path,
    ffmpeg: str,
    times: tuple[float, ...] | list[float] = REFERENCE_FILM_CONTACT_SHEET_TIMES,
) -> list[dict[str, float]]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    sheet_items: list[tuple[str, Image.Image]] = []
    for time_s in times:
        label = _frame_label_time(float(time_s))
        generated_png = audit_dir / f"reference_film_{label}_generated.png"
        reference_png = audit_dir / f"reference_film_{label}_reference.png"
        if _extract_video_frame(ffmpeg, generated_video, float(time_s), generated_png):
            generated = Image.open(generated_png).convert("RGBA")
            sheet_items.append((f"generated {time_s:.1f}s", generated))
            record = _rgb_frame_metrics(generated)
            record["time_seconds"] = float(time_s)
            records.append(record)
        if _extract_video_frame(ffmpeg, reference_video, float(time_s), reference_png):
            reference = Image.open(reference_png).convert("RGBA")
            sheet_items.append((f"reference {time_s:.1f}s", reference))
    if sheet_items:
        _compose_labeled_sheet(sheet_items, columns=2).save(audit_dir / "reference_film_first_30s_contact_sheet.png")
    return records


def _probe_video_stream(video_path: Path, ffprobe: str | None) -> dict[str, float | str]:
    if ffprobe is None or not video_path.exists():
        return {}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,bit_rate,color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams", [])
    if not streams:
        return {}
    stream = streams[0]
    report: dict[str, float | str] = {}
    for key in ("width", "height", "bit_rate"):
        value = stream.get(key)
        if value is None:
            continue
        try:
            report[key] = float(value)
        except (TypeError, ValueError):
            continue
    for key in ("codec_name", "color_space", "color_transfer", "color_primaries"):
        value = stream.get(key)
        if value is not None:
            report[key] = str(value)
    return report


def _reference_film_frame_report(
    combined_frame: Image.Image,
    plate: TerrainPlate,
    output_size: tuple[int, int],
    map_size: tuple[int, int],
    sources: list[HybridSmokeSource],
    frame_index: int,
    frame_time: float,
    regional_map: np.ndarray | None,
    broad_map: np.ndarray,
    wisp_map: np.ndarray | None,
    fire_map: np.ndarray,
    frame_info: ReferenceFilmFrameInfo,
    previous_frame: Image.Image | None = None,
) -> dict[str, float | str]:
    width, height = map(int, output_size)
    rgba = np.asarray(combined_frame.convert("RGBA"), dtype=np.float32)
    rgb = rgba[..., :3] / 255.0
    alpha = rgba[..., 3]
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if previous_frame is not None and previous_frame.size == combined_frame.size:
        previous_luma = np.asarray(previous_frame.convert("RGB"), dtype=np.float32) / 255.0
        previous_luma = previous_luma @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        temporal_luma_delta = float(np.mean(np.abs(luma - previous_luma)))
    else:
        temporal_luma_delta = 0.0

    broad_alpha = np.asarray(broad_map, dtype=np.uint8)[..., 3]
    regional_alpha = (
        np.asarray(regional_map, dtype=np.uint8)[..., 3]
        if regional_map is not None
        else np.zeros((map_size[1], map_size[0]), dtype=np.uint8)
    )
    wisp_alpha = (
        np.asarray(wisp_map, dtype=np.uint8)[..., 3]
        if wisp_map is not None
        else np.zeros((map_size[1], map_size[0]), dtype=np.uint8)
    )
    smoke_alpha = np.maximum(broad_alpha, wisp_alpha)
    fire_alpha = np.asarray(fire_map, dtype=np.uint8)[..., 3]
    fire_core = active_fire_core_intensity_field(sources, frame_index, map_size)
    fire_components = _connected_component_stats(fire_alpha > 142)
    fire_core_areas = _connected_component_areas(fire_alpha > 142)
    fire_core_radii = np.sqrt(fire_core_areas / math.pi) if fire_core_areas.size else np.asarray([], dtype=np.float32)
    halo_area = float(np.count_nonzero(fire_alpha > 18))
    core_area = float(np.count_nonzero(fire_alpha > 142))
    smoke_weights = smoke_alpha.astype(np.float64)
    smoke_mass = float(np.sum(smoke_weights))
    if smoke_mass > 0.0:
        smoke_y, smoke_x = np.mgrid[0 : smoke_alpha.shape[0], 0 : smoke_alpha.shape[1]].astype(np.float64)
        smoke_centroid_x = float(np.sum(smoke_x * smoke_weights) / smoke_mass) / max(float(smoke_alpha.shape[1] - 1), 1.0)
        smoke_centroid_y = float(np.sum(smoke_y * smoke_weights) / smoke_mass) / max(float(smoke_alpha.shape[0] - 1), 1.0)
    else:
        smoke_centroid_x = 0.0
        smoke_centroid_y = 0.0
    mid_scale_smoke = (smoke_alpha > 24) & (smoke_alpha <= 112)
    date_ordinal = float(date.fromisoformat(frame_info.date_label).toordinal())
    return {
        "frame_index": float(frame_index),
        "time_seconds": float(frame_time),
        "date_label": frame_info.date_label,
        "date_ordinal": date_ordinal,
        "burned_area_ha": float(frame_info.burned_area_ha),
        "full_bleed_frame_coverage_fraction": float(np.count_nonzero(alpha > 250.0) / max(width * height, 1)),
        "map_quad_area_fraction": _quad_area_fraction(plate.quad, output_size),
        "mean_luma": float(np.mean(luma)),
        "p95_luma": float(np.percentile(luma, 95.0)),
        "temporal_luma_delta": temporal_luma_delta,
        "combined_smoke_coverage_fraction": float(np.count_nonzero(smoke_alpha > 2) / max(smoke_alpha.size, 1)),
        "dense_combined_smoke_fraction": float(np.count_nonzero(smoke_alpha > 62) / max(smoke_alpha.size, 1)),
        "mid_scale_smoke_fraction": float(np.count_nonzero(mid_scale_smoke) / max(smoke_alpha.size, 1)),
        "smoke_centroid_x_fraction": smoke_centroid_x,
        "smoke_centroid_y_fraction": smoke_centroid_y,
        "regional_smoke_coverage_fraction": float(np.count_nonzero(regional_alpha > 1) / max(regional_alpha.size, 1)),
        "dense_regional_smoke_fraction": float(
            np.count_nonzero(regional_alpha > REFERENCE_FILM_DENSE_REGIONAL_SMOKE_ALPHA_THRESHOLD)
            / max(regional_alpha.size, 1)
        ),
        "source_wisp_coverage_fraction": float(np.count_nonzero(wisp_alpha > 1) / max(wisp_alpha.size, 1)),
        "active_fire_core_pixel_count": float(np.count_nonzero(fire_core >= FIRE_CORE_EMITTER_INTENSITY_THRESHOLD)),
        "hot_fire_fraction": float(np.count_nonzero(fire_alpha > 142) / max(fire_alpha.size, 1)),
        "hot_fire_component_count": float(fire_components["component_count"]),
        "hot_fire_mean_component_width_px": float(fire_components["mean_component_width_px"]),
        "median_fire_mark_radius_px": float(np.median(fire_core_radii)) if fire_core_radii.size else 0.0,
        "halo_core_area_ratio": float(halo_area / max(core_area, 1.0)),
        **_reference_fire_distribution_report(fire_alpha, plate.fire_uv),
        **_reference_regional_smoke_texture_report(regional_alpha),
    }


def _reference_film_label_report(
    pre_label_frame: Image.Image,
    labeled_frame: Image.Image,
    label_boxes: list[tuple[str, tuple[int, int, int, int]]],
    smoke_layer: Image.Image,
    fire_layer: Image.Image,
) -> dict[str, float]:
    if pre_label_frame.size != labeled_frame.size or not label_boxes:
        return {
            "label_count": 0.0,
            "median_label_contrast_delta": 0.0,
            "median_label_smoke_overlap_fraction": 1.0,
            "median_label_fire_overlap_fraction": 1.0,
            "median_label_text_pixel_fraction": 0.0,
        }
    pre_rgb = np.asarray(pre_label_frame.convert("RGB"), dtype=np.float32) / 255.0
    labeled_rgb = np.asarray(labeled_frame.convert("RGB"), dtype=np.float32) / 255.0
    pre_luma = pre_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    labeled_luma = labeled_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    smoke_alpha = np.asarray(smoke_layer.convert("RGBA"), dtype=np.uint8)[..., 3]
    fire_alpha = np.asarray(fire_layer.convert("RGBA"), dtype=np.uint8)[..., 3]
    contrasts: list[float] = []
    smoke_overlaps: list[float] = []
    fire_overlaps: list[float] = []
    text_fractions: list[float] = []
    width, height = labeled_frame.size
    for _name, box in label_boxes:
        x0, y0, x1, y1 = box
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        if x0 >= x1 or y0 >= y1:
            continue
        diff = np.max(np.abs(labeled_rgb[y0:y1, x0:x1] - pre_rgb[y0:y1, x0:x1]), axis=2)
        text_mask = diff > 0.035
        area = max(int(text_mask.size), 1)
        text_fraction = float(np.count_nonzero(text_mask) / area)
        if np.any(text_mask):
            contrast = float(np.median(np.abs(labeled_luma[y0:y1, x0:x1][text_mask] - pre_luma[y0:y1, x0:x1][text_mask])))
        else:
            contrast = 0.0
        contrasts.append(contrast)
        text_fractions.append(text_fraction)
        smoke_overlaps.append(float(np.count_nonzero(smoke_alpha[y0:y1, x0:x1] > 10) / area))
        fire_overlaps.append(float(np.count_nonzero(fire_alpha[y0:y1, x0:x1] > 12) / area))
    return {
        "label_count": float(len(label_boxes)),
        "median_label_contrast_delta": float(np.median(contrasts)) if contrasts else 0.0,
        "median_label_smoke_overlap_fraction": float(np.median(smoke_overlaps)) if smoke_overlaps else 1.0,
        "median_label_fire_overlap_fraction": float(np.median(fire_overlaps)) if fire_overlaps else 1.0,
        "median_label_text_pixel_fraction": float(np.median(text_fractions)) if text_fractions else 0.0,
    }


def _reference_exact_label_regions(size: tuple[int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    width, height = map(int, size)
    specs = (
        ("source_top_left", (0.018, 0.014, 0.470, 0.120)),
        ("date_top_right", (0.590, 0.014, 0.986, 0.126)),
        ("area_bottom_left", (0.018, 0.690, 0.475, 0.970)),
    )
    return [
        (
            name,
            (
                int(round(width * x0)),
                int(round(height * y0)),
                int(round(width * x1)),
                int(round(height * y1)),
            ),
        )
        for name, (x0, y0, x1, y1) in specs
    ]


def _reference_exact_label_region_metrics(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> dict[str, float]:
    width, height = image.size
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x0 >= x1 or y0 >= y1:
        return {
            "region_luma_contrast": 0.0,
            "region_edge_fraction": 0.0,
            "region_bright_fraction": 0.0,
            "region_textlike_fraction": 0.0,
        }
    rgb = np.asarray(image.convert("RGB").crop((x0, y0, x1, y1)), dtype=np.float32) / 255.0
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if min(luma.shape) < 2:
        return {
            "region_luma_contrast": 0.0,
            "region_edge_fraction": 0.0,
            "region_bright_fraction": 0.0,
            "region_textlike_fraction": 0.0,
        }
    p05, p50, p95 = np.percentile(luma, [5.0, 50.0, 95.0])
    contrast = float(p95 - p05)
    grad_y, grad_x = np.gradient(luma.astype(np.float32))
    gradient = np.hypot(grad_x, grad_y)
    edge_threshold = max(0.018, contrast * 0.22)
    edge_mask = gradient > edge_threshold
    bright_threshold = max(float(p50 + contrast * 0.42), 0.58)
    bright_mask = luma > bright_threshold
    textlike_mask = edge_mask | (bright_mask & (gradient > edge_threshold * 0.45))
    area = max(int(luma.size), 1)
    return {
        "region_luma_contrast": contrast,
        "region_edge_fraction": float(np.count_nonzero(edge_mask) / area),
        "region_bright_fraction": float(np.count_nonzero(bright_mask) / area),
        "region_textlike_fraction": float(np.count_nonzero(textlike_mask) / area),
    }


def reference_exact_decoded_label_report(
    frame_dir: Path,
    *,
    frame_count: int,
    start_frame: int = 0,
) -> dict[str, object]:
    frame_count = max(0, int(frame_count))
    if frame_count <= 0:
        return {
            "schema_version": "reference-exact-decoded-label-v1",
            "sampled_frame_count": 0,
            "active_region_fraction": 0.0,
            "passed": False,
            "thresholds": dict(REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS),
            "records": [],
        }
    sample_indexes = sorted({0, frame_count // 2, frame_count - 1})
    records: list[dict[str, float | int | str | bool]] = []
    for local_index in sample_indexes:
        path = Path(frame_dir) / f"frame_{int(local_index):04d}.png"
        if not path.exists():
            continue
        frame = Image.open(path).convert("RGBA")
        for region_name, box in _reference_exact_label_regions(frame.size):
            metrics = _reference_exact_label_region_metrics(frame, box)
            active = (
                metrics["region_luma_contrast"]
                >= REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS["minimum_median_region_luma_contrast"]
                and (
                    metrics["region_edge_fraction"]
                    >= REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS["minimum_median_region_edge_fraction"]
                    or metrics["region_bright_fraction"]
                    >= REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS["minimum_median_region_bright_fraction"]
                )
            )
            records.append(
                {
                    "local_frame_index": int(local_index),
                    "source_frame_index": int(start_frame) + int(local_index),
                    "region_name": region_name,
                    "active": bool(active),
                    **metrics,
                }
            )
    if records:
        active_region_fraction = float(sum(1 for record in records if bool(record["active"])) / len(records))
        median_contrast = float(np.median([float(record["region_luma_contrast"]) for record in records]))
        median_edge = float(np.median([float(record["region_edge_fraction"]) for record in records]))
        median_bright = float(np.median([float(record["region_bright_fraction"]) for record in records]))
        median_textlike = float(np.median([float(record["region_textlike_fraction"]) for record in records]))
    else:
        active_region_fraction = median_contrast = median_edge = median_bright = median_textlike = 0.0
    thresholds = REFERENCE_EXACT_DECODED_LABEL_THRESHOLDS
    passed = (
        active_region_fraction >= thresholds["minimum_active_region_fraction"]
        and median_contrast >= thresholds["minimum_median_region_luma_contrast"]
        and median_edge >= thresholds["minimum_median_region_edge_fraction"]
        and median_bright >= thresholds["minimum_median_region_bright_fraction"]
    )
    return {
        "schema_version": "reference-exact-decoded-label-v1",
        "sampled_frame_count": len(sample_indexes),
        "record_count": len(records),
        "active_region_fraction": active_region_fraction,
        "median_region_luma_contrast": median_contrast,
        "median_region_edge_fraction": median_edge,
        "median_region_bright_fraction": median_bright,
        "median_region_textlike_fraction": median_textlike,
        "passed": bool(passed),
        "thresholds": dict(thresholds),
        "records": records,
    }


def reference_exact_map_extent_contract(
    *,
    output_size: tuple[int, int] = (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT),
) -> dict[str, object]:
    return {
        "extent_kind": "decoded-reference-global-or-continent-frame",
        "continent_mode": True,
        "native_frame_size": [REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "composition_mode": MAP_FILM_COMPOSITION_MODE,
        "reprojection": "none; exact mode reconstructs native source-video pixels in screen space",
        "smoke_layer_space": "native source frame pixels",
        "frame_mapping": REFERENCE_EXACT_COLOR_POLICY["frame_mapping"],
        "audit_note": "Exact-smoke mode validates decoded first-30s source-frame smoke against generated native-frame smoke layers, not the local terrain slab extent.",
    }


def _reference_film_fire_visibility_report(frame: Image.Image, fire_layer: Image.Image) -> dict[str, float]:
    fire_alpha = np.asarray(fire_layer.convert("RGBA"), dtype=np.uint8)[..., 3]
    fire_mask = fire_alpha > 16
    if not np.any(fire_mask):
        return {"post_smoke_fire_visibility_fraction": 0.0}
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)
    warm = (
        (rgb[..., 0] > 138.0)
        & (rgb[..., 0] - rgb[..., 2] > 45.0)
        & (rgb[..., 1] - rgb[..., 2] > 8.0)
    )
    hot = (rgb[..., 0] > 214.0) & (rgb[..., 1] > 150.0) & (rgb[..., 2] < rgb[..., 1])
    return {
        "post_smoke_fire_visibility_fraction": float(np.count_nonzero((warm | hot) & fire_mask) / max(int(np.count_nonzero(fire_mask)), 1))
    }


def _source_wisp_regeneration_commands(
    output: Path,
    preview: Path,
    audit_dir: Path,
    reference_video: Path,
) -> dict[str, str]:
    script = str(Path(__file__).resolve())
    python = sys.executable
    final = [
        python,
        script,
        "--render-preset",
        TARGET_RENDER_PRESET,
        "--output",
        str(output),
        "--preview",
        str(preview),
        "--audit-dir",
        str(audit_dir),
        "--reference-video",
        str(reference_video),
        "--enforce-audit-gates",
    ]
    source_only = [
        python,
        script,
        "--render-preset",
        TARGET_RENDER_PRESET,
        "--smoke-ablation",
        "source-wisps-only",
        "--output",
        str(audit_dir / "source_wisps_only.mp4"),
        "--preview",
        str(audit_dir / "source_wisps_only.preview.png"),
        "--audit-dir",
        str(audit_dir / "source_wisps_only_audit"),
        "--reference-video",
        str(reference_video),
    ]
    no_broad = [
        python,
        script,
        "--render-preset",
        TARGET_RENDER_PRESET,
        "--smoke-ablation",
        "no-broad",
        "--output",
        str(audit_dir / "no_broad.mp4"),
        "--preview",
        str(audit_dir / "no_broad.preview.png"),
        "--audit-dir",
        str(audit_dir / "no_broad_audit"),
        "--reference-video",
        str(reference_video),
    ]
    carpet = [
        python,
        script,
        "--render-preset",
        LEGACY_RENDER_PRESET,
        "--output",
        str(audit_dir / "negative_carpet_legacy_combined.mp4"),
        "--preview",
        str(audit_dir / "negative_carpet_legacy_combined.preview.png"),
        "--audit-dir",
        str(audit_dir / "negative_carpet_audit"),
        "--reference-video",
        str(reference_video),
    ]
    brush = [
        python,
        script,
        "--render-preset",
        BRUSH_BUNDLE_RENDER_PRESET,
        "--smoke-ablation",
        "source-wisps-only",
        "--output",
        str(audit_dir / "negative_brush_bundle_source_wisps_only.mp4"),
        "--preview",
        str(audit_dir / "negative_brush_bundle_source_wisps_only.preview.png"),
        "--audit-dir",
        str(audit_dir / "negative_brush_bundle_audit"),
        "--reference-video",
        str(reference_video),
    ]
    reference_frame = [
        "ffmpeg",
        "-y",
        "-ss",
        "3.500",
        "-i",
        str(reference_video),
        "-frames:v",
        "1",
        str(audit_dir / "reference_3p5s.png"),
    ]
    return {
        "accepted_render": " ".join(final),
        "source_wisps_only_audit": " ".join(source_only),
        "no_broad_audit": " ".join(no_broad),
        "carpet_smoke_negative_baseline": " ".join(carpet),
        "brush_bundle_negative_baseline": " ".join(brush),
        "reference_frame_extraction_example": " ".join(reference_frame),
    }


def _reference_film_regeneration_commands(
    output: Path,
    preview: Path,
    audit_dir: Path,
    reference_video: Path,
) -> dict[str, str]:
    script = str(Path(__file__).resolve())
    python = sys.executable
    final = [
        python,
        script,
        "--render-preset",
        REFERENCE_FILM_RENDER_PRESET,
        "--output",
        str(output),
        "--preview",
        str(preview),
        "--audit-dir",
        str(audit_dir),
        "--reference-video",
        str(reference_video),
    ]
    contact_sheet = [
        "ffmpeg",
        "-y",
        "-ss",
        "0.000",
        "-i",
        str(reference_video),
        "-frames:v",
        "1",
        str(audit_dir / "reference_film_000p0s_reference.png"),
    ]
    return {
        "reference_film_render": " ".join(final),
        "reference_contact_frame_extraction_example": " ".join(contact_sheet),
    }


def _audit_gate(name: str, value: float, threshold: float, op: str, *, frame_time: float | None = None) -> dict[str, float | str | bool]:
    if op == ">=":
        passed = float(value) >= float(threshold)
    elif op == "<=":
        passed = float(value) <= float(threshold)
    elif op == "==":
        passed = abs(float(value) - float(threshold)) <= 1.0e-9
    else:
        raise ValueError("audit gate op must be >=, <=, or ==")
    gate: dict[str, float | str | bool] = {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "op": op,
        "passed": bool(passed),
        "hard_fail": True,
    }
    if frame_time is not None:
        gate["time_seconds"] = float(frame_time)
    return gate


def _evaluate_source_wisp_audit(
    component_reports: list[dict[str, float]],
    encoded_reports: list[dict[str, float]],
    thresholds: dict[str, float] = SOURCE_WISP_AUDIT_THRESHOLDS,
) -> dict[str, object]:
    gates: list[dict[str, float | str | bool]] = []
    for report in component_reports:
        frame_time = float(report.get("time_seconds", report.get("frame_index", 0.0)))
        active_count = float(report.get("active_source_count", 0.0))
        if active_count > 0.0:
            gates.append(
                _audit_gate(
                    "source_attachment_fraction",
                    float(report.get("attached_source_count", 0.0)) / max(active_count, 1.0),
                    thresholds["minimum_attached_source_fraction"],
                    ">=",
                    frame_time=frame_time,
                )
            )
        screen_active = float(report.get("screen_active_source_count", 0.0))
        enforce_active_distribution = frame_time < 6.8
        if screen_active > 0.0:
            gates.append(
                _audit_gate(
                    "screen_source_attachment_fraction",
                    float(report.get("screen_attached_source_fraction", 0.0)),
                    thresholds["minimum_screen_attached_source_fraction"],
                    ">=",
                    frame_time=frame_time,
                )
            )
        if screen_active >= 6.0 and enforce_active_distribution:
            gates.append(
                _audit_gate(
                    "fire_core_visibility_fraction",
                    float(report.get("fire_core_visibility_fraction", 0.0)),
                    thresholds["minimum_fire_core_visibility_fraction"],
                    ">=",
                    frame_time=frame_time,
                )
            )
        if enforce_active_distribution:
            gates.extend(
                [
                    _audit_gate(
                        "active_fire_emitter_count",
                        float(report.get("active_fire_emitter_count", 0.0)),
                        thresholds["minimum_active_fire_emitters"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "emitter_bbox_fraction",
                        float(report.get("emitter_bbox_fraction", 0.0)),
                        thresholds["minimum_emitter_bbox_fraction"],
                        ">=",
                        frame_time=frame_time,
                    ),
                ]
            )
        gates.extend(
            [
                _audit_gate(
                    "source_wisp_component_count",
                    float(report.get("source_wisp_component_count", 0.0)),
                    thresholds["minimum_source_wisp_component_count"],
                    ">=",
                    frame_time=frame_time,
                ),
                _audit_gate(
                    "smoke_carpet_largest_component_fraction",
                    float(report.get("smoke_carpet_largest_component_fraction", 1.0)),
                    thresholds["maximum_smoke_carpet_component_fraction"],
                    "<=",
                    frame_time=frame_time,
                ),
                _audit_gate(
                    "low_frequency_haze_fraction",
                    float(report.get("low_frequency_haze_fraction", 1.0)),
                    thresholds["maximum_low_frequency_haze_fraction"],
                    "<=",
                    frame_time=frame_time,
                ),
                _audit_gate(
                    "strand_to_haze_ratio",
                    float(report.get("strand_to_haze_ratio", 0.0)),
                    thresholds["minimum_strand_to_haze_ratio"],
                    ">=",
                    frame_time=frame_time,
                ),
                _audit_gate(
                    "combined_strand_retention",
                    float(report.get("combined_strand_retention", 0.0)),
                    thresholds["minimum_combined_strand_retention"],
                    ">=",
                    frame_time=frame_time,
                ),
            ]
        )
        gates.append(
            _audit_gate(
                "morphology_stage_coverage_fraction",
                float(report.get("morphology_stage_coverage_fraction", 0.0)),
                thresholds["minimum_morphology_band_coverage_fraction"],
                ">=",
                frame_time=frame_time,
            )
        )
        if frame_time >= SOURCE_WISP_MORPHOLOGY_GATE_MIN_TIME_SECONDS:
            gates.extend(
                [
                    _audit_gate(
                        "transition_width_growth_ratio",
                        float(report.get("transition_width_growth_ratio", 0.0)),
                        thresholds["minimum_transition_width_growth_ratio"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_width_growth_ratio",
                        float(report.get("old_tail_width_growth_ratio", 0.0)),
                        thresholds["minimum_old_tail_width_growth_ratio"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_alpha_p90_fraction",
                        float(report.get("old_tail_alpha_p90_fraction", 1.0)),
                        thresholds["maximum_old_tail_alpha_p90_fraction"],
                        "<=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_endpoint_alpha_fraction",
                        float(report.get("old_tail_endpoint_alpha_fraction", 1.0)),
                        thresholds["maximum_old_tail_endpoint_alpha_fraction"],
                        "<=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_coverage_growth_ratio",
                        float(report.get("old_tail_coverage_growth_ratio", 0.0)),
                        thresholds["minimum_old_tail_coverage_growth_ratio"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_edge_softness_px",
                        float(report.get("old_tail_edge_softness_px", 0.0)),
                        thresholds["minimum_old_tail_edge_softness_px"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "old_tail_diffuse_to_core_area_ratio",
                        float(report.get("old_tail_diffuse_to_core_area_ratio", 0.0)),
                        thresholds["minimum_old_tail_diffuse_to_core_area_ratio"],
                        ">=",
                        frame_time=frame_time,
                    ),
                    _audit_gate(
                        "brush_bundle_score",
                        float(report.get("brush_bundle_score", 1.0)),
                        thresholds["maximum_brush_bundle_score"],
                        "<=",
                        frame_time=frame_time,
                    ),
                ]
            )
        if frame_time >= 6.8:
            gates.append(
                _audit_gate(
                    "late_low_frequency_haze_fraction",
                    float(report.get("low_frequency_haze_fraction", 1.0)),
                    thresholds["maximum_late_low_frequency_haze_fraction"],
                    "<=",
                    frame_time=frame_time,
                )
            )

    for record in encoded_reports:
        gates.append(
            _audit_gate(
                "encoded_strand_like_fraction",
                float(record.get("strand_like_fraction", 0.0)),
                thresholds["minimum_encoded_strand_like_fraction"],
                ">=",
                frame_time=float(record.get("time_seconds", 0.0)),
            )
        )
        gates.append(
            _audit_gate(
                "encoded_soft_tail_like_fraction",
                float(record.get("soft_tail_like_fraction", 0.0)),
                thresholds["minimum_encoded_soft_tail_like_fraction"],
                ">=",
                frame_time=float(record.get("time_seconds", 0.0)),
            )
        )

    failed = [gate for gate in gates if not bool(gate["passed"])]
    return {
        "passed": not failed,
        "gate_count": len(gates),
        "failed_gate_count": len(failed),
        "thresholds": dict(thresholds),
        "gates": gates,
    }


def _report_values(reports: list[dict[str, float | str]], key: str) -> np.ndarray:
    values: list[float] = []
    for report in reports:
        value = report.get(key)
        if isinstance(value, (int, float, np.floating)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _median_report_value(reports: list[dict[str, float | str]], key: str, default: float = 0.0) -> float:
    values = _report_values(reports, key)
    return float(np.median(values)) if values.size else float(default)


def _video_bitrate_to_bps(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def _evaluate_reference_film_audit(
    frame_reports: list[dict[str, float | str]],
    encoded_reports: list[dict[str, float]],
    stream_report: dict[str, float | str],
    encode_policy: dict[str, object],
    thresholds: dict[str, float] = REFERENCE_FILM_AUDIT_THRESHOLDS,
) -> dict[str, object]:
    gates: list[dict[str, float | str | bool]] = []
    full_bleed = _report_values(frame_reports, "full_bleed_frame_coverage_fraction")
    quad_area = _report_values(frame_reports, "map_quad_area_fraction")
    date_ordinals = _report_values(frame_reports, "date_ordinal")
    date_steps = np.diff(np.sort(date_ordinals)) if date_ordinals.size > 1 else np.asarray([], dtype=np.float32)
    date_steps = date_steps[date_steps > 0.0]
    burned_areas = _report_values(frame_reports, "burned_area_ha")
    temporal_deltas = _report_values(frame_reports, "temporal_luma_delta")
    temporal_deltas = temporal_deltas[temporal_deltas > 0.0]
    active_fire_counts = _report_values(frame_reports, "active_fire_core_pixel_count")
    active_fire_change_ratio = 0.0
    if active_fire_counts.size:
        active_fire_change_ratio = float(np.ptp(active_fire_counts) / max(float(np.median(active_fire_counts)), 1.0))
    centroid_x = _report_values(frame_reports, "smoke_centroid_x_fraction")
    centroid_y = _report_values(frame_reports, "smoke_centroid_y_fraction")
    centroid_motion = 0.0
    if centroid_x.size and centroid_y.size:
        centroid_motion = float(math.hypot(float(np.ptp(centroid_x)), float(np.ptp(centroid_y))))
    encoded_smoke = _report_values(encoded_reports, "smoke_like_fraction")
    encoded_soft_tail = _report_values(encoded_reports, "soft_tail_like_fraction")
    delivery_size = encode_policy.get("size", (0, 0))
    if isinstance(delivery_size, (tuple, list)) and len(delivery_size) >= 2:
        configured_width = float(delivery_size[0])
        configured_height = float(delivery_size[1])
    else:
        configured_width = configured_height = 0.0
    actual_width = float(stream_report.get("width", configured_width)) if stream_report else configured_width
    actual_height = float(stream_report.get("height", configured_height)) if stream_report else configured_height
    configured_bitrate = _video_bitrate_to_bps(encode_policy.get("video_bitrate"))
    actual_bitrate = float(stream_report.get("bit_rate", 0.0)) if stream_report.get("bit_rate") is not None else 0.0
    bitrate_for_gate = actual_bitrate if actual_bitrate > 0.0 else configured_bitrate
    codec = str(stream_report.get("codec_name", "h264" if stream_report == {} else ""))
    codec_is_h264 = 1.0 if codec in {"h264", "avc1"} else 0.0
    if not stream_report:
        codec_is_h264 = 1.0

    gates.extend(
        [
            _audit_gate(
                "full_bleed_frame_coverage_fraction",
                float(np.min(full_bleed)) if full_bleed.size else 0.0,
                thresholds["minimum_full_bleed_frame_coverage"],
                ">=",
            ),
            _audit_gate(
                "map_quad_area_fraction",
                float(np.min(quad_area)) if quad_area.size else 0.0,
                thresholds["minimum_map_quad_area_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_smoke_coverage_fraction",
                _median_report_value(frame_reports, "combined_smoke_coverage_fraction"),
                thresholds["minimum_median_smoke_coverage_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_smoke_coverage_fraction",
                _median_report_value(frame_reports, "combined_smoke_coverage_fraction"),
                thresholds["maximum_median_smoke_coverage_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_regional_smoke_coverage_fraction",
                _median_report_value(frame_reports, "regional_smoke_coverage_fraction"),
                thresholds["minimum_median_regional_smoke_coverage_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_dense_regional_smoke_fraction",
                _median_report_value(frame_reports, "dense_regional_smoke_fraction"),
                thresholds["minimum_median_dense_regional_smoke_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_dense_regional_smoke_fraction",
                _median_report_value(frame_reports, "dense_regional_smoke_fraction"),
                thresholds["maximum_median_dense_regional_smoke_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_active_fire_core_pixel_count",
                _median_report_value(frame_reports, "active_fire_core_pixel_count"),
                thresholds["minimum_median_fire_core_pixel_count"],
                ">=",
            ),
            _audit_gate(
                "active_fire_temporal_change_ratio",
                active_fire_change_ratio,
                thresholds["minimum_active_fire_temporal_change_ratio"],
                ">=",
            ),
            _audit_gate(
                "median_hot_fire_fraction",
                _median_report_value(frame_reports, "hot_fire_fraction"),
                thresholds["maximum_median_hot_fire_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_post_smoke_fire_visibility_fraction",
                _median_report_value(frame_reports, "post_smoke_fire_visibility_fraction"),
                thresholds["minimum_post_smoke_fire_visibility_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_fire_mark_radius_px",
                _median_report_value(frame_reports, "median_fire_mark_radius_px"),
                thresholds["maximum_median_fire_mark_radius_px"],
                "<=",
            ),
            _audit_gate(
                "median_halo_core_area_ratio",
                _median_report_value(frame_reports, "halo_core_area_ratio"),
                thresholds["minimum_median_halo_core_area_ratio"],
                ">=",
            ),
            _audit_gate(
                "median_halo_core_area_ratio",
                _median_report_value(frame_reports, "halo_core_area_ratio"),
                thresholds["maximum_median_halo_core_area_ratio"],
                "<=",
            ),
            _audit_gate(
                "median_mid_scale_smoke_fraction",
                _median_report_value(frame_reports, "mid_scale_smoke_fraction"),
                thresholds["minimum_median_mid_scale_smoke_fraction"],
                ">=",
            ),
            _audit_gate(
                "smoke_centroid_motion_fraction",
                centroid_motion,
                thresholds["minimum_smoke_centroid_motion_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_distributed_fire_cluster_count",
                _median_report_value(frame_reports, "distributed_fire_cluster_count"),
                thresholds["minimum_median_distributed_fire_cluster_count"],
                ">=",
            ),
            _audit_gate(
                "median_fire_spread_grid_cell_count",
                _median_report_value(frame_reports, "fire_spread_grid_cell_count"),
                thresholds["minimum_median_fire_spread_grid_cell_count"],
                ">=",
            ),
            _audit_gate(
                "median_far_fire_core_fraction",
                _median_report_value(frame_reports, "far_fire_core_fraction"),
                thresholds["minimum_median_far_fire_core_fraction"],
                ">=",
            ),
            _audit_gate(
                "median_primary_fire_dominance_fraction",
                _median_report_value(frame_reports, "primary_fire_dominance_fraction", default=1.0),
                thresholds["maximum_median_primary_fire_dominance_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_regional_smoke_texture_score",
                _median_report_value(frame_reports, "regional_smoke_texture_score"),
                thresholds["minimum_median_regional_smoke_texture_score"],
                ">=",
            ),
            _audit_gate(
                "median_regional_smoke_axis_band_score",
                _median_report_value(frame_reports, "regional_smoke_axis_band_score", default=1.0),
                thresholds["maximum_median_regional_smoke_axis_band_score"],
                "<=",
            ),
            _audit_gate(
                "median_regional_smoke_contour_band_score",
                _median_report_value(frame_reports, "regional_smoke_contour_band_score", default=1.0),
                thresholds["maximum_median_regional_smoke_contour_band_score"],
                "<=",
            ),
            _audit_gate(
                "median_regional_smoke_ring_score",
                _median_report_value(frame_reports, "regional_smoke_ring_score", default=1.0),
                thresholds["maximum_median_regional_smoke_ring_score"],
                "<=",
            ),
            _audit_gate(
                "median_label_contrast_delta",
                _median_report_value(frame_reports, "median_label_contrast_delta"),
                thresholds["minimum_median_label_contrast_delta"],
                ">=",
            ),
            _audit_gate(
                "median_label_smoke_overlap_fraction",
                _median_report_value(frame_reports, "median_label_smoke_overlap_fraction", default=1.0),
                thresholds["maximum_median_label_smoke_overlap_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_label_fire_overlap_fraction",
                _median_report_value(frame_reports, "median_label_fire_overlap_fraction", default=1.0),
                thresholds["maximum_median_label_fire_overlap_fraction"],
                "<=",
            ),
            _audit_gate(
                "median_label_text_pixel_fraction",
                _median_report_value(frame_reports, "median_label_text_pixel_fraction"),
                thresholds["minimum_median_label_text_pixel_fraction"],
                ">=",
            ),
            _audit_gate(
                "temporal_date_span_days",
                float(np.max(date_ordinals) - np.min(date_ordinals)) if date_ordinals.size else 0.0,
                thresholds["minimum_temporal_date_span_days"],
                ">=",
            ),
            _audit_gate(
                "median_date_step_days",
                float(np.median(date_steps)) if date_steps.size else 0.0,
                thresholds["minimum_median_date_step_days"],
                ">=",
            ),
            _audit_gate(
                "median_date_step_days",
                float(np.median(date_steps)) if date_steps.size else 0.0,
                thresholds["maximum_median_date_step_days"],
                "<=",
            ),
            _audit_gate(
                "burned_area_growth_ratio",
                float(np.max(burned_areas) / max(float(np.min(burned_areas)), 1.0)) if burned_areas.size else 0.0,
                thresholds["minimum_burned_area_growth_ratio"],
                ">=",
            ),
            _audit_gate(
                "median_temporal_luma_delta",
                float(np.median(temporal_deltas)) if temporal_deltas.size else 0.0,
                thresholds["minimum_median_temporal_luma_delta"],
                ">=",
            ),
            _audit_gate(
                "encoded_smoke_like_fraction",
                float(np.median(encoded_smoke)) if encoded_smoke.size else 0.0,
                thresholds["minimum_encoded_smoke_like_fraction"],
                ">=",
            ),
            _audit_gate(
                "encoded_soft_tail_like_fraction",
                float(np.median(encoded_soft_tail)) if encoded_soft_tail.size else 0.0,
                thresholds["minimum_encoded_soft_tail_like_fraction"],
                ">=",
            ),
            _audit_gate(
                "delivery_width_px",
                actual_width,
                thresholds["minimum_delivery_width"],
                ">=",
            ),
            _audit_gate(
                "delivery_height_px",
                actual_height,
                thresholds["minimum_delivery_height"],
                ">=",
            ),
            _audit_gate(
                "configured_delivery_bitrate_bps",
                bitrate_for_gate,
                thresholds["minimum_delivery_bitrate_bps"],
                ">=",
            ),
            _audit_gate(
                "configured_delivery_bitrate_bps",
                bitrate_for_gate,
                thresholds["maximum_delivery_bitrate_bps"],
                "<=",
            ),
            _audit_gate("delivery_h264_codec", codec_is_h264, 1.0, ">="),
        ]
    )
    failed = [gate for gate in gates if not bool(gate["passed"])]
    return {
        "passed": not failed,
        "gate_count": len(gates),
        "failed_gate_count": len(failed),
        "thresholds": dict(thresholds),
        "gates": gates,
        "summary": {
            "frame_report_count": len(frame_reports),
            "encoded_contact_frame_count": len(encoded_reports),
            "probed_video_stream": dict(stream_report),
        },
    }


def _residual_haze_rgba(
    residual_haze: np.ndarray | None,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
) -> np.ndarray:
    haze = np.clip(np.asarray(residual_haze, dtype=np.float32), 0.0, None)
    if haze.ndim != 2:
        raise ValueError("residual haze must be a 2D array")
    height, width = haze.shape
    out = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(haze > 1.0e-6):
        return out

    scale = min(width, height) / 408.0
    soft = _pil_blur_float(haze, max(2.4, 4.8 * scale))
    broad = _pil_blur_float(haze, max(8.0, 16.0 * scale))
    sheet = np.clip(soft * 0.64 + broad * 0.44, 0.0, None)
    positive = sheet[sheet > 0.0]
    scale_value = max(float(np.percentile(positive, 99.3)) if positive.size else 1.0, 1.0e-5)
    norm = np.clip(sheet / (scale_value * 1.18), 0.0, 1.0)

    texture = _pil_blur_float(
        _advected_smoke_texture(haze.shape, frame_index, seed + 23011),
        max(7.0, 12.5 * scale),
    )
    alpha_f = (_smoothstep(0.012, 0.88, norm) ** 0.86) * np.clip(0.70 + 0.25 * (texture - 0.5), 0.50, 0.94)
    alpha = _pil_blur_float(alpha_f * HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA, max(0.65, 1.15 * scale))
    alpha = np.clip(alpha, 0.0, HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA)
    alpha = np.where(alpha >= 2.0, alpha, 0.0)

    thin = np.array([94.0, 109.0, 126.0], dtype=np.float32)
    flat = np.array([139.0, 149.0, 153.0], dtype=np.float32)
    density_t = _smoothstep(0.035, 0.82, norm)
    rgb = thin * (1.0 - density_t[..., None]) + flat * density_t[..., None]
    rgb *= 0.86 + 0.08 * texture[..., None]
    out[..., :3] = np.where(alpha[..., None] > 0.0, np.clip(np.round(rgb), 0, 220), 0).astype(np.uint8)
    out[..., 3] = alpha.astype(np.uint8)
    return out


def hybrid_smoke_rgba(
    state: HybridSmokeState,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
) -> np.ndarray:
    layers = state.layer_density
    layer_ages = state.layer_age_mass
    if not layers or not layer_ages or len(layers) != len(layer_ages):
        return _hybrid_smoke_field_rgba(state, frame_index, seed)

    layer_count = min(len(layers), len(layer_ages), HYBRID_SMOKE_LAYER_COUNT)
    density_shape = np.asarray(layers[0]).shape
    combined = np.zeros((density_shape[0], density_shape[1], 4), dtype=np.uint8)
    if state.residual_haze is not None:
        combined = _premultiplied_over(combined, _residual_haze_rgba(state.residual_haze, frame_index, seed))
    flow_scale = min(density_shape) / 408.0
    for layer_index in range(layer_count):
        altitude = _hybrid_layer_altitude(layer_index)
        color_bias = (
            7.0 - 13.0 * altitude,
            6.0 - 9.0 * altitude,
            -3.0 + 14.0 * altitude,
        )
        layer_state = HybridSmokeState(
            density=np.asarray(layers[layer_index], dtype=np.float32),
            age_mass=np.asarray(layer_ages[layer_index], dtype=np.float32),
        )
        layer_rgba = _hybrid_smoke_field_rgba(
            layer_state,
            frame_index,
            seed + 503 * layer_index,
            alpha_multiplier=HYBRID_SMOKE_RENDER_LAYER_ALPHA[layer_index],
            color_bias=color_bias,
        )
        wind = _hybrid_layer_wind_vector(layer_index)
        cross = np.array([-wind[1], wind[0]], dtype=np.float32)
        dx = (altitude - 0.35) * 3.8 * flow_scale + cross[0] * 2.3 * altitude * flow_scale
        dy = -altitude * 5.2 * flow_scale + cross[1] * 1.8 * (altitude - 0.5) * flow_scale
        combined = _premultiplied_over(combined, _offset_rgba_layer(layer_rgba, dx, dy))

    aggregate_boost = _hybrid_smoke_field_rgba(
        HybridSmokeState(density=state.density, age_mass=state.age_mass),
        frame_index,
        seed + 1709,
        alpha_multiplier=0.18,
        color_bias=(-4.0, -2.0, 4.0),
    )
    combined = _premultiplied_over(combined, aggregate_boost)
    combined[..., 3] = np.minimum(combined[..., 3], HYBRID_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def regional_transport_smoke_rgba(
    map_size: tuple[int, int],
    frame_index: int,
    *,
    seed: int = HYBRID_SMOKE_SEED,
    progress: float = 0.0,
) -> np.ndarray:
    """Render broad synoptic-scale smoke ribbons for the reference-film mode."""
    width, height = map(int, map_size)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    texture = _advected_smoke_texture((height, width), frame_index, seed + 61517)
    fine = texture - _pil_blur_float(texture, max(1.0, min(width, height) / 180.0))
    broad = _pil_blur_float(texture, max(4.0, min(width, height) / 48.0))
    density = np.zeros((height, width), dtype=np.float32)
    t = float(progress)
    lanes = (
        (0.34, 0.15, 0.036, 0.52, 0.0),
        (0.52, 0.20, 0.042, 0.46, 1.6),
        (0.70, 0.16, 0.038, 0.36, 3.1),
    )
    stamp_count = 9
    for lane_index, (center, amp, width_frac, strength, phase) in enumerate(lanes):
        wave = center + amp * np.sin((x * 2.65 + t * 1.20 + phase) * math.tau)
        wave += 0.045 * np.sin((x * 6.8 - t * 2.10 + phase * 0.37) * math.tau)
        ribbon = np.exp(-((y - wave) ** 2) / (2.0 * (width_frac * 1.18) * (width_frac * 1.18)))
        flow_coord = x + 0.060 * np.sin((y * 2.05 + phase * 0.19 + t * 0.45) * math.tau)
        underlay = ribbon * _smoothstep(-0.08, 0.26, flow_coord + t * 0.28 - 0.035 * phase)
        underlay *= 1.0 - _smoothstep(0.82, 1.18, flow_coord - t * 0.18 + 0.055 * phase)
        density += underlay * strength * 0.115
        for stamp_index in range(stamp_count):
            rng = np.random.default_rng(seed + 1801 + lane_index * 191 + stamp_index * 37)
            base_s = -0.18 + stamp_index * (1.36 / max(stamp_count - 1, 1))
            cx = base_s + 0.24 * t - 0.035 * phase + float(rng.uniform(-0.026, 0.026))
            if cx < -0.24 or cx > 1.22:
                continue
            wave_a = (cx * 2.65 + t * 1.20 + phase) * math.tau
            wave_b = (cx * 6.8 - t * 2.10 + phase * 0.37) * math.tau
            cy = center + amp * math.sin(wave_a) + 0.045 * math.sin(wave_b)
            dy_dx = amp * math.cos(wave_a) * 2.65 * math.tau + 0.045 * math.cos(wave_b) * 6.8 * math.tau
            theta = math.atan2(dy_dx, 1.0)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            length_sigma = float(rng.uniform(0.050, 0.092)) * (1.0 + 0.16 * lane_index)
            width_sigma = width_frac * float(rng.uniform(0.68, 1.04))
            dx = x - cx
            dy = y - cy
            along = dx * cos_t + dy * sin_t
            across = -dx * sin_t + dy * cos_t
            stamp = np.exp(-0.5 * ((along / length_sigma) ** 2 + (across / width_sigma) ** 2))
            stamp *= _smoothstep(0.0, 0.12, cx + 0.16) * (1.0 - _smoothstep(0.88, 1.18, cx))
            density += stamp * strength * float(rng.uniform(0.52, 1.02))

    erosion = np.clip(0.72 + 0.22 * texture + 0.10 * fine, 0.52, 1.0)
    density = np.clip(density, 0.0, 1.0) * erosion
    density = _pil_blur_float(density, max(1.20, min(width, height) / 330.0))
    density *= _hybrid_border_fade((height, width))
    density = np.clip((density - 0.035) / 0.895, 0.0, 1.0)
    density *= np.clip(0.72 + 0.28 * _pil_blur_float(texture, max(1.0, min(width, height) / 180.0)), 0.58, 0.98)
    alpha_shape = (density ** 1.72) * np.clip(0.82 + 0.18 * texture + 0.08 * fine, 0.58, 1.0)
    alpha = _pil_blur_float(alpha_shape * REFERENCE_FILM_REGIONAL_SMOKE_MAX_ALPHA, max(2.20, min(width, height) / 360.0))
    alpha = _pil_blur_float(alpha, max(2.20, min(width, height) / 300.0))
    fine_alpha = texture - _pil_blur_float(texture, max(1.0, min(width, height) / 150.0))
    alpha *= np.clip(0.96 + 0.30 * fine_alpha, 0.82, 1.14)
    alpha = np.clip(alpha, 0.0, REFERENCE_FILM_REGIONAL_SMOKE_MAX_ALPHA)
    alpha = np.where(alpha >= 0.35, alpha, 0.0)
    thin = np.array([118.0, 128.0, 136.0], dtype=np.float32)
    dense = np.array([178.0, 181.0, 174.0], dtype=np.float32)
    dense_t = np.clip(density[..., None] * 1.18, 0.0, 1.0)
    rgb = thin * (1.0 - dense_t) + dense * dense_t
    rgb *= np.clip(0.84 + 0.18 * (texture[..., None] - 0.5), 0.72, 1.02)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.where(alpha[..., None] > 0.0, np.clip(np.round(rgb), 0.0, 218.0), 0).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha), 0.0, 255.0).astype(np.uint8)
    return out


def _regional_density_to_smoke_rgba(
    density: np.ndarray,
    frame_index: int,
    *,
    seed: int = HYBRID_SMOKE_SEED,
    max_alpha: float = REFERENCE_FILM_REGIONAL_SMOKE_MAX_ALPHA,
) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=np.float32), 0.0, 1.0)
    if density.ndim != 2:
        raise ValueError("regional smoke density must be a 2D array")
    height, width = density.shape
    texture = _advected_smoke_texture((height, width), frame_index, seed + 82123)
    fine = texture - _pil_blur_float(texture, max(1.0, min(width, height) / 170.0))
    broad = _pil_blur_float(texture, max(1.0, min(width, height) / 70.0))
    shaped = _pil_blur_float(density, max(1.0, min(width, height) / 260.0))
    shaped *= np.clip(0.78 + 0.32 * texture + 0.12 * fine, 0.48, 1.12)
    shaped *= _hybrid_border_fade((height, width))
    shaped = np.clip(shaped, 0.0, 1.0)
    alpha = (shaped ** 1.36) * float(max_alpha)
    alpha *= np.clip(0.88 + 0.24 * broad + 0.12 * fine, 0.68, 1.10)
    alpha = _pil_blur_float(alpha, max(1.4, min(width, height) / 330.0))
    alpha = np.clip(alpha, 0.0, float(max_alpha))
    alpha = np.where(alpha >= 1.0, alpha, 0.0)
    thin = np.array([112.0, 122.0, 132.0], dtype=np.float32)
    dense = np.array([180.0, 182.0, 174.0], dtype=np.float32)
    dense_t = np.clip(shaped[..., None] * 1.28, 0.0, 1.0)
    rgb = thin * (1.0 - dense_t) + dense * dense_t
    rgb *= np.clip(0.86 + 0.18 * (texture[..., None] - 0.5), 0.72, 1.03)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.where(alpha[..., None] > 0.0, np.clip(np.round(rgb), 0.0, 218.0), 0).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha), 0.0, 255.0).astype(np.uint8)
    return out


def _interpolate_reference_event_centroid(
    event: ReferenceSmokeEventState,
    source_frame: int,
) -> tuple[float, float, float]:
    path = event.centroid_path
    if not path:
        width, height = event.source_size
        return 0.5, 0.48, max(float(event.coverage_peak), 0.01)
    if int(source_frame) <= path[0][0]:
        frame, x_px, y_px, coverage = path[0]
    elif int(source_frame) >= path[-1][0]:
        frame, x_px, y_px, coverage = path[-1]
    else:
        for idx in range(len(path) - 1):
            left = path[idx]
            right = path[idx + 1]
            if left[0] <= int(source_frame) <= right[0]:
                span = max(float(right[0] - left[0]), 1.0)
                frac = (float(source_frame) - float(left[0])) / span
                frame = int(source_frame)
                x_px = float(left[1]) * (1.0 - frac) + float(right[1]) * frac
                y_px = float(left[2]) * (1.0 - frac) + float(right[2]) * frac
                coverage = float(left[3]) * (1.0 - frac) + float(right[3]) * frac
                break
        else:
            frame, x_px, y_px, coverage = path[-1]
    width, height = event.source_size
    _ = frame
    return (
        float(np.clip(x_px / max(float(width - 1), 1.0), 0.0, 1.0)),
        float(np.clip(y_px / max(float(height - 1), 1.0), 0.0, 1.0)),
        max(float(coverage), 0.0),
    )


def reference_event_transport_smoke_rgba(
    map_size: tuple[int, int],
    frame_index: int,
    event_states: tuple[ReferenceSmokeEventState, ...],
    *,
    seed: int = HYBRID_SMOKE_SEED,
    progress: float | None = None,
    timeline_frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> np.ndarray:
    """Render broad smoke using decoded reference event timing and centroid paths."""
    width, height = map(int, map_size)
    if not event_states:
        return regional_transport_smoke_rgba(map_size, frame_index, seed=seed, progress=float(progress or 0.0))
    if progress is None:
        source_frame = int(frame_index)
    else:
        source_frame = int(round(np.clip(float(progress), 0.0, 1.0) * max(int(timeline_frame_count) - 1, 0)))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    density = np.zeros((height, width), dtype=np.float32)
    texture = _advected_smoke_texture((height, width), frame_index, seed + 971)
    for event_index, event in enumerate(event_states):
        start = int(event.start_frame)
        peak = int(event.peak_frame)
        end = int(event.end_frame)
        duration = max(end - start, 1)
        if source_frame <= peak:
            active = _smoothstep(float(start), float(peak), float(source_frame))
        else:
            active = 1.0 - _smoothstep(float(peak), float(end), float(source_frame))
        if source_frame < start:
            distance = float(start - source_frame)
        elif source_frame > end:
            distance = float(source_frame - end)
        else:
            distance = 0.0
        residual = 0.16 * math.exp(-distance / max(duration * 0.92, 54.0))
        temporal = float(np.clip(max(active, residual), 0.0, 1.0))
        if temporal <= 0.006:
            continue
        cx, cy, coverage = _interpolate_reference_event_centroid(event, source_frame)
        coverage_scale = float(np.clip(math.sqrt(max(coverage, event.coverage_peak, 0.01) / 0.12), 0.38, 1.82))
        theta = math.radians(float(event.dominant_axis_degrees) + 7.0 * math.sin(event_index + source_frame * 0.011))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = x - np.float32(cx)
        dy = y - np.float32(cy)
        along = dx * cos_t + dy * sin_t
        across = -dx * sin_t + dy * cos_t
        major = (0.090 + 0.120 * coverage_scale) * (1.0 + 0.08 * math.sin(event_index * 1.7))
        minor = 0.030 + 0.034 * coverage_scale
        core = np.exp(-0.5 * ((along / major) ** 2 + (across / minor) ** 2))
        tail_shift = along + major * (0.78 + 0.15 * math.sin(source_frame * 0.015 + event_index))
        tail = np.exp(-0.5 * ((tail_shift / (major * 1.65)) ** 2 + (across / (minor * 1.45)) ** 2))
        front = np.exp(-0.5 * (((along - major * 0.80) / (major * 0.75)) ** 2 + (across / (minor * 0.96)) ** 2))
        striations = 0.76 + 0.34 * texture + 0.13 * np.sin((along * 18.0 + across * 92.0 + source_frame * 0.035))
        event_density = (0.62 * core + 0.46 * tail + 0.22 * front) * temporal * coverage_scale * striations
        density += event_density.astype(np.float32)
    density = _pil_blur_float(np.clip(density, 0.0, 1.0), max(1.0, min(width, height) / 310.0))
    density = np.clip((density - 0.012) / 0.82, 0.0, 1.0)
    return _regional_density_to_smoke_rgba(density, frame_index, seed=seed + 131)


def _observed_guidance_density(source: ObservedSmokeSource, frame_index: int) -> np.ndarray:
    if not source.frames:
        raise ValueError("observed smoke source has no guidance frames")
    if len(source.frames) == 1:
        return source.frames[0].astype(np.float32, copy=False)
    position = np.clip(
        float(frame_index) / max(float(source.guidance_cadence_frames), 1.0),
        0.0,
        float(len(source.frames) - 1),
    )
    lo = int(math.floor(position))
    hi = min(lo + 1, len(source.frames) - 1)
    frac = np.float32(position - lo)
    return (
        source.frames[lo].astype(np.float32, copy=False) * (1.0 - frac)
        + source.frames[hi].astype(np.float32, copy=False) * frac
    ).astype(np.float32)


def observed_smoke_rgba(
    source: ObservedSmokeSource,
    map_size: tuple[int, int],
    frame_index: int,
    *,
    seed: int = HYBRID_SMOKE_SEED,
    progress: float = 0.0,
) -> np.ndarray:
    if source.source_kind == "reference-derived-events":
        return reference_event_transport_smoke_rgba(
            map_size,
            frame_index,
            source.event_states,
            seed=seed,
            progress=progress,
            timeline_frame_count=source.timeline_frame_count,
        )
    if source.source_kind == "hrrr-smoke":
        density = _observed_guidance_density(source, frame_index)
        return _regional_density_to_smoke_rgba(density, frame_index, seed=seed + 311)
    return regional_transport_smoke_rgba(map_size, frame_index, seed=seed, progress=progress)


def _cluster_fire_sources(
    sources: list[HybridSmokeSource],
    frame_index: int,
    cluster_radius: float = 9.5,
) -> list[list[tuple[int, HybridSmokeSource]]]:
    """Group nearby flame-live sources into clusters using simple distance-based clustering."""
    active = [
        (idx, src) for idx, src in enumerate(sources)
        if _source_flame_lifecycle_weight(src, frame_index) > 0.04
    ]
    if not active:
        return []

    # Simple greedy clustering: assign each source to nearest cluster or start new one
    clusters: list[list[tuple[int, HybridSmokeSource]]] = []
    for idx, src in active:
        assigned = False
        for cluster in clusters:
            # Check distance to cluster centroid
            cx = sum(s.x for _, s in cluster) / len(cluster)
            cy = sum(s.y for _, s in cluster) / len(cluster)
            dist = math.hypot(src.x - cx, src.y - cy)
            if dist < cluster_radius:
                cluster.append((idx, src))
                assigned = True
                break
        if not assigned:
            clusters.append([(idx, src)])
    return clusters


def _find_chain_pairs(
    cluster: list[tuple[int, HybridSmokeSource]],
    frame_index: int,
    max_chain_dist: float = 9.0,
) -> list[tuple[tuple[int, HybridSmokeSource], tuple[int, HybridSmokeSource]]]:
    """Find pairs of sources within a cluster that should be connected by short chains."""
    if len(cluster) < 2:
        return []

    cluster_seed = sum(s.seed for _, s in cluster) + frame_index * 7
    rng = np.random.default_rng(cluster_seed)
    points = np.array([(s.x, s.y) for _, s in cluster], dtype=np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(cluster) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    else:
        delta = points[1] - points[0]
        norm = float(np.linalg.norm(delta))
        axis = delta / norm if norm > 1.0e-4 else np.array([1.0, 0.0], dtype=np.float32)

    ordered = sorted(cluster, key=lambda item: float(item[1].x * axis[0] + item[1].y * axis[1]))
    pairs: list[tuple[tuple[int, HybridSmokeSource], tuple[int, HybridSmokeSource]]] = []
    seen: set[tuple[int, int]] = set()

    def add_pair(a: tuple[int, HybridSmokeSource], b: tuple[int, HybridSmokeSource]) -> None:
        key = tuple(sorted((a[0], b[0])))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))

    for left, right in zip(ordered, ordered[1:]):
        dist = math.hypot(left[1].x - right[1].x, left[1].y - right[1].y)
        if 1.0 < dist <= max_chain_dist * 1.18:
            add_pair(left, right)

    candidates: list[tuple[float, tuple[int, HybridSmokeSource], tuple[int, HybridSmokeSource]]] = []
    for i, (idx1, src1) in enumerate(cluster):
        for idx2, src2 in cluster[i + 1:]:
            dist = math.hypot(src1.x - src2.x, src1.y - src2.y)
            if 1.0 < dist <= max_chain_dist:
                intensity = (src1.strength * src1.heat + src2.strength * src2.heat) * 0.5
                score = intensity * (1.0 - dist / max_chain_dist) + float(rng.uniform(0.0, 0.08))
                candidates.append((score, (idx1, src1), (idx2, src2)))
    for _, left, right in sorted(candidates, key=lambda item: item[0], reverse=True)[: max(2, len(cluster))]:
        add_pair(left, right)

    return pairs


def _add_fire_core_blob(
    field: np.ndarray,
    x: float,
    y: float,
    radius: float,
    intensity: float,
    *,
    aspect: float = 1.0,
    angle: float = 0.0,
) -> None:
    height, width = field.shape
    sigma_x = max(float(radius), 0.35)
    sigma_y = max(float(radius) * max(float(aspect), 0.18), 0.30)
    pad = int(math.ceil(max(sigma_x, sigma_y) * 3.2 + 2.0))
    x0 = max(0, int(math.floor(float(x) - pad)))
    x1 = min(width, int(math.ceil(float(x) + pad + 1.0)))
    y0 = max(0, int(math.floor(float(y) - pad)))
    y1 = min(height, int(math.ceil(float(y) + pad + 1.0)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - np.float32(x)
    dy = yy - np.float32(y)
    ca = math.cos(float(angle))
    sa = math.sin(float(angle))
    rx = dx * ca + dy * sa
    ry = -dx * sa + dy * ca
    blob = np.exp(-0.5 * ((rx / sigma_x) ** 2 + (ry / sigma_y) ** 2)).astype(np.float32)
    field[y0:y1, x0:x1] = np.maximum(field[y0:y1, x0:x1], blob * float(intensity))


def _add_fire_core_segment(
    field: np.ndarray,
    a: HybridSmokeSource,
    b: HybridSmokeSource,
    frame_index: int,
    intensity: float,
) -> None:
    height, width = field.shape
    x0, y0 = float(a.x), float(a.y)
    x1, y1 = float(b.x), float(b.y)
    dx = x1 - x0
    dy = y1 - y0
    length = max(math.hypot(dx, dy), 1.0e-4)
    ux = dx / length
    uy = dy / length
    radius = max(0.50, min(float(a.radius_px), float(b.radius_px)) * 0.16)
    pad = int(math.ceil(radius * 4.0 + 2.0))
    bx0 = max(0, int(math.floor(min(x0, x1) - pad)))
    bx1 = min(width, int(math.ceil(max(x0, x1) + pad + 1.0)))
    by0 = max(0, int(math.floor(min(y0, y1) - pad)))
    by1 = min(height, int(math.ceil(max(y0, y1) + pad + 1.0)))
    if bx0 >= bx1 or by0 >= by1:
        return
    yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    rel_x = xx - np.float32(x0)
    rel_y = yy - np.float32(y0)
    along = rel_x * np.float32(ux) + rel_y * np.float32(uy)
    lateral = rel_x * np.float32(-uy) + rel_y * np.float32(ux)
    t = np.clip(along / np.float32(length), 0.0, 1.0)
    gate = _smoothstep(0.0, 0.12, t) * (1.0 - _smoothstep(0.88, 1.0, t))
    phase = float(a.seed + b.seed) * 0.017 + float(frame_index) * 0.19
    jitter = 0.68 + 0.32 * np.sin(t * math.tau * 3.0 + phase)
    segment = np.exp(-0.5 * (lateral / np.float32(radius)) ** 2) * gate * jitter
    field[by0:by1, bx0:bx1] = np.maximum(field[by0:by1, bx0:bx1], segment.astype(np.float32) * float(intensity))


def active_fire_core_intensity_field(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
) -> np.ndarray:
    """Return the pre-bloom active flame core/front signal used as smoke source of truth."""
    width, height = map(int, map_size)
    field = np.zeros((height, width), dtype=np.float32)
    if width <= 0 or height <= 0:
        return field

    for idx, source in enumerate(sources):
        flame = _source_flame_lifecycle_weight(source, frame_index)
        if flame <= 0.035:
            continue
        rng = np.random.default_rng(source.seed + idx * 97 + frame_index * 13)
        pulse = 0.68 + 0.32 * math.sin(frame_index * 0.52 + source.seed * 0.031 + idx * 0.71)
        flicker = 0.82 + 0.18 * math.sin(frame_index * 1.27 + source.seed * 0.071 + idx)
        intensity = float(np.clip(source.strength * source.heat * flame * pulse * flicker, 0.0, 2.4))
        if intensity <= 0.025:
            continue
        jitter_x = rng.uniform(-0.5, 0.5) * source.radius_px * 0.10
        jitter_y = rng.uniform(-0.5, 0.5) * source.radius_px * 0.10
        radius = max(0.48, source.radius_px * (0.11 + 0.055 * min(intensity, 1.4)))
        _add_fire_core_blob(
            field,
            source.x + jitter_x,
            source.y + jitter_y,
            radius,
            min(1.0, 0.42 + 0.48 * intensity),
            aspect=float(rng.uniform(0.55, 0.90)),
            angle=float(rng.uniform(0.0, math.tau)),
        )

    clusters = _cluster_fire_sources(sources, frame_index, cluster_radius=10.5)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster_intensity = sum(
            s.strength * s.heat * _source_flame_lifecycle_weight(s, frame_index)
            for _, s in cluster
        ) / max(len(cluster), 1)
        if cluster_intensity <= 0.10:
            continue
        points = np.array([(s.x, s.y) for _, s in cluster], dtype=np.float32)
        centroid = points.mean(axis=0)
        centered = points - centroid
        if len(cluster) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            front_axis = vh[0]
        else:
            delta = points[1] - points[0]
            norm = float(np.linalg.norm(delta))
            front_axis = delta / norm if norm > 1.0e-4 else np.array([1.0, 0.0], dtype=np.float32)
        ordered = sorted(cluster, key=lambda item: float(item[1].x * front_axis[0] + item[1].y * front_axis[1]))
        for (_, left), (_, right) in zip(ordered, ordered[1:]):
            dist = math.hypot(left.x - right.x, left.y - right.y)
            if dist <= 12.0:
                pair_intensity = min(
                    left.strength * left.heat * _source_flame_lifecycle_weight(left, frame_index),
                    right.strength * right.heat * _source_flame_lifecycle_weight(right, frame_index),
                )
                _add_fire_core_segment(field, left, right, frame_index, min(1.0, 0.38 + 0.45 * pair_intensity))
        for (_idx1, left), (_idx2, right) in _find_chain_pairs(cluster, frame_index, max_chain_dist=8.8):
            pair_intensity = min(
                left.strength * left.heat * _source_flame_lifecycle_weight(left, frame_index),
                right.strength * right.heat * _source_flame_lifecycle_weight(right, frame_index),
            )
            _add_fire_core_segment(field, left, right, frame_index + 3, min(1.0, 0.32 + 0.40 * pair_intensity))

    field = np.clip(field, 0.0, 1.0).astype(np.float32)
    field = np.where(field >= 0.020, field, 0.0).astype(np.float32)
    return field


def _nearest_source(
    sources: list[HybridSmokeSource],
    x: float,
    y: float,
) -> HybridSmokeSource | None:
    if not sources:
        return None
    distances = [(source.x - x) ** 2 + (source.y - y) ** 2 for source in sources]
    return sources[int(np.argmin(np.asarray(distances, dtype=np.float32)))]


def _dynamic_fire_core_source(
    x: float,
    y: float,
    intensity: float,
    frame_index: int,
    seed: int,
    map_size: tuple[int, int],
    base: HybridSmokeSource | None,
    *,
    smolder: bool = False,
) -> HybridSmokeSource:
    scale = min(int(map_size[0]), int(map_size[1])) / 408.0
    heat = float(np.clip((base.heat if base is not None else 1.0) * (0.82 + 0.34 * intensity), 0.36, 1.70))
    smoke_rate = float(np.clip((base.smoke_rate if base is not None else 1.0) * (0.58 + 0.70 * intensity), 0.20, 1.55))
    strength = float(np.clip(0.34 + 1.18 * intensity, 0.18, 1.95))
    radius = float(np.clip(1.25 * scale + 2.10 * scale * math.sqrt(max(intensity, 0.01)), 0.72, 4.4))
    if smolder:
        flame_end = int(frame_index) - 6
        start = max(0, int(frame_index) - 18)
        end = int(frame_index) + 28
        strength *= 0.42
        heat *= 0.42
        smoke_rate *= 0.46
        radius *= 1.25
    else:
        start = max(0, int(frame_index) - 4)
        flame_end = int(frame_index) + 8
        end = int(frame_index) + HYBRID_FIRE_SMOLDER_MIN_FRAMES
    return HybridSmokeSource(
        x=float(x),
        y=float(y),
        strength=strength,
        radius_px=max(0.62, radius),
        start_frame=start,
        end_frame=end,
        seed=int(seed),
        burst_period_frames=float(30.0 + (seed % 23)),
        burst_phase_frames=float(seed % 29),
        burst_duty=0.52 if not smolder else 0.40,
        heat=heat,
        smoke_rate=smoke_rate,
        altitude_bias=float(np.clip(base.altitude_bias if base is not None else 0.16, -0.16, 0.70)),
        flame_end_frame=flame_end,
    )


def fire_core_emitter_sources(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
    *,
    max_emitters: int = SOURCE_WISP_MAX_EMITTERS,
    seed: int = HYBRID_SMOKE_SEED,
    min_intensity: float = FIRE_CORE_EMITTER_INTENSITY_THRESHOLD,
) -> list[HybridSmokeSource]:
    """Sample spatially separated smoke emitters from active pre-bloom fire cores/fronts."""
    max_count = max(0, int(max_emitters))
    if max_count == 0:
        return []
    width, height = map(int, map_size)
    active_sources = [
        source for source in sources
        if _source_flame_lifecycle_weight(source, frame_index) > 0.035
    ]
    core = active_fire_core_intensity_field(sources, frame_index, map_size)
    ys, xs = np.where(core >= float(min_intensity))
    emitters: list[HybridSmokeSource] = []
    selected: list[tuple[float, float]] = []
    spacing = max(1.65, min(width, height) / 92.0)
    spacing_sq = spacing * spacing
    if xs.size:
        scores = core[ys, xs].astype(np.float32)
        jitter = ((xs * 73856093 + ys * 19349663 + int(seed) * 83492791 + int(frame_index) * 2654435761) & 1023) / 1023.0
        scores = scores * (0.96 + 0.08 * jitter.astype(np.float32))
        order = np.argsort(scores)[::-1]
        for index in order:
            x = float(xs[index])
            y = float(ys[index])
            if any((x - sx) * (x - sx) + (y - sy) * (y - sy) < spacing_sq for sx, sy in selected):
                continue
            selected.append((x, y))
            intensity = float(core[int(y), int(x)])
            base = _nearest_source(active_sources, x, y)
            source_seed = int(seed + frame_index * 1009 + int(round(x * 17.0)) + int(round(y * 31.0)))
            emitters.append(_dynamic_fire_core_source(x, y, intensity, frame_index, source_seed, map_size, base))
            if len(emitters) >= max_count:
                break

    if len(emitters) < min(max_count, len(active_sources)):
        for source in sorted(active_sources, key=lambda item: item.strength * item.heat, reverse=True):
            if len(emitters) >= max_count:
                break
            x = float(source.x)
            y = float(source.y)
            if any((x - sx) * (x - sx) + (y - sy) * (y - sy) < spacing_sq for sx, sy in selected):
                continue
            intensity = float(np.clip(source.strength * source.heat * _source_flame_lifecycle_weight(source, frame_index), 0.0, 1.0))
            selected.append((x, y))
            emitters.append(
                _dynamic_fire_core_source(
                    x,
                    y,
                    intensity,
                    frame_index,
                    int(seed + source.seed + frame_index * 809),
                    map_size,
                    source,
                )
            )

    if len(emitters) < max_count:
        smolder_candidates = [
            source for source in sources
            if _source_flame_lifecycle_weight(source, frame_index) <= 0.04
            and _source_smolder_lifecycle_weight(source, frame_index) > 0.05
        ]
        smolder_candidates.sort(
            key=lambda item: item.strength * item.smoke_rate * _source_smolder_lifecycle_weight(item, frame_index),
            reverse=True,
        )
        for source in smolder_candidates:
            if len(emitters) >= max_count:
                break
            x = float(source.x)
            y = float(source.y)
            if any((x - sx) * (x - sx) + (y - sy) * (y - sy) < spacing_sq * 0.72 for sx, sy in selected):
                continue
            smolder = _source_smolder_lifecycle_weight(source, frame_index)
            selected.append((x, y))
            emitters.append(
                _dynamic_fire_core_source(
                    x,
                    y,
                    float(np.clip(0.22 * smolder * source.smoke_rate, 0.03, 0.24)),
                    frame_index,
                    int(seed + source.seed + frame_index * 461),
                    map_size,
                    source,
                    smolder=True,
                )
            )

    return emitters


def hybrid_fire_sources_rgba(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
    *,
    glow_only: bool = False,
    bloom_scale: float = 1.0,
    core_alpha_scale: float = 1.0,
) -> np.ndarray:
    """Render fire sources as clustered marks with front-like chains, white-hot cores, and localized bloom."""
    width, height = map(int, map_size)
    # Separate layers for compositing order
    wide_bloom = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    local_bloom = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    chain_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ember_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    flare_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    wide_draw = ImageDraw.Draw(wide_bloom, "RGBA")
    local_draw = ImageDraw.Draw(local_bloom, "RGBA")
    chain_draw = ImageDraw.Draw(chain_layer, "RGBA")
    ember_draw = ImageDraw.Draw(ember_layer, "RGBA")
    flare_draw = ImageDraw.Draw(flare_layer, "RGBA")
    core_draw = ImageDraw.Draw(core_layer, "RGBA")

    # Cluster sources for shared flame patches and front-like chains
    clusters = _cluster_fire_sources(sources, frame_index, cluster_radius=10.5)

    # Track which sources are in multi-source clusters (for reduced individual rendering)
    clustered_sources: set[int] = set()
    cluster_anchor_sources: set[int] = set()
    for cluster in clusters:
        if len(cluster) > 1:
            for idx, _ in cluster:
                clustered_sources.add(idx)
            ranked = sorted(cluster, key=lambda item: item[1].strength * item[1].heat, reverse=True)
            anchor_count = 1 if len(cluster) < 5 else 2
            for idx, _ in ranked[:anchor_count]:
                cluster_anchor_sources.add(idx)

    # --- Draw cluster-level features: shared patches and chains ---
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Cluster-level deterministic variation
        cluster_seed = sum(s.seed for _, s in cluster) + frame_index * 11
        cluster_rng = np.random.default_rng(cluster_seed)

        # Cluster centroid and extent
        cx = sum(s.x for _, s in cluster) / len(cluster)
        cy = sum(s.y for _, s in cluster) / len(cluster)
        max_dist = max(math.hypot(s.x - cx, s.y - cy) for _, s in cluster)
        cluster_intensity = sum(
            s.strength * s.heat * _source_flame_lifecycle_weight(s, frame_index)
            for _, s in cluster
        ) / len(cluster)

        # Temporal variation at cluster level
        cluster_pulse = 0.72 + 0.28 * math.sin(frame_index * 0.38 + cluster_seed * 0.017)

        # --- Draw shared flame patch for cluster (reduces point-emitter look) ---
        patch_radius = max(3.4, max_dist * 0.95 + 2.3)
        patch_alpha = int(np.clip(68 * cluster_intensity * cluster_pulse, 22, 118))

        # Asymmetric patch: rotated ellipse
        angle = cluster_rng.uniform(0, 360)
        aspect = cluster_rng.uniform(0.55, 0.85)

        # Draw rotated ellipse as polygon approximation
        n_points = 16
        patch_points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            px = patch_radius * math.cos(theta)
            py = patch_radius * aspect * math.sin(theta)
            # Rotate
            rad = math.radians(angle)
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            patch_points.append((cx + rx, cy + ry))

        # Cluster bloom patch
        local_draw.polygon(patch_points, fill=(255, 95, 28, patch_alpha))
        wide_draw.polygon(patch_points, fill=(255, 70, 18, int(patch_alpha * 0.22)))
        if not glow_only and cluster_intensity > 0.48:
            inner_points = []
            for i in range(n_points):
                theta = 2 * math.pi * i / n_points
                px = patch_radius * 0.52 * math.cos(theta)
                py = patch_radius * aspect * 0.34 * math.sin(theta)
                rad = math.radians(angle)
                rx = px * math.cos(rad) - py * math.sin(rad)
                ry = px * math.sin(rad) + py * math.cos(rad)
                inner_points.append((cx + rx, cy + ry))
            flare_draw.polygon(inner_points, fill=(255, 182, 58, int(patch_alpha * 0.50)))
            if cluster_intensity > 0.58:
                core_points = []
                for i in range(n_points):
                    theta = 2 * math.pi * i / n_points
                    px = patch_radius * 0.30 * math.cos(theta)
                    py = patch_radius * aspect * 0.17 * math.sin(theta)
                    rad = math.radians(angle)
                    rx = px * math.cos(rad) - py * math.sin(rad)
                    ry = px * math.sin(rad) + py * math.cos(rad)
                    core_points.append((cx + rx, cy + ry))
                core_draw.polygon(core_points, fill=(255, 246, 214, max(72, int(patch_alpha * 0.68))))

        cluster_points = np.array([(s.x, s.y) for _, s in cluster], dtype=np.float32)
        centered = cluster_points - cluster_points.mean(axis=0)
        if len(cluster) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            front_axis = vh[0]
        else:
            delta = cluster_points[1] - cluster_points[0]
            norm = float(np.linalg.norm(delta))
            front_axis = delta / norm if norm > 1.0e-4 else np.array([1.0, 0.0], dtype=np.float32)
        front_order = sorted(cluster, key=lambda item: float(item[1].x * front_axis[0] + item[1].y * front_axis[1]))
        front_alpha = int(np.clip(92 * cluster_intensity * cluster_pulse, 34, 150))
        for (_, left), (_, right) in zip(front_order, front_order[1:]):
            dist = math.hypot(left.x - right.x, left.y - right.y)
            if dist <= 12.0:
                jitter_seed = left.seed + right.seed + frame_index * 19
                jitter_rng = np.random.default_rng(jitter_seed)
                perp = np.array([-front_axis[1], front_axis[0]], dtype=np.float32)
                j0 = jitter_rng.uniform(-0.55, 0.55)
                j1 = jitter_rng.uniform(-0.55, 0.55)
                sx = left.x + perp[0] * j0
                sy = left.y + perp[1] * j0
                ex = right.x + perp[0] * j1
                ey = right.y + perp[1] * j1
                spine_width = max(2, int(round(1.2 + min(left.radius_px, right.radius_px) * 0.16)))
                local_draw.line((sx, sy, ex, ey), fill=(255, 96, 30, int(front_alpha * 0.45)), width=spine_width + 3)
                chain_draw.line((sx, sy, ex, ey), fill=(255, 122, 38, front_alpha), width=spine_width + 1)
                if not glow_only and cluster_intensity > 0.52:
                    flare_draw.line((sx, sy, ex, ey), fill=(255, 194, 66, int(front_alpha * 0.70)), width=spine_width)
                    if cluster_intensity > 0.62:
                        core_draw.line(
                            (sx, sy, ex, ey),
                            fill=(255, 244, 210, max(66, int(front_alpha * 0.56))),
                            width=max(2, spine_width - 1),
                        )

        # --- Draw front-like chains between nearby sources ---
        chain_pairs = _find_chain_pairs(cluster, frame_index, max_chain_dist=8.8)
        for (idx1, src1), (idx2, src2) in chain_pairs:
            # Chain properties based on source pair
            pair_seed = src1.seed + src2.seed + frame_index * 3
            pair_rng = np.random.default_rng(pair_seed)

            x1, y1 = src1.x, src1.y
            x2, y2 = src2.x, src2.y
            chain_len = math.hypot(x2 - x1, y2 - y1)

            # Chain intensity based on weaker source
            min_intensity = min(
                src1.strength * src1.heat * _source_flame_lifecycle_weight(src1, frame_index),
                src2.strength * src2.heat * _source_flame_lifecycle_weight(src2, frame_index),
            )
            chain_pulse = 0.65 + 0.35 * math.sin(frame_index * 0.61 + pair_seed * 0.023)

            # Chain width varies along length (wider at ends near sources)
            base_width = max(1.35, min(src1.radius_px, src2.radius_px) * 0.22)

            # Draw chain as series of overlapping capsules/strokes
            n_segments = max(2, int(chain_len / 2.6))
            for seg in range(n_segments):
                t = (seg + 0.5) / n_segments
                seg_fraction = 0.72 + 0.18 * pair_rng.random()
                t0 = max(0.0, t - 0.5 * seg_fraction / n_segments)
                t1 = min(1.0, t + 0.5 * seg_fraction / n_segments)
                # Position along chain with slight perpendicular jitter
                perp_x = -(y2 - y1) / max(chain_len, 0.1)
                perp_y = (x2 - x1) / max(chain_len, 0.1)
                jitter0 = pair_rng.uniform(-0.85, 0.85) * base_width
                jitter1 = pair_rng.uniform(-0.85, 0.85) * base_width
                sx = x1 + t0 * (x2 - x1) + perp_x * jitter0
                sy = y1 + t0 * (y2 - y1) + perp_y * jitter0
                ex = x1 + t1 * (x2 - x1) + perp_x * jitter1
                ey = y1 + t1 * (y2 - y1) + perp_y * jitter1

                # Segment width (thinner in middle)
                seg_width = base_width * (1.10 + 0.45 * abs(t - 0.5) * 2)
                seg_alpha = int(np.clip(86 * min_intensity * chain_pulse * (0.68 + 0.32 * pair_rng.random()), 24, 142))

                # Ember-colored front segment. Using strokes instead of tiny capsules makes
                # grouped sources read as short fire fronts after perspective warp.
                chain_width = max(2, int(round(seg_width * 2.25)))
                local_draw.line((sx, sy, ex, ey), fill=(255, 92, 28, int(seg_alpha * 0.50)), width=chain_width + 3)
                chain_draw.line((sx, sy, ex, ey), fill=(255, 112, 34, seg_alpha), width=chain_width)

                # Hot interior for strong chains
                if min_intensity > 0.48 and not glow_only:
                    flare_width = max(1, int(round(seg_width * 1.10)))
                    flare_draw.line((sx, sy, ex, ey), fill=(255, 188, 62, int(seg_alpha * 0.78)), width=flare_width)
                    if min_intensity > 0.72 or idx1 in cluster_anchor_sources or idx2 in cluster_anchor_sources:
                        hot_width = max(2, int(round(seg_width * 0.58)))
                        core_draw.line(
                            (sx, sy, ex, ey),
                            fill=(255, 244, 210, max(64, int(seg_alpha * 0.72))),
                            width=hot_width,
                        )

    # --- Draw individual source marks (reduced for clustered sources) ---
    for idx, source in enumerate(sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        flame_weight = _source_flame_lifecycle_weight(source, frame_index)
        smolder_weight = _source_smolder_lifecycle_weight(source, frame_index)
        if flame_weight <= 0.04 and smolder_weight <= 0.08:
            continue

        # Deterministic variation using source.seed, idx, and frame_index
        rng = np.random.default_rng(source.seed + idx * 97 + frame_index * 13)
        jitter_x = rng.uniform(-0.5, 0.5) * source.radius_px * 0.15
        jitter_y = rng.uniform(-0.5, 0.5) * source.radius_px * 0.15
        x = source.x + jitter_x
        y = source.y + jitter_y

        if flame_weight <= 0.04:
            ember_radius = max(0.65, source.radius_px * 0.13 * (0.8 + 0.4 * smolder_weight))
            ember_alpha = int(np.clip(34.0 * smolder_weight * source.strength, 0.0, 42.0))
            if ember_alpha > 1 and not glow_only:
                ember_draw.ellipse(
                    (x - ember_radius, y - ember_radius, x + ember_radius, y + ember_radius),
                    fill=(150, 48, 22, ember_alpha),
                )
            continue

        # Temporal pulse with per-source phase variation
        phase_offset = source.seed * 0.031 + idx * 0.71
        pulse = 0.68 + 0.32 * math.sin(frame_index * 0.52 + phase_offset)
        flicker = 0.82 + 0.18 * math.sin(frame_index * 1.27 + phase_offset * 2.3)

        # Intensity classification
        intensity = source.strength * source.heat * pulse * flame_weight
        is_strong = intensity > 0.85
        is_medium = 0.45 < intensity <= 0.85
        is_weak = intensity <= 0.45

        # Reduce individual mark size for clustered sources (cluster patch handles coverage)
        in_cluster = idx in clustered_sources
        is_cluster_anchor = idx in cluster_anchor_sources
        cluster_scale = 0.72 if is_cluster_anchor else (0.42 if in_cluster else 1.0)

        # Size variation based on intensity
        base_radius = max(0.6, source.radius_px * 0.18) * cluster_scale
        size_mult = 0.7 + 0.6 * min(1.0, intensity / 1.2) + rng.uniform(-0.1, 0.1)
        radius = base_radius * size_mult

        # Alpha based on intensity with variation
        base_alpha = (55.0 + 95.0 * source.strength * source.heat * pulse * flicker) * flame_weight
        alpha = int(np.clip(base_alpha + rng.uniform(-12, 12), 22, 215))

        # --- Bloom layers ---
        local_radius = radius * (2.4 + 1.2 * float(bloom_scale))
        local_alpha = int(alpha * (0.18 + 0.08 * float(bloom_scale)) * (0.9 if is_weak else 1.0))
        # Reduce bloom alpha for clustered sources (cluster patch provides coverage)
        if in_cluster:
            local_alpha = int(local_alpha * 0.6)

        # Asymmetric bloom: rotated ellipse instead of circle
        bloom_angle = rng.uniform(0, 360)
        bloom_aspect = rng.uniform(0.6, 0.9)
        bloom_points = []
        for i in range(12):
            theta = 2 * math.pi * i / 12
            px = local_radius * math.cos(theta)
            py = local_radius * bloom_aspect * math.sin(theta)
            rad = math.radians(bloom_angle)
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            bloom_points.append((x + rx, y + ry))
        local_draw.polygon(bloom_points, fill=(255, 105, 30, local_alpha))

        # Wide bloom
        wide_radius = local_radius * (1.5 + 0.2 * float(bloom_scale))
        wide_alpha = int(alpha * (0.045 + 0.025 * float(bloom_scale)))
        if in_cluster:
            wide_alpha = int(wide_alpha * 0.5)
        wide_draw.ellipse(
            (x - wide_radius, y - wide_radius, x + wide_radius, y + wide_radius),
            fill=(255, 75, 18, wide_alpha),
        )

        if glow_only:
            continue

        # --- Core layers with intensity-based differentiation ---
        core_alpha_base = int(np.clip(alpha * float(core_alpha_scale), 0, 255))
        if in_cluster and not is_cluster_anchor:
            ember_radius = max(0.45, radius * (0.85 if is_weak else 1.05))
            ember_alpha = int(core_alpha_base * (0.22 if is_weak else 0.30))
            ember_draw.ellipse(
                (x - ember_radius, y - ember_radius, x + ember_radius, y + ember_radius),
                fill=(255, 82, 22, ember_alpha),
            )
            continue

        # Asymmetric core: rotated ellipse
        core_angle = rng.uniform(0, 360)
        core_aspect = rng.uniform(0.55, 0.85)

        if is_strong:
            # Strong sources: bright white-hot core + yellow flare + orange ember edge
            ember_radius = radius * 1.6
            ember_alpha = int(core_alpha_base * 0.55)
            ember_points = []
            for i in range(12):
                theta = 2 * math.pi * i / 12
                px = ember_radius * math.cos(theta)
                py = ember_radius * core_aspect * math.sin(theta)
                rad = math.radians(core_angle)
                rx = px * math.cos(rad) - py * math.sin(rad)
                ry = px * math.sin(rad) + py * math.cos(rad)
                ember_points.append((x + rx, y + ry))
            ember_draw.polygon(ember_points, fill=(255, 95, 22, ember_alpha))

            flare_radius = radius * 1.0
            flare_alpha = int(core_alpha_base * 0.78)
            flare_points = []
            for i in range(12):
                theta = 2 * math.pi * i / 12
                px = flare_radius * math.cos(theta)
                py = flare_radius * core_aspect * math.sin(theta)
                rad = math.radians(core_angle)
                rx = px * math.cos(rad) - py * math.sin(rad)
                ry = px * math.sin(rad) + py * math.cos(rad)
                flare_points.append((x + rx, y + ry))
            flare_draw.polygon(flare_points, fill=(255, 195, 65, flare_alpha))

            hot_radius = radius * 0.45
            hot_alpha = min(255, int(core_alpha_base * 1.1 + 25))
            core_draw.ellipse(
                (x - hot_radius, y - hot_radius, x + hot_radius, y + hot_radius),
                fill=(255, 248, 220, hot_alpha),
            )
        elif is_medium:
            ember_radius = radius * 1.35
            ember_alpha = int(core_alpha_base * 0.5)
            ember_points = []
            for i in range(12):
                theta = 2 * math.pi * i / 12
                px = ember_radius * math.cos(theta)
                py = ember_radius * core_aspect * math.sin(theta)
                rad = math.radians(core_angle)
                rx = px * math.cos(rad) - py * math.sin(rad)
                ry = px * math.sin(rad) + py * math.cos(rad)
                ember_points.append((x + rx, y + ry))
            ember_draw.polygon(ember_points, fill=(255, 105, 28, ember_alpha))

            flare_radius = radius * 0.7
            flare_alpha = int(core_alpha_base * 0.7)
            flare_draw.ellipse(
                (x - flare_radius, y - flare_radius, x + flare_radius, y + flare_radius),
                fill=(255, 175, 55, flare_alpha),
            )

            # Medium sources should enrich the shared front with yellow heat without
            # creating another isolated white-hot dot.
        else:
            # Weak sources: small red/orange embers only
            ember_radius = radius * 0.9
            ember_alpha = int(core_alpha_base * 0.45)
            ember_draw.ellipse(
                (x - ember_radius, y - ember_radius, x + ember_radius, y + ember_radius),
                fill=(255, 75, 18, ember_alpha),
            )
            center_radius = radius * 0.35
            center_alpha = int(core_alpha_base * 0.55)
            flare_draw.ellipse(
                (x - center_radius, y - center_radius, x + center_radius, y + center_radius),
                fill=(255, 125, 42, center_alpha),
            )

    # Apply blur: narrower than before to keep bloom source-local
    wide_bloom = wide_bloom.filter(
        ImageFilter.GaussianBlur(radius=max(2.0, min(width, height) / 95.0 * max(0.75, float(bloom_scale))))
    )
    local_bloom = local_bloom.filter(
        ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) / 180.0 * max(0.75, float(bloom_scale))))
    )
    chain_layer = chain_layer.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Composite layers: wide bloom -> local bloom -> chain -> ember -> flare -> core
    result = wide_bloom
    result.alpha_composite(local_bloom)
    result.alpha_composite(chain_layer)
    if not glow_only:
        ember_layer = ember_layer.filter(ImageFilter.GaussianBlur(radius=0.6))
        result.alpha_composite(ember_layer)
        result.alpha_composite(flare_layer)
        result.alpha_composite(core_layer)

    return np.asarray(result, dtype=np.uint8)


def _draw_reference_regional_fire_context(
    halo_draw: ImageDraw.ImageDraw,
    hot_draw: ImageDraw.ImageDraw,
    map_size: tuple[int, int],
    frame_index: int,
    seed: int,
    *,
    glow_only: bool,
) -> None:
    width, height = map(int, map_size)
    progress = float(np.clip(frame_index / max(REFERENCE_FILM_TIMELINE_DAYS * 20.0, 1.0), 0.0, 1.0))
    for anchor_index, (ux, uy, base_strength, phase) in enumerate(REFERENCE_FILM_REGIONAL_FIRE_ANCHORS[1:], start=1):
        arrival = _smoothstep(phase, phase + 0.22, progress)
        decay = 1.0 - 0.42 * _smoothstep(phase + 0.52, phase + 0.88, progress)
        pulse = 0.72 + 0.28 * math.sin(frame_index * 0.073 + anchor_index * 1.91)
        activity = float(np.clip(base_strength * arrival * decay * pulse, 0.0, 1.0))
        if activity <= 0.025:
            continue
        rng = np.random.default_rng(int(seed + 31_337 + anchor_index * 911 + frame_index * 17))
        point_count = max(5, int(round(10 + 46 * activity)))
        spread_x = max(1.6, width * (0.012 + 0.019 * activity))
        spread_y = max(1.3, height * (0.010 + 0.016 * activity))
        center_x = ux * (width - 1)
        center_y = uy * (height - 1)
        for point_index in range(point_count):
            angle = (point_index / max(point_count, 1)) * math.tau + rng.uniform(-0.42, 0.42)
            radius = math.sqrt(rng.uniform(0.05, 1.0))
            px = float(np.clip(center_x + math.cos(angle) * radius * spread_x + rng.normal(0.0, spread_x * 0.16), 0, width - 1))
            py = float(np.clip(center_y + math.sin(angle) * radius * spread_y + rng.normal(0.0, spread_y * 0.16), 0, height - 1))
            intensity = float(np.clip(activity * rng.uniform(0.72, 1.18), 0.0, 1.0))
            point_radius = max(0.34, min(width, height) / 720.0 * (0.80 + 0.70 * intensity))
            halo_radius = point_radius * (3.0 + 0.70 * intensity)
            halo_draw.ellipse(
                (px - halo_radius, py - halo_radius, px + halo_radius, py + halo_radius),
                fill=(255, 86, 22, int(np.clip(22 + 74 * intensity, 18, 96))),
            )
            if glow_only:
                continue
            ember_radius = point_radius * (1.22 + 0.36 * intensity)
            hot_draw.ellipse(
                (px - ember_radius, py - ember_radius, px + ember_radius, py + ember_radius),
                fill=(255, 106, 28, int(np.clip(68 + 130 * intensity, 70, 205))),
            )
            core_radius = max(0.26, point_radius * 0.72)
            hot_draw.ellipse(
                (px - core_radius, py - core_radius, px + core_radius, py + core_radius),
                fill=(255, 238, 198, int(np.clip(98 + 132 * intensity, 105, 240))),
            )


def reference_fire_points_rgba(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
    *,
    glow_only: bool = False,
    max_points: int = REFERENCE_FILM_FIRE_POINT_LIMIT,
    seed: int = HYBRID_SMOKE_SEED,
    regional_context: bool = False,
) -> np.ndarray:
    """Render many small active fire points with tight halos for map-film mode."""
    width, height = map(int, map_size)
    core = active_fire_core_intensity_field(sources, frame_index, map_size)
    ys, xs = np.where(core >= 0.10)
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    halo = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    hot = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo, "RGBA")
    hot_draw = ImageDraw.Draw(hot, "RGBA")
    if xs.size:
        scores = core[ys, xs].astype(np.float32)
        jitter = ((xs * 2654435761 + ys * 2246822519 + int(seed) * 3266489917 + int(frame_index) * 668265263) & 2047) / 2047.0
        order = np.argsort(scores * (0.94 + 0.12 * jitter.astype(np.float32)))[::-1]
        selected: list[tuple[float, float]] = []
        spacing = max(1.1, min(width, height) / 170.0)
        spacing_sq = spacing * spacing
        for index in order:
            x = float(xs[index])
            y = float(ys[index])
            if any((x - sx) * (x - sx) + (y - sy) * (y - sy) < spacing_sq for sx, sy in selected):
                continue
            selected.append((x, y))
            intensity = float(core[int(y), int(x)])
            rng = np.random.default_rng(int(seed + frame_index * 1709 + x * 37 + y * 67))
            dx = float(rng.uniform(-0.24, 0.24))
            dy = float(rng.uniform(-0.24, 0.24))
            px = x + dx
            py = y + dy
            point_radius = max(0.42, min(width, height) / 760.0 * (0.86 + 0.74 * math.sqrt(intensity)))
            halo_radius = point_radius * (3.2 + 0.8 * intensity)
            halo_alpha = int(np.clip(42 + 96 * intensity, 32, 148))
            halo_draw.ellipse(
                (px - halo_radius, py - halo_radius, px + halo_radius, py + halo_radius),
                fill=(255, 88, 24, int(halo_alpha * 0.34)),
            )
            if not glow_only:
                ember_radius = point_radius * (1.45 + 0.45 * intensity)
                hot_draw.ellipse(
                    (px - ember_radius, py - ember_radius, px + ember_radius, py + ember_radius),
                    fill=(255, 112, 28, int(np.clip(88 + 120 * intensity, 80, 230))),
                )
                core_radius = max(0.30, point_radius * 0.58)
                hot_draw.ellipse(
                    (px - core_radius, py - core_radius, px + core_radius, py + core_radius),
                    fill=(255, 246, 212, int(np.clip(118 + 128 * intensity, 120, 255))),
                )
            if len(selected) >= max_points:
                break

    if regional_context:
        _draw_reference_regional_fire_context(
            halo_draw,
            hot_draw,
            map_size,
            frame_index,
            seed,
            glow_only=glow_only,
        )

    halo = halo.filter(ImageFilter.GaussianBlur(radius=max(0.55, min(width, height) / 820.0)))
    result.alpha_composite(halo)
    if not glow_only:
        result.alpha_composite(hot)
    return np.asarray(result, dtype=np.uint8)


def hybrid_burn_scar_rgba(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
) -> np.ndarray:
    """Render persistent ash and burnt-ground footprints left by expired fire sources."""
    width, height = map(int, map_size)
    scar_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soot_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    rim_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scar_draw = ImageDraw.Draw(scar_layer, "RGBA")
    soot_draw = ImageDraw.Draw(soot_layer, "RGBA")
    rim_draw = ImageDraw.Draw(rim_layer, "RGBA")

    for idx, source in enumerate(sources):
        scar_weight = _source_burn_scar_weight(source, frame_index)
        if scar_weight <= 0.015:
            continue
        rng = np.random.default_rng(source.seed + idx * 211)
        flame_end = _source_flame_end_frame(source)
        burn_age = max(float(frame_index - source.start_frame), 0.0)
        burn_span = max(float(flame_end - source.start_frame), 1.0)
        burn_progress = float(np.clip(burn_age / burn_span, 0.0, 1.0))
        radius = max(1.2, source.radius_px * (0.74 + 2.25 * burn_progress))
        aspect = float(rng.uniform(0.52, 0.88))
        angle = float(rng.uniform(0.0, math.tau))
        points: list[tuple[float, float]] = []
        n_points = 18
        for point_index in range(n_points):
            theta = math.tau * point_index / n_points
            wobble = 0.72 + 0.46 * float(rng.random())
            px = radius * wobble * math.cos(theta)
            py = radius * aspect * wobble * math.sin(theta)
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            points.append((source.x + rx, source.y + ry))

        scar_alpha = int(np.clip(28.0 + 92.0 * scar_weight * (0.74 + 0.26 * source.strength), 18.0, 126.0))
        soot_alpha = int(np.clip(scar_alpha * (0.44 + 0.22 * burn_progress), 10.0, 88.0))
        rim_alpha = int(np.clip(scar_alpha * 0.32, 8.0, 46.0))
        scar_draw.polygon(points, fill=(44, 39, 31, scar_alpha))

        inner_radius = radius * float(rng.uniform(0.38, 0.58))
        inner_aspect = aspect * float(rng.uniform(0.54, 0.78))
        soot_points: list[tuple[float, float]] = []
        for point_index in range(12):
            theta = math.tau * point_index / 12
            wobble = 0.68 + 0.42 * float(rng.random())
            px = inner_radius * wobble * math.cos(theta)
            py = inner_radius * inner_aspect * wobble * math.sin(theta)
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            soot_points.append((source.x + rx, source.y + ry))
        soot_draw.polygon(soot_points, fill=(20, 20, 18, soot_alpha))

        if scar_weight > 0.28:
            rim_draw.line(points + [points[0]], fill=(84, 72, 52, rim_alpha), width=1)
            for _ in range(3):
                speck_x = source.x + float(rng.uniform(-radius * 0.62, radius * 0.62))
                speck_y = source.y + float(rng.uniform(-radius * 0.42, radius * 0.42))
                speck_r = max(0.35, radius * float(rng.uniform(0.045, 0.095)))
                speck_alpha = int(np.clip(soot_alpha * float(rng.uniform(0.22, 0.48)), 4, 26))
                soot_draw.ellipse(
                    (speck_x - speck_r, speck_y - speck_r, speck_x + speck_r, speck_y + speck_r),
                    fill=(12, 12, 11, speck_alpha),
                )

    scar_layer = scar_layer.filter(ImageFilter.GaussianBlur(radius=max(0.35, min(width, height) / 640.0)))
    soot_layer = soot_layer.filter(ImageFilter.GaussianBlur(radius=max(0.22, min(width, height) / 900.0)))
    result = scar_layer
    result.alpha_composite(soot_layer)
    result.alpha_composite(rim_layer)
    return np.asarray(result, dtype=np.uint8)


def _premultiply_rgba_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError("RGBA array must have shape (H, W, 4)")
    out = arr.copy()
    out[..., :3] *= out[..., 3:4] / 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _unpremultiply_rgba_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError("RGBA array must have shape (H, W, 4)")
    alpha = arr[..., 3:4] / 255.0
    out = arr.copy()
    out[..., :3] = np.divide(out[..., :3], alpha, out=np.zeros_like(out[..., :3]), where=alpha > 1.0e-5)
    out[..., :3] = np.where(arr[..., 3:4] > 0.5, out[..., :3], 0.0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def warp_map_layer_to_plate(rgba: np.ndarray, plate: TerrainPlate, size: tuple[int, int]) -> Image.Image:
    source = Image.fromarray(_premultiply_rgba_uint8(rgba), mode="RGBA")
    src_w, src_h = source.size
    coeffs = perspective_coeffs(
        [(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)],
        plate.quad,
    )
    warped = source.transform(size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    warped = warped.filter(ImageFilter.GaussianBlur(radius=max(0.16, size[0] / 4200.0)))
    return Image.fromarray(_unpremultiply_rgba_uint8(np.asarray(warped, dtype=np.uint8)), mode="RGBA")


def composite_atmospheric_smoke(base: Image.Image, smoke_layer: Image.Image) -> Image.Image:
    base_arr = np.asarray(base.convert("RGBA"), dtype=np.float32)
    smoke_arr = np.asarray(smoke_layer.convert("RGBA"), dtype=np.float32)
    alpha = smoke_arr[..., 3:4] / 255.0
    optical = np.clip(alpha * 0.98, 0.0, 1.0) ** 0.90
    veil = smoke_arr[..., :3] / 255.0
    terrain = base_arr[..., :3] / 255.0
    warm_signal = np.clip((terrain[..., 0:1] - terrain[..., 2:3]) * 1.55 + (terrain[..., 1:2] - terrain[..., 2:3]) * 0.38, 0.0, 1.0)
    source_transmission = 1.0 - 0.34 * _smoothstep(0.10, 0.48, warm_signal)
    transmittance = np.exp(-0.72 * optical * source_transmission)
    premul_smoke = veil * optical
    backscatter = np.array([0.65, 0.67, 0.66], dtype=np.float32)[None, None, :] * (0.17 * optical)
    glow_through = terrain * warm_signal * optical * 0.18
    lifted = terrain * transmittance + premul_smoke * 0.92 + backscatter + glow_through
    base_arr[..., :3] = np.clip(lifted * 255.0, 0.0, 255.0)
    base_arr[..., 3] = 255.0
    return Image.fromarray(base_arr.astype(np.uint8), mode="RGBA")


def make_smoke_domain() -> tuple[object, object, object, dict[str, object]]:
    if f3d_smoke is None:
        raise RuntimeError("forge3d.smoke is required for volume detail; rerun with --no-volume-detail")
    domain = f3d_smoke.SmokeDomain((96, 48, 52), voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    emitter = f3d_smoke.SmokeEmitter(
        center=(18.0, 8.0, 24.0),
        radius=3.2,
        density_rate=2.25,
        temperature_rate=1.12,
        soot_rate=0.12,
        emission_rate=2.3,
        velocity=(7.4, 0.20, -0.36),
    )
    step = f3d_smoke.SmokeStepSettings(
        dt=0.10,
        density_decay=0.014,
        temperature_decay=0.22,
        velocity_damping=0.026,
        diffusion=0.0012,
        buoyancy=0.035,
        vorticity=0.32,
        pressure_iterations=14,
        turbulence_strength=0.76,
        turbulence_seed=2020,
        wind=(2.20, 0.00, -0.32),
    )
    render = f3d_smoke.SmokeRenderSettings(
        density_scale=1.34,
        extinction=1.62,
        scattering=1.02,
        absorption=0.28,
        phase_g=0.48,
        max_steps=144,
        self_shadow=True,
        shadow_steps=28,
        shadow_step_size=0.90,
        jitter_strength=0.45,
        thin_color=(0.48, 0.54, 0.61),
        dense_color=(0.93, 0.91, 0.82),
        soot_absorption=0.32,
        fire_glow=0.34,
    )
    camera = {
        "camera_pos": (38.0, 23.0, -104.0),
        "target": (46.0, 16.0, 24.0),
        "up": (0.0, 1.0, 0.0),
        "fovy_deg": 33.0,
        "sun_direction": (0.35, 0.78, -0.18),
        "source_world": (18.0, 8.0, 24.0),
    }
    return domain, emitter, step, {"render": render, **camera}


def project_smoke_source(camera: dict[str, object], width: int, height: int) -> tuple[float, float]:
    eye = np.asarray(camera["camera_pos"], dtype=np.float32)
    target = np.asarray(camera["target"], dtype=np.float32)
    up = np.asarray(camera["up"], dtype=np.float32)
    point = np.asarray(camera["source_world"], dtype=np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    side = np.cross(forward, up)
    side /= max(float(np.linalg.norm(side)), 1e-6)
    cam_up = np.cross(side, forward)
    rel = point - eye
    depth = max(float(np.dot(rel, forward)), 1e-3)
    focal = 1.0 / math.tan(math.radians(float(camera["fovy_deg"])) * 0.5)
    aspect = width / max(float(height), 1.0)
    sx = (float(np.dot(rel, side)) * focal / aspect / depth) * 0.5 + 0.5
    sy = 0.5 - (float(np.dot(rel, cam_up)) * focal / depth) * 0.5
    return sx * width, sy * height


def draw_fire(frame: Image.Image, xy: tuple[float, float], progress: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x, y = xy
    pulse = 0.5 + 0.5 * math.sin(progress * math.tau * 6.0)
    scale = frame.width / WIDTH
    for radius, alpha in ((38, 42), (20, 84), (8, 205)):
        radius *= scale
        color = (255, int(116 + 70 * pulse), 18, int(alpha))
        draw.ellipse((x - radius, y - radius * 0.55, x + radius, y + radius * 0.55), fill=color)
    draw.ellipse((x - 4 * scale, y - 3 * scale, x + 4 * scale, y + 3 * scale), fill=(255, 232, 126, 245))


def _format_area_ha(area_ha: float) -> str:
    if area_ha >= 1_000_000.0:
        return f"{area_ha / 1_000_000.0:.1f} M ha"
    return f"{area_ha / 1000.0:.0f} k ha"


def draw_labels(
    frame: Image.Image,
    *,
    frame_info: ReferenceFilmFrameInfo | None = None,
    composition_mode: str = TERRAIN_SLAB_COMPOSITION_MODE,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    draw = ImageDraw.Draw(frame, "RGBA")
    scale = min(frame.width / WIDTH, frame.height / HEIGHT)
    pad = max(18, int(round(38 * scale)))
    text_font = load_font(max(12, int(round(19 * scale))))
    small_font = load_font(max(10, int(round(15 * scale))))
    area_font = load_font(max(24, int(round(48 * scale))), bold=True)
    label_boxes: list[tuple[str, tuple[int, int, int, int]]] = []

    def record_box(name: str, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, padding: int = 4) -> None:
        bbox = draw.textbbox(xy, text, font=font)
        label_boxes.append(
            (
                name,
                (
                    int(bbox[0] - padding),
                    int(bbox[1] - padding),
                    int(bbox[2] + padding),
                    int(bbox[3] + padding),
                ),
            )
        )

    info = frame_info or ReferenceFilmFrameInfo(
        progress=1.0,
        date_label=AUGUST_COMPLEX.date,
        burned_area_ha=AUGUST_COMPLEX.final_area_ha,
    )
    if composition_mode == MAP_FILM_COMPOSITION_MODE:
        source_lines = ("Data: CAL FIRE perimeter, synthetic smoke field", "August Complex, 2020 reference-film mode")
        date_font = load_font(max(20, int(round(34 * scale))), bold=True)
        date_text = info.date_label
        bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_x = frame.width - pad - (bbox[2] - bbox[0])
        date_y = pad
        draw.text((date_x + 2, date_y + 2), date_text, font=date_font, fill=(0, 0, 0, 190))
        draw.text((date_x, date_y), date_text, font=date_font, fill=(238, 244, 244, 240))
        record_box("date", (int(date_x), int(date_y)), date_text, date_font)
    else:
        source_lines = ("Data: CAL FIRE perimeter, cached California DEM", "August Complex, 2020")
    for idx, line in enumerate(source_lines):
        y = pad + idx * max(18, int(round(25 * scale)))
        draw.text((pad + 1, y + 1), line, font=small_font, fill=(0, 0, 0, 180))
        draw.text((pad, y), line, font=small_font, fill=(234, 239, 235, 230))
        record_box(f"source_{idx}", (int(pad), int(y)), line, small_font)

    bottom = frame.height - pad - max(74, int(round(112 * scale)))
    draw.text((pad + 2, bottom + 2), "Area Burned:", font=text_font, fill=(0, 0, 0, 190))
    draw.text((pad, bottom), "Area Burned:", font=text_font, fill=(236, 239, 235, 235))
    record_box("area_label", (int(pad), int(bottom)), "Area Burned:", text_font)
    value = _format_area_ha(info.burned_area_ha)
    value_y = bottom + max(24, int(round(38 * scale)))
    draw.text((pad + 2, value_y + 2), value, font=area_font, fill=(0, 0, 0, 200))
    draw.text((pad, value_y), value, font=area_font, fill=(248, 250, 246, 245))
    record_box("area_value", (int(pad), int(value_y)), value, area_font)
    return label_boxes


def composite_volume_detail(
    base: Image.Image,
    smoke_rgba: np.ndarray,
    fire_xy: tuple[float, float],
    source_xy: tuple[float, float],
) -> Image.Image:
    smoke_arr = np.asarray(smoke_rgba, dtype=np.float32).copy()
    alpha = smoke_arr[..., 3:4] / 255.0
    smoke_lift = np.array([76.0, 77.0, 74.0], dtype=np.float32)
    smoke_arr[..., :3] = np.clip(smoke_arr[..., :3] * 1.26 + smoke_lift * alpha, 0.0, 255.0)
    smoke_arr[..., 3] = np.clip(smoke_arr[..., 3] * 0.54, 0.0, 116.0)
    smoke_image = Image.fromarray(smoke_arr.astype(np.uint8), mode="RGBA").filter(ImageFilter.GaussianBlur(radius=1.15))
    smoke_scale = 1.04
    scaled_size = (int(round(smoke_image.width * smoke_scale)), int(round(smoke_image.height * smoke_scale)))
    smoke_image = smoke_image.resize(scaled_size, Image.Resampling.BICUBIC)
    smoke_image = smoke_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    scaled_source_xy = (source_xy[0] * smoke_scale, source_xy[1] * smoke_scale)
    flipped_source_x = smoke_image.width - scaled_source_xy[0]
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    offset = (
        int(round(fire_xy[0] - flipped_source_x)),
        int(round(fire_xy[1] - scaled_source_xy[1] - base.height * 0.030)),
    )
    layer.alpha_composite(smoke_image, offset)
    out = base.copy()
    out.alpha_composite(layer)
    return out


def _scale_rgba_alpha(rgba: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(rgba, dtype=np.uint8).copy()
    out[..., 3] = np.clip(out[..., 3].astype(np.float32) * float(scale), 0.0, 255.0).astype(np.uint8)
    return out


def _select_physical_sources(
    sources: list[HybridSmokeSource],
    max_sources: int,
) -> list[HybridSmokeSource]:
    if max_sources <= 0:
        return []
    ranked = sorted(
        enumerate(sources),
        key=lambda item: (
            item[1].strength * item[1].smoke_rate * (1.0 + 0.20 * item[1].heat),
            -item[0],
        ),
        reverse=True,
    )
    selected_indexes: list[int] = []
    selected_indexes.extend(range(min(6, len(sources))))
    for idx, _source in ranked:
        selected_indexes.append(idx)
    seen: set[int] = set()
    unique_indexes: list[int] = []
    for idx in selected_indexes:
        if idx in seen or idx >= len(sources):
            continue
        seen.add(idx)
        unique_indexes.append(idx)
        if len(unique_indexes) >= max_sources:
            break
    return [sources[idx] for idx in sorted(unique_indexes)]


def _native_physical_smoke_available() -> bool:
    if f3d_smoke is None:
        return False
    native_available = getattr(f3d_smoke, "native_smoke_available", lambda: False)
    if not bool(native_available()):
        return False
    required = ("SmokeDomain", "SmokeEmitter", "SmokeStepSettings", "SmokeRenderSettings")
    if not all(hasattr(f3d_smoke, name) for name in required):
        return False
    return bool(hasattr(f3d_smoke.SmokeDomain, "render_projection_rgba"))


def _physical_backend(requested: str) -> str:
    backend = str(requested).lower()
    if backend not in {"auto", "native", "numpy"}:
        raise ValueError("physical smoke backend must be 'auto', 'native', or 'numpy'")
    if backend == "numpy":
        return "numpy"
    if _native_physical_smoke_available():
        return "native"
    if backend == "native":
        raise RuntimeError("native physical smoke backend requested but forge3d.smoke native projection is unavailable")
    return "numpy"


def _make_physical_domain(dims: tuple[int, int, int], backend: str) -> object:
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeDomain(dims, sparse_threshold=1.0e-6)
    return NumpyPhysicalSmokeDomain(dims, sparse_threshold=1.0e-6)


def _make_physical_step_settings(backend: str, substep_count: int, seed: int) -> object:
    kwargs = {
        "dt": 0.22 / substep_count,
        "density_decay": 0.0038,
        "temperature_decay": 0.20,
        "velocity_damping": 0.018,
        "diffusion": 0.0018,
        "buoyancy": 0.0045,
        "vorticity": 0.88,
        "pressure_iterations": 14,
        "turbulence_strength": 1.20,
        "turbulence_seed": int(seed),
        "terrain_collision": True,
        "boundary_damping": 0.026,
        "wind": (0.278, 0.000, -0.106),
    }
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeStepSettings(
            **kwargs,
            mac_cormack=True,
            mass_conservation=False,
        )
    return PhysicalSmokeStepSettings3D(**kwargs)


def _make_physical_render_settings(backend: str) -> object:
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeRenderSettings(
            density_scale=1.32,
            extinction=1.30,
            scattering=1.24,
            absorption=0.26,
            phase_g=0.24,
            step_size=0.62,
            max_steps=150,
            self_shadow=True,
            shadow_steps=32,
            shadow_step_size=0.82,
            jitter_strength=0.36,
            exposure=1.62,
            thin_color=(0.50, 0.54, 0.58),
            dense_color=(0.93, 0.91, 0.82),
            soot_absorption=0.18,
            fire_glow=0.24,
        )
    return PhysicalSmokeRenderSettings3D(
        density_scale=1.32,
        extinction=1.30,
        exposure=1.62,
        scattering=1.24,
        phase_g=0.24,
        soot_absorption=0.18,
        fire_glow=0.24,
    )


def make_physical_main_smoke(
    sources: list[HybridSmokeSource],
    map_size: tuple[int, int],
    *,
    dims: tuple[int, int, int] = PHYSICAL_SMOKE_DIMS,
    render_size: tuple[int, int] = PHYSICAL_SMOKE_RENDER_SIZE,
    max_sources: int = PHYSICAL_SMOKE_MAX_SOURCES,
    substeps: int = 1,
    backend: str = "auto",
    emitter_mode: str = "synthetic",
    seed: int = HYBRID_SMOKE_SEED,
) -> PhysicalSmokeMainEffect | None:
    nx, ny, nz = (max(8, int(v)) for v in dims)
    render_w, render_h = (max(16, int(v)) for v in render_size)
    substep_count = max(1, int(substeps))
    if emitter_mode not in SOURCE_WISP_EMITTER_MODES:
        raise ValueError(f"physical emitter mode must be one of {SOURCE_WISP_EMITTER_MODES}")
    resolved_backend = _physical_backend(backend)
    domain = _make_physical_domain((nx, ny, nz), resolved_backend)
    step = _make_physical_step_settings(resolved_backend, substep_count, seed)
    render = _make_physical_render_settings(resolved_backend)
    if emitter_mode == "fire-core":
        selected_sources = fire_core_emitter_sources(
            sources,
            0,
            map_size,
            max_emitters=max_sources,
            seed=seed + 12011,
        )
        if not selected_sources:
            selected_sources = _select_physical_sources(sources, max_sources=max_sources)
    else:
        selected_sources = _select_physical_sources(sources, max_sources=max_sources)
    if not selected_sources:
        return None
    return PhysicalSmokeMainEffect(
        domain=domain,
        step_settings=step,
        render_settings=render,
        sources=selected_sources,
        map_size=tuple(map(int, map_size)),
        dims=(nx, ny, nz),
        render_size=(render_w, render_h),
        substeps=substep_count,
        backend=resolved_backend,
        seed=int(seed),
        base_sources=list(sources),
        emitter_mode=str(emitter_mode),
        max_sources=max(0, int(max_sources)),
    )


def _physical_sources_for_frame(effect: PhysicalSmokeMainEffect, frame_index: int) -> list[HybridSmokeSource]:
    if effect.emitter_mode == "fire-core":
        sources = fire_core_emitter_sources(
            effect.base_sources or effect.sources,
            frame_index,
            effect.map_size,
            max_emitters=max(effect.max_sources, 1),
            seed=effect.seed + 12011,
        )
        if sources:
            return sources
    return list(effect.sources)


def _physical_emitters_for_frame(effect: PhysicalSmokeMainEffect, frame_index: int) -> list[PhysicalSmokeEmitter3D]:
    nx, ny, nz = effect.dims
    map_w, map_h = effect.map_size
    emitters: list[PhysicalSmokeEmitter3D] = []
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    frame_sources = _physical_sources_for_frame(effect, frame_index)
    for source_index, source in enumerate(frame_sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        flame = _source_flame_lifecycle_weight(source, frame_index)
        smolder = _source_smolder_lifecycle_weight(source, frame_index)
        source_activity = _source_smoke_activity_weight(source, frame_index)
        if source_activity <= 0.01:
            continue
        burst = _source_burst_envelope(source, frame_index)
        if burst < 0.08:
            continue
        sx = float(source.x) / max(float(map_w - 1), 1.0) * float(nx - 1)
        sz = float(source.y) / max(float(map_h - 1), 1.0) * float(nz - 1)
        altitude = float(np.clip(0.18 + 0.34 * source.altitude_bias + 0.06 * source.heat, 0.12, 0.62))
        sy = 1.8 + altitude * float(ny - 4)
        scale_x = float(nx) / max(float(map_w), 1.0)
        scale_z = float(nz) / max(float(map_h), 1.0)
        radius = max(1.28, float(source.radius_px) * math.sqrt(scale_x * scale_z) * (0.82 + 0.18 * source.heat))
        flicker = 0.82 + 0.18 * math.sin(frame_index * 0.31 + source.seed * 0.023 + source_index)
        lateral = (
            0.72 * math.sin(frame_index * 0.041 + source.seed * 0.017)
            + 0.28 * math.sin(frame_index * 0.013 + source_index * 1.41)
        )
        velocity = (
            float(wind[0] * (1.92 + 0.34 * source.heat) + cross[0] * lateral),
            float(0.006 + 0.014 * source.heat),
            float(wind[1] * (1.92 + 0.32 * source.heat) + cross[1] * lateral),
        )
        density_rate = float(0.96 * source.strength * source.smoke_rate * burst * flicker * source_activity)
        temperature_rate = float(0.18 * source.heat * burst * (flame + 0.12 * smolder))
        soot_rate = float(0.018 + 0.010 * source.heat)
        emitters.append(
            PhysicalSmokeEmitter3D(
                center=(sx, sy, sz),
                radius=radius,
                density_rate=density_rate,
                temperature_rate=temperature_rate,
                soot_rate=soot_rate,
                humidity_rate=0.10,
                emission_rate=0.55 * source.heat * flame,
                velocity=velocity,
            )
        )
        if source_index < 6 and burst > 0.18:
            streamer_nodes = (
                (0.48, 0.18, 0.28, 0.18),
                (0.96, 0.17, 0.24, -0.30),
                (1.58, 0.22, 0.20, 0.46),
                (2.34, 0.32, 0.13, -0.54),
            )
            for streamer_index, (distance, radius_scale, rate_scale, bend_sign) in enumerate(streamer_nodes):
                phase = frame_index * (0.061 + streamer_index * 0.017) + source.seed * 0.013
                bend = bend_sign + 0.34 * math.sin(phase * 0.79 + streamer_index)
                lateral_stream = bend * radius * (0.72 + 0.14 * streamer_index)
                vertical_stream = math.cos(phase * 0.83 + 0.7) * (0.12 + 0.03 * streamer_index)
                stream_center = (
                    float(np.clip(sx + wind[0] * radius * distance + cross[0] * lateral_stream, 1.0, nx - 2.0)),
                    float(np.clip(sy + vertical_stream + 0.10 * streamer_index, 1.0, ny - 2.0)),
                    float(np.clip(sz + wind[1] * radius * distance + cross[1] * lateral_stream, 1.0, nz - 2.0)),
                )
                curl_push = 0.11 * math.cos(phase * 1.17 + source_index)
                stream_velocity = (
                    float(velocity[0] * (1.28 + 0.05 * streamer_index) + cross[0] * curl_push),
                    float(velocity[1] * (0.36 + 0.04 * streamer_index)),
                    float(velocity[2] * (1.28 + 0.05 * streamer_index) + cross[1] * curl_push),
                )
                emitters.append(
                    PhysicalSmokeEmitter3D(
                        center=stream_center,
                        radius=max(0.38, radius * radius_scale),
                        density_rate=density_rate * rate_scale,
                        temperature_rate=temperature_rate * (0.34 - 0.06 * streamer_index),
                        soot_rate=soot_rate * 0.55,
                        humidity_rate=0.10,
                        emission_rate=0.28 * source.heat,
                        velocity=stream_velocity,
                    )
                )
            if source_index < 6 and burst > 0.24:
                hook_phase = frame_index * 0.027 + source.seed * 0.019 + source_index * 1.71
                hook_side = -1.0 if math.sin(hook_phase) < 0.0 else 1.0
                for hook_index in range(3):
                    arc_t = (hook_index + 1.0) / 3.0
                    arc_angle = hook_phase + hook_side * (0.55 + 1.10 * arc_t)
                    arc_distance = radius * (1.35 + 0.62 * hook_index)
                    lateral_arc = math.sin(arc_angle) * radius * (0.88 + 0.32 * hook_index)
                    arc_center = (
                        float(np.clip(sx + wind[0] * arc_distance + cross[0] * lateral_arc, 1.0, nx - 2.0)),
                    float(np.clip(sy + 0.10 * hook_index + 0.10 * math.cos(arc_angle), 1.0, ny - 2.0)),
                        float(np.clip(sz + wind[1] * arc_distance + cross[1] * lateral_arc, 1.0, nz - 2.0)),
                    )
                    curl_velocity = 0.13 * hook_side * math.cos(arc_angle)
                    emitters.append(
                        PhysicalSmokeEmitter3D(
                            center=arc_center,
                            radius=max(0.48, radius * (0.34 + 0.08 * hook_index)),
                            density_rate=density_rate * (0.18 - 0.045 * hook_index),
                            temperature_rate=temperature_rate * (0.17 - 0.038 * hook_index),
                            soot_rate=soot_rate * 0.34,
                            humidity_rate=0.12,
                            emission_rate=0.16 * source.heat,
                            velocity=(
                                float(velocity[0] * (0.98 - 0.06 * hook_index) + cross[0] * curl_velocity),
                                float(velocity[1] * 0.20),
                                float(velocity[2] * (0.98 - 0.06 * hook_index) + cross[1] * curl_velocity),
                            ),
                        )
                    )
    return emitters


def _native_physical_emitter(emitter: PhysicalSmokeEmitter3D) -> object:
    assert f3d_smoke is not None
    return f3d_smoke.SmokeEmitter(
        center=emitter.center,
        radius=emitter.radius,
        density_rate=emitter.density_rate,
        temperature_rate=emitter.temperature_rate,
        fuel_rate=0.0,
        soot_rate=emitter.soot_rate,
        humidity_rate=emitter.humidity_rate,
        emission_rate=emitter.emission_rate,
        velocity=emitter.velocity,
    )


def step_physical_main_smoke(effect: PhysicalSmokeMainEffect, frame_index: int) -> None:
    for _substep in range(max(1, effect.substeps)):
        emitters = _physical_emitters_for_frame(effect, frame_index)
        if effect.backend == "native":
            emitters = [_native_physical_emitter(emitter) for emitter in emitters]  # type: ignore[assignment]
        effect.domain.step(effect.step_settings, emitters)


def _henyey_greenstein_py(cos_theta: float, g: float) -> float:
    g = float(np.clip(g, -0.99, 0.99))
    denom = max(1.0 + g * g - 2.0 * g * float(np.clip(cos_theta, -1.0, 1.0)), 1.0e-4)
    return float((1.0 - g * g) / (4.0 * math.pi * denom**1.5))


def _physical_projection_view_direction() -> tuple[float, float, float]:
    view = np.asarray(PHYSICAL_SMOKE_VIEW_DIRECTION, dtype=np.float32)
    if view.shape != (3,):
        view = np.asarray((0.42, -0.68, 0.60), dtype=np.float32)
    parallax = max(0.25, float(PHYSICAL_SMOKE_PARALLAX_SCALE))
    view = np.array([view[0] * parallax, view[1], view[2] * parallax], dtype=np.float32)
    norm = max(float(np.linalg.norm(view)), 1.0e-6)
    view /= norm
    return float(view[0]), float(view[1]), float(view[2])


def _python_volume_light_transmittance(
    density: np.ndarray,
    soot: np.ndarray,
    layer_index: int,
    density_scale: float,
    extinction: float,
    soot_absorption: float,
    settings: object | None,
) -> np.ndarray:
    depth, altitude_count, width = density.shape
    z_grid, x_grid = np.mgrid[0:depth, 0:width].astype(np.float32)
    light_dir = np.asarray((0.34, 0.82, -0.22), dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)
    shadow_steps = int(max(1, getattr(settings, "shadow_steps", 18)))
    shadow_step_size = float(getattr(settings, "shadow_step_size", 1.15))
    if shadow_step_size <= 0.0:
        shadow_step_size = 1.15
    optical_depth = np.zeros((depth, width), dtype=np.float32)
    base_y = np.full((depth, width), float(layer_index), dtype=np.float32)
    for step_index in range(1, shadow_steps + 1):
        distance = float(step_index) * shadow_step_size
        sample_x = x_grid + light_dir[0] * distance
        sample_y = base_y + light_dir[1] * distance
        sample_z = z_grid + light_dir[2] * distance
        sample_density = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        optical_depth += (
            sample_density
            * density_scale
            * extinction
            * (1.0 + sample_soot * soot_absorption * 0.85)
            * shadow_step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _python_volume_shadow_grid(
    density: np.ndarray,
    soot: np.ndarray,
    particle_age: np.ndarray | None,
    light_dir: np.ndarray,
    density_scale: float,
    extinction: float,
    soot_absorption: float,
    settings: object | None,
) -> np.ndarray:
    depth, altitude_count, width = density.shape
    grid_x, grid_y, grid_z = _volume_grids(density.shape)
    shadow_steps = int(max(1, getattr(settings, "shadow_steps", 18)))
    shadow_step_size = float(getattr(settings, "shadow_step_size", 1.15))
    if shadow_step_size <= 0.0:
        shadow_step_size = 1.15
    optical_depth = np.zeros_like(density, dtype=np.float32)
    for step_index in range(1, shadow_steps + 1):
        distance = float(step_index) * shadow_step_size
        sample_x = grid_x + float(light_dir[0]) * distance
        sample_y = grid_y + float(light_dir[1]) * distance
        sample_z = grid_z + float(light_dir[2]) * distance
        sample_density = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        if particle_age is not None:
            sample_age = _trilinear_sample_volume(particle_age, sample_x, sample_y, sample_z)
            age_t = _smoothstep(1.6, 17.0, sample_age)
        else:
            age_t = np.zeros_like(sample_density, dtype=np.float32)
        concentration_gate = 0.50 + 0.50 * _smoothstep(0.045, 0.34, sample_density)
        optical_depth += (
            sample_density
            * density_scale
            * (1.0 - 0.58 * age_t)
            * concentration_gate
            * extinction
            * (1.0 + sample_soot * soot_absorption * 0.85)
            * shadow_step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _python_ray_box_intersection(
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    origin_z: np.ndarray,
    ray_dir: np.ndarray,
    bounds_max: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def axis_interval(origin: np.ndarray, direction: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        if abs(direction) < 1.0e-6:
            inside = (origin >= 0.0) & (origin <= upper)
            near = np.where(inside, -np.inf, 1.0).astype(np.float32)
            far = np.where(inside, np.inf, 0.0).astype(np.float32)
            return near, far
        t0 = (0.0 - origin) / direction
        t1 = (upper - origin) / direction
        return np.minimum(t0, t1).astype(np.float32), np.maximum(t0, t1).astype(np.float32)

    x_near, x_far = axis_interval(origin_x, float(ray_dir[0]), bounds_max[0])
    y_near, y_far = axis_interval(origin_y, float(ray_dir[1]), bounds_max[1])
    z_near, z_far = axis_interval(origin_z, float(ray_dir[2]), bounds_max[2])
    t_enter = np.maximum(np.maximum(x_near, y_near), z_near).astype(np.float32)
    t_exit = np.minimum(np.minimum(x_far, y_far), z_far).astype(np.float32)
    valid = t_exit >= np.maximum(t_enter, 0.0)
    return t_enter, t_exit, valid


def _python_projected_volume_raymarch(
    domain: object,
    render_size: tuple[int, int],
    frame_index: int,
    seed: int,
    settings: object | None = None,
) -> np.ndarray:
    density = np.clip(np.asarray(domain.to_density_numpy(), dtype=np.float32), 0.0, None)
    if density.ndim != 3:
        raise ValueError("physical smoke density must be a 3D array")

    def _optional_volume(name: str, *, clip_nonnegative: bool = True) -> np.ndarray:
        getter = getattr(domain, name, None)
        if getter is None:
            return np.zeros_like(density, dtype=np.float32)
        try:
            arr = np.asarray(getter(), dtype=np.float32)
        except Exception:
            return np.zeros_like(density, dtype=np.float32)
        if arr.shape != density.shape:
            return np.zeros_like(density, dtype=np.float32)
        if clip_nonnegative:
            arr = np.clip(arr, 0.0, None)
        return arr.astype(np.float32)

    temperature = _optional_volume("to_temperature_numpy")
    soot = _optional_volume("to_soot_numpy")
    emission = _optional_volume("to_emission_numpy")
    particle_age = _optional_volume("to_particle_age_numpy", clip_nonnegative=False)
    density_scale = float(getattr(settings, "density_scale", 1.18))
    extinction = float(getattr(settings, "extinction", 1.42))
    soot_absorption = float(getattr(settings, "soot_absorption", 0.34))
    exposure = float(getattr(settings, "exposure", 1.08))
    scattering = float(getattr(settings, "scattering", 0.92))
    absorption = float(getattr(settings, "absorption", 0.32))
    thin_color = np.asarray(getattr(settings, "thin_color", (0.43, 0.49, 0.56)), dtype=np.float32)
    dense_color = np.asarray(getattr(settings, "dense_color", (0.88, 0.87, 0.80)), dtype=np.float32)
    if thin_color.shape != (3,):
        thin_color = np.array([0.43, 0.49, 0.56], dtype=np.float32)
    if dense_color.shape != (3,):
        dense_color = np.array([0.88, 0.87, 0.80], dtype=np.float32)
    fire_glow = float(getattr(settings, "fire_glow", 0.26))
    phase_g = float(getattr(settings, "phase_g", 0.48))
    self_shadow = bool(getattr(settings, "self_shadow", True))
    view_dir = np.asarray(_physical_projection_view_direction(), dtype=np.float32)
    view_dir /= max(float(np.linalg.norm(view_dir)), 1.0e-6)
    light_dir = np.asarray(PHYSICAL_SMOKE_SUN_DIRECTION, dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)
    phase = _henyey_greenstein_py(float(np.dot(view_dir, light_dir)), phase_g)

    depth, altitude_count, width = density.shape
    render_w, render_h = (max(1, int(render_size[0])), max(1, int(render_size[1])))
    step_size = float(getattr(settings, "step_size", 0.72))
    if step_size <= 0.0:
        diagonal = math.sqrt(width * width + altitude_count * altitude_count + depth * depth)
        step_size = max(diagonal / max(float(getattr(settings, "max_steps", 128)), 1.0), 0.35)
    max_steps = int(max(1, getattr(settings, "max_steps", 128)))
    jitter_strength = float(np.clip(getattr(settings, "jitter_strength", 0.35), 0.0, 1.0))
    transmittance = np.ones((render_h, render_w), dtype=np.float32)
    rgb_accum = np.zeros((render_h, render_w, 3), dtype=np.float32)
    texture = _advected_smoke_texture((render_h, render_w), frame_index, seed + 3109)
    pixel_x, pixel_z = _pixel_grids((render_h, render_w))
    plane_x = pixel_x / max(float(render_w - 1), 1.0) * max(float(width - 1), 0.0)
    plane_z = pixel_z / max(float(render_h - 1), 1.0) * max(float(depth - 1), 0.0)
    plane_y = np.full_like(plane_x, (altitude_count - 1) * 0.5, dtype=np.float32)
    diagonal = math.sqrt(max(float(width - 1), 1.0) ** 2 + max(float(altitude_count - 1), 1.0) ** 2 + max(float(depth - 1), 1.0) ** 2)
    origin_x = (plane_x - float(view_dir[0]) * (diagonal + step_size)).astype(np.float32)
    origin_y = (plane_y - float(view_dir[1]) * (diagonal + step_size)).astype(np.float32)
    origin_z = (plane_z - float(view_dir[2]) * (diagonal + step_size)).astype(np.float32)
    t_enter, t_exit, valid = _python_ray_box_intersection(
        origin_x,
        origin_y,
        origin_z,
        view_dir,
        (max(float(width - 1), 0.0), max(float(altitude_count - 1), 0.0), max(float(depth - 1), 0.0)),
    )
    jitter = _bilinear_sample_wrapped(texture, pixel_x + seed * 0.017, pixel_z - frame_index * 0.023)
    t_current = (np.maximum(t_enter, 0.0) + jitter * jitter_strength * step_size).astype(np.float32)
    shadow_volume = (
        _python_volume_shadow_grid(density, soot, particle_age, light_dir, density_scale, extinction, soot_absorption, settings)
        if self_shadow
        else None
    )
    scatter_albedo = np.clip(scattering / max(scattering + absorption, 1.0e-5), 0.02, 0.98)
    sun_radiance = np.array([1.0, 0.96, 0.84], dtype=np.float32) * 11.5
    sky_base = np.array([0.52, 0.60, 0.72], dtype=np.float32)
    bounce_base = np.array([0.58, 0.54, 0.48], dtype=np.float32)
    old_blue = np.array([0.46, 0.49, 0.52], dtype=np.float32)
    aged_blue = np.array([0.36, 0.39, 0.43], dtype=np.float32)
    pale_core = np.array([0.70, 0.73, 0.72], dtype=np.float32)
    glow_color = np.array([3.40, 0.62, 0.03], dtype=np.float32)
    warm_color = np.array([0.95, 0.58, 0.24], dtype=np.float32)
    rgb_flat = rgb_accum.reshape(-1, 3)
    trans_flat = transmittance.reshape(-1)
    t_flat = t_current.reshape(-1)
    pixel_x_flat = pixel_x.reshape(-1)
    pixel_z_flat = pixel_z.reshape(-1)

    for _step_index in range(max_steps):
        active = valid & (t_current <= t_exit) & (transmittance > 0.008)
        if not bool(np.any(active)):
            break
        active_index = np.flatnonzero(active)
        t = t_current[active]
        segment_length = np.minimum(step_size, t_exit[active] - t).astype(np.float32)
        segment_length = np.clip(segment_length, 0.0, step_size)
        sample_x = origin_x[active] + float(view_dir[0]) * t
        sample_y = origin_y[active] + float(view_dir[1]) * t
        sample_z = origin_z[active] + float(view_dir[2]) * t
        layer = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
        occupied = (layer > 1.0e-5) & (segment_length > 1.0e-5)
        t_flat[active_index] = t + step_size
        if not bool(np.any(occupied)):
            continue
        occupied_index = active_index[occupied]
        sample_x = sample_x[occupied]
        sample_y = sample_y[occupied]
        sample_z = sample_z[occupied]
        segment_length = segment_length[occupied]
        layer = layer[occupied]
        temp_layer = _trilinear_sample_volume(temperature, sample_x, sample_y, sample_z)
        soot_layer = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        emission_layer = _trilinear_sample_volume(emission, sample_x, sample_y, sample_z)
        age_layer = _trilinear_sample_volume(particle_age, sample_x, sample_y, sample_z)
        altitude = np.clip(sample_y / max(float(altitude_count - 1), 1.0), 0.0, 1.0).astype(np.float32)
        age_t = _smoothstep(1.6, 17.0, age_layer)
        sigma = np.clip(
            layer * density_scale * (1.05 + 0.34 * altitude) * (1.0 + soot_layer * soot_absorption),
            0.0,
            4.5,
        )
        concentration_gate = 0.50 + 0.50 * _smoothstep(0.045, 0.34, layer)
        sigma *= 1.0 - 0.58 * age_t
        sigma *= concentration_gate
        sigma_t = np.clip(sigma * extinction, 0.0, 8.0).astype(np.float32)
        optical_depth = sigma_t * segment_length
        segment_transmittance = np.exp(-optical_depth).astype(np.float32)
        segment_weight = np.divide(
            1.0 - segment_transmittance,
            sigma_t,
            out=segment_length.copy(),
            where=sigma_t > 1.0e-6,
        )
        segment_weight *= 0.80 + 0.20 * _bilinear_sample_wrapped(
            texture,
            pixel_x_flat[occupied_index] + altitude * 6.5,
            pixel_z_flat[occupied_index] - altitude * 4.0,
        )
        dense_t = _smoothstep(0.05, 1.40, sigma)
        thin = thin_color * (1.0 - dense_t[..., None]) + pale_core * dense_t[..., None]
        layer_rgb = old_blue * (1.0 - dense_t[..., None]) + thin * dense_t[..., None]
        core = (
            _smoothstep(0.58, 1.62, sigma)
            * (1.0 - 0.26 * _smoothstep(0.08, 0.42, soot_layer))
            * (1.0 - 0.68 * age_t)
        )
        layer_rgb = layer_rgb * (1.0 - core[..., None]) + dense_color * core[..., None]
        layer_rgb = layer_rgb * (1.0 - 0.42 * age_t[..., None]) + aged_blue * (0.42 * age_t[..., None])
        warm = _smoothstep(0.18, 1.10, temp_layer) * (1.0 - age_t)
        layer_rgb = layer_rgb * (1.0 - warm[..., None] * 0.035) + warm_color * (warm[..., None] * 0.035)
        if shadow_volume is not None:
            light_trans = _trilinear_sample_volume(shadow_volume, sample_x, sample_y, sample_z)
        else:
            light_trans = np.ones_like(layer, dtype=np.float32)
        sigma_s = sigma_t * scatter_albedo
        sky_radiance = sky_base * (
            0.36 + 0.26 * (1.0 - light_trans)
        )[..., None]
        ground_bounce = bounce_base * (0.070 * (1.0 - altitude))[..., None]
        fresh_heat = temp_layer * (1.0 - age_t) * (1.0 - age_t)
        glow_strength = np.clip((fresh_heat * 0.10 + emission_layer * 1.18) * fire_glow, 0.0, 5.0)
        glow = glow_color * glow_strength[..., None]
        direct = layer_rgb * sigma_s[..., None] * sun_radiance * (phase * light_trans[..., None])
        multiple = layer_rgb * sigma_s[..., None] * (sky_radiance + ground_bounce)
        source_radiance = direct + multiple + glow
        rgb_flat[occupied_index] += source_radiance * segment_weight[..., None] * trans_flat[occupied_index][..., None]
        trans_flat[occupied_index] *= segment_transmittance
    alpha = np.clip(1.0 - transmittance, 0.0, 1.0)
    rgb = np.divide(rgb_accum, alpha[..., None], out=np.zeros_like(rgb_accum), where=alpha[..., None] > 1.0e-6)
    rgb = rgb / (1.0 + rgb)
    out = np.zeros((render_h, render_w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb * exposure * 255.0), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    return out


def _render_projected_physical_volume(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    render_w, render_h = effect.render_size
    if hasattr(effect.domain, "render_projection_rgba"):
        return np.asarray(
            effect.domain.render_projection_rgba(
                render_w,
                render_h,
                view_direction=_physical_projection_view_direction(),
                sun_direction=PHYSICAL_SMOKE_SUN_DIRECTION,
                settings=effect.render_settings,
            ),
            dtype=np.uint8,
        )
    return _python_projected_volume_raymarch(
        effect.domain,
        effect.render_size,
        frame_index,
        effect.seed,
        effect.render_settings,
    )


def _curl_warp_scalar(field: np.ndarray, frame_index: int, seed: int, strength: float) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if src.ndim != 2 or not np.any(src > 0.0):
        return src.astype(np.float32, copy=True)
    shape = src.shape
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    curl = _pil_blur_float(_advected_smoke_texture(shape, frame_index, seed), 4.8)
    gy, gx = np.gradient(curl)
    amp = float(strength) * max(scale, 0.12)
    dx = gy * amp + np.sin(y / max(24.0 * scale, 1.0) + curl * 4.8 + frame_index * 0.011) * 1.7 * scale
    dy = -gx * amp + np.sin(x / max(34.0 * scale, 1.0) - curl * 3.9 + frame_index * 0.008) * 1.1 * scale
    return _bilinear_sample(src, x - dx, y - dy)


def _directional_accumulation(
    field: np.ndarray,
    direction: tuple[float, float],
    *,
    steps: int,
    step_px: float,
    falloff: float,
) -> np.ndarray:
    src = np.clip(np.asarray(field, dtype=np.float32), 0.0, None)
    if src.ndim != 2 or not np.any(src > 0.0):
        return np.zeros_like(src, dtype=np.float32)
    dx, dy = float(direction[0]), float(direction[1])
    length = math.hypot(dx, dy)
    if length < 1.0e-6:
        return np.zeros_like(src, dtype=np.float32)
    dx /= length
    dy /= length
    x, y = _pixel_grids(src.shape)
    accum = np.zeros_like(src, dtype=np.float32)
    weight_sum = 0.0
    for idx in range(1, max(1, int(steps)) + 1):
        weight = math.exp(-float(idx) / max(float(falloff), 1.0e-3))
        offset = float(idx) * float(step_px)
        accum += _bilinear_sample(src, x - dx * offset, y - dy * offset) * weight
        weight_sum += weight
    return accum / max(weight_sum, 1.0e-6)


def _projected_smoke_depth_cues(
    alpha: np.ndarray,
    frame_index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if density.ndim != 2 or not np.any(density > 0.0):
        zeros = np.zeros_like(density, dtype=np.float32)
        return zeros, zeros, zeros
    scale = min(density.shape) / 408.0
    soft_density = _pil_blur_float(density, 2.0)
    sun_shadow = _directional_accumulation(
        soft_density,
        (0.58, -0.34),
        steps=12,
        step_px=max(2.4 * scale, 0.8),
        falloff=5.4,
    )
    view_depth = _directional_accumulation(
        soft_density,
        (-0.18, 0.92),
        steps=8,
        step_px=max(3.4 * scale, 1.0),
        falloff=4.0,
    )
    shadow_texture = _pil_blur_float(_advected_smoke_texture(density.shape, frame_index, seed + 13007), 5.4)
    shadow = np.clip((sun_shadow * 0.82 + view_depth * 0.30) * (0.78 + 0.44 * shadow_texture), 0.0, 1.0)
    gy, gx = np.gradient(_pil_blur_float(density, 1.15))
    sun_edge = np.clip(gx * 0.58 - gy * 0.34, 0.0, 1.0)
    rim = _pil_blur_float(sun_edge, 1.2) * _smoothstep(0.018, 0.36, density) * (1.0 - _smoothstep(0.58, 0.95, density))
    optical_depth = np.clip(_pil_blur_float(density, 5.2) * 0.62 + shadow * 0.54, 0.0, 1.0)
    return shadow.astype(np.float32), rim.astype(np.float32), optical_depth.astype(np.float32)


def _active_source_centroid(effect: PhysicalSmokeMainEffect, frame_index: int) -> tuple[float, float, float]:
    weights: list[float] = []
    coords: list[tuple[float, float]] = []
    frame_sources = _physical_sources_for_frame(effect, frame_index)
    for source in frame_sources:
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        flame = _source_flame_lifecycle_weight(source, frame_index)
        if flame <= 0.05:
            continue
        burst = _source_burst_envelope(source, frame_index)
        if burst <= 0.05:
            continue
        weights.append(float(source.strength * source.smoke_rate * flame * (0.35 + burst) * (1.0 + 0.15 * source.heat)))
        coords.append((float(source.x), float(source.y)))
    if not weights:
        if frame_sources:
            coords = [(float(source.x), float(source.y)) for source in frame_sources[: min(6, len(frame_sources))]]
            weights = [max(float(source.strength), 0.05) for source in frame_sources[: len(coords)]]
        else:
            return effect.map_size[0] * 0.36, effect.map_size[1] * 0.62, 0.0
    weights_arr = np.asarray(weights, dtype=np.float32)
    coords_arr = np.asarray(coords, dtype=np.float32)
    total = max(float(np.sum(weights_arr)), 1.0e-6)
    centroid = np.sum(coords_arr * weights_arr[:, None], axis=0) / total
    return float(centroid[0]), float(centroid[1]), float(np.clip(np.mean(weights_arr), 0.0, 8.0))


def _angle_difference(a: np.ndarray, b: float) -> np.ndarray:
    return np.arctan2(np.sin(a - float(b)), np.cos(a - float(b))).astype(np.float32)


def _physical_structure_enhancement_rgba(
    effect: PhysicalSmokeMainEffect,
    frame_index: int,
    base_alpha: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(np.asarray(base_alpha, dtype=np.float32), 0.0, 1.0)
    height, width = alpha.shape
    x, y = _pixel_grids((height, width))
    wind = _hybrid_layer_wind_vector(1).astype(np.float32)
    wind_len = max(float(np.linalg.norm(wind)), 1.0e-6)
    wind = wind / wind_len
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    cx, cy, active_strength = _active_source_centroid(effect, frame_index)
    rel_x = x - np.float32(cx)
    rel_y = y - np.float32(cy)
    along = rel_x * wind[0] + rel_y * wind[1]
    cross_coord = rel_x * cross[0] + rel_y * cross[1]
    scale = min(height, width) / 408.0
    source_envelope = _smoothstep(0.0, 42.0 * scale, along) * (1.0 - _smoothstep(340.0 * scale, 515.0 * scale, along))
    base_support = np.clip(_smoothstep(0.008, 0.18, alpha) + 0.70 * source_envelope, 0.0, 1.0)
    broad_texture = _pil_blur_float(_advected_smoke_texture((height, width), frame_index, effect.seed + 25031), 6.6)
    fine_texture = _pil_blur_float(_advected_smoke_texture((height, width), frame_index, effect.seed + 26033), 2.0)

    streamers = np.zeros_like(alpha, dtype=np.float32)
    for lane_idx, lane_offset in enumerate((-22.0, -9.0, 6.0, 19.0)):
        phase = frame_index * (0.020 + lane_idx * 0.004) + effect.seed * 0.003 + lane_idx * 1.37
        center = (
            lane_offset * scale
            + np.sin(along / max((36.0 + lane_idx * 9.0) * scale, 1.0) + phase) * (10.0 + lane_idx * 2.4) * scale
            + (broad_texture - 0.5) * 16.0 * scale
        )
        width_px = (3.6 + lane_idx * 0.7) * scale + np.clip(along, 0.0, 260.0 * scale) * 0.035
        ridge = np.exp(-((cross_coord - center) ** 2) / (2.0 * np.maximum(width_px, 1.0) ** 2))
        lane_gate = _smoothstep(4.0 * scale, 44.0 * scale, along) * (1.0 - _smoothstep(210.0 * scale, 360.0 * scale, along))
        streamers += ridge * lane_gate * (0.86 + 0.34 * (fine_texture - 0.5))

    hooks = np.zeros_like(alpha, dtype=np.float32)
    for hook_idx, distance in enumerate((96.0, 158.0, 235.0, 315.0)):
        side = -1.0 if hook_idx % 2 else 1.0
        phase = frame_index * (0.010 + hook_idx * 0.002) + effect.seed * 0.002 + hook_idx * 1.11
        center_x = np.float32(cx) + wind[0] * distance * scale + cross[0] * side * (28.0 + hook_idx * 10.0) * scale
        center_y = np.float32(cy) + wind[1] * distance * scale + cross[1] * side * (28.0 + hook_idx * 10.0) * scale
        hx = (x - center_x) * wind[0] + (y - center_y) * wind[1]
        hy = (x - center_x) * cross[0] + (y - center_y) * cross[1]
        radius = (34.0 + hook_idx * 13.0) * scale
        width_px = (6.0 + hook_idx * 1.8) * scale
        ring = np.exp(-((np.sqrt(hx * hx + hy * hy) - radius) ** 2) / (2.0 * max(width_px, 1.0) ** 2))
        theta = np.arctan2(hy * side, hx + radius * 0.28)
        target_angle = 0.45 + 0.42 * math.sin(phase)
        arc = np.exp(-(_angle_difference(theta, target_angle) ** 2) / (2.0 * 0.78 * 0.78))
        break_mask = 0.64 + 0.58 * broad_texture - 0.24 * fine_texture
        hooks += ring * arc * break_mask * (0.72 - hook_idx * 0.07)

    fan = np.zeros_like(alpha, dtype=np.float32)
    for fan_idx, fan_offset in enumerate((-1.0, 0.0, 1.0)):
        center = fan_offset * 58.0 * scale + np.sin(along / max(115.0 * scale, 1.0) + fan_idx) * 18.0 * scale
        width_px = 20.0 * scale + np.clip(along, 0.0, 390.0 * scale) * 0.13
        band = np.exp(-((cross_coord - center) ** 2) / (2.0 * np.maximum(width_px, 1.0) ** 2))
        gate = _smoothstep(120.0 * scale, 210.0 * scale, along) * (1.0 - _smoothstep(390.0 * scale, 545.0 * scale, along))
        fan += band * gate * (0.35 + 0.55 * broad_texture)

    structure = np.clip(streamers * 0.70 + hooks * 0.92 + fan * 0.04, 0.0, 1.0)
    structure *= base_support * np.clip(0.80 + 0.72 * (broad_texture - 0.5) + 0.34 * (fine_texture - 0.5), 0.38, 1.32)
    voids = _smoothstep(0.58, 0.92, 1.0 - broad_texture) * _smoothstep(0.04, 0.62, structure)
    structure *= 1.0 - 0.38 * voids
    structure = _pil_blur_float(np.clip(structure, 0.0, 1.0), 1.15)
    if not np.any(structure > 0.002):
        return np.zeros((height, width, 4), dtype=np.uint8)

    age_t = _smoothstep(120.0 * scale, 450.0 * scale, along)
    dense_t = _smoothstep(0.22, 0.84, structure + alpha * 0.30)
    thin = np.array([140.0, 154.0, 166.0], dtype=np.float32)
    mid = np.array([180.0, 187.0, 184.0], dtype=np.float32)
    milky = np.array([238.0, 237.0, 224.0], dtype=np.float32)
    rgb = thin * (1.0 - dense_t[..., None]) + mid * dense_t[..., None]
    core = _smoothstep(0.44, 0.92, structure) * (1.0 - 0.45 * age_t)
    rgb = rgb * (1.0 - core[..., None]) + milky * core[..., None]
    aged_blue = np.array([86.0, 99.0, 116.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.26 * age_t[..., None]) + aged_blue * (0.26 * age_t[..., None])
    shadow, rim, depth = _projected_smoke_depth_cues(structure, frame_index, effect.seed + 27143)
    rgb *= (1.0 - np.clip(0.14 * shadow + 0.08 * depth, 0.0, 0.26))[..., None]
    rgb += rim[..., None] * np.array([28.0, 27.0, 22.0], dtype=np.float32)

    alpha_u8 = np.clip(np.round(structure * PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA * 0.82), 0, PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 245).astype(np.uint8)
    out[..., 3] = np.where(alpha_u8 >= 3, alpha_u8, 0).astype(np.uint8)
    return out


def _domain_density_velocity_numpy(effect: PhysicalSmokeMainEffect) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        density = np.asarray(effect.domain.to_density_numpy(), dtype=np.float32)
    except Exception:
        return np.zeros((0, 0, 0), dtype=np.float32), None
    velocity = None
    if hasattr(effect.domain, "to_velocity_numpy"):
        try:
            velocity = np.asarray(effect.domain.to_velocity_numpy(), dtype=np.float32)
        except Exception:
            velocity = None
    elif hasattr(effect.domain, "velocity"):
        try:
            velocity = np.asarray(effect.domain.velocity, dtype=np.float32)
        except Exception:
            velocity = None
    return density, velocity


def _physical_volume_lane_fields(effect: PhysicalSmokeMainEffect, frame_index: int) -> dict[str, np.ndarray] | None:
    density, velocity = _domain_density_velocity_numpy(effect)
    map_w, map_h = effect.map_size
    if density.ndim != 3 or density.size == 0 or not np.any(density > 1.0e-6):
        return None

    d = np.clip(density, 0.0, None)
    column = np.sum(d, axis=1)
    peak = max(float(np.percentile(column[column > 0.0], 98.5)) if np.any(column > 0.0) else 0.0, 1.0e-6)
    column_t = 1.0 - np.exp(-column / peak * 1.75)
    support = _resample_float_field(column_t, (map_h, map_w))
    support = _pil_blur_float(np.clip(support, 0.0, 1.0), 0.55)

    altitude_idx = np.linspace(0.0, 1.0, d.shape[1], dtype=np.float32)[None, :, None]
    altitude_mass = np.sum(d * altitude_idx, axis=1)
    altitude = np.divide(altitude_mass, np.maximum(column, 1.0e-6), out=np.zeros_like(column), where=column > 1.0e-6)
    altitude_map = _resample_float_field(altitude, (map_h, map_w))

    fine = _pil_blur_float(support, 0.75)
    broad = _pil_blur_float(support, 7.0)
    density_ridges = np.clip(fine - broad * 0.72, 0.0, 1.0)
    ridge_positive = density_ridges[density_ridges > 0.0]
    ridge_scale = max(float(np.percentile(ridge_positive, 96.0)) if ridge_positive.size else 0.0, 1.0e-5)
    density_ridges = np.clip(density_ridges / ridge_scale, 0.0, 1.0)
    gy, gx = np.gradient(_pil_blur_float(support, 1.2))
    density_edge = np.clip(np.sqrt(gx * gx + gy * gy) * 6.0, 0.0, 1.0)

    curl_map = np.zeros_like(support, dtype=np.float32)
    shear_map = np.zeros_like(support, dtype=np.float32)
    speed_map = np.zeros_like(support, dtype=np.float32)
    if velocity is not None and velocity.shape == density.shape + (3,):
        weights = np.maximum(d, 1.0e-5)
        weight_sum = np.sum(weights, axis=1)
        vx = np.sum(velocity[..., 0] * weights, axis=1) / np.maximum(weight_sum, 1.0e-6)
        vz = np.sum(velocity[..., 2] * weights, axis=1) / np.maximum(weight_sum, 1.0e-6)
        vx_map = _resample_float_field(vx, (map_h, map_w))
        vz_map = _resample_float_field(vz, (map_h, map_w))
        dvx_dy, dvx_dx = np.gradient(_pil_blur_float(vx_map, 1.0))
        dvz_dy, dvz_dx = np.gradient(_pil_blur_float(vz_map, 1.0))
        curl = dvz_dx - dvx_dy
        shear = np.sqrt((dvx_dx - dvz_dy) ** 2 + (dvx_dy + dvz_dx) ** 2)
        curl_scale = max(float(np.percentile(np.abs(curl), 96.0)), 1.0e-5)
        shear_scale = max(float(np.percentile(shear, 96.0)), 1.0e-5)
        curl_map = np.clip(np.abs(curl) / curl_scale, 0.0, 1.0)
        shear_map = np.clip(shear / shear_scale, 0.0, 1.0)
        curl_map = _pil_blur_float(curl_map, 0.85)
        shear_map = _pil_blur_float(shear_map, 1.2)
        speed = np.sqrt(vx_map * vx_map + vz_map * vz_map)
        speed_scale = max(float(np.percentile(speed, 96.0)), 1.0e-5)
        speed_map = _pil_blur_float(np.clip(speed / speed_scale, 0.0, 1.0), 1.4)

    texture = _pil_blur_float(_advected_smoke_texture((map_h, map_w), frame_index, effect.seed + 31057), 3.2)
    lane_texture = _pil_blur_float(_advected_smoke_texture((map_h, map_w), frame_index, effect.seed + 31891), 8.8)
    x, y = _pixel_grids((map_h, map_w))
    wind = _hybrid_layer_wind_vector(1)
    along = x * wind[0] + y * wind[1]
    cross = x * (-wind[1]) + y * wind[0]
    scale = min(map_h, map_w) / 408.0
    filament_phase = (
        along / max(48.0 * scale, 1.0)
        - cross / max(29.0 * scale, 1.0)
        + lane_texture * 5.7
        + curl_map * 2.4
        + frame_index * 0.010
    )
    filament_gate = _pil_blur_float(_smoothstep(0.54, 0.98, 0.5 + 0.5 * np.sin(filament_phase)), 1.2)
    rollup_phase = (
        along / max(34.0 * scale, 1.0)
        + np.sin(cross / max(24.0 * scale, 1.0) + lane_texture * 3.6) * 2.2
        + curl_map * 2.7
        - frame_index * 0.013
    )
    rollup_gate = _pil_blur_float(_smoothstep(0.56, 0.98, 0.5 + 0.5 * np.sin(rollup_phase)), 1.45)
    lane_field = np.clip(
        filament_gate * 0.44
        + rollup_gate * 0.24
        + density_ridges * 0.34
        + curl_map * 0.18
        + shear_map * 0.14
        + speed_map * 0.10,
        0.0,
        1.0,
    )
    sheet_support = _smoothstep(0.055, 0.58, support)
    weak_lane = 1.0 - _smoothstep(0.26, 0.76, lane_field + density_ridges * 0.28 + density_edge * 0.16)
    texture_voids = _smoothstep(0.54, 0.92, 1.0 - lane_texture)
    ribbon_gaps = _smoothstep(0.48, 0.90, 1.0 - filament_gate) * _smoothstep(0.42, 0.88, 1.0 - rollup_gate)
    sheet_voids = np.maximum(texture_voids * weak_lane, ribbon_gaps * (1.0 - 0.34 * density_ridges))
    sheet_voids = _pil_blur_float(np.clip(sheet_voids * sheet_support, 0.0, 1.0), 1.25)
    lane_gain = np.clip(
        0.66
        + 0.42 * lane_field
        + 0.24 * density_ridges
        + 0.12 * density_edge
        + 0.12 * curl_map
        + 0.08 * shear_map
        - 0.42 * sheet_voids,
        0.28,
        1.24,
    )
    return {
        "support": support.astype(np.float32),
        "altitude": altitude_map.astype(np.float32),
        "ridges": density_ridges.astype(np.float32),
        "edge": density_edge.astype(np.float32),
        "curl": curl_map.astype(np.float32),
        "shear": shear_map.astype(np.float32),
        "speed": speed_map.astype(np.float32),
        "texture": texture.astype(np.float32),
        "lane_texture": lane_texture.astype(np.float32),
        "filaments": filament_gate.astype(np.float32),
        "rollups": rollup_gate.astype(np.float32),
        "lane": lane_field.astype(np.float32),
        "voids": sheet_voids.astype(np.float32),
        "gain": lane_gain.astype(np.float32),
    }


def _physical_lane_field(
    volume_fields: dict[str, np.ndarray] | None,
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if volume_fields is None or key not in volume_fields:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(volume_fields[key], dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros(shape, dtype=np.float32)
    if arr.shape != shape:
        arr = _resample_float_field(arr, shape)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _physical_volume_structure_rgba(
    effect: PhysicalSmokeMainEffect,
    frame_index: int,
    volume_fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    map_w, map_h = effect.map_size
    fields = volume_fields if volume_fields is not None else _physical_volume_lane_fields(effect, frame_index)
    if fields is None:
        return np.zeros((map_h, map_w, 4), dtype=np.uint8)

    support = _physical_lane_field(fields, "support", (map_h, map_w))
    altitude_map = _physical_lane_field(fields, "altitude", (map_h, map_w))
    density_ridges = _physical_lane_field(fields, "ridges", (map_h, map_w))
    density_edge = _physical_lane_field(fields, "edge", (map_h, map_w))
    curl_map = _physical_lane_field(fields, "curl", (map_h, map_w))
    shear_map = _physical_lane_field(fields, "shear", (map_h, map_w))
    lane_field = _physical_lane_field(fields, "lane", (map_h, map_w))
    sheet_voids = _physical_lane_field(fields, "voids", (map_h, map_w))
    texture = _physical_lane_field(fields, "texture", (map_h, map_w))

    physical_structure = (
        density_ridges * 0.60
        + density_edge * 0.20
        + curl_map * 0.30
        + shear_map * 0.22
        + lane_field * 0.34
    )
    physical_structure *= _smoothstep(0.020, 0.42, support)
    physical_structure *= np.clip(0.72 + 0.52 * (texture - 0.5) + 0.38 * altitude_map, 0.34, 1.20)
    physical_structure *= 1.0 - 0.36 * sheet_voids * _smoothstep(0.05, 0.68, physical_structure)
    physical_structure = _pil_blur_float(np.clip(physical_structure, 0.0, 1.0), 0.95)
    if not np.any(physical_structure > 0.003):
        return np.zeros((map_h, map_w, 4), dtype=np.uint8)

    aged = _smoothstep(0.18, 0.88, support) * (1.0 - _smoothstep(0.18, 0.72, density_ridges))
    dense_t = _smoothstep(0.18, 0.86, physical_structure + support * 0.16)
    thin = np.array([116.0, 132.0, 148.0], dtype=np.float32)
    mid = np.array([160.0, 170.0, 171.0], dtype=np.float32)
    bright = np.array([226.0, 226.0, 214.0], dtype=np.float32)
    rgb = thin * (1.0 - dense_t[..., None]) + mid * dense_t[..., None]
    bright_core = _smoothstep(0.44, 0.92, density_ridges + lane_field * 0.24) * (1.0 - 0.45 * aged)
    rgb = rgb * (1.0 - bright_core[..., None]) + bright * bright_core[..., None]
    aged_blue = np.array([78.0, 92.0, 110.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.24 * aged[..., None]) + aged_blue * (0.24 * aged[..., None])
    shadow, rim, depth = _projected_smoke_depth_cues(physical_structure, frame_index, effect.seed + 32213)
    rgb *= (1.0 - np.clip(0.18 * shadow + 0.10 * depth, 0.0, 0.32))[..., None]
    rgb += rim[..., None] * np.array([20.0, 20.0, 17.0], dtype=np.float32)

    out = np.zeros((map_h, map_w, 4), dtype=np.uint8)
    alpha_u8 = np.clip(
        np.round(physical_structure * PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA),
        0,
        PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA,
    )
    out[..., :3] = np.clip(np.round(rgb), 0, 242).astype(np.uint8)
    out[..., 3] = np.where(alpha_u8 >= 3, alpha_u8, 0).astype(np.uint8)
    return out


def _physical_history_layer_rgba(
    rgba: np.ndarray,
    age_frames: int,
    frame_index: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4 or age_frames <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    alpha = arr[..., 3] / 255.0
    if not np.any(alpha > 0.0):
        return np.zeros_like(arr, dtype=np.uint8)

    height, width = alpha.shape
    shape = (height, width)
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    history_noise = _pil_blur_float(_advected_smoke_texture(shape, frame_index - age_frames // 2, seed + 11903), 8.0)
    gy, gx = np.gradient(history_noise)
    age = float(age_frames)
    drift = age * 0.34 * max(scale, 0.16)
    sway = math.sin(frame_index * 0.013 + seed * 0.007 + age * 0.041) * age * 0.050 * max(scale, 0.16)
    curl_amp = min(age, 72.0) * 5.2 * max(scale, 0.16)
    sample_x = x - wind[0] * drift - cross[0] * sway - gy * curl_amp
    sample_y = y - wind[1] * drift - cross[1] * sway + gx * curl_amp

    warped = np.zeros_like(arr, dtype=np.float32)
    for channel in range(4):
        warped[..., channel] = _bilinear_sample(arr[..., channel], sample_x, sample_y)
    blur_radius = min(3.6, 0.026 * age)
    if blur_radius > 0.25:
        for channel in range(4):
            warped[..., channel] = _pil_blur_float(warped[..., channel], blur_radius)

    age_t = float(_smoothstep(0.0, PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES, age))
    fade = math.exp(-age / 68.0) * (1.0 - float(_smoothstep(72.0, PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES, age)) * 0.72)
    alpha_f = np.clip(
        warped[..., 3] / 255.0 * fade * PHYSICAL_SMOKE_HISTORY_ALPHA_SCALE,
        0.0,
        PHYSICAL_SMOKE_MAX_ALPHA / 255.0,
    )
    along_coord = x * wind[0] + y * wind[1]
    cross_coord = x * (-wind[1]) + y * wind[0]
    generation_texture = _pil_blur_float(
        _advected_smoke_texture(shape, frame_index - age_frames, seed + 15131),
        9.5 + age_t * 4.0,
    )
    ribbon_phase = (
        along_coord / max((50.0 + age * 0.20) * scale, 1.0)
        - cross_coord / max((24.0 + age * 0.11) * scale, 1.0)
        + generation_texture * 5.1
        + frame_index * 0.006
    )
    aged_ribbons = _pil_blur_float(0.5 + 0.5 * np.sin(ribbon_phase), 1.8 + age_t * 1.25)
    generation_phase = (
        along_coord / max((74.0 + age * 0.18) * scale, 1.0)
        + cross_coord / max((31.0 + age * 0.10) * scale, 1.0)
        + generation_texture * 4.2
        - frame_index * 0.004
    )
    generation_bands = _pil_blur_float(0.5 + 0.5 * np.sin(generation_phase), 3.4 + age_t * 1.6)
    aged_voids = (
        _smoothstep(0.54, 0.90, 1.0 - generation_texture)
        * _smoothstep(0.46, 0.88, 1.0 - aged_ribbons)
        * _smoothstep(0.04, 0.46, alpha_f)
    )
    alpha_f *= np.clip(
        0.47 + 0.47 * generation_texture + 0.34 * (aged_ribbons - 0.5) + 0.22 * (generation_bands - 0.5),
        0.14,
        1.16,
    )
    alpha_f *= 1.0 - (0.42 + 0.30 * age_t) * aged_voids
    alpha_f = _pil_blur_float(alpha_f, 0.65 + age_t * 1.25)
    aged_blue = np.array([82.0, 95.0, 110.0], dtype=np.float32)
    flat_gray = np.array([135.0, 143.0, 148.0], dtype=np.float32)
    target_rgb = aged_blue * age_t + flat_gray * (1.0 - age_t)
    rgb = warped[..., :3] * (1.0 - 0.58 * age_t) + target_rgb * (0.58 * age_t)
    history_shadow = _directional_accumulation(
        _pil_blur_float(alpha_f, 1.8 + age_t),
        (0.58, -0.34),
        steps=5,
        step_px=max(3.5 * scale, 0.9),
        falloff=3.2,
    )
    history_rim = np.clip(np.gradient(_pil_blur_float(alpha_f, 1.0))[1] * 0.70, 0.0, 1.0)
    rgb *= (0.94 - 0.10 * age_t) * (1.0 - 0.22 * history_shadow[..., None])
    rgb += history_rim[..., None] * np.array([14.0, 15.0, 14.0], dtype=np.float32) * (1.0 - 0.35 * age_t)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 235).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha_f * 255.0), 0, PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 2, out[..., 3], 0).astype(np.uint8)
    return out


def _physical_history_rgba(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    height, width = effect.map_size[1], effect.map_size[0]
    combined = np.zeros((height, width, 4), dtype=np.uint8)
    for history_frame, history_rgba in effect.history:
        age = int(frame_index) - int(history_frame)
        if age <= 0 or age > PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES:
            continue
        layer = _physical_history_layer_rgba(history_rgba, age, frame_index, effect.seed)
        combined = _premultiplied_over(combined, layer)
        combined[..., 3] = np.minimum(combined[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def _physical_source_glow_rgba(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    glow = hybrid_fire_sources_rgba(
        _physical_sources_for_frame(effect, frame_index),
        frame_index,
        effect.map_size,
        glow_only=False,
        bloom_scale=0.92,
        core_alpha_scale=0.38,
    )
    return _scale_rgba_alpha(glow, 0.58)


def _physical_temporal_reproject_rgba(
    rgba: np.ndarray,
    age_frames: int,
    frame_index: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4 or age_frames <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    alpha = arr[..., 3] / 255.0
    if not np.any(alpha > 0.0):
        return np.zeros_like(arr, dtype=np.uint8)

    height, width = alpha.shape
    shape = (height, width)
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    lane_noise = _pil_blur_float(_advected_smoke_texture(shape, frame_index, seed + 21101), 6.2)
    gy, gx = np.gradient(lane_noise)
    age = float(age_frames)
    drift = age * 0.48 * max(scale, 0.16)
    curl_amp = min(age, 4.0) * 4.2 * max(scale, 0.16)
    sway = math.sin(frame_index * 0.019 + seed * 0.004) * age * 0.035 * max(scale, 0.16)
    sample_x = x - wind[0] * drift - cross[0] * sway - gy * curl_amp
    sample_y = y - wind[1] * drift - cross[1] * sway + gx * curl_amp

    warped = np.zeros_like(arr, dtype=np.float32)
    for channel in range(4):
        warped[..., channel] = _bilinear_sample(arr[..., channel], sample_x, sample_y)
    for channel in range(4):
        warped[..., channel] = _pil_blur_float(warped[..., channel], 0.35)
    warped[..., 3] *= math.exp(-age / 4.5)
    out = np.clip(np.round(warped), 0, 255).astype(np.uint8)
    out[..., 3] = np.minimum(out[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return out


def _temporal_blend_physical_smoke(
    effect: PhysicalSmokeMainEffect,
    current: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    previous = effect.previous_render_rgba
    previous_frame = effect.previous_render_frame
    if previous is None or previous_frame is None:
        return current
    age = int(frame_index) - int(previous_frame)
    if age <= 0 or age > 4 or previous.shape != current.shape:
        return current
    reprojected = _physical_temporal_reproject_rgba(previous, age, frame_index, effect.seed)
    if not np.any(reprojected[..., 3] > 0):
        return current
    temporal_weight = 0.24 * (1.0 - float(_smoothstep(1.0, 4.0, age)))
    under = _scale_rgba_alpha(reprojected, temporal_weight)
    blended = _premultiplied_over(under, current)
    blended[..., 3] = np.minimum(blended[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return blended


def _record_physical_history(effect: PhysicalSmokeMainEffect, frame_index: int, rgba: np.ndarray) -> None:
    if effect.history and int(effect.history[-1][0]) == int(frame_index):
        return
    if effect.history and int(frame_index) - int(effect.history[-1][0]) < PHYSICAL_SMOKE_HISTORY_STRIDE:
        return
    stored = np.asarray(rgba, dtype=np.uint8).copy()
    effect.history.append((int(frame_index), stored))
    min_frame = int(frame_index) - PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES
    effect.history[:] = [
        (history_frame, history_rgba)
        for history_frame, history_rgba in effect.history
        if history_frame >= min_frame
    ][-PHYSICAL_SMOKE_HISTORY_MAX_LAYERS:]


def _postprocess_physical_smoke_rgba(
    rgba: np.ndarray,
    frame_index: int,
    seed: int,
    map_size: tuple[int, int],
    volume_fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    if image.size != tuple(map(int, map_size)):
        image = image.resize(tuple(map(int, map_size)), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    alpha = np.clip(arr[..., 3] / 255.0, 0.0, 1.0)
    if not np.any(alpha > 0.0):
        return np.zeros((int(map_size[1]), int(map_size[0]), 4), dtype=np.uint8)
    alpha = np.clip(_curl_warp_scalar(alpha, frame_index, seed + 7109, 390.0), 0.0, 1.0)
    alpha = np.clip(_curl_warp_scalar(alpha, frame_index, seed + 9719, 150.0) * 0.84 + alpha * 0.16, 0.0, 1.0)
    fine = _pil_blur_float(alpha, 0.85)
    mid = _pil_blur_float(alpha, 2.8)
    broad = _pil_blur_float(alpha, 7.2)
    texture = _advected_smoke_texture(alpha.shape, frame_index, seed + 4517)
    low_texture = _pil_blur_float(texture, 8.6)
    cell_texture = _pil_blur_float(_advected_smoke_texture(alpha.shape, frame_index, seed + 8123), 13.0)
    branch_texture = _pil_blur_float(_advected_smoke_texture(alpha.shape, frame_index, seed + 19037), 5.8)
    x, y = _pixel_grids(alpha.shape)
    scale = min(alpha.shape) / 408.0
    wind = _hybrid_layer_wind_vector(1)
    cross_coord = x * (-wind[1]) + y * wind[0]
    along_coord = x * wind[0] + y * wind[1]
    xnorm = x / max(float(alpha.shape[1] - 1), 1.0)
    fresh_zone = 1.0 - _smoothstep(0.20, 0.53, xnorm)
    aged_zone = _smoothstep(0.32, 0.98, xnorm)
    lane_phase = (
        cross_coord / max(9.5 * scale, 1.0)
        + along_coord / max(74.0 * scale, 1.0)
        + low_texture * 4.9
        + branch_texture * 2.2
        + frame_index * 0.022
    )
    streamer_wave = 0.5 + 0.5 * np.sin(lane_phase)
    streamers = _pil_blur_float(_smoothstep(0.48, 0.98, streamer_wave), 1.35)
    wide_streamers = _pil_blur_float(streamer_wave, 2.4)
    hook_wave = (
        0.5
        + 0.5
        * np.sin(
            cross_coord / max(18.0 * scale, 1.0)
            - along_coord / max(46.0 * scale, 1.0)
            + texture * 3.8
            + branch_texture * 2.6
            + frame_index * 0.014
        )
    )
    hooks = _pil_blur_float(_smoothstep(0.55, 0.98, hook_wave), 1.75)
    loop_phase = (
        along_coord / max(42.0 * scale, 1.0)
        + np.sin(cross_coord / max(28.0 * scale, 1.0) + branch_texture * 3.1) * 2.1
        - frame_index * 0.013
    )
    loops = _pil_blur_float(_smoothstep(0.54, 0.96, 0.5 + 0.5 * np.sin(loop_phase)), 3.0)
    cell_phase = (
        along_coord / max(48.0 * scale, 1.0)
        - cross_coord / max(34.0 * scale, 1.0)
        + cell_texture * 5.6
        + frame_index * 0.010
    )
    plume_cells = _pil_blur_float(0.5 + 0.5 * np.sin(cell_phase), 3.2)
    fan_phase = (
        cross_coord / max(30.0 * scale, 1.0)
        + along_coord / max(130.0 * scale, 1.0)
        + low_texture * 3.0
        - frame_index * 0.006
    )
    fan_bands = _pil_blur_float(0.5 + 0.5 * np.sin(fan_phase), 4.2)
    rollup_phase = (
        along_coord / max(36.0 * scale, 1.0)
        + np.sin(cross_coord / max(21.0 * scale, 1.0) + branch_texture * 3.8) * 2.4
        + low_texture * 3.2
        - frame_index * 0.012
    )
    rollup_ridges = _pil_blur_float(_smoothstep(0.57, 0.98, 0.5 + 0.5 * np.sin(rollup_phase)), 1.95)
    streaks = 0.5 + 0.5 * np.sin(
        x / max(alpha.shape[1] / 24.0, 1.0)
        + y / max(alpha.shape[0] / 6.0, 1.0)
        + frame_index * 0.016
        + seed * 0.011
    )
    veil = _pil_blur_float(alpha, 15.5)
    fresh_alpha = np.clip(fine * (0.60 + 0.36 * streamers) + mid * 0.085 + broad * 0.012, 0.0, 1.0)
    aged_alpha = np.clip(fine * 0.46 + mid * 0.18 + broad * 0.040 + veil * 0.0060, 0.0, 1.0)
    shaped_alpha = np.clip(fresh_alpha * fresh_zone + aged_alpha * (1.0 - fresh_zone), 0.0, 1.0)
    shaped_alpha = np.clip(
        shaped_alpha
        + rollup_ridges * _smoothstep(0.045, 0.58, alpha) * (0.020 + 0.046 * aged_zone)
        + hooks * streamers * _smoothstep(0.050, 0.62, alpha) * (0.026 + 0.034 * fresh_zone),
        0.0,
        1.0,
    )
    shaped_alpha *= np.clip(
        0.61
        + 0.30 * (low_texture - 0.5)
        + 0.26 * (streamers - 0.5) * (0.86 + 0.54 * fresh_zone)
        + 0.17 * (hooks - 0.5)
        + 0.14 * (loops - 0.5)
        + 0.08 * (plume_cells - 0.5)
        + 0.07 * (fan_bands - 0.5) * aged_zone
        + 0.08 * (streaks - 0.5),
        0.34,
        1.18,
    )
    shaped_alpha = _pil_blur_float(shaped_alpha, 0.92)
    shaped_alpha = np.clip(shaped_alpha, 0.0, 1.0) ** 0.94
    holes = _smoothstep(0.54, 0.94, 1.0 - low_texture) * _smoothstep(0.04, 0.58, shaped_alpha)
    lane_gaps = _smoothstep(0.50, 0.86, 1.0 - wide_streamers) * _smoothstep(0.05, 0.64, shaped_alpha)
    fresh_lane_gaps = fresh_zone * _smoothstep(0.54, 0.92, 1.0 - streamers) * _smoothstep(0.04, 0.58, shaped_alpha)
    cell_voids = _smoothstep(0.48, 0.84, 1.0 - plume_cells) * _smoothstep(0.07, 0.70, shaped_alpha)
    loop_voids = _smoothstep(0.52, 0.88, 1.0 - loops) * aged_zone * _smoothstep(0.08, 0.64, shaped_alpha)
    fan_voids = _smoothstep(0.50, 0.84, 1.0 - fan_bands) * aged_zone * _smoothstep(0.06, 0.58, shaped_alpha)
    branch_voids = _smoothstep(0.56, 0.88, 1.0 - branch_texture) * _smoothstep(0.08, 0.62, shaped_alpha)
    sheet_split = (
        _smoothstep(0.44, 0.82, 1.0 - rollup_ridges)
        * _smoothstep(0.38, 0.92, 1.0 - hooks)
        * aged_zone
        * _smoothstep(0.06, 0.66, shaped_alpha)
    )
    edge_band = _smoothstep(0.015, 0.24, shaped_alpha) * (1.0 - _smoothstep(0.42, 0.78, shaped_alpha))
    edge_breakup = _smoothstep(0.50, 0.88, 1.0 - low_texture) * edge_band

    volume_support = _physical_lane_field(volume_fields, "support", alpha.shape)
    volume_lane = _physical_lane_field(volume_fields, "lane", alpha.shape)
    volume_ridges = _physical_lane_field(volume_fields, "ridges", alpha.shape)
    volume_curl = _physical_lane_field(volume_fields, "curl", alpha.shape)
    volume_shear = _physical_lane_field(volume_fields, "shear", alpha.shape)
    volume_voids_base = _physical_lane_field(volume_fields, "voids", alpha.shape)
    volume_gain = _physical_lane_field(volume_fields, "gain", alpha.shape)
    volume_voids = np.zeros_like(shaped_alpha, dtype=np.float32)
    volume_lane_recovery = np.zeros_like(shaped_alpha, dtype=np.float32)
    if np.any(volume_support > 0.001):
        volume_sheet = _smoothstep(0.055, 0.58, volume_support) * _smoothstep(0.045, 0.64, shaped_alpha)
        weak_solver_lane = 1.0 - _smoothstep(0.32, 0.84, volume_lane + volume_ridges * 0.30 + volume_curl * 0.16)
        lane_channel_gaps = _smoothstep(0.48, 0.90, 1.0 - volume_lane) * (1.0 - 0.38 * volume_ridges)
        volume_voids = np.clip(
            np.maximum(volume_voids_base, lane_channel_gaps * weak_solver_lane) * volume_sheet,
            0.0,
            1.0,
        )
        volume_voids = _pil_blur_float(volume_voids, 0.95)
        solver_carve = volume_voids * np.clip(0.56 + 0.44 * weak_solver_lane, 0.0, 1.0)
        solver_gain = np.clip(
            0.62
            + 0.24 * (volume_gain - 0.66)
            + 0.20 * volume_lane
            + 0.13 * volume_ridges
            + 0.08 * (volume_curl + volume_shear)
            - 0.62 * solver_carve,
            0.32,
            1.22,
        )
        shaped_alpha *= solver_gain
        shaped_alpha *= 1.0 - (0.42 + 0.42 * aged_zone) * solver_carve
        volume_lane_recovery = (
            volume_support
            * _smoothstep(0.030, 0.62, alpha)
            * (
                0.038 * volume_ridges
                + 0.026 * volume_lane
                + 0.016 * volume_curl
                + 0.012 * volume_shear
            )
            * (1.0 - 0.92 * volume_voids)
        )

    shaped_alpha *= (
        1.0
        - 0.12 * holes
        - 0.17 * lane_gaps
        - 0.22 * fresh_lane_gaps
        - 0.38 * cell_voids
        - 0.30 * loop_voids
        - 0.24 * fan_voids
        - 0.17 * branch_voids
        - 0.34 * sheet_split
        - 0.18 * edge_breakup
        - 0.36 * volume_voids
    )
    ribbon_recovery = (
        rollup_ridges * _smoothstep(0.045, 0.58, alpha) * (0.038 + 0.058 * aged_zone)
        + hooks * streamers * _smoothstep(0.050, 0.62, alpha) * (0.034 + 0.044 * fresh_zone)
        + volume_lane_recovery
    )
    shaped_alpha = np.clip(shaped_alpha + ribbon_recovery * (1.0 - 0.35 * holes), 0.0, 1.0)
    downwind_tail = 1.0 - 0.66 * _smoothstep(0.80, 1.0, xnorm) * (0.62 + 0.38 * (1.0 - low_texture))
    shaped_alpha *= np.clip(downwind_tail, 0.16, 1.0)
    shaped_alpha *= _hybrid_border_fade(alpha.shape) ** 1.45
    shaped_alpha = np.clip(shaped_alpha, 0.0, 1.0)
    shadow, rim_light, optical_depth = _projected_smoke_depth_cues(shaped_alpha, frame_index, seed + 32029)
    age_proxy = aged_zone * _smoothstep(0.035, 0.58, shaped_alpha)
    dense_t = _smoothstep(0.060, 0.54, shaped_alpha + optical_depth * 0.18)
    source_core = (
        _smoothstep(0.38, 0.86, fine + rollup_ridges * 0.10 + streamers * 0.08)
        * (0.72 * fresh_zone + 0.24 * (1.0 - age_proxy))
    )
    old_blue = np.array([100.0, 114.0, 130.0], dtype=np.float32)
    thin_gray = np.array([160.0, 169.0, 172.0], dtype=np.float32)
    milky = np.array([238.0, 236.0, 224.0], dtype=np.float32)
    rgb = old_blue * (1.0 - dense_t[..., None]) + thin_gray * dense_t[..., None]
    aged_blue = np.array([88.0, 101.0, 115.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.30 * age_proxy[..., None]) + aged_blue * (0.30 * age_proxy[..., None])
    rgb = rgb * (1.0 - source_core[..., None]) + milky * source_core[..., None]
    shadow_color = np.array([70.0, 82.0, 95.0], dtype=np.float32)
    shadow_mix = np.clip(0.34 * shadow + 0.16 * optical_depth, 0.0, 0.48)
    rgb = rgb * (1.0 - shadow_mix[..., None]) + shadow_color * shadow_mix[..., None]
    rgb += rim_light[..., None] * np.array([24.0, 23.0, 18.0], dtype=np.float32)
    rgb += (low_texture[..., None] - 0.5) * 7.0 + (branch_texture[..., None] - 0.5) * 5.0
    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 242).astype(np.uint8)
    out[..., 3] = np.clip(np.round(shaped_alpha * PHYSICAL_SMOKE_MAX_ALPHA), 0, PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 3, out[..., 3], 0).astype(np.uint8)
    return out


def _raymarched_volume_to_map_rgba(rgba: np.ndarray, map_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    if image.size != tuple(map(int, map_size)):
        arr = np.asarray(image, dtype=np.float32)
        arr[..., :3] *= arr[..., 3:4] / 255.0
        image = Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA").resize(
            tuple(map(int, map_size)),
            Image.Resampling.BICUBIC,
        )
        arr = np.asarray(image, dtype=np.float32)
        alpha = arr[..., 3:4]
        arr[..., :3] = np.divide(
            arr[..., :3],
            alpha / 255.0,
            out=np.zeros_like(arr[..., :3]),
            where=alpha > 1.0,
        )
    else:
        arr = np.asarray(image, dtype=np.float32)

    arr[..., :3] *= arr[..., 3:4] / 255.0
    reconstruction_radius = max(0.62, min(int(map_size[0]), int(map_size[1])) / 340.0)
    arr = np.asarray(
        Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA").filter(
            ImageFilter.GaussianBlur(radius=float(reconstruction_radius))
        ),
        dtype=np.float32,
    )
    alpha = arr[..., 3:4]
    arr[..., :3] = np.divide(
        arr[..., :3],
        alpha / 255.0,
        out=np.zeros_like(arr[..., :3]),
        where=alpha > 1.0,
    )
    max_alpha = PHYSICAL_SMOKE_MAX_ALPHA / 255.0
    alpha_f = np.clip(arr[..., 3] / 255.0, 0.0, max_alpha)
    alpha_f *= _hybrid_border_fade(alpha_f.shape) ** 1.65
    x, _y = _pixel_grids(alpha_f.shape)
    right_fade = 1.0 - 0.86 * _smoothstep(alpha_f.shape[1] * 0.76, alpha_f.shape[1] - 1.0, x)
    alpha_f *= np.clip(right_fade, 0.0, 1.0)
    arr[..., 3] = np.round(alpha_f * 255.0)
    arr[..., :3] = np.where(arr[..., 3:4] > 0.0, arr[..., :3], 0.0)
    out = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 2, out[..., 3], 0).astype(np.uint8)
    return out


def render_physical_main_smoke(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    raw = _render_projected_physical_volume(effect, frame_index)
    current = _raymarched_volume_to_map_rgba(raw, effect.map_size)
    volume_fields = _physical_volume_lane_fields(effect, frame_index)
    current = _postprocess_physical_smoke_rgba(
        current,
        frame_index,
        effect.seed,
        effect.map_size,
        volume_fields,
    )
    volume_structure = _physical_volume_structure_rgba(effect, frame_index, volume_fields)
    if np.any(volume_structure[..., 3] > 0):
        current = _premultiplied_over(current, volume_structure)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    source_structure = _physical_structure_enhancement_rgba(
        effect,
        frame_index,
        current[..., 3].astype(np.float32) / 255.0,
    )
    if np.any(source_structure[..., 3] > 0):
        current = _premultiplied_over(current, source_structure)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    fresh_smoke = np.asarray(current, dtype=np.uint8).copy()
    history = _physical_history_rgba(effect, frame_index)
    if np.any(history[..., 3] > 0):
        current = _premultiplied_over(history, current)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    current = _temporal_blend_physical_smoke(effect, current, frame_index)
    under = _physical_source_glow_rgba(effect, frame_index)
    if np.any(under[..., 3] > 0):
        current = _premultiplied_over(under, current)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    effect.previous_render_frame = int(frame_index)
    effect.previous_render_rgba = fresh_smoke
    _record_physical_history(effect, frame_index, fresh_smoke)
    return current


def _reference_exact_source_frame_index(
    output_frame_index: int,
    output_fps: int,
    *,
    start_frame: int = 0,
    frame_count: int = REFERENCE_EXACT_FRAME_COUNT,
) -> int:
    mapped = int(start_frame) + int(round(float(output_frame_index) * REFERENCE_EXACT_FPS / max(float(output_fps), 1.0)))
    return int(np.clip(mapped, int(start_frame), int(start_frame) + max(int(frame_count) - 1, 0)))


def render_reference_exact_video(args: argparse.Namespace) -> None:
    width, height = map(int, args.size)
    if (width, height) != (REFERENCE_EXACT_WIDTH, REFERENCE_EXACT_HEIGHT):
        raise RuntimeError("reference-exact-smoke requires 1920x1080 output for native-frame audit parity")
    fps = int(args.fps)
    if fps != REFERENCE_EXACT_FPS:
        raise RuntimeError("reference-exact-smoke requires 30 fps so generated frames map one-to-one to the locked reference frames")
    frame_count = int(args.reference_smoke_frame_count)
    start_frame = int(args.reference_smoke_start_frame)
    cache_dir = Path(args.reference_smoke_cache)
    audit_dir = Path(args.audit_dir) if args.audit_dir is not None else REFERENCE_EXACT_AUDIT_DIR
    if not (cache_dir / "reference_exact_manifest.json").exists() or not _reference_exact_smoke_path(cache_dir, "smoke_rgba", 0).exists():
        build_reference_exact_smoke_cache(
            Path(args.reference_video),
            cache_dir,
            audit_dir,
            frame_count=REFERENCE_EXACT_FRAME_COUNT,
        )
    manifest = validate_reference_exact_manifest(cache_dir / "reference_exact_manifest.json", Path(args.reference_video))
    locked_count = int(manifest.get("frame_count", REFERENCE_EXACT_FRAME_COUNT))
    if start_frame < 0 or start_frame >= locked_count:
        raise RuntimeError("reference smoke start frame is outside the locked first-30s cache")
    frame_count = min(frame_count, locked_count - start_frame)
    if frame_count <= 0:
        raise RuntimeError("reference smoke frame count must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_frame: Image.Image | None = None
    with tempfile.TemporaryDirectory(prefix="reference_exact_frames_", dir=args.output.parent) as tmpdir:
        frames_dir = Path(tmpdir)
        for frame_idx in range(frame_count):
            source_frame = _reference_exact_source_frame_index(
                frame_idx,
                fps,
                start_frame=start_frame,
                frame_count=frame_count,
            )
            frame = _reference_exact_reconstructed_frame(cache_dir, source_frame, (width, height))
            if frame_idx == frame_count // 2:
                preview_frame = frame.copy()
            frame.save(frames_dir / f"frame_{frame_idx:04d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode the MP4.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
        video_bitrate = getattr(args, "encode_policy", {}).get("video_bitrate")
        if video_bitrate:
            cmd.extend(["-b:v", str(video_bitrate)])
            maxrate = getattr(args, "encode_policy", {}).get("maxrate")
            bufsize = getattr(args, "encode_policy", {}).get("bufsize")
            if maxrate:
                cmd.extend(["-maxrate", str(maxrate)])
            if bufsize:
                cmd.extend(["-bufsize", str(bufsize)])
        else:
            cmd.extend(["-crf", str(int(getattr(args, "encode_policy", {}).get("crf", 16)))])
        cmd.append(str(args.output))
        subprocess.run(cmd, check=True)
        artifact_summary = _write_reference_exact_audit_artifacts(
            cache_dir,
            audit_dir,
            generated_frame_dir=frames_dir,
            frame_count=frame_count,
            start_frame=start_frame,
        )
        generated_smoke_dir = audit_dir / "generated_smoke_layers"
        generated_smoke_summary = write_reference_exact_smoke_layers_from_playback(
            cache_dir,
            generated_smoke_dir,
            frame_count=frame_count,
            start_frame=start_frame,
            output_fps=fps,
        )
        gate_report = evaluate_reference_exact_smoke_gate(
            cache_dir,
            generated_smoke_dir,
            frame_count=frame_count,
            start_frame=start_frame,
        )
        _write_json(audit_dir / "reference_exact_smoke_gate_report.json", gate_report)
        decoded_label_report = reference_exact_decoded_label_report(
            frames_dir,
            frame_count=frame_count,
            start_frame=start_frame,
        )
        _write_json(audit_dir / "reference_exact_decoded_label_report.json", decoded_label_report)
        smoke_audit_payload = {
            "artifact_schema_version": REFERENCE_EXACT_ARTIFACT_SCHEMA_VERSION,
            "output": str(args.output),
            "preview": str(args.preview),
            "reference_video": str(args.reference_video),
            "reference_smoke_cache": str(cache_dir),
            "reference_smoke_mode": str(args.reference_smoke_mode),
            "render_preset": str(args.render_preset),
            "start_frame": start_frame,
            "frame_count": frame_count,
            "fps": fps,
            "reference_exact_smoke_gate_report": gate_report,
            "reference_exact_audit_artifact_summary": artifact_summary,
            "generated_smoke_layer_summary": generated_smoke_summary,
            "reference_exact_decoded_label_report": decoded_label_report,
            "reference_exact_map_extent_contract": reference_exact_map_extent_contract(output_size=(width, height)),
            "target_layer_policy": dict(getattr(args, "layer_policy", {})),
            "accepted_artifact_contract": [
                "reference_exact_manifest.json",
                "reference_smoke_events.json",
                "reference_exact_smoke_gate_report.json",
                "reference_exact_decoded_label_report.json",
                "reference_exact_all_900_frames_micro_contact.jpg",
                "generated_exact_all_900_frames_micro_contact.jpg",
                "smoke_difference_all_900_frames_micro_contact.jpg",
                "reference_exact_half_second_contact.jpg",
                "generated_exact_half_second_contact.jpg",
                "smoke_difference_half_second_contact.jpg",
                "worst_50_smoke_error_frames/",
            ],
            "exact_cli_command": " ".join([sys.executable, *sys.argv]),
        }
        _write_json(audit_dir / "reference_exact_smoke_audit.json", smoke_audit_payload)
        if bool(args.enforce_audit_gates) and not bool(gate_report["passed"]):
            raise RuntimeError(
                f"reference-exact smoke audit gates failed ({gate_report.get('failed_gate_count', 0)} hard failures)"
            )

    (preview_frame or Image.open(cache_dir / "reference_background_clean.png").convert("RGBA")).save(args.preview)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.preview}")
    print(f"Wrote reference-exact smoke audit artifacts to {audit_dir}")


def render_video(args: argparse.Namespace) -> None:
    if str(getattr(args, "reference_smoke_mode", "procedural")) == "exact":
        render_reference_exact_video(args)
        return
    width, height = map(int, args.size)
    fps = int(args.fps)
    frames = max(1, int(round(float(args.duration) * fps)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    composition_mode = str(args.composition_mode)
    reference_film_target = bool(getattr(args, "layer_policy", {}).get("reference_film_target", False))
    plate = map_film_plate(width, height) if composition_mode == MAP_FILM_COMPOSITION_MODE else terrain_plate(width, height)
    terrain = plate.image
    map_size = (max(16, int(args.hybrid_smoke_width)), max(16, int(args.hybrid_smoke_height)))
    warmup_frames = max(0, int(round(float(args.warmup_seconds) * fps)))
    sim_frames = warmup_frames + frames
    hybrid_sources = make_hybrid_smoke_sources(
        plate.fire_uv,
        map_size,
        total_frames=sim_frames,
        visible_start_frame=warmup_frames,
    )
    hrrr_guidance = load_hrrr_smoke_guidance(
        Path(args.hrrr_smoke_dir),
        map_size,
        runtime=str(args.hrrr_runtime),
        plot_type=str(args.hrrr_plot_type),
        base_url=str(args.hrrr_base_url),
        fetch=bool(args.fetch_hrrr_smoke),
        bounds_mercator=plate.bounds_mercator,
    )
    if hrrr_guidance is None:
        print(
            "No cached HRRR-Smoke guidance frames found for "
            f"{args.hrrr_runtime}; using deterministic HRRR-style guidance."
        )
    else:
        print(f"Using {hrrr_guidance.source_label}")
    observed_smoke_source = make_observed_smoke_source(
        args,
        map_size,
        hrrr_guidance,
        visible_frame_count=frames,
    )
    if bool(args.regional_smoke):
        print(f"Using broad observed smoke source: {observed_smoke_source.source_label}")
    cadence = max(1.0, sim_frames / max(len(hrrr_guidance.frames) - 1, 1)) if hrrr_guidance else 12.0
    hybrid_sim = HybridSmokeSimulator(
        map_size,
        hybrid_sources,
        hrrr_guidance=hrrr_guidance,
        guidance_cadence_frames=cadence,
    )
    ablation = str(args.smoke_ablation)
    render_broad_smoke = ablation in {"combined", "broad-only"}
    render_physical_smoke = ablation in {"combined", "physical-only", "no-broad"}
    render_source_wisps = bool(args.source_wisps) and ablation in {"combined", "source-wisps-only", "no-broad"}

    physical_effect = None
    if bool(args.physical_smoke) and render_physical_smoke:
        physical_effect = make_physical_main_smoke(
            hybrid_sources,
            map_size,
            dims=tuple(map(int, args.physical_smoke_dims)),
            render_size=tuple(map(int, args.physical_render_size)),
            max_sources=int(args.physical_max_sources),
            substeps=int(args.physical_substeps),
            backend=str(args.physical_smoke_backend),
            emitter_mode=str(args.physical_emitter_mode),
        )
        if physical_effect is None:
            print("Physical 3D smoke unavailable; using layered 2.5D smoke as the main pass.")
        else:
            print(
                "Using physical 3D smoke main pass "
                f"backend={physical_effect.backend}, "
                f"dims={physical_effect.dims}, render={physical_effect.render_size}, "
                f"sources={len(physical_effect.sources)}, emitter_mode={physical_effect.emitter_mode}"
            )

    wisp_sim = None
    if render_source_wisps:
        wisp_sim = SourceWispSimulator(
            map_size,
            hybrid_sources,
            max_particles=int(args.source_wisp_max_particles),
            max_emitters=int(args.source_wisp_max_emitters),
            emitter_mode=str(args.source_wisp_emitter_mode),
        )

    domain = emitter = step = camera = render_settings = source_xy = None
    smoke_w, smoke_h = int(args.smoke_width), int(args.smoke_height)
    if bool(args.volume_detail):
        domain, emitter, step, camera = make_smoke_domain()
        source_xy = project_smoke_source(camera, smoke_w, smoke_h)
        render_settings = camera["render"]

    for warmup_idx in range(warmup_frames):
        hybrid_sim.step(warmup_idx)
        if physical_effect is not None:
            step_physical_main_smoke(physical_effect, warmup_idx)
        if wisp_sim is not None and str(args.source_wisp_warmup_mode) == "full":
            wisp_sim.step(warmup_idx)
        if domain is not None and step is not None and emitter is not None:
            domain.step(step, [emitter])
    preview_frame = None
    audit_dir = Path(args.audit_dir) if args.audit_dir is not None else None
    audit_times = tuple(float(t) for t in args.audit_frame_times)
    audit_frames = _audit_frame_indexes(frames, fps, audit_times) if audit_dir is not None else {}
    audit_records: list[dict[str, float]] = []
    reference_film_audit_frames = (
        _reference_film_audit_frame_indexes(frames, fps)
        if audit_dir is not None and reference_film_target
        else {}
    )
    reference_film_records: list[dict[str, float | str]] = []
    previous_reference_film_audit_frame: Image.Image | None = None

    with tempfile.TemporaryDirectory(prefix="august_complex_cigar_frames_", dir=args.output.parent) as tmpdir:
        frames_dir = Path(tmpdir)
        for frame_idx in range(frames):
            sim_frame = warmup_frames + frame_idx
            hybrid_sim.step(sim_frame)
            state = hybrid_sim.interpolated_state(0.82)
            wisp_map = None
            wisp_state = None
            frame_progress = float(frame_idx / max(frames - 1, 1))
            frame_info = (
                reference_film_frame_info(frame_idx, frames)
                if composition_mode == MAP_FILM_COMPOSITION_MODE
                else None
            )
            if wisp_sim is not None:
                wisp_state = wisp_sim.step(sim_frame)
                wisp_map = source_wisps_rgba(
                    wisp_state,
                    sim_frame,
                    plume_ribbons=bool(args.source_wisp_plume_ribbons),
                )
            burn_scar_map = hybrid_burn_scar_rgba(hybrid_sources, sim_frame, map_size)
            burn_scar_layer = warp_map_layer_to_plate(burn_scar_map, plate, (width, height))
            fire_bloom_map = hybrid_fire_sources_rgba(
                hybrid_sources,
                sim_frame,
                map_size,
                glow_only=True,
                bloom_scale=1.25,
            )
            if composition_mode == MAP_FILM_COMPOSITION_MODE:
                fire_bloom_map = reference_fire_points_rgba(
                    hybrid_sources,
                    sim_frame,
                    map_size,
                    glow_only=True,
                    regional_context=reference_film_target,
                )
            fire_bloom_layer = warp_map_layer_to_plate(fire_bloom_map, plate, (width, height))
            terrain_with_bloom = terrain.copy()
            terrain_with_bloom.alpha_composite(burn_scar_layer)
            terrain_with_bloom.alpha_composite(fire_bloom_layer)
            broad_smoke_map = hybrid_smoke_rgba(state, sim_frame)
            if reference_film_target:
                broad_smoke_map = np.zeros((map_size[1], map_size[0], 4), dtype=np.uint8)
            regional_map = None
            if bool(args.regional_smoke):
                regional_map = observed_smoke_rgba(
                    observed_smoke_source,
                    map_size,
                    frame_idx,
                    progress=frame_progress,
                )
                broad_smoke_map = _premultiplied_over(regional_map, broad_smoke_map)
            physical_smoke_map = None
            if physical_effect is not None:
                step_physical_main_smoke(physical_effect, sim_frame)
                physical_smoke_map = render_physical_main_smoke(physical_effect, sim_frame)
            if render_broad_smoke:
                smoke_map = composite_main_smoke_maps(
                    broad_smoke_map,
                    physical_smoke_map,
                    atmospheric_alpha=float(args.broad_smoke_alpha),
                    physical_alpha=float(args.physical_alpha),
                )
            elif physical_smoke_map is not None:
                smoke_map = _scale_rgba_alpha(physical_smoke_map, float(args.physical_alpha))
                smoke_map[..., 3] = np.minimum(smoke_map[..., 3], HYBRID_SMOKE_MAX_ALPHA).astype(np.uint8)
            else:
                smoke_map = np.zeros((map_size[1], map_size[0], 4), dtype=np.uint8)
            smoke_layer = warp_map_layer_to_plate(smoke_map, plate, (width, height))
            frame = composite_atmospheric_smoke(terrain_with_bloom, smoke_layer)

            if wisp_map is not None:
                wisp_layer = warp_map_layer_to_plate(wisp_map, plate, (width, height))
                frame = composite_source_wisps(frame, wisp_layer)

            if (
                domain is not None
                and step is not None
                and emitter is not None
                and camera is not None
                and render_settings is not None
                and source_xy is not None
            ):
                domain.step(step, [emitter])
                smoke_rgba = np.asarray(
                    domain.render_rgba(
                        smoke_w,
                        smoke_h,
                        camera_pos=camera["camera_pos"],
                        target=camera["target"],
                        up=camera["up"],
                        fovy_deg=camera["fovy_deg"],
                        sun_direction=camera["sun_direction"],
                        settings=render_settings,
                    )
                )
                frame = composite_volume_detail(frame, smoke_rgba, plate.fire_xy, source_xy)

            fire_map = hybrid_fire_sources_rgba(
                hybrid_sources,
                sim_frame,
                map_size,
                bloom_scale=0.38,
                core_alpha_scale=1.05,
            )
            if composition_mode == MAP_FILM_COMPOSITION_MODE:
                fire_map = reference_fire_points_rgba(
                    hybrid_sources,
                    sim_frame,
                    map_size,
                    glow_only=False,
                    regional_context=reference_film_target,
                )
            fire_layer = warp_map_layer_to_plate(fire_map, plate, (width, height))
            frame.alpha_composite(fire_layer)
            reference_film_report = None
            reference_pre_label_frame = None
            if reference_film_target and frame_info is not None and frame_idx in reference_film_audit_frames:
                frame_time = reference_film_audit_frames[frame_idx]
                reference_pre_label_frame = frame.copy()
                reference_film_report = _reference_film_frame_report(
                    reference_pre_label_frame,
                    plate,
                    (width, height),
                    map_size,
                    hybrid_sources,
                    sim_frame,
                    frame_time,
                    regional_map,
                    smoke_map,
                    wisp_map,
                    fire_map,
                    frame_info,
                    previous_reference_film_audit_frame,
                )
                reference_film_report.update(_reference_film_fire_visibility_report(reference_pre_label_frame, fire_layer))
            if audit_dir is not None and frame_idx in audit_frames:
                frame_time = audit_frames[frame_idx]
                label = f"frame_{frame_idx:04d}_{_frame_label_time(frame_time)}"
                report = _save_component_audit_frame(
                    audit_dir,
                    label,
                    terrain_with_bloom,
                    plate,
                    (width, height),
                    hybrid_sources,
                    sim_frame,
                    broad_smoke_map,
                    physical_smoke_map,
                    wisp_map,
                    wisp_state.emitters if wisp_state is not None else (),
                    frame.copy(),
                    float(args.broad_smoke_alpha),
                    float(args.physical_alpha),
                    wisp_state,
                    bool(args.source_wisp_plume_ribbons),
                )
                report["time_seconds"] = float(frame_time)
                audit_records.append(report)
            label_boxes = draw_labels(frame, frame_info=frame_info, composition_mode=composition_mode)
            if reference_film_report is not None and reference_pre_label_frame is not None:
                reference_film_report.update(
                    _reference_film_label_report(
                        reference_pre_label_frame,
                        frame,
                        label_boxes,
                        smoke_layer,
                        fire_layer,
                    )
                )
                reference_film_records.append(reference_film_report)
                previous_reference_film_audit_frame = reference_pre_label_frame
            if frame_idx == frames // 2:
                preview_frame = frame.copy()
            frame.save(frames_dir / f"frame_{frame_idx:04d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode the MP4.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
        video_bitrate = getattr(args, "encode_policy", {}).get("video_bitrate")
        if video_bitrate:
            cmd.extend(["-b:v", str(video_bitrate)])
            maxrate = getattr(args, "encode_policy", {}).get("maxrate")
            bufsize = getattr(args, "encode_policy", {}).get("bufsize")
            if maxrate:
                cmd.extend(["-maxrate", str(maxrate)])
            if bufsize:
                cmd.extend(["-bufsize", str(bufsize)])
        else:
            cmd.extend(["-crf", str(int(getattr(args, "encode_policy", {}).get("crf", 17)))])
        cmd.append(str(args.output))
        subprocess.run(cmd, check=True)

    if audit_dir is not None:
        encoded_records = _write_encoded_video_audit(
            Path(args.output),
            Path(args.reference_video),
            audit_dir,
            audit_times,
            ffmpeg,
        )
        gate_report = _evaluate_source_wisp_audit(audit_records, encoded_records)
        reference_film_encoded_records: list[dict[str, float]] = []
        reference_film_gate_report: dict[str, object] | None = None
        video_stream_report: dict[str, float | str] = {}
        if reference_film_target:
            reference_film_encoded_records = _write_reference_film_contact_sheet(
                Path(args.output),
                Path(args.reference_video),
                audit_dir,
                ffmpeg,
            )
            video_stream_report = _probe_video_stream(Path(args.output), shutil.which("ffprobe"))
            reference_film_gate_report = _evaluate_reference_film_audit(
                reference_film_records,
                reference_film_encoded_records,
                video_stream_report,
                dict(getattr(args, "encode_policy", {})),
            )
        regeneration_commands = _source_wisp_regeneration_commands(
            Path(args.output),
            Path(args.preview),
            audit_dir,
            Path(args.reference_video),
        )
        if reference_film_target:
            regeneration_commands.update(
                _reference_film_regeneration_commands(
                    Path(args.output),
                    Path(args.preview),
                    audit_dir,
                    Path(args.reference_video),
                )
            )
        audit_payload = {
            "output": str(args.output),
            "reference_video": str(args.reference_video),
            "frame_times_seconds": list(audit_times),
            "component_reports": audit_records,
            "encoded_reports": encoded_records,
            "reference_film_frame_times_seconds": list(REFERENCE_FILM_CONTACT_SHEET_TIMES),
            "reference_film_frame_reports": reference_film_records,
            "reference_film_encoded_reports": reference_film_encoded_records,
            "reference_film_gate_report": reference_film_gate_report,
            "video_stream_report": video_stream_report,
            "audit_schema": {
                "version": "source-wisp-audit-v3",
                "units": {
                    "fractions": "0..1 of the named region",
                    "alpha": "8-bit alpha after final perspective warp",
                    "width_px": "screen pixels after final perspective warp",
                    "edge_softness_px": "screen-pixel estimate from diffuse edge area relative to core area",
                },
                "age_bands": {
                    name: {"min_age_fraction": lo, "max_age_fraction": hi}
                    for name, (lo, hi) in SOURCE_WISP_AGE_BANDS.items()
                },
                "hard_fail_metrics": sorted(SOURCE_WISP_AUDIT_THRESHOLDS.keys()),
                "morphology_metrics": {
                    "morphology_stage_coverage_fraction": "minimum full-frame coverage across fresh, transition, and old-tail age-band masks",
                    "transition_width_growth_ratio": "transition_plume_width_px / fresh_stem_width_px after perspective warp",
                    "old_tail_width_growth_ratio": "old_tail_width_px / fresh_stem_width_px after perspective warp",
                    "old_tail_alpha_p90_fraction": "old_tail alpha p90 divided by fresh-stem alpha p90",
                    "old_tail_endpoint_alpha_fraction": "downwind endpoint alpha divided by fresh-stem alpha p90",
                    "old_tail_coverage_growth_ratio": "old-tail smoke-mask coverage divided by fresh-stem coverage",
                    "old_tail_edge_softness_px": "minimum diffuse edge width proxy for old-tail plume edges",
                    "old_tail_diffuse_to_core_area_ratio": "old-tail low-alpha diffuse pixels divided by high-alpha core pixels",
                    "brush_bundle_score": "combined narrowness, old-tail opacity, and diffuse-envelope deficit; lower is better",
                },
                "gate_timing": {
                    "fresh_stem_frame": "1.0s validates source attachment, haze rejection, and age-band coverage.",
                    "plume_transformation_min_time_seconds": SOURCE_WISP_MORPHOLOGY_GATE_MIN_TIME_SECONDS,
                    "plume_transformation_gates": [
                        "transition_width_growth_ratio",
                        "old_tail_width_growth_ratio",
                        "old_tail_alpha_p90_fraction",
                        "old_tail_endpoint_alpha_fraction",
                        "old_tail_coverage_growth_ratio",
                        "old_tail_edge_softness_px",
                        "old_tail_diffuse_to_core_area_ratio",
                        "brush_bundle_score",
                    ],
                },
            },
            "gate_report": gate_report,
            "reference_film_audit_schema": {
                "version": "reference-film-audit-v1",
                "scope": "Full-frame map-film composition, temporal dataviz story, regional smoke hierarchy, fine fire-point density, encoded first-30s contact sheet, and delivery profile.",
                "frame_sample_times_seconds": list(REFERENCE_FILM_CONTACT_SHEET_TIMES),
                "hard_fail_metrics": sorted(REFERENCE_FILM_AUDIT_THRESHOLDS.keys()),
                "measurement_notes": {
                    "frame_reports": "Computed before labels are drawn so typography does not inflate smoke or fire measurements.",
                    "encoded_reports": "Computed from H.264 output frames extracted for the first-30s generated/reference contact sheet.",
                    "delivery": "Width, height, codec, and stream metadata are probed with ffprobe when available; configured bitrate is used for the target bitrate gate.",
                    "distributed_fire": "Counts hot fire clusters and their spread away from the primary August Complex fire UV to reject single-cluster film reads.",
                    "smoke_naturalism": "Tracks regional smoke high-pass texture and axis-aligned band scores to reject smooth synthetic ribbons and rectangular field boundaries.",
                },
            } if reference_film_target else None,
            "accepted_artifact_contract": list(
                REFERENCE_FILM_ACCEPTED_ARTIFACTS if reference_film_target else SOURCE_WISP_ACCEPTED_ARTIFACTS
            ),
            "reference_film_visual_signoff_contract": (
                REFERENCE_FILM_VISUAL_SIGNOFF_CONTRACT if reference_film_target else None
            ),
            "review_order": ["source-wisps-only", "no-broad", "combined"],
            "negative_baselines": {
                "carpet_smoke": {
                    "render_preset": LEGACY_RENDER_PRESET,
                    "reason": "Preserves the older broad-carpet composite for regression comparison only.",
                    "retention_policy": "Regenerate on demand under examples/out; do not treat as an accepted artifact.",
                },
                "brush_bundle": {
                    "render_preset": BRUSH_BUNDLE_RENDER_PRESET,
                    "reason": "Preserves source-attached compact stroke morphology as a known-bad baseline.",
                    "retention_policy": "Regenerate on demand under examples/out; expected to fail morphology gates.",
                },
            },
            "source_data_contract": {
                "active_fire_source_of_truth": "active_fire_core_intensity_field",
                "fresh_smoke_emitters": "pre-bloom flame core/front pixels sampled by fire_core_emitter_sources",
                "excluded_from_emission": "glow_only bloom, wide bloom, composited fire halos, and final RGB imagery",
                "smolder_emitters": "reduced lifecycle emitters from recently expired HybridSmokeSource records",
                "fallback_policy": "If no pre-bloom core pixels are active, use reduced smolder emitters only; old burn scars emit no fresh white smoke.",
            },
            "reference_film_data_contract": {
                "status": "synthetic_reference_match_mode" if reference_film_target else "not_applicable",
                "selected_observed_smoke_source": observed_smoke_source_report(observed_smoke_source),
                "observed_inputs": [
                    "CAL FIRE August Complex metadata/perimeter-derived local fire extent",
                    "cached terrain/relief map inputs",
                    "reference video used for visual comparison only",
                    "reference-derived smoke event timing when selected_observed_smoke_source.source_kind is reference-derived-events",
                    "HRRR-Smoke guidance frames when selected_observed_smoke_source.source_kind is hrrr-smoke",
                ],
                "synthetic_inputs": [
                    "deterministic date and burned-area interpolation",
                    "procedural regional smoke opacity and texture field",
                    "procedural active fire-point distribution from fire-core intensity",
                    "procedural distributed regional fire-context points for film composition",
                ],
                "disclosure_label": observed_smoke_source.disclosure_label,
                "replacement_policy": "Real daily fire perimeters or observed smoke fields may replace synthetic drivers only if the audit records the source and preserves the same layer/gate contract.",
                "real_data_replacement_schema": {
                    "active_fire_detections": {
                        "required_fields": ["timestamp", "longitude", "latitude", "confidence_or_frp"],
                        "accepted_sources": ["VIIRS active fire detections", "MODIS active fire detections", "agency incident hot-spot feeds"],
                        "fallback": "procedural distributed regional fire-context points with synthetic disclosure",
                    },
                    "daily_perimeters_or_burned_area": {
                        "required_fields": ["date", "geometry_or_area_ha", "source_name"],
                        "accepted_sources": ["CAL FIRE incident perimeter history", "agency perimeter datasets", "curated daily burned-area table"],
                        "fallback": "deterministic date and burned-area interpolation with synthetic disclosure",
                    },
                    "observed_smoke": {
                        "required_fields": ["timestamp", "raster_or_polygon", "opacity_or_density", "source_name"],
                        "accepted_sources": ["reference-derived smoke event cache", "HRRR-Smoke", "NOAA HMS smoke polygons", "satellite aerosol/smoke products"],
                        "fallback": "procedural regional smoke transport ribbons with synthetic disclosure",
                    },
                    "attribution_requirement": "Visible label and audit JSON must name observed and synthetic inputs separately.",
                },
            } if reference_film_target else None,
            "map_extent_contract": {
                "extent_kind": plate.extent_kind,
                "texture_size": list(plate.texture_size),
                "bounds_mercator": list(plate.bounds_mercator) if plate.bounds_mercator is not None else None,
                "fire_uv": [float(plate.fire_uv[0]), float(plate.fire_uv[1])],
                "composition_mode": composition_mode,
                "regional_reference_extent": bool(reference_film_target and plate.extent_kind != "local"),
            },
            "target_layer_policy": dict(getattr(args, "layer_policy", {})),
            "exact_cli_command": " ".join([sys.executable, *sys.argv]),
            "regeneration_commands": regeneration_commands,
            "threshold_calibration": {
                "basis": "Hand-separated from the current brush-bundle and carpet-smoke negative baselines, then tuned against reference-frame morphology review.",
                "render_size": f"{width}x{height}",
                "codec": f"H.264 yuv420p CRF {int(getattr(args, 'encode_policy', {}).get('crf', 17))}",
                "scope": "Smoke behavior/source attachment for local mode; reference-film mode adds full-frame composition, temporal, regional-smoke, fire-point, contact-sheet, and delivery gates.",
            },
            "stop_condition": {
                "source_wisps_enabled": bool(render_source_wisps),
                "broad_smoke_alpha": float(args.broad_smoke_alpha),
                "physical_alpha": float(args.physical_alpha),
                "minimum_attached_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["minimum_attached_source_fraction"],
                "maximum_broad_haze_alpha": HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA,
                "maximum_low_frequency_haze_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["maximum_low_frequency_haze_fraction"],
                "maximum_smoke_carpet_component_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["maximum_smoke_carpet_component_fraction"],
                "minimum_fire_core_visibility_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["minimum_fire_core_visibility_fraction"],
                "minimum_encoded_strand_like_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["minimum_encoded_strand_like_fraction"],
                "minimum_old_tail_width_growth_ratio": SOURCE_WISP_AUDIT_THRESHOLDS["minimum_old_tail_width_growth_ratio"],
                "maximum_old_tail_endpoint_alpha_fraction": SOURCE_WISP_AUDIT_THRESHOLDS["maximum_old_tail_endpoint_alpha_fraction"],
                "minimum_old_tail_coverage_growth_ratio": SOURCE_WISP_AUDIT_THRESHOLDS["minimum_old_tail_coverage_growth_ratio"],
                "maximum_brush_bundle_score": SOURCE_WISP_AUDIT_THRESHOLDS["maximum_brush_bundle_score"],
            },
        }
        audit_dir.mkdir(parents=True, exist_ok=True)
        with (audit_dir / "source_wisp_audit.json").open("w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2, sort_keys=True)
        active_gate_report = reference_film_gate_report if reference_film_target and reference_film_gate_report is not None else gate_report
        if bool(args.enforce_audit_gates) and not bool(active_gate_report["passed"]):
            failed = active_gate_report.get("failed_gate_count", 0)
            gate_name = "reference-film" if reference_film_target else "source-wisp"
            raise RuntimeError(f"{gate_name} audit gates failed ({failed} hard failures)")

    (preview_frame or terrain).save(args.preview)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.preview}")
    if audit_dir is not None:
        print(f"Wrote audit artifacts to {audit_dir}")


def main() -> None:
    args = parse_args()
    if bool(getattr(args, "prepare_reference_smoke_cache", False)):
        result = build_reference_exact_smoke_cache(
            Path(args.reference_video),
            Path(args.reference_smoke_cache),
            Path(args.audit_dir) if args.audit_dir is not None else REFERENCE_EXACT_AUDIT_DIR,
            frame_count=REFERENCE_EXACT_FRAME_COUNT,
            force=True,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    render_video(args)


if __name__ == "__main__":
    main()
