# python/forge3d/__init__.py
# Public Python API for forge3d terrain renderer
"""
forge3d - GPU-accelerated terrain rendering library.

Core API:
    open_viewer_async   - Launch the IPC-controlled interactive viewer
    open_viewer         - Launch the blocking interactive viewer
    TerrainRenderer     - Native GPU terrain renderer
    Renderer            - Fallback CPU renderer
    
Configuration:
    TerrainRenderParams - Terrain rendering parameters
    RendererConfig      - Renderer configuration
    
Utilities:
    numpy_to_png        - Save numpy array as PNG
    png_to_numpy        - Load PNG as numpy array
    has_gpu             - Check GPU availability
"""

__version__ = "1.30.0"
version = __version__

import numpy as np

from ._png import load_png_rgba as _load_png_rgba
from ._png import save_png as _save_png

# -----------------------------------------------------------------------------
# Native module loading
# -----------------------------------------------------------------------------
from ._native import (
    get_native_module as _get_native_module,
)
from ._gpu import (
    enumerate_adapters,
    device_probe,
    has_gpu,
    get_device,
)
from .mem import (
    memory_metrics,
    budget_remaining,
    utilization_ratio,
    override_memory_limit,
)

_NATIVE_MODULE = _get_native_module()

# -----------------------------------------------------------------------------
# Native exports (when available)
# -----------------------------------------------------------------------------
if _NATIVE_MODULE is not None:
    for _name in (
        "Scene",
        "Session",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "TerrainRenderParams",
        "TerrainRenderer",
        "Frame",
        "AovFrame",
        "HdrFrame",
        "OfflineBatchResult",
        "OfflineMetrics",
        "Light",
        "Atmosphere",
        "open_viewer",
        "open_terrain_viewer",
        "PickResult",  # Feature B: Picking system (Plan 1)
        "TerrainQueryResult",  # Feature B: Plan 2
        "SelectionStyle",  # Feature B: Plan 2
        "RichPickResult",  # Feature B: Plan 3
        "HighlightStyle",  # Feature B: Plan 3
        "LassoState",  # Feature B: Plan 3
        "HeightfieldHit",  # Feature B: Plan 3
        "CameraKeyframe",  # Feature C: Camera animation keyframe editing
        "CameraAnimation",  # Feature C: Camera animation (Plan 1 MVP)
        "CameraState",  # Feature C: Camera animation (Plan 1 MVP)
        "SunPosition",  # P0.3/M2: Sun ephemeris
        "sun_position",  # P0.3/M2: Sun ephemeris function
        "sun_position_utc",  # P0.3/M2: Sun ephemeris function (components)
        "set_point_shape_mode",
        "set_point_lod_threshold",
        "is_weighted_oit_available",
        "vector_oit_and_pick_demo",
        "vector_render_oit_py",
        "vector_render_pick_map_py",
        "vector_render_oit_and_pick_py",
        "vector_render_polygons_fill_py",
        "ClipmapConfig",  # P2.1/M5: Clipmap terrain
        "ClipmapMesh",  # P2.1/M5: Clipmap terrain
        "clipmap_generate_py",  # P2.1/M5: Clipmap generation function
        "calculate_triangle_reduction_py",  # P2.1/M5: Triangle reduction calculation
    ):
        if hasattr(_NATIVE_MODULE, _name):
            globals()[_name] = getattr(_NATIVE_MODULE, _name)

# -----------------------------------------------------------------------------
# Colormaps
# -----------------------------------------------------------------------------
from .colormaps import (
    get as get_colormap,
    available as available_colormaps,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from .config import RendererConfig, load_renderer_config
from .terrain_params import (
    TerrainRenderParams as TerrainRenderParamsConfig,
    LightSettings,
    IblSettings,
    ShadowSettings,
    FogSettings,
    ReflectionSettings,
    HeightAoSettings,
    SunVisibilitySettings,
    ProbeSettings,
    ReflectionProbeSettings,
    DetailSettings,
    MaterialNoiseSettings,
    MaterialLayerSettings,
    PomSettings,
    TriplanarSettings,
    LodSettings,
    SamplingSettings,
    ClampSettings,
    DenoiseSettings,
    OfflineQualitySettings,
    VTLayerFamily,
    TerrainVTSettings,
    validate_terrain_vt_support,
)
from .offline import OfflineProgress, OfflineResult, render_offline
from .denoise_oidn import oidn_available, oidn_denoise
from . import presets
from . import animation
from . import gis
from . import camera_rigs

# -----------------------------------------------------------------------------
# Core rendering API
# -----------------------------------------------------------------------------
from .path_tracing import ExperimentalSyntheticOutput, PathTracer, make_camera

# -----------------------------------------------------------------------------
# Interactive Viewer API
# -----------------------------------------------------------------------------
from .viewer import LabelBatchResult, ViewerHandle, open_viewer, open_viewer_async
from . import viewer_ipc, colors, interactive, datasets, widgets, smoke
from .datasets import (
    available as available_datasets,
    bundled as bundled_datasets,
    dataset_info,
    fetch as fetch_dataset,
    fetch_cityjson,
    fetch_copc,
    fetch_dem,
    list_datasets,
    mini_dem,
    mini_dem_path,
    remote as remote_datasets,
    sample_boundaries,
    sample_boundaries_path,
)
from .widgets import ViewerWidget, widgets_available
from ._license import LicenseError, set_license_key
from . import terrain_scatter

# -----------------------------------------------------------------------------
# Fallback Renderer class
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Mapping


class Renderer:
    """Fallback CPU renderer for terrain.
    
    Args:
        width: Output image width in pixels
        height: Output image height in pixels
        config: Optional renderer configuration
        **kwargs: Override keywords (brdf, shadows, etc.)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: RendererConfig | Mapping[str, Any] | str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        from .config import split_renderer_overrides
        
        self.width = int(width)
        self.height = int(height)
        overrides, remaining = split_renderer_overrides(dict(kwargs))
        if remaining:
            raise TypeError(f"Unexpected arguments: {', '.join(sorted(str(k) for k in remaining))}")
        self._config = load_renderer_config(config, overrides)
        self._exposure = float(self._config.lighting.exposure)

    def get_config(self) -> dict:
        """Return renderer configuration as dict."""
        return self._config.to_dict()

    def apply_preset(self, name: str, **overrides: Any) -> None:
        """Apply a preset to the renderer configuration."""
        preset_map = presets.get(name)
        self._config = RendererConfig.from_mapping(preset_map, self._config)
        if overrides:
            self._config = load_renderer_config(self._config, overrides)

    def render_triangle_rgba(self) -> np.ndarray:
        """Render a basic triangle pattern (fallback test method)."""
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2
        size = min(self.width, self.height) // 4
        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - cx, y - cy
                if abs(dx) + abs(dy) < size and y > cy - size // 2:
                    img[y, x] = [128, 64, 32, 255]
                else:
                    img[y, x] = [16, 16, 24, 255]
        return img

    def render_triangle_png(self, path) -> None:
        """Render triangle to PNG file."""
        numpy_to_png(path, self.render_triangle_rgba())


# -----------------------------------------------------------------------------
# Image I/O utilities
# -----------------------------------------------------------------------------
def numpy_to_png(path, array: np.ndarray) -> None:
    """Save numpy array as PNG file."""
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError(f"File must have .png extension, got {path_str}")

    arr = np.ascontiguousarray(array)
    if arr.dtype != np.uint8:
        raise RuntimeError("Array must be uint8")

    _save_png(path, arr)


def png_to_numpy(path) -> np.ndarray:
    """Load PNG file as numpy array."""
    return _load_png_rgba(path)


def dem_stats(heightmap: np.ndarray) -> dict:
    """Get DEM statistics."""
    if heightmap.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(heightmap.min()),
        "max": float(heightmap.max()),
        "mean": float(heightmap.mean()),
        "std": float(heightmap.std()),
    }


# -----------------------------------------------------------------------------
# Geometry module
# -----------------------------------------------------------------------------
from . import geometry
from . import io

# -----------------------------------------------------------------------------
# P4: Map Plate / Creator Workflow
# -----------------------------------------------------------------------------
from .map_plate import MapPlate, MapPlateConfig, BBox, PlateRegion
from .legend import Legend, LegendConfig
from .scale_bar import ScaleBar, ScaleBarConfig
from .north_arrow import NorthArrow, NorthArrowConfig
from .graticule import GraticuleSpec, generate_graticule

# -----------------------------------------------------------------------------
# P5-export: Vector Export (SVG/PDF)
# -----------------------------------------------------------------------------
from .export import (
    VectorScene,
    VectorStyle as ExportVectorStyle,
    LabelStyle as ExportLabelStyle,
    Polygon as ExportPolygon,
    Polyline as ExportPolyline,
    Label as ExportLabel,
    Bounds as ExportBounds,
    generate_svg,
    export_svg,
    export_pdf,
    validate_svg,
)

# -----------------------------------------------------------------------------
# Helpers (offscreen rendering, frame dumping)
# -----------------------------------------------------------------------------
from .helpers.offscreen import (
    render_offscreen_rgba,
    save_png_deterministic,
    rgba_to_png_bytes,
)
from .helpers.frame_dump import FrameDumper, dump_frame_sequence

# -----------------------------------------------------------------------------
# Scene Bundle (.forge3d)
# -----------------------------------------------------------------------------
_BUNDLE_EXPORT_NAMES = (
    "save_bundle",
    "load_bundle",
    "is_bundle",
    "BundleManifest",
    "LoadedBundle",
    "CameraBookmark",
    "RasterOverlaySpec",
    "SceneBaseState",
    "ReviewLayer",
    "SceneVariant",
    "SceneState",
    "TerrainMeta",
    "BUNDLE_VERSION",
)
_AVAILABLE_BUNDLE_EXPORTS: list[str] = []
try:
    from . import bundle as _bundle_module
except Exception:
    # Keep unrelated imports working while bundle.py is mid-edit or otherwise
    # unavailable. Direct bundle consumers can still import forge3d.bundle.
    _bundle_module = None
else:
    for _name in _BUNDLE_EXPORT_NAMES:
        if hasattr(_bundle_module, _name):
            globals()[_name] = getattr(_bundle_module, _name)
            _AVAILABLE_BUNDLE_EXPORTS.append(_name)

# -----------------------------------------------------------------------------
# P3-reproject: CRS utilities
# -----------------------------------------------------------------------------
from .crs import (
    proj_available,
    transform_coords,
    reproject_geom,
    crs_to_epsg,
    get_crs_from_rasterio,
    get_crs_from_geopandas,
)

# -----------------------------------------------------------------------------
# P4: 3D Buildings Pipeline
# -----------------------------------------------------------------------------
from .buildings import (
    Building,
    BuildingLayer,
    BuildingMaterial,
    add_buildings,
    add_buildings_cityjson,
    add_buildings_3dtiles,
    validate_building_layer_support,
    infer_roof_type,
    material_from_tags,
    material_from_name,
)

# -----------------------------------------------------------------------------
# Mapbox Style Spec Import
# -----------------------------------------------------------------------------
from .style import (
    load_style,
    parse_style,
    apply_style,
    parse_color,
    validate_style_support,
    vector_overlay_configs_from_style,
    label_layer_contracts_from_style,
    paint_to_vector_style,
    layout_to_label_style,
    layer_to_vector_style,
    layer_to_label_style,
    StyleSpec,
    StyleLayer,
    VectorStyle as StyleVectorStyle,
    LabelStyle as StyleLabelStyle,
    PaintProps,
    LayoutProps,
)

# -----------------------------------------------------------------------------
# Product diagnostics
# -----------------------------------------------------------------------------
from .diagnostics import (
    Diagnostic,
    LayerSummary,
    P2_FEATURE_DIAGNOSTIC_CODES,
    REQUIRED_DIAGNOSTIC_CODES,
    RenderFailurePolicy,
    SeverityPolicy,
    SupportMatrixEntry,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    experimental_feature_diagnostic,
    label_rejection_summary_diagnostic,
    missing_glyphs_diagnostic,
    missing_texture_path_diagnostic,
    missing_uvs_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unavailable_cache_lod_stats_diagnostic,
    unsupported_instancing_path_diagnostic,
    unsupported_style_field_diagnostic,
    unsupported_style_layer_type_diagnostic,
    unsupported_texture_format_diagnostic,
    validate_label_support,
    vt_unsupported_family_diagnostic,
)

# -----------------------------------------------------------------------------
# Deterministic label planning
# -----------------------------------------------------------------------------
from .label_plan import (
    AcceptedLabel,
    KeepoutRegion,
    LabelCandidate,
    LabelPlan,
    PriorityClass,
    RejectedLabel,
)

# -----------------------------------------------------------------------------
# Typed MapScene recipe contract
# -----------------------------------------------------------------------------
from .map_scene import (
    FontAtlas,
    FontFallbackRange,
    LabelLayer,
    LightingPreset,
    MapFurnitureLayer,
    MapScene,
    BuildingLayer as MapSceneBuildingLayer,
    OrbitCamera,
    OutputSpec,
    PointCloudLayer,
    RasterOverlay,
    ReproducibilityProfile,
    SceneRecipe,
    TerrainSource,
    Tiles3DLayer,
    TypographySettings,
    VectorOverlay,
)
from . import recipe_manifest
from .alignment import (
    alignment_report,
    alignment_residual,
    reproject_dem_to_target,
    resample_raster_to_grid,
    transform_features as align_transform_features,
    transform_geometry as align_transform_geometry,
)
from .text_atlas import (
    BakedAtlas,
    bake_atlas,
    default_latin_atlas_paths,
    load_atlas_metrics,
    save_atlas,
    validate_atlas_metrics,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Version
    "__version__",
    "version",
    # Core rendering
    "Renderer",
    "PathTracer",
    "ExperimentalSyntheticOutput",
    "make_camera",
    # Native types (when available)
    "Scene",
    "Session",
    "Colormap1D",
    "MaterialSet",
    "IBL",
    "OverlayLayer",
    "TerrainRenderParams",
    "TerrainRenderer",
    "Frame",
    "AovFrame",
    "HdrFrame",
    "CameraKeyframe",
    "CameraAnimation",
    "CameraState",
    # P0.3/M2: Sun ephemeris
    "SunPosition",
    "sun_position",
    "sun_position_utc",
    # Configuration
    "RendererConfig",
    "TerrainRenderParamsConfig",
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "FogSettings",
    "ReflectionSettings",
    "HeightAoSettings",
    "SunVisibilitySettings",
    "ProbeSettings",
    "ReflectionProbeSettings",
    "DetailSettings",
    "MaterialNoiseSettings",
    "MaterialLayerSettings",
    "PomSettings",
    "TriplanarSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "DenoiseSettings",
    "OfflineQualitySettings",
    "VTLayerFamily",
    "TerrainVTSettings",
    "validate_terrain_vt_support",
    "OfflineProgress",
    "OfflineResult",
    "render_offline",
    "oidn_available",
    "oidn_denoise",
    "presets",
    # Colormaps
    "get_colormap",
    "available_colormaps",
    # GPU utilities
    "has_gpu",
    "get_device",
    "enumerate_adapters",
    "device_probe",
    "set_point_shape_mode",
    "set_point_lod_threshold",
    "is_weighted_oit_available",
    "vector_oit_and_pick_demo",
    "vector_render_oit_py",
    "vector_render_pick_map_py",
    "vector_render_oit_and_pick_py",
    "vector_render_polygons_fill_py",
    "memory_metrics",
    "budget_remaining",
    "utilization_ratio",
    "override_memory_limit",
    # Image I/O
    "numpy_to_png",
    "png_to_numpy",
    "dem_stats",
    # Helpers
    "render_offscreen_rgba",
    "save_png_deterministic",
    "rgba_to_png_bytes",
    "FrameDumper",
    "dump_frame_sequence",
    # Modules
    "geometry",
    "gis",
    "io",
    "terrain_scatter",
    "animation",
    "camera_rigs",
    "datasets",
    "widgets",
    # Interactive viewer
    "open_viewer",
    "open_viewer_async",
    "ViewerHandle",
    "LabelBatchResult",
    "ViewerWidget",
    "widgets_available",
    # P4: Map Plate / Creator Workflow
    "MapPlate",
    "MapPlateConfig",
    "BBox",
    "PlateRegion",
    "Legend",
    "LegendConfig",
    "ScaleBar",
    "ScaleBarConfig",
    "NorthArrow",
    "NorthArrowConfig",
    "GraticuleSpec",
    "generate_graticule",
    # Viewer utilities
    "viewer_ipc",
    "colors",
    "interactive",
    # Datasets
    "mini_dem",
    "mini_dem_path",
    "sample_boundaries",
    "sample_boundaries_path",
    "available_datasets",
    "bundled_datasets",
    "remote_datasets",
    "list_datasets",
    "dataset_info",
    "fetch_dataset",
    "fetch_dem",
    "fetch_cityjson",
    "fetch_copc",
    # P5-export: Vector Export (SVG/PDF)
    "VectorScene",
    "ExportVectorStyle",
    "ExportLabelStyle",
    "ExportPolygon",
    "ExportPolyline",
    "ExportLabel",
    "ExportBounds",
    "generate_svg",
    "export_svg",
    "export_pdf",
    "validate_svg",
    # License management
    "set_license_key",
    "LicenseError",
    # Mapbox Style Spec
    "load_style",
    "parse_style",
    "apply_style",
    "parse_color",
    "validate_style_support",
    "vector_overlay_configs_from_style",
    "label_layer_contracts_from_style",
    "paint_to_vector_style",
    "layout_to_label_style",
    "layer_to_vector_style",
    "layer_to_label_style",
    "StyleSpec",
    "StyleLayer",
    "StyleVectorStyle",
    "StyleLabelStyle",
    "PaintProps",
    "LayoutProps",
    # Product diagnostics
    "Diagnostic",
    "LayerSummary",
    "P2_FEATURE_DIAGNOSTIC_CODES",
    "REQUIRED_DIAGNOSTIC_CODES",
    "RenderFailurePolicy",
    "SeverityPolicy",
    "SupportMatrixEntry",
    "ValidationReport",
    "crs_mismatch_diagnostic",
    "estimated_gpu_memory_diagnostic",
    "experimental_feature_diagnostic",
    "label_rejection_summary_diagnostic",
    "missing_glyphs_diagnostic",
    "missing_texture_path_diagnostic",
    "missing_uvs_diagnostic",
    "placeholder_fallback_diagnostic",
    "pro_gated_path_diagnostic",
    "python_public_3dtiles_incomplete_diagnostic",
    "unavailable_cache_lod_stats_diagnostic",
    "unsupported_instancing_path_diagnostic",
    "unsupported_style_field_diagnostic",
    "unsupported_style_layer_type_diagnostic",
    "unsupported_texture_format_diagnostic",
    "validate_label_support",
    "vt_unsupported_family_diagnostic",
    # Deterministic label planning
    "AcceptedLabel",
    "KeepoutRegion",
    "LabelCandidate",
    "LabelPlan",
    "PriorityClass",
    "RejectedLabel",
    # Typed MapScene recipe contract
    "MapScene",
    "SceneRecipe",
    "TerrainSource",
    "RasterOverlay",
    "VectorOverlay",
    "FontAtlas",
    "FontFallbackRange",
    "TypographySettings",
    "LabelLayer",
    "PointCloudLayer",
    "Tiles3DLayer",
    "MapSceneBuildingLayer",
    "MapFurnitureLayer",
    "OrbitCamera",
    "LightingPreset",
    "OutputSpec",
    "ReproducibilityProfile",
    "recipe_manifest",
    "alignment_report",
    "alignment_residual",
    "reproject_dem_to_target",
    "resample_raster_to_grid",
    "align_transform_features",
    "align_transform_geometry",
    "BakedAtlas",
    "bake_atlas",
    "default_latin_atlas_paths",
    "load_atlas_metrics",
    "save_atlas",
    "validate_atlas_metrics",
    # P3-reproject: CRS utilities
    "proj_available",
    "transform_coords",
    "reproject_geom",
    "crs_to_epsg",
    "get_crs_from_rasterio",
    "get_crs_from_geopandas",
    # P4: 3D Buildings Pipeline
    "Building",
    "BuildingLayer",
    "BuildingMaterial",
    "add_buildings",
    "add_buildings_cityjson",
    "add_buildings_3dtiles",
    "validate_building_layer_support",
    "infer_roof_type",
    "material_from_tags",
    "material_from_name",
]
__all__.extend(_AVAILABLE_BUNDLE_EXPORTS)
