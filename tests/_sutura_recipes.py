"""SUTURA definition-of-done recipe builders.

Four first-class validated MapScene configurations used by
tests/test_mapscene_sutura_integrity.py:

- ``terrain_raster_labels_buildings``
- ``terrain_point_cloud``
- ``terrain_tiles3d``
- ``all_layers``

Every builder produces a scene whose inputs are fully serializable so a
``save_bundle`` -> ``load_bundle`` round-trip reconstructs the identical
recipe (terrain and raster overlays are written to files; labels, buildings,
and point clouds are inline).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

import forge3d as f3d

RECIPE_NAMES = (
    "terrain_raster_labels_buildings",
    "terrain_point_cloud",
    "terrain_tiles3d",
    "all_layers",
)

_CRS = "EPSG:32610"


def _write_terrain(tmp_path: Path) -> Path:
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    heightmap = 12.0 * np.sin(xx * 0.35) * np.cos(yy * 0.25) + 0.5 * (xx + yy)
    path = tmp_path / "sutura_dem.npy"
    np.save(path, heightmap.astype(np.float32))
    return path


def _write_raster_overlay(tmp_path: Path) -> Path:
    from forge3d.helpers.offscreen import save_png_deterministic

    raster = np.zeros((32, 32, 4), dtype=np.uint8)
    raster[..., 0] = np.linspace(30, 220, 32).astype(np.uint8)[None, :]
    raster[..., 1] = 128
    raster[..., 2] = np.linspace(200, 40, 32).astype(np.uint8)[:, None]
    raster[..., 3] = 255
    path = tmp_path / "sutura_overlay.png"
    save_png_deterministic(path, raster)
    return path


def write_pnts(tmp_path: Path) -> Path:
    """Write a minimal spec-conformant 3D Tiles PNTS payload (POSITION + RGB)."""
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, 2.0, 1.0], [8.0, 6.0, 2.0], [2.0, 7.0, 0.5]],
        dtype="<f4",
    )
    colors = np.asarray(
        [[255, 64, 32], [32, 255, 64], [64, 32, 255], [200, 200, 40]],
        dtype=np.uint8,
    )
    feature_table = {
        "POINTS_LENGTH": int(positions.shape[0]),
        "POSITION": {"byteOffset": 0},
        "RGB": {"byteOffset": positions.nbytes},
    }
    ft_json = json.dumps(feature_table).encode("utf-8")
    ft_json += b" " * ((8 - len(ft_json) % 8) % 8)
    ft_bin = positions.tobytes() + colors.tobytes()
    ft_bin += b"\x00" * ((8 - len(ft_bin) % 8) % 8)
    total = 28 + len(ft_json) + len(ft_bin)
    header = b"pnts" + struct.pack("<6I", 1, total, len(ft_json), len(ft_bin), 0, 0)
    path = tmp_path / "sutura_points.pnts"
    path.write_bytes(header + ft_json + ft_bin)
    return path


def _terrain(tmp_path: Path) -> "f3d.TerrainSource":
    return f3d.TerrainSource(
        path=str(_write_terrain(tmp_path)),
        crs=_CRS,
        metadata={"width": 32, "height": 32, "source_id": "sutura-dem"},
        elevation_sampling_available=True,
    )


def _raster_layer(tmp_path: Path) -> "f3d.RasterOverlay":
    return f3d.RasterOverlay(
        layer_id="ortho",
        path=str(_write_raster_overlay(tmp_path)),
        crs=_CRS,
        opacity=0.75,
        metadata={"width": 32, "height": 32, "source_id": "sutura-overlay"},
    )


def _label_layer() -> "f3d.LabelLayer":
    return f3d.LabelLayer(
        layer_id="labels",
        labels=[
            {
                "id": "summit",
                "kind": "point",
                "text": "Summit",
                "geometry": {"type": "Point", "coordinates": (30.0, 20.0, 0.0)},
            },
            {
                "id": "valley",
                "kind": "point",
                "text": "Valley",
                "geometry": {"type": "Point", "coordinates": (66.0, 42.0, 0.0)},
            },
        ],
        glyph_atlas={"glyphs": sorted(set("SummitValley"))},
        metadata={"source_id": "sutura-labels", "seed": 7},
    )


def _building_layer() -> "f3d.MapSceneBuildingLayer":
    features = []
    for idx, roof_type in enumerate(("flat", "gabled")):
        x0 = 0.18 + idx * 0.34
        x1 = x0 + 0.22
        features.append(
            {
                "id": f"b-{roof_type}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, 0.28), (x1, 0.28), (x1, 0.62), (x0, 0.62), (x0, 0.28)]],
                },
                "properties": {
                    "height": 24.0 + idx * 8.0,
                    "roof:shape": roof_type,
                    "building:material": "brick",
                },
            }
        )
    return f3d.MapSceneBuildingLayer(
        layer_id="buildings",
        source={"source_id": "sutura-buildings", "asset_status": "fixture"},
        support_level="supported",
        geometry_count=len(features),
        material_status="scalar_pbr_underdeveloped",
        features=features,
        metadata={"source_id": "sutura-buildings", "asset_status": "fixture"},
    )


def _point_cloud_layer() -> "f3d.PointCloudLayer":
    return f3d.PointCloudLayer(
        layer_id="points",
        crs=_CRS,
        point_count=4,
        metadata={
            "source_id": "sutura-points",
            "positions": [
                [0.0, 0.0, 0.0],
                [10.0, 5.0, 1.0],
                [4.0, 8.0, 0.5],
                [7.0, 2.0, 2.0],
            ],
            "bounds": [0.0, 0.0, 10.0, 8.0],
            "color": "#ff8000",
            "point_size": 5.0,
        },
    )


def _tiles3d_layer(tmp_path: Path) -> "f3d.Tiles3DLayer":
    return f3d.Tiles3DLayer(
        layer_id="tiles",
        source={"path": str(write_pnts(tmp_path)), "source_format": "pnts"},
        metadata={"source_id": "sutura-tiles", "point_size": 6.0},
    )


def _vector_layer() -> "f3d.VectorOverlay":
    return f3d.VectorOverlay(
        layer_id="roads",
        crs=_CRS,
        features=[
            {
                "id": "road-1",
                "geometry": {"type": "LineString", "coordinates": [(0.1, 0.2), (0.9, 0.7)]},
                "properties": {"class": "primary"},
            }
        ],
        width_px=3,
        style={
            "version": 8,
            "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#f9fafb"}}],
        },
    )


def build_scene(name: str, tmp_path: Path) -> "f3d.MapScene":
    """Build one of the four SUTURA DoD recipes rooted at ``tmp_path``."""
    if name == "terrain_raster_labels_buildings":
        layers = [_raster_layer(tmp_path), _label_layer(), _building_layer()]
    elif name == "terrain_point_cloud":
        layers = [_point_cloud_layer()]
    elif name == "terrain_tiles3d":
        layers = [_tiles3d_layer(tmp_path)]
    elif name == "all_layers":
        layers = [
            _raster_layer(tmp_path),
            _vector_layer(),
            _label_layer(),
            _building_layer(),
            _point_cloud_layer(),
            _tiles3d_layer(tmp_path),
        ]
    else:
        raise ValueError(f"Unknown SUTURA recipe: {name}")

    return f3d.MapScene(
        terrain=_terrain(tmp_path),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=250.0, azimuth_deg=30.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.2),
        output=f3d.OutputSpec(width=96, height=64, format="png"),
        layers=layers,
        reproducibility_profile=f3d.ReproducibilityProfile(seed=7),
    )
