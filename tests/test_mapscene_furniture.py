from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.graticule import GraticuleSpec, generate_graticule
from forge3d.map_plate import BBox
from forge3d.scale_bar import ScaleBar


def test_scale_bar_uses_geodesic_distance_when_pyproj_is_available():
    pyproj = pytest.importorskip("pyproj")
    bbox = BBox(west=-1.0, south=59.5, east=1.0, north=60.5)

    meters_per_pixel = ScaleBar.compute_meters_per_pixel(bbox, 200, geodesic=True)
    _az12, _az21, expected_distance = pyproj.Geod(ellps="WGS84").inv(-1.0, 60.0, 1.0, 60.0)

    assert meters_per_pixel == pytest.approx(abs(expected_distance) / 200.0, rel=1.0e-6)


@pytest.mark.parametrize("bounds,crs", [
    ((-122.5, 46.6, -121.9, 47.0), "EPSG:32610"),
    ((3, 50.5, 7.3, 53.7), "EPSG:3035"),
    ((0, 90, 1, 91), "EPSG:4326"),
])
def test_automatic_scale_rejects_bounds_outside_declared_crs(tmp_path, bounds, crs):
    from dataclasses import replace
    from forge3d._map_scene_render import _compose_scale_bar
    scene = _furniture_scene(tmp_path / "invalid.png", furniture=None)
    scene.recipe.terrain = replace(scene.recipe.terrain, crs=crs, metadata={"bounds": bounds})
    with pytest.raises(ValueError, match="bounds cannot be validated for CRS"):
        _compose_scale_bar(np.zeros((224, 128, 4), np.uint8), scene.recipe, {})


def test_automatic_scale_uses_registered_bounds_not_graticule_bounds(tmp_path, monkeypatch):
    from dataclasses import replace
    from forge3d._map_scene_render import _compose_scale_bar, _validate_scale_bounds
    from forge3d.scale_bar import ScaleBar
    # Small projected coordinates can be valid: do not guess units by magnitude.
    _validate_scale_bounds((0, 0, 1, 1), "EPSG:3857")
    scene = _furniture_scene(tmp_path / "registered.png", furniture=f3d.MapFurnitureLayer(
        graticule={"bounds": [-122.5, 46.6, -121.9, 47.0]}))
    scene.recipe.terrain = replace(scene.recipe.terrain, crs="EPSG:32610",
        metadata={"bounds": [500000, 5200000, 510000, 5210000]})
    seen = []
    original = ScaleBar.compute_meters_per_pixel
    def measure(bbox, width, **kwargs):
        seen.append(bbox)
        return original(bbox, width, **kwargs)
    monkeypatch.setattr(ScaleBar, "compute_meters_per_pixel", staticmethod(measure))
    _compose_scale_bar(np.zeros((224, 128, 4), np.uint8), scene.recipe, {"width_px": 100})
    assert seen[0].west == 500000 and seen[0].crs == "EPSG:32610"


@pytest.mark.parametrize("crs,geographic_bounds,geodesic", [
    ("EPSG:32631", (3, 50.5, 7.3, 53.7), True),  # Netherlands crossing 6 E.
    ("EPSG:32632", (5.5, 47, 15.5, 55.5), True),  # Germany crossing 12 E.
    # The native projection engine does not support these two transforms.
    # Exercise the existing explicit projected-distance option, without fallback.
    ("EPSG:28992", (2.5, 50.5, 7.5, 54), False),  # RD New with the North Sea.
    ("EPSG:2056", (5.5, 45.5, 11, 48.5), False),  # LV95 beyond Switzerland.
])
def test_automatic_scale_accepts_national_map_edges_outside_area(tmp_path, crs, geographic_bounds, geodesic):
    from dataclasses import replace
    from pyproj import CRS, Transformer
    from forge3d._map_scene_render import _compose_scale_bar, _bounds_edge_samples, _validate_scale_bounds
    projected = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform_bounds(*geographic_bounds)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform(*np.asarray(_bounds_edge_samples(projected)).T)
    area = CRS.from_user_input(crs).area_of_use
    # These fixtures exercise the old edge rejection, not just small valid maps.
    assert np.any((lon < area.west) | (lon > area.east) | (lat < area.south) | (lat > area.north))
    centre = inverse.transform((projected[0]+projected[2])/2, (projected[1]+projected[3])/2)
    assert area.west <= centre[0] <= area.east and area.south <= centre[1] <= area.north
    _validate_scale_bounds(projected, crs)
    scene = _furniture_scene(tmp_path / "national.png", furniture=None)
    scene.recipe.terrain = replace(scene.recipe.terrain, crs=crs, metadata={"bounds": projected})
    image = np.zeros((224, 128, 4), np.uint8)
    _compose_scale_bar(image, scene.recipe, {"width_px": 100, "geodesic": geodesic})
    assert np.any(image)


def test_explicit_scale_does_not_infer_crs_or_bounds(tmp_path, monkeypatch):
    from forge3d import _map_scene_render as compositor
    scene = _furniture_scene(tmp_path / "measured.png", furniture=None)
    def unexpected(*args):
        raise AssertionError("An explicit measured scale must not infer scale from bounds")
    monkeypatch.setattr(compositor, "_validate_scale_bounds", unexpected)
    image = np.zeros((224, 128, 4), np.uint8)
    compositor._compose_scale_bar(image, scene.recipe, {"meters_per_pixel": 358, "width_px": 100})
    assert np.any(image)


def test_graticule_generates_lines_labels_and_explicit_target_crs():
    graticule = generate_graticule(
        GraticuleSpec(bounds=(0.0, 0.0, 2.0, 2.0), interval_deg=1.0, line_steps=4)
    )

    assert graticule["source_crs"] == "EPSG:4326"
    assert graticule["target_crs"] == "EPSG:4326"
    assert len(graticule["features"]) == 6
    assert len(graticule["labels"]) == 6
    assert graticule["features"][0]["geometry"]["type"] == "LineString"
    assert graticule["features"][0]["properties"]["kind"] == "meridian"


def test_graticule_transforms_lines_and_labels_to_projected_target_crs():
    pytest.importorskip("pyproj")

    graticule = generate_graticule(
        GraticuleSpec(
            bounds=(0.0, 0.0, 1.0, 1.0),
            interval_deg=1.0,
            target_crs="EPSG:3857",
            line_steps=2,
        )
    )

    assert graticule["target_crs"] == "EPSG:3857"
    meridian = next(
        feature
        for feature in graticule["features"]
        if feature["properties"]["kind"] == "meridian" and feature["properties"]["value"] == 1.0
    )
    first_point = meridian["geometry"]["coordinates"][0]
    assert first_point[0] == pytest.approx(111319.49, rel=1.0e-4)
    assert first_point[1] == pytest.approx(0.0, abs=1.0e-6)

    label = next(
        label
        for label in graticule["labels"]
        if label["kind"] == "meridian" and label["value"] == 1.0
    )
    assert label["coordinate"][0] == pytest.approx(111319.49, rel=1.0e-4)
    assert label["coordinate"][1] == pytest.approx(0.0, abs=1.0e-6)
    assert label["text"] == "1 degE"


def _furniture_scene(
    path: Path,
    *,
    furniture: f3d.MapFurnitureLayer | None,
    target_crs: str | None = None,
    output_metadata: dict[str, object] | None = None,
) -> f3d.MapScene:
    terrain = f3d.TerrainSource(
        data=np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8),
        crs="EPSG:4326",
        metadata={"bounds": [0.0, 0.0, 2.0, 2.0], "width": 8, "height": 8},
        elevation_sampling_available=True,
    )
    return f3d.MapScene(
        terrain=terrain,
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        layers=[],
        output=f3d.OutputSpec(width=180, height=120, path=str(path), metadata=output_metadata),
        target_crs=target_crs,
        map_furniture=furniture,
    )


def test_mapscene_render_composites_map_furniture(tmp_path):
    plain_path = tmp_path / "plain.png"
    furnished_path = tmp_path / "furnished.png"
    plain = _furniture_scene(plain_path, furniture=None)
    furniture = f3d.MapFurnitureLayer(
        title="Grid",
        legend={"items": ["roads", "labels"]},
        scale_bar={"bounds": [0.0, 0.0, 2.0, 2.0], "units": "km", "width_px": 70},
        north_arrow={"style": "simple", "size": 36},
        graticule={"bounds": [0.0, 0.0, 2.0, 2.0], "interval_deg": 1.0, "line_steps": 8},
    )
    furnished = _furniture_scene(furnished_path, furniture=furniture)

    plain.render()
    report = furnished.render()

    from PIL import Image

    plain_rgba = np.asarray(Image.open(plain_path).convert("RGBA"))
    furnished_rgba = np.asarray(Image.open(furnished_path).convert("RGBA"))
    assert not np.array_equal(plain_rgba, furnished_rgba)
    assert report.supported_features["mapscene.furniture"] == "supported"
    assert report.supported_features["mapscene.graticule"] == "supported"
    assert report.supported_features["mapscene.scale_bar.geodesic"] == "supported"
    assert report.supported_features["mapscene.furniture_composite"] == "supported"


def test_map_furniture_roundtrip_preserves_graticule():
    furniture = f3d.MapFurnitureLayer(graticule={"bounds": [0.0, 0.0, 1.0, 1.0], "interval_deg": 0.5})

    payload = furniture.to_dict()

    assert payload["graticule"]["interval_deg"] == 0.5
    assert payload["scale_bar"] is None


def test_map_furniture_accepts_graticule_spec_roundtrip():
    furniture = f3d.MapFurnitureLayer(
        graticule=GraticuleSpec(bounds=(0.0, 0.0, 1.0, 1.0), interval_deg=0.25, target_crs="EPSG:3857")
    )

    payload = furniture.to_dict()

    assert payload["graticule"]["bounds"] == [0.0, 0.0, 1.0, 1.0]
    assert payload["graticule"]["interval_deg"] == 0.25
    assert payload["graticule"]["target_crs"] == "EPSG:3857"


def test_mapscene_graticule_render_uses_scene_target_crs(monkeypatch, tmp_path):
    calls = []

    def fake_generate_graticule(bounds, **kwargs):
        calls.append({"bounds": bounds, **kwargs})
        return {"features": [], "labels": []}

    monkeypatch.setattr("forge3d.graticule.generate_graticule", fake_generate_graticule)
    furniture = f3d.MapFurnitureLayer(graticule={"bounds": [0.0, 0.0, 1.0, 1.0], "interval_deg": 0.5})
    scene = _furniture_scene(
        tmp_path / "projected-graticule.png",
        furniture=furniture,
        target_crs="EPSG:3857",
    )

    scene.render()

    assert calls
    assert calls[-1]["target_crs"] == "EPSG:3857"


def test_place_overlay_avoids_keepout_rect():
    from forge3d._map_scene_render import _place_overlay

    image = np.zeros((20, 40, 4), dtype=np.uint8)
    image[..., 3] = 255
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = 255

    _place_overlay(image, overlay, "bottom-left", 0, keepouts=[{"bounds": [0, 10, 12, 20]}])

    assert image[15, 5, 0] == 0
    assert image[15, 35, 0] == 255


def test_mapscene_scale_bar_config_scales_with_output_dpi(monkeypatch, tmp_path):
    from forge3d import scale_bar as scale_bar_module

    configs = []

    class RecordingScaleBar:
        def __init__(self, meters_per_pixel, config=None):
            self.meters_per_pixel = meters_per_pixel
            self.config = config
            configs.append(config)

        @staticmethod
        def compute_meters_per_pixel(*_args, **_kwargs):
            return 1.0

        def render_geometry(self):
            return np.zeros((4, 6, 4), dtype=np.uint8), "1 m", (0, 0)

    monkeypatch.setattr(scale_bar_module, "ScaleBar", RecordingScaleBar)
    furniture = f3d.MapFurnitureLayer(
        scale_bar={
            "bounds": [0.0, 0.0, 2.0, 2.0],
            "width_px": 70,
            "height_px": 20,
            "font_size": 10,
            "padding": 4,
            "bar_height": 6,
        }
    )
    scene = _furniture_scene(
        tmp_path / "dpi-scale-bar.png",
        furniture=furniture,
        output_metadata={"dpi": 300},
    )

    scene.render()

    assert configs
    assert configs[-1].width_px == 140
    assert configs[-1].height_px == 40
    assert configs[-1].font_size == 20
    assert configs[-1].padding == 8
    assert configs[-1].bar_height == 12
