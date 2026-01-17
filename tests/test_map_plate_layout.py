"""P4: Map plate layout, legend, and scale bar tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from forge3d.map_plate import MapPlate, MapPlateConfig, BBox, PlateRegion
from forge3d.legend import Legend, LegendConfig
from forge3d.scale_bar import ScaleBar, ScaleBarConfig


class TestBBox:
    def test_bbox_properties(self):
        bbox = BBox(west=-122.0, south=46.0, east=-121.0, north=47.0)
        assert bbox.width == pytest.approx(1.0)
        assert bbox.height == pytest.approx(1.0)
        assert bbox.center_lat == pytest.approx(46.5)

    def test_bbox_immutable(self):
        bbox = BBox(west=0, south=0, east=1, north=1)
        with pytest.raises(Exception):
            bbox.west = 10


class TestPlateRegion:
    def test_region_dimensions(self):
        region = PlateRegion(x0=10, y0=20, x1=110, y1=220)
        assert region.width == 100
        assert region.height == 200
        assert region.rect == (10, 20, 110, 220)


class TestMapPlateConfig:
    def test_default_config(self):
        cfg = MapPlateConfig()
        assert cfg.width == 1600
        assert cfg.height == 1000
        assert cfg.margin == (60, 200, 80, 40)
        assert cfg.background == (255, 255, 255, 255)

    def test_custom_config(self):
        cfg = MapPlateConfig(width=800, height=600, margin=(10, 10, 10, 10))
        assert cfg.width == 800
        assert cfg.height == 600


class TestMapPlateLayout:
    def test_plate_dimensions(self):
        plate = MapPlate(MapPlateConfig(width=800, height=600))
        composed = plate.compose()
        assert composed.shape == (600, 800, 4)
        assert composed.dtype == np.uint8

    def test_default_background_white(self):
        plate = MapPlate(MapPlateConfig(width=100, height=100))
        composed = plate.compose()
        assert np.all(composed[50, 50] == [255, 255, 255, 255])

    def test_custom_background(self):
        plate = MapPlate(MapPlateConfig(
            width=100, height=100,
            background=(128, 64, 32, 255)
        ))
        composed = plate.compose()
        assert composed[50, 50, 0] == 128
        assert composed[50, 50, 1] == 64
        assert composed[50, 50, 2] == 32

    def test_map_region_blitted(self):
        plate = MapPlate(MapPlateConfig(
            width=200, height=200,
            margin=(20, 20, 20, 20),
            background=(255, 255, 255, 255)
        ))
        map_img = np.zeros((100, 100, 4), dtype=np.uint8)
        map_img[..., 0] = 200
        map_img[..., 3] = 255
        plate.set_map_region(map_img, BBox(0, 0, 1, 1))
        composed = plate.compose()
        center = composed[100, 100]
        assert center[0] == 200

    def test_rgb_image_converted_to_rgba(self):
        plate = MapPlate(MapPlateConfig(width=100, height=100, margin=(10, 10, 10, 10)))
        rgb_img = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb_img[..., 1] = 128
        plate.set_map_region(rgb_img, BBox(0, 0, 1, 1))
        assert plate._map_image.shape[2] == 4

    def test_grayscale_image_converted_to_rgba(self):
        plate = MapPlate(MapPlateConfig(width=100, height=100, margin=(10, 10, 10, 10)))
        gray_img = np.full((50, 50), 100, dtype=np.uint8)
        plate.set_map_region(gray_img, BBox(0, 0, 1, 1))
        assert plate._map_image.shape[2] == 4


class TestMapPlateTitle:
    def test_add_title(self):
        plate = MapPlate(MapPlateConfig(width=400, height=300))
        plate.add_title("Test Title", font_size=20)
        assert plate._title is not None
        assert plate._title.text == "Test Title"
        assert plate._title.font_size == 20


class TestMapPlateExport:
    def test_export_png(self, tmp_path):
        plate = MapPlate(MapPlateConfig(width=200, height=150))
        output_path = tmp_path / "test_plate.png"
        plate.export_png(output_path)
        assert output_path.exists()
        from PIL import Image
        img = Image.open(output_path)
        assert img.size == (200, 150)

    def test_export_jpeg(self, tmp_path):
        plate = MapPlate(MapPlateConfig(width=200, height=150))
        output_path = tmp_path / "test_plate.jpg"
        plate.export_jpeg(output_path)
        assert output_path.exists()


class TestLegendConfig:
    def test_default_config(self):
        cfg = LegendConfig()
        assert cfg.orientation == "vertical"
        assert cfg.tick_count == 5
        assert cfg.bar_height == 250

    def test_custom_config(self):
        cfg = LegendConfig(orientation="horizontal", tick_count=10, bar_height=100)
        assert cfg.orientation == "horizontal"
        assert cfg.tick_count == 10


class TestLegendGeneration:
    @pytest.fixture
    def sample_colormap(self):
        n = 256
        rgba = np.zeros((n, 4), dtype=np.float32)
        rgba[:, 0] = np.linspace(0, 1, n)
        rgba[:, 1] = np.linspace(0.5, 0.5, n)
        rgba[:, 2] = np.linspace(1, 0, n)
        rgba[:, 3] = 1.0
        return rgba

    def test_legend_ticks_count(self, sample_colormap):
        legend = Legend(sample_colormap, domain=(0, 1000), config=LegendConfig(tick_count=5))
        ticks = legend.get_ticks()
        assert len(ticks) == 5

    def test_legend_ticks_values(self, sample_colormap):
        legend = Legend(sample_colormap, domain=(0, 4000), config=LegendConfig(tick_count=5))
        ticks = legend.get_ticks()
        values = [t[0] for t in ticks]
        assert values[0] == pytest.approx(0)
        assert values[-1] == pytest.approx(4000)

    def test_legend_ticks_labels(self, sample_colormap):
        legend = Legend(
            sample_colormap,
            domain=(0, 1000),
            config=LegendConfig(tick_count=3, label_format="{:.1f}", label_suffix=" m")
        )
        ticks = legend.get_ticks()
        assert ticks[0][1] == "0.0 m"
        assert ticks[1][1] == "500.0 m"
        assert ticks[2][1] == "1000.0 m"

    def test_legend_render_shape(self, sample_colormap):
        legend = Legend(
            sample_colormap,
            domain=(0, 1000),
            config=LegendConfig(bar_height=200, bar_width=20)
        )
        img = legend.render()
        assert img.ndim == 3
        assert img.shape[2] == 4
        assert img.dtype == np.uint8

    def test_legend_render_vertical(self, sample_colormap):
        legend = Legend(
            sample_colormap,
            domain=(0, 100),
            config=LegendConfig(orientation="vertical", bar_height=100, bar_width=15)
        )
        img = legend.render()
        assert img.shape[0] >= 100

    def test_legend_render_horizontal(self, sample_colormap):
        legend = Legend(
            sample_colormap,
            domain=(0, 100),
            config=LegendConfig(orientation="horizontal", bar_height=100, bar_width=15)
        )
        img = legend.render()
        assert img.shape[1] >= 100


class TestScaleBarConfig:
    def test_default_config(self):
        cfg = ScaleBarConfig()
        assert cfg.units == "km"
        assert cfg.style == "alternating"
        assert cfg.divisions == 4

    def test_custom_units(self):
        cfg = ScaleBarConfig(units="mi")
        assert cfg.units == "mi"


class TestScaleBarGeodetic:
    def test_meters_per_pixel_equator(self):
        bbox = BBox(west=0, south=0, east=1, north=1)
        mpp = ScaleBar.compute_meters_per_pixel(bbox, 1000)
        assert 100 < mpp < 120

    def test_meters_per_pixel_mid_latitude(self):
        bbox = BBox(west=0, south=45, east=1, north=46)
        mpp = ScaleBar.compute_meters_per_pixel(bbox, 1000)
        assert 70 < mpp < 90

    def test_meters_per_pixel_high_latitude(self):
        bbox = BBox(west=0, south=70, east=1, north=71)
        mpp = ScaleBar.compute_meters_per_pixel(bbox, 1000)
        assert 30 < mpp < 50

    def test_meters_per_pixel_scales_with_width(self):
        bbox = BBox(west=0, south=0, east=1, north=1)
        mpp_1000 = ScaleBar.compute_meters_per_pixel(bbox, 1000)
        mpp_2000 = ScaleBar.compute_meters_per_pixel(bbox, 2000)
        assert mpp_1000 == pytest.approx(mpp_2000 * 2, rel=0.01)


class TestScaleBarRender:
    def test_scale_bar_renders(self):
        sb = ScaleBar(meters_per_pixel=100, config=ScaleBarConfig())
        img = sb.render()
        assert img.ndim == 3
        assert img.shape[2] == 4
        assert img.dtype == np.uint8

    def test_scale_bar_simple_style(self):
        sb = ScaleBar(meters_per_pixel=100, config=ScaleBarConfig(style="simple"))
        img = sb.render()
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_scale_bar_different_units(self):
        for units in ["m", "km", "mi", "ft"]:
            sb = ScaleBar(meters_per_pixel=100, config=ScaleBarConfig(units=units))
            img = sb.render()
            assert img.shape[0] > 0


class TestMapPlateIntegration:
    @pytest.fixture
    def sample_colormap(self):
        n = 256
        rgba = np.zeros((n, 4), dtype=np.float32)
        rgba[:, 0] = np.linspace(0, 1, n)
        rgba[:, 1] = 0.5
        rgba[:, 2] = np.linspace(1, 0, n)
        rgba[:, 3] = 1.0
        return rgba

    def test_full_plate_composition(self, sample_colormap, tmp_path):
        plate = MapPlate(MapPlateConfig(
            width=800,
            height=600,
            margin=(50, 150, 50, 30),
        ))
        map_img = np.zeros((400, 500, 4), dtype=np.uint8)
        map_img[..., 1] = 128
        map_img[..., 3] = 255
        bbox = BBox(west=-122, south=46, east=-121, north=47)
        plate.set_map_region(map_img, bbox)
        plate.add_title("Integration Test Map")
        legend = Legend(sample_colormap, domain=(0, 1000), config=LegendConfig(bar_height=200))
        plate.add_legend(legend.render())
        mpp = ScaleBar.compute_meters_per_pixel(bbox, 500)
        sb = ScaleBar(mpp, config=ScaleBarConfig())
        plate.add_scale_bar(sb.render())
        composed = plate.compose()
        assert composed.shape == (600, 800, 4)
        output_path = tmp_path / "integration_test.png"
        plate.export_png(output_path)
        assert output_path.exists()
