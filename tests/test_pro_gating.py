"""Regression tests for the Phase 3 Pro gate."""

from pathlib import Path

import pytest

from forge3d._license import set_license_key


@pytest.fixture(autouse=True)
def clear_license(monkeypatch):
    """Ensure these tests run without a license."""

    monkeypatch.delenv("FORGE3D_LICENSE_KEY", raising=False)
    set_license_key("")
    yield
    set_license_key("")


def test_map_plate_gated():
    """MapPlate instantiation requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.map_plate import MapPlate

    with pytest.raises(LicenseError, match="Map plate composition requires a Pro license"):
        MapPlate()


def test_export_svg_gated(tmp_path: Path):
    """SVG export requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.export import VectorScene, export_svg

    with pytest.raises(LicenseError, match="SVG export requires a Pro license"):
        export_svg(VectorScene(), tmp_path / "test.svg")


def test_export_pdf_gated(tmp_path: Path):
    """PDF export requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.export import VectorScene, export_pdf

    with pytest.raises(LicenseError, match="PDF export requires a Pro license"):
        export_pdf(VectorScene(), tmp_path / "test.pdf")


def test_buildings_add_gated():
    """GeoJSON building import requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.buildings import add_buildings

    with pytest.raises(LicenseError, match="GeoJSON building import requires a Pro license"):
        add_buildings("test.geojson")


def test_buildings_cityjson_gated():
    """CityJSON building import requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.buildings import add_buildings_cityjson

    with pytest.raises(LicenseError, match="CityJSON building import requires a Pro license"):
        add_buildings_cityjson("test.city.json")


def test_buildings_3dtiles_gated():
    """3D Tiles building import requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.buildings import add_buildings_3dtiles

    with pytest.raises(LicenseError, match="3D Tiles building import requires a Pro license"):
        add_buildings_3dtiles("tileset.json")


def test_style_load_gated():
    """Mapbox style loading requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.style import load_style

    with pytest.raises(LicenseError, match="Mapbox style loading requires a Pro license"):
        load_style("test.json")


def test_style_apply_gated():
    """Mapbox style application requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.style import StyleSpec, apply_style

    with pytest.raises(LicenseError, match="Mapbox style application requires a Pro license"):
        apply_style(StyleSpec(), [])


def test_bundle_save_gated(tmp_path: Path):
    """Bundle save requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.bundle import save_bundle

    with pytest.raises(LicenseError, match="Scene bundle save requires a Pro license"):
        save_bundle(tmp_path / "test.forge3d")


def test_bundle_load_gated(tmp_path: Path):
    """Bundle load requires Pro."""

    from forge3d._license import LicenseError
    from forge3d.bundle import load_bundle

    with pytest.raises(LicenseError, match="Scene bundle load requires a Pro license"):
        load_bundle(tmp_path / "test.forge3d")
