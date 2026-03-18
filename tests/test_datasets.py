"""Tests for the bundled and on-demand Phase 2 dataset registry."""

from pathlib import Path

import numpy as np
import pytest

from forge3d import datasets


def test_mini_dem_loads_with_expected_shape_and_dtype():
    """The bundled mini DEM should be a small float32 height grid."""
    dem = datasets.mini_dem()

    assert isinstance(dem, np.ndarray)
    assert dem.shape == (256, 256)
    assert dem.dtype == np.float32
    assert float(dem.max()) > 500.0
    assert float(dem.min()) < 0.0


def test_mini_dem_path_exists():
    """The packaged mini DEM path resolves to a real file."""
    path = datasets.mini_dem_path()

    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix == ".npy"


def test_sample_boundaries_load_as_geojson():
    """The bundled boundary sample should parse as a small feature collection."""
    geojson = datasets.sample_boundaries()

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 3
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"


def test_registry_lists_bundled_and_remote_datasets():
    """The registry should expose the Phase 2 bundled and fetchable names."""
    names = datasets.available()

    assert "mini_dem" in names
    assert "sample_boundaries" in names
    assert "rainier" in names
    assert "mt-st-helens" in names


def test_list_datasets_returns_named_records():
    """list_datasets() should produce metadata rows with stable names."""
    records = datasets.list_datasets()

    assert isinstance(records, list)
    assert any(record["name"] == "mini_dem" for record in records)
    assert any(record["name"] == "rainier" for record in records)


def test_fetch_returns_bundled_path_for_bundled_dataset():
    """fetch() should resolve bundled datasets without network access."""
    assert datasets.fetch("mini_dem") == datasets.mini_dem_path()


def test_fetch_dem_prefers_local_asset_checkout():
    """Large DEM samples should resolve locally inside the repo checkout when present."""
    path = datasets.fetch_dem("rainier")

    assert path.exists()
    assert path.suffix.lower() in (".tif", ".tiff")
    assert path.name == "dem_rainier.tif"


def test_fetch_cityjson_prefers_local_asset_checkout():
    """CityJSON samples should resolve locally when the repo assets are present."""
    path = datasets.fetch_cityjson("sample-buildings")

    assert path.exists()
    assert path.suffix.lower() == ".json"


def test_fetch_copc_prefers_local_asset_checkout():
    """Point cloud samples should resolve locally when the repo assets are present."""
    path = datasets.fetch_copc("mt-st-helens")

    assert path.exists()
    assert path.suffix.lower() == ".laz"


def test_fetch_rejects_unknown_dataset():
    """Unknown dataset names should fail with a helpful error."""
    with pytest.raises(KeyError, match="Unknown dataset"):
        datasets.fetch("does-not-exist")


def test_fetch_kind_validation_rejects_wrong_registry_kind():
    """Kind-specific helpers should reject datasets from the wrong category."""
    with pytest.raises(ValueError, match="expected 'dem'"):
        datasets.fetch_dem("sample-buildings")


def test_default_dataset_base_url_uses_live_repository_path():
    """The fallback download base URL should use the live GitHub repository."""

    assert "milos-agathon/forge3d" in datasets._DEFAULT_DATASET_BASE_URL
    assert "github.com/forge3d/forge3d" not in datasets._DEFAULT_DATASET_BASE_URL
