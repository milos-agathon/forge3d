"""Independent Landscape encoding/coordinates; no editor execution is implied."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest

from forge3d.landscape import LandscapeLayout, export_landscape, verify_landscape_import
from forge3d.preparation import PreparationError, PreparationRecipe, PreparationSource, prepare_scene
from test_preparation import elevation, grid, raster


def prepared_fixture(tmp_path, heights=None, source_grid=None, *, void=False):
    source_grid = source_grid or grid(8, 8, spacing=2)
    heights = (np.linspace(-431.2, 3999.7, 64, dtype=np.float64).reshape(8, 8)
               if heights is None else heights)
    dem = elevation(tmp_path, source_grid, heights)
    if void:
        np.save(tmp_path / "valid.npy", np.isfinite(heights))
    classes = np.tile(np.array([0, 1] * 4, dtype=np.uint8), (8, 1))
    category = PreparationSource("landcover", raster(tmp_path / "cover.tif", classes, source_grid), "categorical")
    return prepare_scene(PreparationRecipe(source_grid, (dem, category), "NAP"))


def export(prepared, path, **kwargs):
    return export_landscape(prepared, path, layout=LandscapeLayout(7, 1, 1, 1),
                            origin=(100_000, 440_000, 0), categorical_source="landcover",
                            material_layers={0: "Water", 1: "Land"}, **kwargs)


@pytest.mark.parametrize("flat_height", [None, -100.0, 0.0, 9_000.0])
def test_independent_height_decode_and_quantization(tmp_path, flat_height):
    heights = (None if flat_height is None else np.full((8, 8), flat_height, dtype=np.float64))
    prepared = prepared_fixture(tmp_path, heights)
    root = export(prepared, tmp_path / "landscape")
    manifest = json.loads((root / "landscape.json").read_text())
    samples = np.array(Image.open(root / "height.png"))
    assert samples.dtype == np.uint16
    assert samples.shape == (8, 8)
    encoding = manifest["encoding"]
    reconstructed = encoding["intercept_m"] + samples.astype(np.float64) * encoding["step_m"]
    np.testing.assert_allclose(reconstructed, prepared.layers["dem"].data, rtol=0,
                               atol=encoding["quantization_bound_m"] + encoding["floating_bound_m"])
    settings = manifest["import_settings"]
    engine_heights = settings["vertex_0_0_base_location_cm"][2] + (samples.astype(np.float64) - 32768) / 128 * settings["scale_cm"][2]
    np.testing.assert_allclose(engine_heights / 100, reconstructed, rtol=0,
                               atol=max(encoding["floating_bound_m"], np.finfo(float).eps))
    assert manifest["editor_import_verified"] is False
    assert manifest["qualified_editor_versions"] == []
    assert manifest["geographic"]["vertical_reference"] == "NAP"


def test_masks_and_physical_orientation_and_spacing(tmp_path):
    prepared = prepared_fixture(tmp_path)
    root = export(prepared, tmp_path / "landscape")
    manifest = json.loads((root / "landscape.json").read_text())
    settings = manifest["import_settings"]
    assert settings["scale_cm"][:2] == [200, 200]
    assert settings["vertex_0_0_base_location_cm"][:2] == [100, 100]
    assert settings["flip_y"] is False
    assert manifest["geographic"]["local_axes"] == ["east", "south", "up"]
    water = np.array(Image.open(root / "layer_Water.png"))
    land = np.array(Image.open(root / "layer_Land.png"))
    np.testing.assert_array_equal(water.astype(int) + land.astype(int), 255)
    np.testing.assert_array_equal(land == 255, prepared.layers["landcover"].data == 1)
    assert (root / "IMPORT.md").is_file()


def test_voids_need_explicit_fill_and_keep_validity(tmp_path):
    heights = np.full((8, 8), -20., dtype=np.float64)
    heights[2, 3] = np.nan
    prepared = prepared_fixture(tmp_path, heights, void=True)
    with pytest.raises(PreparationError, match="landscape_voids"):
        export(prepared, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    root = export(prepared, tmp_path / "filled", void_fill_m=-25)
    valid = np.array(Image.open(root / "validity.png"))
    assert valid[2, 3] == 0
    manifest = json.loads((root / "landscape.json").read_text())
    assert manifest["void_policy"]["filled_vertices"] == 1
    assert manifest["void_policy"]["fill_m"] == -25


@pytest.mark.parametrize("size,sections,components,expected", [
    (63, 1, (10, 10), (631, 631)),
    (63, 4, (32, 32), (4033, 4033)),
    (127, 4, (32, 32), (8129, 8129)),
    (7, 1, (2, 1), (8, 15)),
])
def test_component_contract_from_epic(size, sections, components, expected):
    assert LandscapeLayout(size, sections, *components).shape == expected


@pytest.mark.parametrize("args", [(8, 1, 1, 1), (63, 2, 1, 1), (63, 1, 0, 1), (63, 1, 1.5, 1)])
def test_invalid_layouts_refuse(args):
    with pytest.raises(PreparationError, match="invalid_landscape_layout"):
        LandscapeLayout(*args)


def test_wrong_dimensions_rotated_geographic_and_unmapped_classes_refuse(tmp_path):
    prepared = prepared_fixture(tmp_path)
    prepared.layers["dem"].data = np.ones((9, 9))
    with pytest.raises(PreparationError, match="landscape_dimensions"):
        export(prepared, tmp_path / "bad")
    prepared.layers["dem"].data = np.ones((8, 8))
    from dataclasses import replace
    original = prepared.layers["dem"].grid
    prepared.layers["dem"].grid = replace(original, geotransform=(100000, 2, 1, 440000, 0, -2))
    with pytest.raises(PreparationError, match="landscape_grid_orientation"):
        export(prepared, tmp_path / "bad")
    prepared.layers["dem"].grid = replace(original, crs="EPSG:4326")
    with pytest.raises(PreparationError, match="landscape_projected_crs"):
        export(prepared, tmp_path / "bad")
    prepared.layers["dem"].grid = original
    prepared.layers["landcover"].data[0, 0] = 5
    with pytest.raises(PreparationError, match="landscape_material_mapping"):
        export(prepared, tmp_path / "bad")


def test_editor_verifier_detects_shift_and_height_error(tmp_path):
    prepared = prepared_fixture(tmp_path)
    root = export(prepared, tmp_path / "landscape")
    manifest = json.loads((root / "landscape.json").read_text())
    samples = np.array(Image.open(root / "height.png"))
    decoded = manifest["encoding"]["intercept_m"] + samples.astype(float) * manifest["encoding"]["step_m"]
    rows, cols = np.indices((8, 8))
    positions = np.stack(((cols * 2 + 1) * 100, (rows * 2 + 1) * 100, decoded * 100), axis=-1)
    # Controlled verifier fixture, explicitly not evidence of an Unreal run.
    tolerance = np.spacing(float(np.max(abs(positions)))) / 100 * 4
    result = verify_landscape_import(root, positions, editor_version="controlled test fixture, not an editor run",
                                     measurement_tolerance_m=tolerance)
    assert result["measured_vertices"] == 64
    assert result["material_verified"] is False
    for axis in (0, 1, 2):
        shifted = positions.copy()
        shifted[..., axis] += 100
        with pytest.raises(PreparationError, match="landscape_editor_mismatch"):
            verify_landscape_import(root, shifted, editor_version="controlled test fixture",
                                     measurement_tolerance_m=tolerance)


def test_package_hashes_protect_editor_comparison(tmp_path):
    root = export(prepared_fixture(tmp_path), tmp_path / "landscape")
    (root / "height.png").write_bytes(b"changed")
    with pytest.raises(PreparationError, match="landscape_asset_integrity"):
        verify_landscape_import(root, np.zeros((8, 8, 3)), editor_version="fixture",
                                 measurement_tolerance_m=0)
