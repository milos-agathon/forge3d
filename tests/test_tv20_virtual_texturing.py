"""TV20: Terrain material virtual texturing."""
from __future__ import annotations
import pytest


# --- Pure-Python tests (no native module needed) ---

def test_vt_layer_family_defaults():
    from forge3d.terrain_params import VTLayerFamily
    f = VTLayerFamily(family="albedo")
    assert f.tile_size == 248
    assert f.tile_border == 4
    assert f.slot_size == 256
    assert f.pages_x0 == 17  # ceil(4096/248)
    assert f.pages_y0 == 17


def test_vt_layer_family_rejects_unknown():
    from forge3d.terrain_params import VTLayerFamily
    with pytest.raises(ValueError, match="family must be one of"):
        VTLayerFamily(family="diffuse")


def test_vt_layer_family_accepts_normal_mask():
    from forge3d.terrain_params import VTLayerFamily
    VTLayerFamily(family="normal")
    VTLayerFamily(family="mask")


def test_vt_settings_rejects_duplicate_family():
    from forge3d.terrain_params import VTLayerFamily, TerrainVTSettings
    with pytest.raises(ValueError, match="duplicate"):
        TerrainVTSettings(layers=[
            VTLayerFamily(family="albedo"),
            VTLayerFamily(family="albedo"),
        ])


def test_vt_settings_rejects_indivisible_atlas():
    from forge3d.terrain_params import VTLayerFamily, TerrainVTSettings
    with pytest.raises(ValueError, match="divisible"):
        TerrainVTSettings(atlas_size=1000, layers=[
            VTLayerFamily(family="albedo", tile_size=248, tile_border=4),
        ])


def test_vt_settings_actual_mip_count():
    from forge3d.terrain_params import TerrainVTSettings
    s = TerrainVTSettings(max_mip_levels=8)
    assert s.actual_mip_count("albedo") >= 1


def test_vt_layer_pages_at_mip_non_pot():
    from forge3d.terrain_params import VTLayerFamily
    f = VTLayerFamily(family="albedo", virtual_size_px=(3000, 2000), tile_size=248)
    px, py = f.pages_at_mip(0)
    assert px == 13  # ceil(3000/248)
    assert py == 9   # ceil(2000/248)
    px1, py1 = f.pages_at_mip(1)
    assert px1 == 7  # ceil(13/2)
    assert py1 == 5  # ceil(9/2)
