# tests/test_gallery_smoke.py
# CI-friendly smoke tests for the gallery examples. These tests avoid pixel
# equality and only assert that the scripts render mosaics with plausible
# dimensions and create the expected files. They work with or without a GPU.
from __future__ import annotations

import types
from pathlib import Path
import importlib.util

import numpy as np
import pytest


def _load_module_by_path(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _maybe_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            return (int(im.width), int(im.height))
    except Exception:
        return None


@pytest.mark.parametrize("tile_size,frames", [(64, 1)])
def test_lighting_gallery_smoke(tmp_path: Path, tile_size: int, frames: int) -> None:
    repo = Path(__file__).resolve().parents[1]
    mod = _load_module_by_path(repo / "examples" / "lighting_gallery.py")

    mesh = repo / "assets" / "bunny.obj"
    outdir = tmp_path / "out"
    outpng = tmp_path / "lighting_gallery.png"

    mod.render_lighting_gallery(
        mesh_path=mesh,
        output_path=outpng,
        outdir=outdir,
        tile_size=int(tile_size),
        frames=int(frames),
        save_tiles=False,
    )

    # Either PNG or (rare) .npy fallback
    assert outpng.exists() or outpng.with_suffix(".npy").exists()

    size = _maybe_image_size(outpng)
    if size is not None:
        w, h = size
        assert w % tile_size == 0 and h % tile_size == 0


@pytest.mark.parametrize("techniques,tile_size,map_res", [(["Hard", "PCF"], 64, 512)])
def test_shadow_gallery_smoke(tmp_path: Path, techniques, tile_size: int, map_res: int) -> None:
    repo = Path(__file__).resolve().parents[1]
    mod = _load_module_by_path(repo / "examples" / "shadow_gallery.py")

    outdir = tmp_path / "out"
    outpng = tmp_path / "shadow_gallery.png"
    hdr = repo / "assets" / "snow_field_4k.hdr"

    mod.render_shadow_comparison(
        techniques=techniques,
        output_path=outpng,
        hdr_path=hdr,
        outdir=outdir,
        map_res=int(map_res),
        tile_size=int(tile_size),
        cols=2,
        save_tiles=False,
    )

    assert outpng.exists() or outpng.with_suffix(".npy").exists()
    size = _maybe_image_size(outpng)
    if size is not None:
        w, h = size
        assert w % tile_size == 0 and h % tile_size == 0


@pytest.mark.parametrize("tile_size", [64])
def test_ibl_gallery_smoke(tmp_path: Path, tile_size: int) -> None:
    repo = Path(__file__).resolve().parents[1]
    mod = _load_module_by_path(repo / "examples" / "ibl_gallery.py")

    hdr = repo / "assets" / "snow_field_4k.hdr"
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    # Rotation sweep (native or fallback)
    out_rot = tmp_path / "ibl_rotation.png"
    mod.render_rotation_sweep_native_or_fallback(
        hdr_path=hdr,
        output_path=out_rot,
        tile_size=int(tile_size),
        rotation_steps=4,
        outdir=outdir,
    )
    assert out_rot.exists() or out_rot.with_suffix(".npy").exists()

    # Roughness sweep (mesh tracer)
    out_rough = tmp_path / "ibl_roughness.png"
    mod.render_roughness_sweep_mesh(
        hdr_path=hdr,
        output_path=out_rough,
        tile_size=int(tile_size),
        roughness_steps=3,
        outdir=outdir,
    )
    assert out_rough.exists() or out_rough.with_suffix(".npy").exists()

    # Metallic comparison (mesh tracer)
    out_metal = tmp_path / "ibl_metallic.png"
    mod.render_metallic_comparison_mesh(
        hdr_path=hdr,
        output_path=out_metal,
        tile_size=int(tile_size),
        outdir=outdir,
    )
    assert out_metal.exists() or out_metal.with_suffix(".npy").exists()
