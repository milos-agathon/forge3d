from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.diagnostics import render_certificate, visibility_stats, vt_stats
from forge3d.mem import memory_metrics
from forge3d.terrain_params import (
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)
from test_terrain_clipmap_streaming import _render_rgba


VIRTUAL_SIDE = 1 << 18
LOGICAL_MIN_BYTES = 256 * 1024**3


def _build_sparse_store(tmp_path: Path) -> tuple[f3d.VTStore, dict]:
    store_path = tmp_path / "switzerland-procedural.f3dvt"
    manifest_path = tmp_path / "switzerland-procedural.manifest.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--bin",
            "forge3d-vtpack",
            "--",
            "--procedural",
            "--output",
            str(store_path),
            "--manifest",
            str(manifest_path),
            "--virtual-width",
            str(VIRTUAL_SIDE),
            "--virtual-height",
            str(VIRTUAL_SIDE),
            "--tile-size",
            "128",
            "--tile-border",
            "0",
            "--seed",
            "19",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(result.stdout)
    return f3d.open_vt_store(store_path), manifest


def test_sparse_store_declares_at_least_256_gib_without_allocating_it(tmp_path):
    store, manifest = _build_sparse_store(tmp_path)
    assert int(manifest["logical_texel_bytes"]) >= LOGICAL_MIN_BYTES
    assert manifest["procedural"] is True
    assert manifest["page_order"] == "family,mip,morton2"
    assert Path(store.path).stat().st_size < 4096
    assert Path(store.path).read_bytes()[:8] == b"F3DVT1\0\0"


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_256_gib_store_settles_within_eight_frames_under_host_budget(tmp_path):
    store, manifest = _build_sparse_store(tmp_path)
    hdr = tmp_path / "probe.hdr"
    _write_test_hdr(hdr)
    ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    layers = [
        VTLayerFamily(
            family=family,
            virtual_size_px=(VIRTUAL_SIDE, VIRTUAL_SIDE),
            tile_size=128,
            tile_border=0,
        )
        for family in ("albedo", "normal", "mask")
    ]
    config = make_terrain_params_config(
        size_px=(3840, 2160),
        terrain_span=50.0,
        camera_mode="clipmap:4:32:32:10:0.3",
        culling="frustum",
        shading="visibility",
        vt_store=store,
        vt_upload_budget_bytes=64 * 1024 * 1024,
        prefetch_horizon_ms=100.0,
        vt=TerrainVTSettings(
            enabled=True,
            layers=layers,
            atlas_size=4096,
            residency_budget_mb=192.0,
            max_mip_levels=8,
            use_feedback=True,
        ),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    params = f3d.TerrainRenderParams(config)
    dem = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)

    stats = {}
    for settling_frame in range(1, 9):
        frame = _render_rgba(renderer, params, dem, ibl)
        stats = renderer.get_material_vt_stats()
        if stats["retained_requests"] == 0 and stats["miss_rate"] == 0:
            break

    assert frame.shape == (2160, 3840, 4)
    assert settling_frame <= 8
    assert stats["retained_requests"] == 0
    assert stats["evictions"] <= stats["tiles_streamed"]
    public_stats = vt_stats()
    assert all(public_stats[key] == value for key, value in stats.items())
    assert stats["atlas_device_local_bytes"] > 0
    assert stats["atlas_uncompressed_equivalent_bytes"] >= stats["atlas_device_local_bytes"]
    assert stats["atlas_compression_ratio"] >= 1.0
    assert visibility_stats()["fallback_texels"] == 0
    degradations = render_certificate(sign=False)["degradations"]
    assert not {
        "terrain_vt_bc_atlas",
        "terrain_vt_bindless_atlas",
    }.intersection(entry["name"] for entry in degradations), degradations
    metrics = memory_metrics()
    peak_host = max(
        metrics.get("peak_host_visible_bytes", 0),
        metrics.get("host_visible_bytes", 0),
    )
    assert peak_host < 512 * 1024**2, {
        "peak_host_visible_bytes": peak_host,
        "manifest": manifest,
        "vt_stats": stats,
    }
    record_tessella_result(
        "vt_out_of_core",
        {
            "logical_texel_bytes": int(manifest["logical_texel_bytes"]),
            "sparse_store_bytes": Path(store.path).stat().st_size,
            "settling_frames": settling_frame,
            "retained_requests": int(stats["retained_requests"]),
            "miss_rate": float(stats["miss_rate"]),
            "fallback_texels": int(visibility_stats()["fallback_texels"]),
            "atlas_device_local_bytes": int(stats["atlas_device_local_bytes"]),
            "atlas_uncompressed_equivalent_bytes": int(
                stats["atlas_uncompressed_equivalent_bytes"]
            ),
            "atlas_compression_ratio": float(stats["atlas_compression_ratio"]),
            "peak_host_visible_bytes": int(peak_host),
        },
    )
