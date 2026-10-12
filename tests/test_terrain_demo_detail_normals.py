"""DEM-detail CLI/preset forwarding, real generation and physical demo rendering."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from forge3d import terrain_demo as demo
from _terrain_runtime import terrain_rendering_available
from test_terrain_demo_cli_smoke import _load_module_by_path
from test_terrain_detail_normals import write_ramp_map


ROOT = Path(__file__).resolve().parents[1]


def detail_args(tmp_path, **overrides):
    values = dict(detail_normals=None, detail_strength=0.0, detail_sigma_px=3.0,
                  detail_fade_start=None, detail_fade_end=None, render=None,
                  dem=tmp_path / "dem.tif", output=tmp_path / "render.png",
                  overwrite=False, _explicit_cli_args=set())
    values.update(overrides)
    return SimpleNamespace(**values)


def test_demo_detail_defaults_and_forwarding(tmp_path, monkeypatch):
    assert demo._build_detail_settings(detail_args(tmp_path)) is None
    detail = demo._build_detail_settings(detail_args(
        tmp_path, detail_normals=tmp_path / "detail.png", detail_strength=0.75,
        detail_sigma_px=2.0, detail_fade_start=1000.0, detail_fade_end=2000.0,
    ))
    assert detail.enabled and detail.detail_strength == 0.75
    assert detail.detail_sigma_px == 2.0
    assert (detail.fade_start, detail.fade_end) == (1000.0, 2000.0)
    assert detail.normal_strength == detail.albedo_noise == 0.0
    native_params = f3d.TerrainRenderParams
    monkeypatch.setattr(f3d, "TerrainRenderParams", lambda config: config)
    config = demo._build_params(
        size=(64, 64), render_scale=1.0, terrain_span=6.0, msaa=1,
        z_scale=1.0, exposure=1.0, domain=(0.0, 1.0),
        colormap=f3d.Colormap1D.from_stops([(0.0, "#ffffff"), (1.0, "#ffffff")], domain=(0.0, 1.0)),
        detail_config=detail,
    )
    assert config.detail is detail
    native_params(config)  # Verify the forwarded configuration actually decodes.


@pytest.mark.parametrize("field,value,error", [
    ("detail_strength", -0.1, "detail_strength"),
    ("detail_strength", float("nan"), "detail_strength"),
    ("detail_sigma_px", 0.0, "detail_sigma_px"),
    ("detail_fade_start", float("nan"), "fade_start"),
    ("detail_fade_end", float("inf"), "fade_end"),
])
def test_demo_rejects_invalid_detail_flags(tmp_path, field, value, error):
    with pytest.raises(ValueError, match=error):
        demo._build_detail_settings(detail_args(tmp_path, **{field: value}))


def test_demo_positive_strength_requires_path(tmp_path):
    with pytest.raises(ValueError, match="detail_normal_path"):
        demo._build_detail_settings(detail_args(tmp_path, detail_strength=1.0))


@pytest.mark.parametrize("flag", [["--detail-strength", "0"], ["--detail-strength=0"]])
def test_p6_explicit_zero_does_not_generate(tmp_path, monkeypatch, flag):
    def unexpected(*args, **kwargs):
        pytest.fail("explicit zero strength must not generate a map")
    monkeypatch.setattr("subprocess.run", unexpected)
    cli = _load_module_by_path(ROOT / "examples/terrain_demo.py")
    monkeypatch.setattr("sys.argv", ["terrain_demo", "--render", "p6", *flag])
    assert demo._build_detail_settings(cli._parse_args()) is None


@pytest.mark.parametrize("returncode", [0, 1])
def test_p6_generation_failure_is_explicit(tmp_path, monkeypatch, returncode):
    def failed(command, **kwargs):
        assert Path(command[1]) == ROOT / "scripts" / "detail_normals.py"
        assert Path(command[1]).is_file()
        assert "--flip-y" in command
        return SimpleNamespace(returncode=returncode, stderr="generation failed", stdout="")
    monkeypatch.setattr("subprocess.run", failed)
    with pytest.raises(RuntimeError, match="detail_normal_path generation failed"):
        demo._build_detail_settings(detail_args(tmp_path, render="p6"))


@pytest.mark.parametrize("explicit", [False, True, "equals"])
def test_detail_json_preset_honors_cli_overrides(tmp_path, monkeypatch, explicit):
    cli = _load_module_by_path(ROOT / "examples" / "terrain_demo.py")
    argv = ["terrain_demo"]
    if explicit:
        argv += ["--detail-normals", str(tmp_path / "cli.png"),
                 "--detail-strength", "0", "--detail-sigma-px", "4",
                 "--detail-fade-start", "800", "--detail-fade-end", "1600"]
        if explicit == "equals":
            argv = [argv[0]] + [f"{argv[i]}={argv[i + 1]}" for i in range(1, len(argv), 2)]
    monkeypatch.setattr("sys.argv", argv)
    args = cli._parse_args()
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"detail_normals": {
        "enabled": True, "detail_normal_path": "assets/preset.png",
        "detail_strength": 0.75, "detail_sigma_px": 2.0,
        "fade_start": 1000.0, "fade_end": 2000.0,
    }}), encoding="utf-8")
    cli._apply_json_preset(args, preset)
    detail = demo._build_detail_settings(args)
    assert detail.detail_normal_path == str(tmp_path / "cli.png" if explicit else ROOT / "assets/preset.png")
    assert detail.detail_strength == (0.0 if explicit else 0.75)
    assert detail.detail_sigma_px == (4.0 if explicit else 2.0)
    assert (detail.fade_start, detail.fade_end) == ((800.0, 1600.0) if explicit else (1000.0, 2000.0))
    assert detail.enabled is (not explicit)


def test_p6_real_generation_matches_demo_dem_orientation(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("scipy.ndimage")
    rows, cols = np.mgrid[:32, :32].astype(np.float32)
    height = np.sin(rows * 0.4) * 2.0 + np.cos(cols * 0.2)
    args = detail_args(tmp_path, render="p6", detail_sigma_px=2.0)
    with rasterio.open(args.dem, "w", driver="GTiff", width=32, height=32,
                       count=1, dtype="float32", crs="EPSG:32610",
                       transform=rasterio.transform.from_origin(100.0, 200.0, 1.0, 1.0)) as dst:
        dst.write(height, 1)
    detail = demo._build_detail_settings(args)
    assert detail.enabled and detail.detail_strength == 0.5
    generated = Path(detail.detail_normal_path)
    assert generated == args.output.with_suffix(".detail-normal.png")
    spec = importlib.util.spec_from_file_location("detail_encoder", ROOT / "scripts/detail_normals.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    expected = encoder.detail_to_normal_map(encoder.compute_detail_residual(np.flipud(height), 2.0))
    np.testing.assert_array_equal(np.asarray(Image.open(generated)), expected)
    assert json.loads(generated.with_suffix(".json").read_text())["flip_y"] is True
    # Existing generated output must raise, rather than silently replay stale data.
    with pytest.raises(RuntimeError, match="detail_normal_path generation failed"):
        demo._build_detail_settings(args)


def test_demo_entry_point_renders_detail_map_on_physical_gpu(tmp_path, monkeypatch):
    if not terrain_rendering_available():
        pytest.skip("terrain demo requires a physical GPU")
    probe = f3d.device_probe()
    assert probe["device_type"] in {"DiscreteGpu", "IntegratedGpu"}, probe
    print(f"Terrain demo detail physical adapter: {probe}")
    dem = tmp_path / "dem.npy"
    np.save(dem, np.full((32, 32), 0.5, dtype=np.float32))
    hdr = tmp_path / "env.hdr"
    hdr.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n" + bytes([128, 128, 128, 128]) * 4)
    normal = tmp_path / "east.png"
    write_ramp_map(normal, "east")
    cli = _load_module_by_path(ROOT / "examples/terrain_demo.py")
    frames = []
    for strength in (0.0, 1.0):
        output = tmp_path / f"render-{strength}.png"
        monkeypatch.setattr("sys.argv", [
            "terrain_demo", "--dem", str(dem), "--hdr", str(hdr), "--output", str(output),
            "--size", "64", "64", "--msaa", "1", "--z-scale", "1",
            "--colormap-domain", "0", "1",
            "--cam-radius", "4", "--cam-theta", "55", "--camera-mode", "screen",
            "--sun-azimuth", "0", "--sun-elevation", "30", "--sun-intensity", "1",
            "--ibl-intensity", "0", "--pom-disabled", "--shadow-technique", "none",
            "--albedo-mode", "colormap", "--colormap-strength", "1", "--normal-strength", "0",
            "--detail-normals", str(normal), "--detail-strength", str(strength),
            "--detail-fade-start", "100", "--detail-fade-end", "200",
        ])
        assert demo.run(cli._parse_args()) == 0
        frames.append(np.asarray(Image.open(output)).copy())
    assert not np.array_equal(*frames)
