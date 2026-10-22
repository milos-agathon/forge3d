# Smoke test for examples/splat_fused_map.py.
# Verifies: CLI parsing, default offline run, non-blank beauty, hit-kind variety,
# deterministic rerun hash, and report.json structure.
# RELEVANT FILES: examples/splat_fused_map.py, tests/_splat_fusion.py
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forge3d as f3d
from forge3d import splat

from _splat_fusion import gpu_available, FIXTURE_DIR

pytestmark = pytest.mark.skipif(
    not splat.splat_fusion_available(),
    reason="wheel built without the splat-fusion feature",
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "splat_fused_map.py"


def _load_module():
    import importlib.util
    examples_dir = str(EXAMPLE.parent.resolve())
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("splat_fused_map", str(EXAMPLE))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_example_module_loads():
    mod = _load_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "build_parser")


def test_parser_defaults():
    mod = _load_module()
    p = mod.build_parser()
    args = p.parse_args([])
    assert args.width == 800
    assert args.height == 560
    assert args.samples == 64
    assert args.seed == 7
    assert args.dem is None
    assert args.pointcloud is None
    assert args.splats is None
    assert args.no_lidar is False
    assert args.view is None


def test_hdr_cache_is_checksum_validated(tmp_path, monkeypatch):
    """A cached preset must be md5-checked, not trusted on size alone."""
    monkeypatch.setenv("HOME", str(tmp_path))
    mod = _load_module()
    cache = tmp_path / ".forge3d" / "datasets" / "hdri" / f"{mod.HDR_PRESET}_2k.hdr"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"corrupt" + b"\x00" * 4000)  # non-trivial size, wrong hash

    def _offline(*args, **kwargs):  # noqa: ARG001
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlretrieve", _offline)
    # Old behavior returned the corrupt path; a validated cache falls back.
    assert mod._resolve_hdr("auto", use_dem=True) is None
    assert not cache.exists(), "the rejected cache must be discarded"

    # A cache whose hash matches the preset is accepted without any network.
    monkeypatch.setattr(mod, "HDR_PRESET_MD5", hashlib.md5(cache_bytes := b"good" * 2000).hexdigest())
    cache.write_bytes(cache_bytes)
    assert mod._resolve_hdr("auto", use_dem=True) == str(cache)


@pytest.mark.parametrize(
    "argv",
    [
        # A camera needs both halves; one alone is a mistake, not a request
        # for the default camera.
        ["--look-at", "1", "2", "3"],
        ["--origin", "1", "2", "3"],
        # A rotation needs an environment to rotate. "auto" without a --dem
        # render never names one, and "none" explicitly refuses one.
        ["--hdr-rotation", "30"],
        ["--hdr-rotation", "30", "--hdr", "none"],
        # The HDRI must be a format the seam can read.
        ["--hdr", "sky.png"],
        # An explicit point cloud that is not there must not be replaced by a
        # synthetic one.
        ["--pointcloud", "missing.copc.laz"],
    ],
)
def test_bad_arguments_exit_with_a_clean_error(tmp_path, argv, capsys):
    """Every argument error prints ``error: ...`` and exits 2, not a traceback."""
    mod = _load_module()
    code = mod.main([*argv, "--out", str(tmp_path / "out")])
    assert code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err, f"no error message on stderr: {captured.err!r}"
    assert "Traceback" not in captured.err, "the error was an uncaught traceback"


def test_the_dem_download_is_checked_against_its_pointer(tmp_path, monkeypatch):
    """The LFS pointer names the SHA256 of the real bytes: use it.

    The HDRI preset is md5-checked on every hit; the DEM download was
    trusted on size alone, so a truncated or corrupted download rendered as
    a wrong terrain.
    """
    mod = _load_module()
    payload = b"not a real raster, but the hash is what matters" * 100
    digest = hashlib.sha256(payload).hexdigest()
    pointer = tmp_path / "dem.tif"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\n"
        f"size {len(payload)}\n"
    )
    assert mod._lfs_sha256(pointer) == digest

    # A file that is not a pointer has no hash to check.
    plain = tmp_path / "plain.tif"
    plain.write_bytes(payload)
    assert mod._lfs_sha256(plain) is None

    # A download whose bytes do not match the pointer is not trusted: the
    # corrupt cache is discarded and re-fetched, not rendered as a wrong
    # terrain. (The re-fetch itself needs the network, so the run may end in
    # a download error — what matters is that the bad cache is gone.)
    cache = tmp_path / ".forge3d" / "datasets"
    cache.mkdir(parents=True)
    bad = cache / "dem.tif"
    bad.write_bytes(b"corrupted" * 200)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(RuntimeError):
        mod._resolve_dem_path(pointer)
    assert not bad.exists(), "the corrupt cache was kept"

    # And a cache that DOES match is used as it is.
    good = cache / "dem.tif"
    good.write_bytes(payload)
    assert mod._resolve_dem_path(pointer) == good


def test_default_run_nonblank_and_deterministic(tmp_path):
    if not gpu_available():
        pytest.skip("no GPU adapter available")
    try:
        import PIL  # noqa: F401
    except ImportError:
        pytest.skip("Pillow not available in this environment")
    import subprocess
    import os

    env = os.environ.copy()
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out in (out1, out2):
        subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--out",
                str(out),
                "--samples",
                "8",
                "--certificate",
            ],
            check=True,
            env=env,
            cwd=str(REPO),
        )
    # Beauty must be non-blank (not all one color).
    import numpy as np

    # The certificate must be the one the BEAUTY render produced, and it
    # must be signed. Reading it back from the diagnostics module after the
    # run would give the LAST render in the process (the last fly-through
    # frame), not the sheet's.
    cert_path = out1 / "certificate.json"
    assert cert_path.exists(), "the certificate was not written"
    cert = json.loads(cert_path.read_text())
    assert "signature" in cert, "the certificate was written unsigned"
    assert cert["inputs"]["fused.seed"] == "7"

    # Check non-blank by reading raw PNG bytes size (a blank image would be tiny).
    # Also verify the file exists and is a valid PNG.
    beauty_path = out1 / "splat_fused_beauty.png"
    assert beauty_path.exists()
    assert beauty_path.stat().st_size > 10000, "beauty PNG is suspiciously small (likely blank)"
    # Deterministic rerun already checked via hash below.
    # Deterministic rerun.
    h1 = hashlib.sha256((out1 / "splat_fused_beauty.png").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out2 / "splat_fused_beauty.png").read_bytes()).hexdigest()
    assert h1 == h2, "rerun is not deterministic"
    # The SHEET is what the sheet's own footer promises ("seed N reproduces
    # this sheet byte-for-byte"), so it earns the same byte-identity check as
    # the beauty. A sheet that drifts between runs would make that claim
    # false while the beauty test still passed.
    sheet1 = out1 / "splat_fused_map.png"
    sheet2 = out2 / "splat_fused_map.png"
    assert sheet1.exists() and sheet1.stat().st_size > 10000, "sheet PNG is missing or blank"
    s1 = hashlib.sha256(sheet1.read_bytes()).hexdigest()
    s2 = hashlib.sha256(sheet2.read_bytes()).hexdigest()
    assert s1 == s2, "the map sheet is not byte-identical across reruns"
    # Report structure.
    report = json.loads((out1 / "report.json").read_text())
    assert "hit_counts" in report
    assert "stats" in report
    assert report["stats"]["logical_primitives"] > 0
    # At least terrain or splat hits (not all miss).
    hc = report["hit_counts"]
    assert hc["terrain"] + hc["splat"] + hc["lidar"] > 0, "no surface hits at all"