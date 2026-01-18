# tests/test_bundle_cli.py
"""CLI smoke tests for --save-bundle and --load-bundle flags."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
from pathlib import Path

import pytest


def _load_terrain_demo() -> types.ModuleType:
    """Load terrain_demo.py module by path."""
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "terrain_demo", str(repo / "examples" / "terrain_demo.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_bundle_flag_exists():
    """--save-bundle flag is recognized by argparse."""
    mod = _load_terrain_demo()
    # Create a minimal parser and check the flag exists
    import argparse
    import sys
    
    # Temporarily modify sys.argv
    original_argv = sys.argv
    try:
        sys.argv = ["terrain_demo.py", "--save-bundle", "/tmp/test.forge3d"]
        args = mod._parse_args()
        assert args.save_bundle == Path("/tmp/test.forge3d")
    finally:
        sys.argv = original_argv


def test_load_bundle_flag_exists():
    """--load-bundle flag is recognized by argparse."""
    mod = _load_terrain_demo()
    import sys
    
    original_argv = sys.argv
    try:
        sys.argv = ["terrain_demo.py", "--load-bundle", "/tmp/test.forge3d"]
        args = mod._parse_args()
        assert args.load_bundle == Path("/tmp/test.forge3d")
    finally:
        sys.argv = original_argv


def test_was_cli_set_detects_explicit_flags():
    """_was_cli_set correctly detects explicitly set CLI flags."""
    mod = _load_terrain_demo()
    import argparse
    import sys
    
    original_argv = sys.argv
    try:
        sys.argv = ["terrain_demo.py", "--exposure", "2.0", "--z-scale", "3.0"]
        args = mod._parse_args()
        
        assert mod._was_cli_set(args, "exposure") is True
        assert mod._was_cli_set(args, "z_scale") is True
        assert mod._was_cli_set(args, "colormap") is False
        assert mod._was_cli_set(args, "cam_phi") is False
    finally:
        sys.argv = original_argv


def test_bundle_directory_structure():
    """save_bundle creates expected directory structure."""
    from forge3d.bundle import save_bundle
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "structure_test"
        
        save_bundle(
            bundle_path,
            name="structure_test",
            preset={"test": True},
        )
        
        # Check expected directories exist
        assert bundle_path.with_suffix(".forge3d").exists()
        bundle = bundle_path.with_suffix(".forge3d")
        assert (bundle / "manifest.json").exists()
        assert (bundle / "terrain").is_dir()
        assert (bundle / "overlays").is_dir()
        assert (bundle / "camera").is_dir()
        assert (bundle / "render").is_dir()
        assert (bundle / "assets").is_dir()


def test_manifest_has_version_1():
    """Bundle manifest has version == 1."""
    from forge3d.bundle import save_bundle, BUNDLE_VERSION
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "version_test.forge3d"
        
        save_bundle(bundle_path, name="version_test")
        
        manifest_path = bundle_path / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest["version"] == BUNDLE_VERSION
        assert manifest["version"] == 1


def test_bundle_extension_added_if_missing():
    """save_bundle adds .forge3d extension if not present."""
    from forge3d.bundle import save_bundle
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "no_extension"
        
        result = save_bundle(bundle_path, name="no_extension")
        
        assert result.suffix == ".forge3d"
        assert result.exists()
