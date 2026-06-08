from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "humanity_globe_video.py"


def load_module():
    spec = importlib.util.spec_from_file_location("humanity_globe_video", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")

    def numpy_to_png(path, array):
        from PIL import Image

        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGBA").save(path)

    forge3d_stub.numpy_to_png = numpy_to_png
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = False
    previous_forge3d = sys.modules.get("forge3d")
    previous_module = sys.modules.get(spec.name)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    sys.modules["forge3d"] = forge3d_stub
    sys.modules[spec.name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_parse_args_defaults_match_reference_video(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["humanity_globe_video.py"])

    args = module.parse_args()

    assert args.size == 720
    assert args.fps == 25
    assert args.duration == pytest.approx(28.8)
    assert args.frames is None
    assert module.frame_count(args) == 720
    assert args.output == module.DEFAULT_OUTPUT
    assert args.preview == module.DEFAULT_PREVIEW


def test_parse_args_accepts_explicit_frame_override(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["humanity_globe_video.py", "--frames", "25", "--size", "360"])

    args = module.parse_args()

    assert args.frames == 25
    assert args.size == 360
    assert module.frame_count(args) == 25
