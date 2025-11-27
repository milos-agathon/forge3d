import inspect
import os

import pytest

import forge3d as f3d


def test_open_viewer_signature_params_and_kinds() -> None:
    sig = inspect.signature(f3d.open_viewer)
    params = sig.parameters

    # Order and names are part of the public API.
    assert list(params.keys()) == [
        "width",
        "height",
        "title",
        "vsync",
        "fov_deg",
        "znear",
        "zfar",
        "obj_path",
        "gltf_path",
        "snapshot_path",
        "snapshot_width",
        "snapshot_height",
        "initial_commands",
    ]

    # First seven are positional-or-keyword, the rest are keyword-only.
    for name in ["width", "height", "title", "vsync", "fov_deg", "znear", "zfar"]:
        assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in [
        "obj_path",
        "gltf_path",
        "snapshot_path",
        "snapshot_width",
        "snapshot_height",
        "initial_commands",
    ]:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_open_viewer_obj_and_gltf_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="obj_path and gltf_path are mutually exclusive"):
        f3d.open_viewer(obj_path="a.obj", gltf_path="b.gltf")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot_width": 1920},
        {"snapshot_height": 1080},
    ],
)
def test_open_viewer_snapshot_dims_must_be_paired(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="snapshot_width and snapshot_height must be provided together"):
        f3d.open_viewer(**kwargs)


@pytest.mark.parametrize("w,h", [(0, 1080), (1920, 0), (-1, 1080), (1920, -1)])
def test_open_viewer_snapshot_dims_must_be_positive(w: int, h: int) -> None:
    kw = {}
    if w is not None:
        kw["snapshot_width"] = w
    if h is not None:
        kw["snapshot_height"] = h
    with pytest.raises(ValueError, match="must be positive"):
        f3d.open_viewer(**kw)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot_path": "out.png"},
        {"snapshot_width": 1920, "snapshot_height": 1080},
        {
            "snapshot_path": "out.png",
            "snapshot_width": 1920,
            "snapshot_height": 1080,
        },
    ],
)
def test_open_viewer_env_auto_snapshot_conflicts_with_snapshot_args(monkeypatch, kwargs: dict) -> None:
    monkeypatch.setenv("FORGE3D_AUTO_SNAPSHOT_PATH", "auto.png")
    with pytest.raises(ValueError, match=r"FORGE3D_AUTO_SNAPSHOT_PATH and snapshot_\* arguments cannot be used together"):
        f3d.open_viewer(**kwargs)


def test_open_viewer_env_auto_snapshot_allowed_without_snapshot_args(monkeypatch) -> None:
    # This should not raise at validation time; avoid actually launching the viewer by
    # clearing the environment afterwards if needed. The call itself should either
    # succeed or raise a RuntimeError due to missing native module, but not ValueError.
    monkeypatch.setenv("FORGE3D_AUTO_SNAPSHOT_PATH", "auto.png")
    try:
        f3d.open_viewer()
    except (RuntimeError, TypeError):
        # Acceptable outcome when the native extension is not available or not yet
        # rebuilt with the extended open_viewer signature. The key contract is that
        # Python-level validation does not raise ValueError in this configuration.
        pass
