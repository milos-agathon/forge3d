import builtins
import sys

import numpy as np
import pytest

import forge3d as f3d
from forge3d.helpers.offscreen import rgba_to_png_bytes


def _deny_pillow_imports(monkeypatch) -> None:
    real_import = builtins.__import__

    for name in tuple(sys.modules):
        if name == "PIL" or name.startswith("PIL."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            np.array([[0, 128], [255, 64]], dtype=np.uint8),
            np.array(
                [
                    [[0, 0, 0, 255], [128, 128, 128, 255]],
                    [[255, 255, 255, 255], [64, 64, 64, 255]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[10, 20, 30, 255], [40, 50, 60, 255]],
                    [[70, 80, 90, 255], [100, 110, 120, 255]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            np.array(
                [
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_public_png_roundtrip_without_pillow(monkeypatch, tmp_path, src, expected):
    _deny_pillow_imports(monkeypatch)

    path = tmp_path / "roundtrip.png"
    f3d.numpy_to_png(path, src)
    loaded = f3d.png_to_numpy(path)

    assert np.array_equal(loaded, expected)


def test_rgba_to_png_bytes_without_pillow(monkeypatch, tmp_path):
    _deny_pillow_imports(monkeypatch)

    src = np.array(
        [
            [[12, 34, 56, 255], [78, 90, 123, 200]],
            [[45, 67, 89, 128], [210, 220, 230, 64]],
        ],
        dtype=np.uint8,
    )

    data = rgba_to_png_bytes(src)
    path = tmp_path / "bytes.png"
    path.write_bytes(data)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert np.array_equal(f3d.png_to_numpy(path), src)
