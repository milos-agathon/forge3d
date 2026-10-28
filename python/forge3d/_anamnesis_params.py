"""Private identity of the input decoded by TerrainRenderParams at construction."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import os
import struct

import numpy as np


def input_snapshot(params):
    """Hash owned values without deepcopying large LUT or water arrays.

    Unsupported custom values raise: native code keeps ordinary rendering
    available and reports parameter_identity_unavailable when cache is asked for.
    This digest supplements the live input identity, since some native fields
    are decoded once while other Python settings remain live references.
    """
    digest = hashlib.sha256(b"forge3d.terrain.decoded-input/v1\0")

    def token(tag, data=b""):
        digest.update(tag)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    def visit(value):
        if value is None:
            token(b"n")
        elif isinstance(value, (bool, np.bool_)):
            token(b"b", bytes([bool(value)]))
        elif isinstance(value, (int, np.integer)):
            token(b"i", str(int(value)).encode("ascii"))
        elif isinstance(value, (float, np.floating)):
            token(b"f", struct.pack("<d", float(value)))
        elif isinstance(value, str):
            token(b"s", value.encode("utf-8"))
        elif isinstance(value, bytes):
            token(b"y", value)
        elif isinstance(value, os.PathLike):
            token(b"p")
            visit(os.fspath(value))
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays have no native terrain identity")
            array = np.ascontiguousarray(value)
            token(b"a", array.dtype.str.encode("ascii"))
            visit(array.shape)
            token(b"h", hashlib.sha256(memoryview(array).cast("B")).digest())
        elif is_dataclass(value) and not isinstance(value, type):
            token(b"d")
            for field in fields(value):
                visit(field.name)
                visit(getattr(value, field.name))
            token(b"e")
        elif isinstance(value, Mapping):
            token(b"m")
            for name in sorted(value):
                visit(name)
                visit(value[name])
            token(b"e")
        elif isinstance(value, (tuple, list)):
            token(b"l")
            for item in value:
                visit(item)
            token(b"e")
        else:
            from .terrain_params import _WaterSurfaceGrid
            if isinstance(value, _WaterSurfaceGrid):
                visit(np.asarray(value))
            else:
                raise TypeError(f"unsupported native terrain input: {type(value).__name__}")

    visit(params)
    return digest.digest()
