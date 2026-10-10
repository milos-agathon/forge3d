"""Validated content-addressed D04 snapshot assets for version 4 bundles."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping

from .render_pass import RenderPassInput


def write_snapshots(root: Path, snapshots: Mapping[str, RenderPassInput],
                    checksums: dict[str, str]) -> None:
    for image in snapshots.values():
        descriptor = image._asset_descriptor()
        relative = descriptor['asset']
        if relative in checksums:
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('render pass snapshot escapes bundle root')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image._binary)
        checksums[relative] = descriptor['sha256']


def load_snapshots(root: Path, payload: Mapping[str, Any],
                   checksums: Mapping[str, str]) -> dict[str, RenderPassInput]:
    import numpy as np

    snapshots = {}
    for name, descriptor in payload.get('inputs', {}).items():
        digest = descriptor.get('sha256')
        if not isinstance(digest, str) or len(digest) != hashlib.sha256().digest_size * 2 or any(
                value not in '0123456789abcdef' for value in digest):
            raise ValueError('invalid render pass snapshot SHA-256')
        relative = f'scene/render_pass_inputs/{digest}.npy'
        if descriptor.get('asset') != relative or checksums.get(relative) != digest:
            raise ValueError('render pass snapshot reference/checksum mismatch')
        # Refuse symlinks or junctions escaping the declared bundle root.
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('render pass snapshot escapes bundle root')
        binary = path.read_bytes()
        if hashlib.sha256(binary).hexdigest() != digest:
            raise ValueError('render pass snapshot checksum mismatch')
        stream = io.BytesIO(binary)
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError('render pass snapshots require NumPy format 1.0')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if (dtype.str != '<f8' or fortran or list(shape) != descriptor.get('shape')
                or descriptor.get('dtype') != '<f8' or len(shape) != 3
                or shape[2] != 4 or any(not isinstance(value, int) or value <= 0 for value in shape)
                or len(binary) - stream.tell() != math.prod(shape) * dtype.itemsize):
            raise ValueError('render pass snapshot dtype/dimensions mismatch')
        data = np.frombuffer(binary, dtype=dtype, offset=stream.tell()).reshape(shape)
        image = RenderPassInput(data, color_space=descriptor['color_space'], alpha_mode=descriptor['alpha_mode'])
        if image._asset_descriptor() != descriptor:
            raise ValueError('noncanonical render pass snapshot')
        snapshots[name] = image
    return snapshots
