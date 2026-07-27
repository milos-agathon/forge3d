from __future__ import annotations

import numpy as np

import forge3d as f3d
from _deltae import delta_e_2000, srgb_to_lab
from _ssim import ssim


def _smooth_albedo(side: int = 64) -> np.ndarray:
    y, x = np.mgrid[:side, :side].astype(np.float32)
    rgba = np.empty((side, side, 4), dtype=np.uint8)
    rgba[..., 0] = np.round(72.0 + x * 0.45).astype(np.uint8)
    rgba[..., 1] = np.round(96.0 + y * 0.40).astype(np.uint8)
    rgba[..., 2] = np.round(128.0 + (x + y) * 0.20).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def test_bc7_mode6_clears_tessella_fidelity_gate():
    source = _smooth_albedo()
    height, width = source.shape[:2]
    encoded = f3d.encode_bc7_rgba8(source.tobytes(), width, height)
    decoded = np.frombuffer(
        f3d.decode_bc7_rgba8(encoded, width, height), dtype=np.uint8
    ).reshape(source.shape)

    measured_ssim = ssim(source[..., :3], decoded[..., :3])
    mean_delta_e = float(
        delta_e_2000(
            srgb_to_lab(source[..., :3]), srgb_to_lab(decoded[..., :3])
        ).mean()
    )
    assert len(encoded) * 4 == source.nbytes
    assert measured_ssim >= 0.98, measured_ssim
    assert mean_delta_e < 1.5, mean_delta_e


def test_bc5_normal_angular_error_clears_gate():
    side = 64
    y, x = np.mgrid[:side, :side].astype(np.float32)
    nx = (x / (side - 1) - 0.5) * 0.28
    ny = (y / (side - 1) - 0.5) * 0.28
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0))
    source_normals = np.stack([nx, ny, nz], axis=-1)
    rg = np.round((source_normals[..., :2] * 0.5 + 0.5) * 255.0).astype(
        np.uint8
    )

    encoded = f3d.encode_bc5_rg8(rg.tobytes(), side, side)
    decoded_rg = np.frombuffer(
        f3d.decode_bc5_rg8(encoded, side, side), dtype=np.uint8
    ).reshape(side, side, 2)
    decoded_xy = decoded_rg.astype(np.float32) / 127.5 - 1.0
    decoded_z = np.sqrt(
        np.maximum(1.0 - np.sum(decoded_xy * decoded_xy, axis=-1), 0.0)
    )
    decoded_normals = np.dstack([decoded_xy, decoded_z])
    dots = np.clip(
        np.sum(source_normals * decoded_normals, axis=-1), -1.0, 1.0
    )
    angular = np.degrees(np.arccos(dots))

    assert len(encoded) * 2 == rg.nbytes
    assert float(angular.mean()) < 1.0, float(angular.mean())
    assert float(angular.max()) < 4.0, float(angular.max())
