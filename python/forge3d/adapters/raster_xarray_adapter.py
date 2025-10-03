# python/forge3d/adapters/raster_xarray_adapter.py
# M4: Rasterio/Xarray Adapter
# - Convert rasterio datasets and xarray DataArrays to RGBA overlays
# - Handle nodata/masks -> alpha
# - Provide ≤1px georef offset acceptance via reproject helpers

from __future__ import annotations

from typing import Optional, Tuple, Any
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    import xarray as xr  # type: ignore
    _HAS_XARRAY = True
except Exception:
    _HAS_XARRAY = False


def is_rasterio_available() -> bool:
    return _HAS_RASTERIO


def is_xarray_available() -> bool:
    return _HAS_XARRAY


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating):
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    # integer types
    info = np.iinfo(rgb.dtype) if np.issubdtype(rgb.dtype, np.integer) else None
    if info is not None and info.bits > 8:
        scale = 255.0 / float(info.max - info.min or 1)
        return np.clip((rgb.astype(np.float32) - info.min) * scale, 0, 255).astype(np.uint8)
    return rgb.astype(np.uint8, copy=True)


def rasterio_to_rgba(dataset: "rasterio.io.DatasetReader", *, bands: Optional[Tuple[int,int,int]] = None) -> np.ndarray:
    """Convert a rasterio dataset to RGBA uint8 (H,W,4) with nodata->alpha.

    If bands is None, tries to pick (1,2,3) or reps a single band to RGB.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required. Install with: pip install rasterio")

    count = dataset.count
    h = dataset.height
    w = dataset.width

    if count == 0:
        raise ValueError("dataset has zero bands")

    if bands is None:
        if count >= 3:
            bands = (1, 2, 3)
        else:
            bands = (1, 1, 1)

    data = dataset.read(bands)
    # data shape: (3, H, W)
    rgb = np.transpose(data, (1,2,0))  # (H,W,3)
    rgb8 = _to_uint8(rgb)

    # Alpha: prefer explicit mask if present
    alpha = None
    try:
        m1 = dataset.read_masks(bands[0])  # (H,W)
        if m1 is not None:
            alpha = m1
    except Exception:
        alpha = None

    if alpha is None:
        nodata = getattr(dataset, 'nodata', None)
        if nodata is not None:
            # Any band equals nodata -> transparent
            mask = np.zeros((h,w), dtype=bool)
            for b in bands:
                band_data = dataset.read(b)
                mask |= (band_data == nodata)
            alpha = np.where(mask, 0, 255).astype(np.uint8)
        else:
            alpha = np.full((h,w), 255, dtype=np.uint8)

    rgba = np.empty((h,w,4), dtype=np.uint8)
    rgba[...,:3] = rgb8
    rgba[...,3] = alpha
    return rgba


def dataarray_to_rgba(da: "xr.DataArray", *, band_dim: Optional[str] = None) -> np.ndarray:
    """Convert an xarray DataArray to RGBA uint8 (H,W,4) with rudimentary nodata->alpha.

    If band_dim is present or inferred, maps first 3 bands to RGB; single-band is gray.
    """
    if not _HAS_XARRAY:
        raise ImportError("xarray is required. Install with: pip install 'xarray[complete]'")

    dims = list(getattr(da, 'dims', ()))
    if not dims:
        raise ValueError("DataArray has no dims")

    # Identify spatial dims
    y_dim = None
    x_dim = None
    for d in dims:
        dl = d.lower()
        if dl in ("y", "lat", "latitude", "north"):
            y_dim = d
        elif dl in ("x", "lon", "longitude", "east"):
            x_dim = d
    if y_dim is None or x_dim is None:
        # fall back to last two dims
        y_dim, x_dim = dims[-2], dims[-1]

    # Identify band dim
    if band_dim is None:
        for d in dims:
            if d not in (y_dim, x_dim):
                band_dim = d
                break

    arr = da
    if band_dim is not None:
        arr = arr.transpose(y_dim, x_dim, band_dim)
        data = arr.values  # (H,W,C)
    else:
        data = arr.transpose(y_dim, x_dim).values[..., None]  # (H,W,1)

    if data.ndim != 3:
        raise ValueError("Expected 2D or 3D DataArray with spatial dims")

    # Build RGB
    c = data.shape[2]
    if c >= 3:
        rgb = data[..., :3]
    else:
        rgb = np.repeat(data[..., :1], 3, axis=2)

    rgb8 = _to_uint8(rgb)

    # Alpha from nodata attribute, if present
    nodata = getattr(da, 'rio', None)
    nd = None
    try:
        if nodata is not None:
            nd = getattr(nodata, 'nodata', None)
    except Exception:
        nd = None

    h, w = rgb8.shape[:2]
    if nd is not None and band_dim is not None:
        # any band equals nodata
        mask = np.zeros((h,w), dtype=bool)
        for i in range(min(c, 3)):
            mask |= (data[..., i] == nd)
        alpha = np.where(mask, 0, 255).astype(np.uint8)
    else:
        alpha = np.full((h,w), 255, dtype=np.uint8)

    rgba = np.empty((h,w,4), dtype=np.uint8)
    rgba[...,:3] = rgb8
    rgba[...,3] = alpha
    return rgba
