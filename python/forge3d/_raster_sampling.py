"""Neighbour derivatives for grids with missing samples."""

import numpy as np


def _typed_marker(dtype, marker):
    if marker is None or not np.isfinite(marker):
        return marker
    if dtype.kind in "iu":
        integer = int(marker)
        limits = np.iinfo(dtype)
        return integer if integer == marker and limits.min <= integer <= limits.max else None
    with np.errstate(over="ignore"):
        return dtype.type(marker).item()


def source_validity(band, marker):
    """Combine the stored marker with the explicit mask before dtype conversion."""
    values = np.ma.getdata(band)
    valid = np.isfinite(values) & ~np.ma.getmaskarray(band)
    marker = _typed_marker(values.dtype, marker)
    if marker is not None and np.isfinite(marker):
        valid &= values != marker
    return valid


def read_band_validity(dataset, index, values=None, *, window=None):
    """Read stored samples and intersect only independent GDAL masks with the exact marker."""
    from rasterio.enums import MaskFlags

    if values is None:
        values = dataset.read(index, window=window)
    valid = source_validity(values, dataset.nodatavals[index - 1])
    flags = dataset.mask_flag_enums[index - 1]
    # GDAL's synthesized nodata mask can compare neighbouring Float64 values as equal.
    # Empty flags describe an explicit per-band mask.
    if MaskFlags.nodata not in flags and MaskFlags.all_valid not in flags:
        valid &= dataset.read_masks(index, window=window) != 0
    return values, valid


def output_nodata(values, valid, marker):
    """Choose a marker distinct from every valid sample in the destination dtype."""
    marker = _typed_marker(values.dtype, marker)
    if marker is None:
        return None
    if not np.isfinite(marker) or np.any((values == marker) & valid):
        # Integer GeoTIFFs can encode missing samples with their explicit mask alone.
        return np.nan if values.dtype.kind == "f" else None
    return marker


def height_array(values):
    """Represent missing CPU samples uniformly before native float32 upload."""
    if np.ma.isMaskedArray(values):
        return np.ascontiguousarray(values.astype(np.float32).filled(np.nan))
    return np.ascontiguousarray(values, dtype=np.float32)


def gradient(values):
    """Return row and column derivatives. Each axis needs one valid neighbour."""
    if np.isfinite(values).all():
        return np.gradient(values)
    result = []
    for axis in (0, 1):
        rows = np.moveaxis(values, axis, 0)
        padded = np.pad(rows, ((1, 1), (0, 0)), constant_values=np.nan)
        before, after = padded[:-2], padded[2:]
        left, right = np.isfinite(before), np.isfinite(after)
        derivative = np.full_like(rows, np.nan)
        both = left & right & np.isfinite(rows)
        derivative[both] = (after[both] - before[both]) * 0.5
        forward = ~left & right & np.isfinite(rows)
        derivative[forward] = after[forward] - rows[forward]
        backward = left & ~right & np.isfinite(rows)
        derivative[backward] = rows[backward] - before[backward]
        result.append(np.moveaxis(derivative, 0, axis))
    return tuple(result)
