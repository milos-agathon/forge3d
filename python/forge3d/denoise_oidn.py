from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _load_backend() -> Any | None:
    for module_name in ("oidn", "pyoidn"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _first_attr(obj: Any, *names: str) -> Any | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _require_image(name: str, array: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.ascontiguousarray(array, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def oidn_available() -> bool:
    """Return True when an OIDN Python binding is importable."""

    return _load_backend() is not None


def _maybe_set_image(filter_obj: Any, backend: Any, name: str, array: np.ndarray) -> None:
    set_image = _first_attr(filter_obj, "set_image", "SetImage")
    if set_image is None:
        raise RuntimeError("OIDN filter object does not expose set_image/SetImage")

    height, width, channels = array.shape
    fmt_enum = _first_attr(backend, "Format", "FORMAT")
    fmt_float3 = None
    if fmt_enum is not None:
        fmt_float3 = _first_attr(fmt_enum, "Float3", "FLOAT3")
    if fmt_float3 is None:
        fmt_float3 = _first_attr(backend, "OIDN_FORMAT_FLOAT3", "FORMAT_FLOAT3")

    attempts = [
        (name, array, fmt_float3, width, height),
        (name, array, width, height),
        (name, array),
    ]
    for args in attempts:
        filtered_args = tuple(arg for arg in args if arg is not None)
        try:
            set_image(*filtered_args)
            return
        except TypeError:
            continue

    raise RuntimeError(f"Unable to bind OIDN image '{name}' using the detected Python API")


def _maybe_set_scalar(filter_obj: Any, backend: Any, name: str, value: Any) -> None:
    if isinstance(value, bool):
        setter = _first_attr(filter_obj, "set_bool", "SetBool")
        if callable(setter):
            try:
                setter(name, value)
                return
            except TypeError:
                pass

    if name == "quality":
        setter = _first_attr(filter_obj, "set_quality", "SetQuality")
        if callable(setter):
            quality_map = {
                "default": _first_attr(backend, "OIDN_QUALITY_DEFAULT"),
                "high": _first_attr(backend, "OIDN_QUALITY_HIGH"),
                "balanced": _first_attr(backend, "OIDN_QUALITY_BALANCED"),
                "fast": _first_attr(backend, "OIDN_QUALITY_FAST"),
            }
            mapped = quality_map.get(str(value).lower())
            if mapped is not None:
                try:
                    setter(mapped)
                    return
                except TypeError:
                    pass

    setter = _first_attr(filter_obj, "set", "Set")
    if setter is None:
        return
    try:
        setter(name, value)
    except TypeError:
        pass


def _run_device_filter(
    backend: Any,
    beauty: np.ndarray,
    albedo: np.ndarray | None,
    normal: np.ndarray | None,
    hdr: bool,
    quality: str,
) -> np.ndarray:
    device_ctor = _first_attr(backend, "NewDevice", "Device")
    if device_ctor is None:
        raise RuntimeError("OIDN backend does not expose a device constructor")

    device = device_ctor() if callable(device_ctor) else device_ctor
    commit_device = _first_attr(device, "commit", "Commit")
    if callable(commit_device):
        commit_device()

    filter_ctor = _first_attr(device, "new_filter", "NewFilter")
    if filter_ctor is not None:
        filter_obj = filter_ctor("RT")
    else:
        filter_cls = _first_attr(backend, "Filter")
        if filter_cls is None:
            raise RuntimeError("OIDN backend does not expose a filter constructor")
        try:
            filter_obj = filter_cls(device, "RT")
        except TypeError:
            filter_obj = filter_cls(device)

    _maybe_set_image(filter_obj, backend, "color", beauty)
    if albedo is not None:
        _maybe_set_image(filter_obj, backend, "albedo", albedo)
    if normal is not None:
        _maybe_set_image(filter_obj, backend, "normal", normal)

    output = np.empty_like(beauty, dtype=np.float32)
    _maybe_set_image(filter_obj, backend, "output", output)
    _maybe_set_scalar(filter_obj, backend, "hdr", bool(hdr))
    _maybe_set_scalar(filter_obj, backend, "quality", str(quality))

    commit_filter = _first_attr(filter_obj, "commit", "Commit")
    if callable(commit_filter):
        commit_filter()

    execute = _first_attr(filter_obj, "execute", "Execute")
    if execute is None:
        raise RuntimeError("OIDN filter does not expose execute/Execute")
    execute()
    return output


def oidn_denoise(
    beauty: np.ndarray,
    albedo: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    *,
    hdr: bool = True,
    quality: str = "high",
) -> np.ndarray:
    """Denoise a linear HDR beauty pass with optional albedo/normal guidance."""

    backend = _load_backend()
    if backend is None:
        raise RuntimeError("OIDN runtime is not available")

    beauty_arr = np.ascontiguousarray(beauty, dtype=np.float32)
    if beauty_arr.ndim != 3 or beauty_arr.shape[2] != 3:
        raise ValueError("beauty must be a float32 array of shape (H, W, 3)")

    albedo_arr = _require_image("albedo", albedo, beauty_arr.shape)
    normal_arr = _require_image("normal", normal, beauty_arr.shape)

    for fn_name in ("denoise", "oidn_denoise"):
        fn = getattr(backend, fn_name, None)
        if not callable(fn):
            continue
        try:
            result = fn(
                beauty_arr,
                albedo=albedo_arr,
                normal=normal_arr,
                hdr=bool(hdr),
                quality=str(quality),
            )
            return np.ascontiguousarray(result, dtype=np.float32)
        except TypeError:
            continue

    result = _run_device_filter(
        backend,
        beauty_arr,
        albedo_arr,
        normal_arr,
        hdr=hdr,
        quality=quality,
    )
    return np.ascontiguousarray(result, dtype=np.float32)


__all__ = ["oidn_available", "oidn_denoise"]
