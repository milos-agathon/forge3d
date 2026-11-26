"""Scoped reuse of CPU material preparation, never of a rendered frame."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from functools import wraps
import hashlib
import json

import numpy as np

_ACTIVE = ContextVar("forge3d_material_preparation", default=None)


@contextmanager
def material_preparation_context():
    if _ACTIVE.get() is not None:
        yield
        return
    entries = {}
    token = _ACTIVE.set(entries)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        for _, _, _, reservation in entries.values():
            reservation.close()


def scoped_material_preparation(function):
    @wraps(function)
    def call(*args, **kwargs):
        with material_preparation_context():
            return function(*args, **kwargs)
    return call


def _clone(result):
    cloned = copy.copy(result)
    for name, value in vars(result).items():
        if isinstance(value, np.ndarray):
            setattr(cloned, name, value.copy())
        elif name in ("masks", "legend"):
            setattr(cloned, name, copy.deepcopy(value))
    return cloned


def prepared_material(style, source, valid_mask, evaluate):
    entries = _ACTIVE.get()
    if entries is None:
        return evaluate()
    from ._native import get_native_module
    native = get_native_module()
    reserve = getattr(native, "_reserve_label_depth_host_allocation", None)
    if reserve is None:
        return evaluate()
    digest = hashlib.sha256()
    digest.update(json.dumps([style.recipe.to_dict(), style.grid.to_dict()],
                             sort_keys=True, allow_nan=False).encode())
    heights = np.asarray(source, dtype=np.float64)
    validity = np.isfinite(heights)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != heights.shape or mask.dtype != np.bool_:
            return evaluate()  # Preserve the recipe boundary's validation.
        validity &= mask
    for array in (source, validity, style.classes, style.derivative_halo,
                  *(values for values, _ in style.fields.values())):
        if array is None:
            digest.update(b"None")
        else:
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    digest.update(repr([(name, units) for name, (_, units) in style.fields.items()]).encode())
    for layer in style.recipe.layers:
        for texture in (layer.texture, layer.normal, layer.roughness_texture):
            if texture is not None:
                digest.update(texture.resolved_path.read_bytes())
    key = digest.digest()
    previous = entries.get(id(style))
    if previous is not None:
        if previous[0] is style and previous[1] == key:
            return _clone(previous[2])
        # One current source per style; changed inputs release their predecessor.
        previous[3].close()
        del entries[id(style)]
    result = evaluate()
    arrays = [v for v in vars(result).values() if isinstance(v, np.ndarray)]
    arrays.extend(result.masks.values())
    try:
        reservation = reserve(sum(a.nbytes for a in arrays), "mapscene.material_preparation")
    except native.MemoryBudgetExceeded:
        # Reuse is optional. A denied retention budget must not reject a scene
        # whose uncached preparation/render remains valid.
        return result
    try:
        entries[id(style)] = (style, key, _clone(result), reservation)
    except BaseException:
        reservation.close()
        raise
    return result
