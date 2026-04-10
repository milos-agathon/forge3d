# python/dask/base.py
# Minimal dask.base stub for optional xarray compatibility.
# RELEVANT FILES:python/dask/__init__.py,examples/belgium_bivariate_climate_map.py,tests/test_dask_stub.py

"""Small subset of ``dask.base`` used by xarray in non-dask workflows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from hashlib import sha1
from typing import Any


def is_dask_collection(x: object) -> bool:
    """Return True for objects exposing basic dask collection hooks."""

    return any(
        hasattr(x, attr)
        for attr in ("__dask_graph__", "__dask_keys__", "__dask_layers__", "__dask_tokenize__")
    )


def normalize_token(value: Any) -> Any:
    """Normalize nested values into a deterministic token-friendly form."""

    if hasattr(value, "__dask_tokenize__"):
        return normalize_token(value.__dask_tokenize__())
    if isinstance(value, Mapping):
        items = sorted(
            ((normalize_token(k), normalize_token(v)) for k, v in value.items()),
            key=repr,
        )
        return ("dict", tuple(items))
    if isinstance(value, tuple):
        return ("tuple", tuple(normalize_token(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(normalize_token(v) for v in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((normalize_token(v) for v in value), key=repr)))
    return value


def tokenize(*args: Any, **kwargs: Any) -> str:
    """Generate a stable hash for cache keys in lightweight workflows."""

    payload = normalize_token((args, kwargs))
    return sha1(repr(payload).encode("utf-8")).hexdigest()


def flatten(seq: Iterable[Any]) -> Iterator[Any]:
    """Flatten nested key lists while preserving tuple-style dask keys."""

    for item in seq:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def replace_name_in_key(key: Any, rename: Mapping[str, str]) -> Any:
    """Replace the leading name in a dask-style key tuple."""

    if isinstance(key, tuple) and key:
        head = rename.get(key[0], key[0])
        return (head, *key[1:])
    return rename.get(key, key)


def get_scheduler(get: Any = None, collection: Any = None) -> Any:
    """Mirror dask.base.get_scheduler enough for xarray lock helpers."""

    _ = collection  # Kept for dask-compatible call signatures.
    return get
