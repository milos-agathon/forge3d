# python/forge3d/provenance.py
# VERITAS: pure-Python per-pixel provenance — canonical leaf encodings, the
# SHA256 Merkle commitment, and Ed25519 seal/verify on top of the offline
# fallback in _ed25519.py. Byte-for-byte mirror of src/core/provenance.rs;
# changing any encoding rule here is a schema bump on both sides.
# RELEVANT FILES: src/core/provenance.rs, src/py_functions/provenance.rs,
# tools/verify_provenance.py, python/forge3d/_ed25519.py

"""Offline verification of forge3d per-pixel provenance.

This module intentionally depends only on :mod:`io`, :mod:`json`,
:mod:`struct`, :mod:`numpy`, the shared SHA256 helper in
``forge3d.bundle``, and the pure-Python Ed25519 fallback in
``forge3d._ed25519`` so third parties can re-verify a rendered
``(image, source_map, provenance.json)`` triple WITHOUT the compiled
``_forge3d`` extension.

Encoding contract (schema_version 2):

- Tile leaf (56 bytes): ``b"VTLF" || family_slot:u32le || source_id:u32le ||
  tile_x:u32le || tile_y:u32le || mip_level:u32le || content_hash[32]``.
- Source-map leaf (44 bytes): ``b"VTSM" || width:u32le || height:u32le ||
  sha256(row-major little-endian u32 raster)[32]``.
- Image leaf (36 bytes): ``b"VTIM" || sha256(rendered image bytes)[32]``.
- Leaf hash = SHA256(encoding); leaves sorted ascending by raw encoding.
- Interior node = SHA256(left || right); an odd trailing node is promoted
  unchanged.
- Empty leaf set: root = SHA256(b"forge3d.provenance.v2.empty").
- Signature message = ``b"forge3d.provenance.v2" || root`` (Ed25519).

Calls that omit ``image_bytes`` retain deprecated schema v1 compatibility:
the image leaf and ``image_sha256`` field are absent, and the signature uses
the ``forge3d.provenance.v1`` context.

All SHA256 hashing routes through :func:`forge3d.bundle._compute_sha256`
(the bundle-integrity helper), which also hashes *files* directly.
"""

from __future__ import annotations

import io
import json
import struct
import warnings
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .bundle import _compute_sha256

SCHEMA_VERSION = 2
SOURCE_ID_NONE = 0
FAMILY_NAMES = ("albedo", "normal", "mask")
SIGN_CONTEXT = b"forge3d.provenance.v2"

_EMPTY_ROOT_PREIMAGE = b"forge3d.provenance.v2.empty"
_LEGACY_SCHEMA_VERSION = 1
_LEGACY_SIGN_CONTEXT = b"forge3d.provenance.v1"
_TILE_LEAF_TAG = b"VTLF"
_SOURCE_MAP_LEAF_TAG = b"VTSM"
_IMAGE_LEAF_TAG = b"VTIM"
_TILE_LEAF_STRUCT = struct.Struct("<5I")
_SOURCE_MAP_LEAF_STRUCT = struct.Struct("<2I")

__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_ID_NONE",
    "FAMILY_NAMES",
    "build_merkle_root",
    "encode_tile_leaf",
    "decode_tile_leaf",
    "encode_source_map_leaf",
    "encode_image_leaf",
    "source_map_digest",
    "manifest_leaf_encodings",
    "seal_provenance_offline",
    "verify_provenance_offline",
]


def _sha256(data: bytes) -> bytes:
    return bytes.fromhex(_compute_sha256(io.BytesIO(data)))


def encode_tile_leaf(
    family_slot: int,
    source_id: int,
    tile_x: int,
    tile_y: int,
    mip_level: int,
    content_hash: bytes,
) -> bytes:
    """Canonical 56-byte tile-leaf preimage."""
    if len(content_hash) != 32:
        raise ValueError("content_hash must be 32 bytes")
    return (
        _TILE_LEAF_TAG
        + _TILE_LEAF_STRUCT.pack(family_slot, source_id, tile_x, tile_y, mip_level)
        + content_hash
    )


def decode_tile_leaf(leaf: bytes) -> Dict[str, Any]:
    """Inverse of :func:`encode_tile_leaf` (round-trip helper for tests)."""
    if len(leaf) != 56 or leaf[:4] != _TILE_LEAF_TAG:
        raise ValueError("not a canonical tile leaf")
    family_slot, source_id, tile_x, tile_y, mip_level = _TILE_LEAF_STRUCT.unpack(
        leaf[4:24]
    )
    return {
        "family_slot": family_slot,
        "source_id": source_id,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "mip_level": mip_level,
        "content_hash": leaf[24:56],
    }


def encode_source_map_leaf(width: int, height: int, digest: bytes) -> bytes:
    """Canonical 44-byte source-map-leaf preimage."""
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    return _SOURCE_MAP_LEAF_TAG + _SOURCE_MAP_LEAF_STRUCT.pack(width, height) + digest


def encode_image_leaf(image_sha256: bytes) -> bytes:
    """Canonical 36-byte image-leaf preimage (``b"VTIM" || digest[32]``)."""
    if len(image_sha256) != 32:
        raise ValueError("image_sha256 must be 32 bytes")
    return _IMAGE_LEAF_TAG + image_sha256


def source_map_digest(source_map: np.ndarray) -> bytes:
    """SHA256 over the row-major little-endian u32 raster of the source map."""
    arr = np.ascontiguousarray(np.asarray(source_map, dtype="<u4"))
    if arr.ndim != 2:
        raise ValueError("source_map must be a 2D (H, W) uint32 array")
    return _sha256(arr.tobytes())


def build_merkle_root(leaf_encodings: Iterable[bytes]) -> bytes:
    """Binary SHA256 Merkle root over leaf preimages (order-independent)."""
    leaves = sorted(bytes(leaf) for leaf in leaf_encodings)
    if not leaves:
        return _sha256(_EMPTY_ROOT_PREIMAGE)
    level = [_sha256(leaf) for leaf in leaves]
    while len(level) > 1:
        next_level = [
            _sha256(level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2 == 1:
            # Odd node count: promote the lone node unchanged.
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _tile_leaf_from_record(index: int, record: Mapping[str, Any]) -> bytes:
    try:
        content_hash = bytes.fromhex(str(record["content_hash"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"tile record {index}: bad content_hash") from exc
    try:
        return encode_tile_leaf(
            int(record["family_slot"]),
            int(record["source_id"]),
            int(record["tile_x"]),
            int(record["tile_y"]),
            int(record["mip_level"]),
            content_hash,
        )
    except KeyError as exc:
        raise ValueError(f"tile record {index}: missing field {exc}") from exc


def manifest_leaf_encodings(
    manifest: Mapping[str, Any],
    source_map: np.ndarray,
    image_bytes: bytes | None = None,
) -> Tuple[bytes, ...]:
    """Rebuild all leaf preimages for a schema v1 or v2 manifest."""
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in (_LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
        raise ValueError(
            f"unsupported provenance schema_version {manifest.get('schema_version')!r}"
        )
    if schema_version == SCHEMA_VERSION and image_bytes is None:
        raise ValueError("image_bytes is required for schema_version 2 provenance")

    leaves = [
        _tile_leaf_from_record(index, record)
        for index, record in enumerate(manifest.get("leaves", ()))
    ]
    height, width = np.asarray(source_map).shape
    leaves.append(encode_source_map_leaf(int(width), int(height), source_map_digest(source_map)))
    if schema_version == SCHEMA_VERSION:
        leaves.append(encode_image_leaf(_sha256(bytes(image_bytes))))
    return tuple(leaves)


def seal_provenance_offline(
    source_map: np.ndarray,
    contributing_tiles: Sequence[Mapping[str, Any]],
    private_key: bytes,
    image_bytes: bytes | None = None,
) -> bytes:
    """Pure-Python twin of the native ``forge3d.seal_provenance``.

    Produces a byte-compatible manifest (same schema, same root and signature
    for the same inputs and key) using the RFC 8032 fallback in
    ``forge3d._ed25519``. Intended for fixtures and extension-absent tests —
    prefer the native function when ``_forge3d`` is available.
    """
    from . import _ed25519

    if image_bytes is None:
        warnings.warn(
            "calling seal_provenance_offline without image_bytes is deprecated; "
            "the legacy schema_version 1 manifest does not bind the rendered image",
            DeprecationWarning,
            stacklevel=2,
        )
        schema_version = _LEGACY_SCHEMA_VERSION
        sign_context = _LEGACY_SIGN_CONTEXT
    else:
        schema_version = SCHEMA_VERSION
        sign_context = SIGN_CONTEXT

    if len(private_key) != 32:
        raise ValueError("private_key must be a 32-byte Ed25519 seed")
    arr = np.asarray(source_map)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("source_map must be a non-empty 2D (H, W) uint32 array")
    height, width = arr.shape

    records = []
    for index, record in enumerate(contributing_tiles):
        leaf = _tile_leaf_from_record(index, record)
        records.append((leaf, decode_tile_leaf(leaf)))
    records.sort(key=lambda item: item[0])
    deduped = []
    for leaf, decoded in records:
        if not deduped or deduped[-1][0] != leaf:
            deduped.append((leaf, decoded))

    digest = source_map_digest(arr)
    leaves = [leaf for leaf, _ in deduped]
    leaves.append(encode_source_map_leaf(int(width), int(height), digest))
    if image_bytes is not None:
        image_digest = _sha256(bytes(image_bytes))
        leaves.append(encode_image_leaf(image_digest))
    root = build_merkle_root(leaves)
    signature = _ed25519.sign(private_key, sign_context + root)
    public_key = _ed25519.public_key_from_private(private_key)

    def _family_name(slot: int) -> str:
        return FAMILY_NAMES[slot] if 0 <= slot < len(FAMILY_NAMES) else "unknown"

    source_table = sorted(
        {
            (
                decoded["source_id"],
                decoded["family_slot"],
                decoded["content_hash"],
            )
            for _, decoded in deduped
        }
    )
    manifest = {
        "schema_version": schema_version,
        "merkle_root": root.hex(),
        "signature": signature.hex(),
        "public_key": public_key.hex(),
        "image_dims": [int(width), int(height)],
    }
    if image_bytes is not None:
        manifest["image_sha256"] = image_digest.hex()
    manifest.update(
        {
            "albedo_family_index": 0,
            "source_map_encoding": "u32le-row-major",
            "source_map_sha256": digest.hex(),
            "leaves": [
                {
                    "family": _family_name(decoded["family_slot"]),
                    "family_slot": decoded["family_slot"],
                    "source_id": decoded["source_id"],
                    "tile_x": decoded["tile_x"],
                    "tile_y": decoded["tile_y"],
                    "mip_level": decoded["mip_level"],
                    "content_hash": decoded["content_hash"].hex(),
                }
                for _, decoded in deduped
            ],
            "source_table": [
                {
                    "source_id": source_id,
                    "family": _family_name(family_slot),
                    "content_hash": content_hash.hex(),
                }
                for source_id, family_slot, content_hash in source_table
            ],
        }
    )
    return json.dumps(manifest, indent=2).encode("utf-8")


def verify_provenance_offline(
    source_map: np.ndarray,
    manifest: Mapping[str, Any] | bytes | str,
    image_bytes: bytes | None = None,
) -> Dict[str, Any]:
    """Re-verify a provenance manifest with only numpy + stdlib + _ed25519.

    Returns a report dict::

        {
            "ok": bool,                # root_match AND signature_valid AND dims
                                       # AND image_sha256_match
            "root_match": bool,
            "signature_valid": bool,
            "dims_match": bool,
            "image_sha256_match": bool,  # schema v2 only
            "computed_root": "<hex>",
            "signed_root": "<hex>",
            "coverage": {source_id: pixel_count},   # SOURCE_ID_NONE excluded
            "unattributed_pixels": int,             # SOURCE_ID_NONE count
            "unknown_source_ids": [int, ...],       # ids absent from source_table
        }
    """
    from . import _ed25519

    if isinstance(manifest, (bytes, bytearray, str)):
        manifest = json.loads(manifest)
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in (_LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
        raise ValueError(
            f"unsupported provenance schema_version {manifest.get('schema_version')!r}"
        )
    if image_bytes is None:
        warnings.warn(
            "calling verify_provenance_offline without image_bytes is deprecated; "
            "only legacy schema_version 1 manifests can be verified without it",
            DeprecationWarning,
            stacklevel=2,
        )
        if schema_version == SCHEMA_VERSION:
            raise ValueError("image_bytes is required for schema_version 2 provenance")

    arr = np.ascontiguousarray(np.asarray(source_map, dtype="<u4"))
    height, width = arr.shape
    dims = manifest.get("image_dims", [None, None])
    dims_match = [int(dims[0]), int(dims[1])] == [int(width), int(height)]

    if schema_version == SCHEMA_VERSION:
        image_digest = _sha256(bytes(image_bytes))
        try:
            manifest_image = bytes.fromhex(str(manifest.get("image_sha256", "")))
        except ValueError:
            manifest_image = b""
        image_sha256_match = manifest_image == image_digest
    computed_root = build_merkle_root(
        manifest_leaf_encodings(manifest, arr, image_bytes)
    )
    signed_root = bytes.fromhex(str(manifest["merkle_root"]))
    root_match = computed_root == signed_root

    signature = bytes.fromhex(str(manifest["signature"]))
    public_key = bytes.fromhex(str(manifest["public_key"]))
    # The signature covers the *signed* root; root_match ties it to this data.
    sign_context = (
        SIGN_CONTEXT if schema_version == SCHEMA_VERSION else _LEGACY_SIGN_CONTEXT
    )
    signature_valid = _ed25519.verify(
        public_key, sign_context + signed_root, signature
    )

    ids, counts = np.unique(arr, return_counts=True)
    coverage = {
        int(i): int(c) for i, c in zip(ids, counts) if int(i) != SOURCE_ID_NONE
    }
    unattributed = int(counts[ids == SOURCE_ID_NONE].sum()) if SOURCE_ID_NONE in ids else 0
    known_ids = {int(entry["source_id"]) for entry in manifest.get("source_table", ())}
    unknown_source_ids = sorted(set(coverage) - known_ids)

    ok = bool(root_match and signature_valid and dims_match)
    if schema_version == SCHEMA_VERSION:
        ok = bool(ok and image_sha256_match)

    report: Dict[str, Any] = {
        "ok": ok,
        "root_match": bool(root_match),
        "signature_valid": bool(signature_valid),
        "dims_match": bool(dims_match),
    }
    if schema_version == SCHEMA_VERSION:
        report["image_sha256_match"] = bool(image_sha256_match)
    report.update(
        {
            "computed_root": computed_root.hex(),
            "signed_root": signed_root.hex(),
            "coverage": coverage,
            "unattributed_pixels": unattributed,
            "unknown_source_ids": unknown_source_ids,
        }
    )
    return report
