"""Content-addressed execution for deterministic offline render sequences.

ANAMNESIS performs no cache I/O unless a cache path is supplied. Recognized
MapScene recipes execute native terrain, frozen labels and image encoding;
unsupported scene combinations execute freshly. Explicit external executors
can provide their own declared passes. Arbitrary reference mappings use a
GPU-independent DAG for key propagation and store tests. The interactive
viewer does not participate.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import threading
from types import BuiltinFunctionType, FunctionType
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

from ._canonical_json import canonical_json_bytes, canonical_json_value
from ._native import get_native_module

__all__ = [
    "CacheReport",
    "SequenceResult",
    "render_sequence",
    "explain",
    "gc",
    "verify",
    "leaf_key",
    "pass_key",
    "capability_fingerprint",
    "engine_fingerprint",
]

_DOMAIN = b"forge3d.anamnesis/1"
_LEAF_DOMAIN = b"forge3d.anamnesis/1/leaf"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024
_META_NAME = "meta.json"
_BLOB_NAME = "blob"
_LAST_MANIFEST = "last_manifest.json"
_ACCESS_LOG = "access.log"
_STORE_POOL_LIMIT = 4


def _segment(tag: bytes, value: bytes) -> bytes:
    return (
        len(tag).to_bytes(8, "little")
        + tag
        + len(value).to_bytes(8, "little")
        + value
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def leaf_key(content: bytes | bytearray | memoryview) -> str:
    """Return the domain-separated content key of a leaf resource."""
    data = bytes(content)
    native = get_native_module()
    if native is not None and hasattr(native, "anamnesis_leaf_key"):
        return str(native.anamnesis_leaf_key(data))
    return _sha256(_segment(b"domain", _LEAF_DOMAIN) + _segment(b"content", data))


def pass_key(
    label: str,
    pipeline_descriptor: bytes,
    uniform_bytes: bytes,
    input_keys: Sequence[str | tuple[str, str]],
    capability_fingerprint_bytes: bytes,
    engine_fingerprint_bytes: bytes,
) -> str:
    """Compute the hermetic key of one pass.

    ``pipeline_descriptor`` must include exact WGSL hashes plus sampler,
    blend/depth/primitive/target state, viewport, scissor, clear values, RNG
    seed, accumulation frame, backend and DX12 compiler. The function refuses
    malformed input keys and hashes exact uniform bytes including padding.
    """
    normalized_inputs: list[tuple[str, str]] = []
    for index, item in enumerate(input_keys):
        if isinstance(item, tuple) and len(item) == 2:
            binding, key = str(item[0]), str(item[1]).lower()
        else:
            binding, key = f"input@{index}", str(item).lower()
        if not binding:
            raise ValueError("ANAMNESIS input binding identities must be non-empty")
        if len(key) != 64:
            raise ValueError(f"invalid ANAMNESIS input key: {key!r}")
        try:
            bytes.fromhex(key)
        except ValueError as exc:
            raise ValueError(f"invalid ANAMNESIS input key: {key!r}") from exc
        normalized_inputs.append((binding, key))
    normalized_inputs.sort()
    native = get_native_module()
    if native is not None and hasattr(native, "anamnesis_pass_key"):
        return str(
            native.anamnesis_pass_key(
                label,
                bytes(pipeline_descriptor),
                bytes(uniform_bytes),
                normalized_inputs,
                bytes(capability_fingerprint_bytes),
                bytes(engine_fingerprint_bytes),
            )
        )
    payload = bytearray()
    payload += _segment(b"domain", _DOMAIN)
    payload += _segment(b"label", label.encode("utf-8"))
    payload += _segment(
        b"pipeline_descriptor_hash",
        hashlib.sha256(pipeline_descriptor).digest(),
    )
    payload += _segment(b"uniform_bytes", uniform_bytes)
    for binding, key in normalized_inputs:
        payload += _segment(b"input_binding", binding.encode("utf-8"))
        payload += _segment(b"input_key", bytes.fromhex(key))
    payload += _segment(b"capability_fingerprint", capability_fingerprint_bytes)
    payload += _segment(b"engine_fingerprint", engine_fingerprint_bytes)
    return _sha256(bytes(payload))


def engine_fingerprint() -> bytes:
    """Return the canonical engine fingerprint used in every pass key."""
    native = get_native_module()
    if native is not None and hasattr(native, "anamnesis_engine_fingerprint"):
        return canonical_json_bytes(
            json.loads(native.anamnesis_engine_fingerprint()),
            error_context="ANAMNESIS engine fingerprint",
        )
    from . import __version__

    return canonical_json_bytes(
        {
            "crate_version": __version__,
            "git_sha": os.environ.get(
                "FORGE3D_GIT_SHA_FULL",
                os.environ.get("FORGE3D_GIT_SHA", "unknown"),
            ),
            "naga_version": "0.19.2",
            "wgsl_tree_sha256": "unknown",
        },
        error_context="ANAMNESIS engine fingerprint",
    )


def capability_fingerprint(
    value: Mapping[str, Any] | None = None,
    *,
    backend: str | None = None,
    dx12_compiler: str | None = None,
    naga_capabilities: Sequence[str] = (),
) -> bytes:
    """Canonicalize CENSOR's granted capabilities and codegen limits."""
    if value is None:
        value = {}
        native = get_native_module()
        if native is not None and hasattr(native, "capabilities"):
            try:
                value = dict(native.capabilities())
            except Exception:
                # Sound fallback: an incomplete fingerprint gets a distinct
                # explicit value, never one that aliases a known GPU.
                value = {"requested": [], "granted": [], "limits": {}, "status": "unavailable"}
    payload = {
        "granted_features": sorted({str(item) for item in value.get("granted", ())}),
        "limits": {
            str(key): int(item)
            for key, item in sorted(dict(value.get("limits", {})).items())
        },
        "backend": str(
            backend
            if backend is not None
            else value.get("backend", os.environ.get("WGPU_BACKENDS", "unknown"))
        ).lower(),
        "dx12_compiler": str(
            dx12_compiler
            if dx12_compiler is not None
            else value.get("dx12_compiler", "fxc" if os.environ.get("FORGE3D_DETERMINISTIC") else "default")
        ).lower(),
        "naga_capabilities": sorted({
            str(item)
            for item in (
                naga_capabilities
                if naga_capabilities
                else value.get("naga_capabilities", ())
            )
        }),
    }
    return canonical_json_bytes(payload, error_context="ANAMNESIS capability fingerprint")


@dataclass
class CacheReport:
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    bytes_read: int = 0
    bytes_written: int = 0
    wall_ms_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = len(self.hits) + len(self.misses)
        return len(self.hits) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hit_rate": self.hit_rate}


@dataclass
class SequenceResult:
    frame_hashes: list[str]
    frame_blobs: list[bytes]
    cache_report: CacheReport
    predicted_recompute: list[tuple[int, str]]
    observed_recompute: list[tuple[int, str]]
    elapsed_seconds: float
    pass_keys: dict[str, str] = field(default_factory=dict)
    reference_only: bool = False
    bypass_reason: str | None = None

    @property
    def prediction_matches(self) -> bool:
        return set(self.predicted_recompute) == set(self.observed_recompute)


class _PlannedPass(NamedTuple):
    instance: str
    label: str
    frame: int
    pipeline: bytes
    uniforms: bytes
    inputs: tuple[tuple[str, str], ...]
    key: str
    state_value: Any


_SIGNATURE_EXECUTOR = None
_SIGNATURE_EXECUTOR_PID = None


def _signature_executor():
    # Reuse the process's metadata workers instead of joining/recreating them
    # on every frame sequence. A forked child must create its own executor.
    global _SIGNATURE_EXECUTOR, _SIGNATURE_EXECUTOR_PID
    pid = os.getpid()
    if _SIGNATURE_EXECUTOR is None or _SIGNATURE_EXECUTOR_PID != pid:
        _SIGNATURE_EXECUTOR = ThreadPoolExecutor()
        _SIGNATURE_EXECUTOR_PID = pid
    return _SIGNATURE_EXECUTOR


_PACKED_PRELOADS = threading.local()

def _packed_signature(root):
    try:
        data, index = (root / _Store.DATA).stat(), (root / _Store.INDEX).stat()
        return data.st_size, data.st_mtime_ns, index.st_size, index.st_mtime_ns
    except OSError:
        return None

def _native_packed_snapshot(index, data):
    native = get_native_module()
    factory = getattr(native, "_anamnesis_packed_snapshot", None)
    if factory is not None:
        try:
            return factory(index, data)
        except (ValueError, TypeError):
            # A malformed entry still reaches the ordinary per-entry
            # quarantine path; a failed optional parse never serves bytes.
            return None
    return None

def _verify_packed_reads(snapshot, keys):
    return snapshot.verify(keys)

def _load_packed_snapshot(root):
    # Read only: neither admission nor entry verification may run ahead of
    # scene validation and the independently frozen recompute prediction.
    signature = _packed_signature(root)
    if signature is None:
        return None
    try:
        raw = (root / _Store.INDEX).read_bytes()
        index = json.loads(raw)
        data = (root / _Store.DATA).read_bytes()
        snapshot = _native_packed_snapshot(raw, data)
        return (signature, raw, index, data, snapshot) if signature == _packed_signature(root) else None
    except (OSError, ValueError, TypeError):
        return None


def _manifest_signature(path):
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return None


def _load_manifest_snapshot(root):
    path = root / _LAST_MANIFEST
    signature = _manifest_signature(path)
    if signature is None:
        return None
    try:
        manifest = _read_manifest(root)
        return (signature, manifest) if signature == _manifest_signature(path) else None
    except (OSError, ValueError, TypeError, AttributeError):
        # Speculative reads must not move validation or errors before admission.
        return None


@contextmanager
def _preload_packed_store(root):
    previous = getattr(_PACKED_PRELOADS, "pending", {})
    previous_manifests = getattr(_PACKED_PRELOADS, "manifests", {})
    pending = dict(previous)
    manifests = dict(previous_manifests)
    started = manifest_started = None
    if root is not None:
        key = os.path.abspath(root)
        # An open pooled store already owns its immutable snapshot. Reuse
        # checks its signatures and refreshes synchronously if maintenance ran.
        pooled = any(os.path.abspath(store.root) == key for store in _STORE_POOL.values())
        if key not in pending and not pooled:
            started = _signature_executor().submit(_load_packed_snapshot, Path(root))
            pending[key] = started
            manifest_started = _signature_executor().submit(_load_manifest_snapshot, Path(root))
            manifests[key] = manifest_started
    _PACKED_PRELOADS.pending = pending
    _PACKED_PRELOADS.manifests = manifests
    try:
        yield
    finally:
        try:
            try:
                if started is not None:
                    started.result()  # All work stays inside the caller's timer.
            finally:
                if manifest_started is not None:
                    manifest_started.result()
        finally:
            _PACKED_PRELOADS.pending = previous
            _PACKED_PRELOADS.manifests = previous_manifests


class _PackedBytes:
    """Own immutable payload segments; appending never copies resident bytes."""
    __slots__ = ("chunks", "ends", "length")

    def __init__(self, initial=b""):
        self.chunks, self.ends, self.length = [], [], 0
        self.extend(initial)

    def __len__(self):
        return self.length

    def extend(self, value):
        value = bytes(value)
        if value:
            self.chunks.append(value)
            self.length += len(value)
            self.ends.append(self.length)

    def __getitem__(self, span):
        start, stop, step = span.indices(self.length)
        if step != 1:
            return b"".join(self.chunks)[span]
        if start >= stop:
            return b""
        index = bisect_right(self.ends, start)
        base = self.ends[index - 1] if index else 0
        if stop <= self.ends[index]:
            return self.chunks[index][start - base:stop - base]
        pieces = []
        while start < stop:
            end = min(stop, self.ends[index])
            pieces.append(self.chunks[index][start - base:end - base])
            start, base, index = end, self.ends[index], index + 1
        return b"".join(pieces)


class _CompletedRead:
    """A synchronous packed read needs no Future lock or condition variable."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Store:
    """Version 2: one append-only payload file and one self-describing index."""

    DATA = "data.v2.bin"
    INDEX = "index.v2.json"
    SCHEMA = "forge3d.anamnesis.packed-index/2"

    def __init__(self, root, max_bytes, verify_reads):
        if int(max_bytes) <= 0:
            raise ValueError("ANAMNESIS max_bytes must be positive")
        self.root, self.max_bytes, self.verify_reads = Path(root), int(max_bytes), bool(verify_reads)
        self._memory_entries, self._protected_reads, self._pending_reads = {}, {}, {}
        self._touched_keys, self._active_plan_keys, self._resident_keys = set(), set(), set()
        self._key_material = {}
        self._records, self._data, self._signature = {}, _PackedBytes(), None
        self._appendable_index = False
        self._native_snapshot = None
        self._invalid_index = False
        self._bad_index_bytes = b""
        self._manifest_may_be_incomplete = False
        self._batch_depth, self._admitted_limit = 0, None
        self._admitting, self._keep_packed_files = False, False
        self.admission_cleanup_bytes = 0
        self._current_bytes = self.disk_bytes()
        if self.root.exists():
            self._refresh_residency()
        self._admit()
        if not self.root.exists():
            directory_bytes = 0 if os.name == "nt" else self._index_bytes()
            self._reserve_growth(2 * directory_bytes)
            self.root.mkdir(parents=True)
        quarantine = self.root / "quarantine"
        if not quarantine.exists():
            growth = 0 if os.name == "nt" and not any(self.root.iterdir()) else 2 * self._index_bytes()
            self._reserve_growth(growth)
            quarantine.mkdir()
        self._current_bytes = self.disk_bytes()
        self._root_identity = self._identity()

    def _packed_signature(self):
        return _packed_signature(self.root)

    def _resident_inventory(self):
        return {key for key, record in self._records.items()
                if isinstance(record, dict)
                and type(record.get("offset")) is int and type(record.get("length")) is int
                and 0 <= record["offset"] <= record["offset"] + record["length"] <= len(self._data)
                and isinstance(record.get("metadata"), dict)
                and record["metadata"].get("schema") == "forge3d.anamnesis.store/2"}

    def _refresh_residency(self):
        signature = self._packed_signature()
        if signature is not None and signature == self._signature:
            return
        self._records, self._data, self._resident_keys = {}, _PackedBytes(), set()
        self._appendable_index = False
        self._native_snapshot = None
        self._invalid_index = False
        self._bad_index_bytes = b""
        if signature is not None:
            try:
                pending = getattr(_PACKED_PRELOADS, "pending", {}).get(os.path.abspath(self.root))
                snapshot = pending.result() if pending is not None else None
                if snapshot is not None and snapshot[0] == signature:
                    _, self._bad_index_bytes, index, data, self._native_snapshot = snapshot
                else:
                    self._bad_index_bytes = (self.root / self.INDEX).read_bytes()
                    index = json.loads(self._bad_index_bytes)
                    data = (self.root / self.DATA).read_bytes()
                    self._native_snapshot = _native_packed_snapshot(self._bad_index_bytes, data)
                if index.get("schema") != self.SCHEMA or not isinstance(index.get("entries"), dict):
                    raise ValueError("unsupported packed index")
                if signature != self._packed_signature():
                    raise RuntimeError("ANAMNESIS packed store changed while opening")
                self._records, self._data = index["entries"], _PackedBytes(data)
                self._appendable_index = (next(reversed(index)) == "entries"
                                          and self._bad_index_bytes.endswith(b"}}"))
                self._resident_keys = self._resident_inventory()
            except (OSError, ValueError, TypeError):
                self._invalid_index = True
                (self.root / _LAST_MANIFEST).unlink(missing_ok=True)
        if signature is None:
            self._invalid_index = (self.root / self.DATA).exists() or (self.root / self.INDEX).exists()
            if (self.root / self.INDEX).exists():
                self._bad_index_bytes = (self.root / self.INDEX).read_bytes()
            (self.root / _LAST_MANIFEST).unlink(missing_ok=True)
        self._signature = signature

    def _encoded_index(self, records=None):
        # The entries object stays last so a new verified record can replace
        # its two closing braces without serializing every resident record.
        entries = canonical_json_bytes(self._records if records is None else records,
                                       error_context="ANAMNESIS packed index")
        return b'{"schema":' + json.dumps(self.SCHEMA).encode("ascii") + b',"entries":' + entries + b'}'

    def _ensure_format(self):
        if self._signature is not None and not self._invalid_index:
            return
        if self._invalid_index:
            self._quarantine_index()
        missing = [name for name in (self.DATA, self.INDEX) if not (self.root / name).exists()]
        if not missing:
            return
        payload = self._encoded_index()
        self._reserve_growth(len(payload) + len(missing) * self._index_bytes())
        if self.DATA in missing:
            (self.root / self.DATA).write_bytes(b"")
        if self.INDEX in missing:
            (self.root / self.INDEX).write_bytes(payload)
            self._appendable_index = True
        self._current_bytes = self.disk_bytes()
        self._signature = self._packed_signature()

    def _identity(self):
        value = self.root.stat()
        return value.st_dev, value.st_ino

    def reusable(self):
        try:
            if self._identity() != self._root_identity:
                return False
            self._refresh_residency()
            self._admit()
            return True
        except OSError:
            return False

    def _admit(self):
        if self._batch_depth and self._admitted_limit == self.max_bytes:
            return
        self._current_bytes = self.disk_bytes()
        if self._current_bytes > self.max_bytes:
            excess = self._current_bytes - self.max_bytes
            self.admission_cleanup_bytes += excess
            print(f"ANAMNESIS admission_cleanup starting_excess_bytes={excess}", file=sys.stderr)
            self._admitting = True
            try:
                self.gc(self.max_bytes)
            finally:
                self._admitting = False
            if self.disk_bytes() > self.max_bytes:
                raise ValueError("ANAMNESIS store control/quarantine footprint exceeds max_bytes")
        self._admitted_limit = self.max_bytes

    def _reserve_growth(self, additional):
        if self._current_bytes + additional > self.max_bytes:
            self.gc(max(0, self.max_bytes - additional))
        if self._current_bytes + additional > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")

    def disk_bytes(self):
        return self._tree_bytes(self.root)

    def entries(self):
        return sorted(self._records)

    def _raw_entry(self, key):
        try:
            record = self._records[key]
            offset, length = record["offset"], record["length"]
            if type(offset) is not int or type(length) is not int or offset < 0 or length < 0 or offset + length > len(self._data):
                return None
            return self._data[offset:offset + length], dict(record["metadata"])
        except (KeyError, TypeError, ValueError):
            return None

    def _read_entry(self, key):
        restored = self._raw_entry(key)
        try:
            if restored is not None and self._metadata_valid(key, *restored):
                return *restored, self._signature
        except (ValueError, TypeError, KeyError):
            pass
        return None

    def memory_signatures(self, keys):
        signature = self._packed_signature()
        task = Future()
        names = set(keys) & self._memory_entries.keys()
        task.set_result(dict.fromkeys(names, signature))
        return {key: (task, key) for key in names}

    @contextmanager
    def protect_reads(self, keys):
        self._admit()
        self._batch_depth += 1
        previous = self._protected_reads
        active = dict(previous)
        active.update((key, active.get(key)) for key in keys)
        self._protected_reads = active
        try:
            yield
        finally:
            for key in previous:
                if active.get(key) is not None:
                    previous[key] = active[key]
            self._protected_reads = previous
            self._batch_depth -= 1

    @contextmanager
    def prefetch_reads(self, keys):
        previous = self._pending_reads
        pending = dict(previous)
        # One packed snapshot removes per-entry opens; full entry verification
        # is performed here, after prediction freezes and inside its timer.
        required = tuple(key for key in dict.fromkeys(keys)
                         if key not in self._memory_entries and key not in pending)
        verified = (_verify_packed_reads(self._native_snapshot, [key for key in required if key in self._records])
                    if self._native_snapshot is not None else {})
        for key in required:
            if key not in self._memory_entries and key not in pending:
                if key in verified:
                    blob = verified[key]
                    read = ((blob, dict(self._records[key]["metadata"]), self._signature)
                            if blob is not None else None)
                else:
                    read = self._read_entry(key)
                pending[key] = _CompletedRead(read)
                if read is not None and key in self._protected_reads:
                    # These independently verified bytes belong to this run's
                    # protected snapshot, just like a prior protected get().
                    self._protected_reads[key] = read
                    self._memory_entries[key] = read
        # Required bytes now have independently verified protected copies.
        # Do not retain a second parsed metadata tree in the pooled store.
        self._native_snapshot = None
        self._pending_reads = pending
        try:
            yield
        finally:
            self._pending_reads = previous

    def get(self, key, *, touch=True, quarantine=True, known_signature=None):
        if not self._admitting:
            self._admit()
        protected = self._protected_reads.get(key)
        if protected is not None:
            if touch:
                self._touched_keys.add(key)
            return protected[0], dict(protected[1])
        prefetched = self._pending_reads.pop(key, None)
        signature = known_signature or self._packed_signature()
        remembered = self._memory_entries.get(key)
        if remembered is not None and signature is not None and remembered[2] == signature:
            restored = remembered[:2]
        else:
            if signature != self._signature:
                self._refresh_residency()
            if self._invalid_index:
                self._quarantine_index()
                return None
            read = prefetched.result() if prefetched is not None else None
            if read is None or read[2] != self._signature:
                read = self._read_entry(key)
            if read is None:
                self._memory_entries.pop(key, None)
                if quarantine and key in self._records:
                    self._quarantine(key)
                return None
            restored = read[:2]
            self._memory_entries[key] = (*restored, self._signature)
        if touch:
            self._touched_keys.add(key)
        return restored[0], dict(restored[1])

    def put_pass(self, key, blob, material, *, frame=None, measured_wall_ms=0.0):
        self._put(key, blob, pass_label=str(material["label"]), input_keys=list(material.get("input_keys", ())),
                  derivation={"kind": "pass", "material": dict(material)}, frame=frame, measured_wall_ms=measured_wall_ms)

    def put_leaf(self, key, blob, *, label):
        if leaf_key(blob) != key:
            raise ValueError("ANAMNESIS leaf key does not match leaf content")
        self._put(key, blob, pass_label=label, input_keys=[],
                  derivation={"kind": "leaf", "content_sha256": _sha256(blob)}, frame=None, measured_wall_ms=0.0)

    @staticmethod
    def _entry_metadata(key, blob, *, pass_label, input_keys, derivation, frame, measured_wall_ms):
        try:
            engine = json.loads(bytes.fromhex(derivation.get("material", {}).get("engine_fingerprint_hex", "")).decode())
        except (ValueError, UnicodeError, TypeError):
            engine = dict.fromkeys(("crate_version", "git_sha", "naga_version", "wgsl_tree_sha256"), "unknown")
        now = time.time_ns() // 1_000_000
        meta = {"schema": "forge3d.anamnesis.store/2", "key": key, "pass_label": pass_label,
                "input_keys": input_keys, "byte_length": len(blob), "creation_engine_fingerprint": engine,
                "self_hash": _sha256(blob), "created_unix_ms": now, "last_access_unix_ms": now,
                "derivation": derivation, "frame": frame, "measured_wall_ms": max(0.0, float(measured_wall_ms))}
        return meta

    def put_pass_batch(self, writes):
        """Append a verified native-scene write set with one file pair open.

        Pressure, existing keys and non-appendable indexes retain the ordinary
        per-pass path, including its admission, LRU and protected-read logic.
        """
        writes = [(key, bytes(blob), dict(material), frame, elapsed)
                  for key, blob, material, frame, elapsed in writes]
        if not writes:
            return
        if not self._batch_depth:
            self._refresh_residency()
        self._admit()
        prepared = []
        for key, blob, material, frame, elapsed in writes:
            meta = self._entry_metadata(key, blob, pass_label=str(material["label"]),
                input_keys=list(material.get("input_keys", ())),
                derivation={"kind": "pass", "material": material},
                frame=frame, measured_wall_ms=elapsed)
            try:
                valid = self._metadata_valid(key, blob, meta)
            except (KeyError, ValueError, TypeError):
                valid = False
            if not valid:
                raise ValueError("ANAMNESIS batch pass key does not match its material")
            prepared.append((key, blob, meta))
        keys = [key for key, _, _ in prepared]
        appendable = (self._signature is not None and self._appendable_index
                      and len(set(keys)) == len(keys)
                      and not any(key in self._records for key in keys))
        records, fragments = dict(self._records), []
        offset = len(self._data)
        if appendable:
            for key, blob, meta in prepared:
                record = {"offset": offset, "length": len(blob), "metadata": meta}
                records[key] = record
                fragments.append(canonical_json_bytes({key: record},
                    error_context="ANAMNESIS packed batch record")[1:-1])
                offset += len(blob)
            # Existing files require no directory-entry growth. Reserve all
            # data and index growth before opening either file for writing.
            index_growth = sum(map(len, fragments)) + len(fragments) - 1 + bool(self._records)
            data_growth = offset - len(self._data)
            growth = data_growth + index_growth
            appendable = self._current_bytes + growth <= self.max_bytes
        if not appendable:
            for key, blob, material, frame, elapsed in writes:
                self.put_pass(key, blob, material, frame=frame, measured_wall_ms=elapsed)
            return
        # A concurrent store writer cannot be merged with this frozen append.
        # The next pooled reuse refreshes residency from its new snapshot.
        if self._packed_signature() != self._signature:
            raise RuntimeError("ANAMNESIS packed store changed during batched append")
        before_reservation = self._signature
        self._reserve_growth(growth)
        if self._signature != before_reservation:
            # Admission/GC may relocate offsets. Reuse the established path
            # rather than publishing a batch built against the old snapshot.
            for key, blob, material, frame, elapsed in writes:
                self.put_pass(key, blob, material, frame=frame, measured_wall_ms=elapsed)
            return
        old_index_size = self._signature[2]
        try:
            with (self.root / self.DATA).open("ab") as stream:
                stream.writelines(blob for _, blob, _ in prepared)
            self._current_bytes += data_growth
            with (self.root / self.INDEX).open("r+b") as stream:
                stream.seek(old_index_size - 2)
                for index, fragment in enumerate(fragments):
                    if self._records or index:
                        stream.write(b",")
                    stream.write(fragment)
                stream.write(b"}}")
            signature = self._packed_signature()
            if signature is None or signature[0] != offset or signature[2] != old_index_size + index_growth:
                raise OSError("ANAMNESIS packed batch was not fully published")
        except BaseException:
            self._signature, self._native_snapshot = None, None
            self._memory_entries.clear()
            self._current_bytes = self.disk_bytes()
            self._invalidate_manifest()
            raise
        self._records = records
        for _, blob, _ in prepared:
            self._data.extend(blob)
        self._resident_keys.update(keys)
        self._signature, self._native_snapshot = signature, None
        self._current_bytes += index_growth
        for key, blob, meta in prepared:
            self._memory_entries[key] = (blob, meta, signature)
        if self._current_bytes > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")

    def _put(self, key, blob, *, pass_label, input_keys, derivation, frame, measured_wall_ms):
        if not self._batch_depth:
            self._refresh_residency()
        self._admit()
        if key in self._records and self.get(key) is not None:
            return
        meta = self._entry_metadata(key, blob, pass_label=pass_label,
            input_keys=input_keys, derivation=derivation, frame=frame,
            measured_wall_ms=measured_wall_ms)
        self._ensure_format()
        while True:
            records = dict(self._records)
            records[key] = {"offset": len(self._data), "length": len(blob), "metadata": meta}
            old_size = self._signature[2]
            append = None
            if self._appendable_index and key not in self._records:
                entry = canonical_json_bytes({key: records[key]}, error_context="ANAMNESIS packed record")[1:-1]
                append = (b"," if self._records else b"") + entry + b"}}"
                growth = len(blob) + len(append) - 2
            else:
                encoded = self._encoded_index(records)
                growth = len(blob) + max(0, len(encoded) - old_size)
            if self._current_bytes + growth <= self.max_bytes:
                break
            before = self._current_bytes
            self._keep_packed_files = True
            try:
                self.gc(max(0, self.max_bytes - growth))
            finally:
                self._keep_packed_files = False
            if self._current_bytes >= before:
                raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")
        # Both payload growth and the entire index growth are reserved before
        # append. A crash before index publication leaves an unindexed miss.
        with (self.root / self.DATA).open("ab") as stream:
            stream.write(blob)
        self._current_bytes += len(blob)
        if append is None:
            (self.root / self.INDEX).write_bytes(encoded)
        else:
            with (self.root / self.INDEX).open("r+b") as stream:
                stream.seek(old_size - 2)
                stream.write(append)
        self._appendable_index = True
        self._records = records
        self._data.extend(blob)
        self._resident_keys.add(key)
        self._signature = self._packed_signature()
        self._memory_entries[key] = (bytes(blob), dict(meta), self._signature)
        # Both files already exist: changing their contents cannot change a
        # directory entry's length. Admission/GC/control creation still measure
        # the full tree; this reserved append accounts for its exact growth.
        self._current_bytes += (len(encoded) - old_size if append is None else len(append) - 2)
        if self._current_bytes > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")

    def _invalidate_manifest(self):
        (self.root / _LAST_MANIFEST).unlink(missing_ok=True)
        self._manifest_may_be_incomplete = True

    def _rewrite(self, records):
        self._native_snapshot = None
        chunks, rewritten, offset = [], {}, 0
        for key, record in sorted(records.items(), key=lambda item: item[1]["offset"]):
            blob = self._data[record["offset"]:record["offset"] + record["length"]]
            chunks.append(blob)
            rewritten[key] = {**record, "offset": offset}
            offset += len(blob)
        data, index = b"".join(chunks), self._encoded_index(rewritten)
        self._invalidate_manifest()
        # Invalidate the index before relocating bytes. Truncate first so
        # admission cleanup never writes while the starting excess remains.
        for name in (self.INDEX, self.DATA):
            if (self.root / name).exists():
                os.truncate(self.root / name, 0)
        self._current_bytes = self.disk_bytes()
        if self._current_bytes + len(data) + len(index) > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")
        (self.root / self.DATA).write_bytes(data)
        (self.root / self.INDEX).write_bytes(index)
        self._records, self._data = rewritten, _PackedBytes(data)
        self._appendable_index = True
        self._resident_keys = self._resident_inventory()
        self._signature = self._packed_signature()
        self._current_bytes = self.disk_bytes()

    def gc(self, target_bytes):
        if not self._batch_depth:
            self._refresh_residency()
        target = max(0, int(target_bytes))
        before = self.disk_bytes()
        access = self._read_access_times()
        records = dict(self._records)
        estimated = before
        for key in sorted(records, key=lambda key: (key in self._active_plan_keys,
                           max(int(records[key]["metadata"].get("last_access_unix_ms", 0)), access.get(key, 0)), key)):
            if estimated <= target:
                break
            if key in self._protected_reads:
                preserved = self.get(key, touch=False)
                if preserved is None:
                    raise RuntimeError(f"ANAMNESIS prediction mismatch: unread entry {key!r}")
                self._protected_reads[key] = preserved
            record = records.pop(key)
            estimated -= record["length"] + len(canonical_json_bytes(record, error_context="ANAMNESIS eviction")) + len(key) + 4
            self._memory_entries.pop(key, None)
            self._touched_keys.discard(key)
        # Legacy schema directories are never read or migrated. GC can reclaim
        # them, quarantine and orphan staging bytes under the same full bound.
        if self.root.exists():
            extras = [p for p in self.root.iterdir() if p.is_dir() and len(p.name) == 2]
            quarantine = self.root / "quarantine"
            extras += sorted(quarantine.iterdir(), key=lambda p: p.stat().st_mtime_ns) if quarantine.exists() else []
            for path in extras:
                if estimated <= target:
                    break
                size = self._tree_bytes(path)
                self._invalidate_manifest()
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                estimated -= size
        if records != self._records or (not records and self._data):
            if records or self._keep_packed_files:
                self._rewrite(records)
            else:
                self._invalidate_manifest()
                for name in (self.INDEX, self.DATA):
                    (self.root / name).unlink(missing_ok=True)
                self._records, self._data, self._resident_keys, self._signature = {}, _PackedBytes(), set(), None
        if not records and not self._keep_packed_files and self.disk_bytes() > target:
            for name in (self.INDEX, self.DATA):
                (self.root / name).unlink(missing_ok=True)
            self._signature = None
        if self.root.exists() and self.disk_bytes() <= self.max_bytes:
            self._compact_access_log()
        self._current_bytes = self.disk_bytes()
        return max(0, before - self._current_bytes)

    def _quarantine(self, key):
        raw = self._raw_entry(key)
        packet = canonical_json_bytes({"key": key, "metadata": self._records[key]}, error_context="ANAMNESIS quarantine") + b"\n"
        if raw is not None:
            packet += raw[0]
        else:
            record = self._records[key]
            offset, length = record.get("offset"), record.get("length")
            if type(offset) is int and type(length) is int and 0 <= offset <= offset + length <= len(self._data):
                packet += self._data[offset:offset + length]
        records = dict(self._records)
        records.pop(key, None)
        self._rewrite(records)
        self._memory_entries.pop(key, None)
        self._touched_keys.discard(key)
        self._write_quarantine_packet(f"{key}-{time.time_ns()}", packet)

    def _write_quarantine_packet(self, name, packet):
        quarantine = self.root / "quarantine"
        # Admission already created this directory. Only its new file entry
        # can grow the directory index; reserving a second directory block
        # can reject quarantine even after every live cache entry is evicted.
        growth = 0 if os.name == "nt" and quarantine.stat().st_size == 0 and not any(quarantine.iterdir()) else self._index_bytes()
        self._reserve_growth(len(packet) + growth)
        (quarantine / name).write_bytes(packet)
        self._current_bytes = self.disk_bytes()

    def _quarantine_index(self):
        self._native_snapshot = None
        packet = b"ANAMNESIS invalid packed index\n" + self._bad_index_bytes
        self._invalidate_manifest()
        for name in (self.INDEX, self.DATA):
            (self.root / name).unlink(missing_ok=True)
        self._records, self._data, self._resident_keys, self._signature = {}, _PackedBytes(), set(), None
        self._invalid_index, self._bad_index_bytes = False, b""
        self._memory_entries.clear()
        self._current_bytes = self.disk_bytes()
        self._write_quarantine_packet(f"index-{time.time_ns()}.invalid", packet)

    def verify(self):
        self._refresh_residency()
        self._admit()
        result = {"valid": 0, "quarantined": 0, "bytes_checked": 0}
        if self._invalid_index:
            self._quarantine_index()
            result["quarantined"] = 1
            return result
        for key in list(self._records):
            raw = self._raw_entry(key)
            result["bytes_checked"] += len(raw[0]) if raw is not None else 0
            if self._read_entry(key) is not None:
                result["valid"] += 1
            else:
                self._quarantine(key)
                result["quarantined"] += 1
        return result

    def _resident_manifest(self, payload):
        manifest = json.loads(payload)
        manifest["entries"] = {instance: key for instance, key in manifest["entries"].items() if key in self._resident_keys}
        return canonical_json_bytes(manifest, error_context="ANAMNESIS resident manifest")


    def _index_bytes(self) -> int:
        if os.name == "nt":
            return 4096  # Measured NTFS directory-index growth in this store.
        ancestor = next(path for path in (self.root, *self.root.parents) if path.exists())
        return os.statvfs(ancestor).f_frsize


    @staticmethod
    def _tree_bytes(path: Path) -> int:
        try:
            metadata = path.stat()
        except OSError:
            return 0

        def visit(directory, metadata):
            total = metadata.st_size
            if stat.S_ISDIR(metadata.st_mode):
                # Reuse file attributes from enumeration. Windows directory
                # indexes still need an explicit size query below.
                with os.scandir(directory) as children:
                    for child in children:
                        try:
                            child_metadata = child.stat()
                            if os.name == "nt" and stat.S_ISDIR(child_metadata.st_mode):
                                # Find-data sizes are zero for NTFS directories
                                # even when their index occupies real bytes.
                                child_metadata = os.stat(child.path)
                        except OSError:
                            continue
                        total += visit(child.path, child_metadata)
            return total

        return visit(path, metadata)


    @staticmethod
    def _entry_signature(path: Path) -> tuple[int, int, int, int] | None:
        try:
            if os.name == "nt":
                # Windows directory enumeration carries these same size/mtime
                # attributes, avoiding two separate metadata queries. Reparse
                # points still follow DirEntry.stat's normal link semantics.
                with os.scandir(path) as entries:
                    attributes = {item.name: item.stat() for item in entries
                                  if item.name in {_BLOB_NAME, _META_NAME}}
                blob, meta = attributes[_BLOB_NAME], attributes[_META_NAME]
            else:
                blob = (path / _BLOB_NAME).stat()
                meta = (path / _META_NAME).stat()
        except (OSError, KeyError):
            return None
        return (
            int(blob.st_size),
            int(blob.st_mtime_ns),
            int(meta.st_size),
            int(meta.st_mtime_ns),
        )


    @staticmethod
    def _metadata_valid(key: str, blob: bytes, meta: Mapping[str, Any]) -> bool:
        if not (
            meta.get("schema") == "forge3d.anamnesis.store/2"
            and meta.get("key") == key
            and int(meta.get("byte_length", -1)) == len(blob)
            and meta.get("self_hash") == _sha256(blob)
        ):
            return False
        derivation = dict(meta.get("derivation", {}))
        if derivation.get("kind") == "leaf":
            return (
                derivation.get("content_sha256") == _sha256(blob)
                and leaf_key(blob) == key
            )
        if derivation.get("kind") != "pass":
            return False
        material = dict(derivation.get("material", {}))
        try:
            pipeline = bytes.fromhex(str(material["pipeline_descriptor_hex"]))
            uniforms = bytes.fromhex(str(material["uniform_hex"]))
            capabilities = bytes.fromhex(
                str(material["capability_fingerprint_hex"])
            )
            engine = bytes.fromhex(str(material["engine_fingerprint_hex"]))
            inputs = [
                (str(item["binding"]), str(item["key"]))
                for item in material.get("input_keys", ())
            ]
            reconstructed = pass_key(
                str(material["label"]),
                pipeline,
                uniforms,
                inputs,
                capabilities,
                engine,
            )
        except (KeyError, TypeError, ValueError):
            return False
        return (
            reconstructed == key
            and material.get("pipeline_descriptor_hash") == _sha256(pipeline)
            and material.get("uniform_sha256") == _sha256(uniforms)
            and int(material.get("uniform_byte_length", -1)) == len(uniforms)
            and material.get("capability_fingerprint_sha256")
            == _sha256(capabilities)
            and material.get("engine_fingerprint_sha256") == _sha256(engine)
        )


    def _read_access_times(self) -> dict[str, int]:
        result: dict[str, int] = {}
        try:
            lines = (self.root / _ACCESS_LOG).read_text(encoding="ascii").splitlines()
        except OSError:
            return result
        for line in lines:
            try:
                timestamp, key = line.split("\t", 1)
                if len(key) != 64:
                    continue
                int(key, 16)
                result[key] = max(result.get(key, 0), int(timestamp))
            except (ValueError, TypeError):
                continue
        return result


    def _compact_access_log(self) -> None:
        target = self.root / _ACCESS_LOG
        if not target.exists():
            return
        access_times = self._read_access_times()
        live = set(self.entries())
        payload = "".join(
            f"{access_times[key]}\t{key}\n"
            for key in sorted(access_times.keys() & live)
        ).encode("ascii")
        self._replace_control(_ACCESS_LOG, payload, enforce=False)


    def flush_accesses(self) -> None:
        self._admit()
        if not self._touched_keys:
            return
        now = time.time_ns() // 1_000_000
        def pending_payload():
            return "".join(f"{now}\t{key}\n" for key in sorted(self._touched_keys)).encode("ascii")
        payload = pending_payload()
        target = self.root / _ACCESS_LOG
        before = target.stat().st_size if target.exists() else 0
        directory_growth = self._index_bytes() if not target.exists() else 0
        if self._current_bytes + len(payload) + directory_growth > self.max_bytes:
            self.gc(max(0, self.max_bytes - len(payload) - directory_growth))
            # Eviction discards touches for entries that no longer exist.
            payload = pending_payload()
        if not payload:
            return
        directory_growth = self._index_bytes() if not target.exists() else 0
        self._reserve_growth(len(payload) + directory_growth)
        before_root = self.root.stat().st_size
        with target.open("ab") as stream:
            stream.write(payload)
        self._current_bytes += len(payload) + self.root.stat().st_size - before_root
        self._touched_keys.clear()
        if self._current_bytes > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")
        if target.stat().st_size > max(1024 * 1024, before * 2):
            self._compact_access_log()


    def _replace_control(
        self,
        name: str,
        payload: bytes,
        *,
        enforce: bool,
    ) -> None:
        self._admit()
        target = self.root / name
        old_size = target.stat().st_size if target.exists() else 0
        if not enforce:
            # GC compaction rewrites an existing, shrinking advisory log. It
            # cannot create a directory entry or grow the file during cleanup.
            if not target.exists() or len(payload) > old_size:
                raise ValueError("ANAMNESIS access-log compaction must only shrink")
            before_root = self.root.stat().st_size
            target.write_bytes(payload)
            self._current_bytes += len(payload) - old_size + self.root.stat().st_size - before_root
            return
        if name == _LAST_MANIFEST:
            payload = self._resident_manifest(payload)
            if not target.exists():
                # Reserve the empty control entry's index growth before it
                # exists. A clean NTFS singleton uses a resident root index.
                growth = self._index_bytes()
                if os.name == "nt" and self._current_bytes == 0:
                    growth = 0
                self._reserve_growth(growth)
            before_root = self.root.stat().st_size
            target.write_bytes(b"")
            self._current_bytes += self.root.stat().st_size - before_root - old_size
            if self._current_bytes + len(payload) > self.max_bytes:
                self.gc(max(0, self.max_bytes - len(payload)))
                payload = self._resident_manifest(payload)
            self._reserve_growth(len(payload))
            # GC can remove more inputs while reserving. Publish only the
            # resident plan, preserving the existing empty directory entry.
            payload = self._resident_manifest(payload)
            before_root = self.root.stat().st_size
            target.write_bytes(payload)
            self._current_bytes += len(payload) + self.root.stat().st_size - before_root
            self._manifest_may_be_incomplete = False
        else:
            # Generic controls retain atomic publication. Reserve both the
            # staging-file data and the create/rename index mutations first.
            self._reserve_growth(len(payload) + 2 * self._index_bytes())
            old_size = target.stat().st_size if target.exists() else 0
            temporary = self.root / f".{name}.tmp-{os.getpid()}"
            before_root = self.root.stat().st_size
            try:
                temporary.write_bytes(payload)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            self._current_bytes += len(payload) - old_size + self.root.stat().st_size - before_root
        if self._current_bytes > self.max_bytes:
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")


    def write_control(self, name: str, payload: bytes) -> None:
        self._replace_control(name, payload, enforce=True)



_STORE_POOL: OrderedDict[tuple[str, int, bool], _Store] = OrderedDict()


def _store_pool_key(root, max_bytes, verify_reads):
    return (os.path.normcase(os.path.abspath(os.fspath(root))), int(max_bytes), bool(verify_reads))


def _pooled_store(
    root: str | os.PathLike[str],
    max_bytes: int,
    verify_reads: bool,
) -> _Store:
    key = _store_pool_key(root, max_bytes, verify_reads)
    remembered = _STORE_POOL.pop(key, None)
    if remembered is None or not remembered.reusable():
        remembered = _Store(root, max_bytes, verify_reads)
    _STORE_POOL[key] = remembered
    while len(_STORE_POOL) > _STORE_POOL_LIMIT:
        _STORE_POOL.popitem(last=False)
    return remembered


def _recipe_payload(recipe: Any) -> dict[str, Any]:
    value = getattr(recipe, "recipe", recipe)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    if isinstance(value, (str, os.PathLike)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("render_sequence recipe must be a mapping, dataclass, MapScene, or JSON path")
    return canonical_json_value(dict(value), error_context="ANAMNESIS recipe")


def _label_payload(recipe: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    labels: list[Mapping[str, Any]] = []
    for layer in recipe.get("layers", ()) or ():
        if not isinstance(layer, Mapping):
            continue
        kind = str(layer.get("kind", layer.get("type", ""))).lower()
        if "label" in kind or "text" in layer or "labels" in layer:
            labels.append(layer)
    return labels


def _labels_visible(labels: Sequence[Mapping[str, Any]], frame: int) -> bool:
    if not labels:
        return False
    for label in labels:
        visibility = label.get("visible_frames")
        if visibility is None and isinstance(label.get("metadata"), Mapping):
            visibility = label["metadata"].get("visible_frames")
        if visibility is None:
            return True
        if isinstance(visibility, Mapping):
            start = int(visibility.get("start", frame))
            stop = int(visibility.get("stop", frame + 1))
            if start <= frame < stop:
                return True
        elif frame in {int(item) for item in visibility}:
            return True
    return False


def _pass_descriptor(label: str, frame: int | None, state: Mapping[str, Any]) -> bytes:
    # This is the complete reference pipeline identity. GPU integrations must
    # replace `shader_hashes` with CENSOR's exact preprocessed WGSL hashes and
    # fill the same state categories; absence disables caching at that callsite.
    structured_fingerprints = dict(
        state.get("structured_executor_sha256", {}) or {}
    )
    if label in structured_fingerprints:
        executor_kind = "structured-pass"
    elif label == "frame.output" and state.get("external_renderer_sha256"):
        executor_kind = "opaque-render-frame"
    else:
        executor_kind = "reference"
    descriptor = {
        "label": label,
        "shader_hashes": state.get("shader_hashes", {label: _sha256(label.encode())}),
        "sampler": state.get("sampler", "nearest-clamp"),
        "blend": state.get("blend", "replace"),
        "depth": state.get("depth", "disabled"),
        "formats": state.get("formats", ["rgba8unorm"]),
        "mip_counts": state.get("mip_counts", [1]),
        "primitive": state.get("primitive", "triangle-list"),
        "viewport": state.get("viewport", [0, 0, 1, 1]),
        "scissor": state.get("scissor", [0, 0, 1, 1]),
        "clear": state.get("clear", [0, 0, 0, 0]),
        "rng_seed": int(state.get("seed", 0)),
        "accumulation_frame_index": frame,
        "backend": state.get("backend", "reference-cpu"),
        "dx12_compiler": state.get("dx12_compiler", "not-applicable"),
        "external_renderer_sha256": (
            state.get("external_renderer_sha256")
            if label == "frame.output"
            else None
        ),
        "external_renderer_context_sha256": (
            state.get("external_renderer_context_sha256")
            if label == "frame.output"
            else None
        ),
        "structured_executor_sha256": structured_fingerprints.get(label),
        "structured_executor_context_sha256": dict(
            state.get("structured_executor_context_sha256", {}) or {}
        ).get(label),
        "executor_kind": executor_kind,
        "reference_work_factor": (
            int(state.get("reference_work_factor", 0))
            if executor_kind == "reference"
            else 0
        ),
    }
    return canonical_json_bytes(descriptor, error_context="ANAMNESIS pipeline descriptor")


def _execute_reference(
    label: str,
    inputs: Sequence[bytes],
    state: bytes,
    *,
    work_factor: int = 0,
) -> bytes:
    payload = bytearray(b"forge3d.anamnesis.reference-output/1")
    payload.extend(label.encode("utf-8"))
    payload.extend(state)
    for blob in inputs:
        payload.extend(len(blob).to_bytes(8, "little"))
        payload.extend(blob)
    digest = hashlib.sha256(payload).digest()
    for _ in range(max(0, int(work_factor))):
        digest = hashlib.sha256(digest + payload[:32]).digest()
    return digest


def _pass_material(
    label: str,
    pipeline: bytes,
    uniforms: bytes,
    inputs: Sequence[tuple[str, str]],
    capabilities: bytes,
    engine: bytes,
) -> dict[str, Any]:
    return {
        "label": label,
        "pipeline_descriptor_hash": _sha256(pipeline),
        "pipeline_descriptor_hex": pipeline.hex(),
        "uniform_sha256": _sha256(uniforms),
        "uniform_byte_length": len(uniforms),
        "uniform_hex": uniforms.hex(),
        "input_keys": [
            {"binding": binding, "key": key}
            for binding, key in sorted(inputs)
        ],
        "capability_fingerprint_sha256": _sha256(capabilities),
        "capability_fingerprint_hex": capabilities.hex(),
        "engine_fingerprint_sha256": _sha256(engine),
        "engine_fingerprint_hex": engine.hex(),
    }


def _pixel_recipe_payload(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical pixel-affecting recipe state for opaque render callbacks.

    Destination-only output fields are the only committed irrelevant inputs.
    Unknown fields remain in the projection: a conservative miss is preferable
    to a stale hit.
    """
    payload = json.loads(
        canonical_json_bytes(
            recipe, error_context="ANAMNESIS pixel recipe"
        ).decode("utf-8")
    )
    output = payload.get("output")
    if isinstance(output, dict):
        for name in ("path", "directory", "filename"):
            output.pop(name, None)
    return payload


def _load_manifest(store: _Store | Path | None) -> dict[str, str]:
    if store is None:
        return {}
    root = store.root if isinstance(store, _Store) else store
    pending = getattr(_PACKED_PRELOADS, "manifests", {}).get(os.path.abspath(root))
    snapshot = pending.result() if pending is not None else None
    # Admission may evict entries or invalidate build state. Only consume the
    # read-only snapshot after maintenance and a fresh file-identity check.
    if snapshot is not None and snapshot[0] == _manifest_signature(root / _LAST_MANIFEST):
        return dict(snapshot[1])
    return _read_manifest(root)


def _read_manifest(root: Path) -> dict[str, str]:
    path = root / _LAST_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if manifest.get("schema") != "forge3d.anamnesis.manifest/2":
        return {}
    entries = manifest.get("entries", {})
    if not isinstance(entries, Mapping):
        return {}
    return {str(name): str(key) for name, key in entries.items()}


def _instance_name(frame: int, label: str) -> str:
    return f"{frame}:{label}"


def render_sequence(
    recipe: Any,
    frames: Iterable[int] = range(600),
    cache: str | os.PathLike[str] | None = ".forge3d/cache",
    *,
    dry_run: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    verify_reads: bool = True,
    render_frame: Callable[[Mapping[str, Any], int], bytes | bytearray | memoryview] | None = None,
    render_frame_fingerprint: bytes | None = None,
    render_frame_context: bytes | None = None,
    pass_executors: Mapping[
        str,
        Callable[[Any, int, Sequence[bytes]], bytes | bytearray | memoryview],
    ]
    | None = None,
    pass_executor_fingerprints: Mapping[str, bytes] | None = None,
    pass_executor_contexts: Mapping[str, bytes] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    reference_work_factor: int = 0,
    reference_only: bool = False,
) -> SequenceResult:
    """Render a deterministic frame sequence through a Merkle pass graph.

    The predicted recompute set is calculated before any pass executes and is
    asserted equal to observed misses. Supplying ``cache=None`` executes every
    pass and performs no cache I/O. MapScene and SceneRecipe inputs use the
    built-in native adapter; arbitrary mappings retain the reference DAG.
    A real renderer may also be supplied for final frame bytes; cache hits
    skip that call entirely. Explicit external
    capabilities are retained without probing a local device. If that probe
    is needed and unavailable, rendering proceeds without cache I/O.
    Without an executor, ``reference_only=True`` is required for digest-only
    reference output; those bytes are not rendered images.
    """
    started = time.perf_counter()
    frame_numbers = [int(frame) for frame in frames]
    if render_frame is None and pass_executors is None:
        from ._anamnesis_scene import try_render_sequence

        native_sequence = try_render_sequence(
            recipe, frame_numbers, cache, dry_run=dry_run, max_bytes=max_bytes,
            verify_reads=verify_reads, capabilities=capabilities,
        )
        if native_sequence is not None:
            native_sequence.elapsed_seconds = time.perf_counter() - started
            return native_sequence
    payload = _recipe_payload(recipe)
    labels = _label_payload(payload)
    terrain = payload.get("terrain", {})
    lighting = payload.get("lighting", payload.get("sun", {}))
    atmosphere = payload.get("atmosphere", {})
    camera = payload.get("camera", {})
    state = dict(payload.get("anamnesis_state", {}) or {})
    executors = dict(pass_executors or {})
    if render_frame is None and not executors and not reference_only and not dry_run:
        raise RuntimeError("No frame renderer is available; supply render_frame/pass_executors or explicitly select reference_only=True for digest-only output")

    executor_fingerprints = dict(pass_executor_fingerprints or {})
    executor_contexts = dict(pass_executor_contexts or {})
    if render_frame is not None and executors:
        raise ValueError("render_frame and pass_executors are mutually exclusive")
    if set(executors) != set(executor_fingerprints):
        raise ValueError(
            "pass_executor_fingerprints must identify every structured pass executor"
        )
    if any(not bytes(value) for value in executor_fingerprints.values()):
        raise ValueError("structured pass executor fingerprints must be non-empty")
    if set(executors) != set(executor_contexts):
        raise ValueError(
            "pass_executor_contexts must enumerate every captured pixel-affecting input"
        )
    if any(not bytes(value) for value in executor_contexts.values()):
        raise ValueError("structured pass executor contexts must be non-empty")
    state["structured_executor_sha256"] = {
        str(label): _sha256(bytes(fingerprint))
        for label, fingerprint in executor_fingerprints.items()
    }
    state["structured_executor_context_sha256"] = {
        str(label): _sha256(bytes(context))
        for label, context in executor_contexts.items()
    }
    if render_frame is not None and not render_frame_fingerprint:
        raise ValueError(
            "render_frame requires render_frame_fingerprint containing the renderer/code identity"
        )
    if render_frame is not None and not render_frame_context:
        raise ValueError(
            "render_frame requires render_frame_context enumerating captured pixel inputs"
        )
    if render_frame_fingerprint is not None:
        state["external_renderer_sha256"] = _sha256(bytes(render_frame_fingerprint))
    if render_frame_context is not None:
        state["external_renderer_context_sha256"] = _sha256(
            bytes(render_frame_context)
        )
    if int(reference_work_factor) < 0:
        raise ValueError("reference_work_factor must be non-negative")
    state["reference_work_factor"] = int(reference_work_factor)
    capability_source = capabilities
    cache_eligible = True
    if capability_source is None:
        capability_source = {}
        if render_frame is not None or executors:
            native = get_native_module()
            try:
                capability_source = dict(native.capabilities())
                cache_eligible = all(
                    name in capability_source
                    for name in ("granted", "limits", "backend", "dx12_compiler", "naga_capabilities")
                )
            except Exception:
                # An unknown device must never share a persistent key with
                # another unknown device. Preserve rendering, without reuse.
                cache_eligible = False
    if "backend" not in state:
        state["backend"] = str(
            capability_source.get("backend")
            or (
                "reference-cpu"
                if render_frame is None and not executors
                else "external-renderer-unknown"
            )
        )
    caps = capability_fingerprint(
        capability_source,
        backend=None if "backend" in capability_source else str(state["backend"]),
    )
    engine = engine_fingerprint()
    cache_root = Path(cache) if cache is not None and cache_eligible else None
    store = None
    report = CacheReport()
    pixel_recipe = _pixel_recipe_payload(payload)
    plans: list[_PlannedPass] = []
    plans_by_instance: dict[str, _PlannedPass] = {}
    descriptor_templates: dict[str, bytes] = {}
    scene_adapter = getattr(recipe, "_anamnesis_scene_adapter", None)
    scene_encoder = (scene_adapter.encode_uniforms if scene_adapter is not None
                     and scene_adapter.cache_requested and scene_adapter.cache_eligible else None)
    # Reuse key calculations only for identical canonical bytes and context.
    # Store reads remain independently verified; this memo contains no pixels.
    key_memo_enabled = (cache_root is not None and not dry_run and scene_encoder is not None
                        and getattr(scene_adapter, "fast_uniforms", False))
    remembered = (_STORE_POOL.get(_store_pool_key(cache_root, max_bytes, verify_reads))
                  if key_memo_enabled else None)
    previous_key_material = remembered._key_material if remembered is not None else {}
    current_key_material = {}
    key_context = (caps, engine, pass_key, canonical_json_bytes, json.dumps,
                   getattr(scene_encoder, "__func__", None),
                   getattr(get_native_module(), "anamnesis_pass_key", None))
    key_memo_enabled = key_memo_enabled and all(
        fn is None or isinstance(fn, (FunctionType, BuiltinFunctionType)) for fn in key_context[2:])

    def declare(
        label: str,
        frame: int,
        leaf: Any,
        inputs: Sequence[tuple[str, str]],
    ) -> str:
        template = descriptor_templates.get(label)
        if template is None:
            template = _pass_descriptor(label, None, state)
            descriptor_templates[label] = template
        # The immutable descriptor changes only its frame index within one
        # sequence. Preserve its exact canonical JSON bytes, including null
        # for global passes; the alphabetically first field is unambiguous.
        prefix = b'{"accumulation_frame_index":'
        if not template.startswith(prefix + b"null"):
            raise RuntimeError("ANAMNESIS descriptor lost its canonical frame field")
        pipeline = (template if frame < 0 else
                    prefix + str(frame).encode("ascii") + template[len(prefix) + len(b"null"):])
        uniform_state: dict[str, Any] = {"frame": frame, "state": leaf}
        if label == "frame.output" and render_frame is not None:
            # An opaque callback may read any recipe field. Key its final
            # output from the complete pixel projection; structured pass
            # executors are required for finer invalidation.
            uniform_state["opaque_pixel_recipe"] = pixel_recipe
        input_keys = [
            (binding, plans_by_instance[input_instance].key)
            for binding, input_instance in inputs
        ]
        context = f"ANAMNESIS {label} uniforms"
        uniforms = (scene_encoder(label, uniform_state, error_context=context) if scene_encoder is not None else
                    canonical_json_bytes(uniform_state, error_context=context))
        instance = _instance_name(frame, label)
        memo_inputs = ((key_context, label, pipeline, uniforms, tuple(input_keys))
                       if key_memo_enabled else None)
        material = previous_key_material.get(instance) if memo_inputs is not None else None
        # Locate the previous instance cheaply, then compare every exact key
        # input. Hashing the complete uniform payload again for a Python dict
        # duplicates work already performed by the canonical native key.
        if material is None or material[0] != memo_inputs:
            key = pass_key(label, pipeline, uniforms, input_keys, caps, engine)
            material = (memo_inputs, key)
        else:
            key = material[1]
            # Retain the already owned bytes, rather than a second equal copy.
            pipeline, uniforms = material[0][2:4]
        if memo_inputs is not None:
            current_key_material[instance] = material
        plan = _PlannedPass(
            instance=instance,
            label=label,
            frame=frame,
            pipeline=pipeline,
            uniforms=uniforms,
            inputs=tuple(inputs),
            key=key,
            state_value=leaf,
        )
        plans.append(plan)
        plans_by_instance[instance] = plan
        return instance

    if scene_adapter is not None:
        output_instances = scene_adapter.declare(declare, frame_numbers)
    else:
        output_instances = []
    terrain_instance = declare("terrain.geometry", -1, terrain, ()) if scene_adapter is None else None
    atmosphere_instance = declare("atmosphere.lut", -1, atmosphere, ()) if scene_adapter is None else None
    label_instance: str | None = None
    if labels and scene_adapter is None:
        label_instance = declare("label.compile", -1, labels, ())

    for frame in frame_numbers if scene_adapter is None else ():
        frame_camera = {"camera": camera, "frame": frame}
        shadow_instance = declare(
            "shadow.map",
            frame,
            {"lighting": lighting, **frame_camera},
            (("terrain.geometry@0", terrain_instance),),
        )
        shade_instance = declare(
            "terrain.shade",
            frame,
            frame_camera,
            (
                ("terrain.geometry@0", terrain_instance),
                ("atmosphere.lut@1", atmosphere_instance),
                ("shadow.map@2", shadow_instance),
            ),
        )
        accumulation_instance = declare(
            "accumulation",
            frame,
            {"samples": payload.get("output", {}).get("samples", 1)},
            (("terrain.shade@0", shade_instance),),
        )
        final_instance = accumulation_instance
        if label_instance is not None and _labels_visible(labels, frame):
            final_instance = declare(
                "label.composite",
                frame,
                {"visible": True},
                (
                    ("accumulation@0", accumulation_instance),
                    ("label.compile@1", label_instance),
                ),
            )
        output_instance = declare(
            "frame.output",
            frame,
            {"output": _pixel_recipe_payload({"output": payload.get("output", {})})["output"]},
            (("frame.input@0", final_instance),),
        )
        output_instances.append(output_instance)

    # Freeze the semantic recipe-diff prediction before inspecting the content
    # store or executing an encoder. The previous manifest is the independent
    # build-state contract: changed Merkle keys must execute, unchanged keys
    # must restore. A missing/corrupt unchanged entry therefore becomes an
    # explicit prediction mismatch instead of silently redefining prediction
    # from the same store lookup used by execution.
    if cache_root is not None and not dry_run:
        store = _pooled_store(cache_root, max_bytes, verify_reads)
        store._active_plan_keys = {plan.key for plan in plans}
    # Initial budget enforcement can invalidate the prior manifest. Read it
    # after maintenance, before independently freezing the recompute set.
    previous_manifest = _load_manifest(cache_root)
    predicted_plans = [
        plan for plan in plans if previous_manifest.get(plan.instance) != plan.key
    ]
    predicted = [(plan.frame, plan.label) for plan in predicted_plans]
    predicted_instances = {plan.instance for plan in predicted_plans}

    # An unchanged cached node is already represented by its Merkle key in a
    # cached descendant. Restore only terminal outputs and the direct inputs
    # needed by actual misses; reading every shadow/shade/accumulation ancestor
    # would turn a 600-frame label edit into thousands of pointless filesystem
    # operations. Elided ancestors remain logical hits because the restored
    # descendant key commits to them transitively.
    required_instances = set(output_instances)
    pending_instances = list(output_instances)
    while pending_instances:
        instance = pending_instances.pop()
        plan = plans_by_instance[instance]
        if instance not in predicted_instances:
            continue
        for _, input_instance in plan.inputs:
            if input_instance not in required_instances:
                required_instances.add(input_instance)
                pending_instances.append(input_instance)

    # A resident descendant commits to all its ancestors. A partial manifest
    # need not keep their evicted payloads, unless an actual miss needs them.
    predicted_plans = [plan for plan in predicted_plans if plan.instance in required_instances]
    predicted = [(plan.frame, plan.label) for plan in predicted_plans]
    predicted_instances = {plan.instance for plan in predicted_plans}

    if store is not None:
        # Restored descendants commit to elided ancestors. Retain the bytes
        # this run actually needs before those ancestor payloads during LRU
        # maintenance, so an oversized historical label cannot displace all
        # terminal outputs solely because its key is still in the full DAG.
        store._active_plan_keys = {plan.key for plan in plans if plan.instance in required_instances}

    # Prediction and reachability are already frozen. Only now inspect the
    # memory-hit signatures in parallel; dry runs never touch the store.
    restore_keys = tuple(dict.fromkeys(
        plan.key for plan in plans
        if plan.instance in required_instances and plan.instance not in predicted_instances
    ))
    signatures = (store.memory_signatures(restore_keys)
                  if store is not None and not dry_run else {})
    observed: list[tuple[int, str]] = []
    blobs_by_instance: dict[str, bytes] = {}
    memo: dict[str, bytes] = {}
    protected_keys = (plan.key for plan in plans
                      if plan.instance in required_instances and plan.instance not in predicted_instances)
    prepare_executors = getattr(scene_adapter, "prepare_executors", None)
    deferred_writes = None
    if (store is not None and not dry_run and scene_adapter is not None
            and store._signature is not None and store._appendable_index):
        from ._anamnesis_scene import _SceneExecution
        if type(scene_adapter) is _SceneExecution and scene_adapter.fast_uniforms:
            # Built-in executors consume owned inputs and prior pass bytes,
            # never intermediate store writes. Keep the complete write set
            # inside the same timer and protected-read scope.
            deferred_writes = []
    with (prepare_executors(predicted_plans) if store is not None and not dry_run and prepare_executors is not None else nullcontext(),
          store.protect_reads(protected_keys) if store is not None and not dry_run else nullcontext(),
          store.prefetch_reads(restore_keys) if store is not None and not dry_run else nullcontext()):
        if not dry_run:
            for plan in plans:
                force_recompute = plan.instance in predicted_instances
                if plan.instance not in required_instances:
                    if force_recompute:
                        raise RuntimeError(
                            f"ANAMNESIS internal reachability error: {plan.instance}"
                        )
                    report.hits.append(plan.label)
                    continue
                if not force_recompute and plan.key in memo:
                    blob = memo[plan.key]
                    report.hits.append(plan.label)
                    blobs_by_instance[plan.instance] = blob
                    continue
                cached = (
                    store.get(plan.key, known_signature=(signatures[plan.key][0].result()[plan.key] if plan.key in signatures else None))
                    if store is not None and not force_recompute
                    else None
                )
                if cached is not None:
                    blob, meta = cached
                    report.hits.append(plan.label)
                    report.bytes_read += len(blob)
                    report.wall_ms_saved += max(
                        0.0, float(meta.get("measured_wall_ms", 0.0))
                    )
                else:
                    if not force_recompute:
                        raise RuntimeError(
                            "ANAMNESIS prediction mismatch: "
                            f"{[(plan.frame, plan.label)]!r}"
                        )
                    pass_started = time.perf_counter()
                    if plan.label in executors:
                        blob = bytes(
                            executors[plan.label](
                                plan.state_value,
                                plan.frame,
                                [
                                    blobs_by_instance[input_instance]
                                    for _, input_instance in plan.inputs
                                ],
                            )
                        )
                    elif plan.label == "frame.output" and render_frame is not None:
                        blob = bytes(render_frame(payload, plan.frame))
                    else:
                        blob = _execute_reference(
                            plan.label,
                            [
                                blobs_by_instance[input_instance]
                                for _, input_instance in plan.inputs
                            ],
                            plan.pipeline + plan.uniforms + caps + engine,
                            work_factor=reference_work_factor,
                        )
                    elapsed_ms = (time.perf_counter() - pass_started) * 1000.0
                    observed.append((plan.frame, plan.label))
                    report.misses.append(plan.label)
                    if store is not None:
                        material = _pass_material(
                            plan.label,
                            plan.pipeline,
                            plan.uniforms,
                            [
                                (binding, plans_by_instance[input_instance].key)
                                for binding, input_instance in plan.inputs
                            ],
                            caps,
                            engine,
                        )
                        if deferred_writes is None:
                            store.put_pass(plan.key, blob, material,
                                frame=plan.frame, measured_wall_ms=elapsed_ms)
                        else:
                            deferred_writes.append((plan.key, blob, material, plan.frame, elapsed_ms))
                        report.bytes_written += len(blob)
                memo[plan.key] = blob
                blobs_by_instance[plan.instance] = blob
            if deferred_writes is not None:
                store.put_pass_batch(deferred_writes)

    outputs = [blobs_by_instance.get(instance, b"") for instance in output_instances]
    hashes = [_sha256(output) for output in outputs]
    current_manifest = {plan.instance: plan.key for plan in plans}

    if not dry_run and set(predicted) != set(observed):
        difference = sorted(set(predicted).symmetric_difference(observed))
        raise RuntimeError(f"ANAMNESIS prediction mismatch: {difference!r}")
    if store is not None and not dry_run:
        manifest = canonical_json_bytes(
            {
                "schema": "forge3d.anamnesis.manifest/2",
                "recipe_sha256": _sha256(
                    canonical_json_bytes(
                        pixel_recipe, error_context="ANAMNESIS manifest recipe"
                    )
                ),
                "entries": current_manifest,
            },
            error_context="ANAMNESIS manifest",
        )
        store.flush_accesses()
        store.write_control(_LAST_MANIFEST, manifest)
        # Only the latest successful plan is retained by the existing store
        # pool; old scene states, adapters and GPU resources are not retained.
        store._key_material = current_key_material
    return SequenceResult(
        frame_hashes=hashes,
        frame_blobs=outputs,
        cache_report=report,
        predicted_recompute=predicted,
        observed_recompute=observed,
        elapsed_seconds=time.perf_counter() - started,
        pass_keys=current_manifest,
        reference_only=reference_only,
        bypass_reason=("capability_fingerprint_unavailable" if cache is not None and not cache_eligible else None),
    )


def _explain_tree(store: _Store, key: str, seen: set[str]) -> dict[str, Any]:
    if key in seen:
        return {"key": key, "cycle": True}
    seen.add(key)
    restored = store.get(key, touch=False)
    if restored is None:
        return {"key": key, "status": "missing"}
    _, meta = restored
    derivation = dict(meta.get("derivation", {}))
    if derivation.get("kind") == "leaf":
        return {
            "key": key,
            "pass_label": meta.get("pass_label"),
            "kind": "leaf",
            "content_sha256": derivation.get("content_sha256"),
            "byte_length": meta.get("byte_length"),
        }
    material = dict(derivation.get("material", {}))
    input_records = list(material.get("input_keys", meta.get("input_keys", ())))
    reconstructed = None
    try:
        reconstructed = pass_key(
            str(material["label"]),
            bytes.fromhex(str(material["pipeline_descriptor_hex"])),
            bytes.fromhex(str(material["uniform_hex"])),
            [
                (str(item["binding"]), str(item["key"]))
                for item in input_records
            ],
            bytes.fromhex(str(material["capability_fingerprint_hex"])),
            bytes.fromhex(str(material["engine_fingerprint_hex"])),
        )
    except (KeyError, TypeError, ValueError):
        pass
    return {
        "key": key,
        "pass_label": meta.get("pass_label"),
        "frame": meta.get("frame"),
        "kind": "pass",
        "reconstructed_key": reconstructed,
        "reconstructs": reconstructed == key,
        "pipeline_descriptor_hash": material.get("pipeline_descriptor_hash"),
        "pipeline_descriptor_hex": material.get("pipeline_descriptor_hex"),
        "uniform_sha256": material.get("uniform_sha256"),
        "uniform_byte_length": material.get("uniform_byte_length"),
        "uniform_hex": material.get("uniform_hex"),
        "capability_fingerprint_sha256": material.get(
            "capability_fingerprint_sha256"
        ),
        "capability_fingerprint_hex": material.get(
            "capability_fingerprint_hex"
        ),
        "engine_fingerprint_sha256": material.get("engine_fingerprint_sha256"),
        "engine_fingerprint_hex": material.get("engine_fingerprint_hex"),
        "inputs": [
            {
                "binding": item.get("binding"),
                "derivation": _explain_tree(
                    store, str(item.get("key")), seen.copy()
                ),
            }
            for item in input_records
        ],
    }


def explain(key: str, cache: str | os.PathLike[str] = ".forge3d/cache") -> dict[str, Any]:
    """Print and return a key's recursive derivation tree."""
    tree = _explain_tree(_Store(cache, _DEFAULT_MAX_BYTES, False), key.lower(), set())
    print(json.dumps(tree, indent=2, sort_keys=True))
    return tree


def gc(max_bytes: int, cache: str | os.PathLike[str] = ".forge3d/cache") -> int:
    """Evict least-recently-used entries until the store fits ``max_bytes``."""
    native = get_native_module()
    if native is not None and hasattr(native, "anamnesis_store_gc"):
        removed = int(native.anamnesis_store_gc(os.fspath(cache), int(max_bytes)))
    else:
        store = _Store(cache, max(max_bytes, 1), False)
        removed = store.gc(max_bytes)
    if removed:
        (Path(cache) / _LAST_MANIFEST).unlink(missing_ok=True)
    return removed


def verify(cache: str | os.PathLike[str] = ".forge3d/cache") -> dict[str, int]:
    """Re-hash all blobs and quarantine any corrupt or incomplete entry."""
    native = get_native_module()
    if native is not None and hasattr(native, "anamnesis_store_verify"):
        result = dict(
            native.anamnesis_store_verify(
                os.fspath(cache),
                _DEFAULT_MAX_BYTES,
            )
        )
    else:
        result = _Store(cache, _DEFAULT_MAX_BYTES, True).verify()
    if result.get("quarantined", 0):
        (Path(cache) / _LAST_MANIFEST).unlink(missing_ok=True)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge3d.anamnesis")
    parser.add_argument("command", choices=("explain", "verify", "gc", "render-sequence"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("--cache", default=".forge3d/cache")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Produce reference digests, not rendered frames")
    args = parser.parse_args(argv)
    if args.command == "explain":
        if not args.value:
            parser.error("explain requires a key")
        explain(args.value, args.cache)
    elif args.command == "verify":
        print(json.dumps(verify(args.cache), sort_keys=True))
    elif args.command == "gc":
        if args.value is None:
            parser.error("gc requires max_bytes")
        print(gc(int(args.value), args.cache))
    else:
        if args.value is None:
            parser.error("render-sequence requires a recipe JSON path")
        result = render_sequence(
            args.value,
            frames=range(args.frames),
            cache=args.cache,
            dry_run=args.dry_run,
            reference_only=args.reference_only,
        )
        print(
            json.dumps(
                {
                    "predicted_recompute": result.predicted_recompute,
                    "observed_recompute": result.observed_recompute,
                    "prediction_matches": result.prediction_matches,
                    "reference_only": result.reference_only,
                    "cache_report": result.cache_report.to_dict(),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
