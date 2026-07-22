"""Content-addressed execution for deterministic offline render sequences.

ANAMNESIS is deliberately inert unless a cache path is supplied. It does not
participate in the interactive viewer. The sequence scheduler models the
offline terrain, atmosphere, shadow, accumulation, label and output passes as
a Merkle DAG; callers may provide ``render_frame`` to make the final pass emit
real image bytes, while the built-in reference executor is deterministic and
GPU-independent for hermeticity and store testing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

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
_LAST_RECIPE = "last_recipe.json"
_FAST_PACK = "fastpack.json"
_FAST_PACK_MAX_BYTES = 64 * 1024 * 1024


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
    input_keys: Sequence[str],
    capability_fingerprint_bytes: bytes,
    engine_fingerprint_bytes: bytes,
) -> str:
    """Compute the hermetic key of one pass.

    ``pipeline_descriptor`` must include exact WGSL hashes plus sampler,
    blend/depth/primitive/target state, viewport, scissor, clear values, RNG
    seed, accumulation frame, backend and DX12 compiler. The function refuses
    malformed input keys and hashes exact uniform bytes including padding.
    """
    normalized_inputs = sorted(str(key).lower() for key in input_keys)
    for key in normalized_inputs:
        if len(key) != 64:
            raise ValueError(f"invalid ANAMNESIS input key: {key!r}")
        try:
            bytes.fromhex(key)
        except ValueError as exc:
            raise ValueError(f"invalid ANAMNESIS input key: {key!r}") from exc
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
    for key in normalized_inputs:
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
            "git_sha": os.environ.get("FORGE3D_GIT_SHA", "unknown"),
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
        "granted_features": sorted(str(item) for item in value.get("granted", ())),
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
        "naga_capabilities": sorted(str(item) for item in naga_capabilities),
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

    @property
    def prediction_matches(self) -> bool:
        return set(self.predicted_recompute) == set(self.observed_recompute)


class _Store:
    def __init__(self, root: str | os.PathLike[str], max_bytes: int, verify_reads: bool):
        if int(max_bytes) <= 0:
            raise ValueError("ANAMNESIS max_bytes must be positive")
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.verify_reads = bool(verify_reads)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "quarantine").mkdir(exist_ok=True)
        self._fast: dict[str, dict[str, Any]] = {}
        self._fast_pack_bytes = 0
        fast_path = self.root / _FAST_PACK
        if not self.verify_reads and fast_path.is_file():
            try:
                self._fast_pack_bytes = fast_path.stat().st_size
                packed = json.loads(fast_path.read_text(encoding="utf-8"))
                if packed.get("schema") == "forge3d.anamnesis.fastpack/1":
                    entries = dict(packed.get("entries", {}))
                    total = 0
                    for key, item in entries.items():
                        blob = bytes.fromhex(str(item["blob_hex"]))
                        meta = dict(item["meta"])
                        if not (
                            meta.get("key") == key
                            and int(meta.get("byte_length", -1)) == len(blob)
                            and meta.get("self_hash") == _sha256(blob)
                        ):
                            raise ValueError("invalid ANAMNESIS fast-pack entry")
                        total += len(blob)
                    if {path.name for path in self.entries()} == set(entries):
                        self._fast = entries
                        self._current_bytes = total
            except (KeyError, OSError, TypeError, ValueError):
                self._fast = {}
                self._fast_pack_bytes = 0
        if not self._fast:
            self._current_bytes = self._inventory_bytes()
        if self._current_bytes + self._fast_pack_bytes > self.max_bytes:
            fast_path.unlink(missing_ok=True)
            self._fast.clear()
            self._fast_pack_bytes = 0
            self.gc(self.max_bytes)

    def _inventory_bytes(self) -> int:
        total = 0
        for path in self.entries():
            try:
                meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
                total += int(meta.get("byte_length", 0))
            except (OSError, ValueError):
                continue
        return total

    def entry(self, key: str) -> Path:
        return self.root / key[:2] / key

    def peek(self, key: str) -> bool:
        # Prediction uses the same integrity decision as execution. A corrupt
        # entry is quarantined now and predicted as a miss, so dry-run and
        # observed recompute sets cannot diverge on damaged storage.
        return self.get(key) is not None

    def get(self, key: str, *, touch: bool = True) -> tuple[bytes, dict[str, Any]] | None:
        packed = self._fast.get(key)
        if packed is not None:
            try:
                blob = bytes.fromhex(str(packed["blob_hex"]))
                meta = dict(packed["meta"])
                valid = (
                    meta.get("key") == key
                    and int(meta.get("byte_length", -1)) == len(blob)
                    and meta.get("self_hash") == _sha256(blob)
                )
                if valid:
                    return blob, meta
            except (KeyError, TypeError, ValueError):
                pass
            # A fast-pack is only an I/O optimization, never a relaxation of
            # the content-addressed integrity boundary. Discard any malformed
            # or tampered entry and fall through to the self-verifying store.
            self._fast.pop(key, None)
        path = self.entry(key)
        if not path.is_dir():
            return None
        try:
            blob = (path / _BLOB_NAME).read_bytes()
            meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._quarantine(path, key)
            return None
        valid = (
            meta.get("key") == key
            and int(meta.get("byte_length", -1)) == len(blob)
            and meta.get("self_hash") == _sha256(blob)
        )
        if not valid:
            self._quarantine(path, key)
            return None
        # Directory mtime is the LRU clock. Updating it avoids rewriting and
        # fsyncing self-describing JSON on every cache hit.
        if touch:
            os.utime(path, None)
        return blob, meta

    def put(self, key: str, blob: bytes, meta: Mapping[str, Any]) -> None:
        if len(blob) > self.max_bytes:
            raise ValueError("ANAMNESIS blob is larger than the entire store budget")
        target = self.entry(key)
        if target.is_dir() and self.get(key) is not None:
            return
        (self.root / _FAST_PACK).unlink(missing_ok=True)
        self._fast_pack_bytes = 0
        if self._current_bytes + len(blob) > self.max_bytes:
            self.gc(self.max_bytes - len(blob))
        target.parent.mkdir(parents=True, exist_ok=True)
        now = time.time_ns()
        complete = {
            "schema": "forge3d.anamnesis.store/1",
            **dict(meta),
            "key": key,
            "byte_length": len(blob),
            "self_hash": _sha256(blob),
            "created_ns": now,
            "last_access_ns": now,
        }
        temp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=target.parent))
        try:
            (temp / _BLOB_NAME).write_bytes(blob)
            self._write_meta(temp, complete)
            try:
                temp.replace(target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        self._current_bytes += len(blob)
        if not self.verify_reads and self._current_bytes <= _FAST_PACK_MAX_BYTES:
            self._fast[key] = {"blob_hex": blob.hex(), "meta": complete}

    def flush_fast_pack(self) -> None:
        if self.verify_reads or self._current_bytes > _FAST_PACK_MAX_BYTES:
            return
        payload = {
            "schema": "forge3d.anamnesis.fastpack/1",
            "entries": self._fast,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if self._current_bytes + len(data.encode("utf-8")) > self.max_bytes:
            (self.root / _FAST_PACK).unlink(missing_ok=True)
            return
        temporary = self.root / f".{_FAST_PACK}.tmp-{os.getpid()}"
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(self.root / _FAST_PACK)
        self._fast_pack_bytes = len(data.encode("utf-8"))

    def _write_meta(self, path: Path, meta: Mapping[str, Any]) -> None:
        data = json.dumps(
            canonical_json_value(meta, error_context="ANAMNESIS store metadata"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        temporary = path / f".{_META_NAME}.tmp-{os.getpid()}"
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(path / _META_NAME)

    def entries(self) -> list[Path]:
        return sorted(
            path
            for prefix in self.root.iterdir()
            if prefix.is_dir() and prefix.name != "quarantine" and len(prefix.name) == 2
            for path in prefix.iterdir()
            if path.is_dir()
        )

    def gc(self, target_bytes: int) -> int:
        (self.root / _FAST_PACK).unlink(missing_ok=True)
        self._fast_pack_bytes = 0
        records: list[tuple[int, str, int, Path]] = []
        total = 0
        for path in self.entries():
            try:
                meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
                size = int(meta["byte_length"])
                accessed = path.stat().st_mtime_ns
            except (OSError, ValueError, KeyError):
                size, accessed = 0, 0
            total += size
            records.append((accessed, path.name, size, path))
        removed = 0
        for _, _, size, path in sorted(records):
            if total <= max(0, int(target_bytes)):
                break
            shutil.rmtree(path)
            self._fast.pop(path.name, None)
            total -= size
            removed += size
        self._current_bytes = total
        return removed

    def verify(self) -> dict[str, int]:
        result = {"valid": 0, "quarantined": 0, "bytes_checked": 0}
        for path in self.entries():
            key = path.name
            try:
                blob = (path / _BLOB_NAME).read_bytes()
                meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
                valid = (
                    len(key) == 64
                    and meta.get("key") == key
                    and int(meta.get("byte_length", -1)) == len(blob)
                    and meta.get("self_hash") == _sha256(blob)
                )
            except (OSError, ValueError):
                blob, valid = b"", False
            result["bytes_checked"] += len(blob)
            if valid:
                result["valid"] += 1
            else:
                self._quarantine(path, key)
                result["quarantined"] += 1
        if result["quarantined"]:
            (self.root / _FAST_PACK).unlink(missing_ok=True)
        return result

    def _quarantine(self, path: Path, key: str) -> None:
        if not path.exists():
            return
        target = self.root / "quarantine" / f"{key}-{time.time_ns()}"
        path.replace(target)
        self._fast.pop(key, None)


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


def _derivation(
    label: str,
    pipeline: bytes,
    uniforms: bytes,
    inputs: Sequence[str],
    capabilities: bytes,
    engine: bytes,
    frame: int,
) -> dict[str, Any]:
    return {
        "pass_label": label,
        "frame": frame,
        "input_keys": sorted(inputs),
        "pipeline_descriptor_hash": _sha256(pipeline),
        "uniform_sha256": _sha256(uniforms),
        "uniform_byte_length": len(uniforms),
        "capability_fingerprint_sha256": _sha256(capabilities),
        "engine_fingerprint_sha256": _sha256(engine),
        "creation_engine_fingerprint": json.loads(engine),
    }


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
    capabilities: Mapping[str, Any] | None = None,
    reference_work_factor: int = 0,
) -> SequenceResult:
    """Render a deterministic frame sequence through a Merkle pass graph.

    The predicted recompute set is calculated before any pass executes and is
    asserted equal to observed misses. Supplying ``cache=None`` executes every
    pass and performs no cache I/O. A real renderer may be supplied for final
    frame bytes; cache hits skip that call entirely.
    """
    started = time.perf_counter()
    payload = _recipe_payload(recipe)
    frame_numbers = [int(frame) for frame in frames]
    labels = _label_payload(payload)
    terrain = payload.get("terrain", {})
    lighting = payload.get("lighting", payload.get("sun", {}))
    atmosphere = payload.get("atmosphere", {})
    camera = payload.get("camera", {})
    state = dict(payload.get("anamnesis_state", {}) or {})
    if render_frame is not None and not render_frame_fingerprint:
        raise ValueError(
            "render_frame requires render_frame_fingerprint containing the renderer/code identity"
        )
    if render_frame_fingerprint is not None:
        state["external_renderer_sha256"] = _sha256(bytes(render_frame_fingerprint))
    if int(reference_work_factor) < 0:
        raise ValueError("reference_work_factor must be non-negative")
    state["reference_work_factor"] = int(reference_work_factor)
    if "backend" not in state:
        detected_backend = None
        if render_frame is not None and capabilities is None:
            native = get_native_module()
            if native is not None and hasattr(native, "engine_info"):
                try:
                    detected_backend = dict(native.engine_info()).get("backend")
                except Exception:
                    detected_backend = None
        state["backend"] = str(
            detected_backend
            or ("reference-cpu" if render_frame is None else "external-renderer-unknown")
        )
    capability_source = (
        capabilities
        if capabilities is not None
        else ({} if render_frame is None else None)
    )
    caps = capability_fingerprint(capability_source, backend=str(state["backend"]))
    engine = engine_fingerprint()
    store = _Store(cache, max_bytes, verify_reads) if cache is not None else None
    report = CacheReport()
    predicted: list[tuple[int, str]] = []
    observed: list[tuple[int, str]] = []
    outputs: list[bytes] = []
    hashes: list[str] = []
    memo: dict[str, bytes] = {}

    def run(label: str, frame: int, leaf: Any, inputs: Sequence[tuple[str, bytes]]) -> tuple[str, bytes]:
        pipeline = _pass_descriptor(label, None if frame < 0 else frame, state)
        uniforms = canonical_json_bytes(
            {"frame": frame, "state": leaf},
            error_context=f"ANAMNESIS {label} uniforms",
        )
        state_key = leaf_key(uniforms)
        input_keys = [state_key, *(key for key, _ in inputs)]
        state_hit = store.get(state_key) if store is not None and not dry_run else None
        if store is not None and not dry_run and state_hit is None:
            store.put(
                state_key,
                uniforms,
                {
                    "pass_label": f"leaf:{label}",
                    "frame": frame,
                    "input_keys": [],
                    "content_sha256": _sha256(uniforms),
                    "creation_engine_fingerprint": json.loads(engine),
                },
            )
        key = pass_key(label, pipeline, uniforms, input_keys, caps, engine)
        if key in memo:
            report.hits.append(label)
            return key, memo[key]
        cached = store.get(key, touch=not dry_run) if store is not None else None
        predicted_miss = cached is None
        if predicted_miss:
            predicted.append((frame, label))
        if dry_run:
            blob = b""
            if cached is not None:
                blob = cached[0]
            memo[key] = blob
            return key, blob
        if cached is not None:
            blob = cached[0]
            report.hits.append(label)
            report.bytes_read += len(blob)
            report.wall_ms_saved += max(0.0, float(cached[1].get("measured_wall_ms", 0.0)))
            memo[key] = blob
            return key, blob
        pass_started = time.perf_counter()
        if label == "frame.output" and render_frame is not None:
            blob = bytes(render_frame(payload, frame))
        else:
            blob = _execute_reference(
                label,
                [item for _, item in inputs],
                pipeline + uniforms + caps + engine,
                work_factor=reference_work_factor,
            )
        elapsed_ms = (time.perf_counter() - pass_started) * 1000.0
        observed.append((frame, label))
        report.misses.append(label)
        report.bytes_written += len(blob)
        if store is not None:
            meta = _derivation(label, pipeline, uniforms, input_keys, caps, engine, frame)
            meta["measured_wall_ms"] = elapsed_ms
            store.put(key, blob, meta)
        memo[key] = blob
        return key, blob

    terrain_key, terrain_blob = run("terrain.geometry", -1, terrain, ())
    atmosphere_key, atmosphere_blob = run("atmosphere.lut", -1, atmosphere, ())
    label_key: str | None = None
    label_blob: bytes | None = None
    if labels:
        label_key, label_blob = run("label.compile", -1, labels, ())

    for frame in frame_numbers:
        frame_camera = {"camera": camera, "frame": frame}
        shadow_key, shadow_blob = run(
            "shadow.map", frame, {"lighting": lighting, **frame_camera}, ((terrain_key, terrain_blob),)
        )
        shade_key, shade_blob = run(
            "terrain.shade",
            frame,
            frame_camera,
            ((terrain_key, terrain_blob), (atmosphere_key, atmosphere_blob), (shadow_key, shadow_blob)),
        )
        accum_key, accum_blob = run(
            "accumulation", frame, {"samples": payload.get("output", {}).get("samples", 1)}, ((shade_key, shade_blob),)
        )
        final_input = (accum_key, accum_blob)
        if label_key is not None and label_blob is not None and _labels_visible(labels, frame):
            final_input = run(
                "label.composite", frame, {"visible": True}, ((accum_key, accum_blob), (label_key, label_blob))
            )
        output_key, output_blob = run(
            "frame.output", frame, {"output": payload.get("output", {})}, (final_input,)
        )
        outputs.append(output_blob)
        hashes.append(_sha256(output_blob))

    if not dry_run and set(predicted) != set(observed):
        difference = sorted(set(predicted).symmetric_difference(observed))
        raise RuntimeError(f"ANAMNESIS prediction mismatch: {difference!r}")
    if store is not None and not dry_run:
        store.flush_fast_pack()
        (store.root / _LAST_RECIPE).write_bytes(
            canonical_json_bytes(payload, error_context="ANAMNESIS last recipe") + b"\n"
        )
    return SequenceResult(
        frame_hashes=hashes,
        frame_blobs=outputs,
        cache_report=report,
        predicted_recompute=predicted,
        observed_recompute=observed,
        elapsed_seconds=time.perf_counter() - started,
    )


def _explain_tree(store: _Store, key: str, seen: set[str]) -> dict[str, Any]:
    if key in seen:
        return {"key": key, "cycle": True}
    seen.add(key)
    path = store.entry(key) / _META_NAME
    if not path.is_file():
        return {"key": key, "status": "missing"}
    meta = json.loads(path.read_text(encoding="utf-8"))
    return {
        "key": key,
        "pass_label": meta.get("pass_label"),
        "frame": meta.get("frame"),
        "pipeline_descriptor_hash": meta.get("pipeline_descriptor_hash"),
        "uniform_sha256": meta.get("uniform_sha256"),
        "capability_fingerprint_sha256": meta.get("capability_fingerprint_sha256"),
        "engine_fingerprint_sha256": meta.get("engine_fingerprint_sha256"),
        "inputs": [_explain_tree(store, child, seen.copy()) for child in meta.get("input_keys", ())],
    }


def explain(key: str, cache: str | os.PathLike[str] = ".forge3d/cache") -> dict[str, Any]:
    """Print and return a key's recursive derivation tree."""
    tree = _explain_tree(_Store(cache, _DEFAULT_MAX_BYTES, False), key.lower(), set())
    print(json.dumps(tree, indent=2, sort_keys=True))
    return tree


def gc(max_bytes: int, cache: str | os.PathLike[str] = ".forge3d/cache") -> int:
    """Evict least-recently-used entries until the store fits ``max_bytes``."""
    store = _Store(cache, max(max_bytes, 1), False)
    removed = store.gc(max_bytes)
    store.flush_fast_pack()
    return removed


def verify(cache: str | os.PathLike[str] = ".forge3d/cache") -> dict[str, int]:
    """Re-hash all blobs and quarantine any corrupt or incomplete entry."""
    return _Store(cache, _DEFAULT_MAX_BYTES, True).verify()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge3d.anamnesis")
    parser.add_argument("command", choices=("explain", "verify", "gc", "render-sequence"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("--cache", default=".forge3d/cache")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
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
        )
        print(
            json.dumps(
                {
                    "predicted_recompute": result.predicted_recompute,
                    "observed_recompute": result.observed_recompute,
                    "prediction_matches": result.prediction_matches,
                    "cache_report": result.cache_report.to_dict(),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
