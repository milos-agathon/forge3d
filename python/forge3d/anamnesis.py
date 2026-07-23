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
_LAST_MANIFEST = "last_manifest.json"


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
        "naga_capabilities": sorted(
            str(item)
            for item in (
                naga_capabilities
                if naga_capabilities
                else value.get("naga_capabilities", ())
            )
        ),
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

    @property
    def prediction_matches(self) -> bool:
        return set(self.predicted_recompute) == set(self.observed_recompute)


@dataclass(frozen=True)
class _PlannedPass:
    instance: str
    label: str
    frame: int
    pipeline: bytes
    uniforms: bytes
    inputs: tuple[tuple[str, str], ...]
    key: str
    state_value: Any


class _Store:
    def __init__(self, root: str | os.PathLike[str], max_bytes: int, verify_reads: bool):
        if int(max_bytes) <= 0:
            raise ValueError("ANAMNESIS max_bytes must be positive")
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.verify_reads = bool(verify_reads)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "quarantine").mkdir(exist_ok=True)
        self._current_bytes = self.disk_bytes()
        if self._current_bytes > self.max_bytes:
            self.gc(self.max_bytes)
        if self._current_bytes > self.max_bytes:
            raise ValueError(
                "ANAMNESIS store control/quarantine footprint exceeds max_bytes"
            )

    @staticmethod
    def _tree_bytes(path: Path) -> int:
        try:
            total = path.stat().st_size
        except OSError:
            return 0
        if path.is_dir():
            for child in path.iterdir():
                total += _Store._tree_bytes(child)
        return total

    def disk_bytes(self) -> int:
        """Return the complete footprint, including metadata and directories."""
        return self._tree_bytes(self.root)

    def entry(self, key: str) -> Path:
        return self.root / key[:2] / key

    def get(
        self,
        key: str,
        *,
        touch: bool = True,
        quarantine: bool = True,
    ) -> tuple[bytes, dict[str, Any]] | None:
        path = self.entry(key)
        if not path.is_dir():
            return None
        try:
            blob = (path / _BLOB_NAME).read_bytes()
            meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if quarantine:
                self._quarantine(path, key)
            return None
        valid = self._metadata_valid(key, blob, meta)
        if not valid:
            if quarantine:
                self._quarantine(path, key)
            return None
        if touch:
            meta_path = path / _META_NAME
            before = meta_path.stat().st_size
            meta["last_access_unix_ms"] = time.time_ns() // 1_000_000
            self._write_meta(path, meta)
            self._current_bytes += meta_path.stat().st_size - before
            os.utime(path, None)
        return blob, meta

    def put_pass(
        self,
        key: str,
        blob: bytes,
        material: Mapping[str, Any],
        *,
        frame: int | None = None,
        measured_wall_ms: float = 0.0,
    ) -> None:
        input_keys = list(material.get("input_keys", ()))
        self._put(
            key,
            blob,
            pass_label=str(material["label"]),
            input_keys=input_keys,
            derivation={"kind": "pass", "material": dict(material)},
            frame=frame,
            measured_wall_ms=measured_wall_ms,
        )

    def put_leaf(self, key: str, blob: bytes, *, label: str) -> None:
        if leaf_key(blob) != key:
            raise ValueError("ANAMNESIS leaf key does not match leaf content")
        self._put(
            key,
            blob,
            pass_label=label,
            input_keys=[],
            derivation={"kind": "leaf", "content_sha256": _sha256(blob)},
            frame=None,
            measured_wall_ms=0.0,
        )

    def _put(
        self,
        key: str,
        blob: bytes,
        *,
        pass_label: str,
        input_keys: Sequence[Mapping[str, Any]],
        derivation: Mapping[str, Any],
        frame: int | None,
        measured_wall_ms: float,
    ) -> None:
        target = self.entry(key)
        if target.is_dir() and self.get(key) is not None:
            return
        prefix = target.parent
        if not prefix.exists():
            before_root = self.root.stat().st_size
            prefix.mkdir(parents=True)
            self._current_bytes += prefix.stat().st_size
            self._current_bytes += self.root.stat().st_size - before_root
        now = time.time_ns() // 1_000_000
        try:
            creation_engine = json.loads(
                bytes.fromhex(
                    str(
                        dict(derivation.get("material", {})).get(
                            "engine_fingerprint_hex", ""
                        )
                    )
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            creation_engine = {
                "crate_version": "unknown",
                "git_sha": "unknown",
                "naga_version": "unknown",
                "wgsl_tree_sha256": "unknown",
            }
        complete = {
            "schema": "forge3d.anamnesis.store/1",
            "key": key,
            "pass_label": pass_label,
            "input_keys": list(input_keys),
            "byte_length": len(blob),
            "creation_engine_fingerprint": creation_engine,
            "self_hash": _sha256(blob),
            "created_unix_ms": now,
            "last_access_unix_ms": now,
            "derivation": dict(derivation),
            "frame": frame,
            "measured_wall_ms": max(0.0, float(measured_wall_ms)),
        }
        before_prefix = prefix.stat().st_size
        temp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=prefix))
        prefix_growth = prefix.stat().st_size - before_prefix
        try:
            (temp / _BLOB_NAME).write_bytes(blob)
            self._write_meta(temp, complete)
            staged_bytes = self._tree_bytes(temp)
            if staged_bytes > self.max_bytes:
                raise ValueError(
                    f"ANAMNESIS entry requires {staged_bytes} bytes, "
                    f"exceeding max_bytes={self.max_bytes}"
                )
            projected = self._current_bytes + prefix_growth + staged_bytes
            if projected > self.max_bytes:
                # This full scan is reserved for the near/full-store path. The
                # staged temp directory is included in total footprint but is
                # excluded from active-entry eviction.
                self._current_bytes = self.disk_bytes()
                self.gc(self.max_bytes)
                if self._current_bytes > self.max_bytes:
                    raise ValueError(
                        "ANAMNESIS complete on-disk footprint exceeds max_bytes"
                    )
            try:
                temp.replace(target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        if projected <= self.max_bytes:
            self._current_bytes = projected
        else:
            self._current_bytes = self.disk_bytes()
        if self._current_bytes > self.max_bytes:
            shutil.rmtree(target, ignore_errors=True)
            self._current_bytes = self.disk_bytes()
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")
        if self._current_bytes >= int(self.max_bytes * 0.95):
            self._current_bytes = self.disk_bytes()
            if self._current_bytes > self.max_bytes:
                shutil.rmtree(target, ignore_errors=True)
                self._current_bytes = self.disk_bytes()
                raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")

    def _write_meta(self, path: Path, meta: Mapping[str, Any]) -> None:
        data = json.dumps(
            canonical_json_value(meta, error_context="ANAMNESIS store metadata"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        temporary = path / f".{_META_NAME}.tmp-{os.getpid()}"
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(path / _META_NAME)

    def entries(self) -> list[Path]:
        return sorted(
            path
            for prefix in self.root.iterdir()
            if prefix.is_dir() and prefix.name != "quarantine" and len(prefix.name) == 2
            for path in prefix.iterdir()
            if path.is_dir() and len(path.name) == 64
        )

    def gc(self, target_bytes: int) -> int:
        records: list[tuple[int, str, int, Path]] = []
        total = self.disk_bytes()
        for path in self.entries():
            try:
                meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
                size = self._tree_bytes(path)
                accessed = int(meta["last_access_unix_ms"])
            except (OSError, ValueError, KeyError):
                size, accessed = self._tree_bytes(path), 0
            records.append((accessed, path.name, size, path))
        for path in sorted((self.root / "quarantine").iterdir()):
            if path.is_dir():
                records.append(
                    (
                        path.stat().st_mtime_ns // 1_000_000,
                        path.name,
                        self._tree_bytes(path),
                        path,
                    )
                )
        removed = 0
        for _, _, size, path in sorted(records):
            if total <= max(0, int(target_bytes)):
                break
            shutil.rmtree(path)
            total = self.disk_bytes()
            removed += size
        self._current_bytes = self.disk_bytes()
        return removed

    def verify(self) -> dict[str, int]:
        result = {"valid": 0, "quarantined": 0, "bytes_checked": 0}
        for path in self.entries():
            key = path.name
            try:
                blob = (path / _BLOB_NAME).read_bytes()
                meta = json.loads((path / _META_NAME).read_text(encoding="utf-8"))
                valid = len(key) == 64 and self._metadata_valid(key, blob, meta)
            except (OSError, ValueError):
                blob, valid = b"", False
            result["bytes_checked"] += len(blob)
            if valid:
                result["valid"] += 1
            else:
                self._quarantine(path, key)
                result["quarantined"] += 1
        return result

    def _quarantine(self, path: Path, key: str) -> None:
        if not path.exists():
            return
        target = self.root / "quarantine" / f"{key}-{time.time_ns()}"
        path.replace(target)
        # The prior manifest asserted this entry was restorable. Once runtime
        # integrity rejects it, that manifest is no longer a valid independent
        # prediction baseline for a retry.
        (self.root / _LAST_MANIFEST).unlink(missing_ok=True)
        self._current_bytes = self.disk_bytes()

    @staticmethod
    def _metadata_valid(key: str, blob: bytes, meta: Mapping[str, Any]) -> bool:
        if not (
            meta.get("schema") == "forge3d.anamnesis.store/1"
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

    def write_control(self, name: str, payload: bytes) -> None:
        target = self.root / name
        temporary = self.root / f".{name}.tmp-{os.getpid()}"
        temporary.write_bytes(payload)
        projected = self.disk_bytes()
        self._current_bytes = projected
        if projected > self.max_bytes:
            self.gc(self.max_bytes - temporary.stat().st_size)
        if self.disk_bytes() > self.max_bytes:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"ANAMNESIS control file {name!r} exceeds the complete store budget"
            )
        temporary.replace(target)
        self._current_bytes = self.disk_bytes()
        if self._current_bytes > self.max_bytes:
            target.unlink(missing_ok=True)
            self._current_bytes = self.disk_bytes()
            raise ValueError("ANAMNESIS complete on-disk footprint exceeds max_bytes")


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


def _load_manifest(store: _Store | None) -> dict[str, str]:
    if store is None:
        return {}
    path = store.root / _LAST_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if manifest.get("schema") != "forge3d.anamnesis.manifest/1":
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
    executors = dict(pass_executors or {})
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
    if "backend" not in state:
        detected_backend = None
        if (render_frame is not None or executors) and capabilities is None:
            native = get_native_module()
            if native is not None and hasattr(native, "engine_info"):
                try:
                    detected_backend = dict(native.engine_info()).get("backend")
                except Exception:
                    detected_backend = None
        state["backend"] = str(
            detected_backend
            or (
                "reference-cpu"
                if render_frame is None and not executors
                else "external-renderer-unknown"
            )
        )
    capability_source = (
        capabilities
        if capabilities is not None
        else ({} if render_frame is None and not executors else None)
    )
    caps = capability_fingerprint(capability_source, backend=str(state["backend"]))
    engine = engine_fingerprint()
    store = _Store(cache, max_bytes, verify_reads) if cache is not None else None
    report = CacheReport()
    previous_manifest = _load_manifest(store)
    pixel_recipe = _pixel_recipe_payload(payload)
    plans: list[_PlannedPass] = []
    plans_by_instance: dict[str, _PlannedPass] = {}

    def declare(
        label: str,
        frame: int,
        leaf: Any,
        inputs: Sequence[tuple[str, str]],
    ) -> str:
        pipeline = _pass_descriptor(label, None if frame < 0 else frame, state)
        uniform_state: dict[str, Any] = {"frame": frame, "state": leaf}
        if label == "frame.output" and render_frame is not None:
            # An opaque callback may read any recipe field. Key its final
            # output from the complete pixel projection; structured pass
            # executors are required for finer invalidation.
            uniform_state["opaque_pixel_recipe"] = pixel_recipe
        uniforms = canonical_json_bytes(
            uniform_state,
            error_context=f"ANAMNESIS {label} uniforms",
        )
        input_keys = [
            (binding, plans_by_instance[input_instance].key)
            for binding, input_instance in inputs
        ]
        key = pass_key(label, pipeline, uniforms, input_keys, caps, engine)
        instance = _instance_name(frame, label)
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

    terrain_instance = declare("terrain.geometry", -1, terrain, ())
    atmosphere_instance = declare("atmosphere.lut", -1, atmosphere, ())
    label_instance: str | None = None
    if labels:
        label_instance = declare("label.compile", -1, labels, ())

    output_instances: list[str] = []
    for frame in frame_numbers:
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
    predicted_plans = [
        plan for plan in plans if previous_manifest.get(plan.instance) != plan.key
    ]
    predicted = [(plan.frame, plan.label) for plan in predicted_plans]
    predicted_instances = {plan.instance for plan in predicted_plans}

    observed: list[tuple[int, str]] = []
    blobs_by_instance: dict[str, bytes] = {}
    memo: dict[str, bytes] = {}
    if not dry_run:
        for plan in plans:
            force_recompute = plan.instance in predicted_instances
            if not force_recompute and plan.key in memo:
                blob = memo[plan.key]
                report.hits.append(plan.label)
                blobs_by_instance[plan.instance] = blob
                continue
            cached = (
                store.get(plan.key)
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
                report.bytes_written += len(blob)
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
                    store.put_pass(
                        plan.key,
                        blob,
                        material,
                        frame=plan.frame,
                        measured_wall_ms=elapsed_ms,
                    )
            memo[plan.key] = blob
            blobs_by_instance[plan.instance] = blob

    outputs = [blobs_by_instance.get(instance, b"") for instance in output_instances]
    hashes = [_sha256(output) for output in outputs]
    current_manifest = {plan.instance: plan.key for plan in plans}

    if not dry_run and set(predicted) != set(observed):
        difference = sorted(set(predicted).symmetric_difference(observed))
        raise RuntimeError(f"ANAMNESIS prediction mismatch: {difference!r}")
    if store is not None and not dry_run:
        manifest = canonical_json_bytes(
            {
                "schema": "forge3d.anamnesis.manifest/1",
                "recipe_sha256": _sha256(
                    canonical_json_bytes(
                        pixel_recipe, error_context="ANAMNESIS manifest recipe"
                    )
                ),
                "entries": current_manifest,
            },
            error_context="ANAMNESIS manifest",
        )
        store.write_control(_LAST_MANIFEST, manifest)
    return SequenceResult(
        frame_hashes=hashes,
        frame_blobs=outputs,
        cache_report=report,
        predicted_recompute=predicted,
        observed_recompute=observed,
        elapsed_seconds=time.perf_counter() - started,
        pass_keys=current_manifest,
    )


def _explain_tree(store: _Store, key: str, seen: set[str]) -> dict[str, Any]:
    if key in seen:
        return {"key": key, "cycle": True}
    seen.add(key)
    path = store.entry(key) / _META_NAME
    if not path.is_file():
        return {"key": key, "status": "missing"}
    meta = json.loads(path.read_text(encoding="utf-8"))
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
