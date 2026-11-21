"""Progress events and durable tile records for recoverable terrain jobs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from operator import index
from pathlib import Path
import struct
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np

from .certificate import _call_render_progress


@dataclass(frozen=True)
class TerrainRenderProgress:
    """Progress at a completed frame or an atomically published poster tile."""

    completed_frames: int
    tile: tuple[int, int] | None
    variance: float | None
    variance_at_frame: int | None
    variance_age_frames: int | None
    elapsed_seconds: float
    checkpoint_path: Path | None
    tile_complete: bool


class TerrainRenderCancelled(RuntimeError):
    """A progress callback requested cancellation at a completed boundary."""

    code = "terrain_render_cancelled"

    def __init__(self, progress: TerrainRenderProgress):
        super().__init__("Terrain rendering cancelled by the progress callback")
        self.progress = progress
        self.checkpoint_path = progress.checkpoint_path


ProgressCallback = Callable[[TerrainRenderProgress], bool | None]


class Progress:
    def __init__(self, callback: ProgressCallback | None, last_path: Path | None = None):
        self.callback = callback
        self.last_path = last_path
        self.started = perf_counter()

    def notify(self, frames, variance_at, variance, *, tile=None, path=None, tile_complete=False):
        if path is not None:
            self.last_path = path
        if self.callback is None:
            return
        event = TerrainRenderProgress(
            completed_frames=frames,
            tile=tile,
            variance=variance if variance_at else None,
            variance_at_frame=variance_at or None,
            variance_age_frames=frames - variance_at if variance_at else None,
            elapsed_seconds=perf_counter() - self.started,
            checkpoint_path=self.last_path,
            tile_complete=tile_complete,
        )
        if _call_render_progress(self.callback, (event,)) is False:
            raise TerrainRenderCancelled(event)


def options(callback, checkpoint_path, checkpoint_every, resume, *, poster=False):
    if callback is not None and not callable(callback):
        raise TypeError("on_progress must be callable")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    path = Path(checkpoint_path).absolute() if checkpoint_path is not None else None
    if checkpoint_every is not None:
        if isinstance(checkpoint_every, (bool, np.bool_)):
            raise TypeError("checkpoint_every must be an integer frame count")
        checkpoint_every = index(checkpoint_every)
        if not 0 < checkpoint_every <= 0xFFFFFFFF:  # native absolute frame counter
            raise ValueError("checkpoint_every must be a positive u32 frame count")
    if path is None and (resume or checkpoint_every is not None):
        raise ValueError("resume and checkpoint_every require checkpoint_path")
    if path is not None and not (poster or resume or checkpoint_every is not None):
        raise ValueError("starting a terrain checkpoint requires checkpoint_every")
    return path, checkpoint_every


def array_digest(array, dtype):
    if array is None:
        return None
    data = np.ascontiguousarray(array, dtype=dtype)
    pixels = memoryview(data).cast("B") if data.size else b""
    return data.shape, hashlib.sha256(pixels).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _fingerprint(value):
    return hashlib.sha256(_json_bytes(value)).digest()


@contextmanager
def _atomic_file(path):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".continua-", delete=False) as out:
        temp = Path(out.name)
        try:
            yield out
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temp, path)
            if os.name == "posix":
                fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            out.close()
            temp.unlink(missing_ok=True)


@contextmanager
def _poster_lock(path):
    # The OS releases the lock when a worker dies. Leave the inode in place
    # so a concurrent opener can never lock a replacement inode.
    with open(path / ".continua.lock", "a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Terrain poster checkpoint is already in use: {path}") from error
        yield


_POSTER_MAGIC = b"F3DPOST\x01"
_TILE_MAGIC = b"F3DTILE\x01"
_DIGEST_BYTES = hashlib.sha256().digest_size
_TILE_FIELDS = (
    "frames", "variance", "peak_host_visible_bytes", "peak_device_local_bytes",
    "reservoir_valid_count", "reservoir_m_min", "reservoir_m_max",
    "minmax_pyramid_bytes", "gpu_resource_bytes",
    "solar_azimuth_deg", "solar_elevation_deg",
)
_TRAVERSAL_FIELDS = ("node_visits", "minmax_loads", "height_loads", "rays")
_TILE_DATA = struct.Struct("<4IIdQQQIIQQddII8Q")
_TILE_HOST_BUDGET = 512 * 1024 * 1024  # terrain tile contract


def _read_exact(stream, size, path):
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError(f"Truncated terrain poster checkpoint: {path}")
    return data


class PosterCheckpoint:
    """One locked job directory with a manifest, frame checkpoints and tile records."""

    def __init__(self, path, resume, scene_identity, runtime):
        self.path = path
        self.resume = resume
        self.runtime = runtime
        fields = scene_identity | runtime
        self.identity = {key: _fingerprint(value) for key, value in sorted(fields.items())}
        field_bytes = b"".join(self.identity.values())
        self.identity_digest = hashlib.sha256(field_bytes).digest()
        self.manifest_body = _POSTER_MAGIC + field_bytes
        self.manifest_digest = hashlib.sha256(self.manifest_body).digest()
        self.manifest = path / "poster.continua"

    @contextmanager
    def open(self):
        if self.resume and not self.path.is_dir():
            raise FileNotFoundError(f"Terrain poster checkpoint directory does not exist: {self.path}")
        self.path.mkdir(exist_ok=True)
        with _poster_lock(self.path):
            if self.resume:
                self._check_manifest()
            else:
                if self.manifest.exists():
                    raise FileExistsError(f"Use resume=True or a new checkpoint directory: {self.path}")
                if any(entry.name != ".continua.lock" for entry in self.path.iterdir()):
                    raise ValueError(f"Terrain poster checkpoint directory must be empty: {self.path}")
                with _atomic_file(self.manifest) as out:
                    out.write(self.manifest_body)
                    out.write(self.manifest_digest)
            yield self

    def _check_manifest(self):
        size = len(_POSTER_MAGIC) + _DIGEST_BYTES * (len(self.identity) + 1)
        with self.manifest.open("rb") as stream:
            data = _read_exact(stream, size, self.manifest)
            if stream.read(1):
                raise RuntimeError(f"Invalid terrain poster checkpoint length: {self.manifest}")
        if not data.startswith(_POSTER_MAGIC):
            raise RuntimeError("Unsupported terrain poster checkpoint version")
        if hashlib.sha256(data[:-_DIGEST_BYTES]).digest() != data[-_DIGEST_BYTES:]:
            raise RuntimeError("Terrain poster checkpoint checksum mismatch")
        for offset, (name, expected) in enumerate(self.identity.items()):
            start = len(_POSTER_MAGIC) + offset * _DIGEST_BYTES
            if data[start:start + _DIGEST_BYTES] != expected:
                raise RuntimeError(f"Terrain poster checkpoint {name} mismatch")

    def paths(self, ix, iy):
        return (self.path / f"tile-{ix}-{iy}.frame", self.path / f"tile-{ix}-{iy}.rgba")

    def save_tile(self, path, window, rgba, stats):
        traversal = stats["traversal"]
        data = _TILE_DATA.pack(
            *window,
            *(stats[key] for key in _TILE_FIELDS),
            traversal["frame_index"], traversal["mip_count"],
            *(traversal[lane][field] for lane in ("primary", "shadow") for field in _TRAVERSAL_FIELDS),
        )
        header = _TILE_MAGIC + self.identity_digest + data
        digest = hashlib.sha256(header)
        pixels = memoryview(rgba).cast("B")
        with _atomic_file(path) as out:
            out.write(header)
            out.write(pixels)
            digest.update(pixels)
            out.write(digest.digest())

    def restore_tile(self, path, window, destination, max_frames, min_frames, variance_threshold):
        header_size = len(_TILE_MAGIC) + _DIGEST_BYTES + _TILE_DATA.size
        _, _, tw, th = window
        with path.open("rb", buffering=0) as stream:
            if os.fstat(stream.fileno()).st_size != header_size + tw * th * 4 + _DIGEST_BYTES:
                raise RuntimeError(f"Terrain poster tile length does not match its dimensions: {path}")
            header = _read_exact(stream, header_size, path)
            if not header.startswith(_TILE_MAGIC):
                raise RuntimeError(f"Unsupported terrain poster tile version: {path}")
            if header[len(_TILE_MAGIC):len(_TILE_MAGIC) + _DIGEST_BYTES] != self.identity_digest:
                raise RuntimeError(f"Terrain poster tile identity mismatch: {path}")
            x, y, width, height, *values = _TILE_DATA.unpack(header[-_TILE_DATA.size:])
            if (x, y, width, height) != window:
                raise RuntimeError(f"Terrain poster tile pixel offset or dimensions mismatch: {path}")
            digest = hashlib.sha256(header)
            # Stream directly into the existing plate, including edge tiles
            # whose destination is not contiguous across rows.
            for row in destination:
                pixels = memoryview(row).cast("B")
                if stream.readinto(pixels) != len(pixels):
                    raise RuntimeError(f"Truncated terrain poster tile: {path}")
                digest.update(pixels)
            if digest.digest() != _read_exact(stream, _DIGEST_BYTES, path):
                raise RuntimeError(f"Terrain poster tile checksum mismatch: {path}")
        stats = dict(zip(_TILE_FIELDS, values))
        frame_index, mip_count, *counts = values[len(_TILE_FIELDS):]
        frames = stats["frames"]
        variance = stats["variance"]
        if (
            not max(2, min_frames) <= frames <= max_frames
            or not math.isfinite(variance)
            or not 0 <= variance < variance_threshold
            or frame_index != frames - 1
            or stats["peak_host_visible_bytes"] > _TILE_HOST_BUDGET
        ):
            raise RuntimeError(f"Invalid terrain poster tile convergence or diagnostics: {path}")
        stats.update(
            convergence_metric="frame_mean_estimator_variance",
            converged=True,
            traversal={
                "frame_index": frame_index,
                "mip_count": mip_count,
                "primary": dict(zip(_TRAVERSAL_FIELDS, counts[:4])),
                "shadow": dict(zip(_TRAVERSAL_FIELDS, counts[4:])),
            },
            sun_source="manual_angles",
            resumed=True,
            resume_frame=frames,
            frames_dispatched=0,
        )
        return stats
