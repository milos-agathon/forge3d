"""Out-of-core terrain data handles."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path


@dataclass(frozen=True)
class VTStore:
    """Validated handle to a forge3d ``.f3dvt`` packed page store."""

    path: str

    def __fspath__(self) -> str:
        return self.path


def open_vt_store(path: str | PathLike[str]) -> VTStore:
    """Open a disk-backed TESSELLA virtual-texture store.

    The native renderer performs full header, directory, and per-page SHA-256
    validation when the handle is first used. This lightweight boundary check
    catches wrong paths and formats before a GPU render begins.
    """

    resolved = Path(fspath(path)).expanduser().resolve()
    with resolved.open("rb") as stream:
        magic = stream.read(8)
    if magic != b"F3DVT1\0\0":
        raise ValueError(f"{resolved} is not a forge3d VT store")
    return VTStore(str(resolved))


__all__ = ["VTStore", "open_vt_store"]
