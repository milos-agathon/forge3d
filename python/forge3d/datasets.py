"""Bundled and on-demand sample datasets for tutorials, tests, and notebooks."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "available",
    "bundled",
    "dataset_info",
    "fetch",
    "fetch_cityjson",
    "fetch_copc",
    "fetch_dem",
    "list_datasets",
    "mini_dem",
    "mini_dem_path",
    "remote",
    "sample_boundaries",
    "sample_boundaries_path",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = Path.home() / ".forge3d" / "datasets"
_DEFAULT_DATASET_BASE_URL = "https://raw.githubusercontent.com/forge3d/forge3d/main/assets/"


@dataclass(frozen=True)
class _RemoteDataset:
    """Metadata for a larger dataset that may be resolved locally or downloaded."""

    name: str
    kind: str
    filename: str
    description: str
    relative_url: str
    known_hash: str
    local_path: Optional[str] = None


_REMOTE_DATASETS: Dict[str, _RemoteDataset] = {
    "rainier": _RemoteDataset(
        name="rainier",
        kind="dem",
        filename="dem_rainier.tif",
        description="Mount Rainier DEM used in viewer tutorials and gallery scenes.",
        relative_url="tif/dem_rainier.tif",
        known_hash="sha256:875b243474b151175f76037acd60c2149ac2e46fba9ba2bbce0c9a6998015dd3",
        local_path="assets/tif/dem_rainier.tif",
    ),
    "fuji": _RemoteDataset(
        name="fuji",
        kind="dem",
        filename="Mount_Fuji_30m.tif",
        description="Mount Fuji DEM used in labels and buildings examples.",
        relative_url="tif/Mount_Fuji_30m.tif",
        known_hash="sha256:cff39b4e02d7ba13c48f3d8b1a4080d40ada753ade62fa951459fe4e01e98b48",
        local_path="assets/tif/Mount_Fuji_30m.tif",
    ),
    "swiss": _RemoteDataset(
        name="swiss",
        kind="dem",
        filename="switzerland_dem.tif",
        description="Swiss Alps DEM used in overlay and legend examples.",
        relative_url="tif/switzerland_dem.tif",
        known_hash="sha256:d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478",
        local_path="assets/tif/switzerland_dem.tif",
    ),
    "luxembourg": _RemoteDataset(
        name="luxembourg",
        kind="dem",
        filename="luxembourg_dem.tif",
        description="Luxembourg DEM used with the rail overlay gallery example.",
        relative_url="tif/luxembourg_dem.tif",
        known_hash="sha256:c332f7abb41a911449596f86277e05ef340cef37620115c1c78a56af35cc83e8",
        local_path="assets/tif/luxembourg_dem.tif",
    ),
    "swiss-land-cover": _RemoteDataset(
        name="swiss-land-cover",
        kind="overlay",
        filename="switzerland_land_cover.tif",
        description="Swiss land-cover raster used as a draped terrain overlay.",
        relative_url="tif/switzerland_land_cover.tif",
        known_hash="sha256:6b254585be4982ed9e8da63b8536ecc2f5fa4c64c6545db06c73eb1fe39a8f7f",
        local_path="assets/tif/switzerland_land_cover.tif",
    ),
    "luxembourg-rail": _RemoteDataset(
        name="luxembourg-rail",
        kind="vector",
        filename="luxembourg_rail.gpkg",
        description="Rail network overlay used in the Luxembourg gallery scene.",
        relative_url="gpkg/luxembourg_rail.gpkg",
        known_hash="sha256:980dc1659c712c67a80c9b57acf31eb3b26b14213f69a5c6c7f7ffb84385e1ec",
        local_path="assets/gpkg/luxembourg_rail.gpkg",
    ),
    "mount-fuji-places": _RemoteDataset(
        name="mount-fuji-places",
        kind="vector",
        filename="Mount_Fuji_places.gpkg",
        description="Sample placenames around Mount Fuji for labels and callouts.",
        relative_url="gpkg/Mount_Fuji_places.gpkg",
        known_hash="sha256:9e46ad2e55ba9b945b3dc5c29ad29e80d881a32d4a90ecf8a2637f864567a530",
        local_path="assets/gpkg/Mount_Fuji_places.gpkg",
    ),
    "sample-buildings": _RemoteDataset(
        name="sample-buildings",
        kind="cityjson",
        filename="sample_buildings.city.json",
        description="Small CityJSON building set for tutorial and test scenes.",
        relative_url="geojson/sample_buildings.city.json",
        known_hash="sha256:ac7e90c90d11bd83259e1af41228447dba9afd3b2fde49e319ef286e960e71c9",
        local_path="assets/geojson/sample_buildings.city.json",
    ),
    "mount-fuji-buildings": _RemoteDataset(
        name="mount-fuji-buildings",
        kind="geojson",
        filename="mount_fuji_buildings.geojson",
        description="GeoJSON building footprints used in the Mount Fuji buildings demo.",
        relative_url="geojson/mount_fuji_buildings.geojson",
        known_hash="sha256:19a124e80b12c7cd7181020d70e1b2e2004acc7a8b8a72b3463a701575c07f6e",
        local_path="assets/geojson/mount_fuji_buildings.geojson",
    ),
    "mt-st-helens": _RemoteDataset(
        name="mt-st-helens",
        kind="copc",
        filename="MtStHelens.laz",
        description="LAZ point cloud used in the point cloud tutorial and gallery entry.",
        relative_url="lidar/MtStHelens.laz",
        known_hash="sha256:4474530433fda8c40fbb621ed4dd78b02c9c90cbe4ef33588a73883663d5bd57",
        local_path="assets/lidar/MtStHelens.laz",
    ),
}


def _require_data_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled dataset not found: {path}. Reinstall forge3d or rebuild the wheel."
        )
    return path


def _dataset_base_url(base_url: Optional[str]) -> str:
    if base_url:
        return base_url.rstrip("/") + "/"
    env_url = os.environ.get("FORGE3D_DATASETS_BASE_URL")
    if env_url:
        return env_url.rstrip("/") + "/"
    return _DEFAULT_DATASET_BASE_URL


def _local_dataset_path(meta: _RemoteDataset) -> Optional[Path]:
    if meta.local_path is None:
        return None

    local_rel_path = Path(meta.local_path)
    roots: list[Path] = []
    env_root = os.environ.get("FORGE3D_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path.cwd())
    roots.append(_REPO_ROOT)

    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        candidate = root / local_rel_path
        if candidate.exists():
            return candidate
    return None


def _download_dataset(meta: _RemoteDataset, cache_dir: Optional[str], base_url: Optional[str]) -> Path:
    try:
        import pooch
    except ImportError as exc:
        raise ImportError(
            "Remote dataset downloads require pooch. Install with: pip install forge3d[datasets]"
        ) from exc

    target_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        pooch.retrieve(
            url=_dataset_base_url(base_url) + meta.relative_url,
            known_hash=meta.known_hash,
            fname=meta.filename,
            path=target_dir,
            progressbar=False,
        )
    )


def _fetch_kind(name: str, expected_kind: str, cache_dir: Optional[str], base_url: Optional[str]) -> Path:
    meta = _REMOTE_DATASETS.get(name)
    if meta is not None and meta.kind != expected_kind:
        raise ValueError(
            f"Dataset '{name}' has kind '{meta.kind}', expected '{expected_kind}'."
        )
    return fetch(name, cache_dir=cache_dir, base_url=base_url)


def bundled() -> List[str]:
    """Return the names of datasets bundled inside the wheel."""
    return ["mini_dem", "sample_boundaries"]


def remote() -> List[str]:
    """Return the names of datasets that can be fetched on demand."""
    return sorted(_REMOTE_DATASETS)


def available() -> List[str]:
    """Return all bundled and remote dataset names."""
    return bundled() + remote()


def list_datasets() -> List[Dict[str, str]]:
    """Return the dataset registry as a list of metadata records."""
    return [{"name": name, **meta} for name, meta in sorted(dataset_info().items())]


def dataset_info() -> Dict[str, Dict[str, str]]:
    """Return short metadata for the built-in dataset registry."""
    info: Dict[str, Dict[str, str]] = {
        "mini_dem": {
            "kind": "bundled",
            "path": str(mini_dem_path()),
            "description": "Synthetic 256x256 DEM bundled for tutorials and notebook examples.",
        },
        "sample_boundaries": {
            "kind": "bundled",
            "path": str(sample_boundaries_path()),
            "description": "Tiny GeoJSON polygon dataset for vector-overlay tutorials.",
        },
    }
    for name, meta in sorted(_REMOTE_DATASETS.items()):
        info[name] = {
            "kind": meta.kind,
            "filename": meta.filename,
            "description": meta.description,
        }
        local_path = _local_dataset_path(meta)
        if local_path is not None:
            info[name]["path"] = str(local_path)
    return info


def mini_dem_path() -> Path:
    """Return the packaged path to the bundled mini DEM."""
    return _require_data_file(_DATA_DIR / "mini_dem.npy")


def mini_dem() -> np.ndarray:
    """Load the bundled mini DEM as ``float32``."""
    data = np.load(mini_dem_path())
    return np.asarray(data, dtype=np.float32)


def sample_boundaries_path() -> Path:
    """Return the packaged path to the bundled sample boundaries GeoJSON."""
    return _require_data_file(_DATA_DIR / "sample_boundaries.geojson")


def sample_boundaries() -> dict:
    """Load the bundled vector overlay example as a GeoJSON dict."""
    return json.loads(sample_boundaries_path().read_text(encoding="utf-8"))


def fetch(name: str, cache_dir: Optional[str] = None, base_url: Optional[str] = None) -> Path:
    """Resolve a dataset by name and return a local path.

    Bundled datasets always resolve immediately. Larger datasets are first resolved
    from the local repository checkout when available, then downloaded and cached.
    """
    if name == "mini_dem":
        return mini_dem_path()
    if name == "sample_boundaries":
        return sample_boundaries_path()

    try:
        meta = _REMOTE_DATASETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {', '.join(available())}"
        ) from exc

    local_path = _local_dataset_path(meta)
    if local_path is not None:
        return local_path

    return _download_dataset(meta, cache_dir=cache_dir, base_url=base_url)


def fetch_dem(name: str, cache_dir: Optional[str] = None, base_url: Optional[str] = None) -> Path:
    """Fetch a sample DEM dataset and return its local path."""
    return _fetch_kind(name, expected_kind="dem", cache_dir=cache_dir, base_url=base_url)


def fetch_cityjson(name: str, cache_dir: Optional[str] = None, base_url: Optional[str] = None) -> Path:
    """Fetch a sample CityJSON dataset and return its local path."""
    return _fetch_kind(name, expected_kind="cityjson", cache_dir=cache_dir, base_url=base_url)


def fetch_copc(name: str, cache_dir: Optional[str] = None, base_url: Optional[str] = None) -> Path:
    """Fetch a sample point cloud dataset and return its local path."""
    return _fetch_kind(name, expected_kind="copc", cache_dir=cache_dir, base_url=base_url)
