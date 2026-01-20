# python/forge3d/bundle.py
"""Scene bundle (.forge3d) for portable scene packages.

A bundle is a directory containing:
- manifest.json - version, name, checksums
- terrain/ - DEM and colormap data
- overlays/ - vectors and labels
- camera/ - camera bookmarks
- render/ - preset configuration
- assets/ - fonts, HDRI, etc.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Bundle format version
BUNDLE_VERSION = 1
BUNDLE_EXTENSION = "forge3d"


@dataclass
class TerrainMeta:
    """Terrain metadata in bundle manifest."""
    dem_path: str
    crs: Optional[str] = None
    domain: Optional[Tuple[float, float]] = None
    colormap: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"dem_path": self.dem_path}
        if self.crs:
            d["crs"] = self.crs
        if self.domain:
            d["domain"] = list(self.domain)
        if self.colormap:
            d["colormap"] = self.colormap
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerrainMeta":
        return cls(
            dem_path=data["dem_path"],
            crs=data.get("crs"),
            domain=tuple(data["domain"]) if data.get("domain") else None,
            colormap=data.get("colormap"),
        )


@dataclass
class CameraBookmark:
    """Camera bookmark for saved viewpoints."""
    name: str
    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_deg": self.fov_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraBookmark":
        return cls(
            name=data["name"],
            eye=tuple(data["eye"]),
            target=tuple(data["target"]),
            up=tuple(data.get("up", [0.0, 1.0, 0.0])),
            fov_deg=data.get("fov_deg", 45.0),
        )


@dataclass
class BundleManifest:
    """Bundle manifest containing metadata and checksums."""
    version: int
    name: str
    created_at: str
    description: Optional[str] = None
    checksums: Dict[str, str] = field(default_factory=dict)
    terrain: Optional[TerrainMeta] = None
    camera_bookmarks: List[CameraBookmark] = field(default_factory=list)
    preset: Optional[Dict[str, Any]] = None

    @classmethod
    def new(cls, name: str) -> "BundleManifest":
        """Create a new manifest with current timestamp."""
        return cls(
            version=BUNDLE_VERSION,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
        }
        if self.description:
            d["description"] = self.description
        if self.checksums:
            d["checksums"] = self.checksums
        if self.terrain:
            d["terrain"] = self.terrain.to_dict()
        if self.camera_bookmarks:
            d["camera_bookmarks"] = [b.to_dict() for b in self.camera_bookmarks]
        if self.preset:
            d["preset"] = self.preset
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleManifest":
        terrain = None
        if data.get("terrain"):
            terrain = TerrainMeta.from_dict(data["terrain"])
        bookmarks = [
            CameraBookmark.from_dict(b) for b in data.get("camera_bookmarks", [])
        ]
        return cls(
            version=data["version"],
            name=data["name"],
            created_at=data["created_at"],
            description=data.get("description"),
            checksums=data.get("checksums", {}),
            terrain=terrain,
            camera_bookmarks=bookmarks,
            preset=data.get("preset"),
        )

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BundleManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        manifest = cls.from_dict(data)
        if manifest.version > BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version {manifest.version} > supported version {BUNDLE_VERSION}"
            )
        return manifest


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(
    path: Union[str, Path],
    *,
    name: Optional[str] = None,
    dem_path: Optional[Union[str, Path]] = None,
    colormap_name: Optional[str] = None,
    domain: Optional[Tuple[float, float]] = None,
    crs: Optional[str] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    labels: Optional[List[Dict[str, Any]]] = None,
    preset: Optional[Dict[str, Any]] = None,
    camera_bookmarks: Optional[List[CameraBookmark]] = None,
    hdr_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save a scene bundle to disk.
    
    Args:
        path: Output bundle directory path (will be created)
        name: Bundle name (defaults to directory name)
        dem_path: Path to DEM file to include
        colormap_name: Colormap name or path
        domain: Elevation domain [min, max]
        crs: Coordinate reference system
        overlays: List of overlay dictionaries (vectors)
        labels: List of label dictionaries
        preset: Render preset dictionary
        camera_bookmarks: List of camera bookmarks
        hdr_path: Path to HDR environment map
    
    Returns:
        Path to created bundle directory
    """
    bundle_path = Path(path)
    if not bundle_path.suffix:
        bundle_path = bundle_path.with_suffix(f".{BUNDLE_EXTENSION}")
    
    # Create bundle directory structure
    bundle_path.mkdir(parents=True, exist_ok=True)
    (bundle_path / "terrain").mkdir(exist_ok=True)
    (bundle_path / "overlays").mkdir(exist_ok=True)
    (bundle_path / "camera").mkdir(exist_ok=True)
    (bundle_path / "render").mkdir(exist_ok=True)
    (bundle_path / "assets").mkdir(exist_ok=True)
    (bundle_path / "assets" / "hdri").mkdir(exist_ok=True)
    
    # Create manifest
    manifest = BundleManifest.new(name or bundle_path.stem)
    
    # Copy DEM if provided
    if dem_path:
        dem_src = Path(dem_path)
        dem_dst = bundle_path / "terrain" / dem_src.name
        shutil.copy2(dem_src, dem_dst)
        rel_path = f"terrain/{dem_src.name}"
        manifest.checksums[rel_path] = _compute_sha256(dem_dst)
        manifest.terrain = TerrainMeta(
            dem_path=rel_path,
            crs=crs,
            domain=domain,
            colormap=colormap_name,
        )
    
    # Save overlays
    if overlays:
        overlays_path = bundle_path / "overlays" / "vectors.geojson"
        with open(overlays_path, "w") as f:
            json.dump(overlays, f, indent=2)
        manifest.checksums["overlays/vectors.geojson"] = _compute_sha256(overlays_path)
    
    # Save labels
    if labels:
        labels_path = bundle_path / "overlays" / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        manifest.checksums["overlays/labels.json"] = _compute_sha256(labels_path)
    
    # Save preset
    if preset:
        preset_path = bundle_path / "render" / "preset.json"
        with open(preset_path, "w") as f:
            json.dump(preset, f, indent=2)
        manifest.checksums["render/preset.json"] = _compute_sha256(preset_path)
        manifest.preset = preset
    
    # Copy HDR if provided
    if hdr_path:
        hdr_src = Path(hdr_path)
        hdr_dst = bundle_path / "assets" / "hdri" / hdr_src.name
        shutil.copy2(hdr_src, hdr_dst)
        rel_path = f"assets/hdri/{hdr_src.name}"
        manifest.checksums[rel_path] = _compute_sha256(hdr_dst)
    
    # Add camera bookmarks
    if camera_bookmarks:
        manifest.camera_bookmarks = camera_bookmarks
        bookmarks_path = bundle_path / "camera" / "bookmarks.json"
        with open(bookmarks_path, "w") as f:
            json.dump([b.to_dict() for b in camera_bookmarks], f, indent=2)
        manifest.checksums["camera/bookmarks.json"] = _compute_sha256(bookmarks_path)
    
    # Save manifest
    manifest.save(bundle_path / "manifest.json")
    
    return bundle_path


@dataclass
class LoadedBundle:
    """Result of loading a bundle."""
    path: Path
    manifest: BundleManifest
    dem_path: Optional[Path] = None
    overlays: Optional[List[Dict[str, Any]]] = None
    labels: Optional[List[Dict[str, Any]]] = None
    preset: Optional[Dict[str, Any]] = None
    hdr_path: Optional[Path] = None


def load_bundle(path: Union[str, Path], verify_checksums: bool = True) -> LoadedBundle:
    """Load a scene bundle from disk.
    
    Args:
        path: Bundle directory path
        verify_checksums: Whether to verify file checksums
    
    Returns:
        LoadedBundle with all data loaded
    
    Raises:
        ValueError: If bundle is invalid or checksums don't match
    """
    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    
    if not manifest_path.exists():
        raise ValueError(f"Not a valid bundle: {bundle_path} (no manifest.json)")
    
    manifest = BundleManifest.load(manifest_path)
    
    # Verify checksums if requested
    if verify_checksums:
        for rel_path, expected_hash in manifest.checksums.items():
            file_path = bundle_path / rel_path
            if file_path.exists():
                actual_hash = _compute_sha256(file_path)
                if actual_hash != expected_hash:
                    raise ValueError(f"Checksum mismatch for {rel_path}")
    
    result = LoadedBundle(path=bundle_path, manifest=manifest)
    
    # Load DEM path
    if manifest.terrain:
        dem_file = bundle_path / manifest.terrain.dem_path
        if dem_file.exists():
            result.dem_path = dem_file
    
    # Load overlays
    overlays_path = bundle_path / "overlays" / "vectors.geojson"
    if overlays_path.exists():
        with open(overlays_path) as f:
            result.overlays = json.load(f)
    
    # Load labels
    labels_path = bundle_path / "overlays" / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            result.labels = json.load(f)
    
    # Load preset (from file or manifest)
    preset_path = bundle_path / "render" / "preset.json"
    if preset_path.exists():
        with open(preset_path) as f:
            result.preset = json.load(f)
    elif manifest.preset:
        result.preset = manifest.preset
    
    # Find HDR file
    hdri_dir = bundle_path / "assets" / "hdri"
    if hdri_dir.exists():
        for ext in (".hdr", ".exr"):
            for hdr_file in hdri_dir.glob(f"*{ext}"):
                result.hdr_path = hdr_file
                break
    
    return result


def is_bundle(path: Union[str, Path]) -> bool:
    """Check if a path is a valid bundle directory."""
    p = Path(path)
    return p.is_dir() and (p / "manifest.json").exists()
