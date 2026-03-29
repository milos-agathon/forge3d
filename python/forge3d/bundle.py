"""Scene bundle (.forge3d) for portable scene packages.

A bundle is a directory containing:
- manifest.json - version, name, checksums
- terrain/ - DEM and colormap data
- overlays/ - legacy raw vector and label payload mirrors
- camera/ - camera bookmarks
- render/ - base preset mirror
- scene/ - canonical TV16 scene review state
- assets/ - referenced HDRI and raster-overlay assets
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._license import _check_pro_access

BUNDLE_VERSION = 2
BUNDLE_EXTENSION = "forge3d"
_LABEL_KINDS = {"point", "line", "curved", "callout"}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def _deepcopy_json(value: Any) -> Any:
    return copy.deepcopy(_json_compatible(value))


def _copy_json_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    return dict(_deepcopy_json(dict(value)))


def _coerce_bookmarks(
    bookmarks: Iterable["CameraBookmark | Mapping[str, Any]"] | None,
) -> list["CameraBookmark"]:
    if bookmarks is None:
        return []
    result: list[CameraBookmark] = []
    for bookmark in bookmarks:
        if isinstance(bookmark, CameraBookmark):
            result.append(copy.deepcopy(bookmark))
        elif isinstance(bookmark, Mapping):
            result.append(CameraBookmark.from_dict(dict(bookmark)))
        else:
            raise TypeError("camera_bookmarks entries must be CameraBookmark or mapping")
    return result


def _coerce_vector_payloads(
    payloads: Iterable[Mapping[str, Any]] | None,
    *,
    name: str,
) -> list[dict[str, Any]]:
    if payloads is None:
        return []
    result: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{name} entries must be mappings")
        result.append(dict(_deepcopy_json(dict(payload))))
    return result


def _infer_label_kind(label: Mapping[str, Any]) -> str:
    if "kind" in label:
        kind = str(label["kind"])
        if kind not in _LABEL_KINDS:
            raise ValueError(f"Unsupported label kind: {kind}")
        return kind
    if "anchor" in label:
        return "callout"
    if "polyline" in label:
        return "line"
    if "world_pos" in label:
        return "point"
    raise ValueError("Label entries must include kind or enough shape fields to infer one")


def _coerce_labels(labels: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if labels is None:
        return []
    result: list[dict[str, Any]] = []
    for label in labels:
        if not isinstance(label, Mapping):
            raise TypeError("labels entries must be mappings")
        label_dict = dict(_deepcopy_json(dict(label)))
        label_dict["kind"] = _infer_label_kind(label_dict)
        result.append(label_dict)
    return result


def _collect_runtime_vector_overlays(overlays: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if overlays is None:
        return result
    for overlay in overlays:
        if not isinstance(overlay, Mapping):
            continue
        overlay_dict = dict(_deepcopy_json(dict(overlay)))
        if "vertices" in overlay_dict and "indices" in overlay_dict:
            result.append(overlay_dict)
    return result


def _canonical_bundle_path(path: str | Path) -> Path:
    bundle_path = Path(path)
    if not bundle_path.suffix:
        bundle_path = bundle_path.with_suffix(f".{BUNDLE_EXTENSION}")
    return bundle_path


def _bundle_rel_to_abs(bundle_path: Path, rel_path: str | None) -> str | None:
    if rel_path is None:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((bundle_path / candidate).resolve())


def _rewrite_preset_asset_paths_for_save(
    value: Any,
    *,
    bundle_path: Path,
    checksums: dict[str, str],
    asset_cache: dict[Path, str],
) -> Any:
    if isinstance(value, list):
        return [
            _rewrite_preset_asset_paths_for_save(
                item,
                bundle_path=bundle_path,
                checksums=checksums,
                asset_cache=asset_cache,
            )
            for item in value
        ]
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"hdr", "hdr_path"} and item is not None:
                rel_path = _copy_asset_into_bundle(
                    Path(item),
                    bundle_path=bundle_path,
                    target_dir=Path("assets") / "hdri",
                    checksums=checksums,
                    asset_cache=asset_cache,
                )
                rewritten[key] = rel_path
            else:
                rewritten[key] = _rewrite_preset_asset_paths_for_save(
                    item,
                    bundle_path=bundle_path,
                    checksums=checksums,
                    asset_cache=asset_cache,
                )
        return rewritten
    return _json_compatible(value)


def _rewrite_preset_asset_paths_for_load(value: Any, *, bundle_path: Path) -> Any:
    if isinstance(value, list):
        return [_rewrite_preset_asset_paths_for_load(item, bundle_path=bundle_path) for item in value]
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"hdr", "hdr_path"} and isinstance(item, str):
                rewritten[key] = _bundle_rel_to_abs(bundle_path, item)
            else:
                rewritten[key] = _rewrite_preset_asset_paths_for_load(item, bundle_path=bundle_path)
        return rewritten
    return value


def _copy_asset_into_bundle(
    src_path: Path,
    *,
    bundle_path: Path,
    target_dir: Path,
    checksums: dict[str, str],
    asset_cache: dict[Path, str],
) -> str:
    src = src_path.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Asset not found: {src}")
    if src in asset_cache:
        return asset_cache[src]

    destination_dir = bundle_path / target_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_name = src.name
    destination = destination_dir / destination_name
    counter = 1
    while destination.exists() and destination.resolve() != src:
        destination_name = f"{src.stem}_{counter}{src.suffix}"
        destination = destination_dir / destination_name
        counter += 1

    shutil.copy2(src, destination)
    rel_path = str((target_dir / destination.name).as_posix())
    checksums[rel_path] = _compute_sha256(destination)
    asset_cache[src] = rel_path
    return rel_path


@dataclass
class TerrainMeta:
    """Terrain metadata in bundle manifest."""

    dem_path: str
    crs: str | None = None
    domain: tuple[float, float] | None = None
    colormap: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"dem_path": self.dem_path}
        if self.crs:
            data["crs"] = self.crs
        if self.domain:
            data["domain"] = list(self.domain)
        if self.colormap:
            data["colormap"] = self.colormap
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerrainMeta":
        return cls(
            dem_path=str(data["dem_path"]),
            crs=data.get("crs"),
            domain=tuple(data["domain"]) if data.get("domain") else None,
            colormap=data.get("colormap"),
        )


@dataclass
class CameraBookmark:
    """Camera bookmark for saved viewpoints."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_deg": self.fov_deg,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraBookmark":
        return cls(
            name=str(data["name"]),
            eye=tuple(data["eye"]),
            target=tuple(data["target"]),
            up=tuple(data.get("up", [0.0, 1.0, 0.0])),
            fov_deg=float(data.get("fov_deg", 45.0)),
        )


@dataclass
class RasterOverlaySpec:
    """Runtime-ready raster overlay payload for review-state application."""

    name: str
    path: str
    extent: tuple[float, float, float, float] | None = None
    opacity: float | None = None
    z_order: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RasterOverlaySpec.name must be non-empty")
        self.path = str(self.path)
        if not self.path:
            raise ValueError("RasterOverlaySpec.path must be non-empty")
        if self.opacity is not None:
            self.opacity = float(self.opacity)
        if self.z_order is not None:
            self.z_order = int(self.z_order)
        if self.extent is not None:
            self.extent = tuple(float(value) for value in self.extent)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "path": self.path}
        if self.extent is not None:
            data["extent"] = list(self.extent)
        if self.opacity is not None:
            data["opacity"] = self.opacity
        if self.z_order is not None:
            data["z_order"] = self.z_order
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RasterOverlaySpec":
        return cls(
            name=str(data["name"]),
            path=str(data["path"]),
            extent=tuple(data["extent"]) if data.get("extent") else None,
            opacity=data.get("opacity"),
            z_order=data.get("z_order"),
        )


@dataclass
class SceneBaseState:
    """Base review-managed scene state shared by all variants."""

    preset: dict[str, Any] | None = None
    camera_bookmarks: list[CameraBookmark] = field(default_factory=list)
    raster_overlays: list[RasterOverlaySpec] = field(default_factory=list)
    vector_overlays: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    scatter_batches: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.preset = _copy_json_mapping(self.preset, name="SceneBaseState.preset")
        self.camera_bookmarks = _coerce_bookmarks(self.camera_bookmarks)
        self.raster_overlays = [
            overlay if isinstance(overlay, RasterOverlaySpec) else RasterOverlaySpec.from_dict(overlay)
            for overlay in self.raster_overlays
        ]
        self.vector_overlays = _coerce_vector_payloads(
            self.vector_overlays,
            name="SceneBaseState.vector_overlays",
        )
        self.labels = _coerce_labels(self.labels)
        self.scatter_batches = _coerce_vector_payloads(
            self.scatter_batches,
            name="SceneBaseState.scatter_batches",
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.preset is not None:
            data["preset"] = _deepcopy_json(self.preset)
        if self.camera_bookmarks:
            data["camera_bookmarks"] = [bookmark.to_dict() for bookmark in self.camera_bookmarks]
        if self.raster_overlays:
            data["raster_overlays"] = [overlay.to_dict() for overlay in self.raster_overlays]
        if self.vector_overlays:
            data["vector_overlays"] = _deepcopy_json(self.vector_overlays)
        if self.labels:
            data["labels"] = _deepcopy_json(self.labels)
        if self.scatter_batches:
            data["scatter_batches"] = _deepcopy_json(self.scatter_batches)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SceneBaseState":
        if not data:
            return cls()
        return cls(
            preset=data.get("preset"),
            camera_bookmarks=data.get("camera_bookmarks", []),
            raster_overlays=data.get("raster_overlays", []),
            vector_overlays=data.get("vector_overlays", []),
            labels=data.get("labels", []),
            scatter_batches=data.get("scatter_batches", []),
        )


@dataclass
class ReviewLayer:
    """Named review layer containing runtime-ready payloads."""

    id: str
    name: str | None = None
    description: str | None = None
    raster_overlays: list[RasterOverlaySpec] = field(default_factory=list)
    vector_overlays: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    scatter_batches: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ReviewLayer.id must be non-empty")
        self.raster_overlays = [
            overlay if isinstance(overlay, RasterOverlaySpec) else RasterOverlaySpec.from_dict(overlay)
            for overlay in self.raster_overlays
        ]
        self.vector_overlays = _coerce_vector_payloads(
            self.vector_overlays,
            name="ReviewLayer.vector_overlays",
        )
        self.labels = _coerce_labels(self.labels)
        self.scatter_batches = _coerce_vector_payloads(
            self.scatter_batches,
            name="ReviewLayer.scatter_batches",
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id}
        if self.name is not None:
            data["name"] = self.name
        if self.description is not None:
            data["description"] = self.description
        if self.raster_overlays:
            data["raster_overlays"] = [overlay.to_dict() for overlay in self.raster_overlays]
        if self.vector_overlays:
            data["vector_overlays"] = _deepcopy_json(self.vector_overlays)
        if self.labels:
            data["labels"] = _deepcopy_json(self.labels)
        if self.scatter_batches:
            data["scatter_batches"] = _deepcopy_json(self.scatter_batches)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewLayer":
        return cls(
            id=str(data["id"]),
            name=data.get("name"),
            description=data.get("description"),
            raster_overlays=data.get("raster_overlays", []),
            vector_overlays=data.get("vector_overlays", []),
            labels=data.get("labels", []),
            scatter_batches=data.get("scatter_batches", []),
        )


@dataclass
class SceneVariant:
    """Named scene variant composed from review layers plus an optional preset replacement."""

    id: str
    name: str | None = None
    description: str | None = None
    active_layer_ids: list[str] = field(default_factory=list)
    preset: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SceneVariant.id must be non-empty")
        self.active_layer_ids = [str(layer_id) for layer_id in self.active_layer_ids]
        self.preset = _copy_json_mapping(self.preset, name="SceneVariant.preset")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "active_layer_ids": list(self.active_layer_ids),
        }
        if self.name is not None:
            data["name"] = self.name
        if self.description is not None:
            data["description"] = self.description
        if self.preset is not None:
            data["preset"] = _deepcopy_json(self.preset)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneVariant":
        return cls(
            id=str(data["id"]),
            name=data.get("name"),
            description=data.get("description"),
            active_layer_ids=list(data.get("active_layer_ids", [])),
            preset=data.get("preset"),
        )


@dataclass
class SceneState:
    """Canonical TV16 scene review state."""

    base: SceneBaseState = field(default_factory=SceneBaseState)
    review_layers: list[ReviewLayer] = field(default_factory=list)
    variants: list[SceneVariant] = field(default_factory=list)
    active_variant_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, SceneBaseState):
            self.base = SceneBaseState.from_dict(self.base)
        self.review_layers = [
            layer if isinstance(layer, ReviewLayer) else ReviewLayer.from_dict(layer)
            for layer in self.review_layers
        ]
        self.variants = [
            variant if isinstance(variant, SceneVariant) else SceneVariant.from_dict(variant)
            for variant in self.variants
        ]
        if self.active_variant_id is not None:
            self.active_variant_id = str(self.active_variant_id)
        _validate_scene_state(self)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"base_state": self.base.to_dict()}
        if self.review_layers:
            data["review_layers"] = [layer.to_dict() for layer in self.review_layers]
        if self.variants:
            data["variants"] = [variant.to_dict() for variant in self.variants]
        if self.active_variant_id is not None:
            data["active_variant_id"] = self.active_variant_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SceneState":
        if not data:
            return cls()
        return cls(
            base=SceneBaseState.from_dict(data.get("base_state")),
            review_layers=data.get("review_layers", []),
            variants=data.get("variants", []),
            active_variant_id=data.get("active_variant_id"),
        )


def _validate_scene_state(scene_state: SceneState) -> None:
    layer_ids = [layer.id for layer in scene_state.review_layers]
    if len(layer_ids) != len(set(layer_ids)):
        duplicates = sorted({layer_id for layer_id in layer_ids if layer_ids.count(layer_id) > 1})
        raise ValueError(f"Duplicate review layer IDs: {duplicates}")

    variant_ids = [variant.id for variant in scene_state.variants]
    if len(variant_ids) != len(set(variant_ids)):
        duplicates = sorted({variant_id for variant_id in variant_ids if variant_ids.count(variant_id) > 1})
        raise ValueError(f"Duplicate scene variant IDs: {duplicates}")

    layer_id_set = set(layer_ids)
    for variant in scene_state.variants:
        missing = [layer_id for layer_id in variant.active_layer_ids if layer_id not in layer_id_set]
        if missing:
            raise ValueError(
                f"Variant '{variant.id}' references unknown review layer IDs: {missing}"
            )

    if scene_state.active_variant_id is not None and scene_state.active_variant_id not in set(variant_ids):
        raise ValueError(
            f"active_variant_id '{scene_state.active_variant_id}' does not match any variant"
        )


@dataclass
class BundleManifest:
    """Bundle manifest containing metadata and checksums."""

    version: int
    name: str
    created_at: str
    description: str | None = None
    checksums: dict[str, str] = field(default_factory=dict)
    terrain: TerrainMeta | None = None
    camera_bookmarks: list[CameraBookmark] = field(default_factory=list)
    preset: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.terrain is not None and not isinstance(self.terrain, TerrainMeta):
            self.terrain = TerrainMeta.from_dict(self.terrain)
        self.camera_bookmarks = _coerce_bookmarks(self.camera_bookmarks)
        self.preset = _copy_json_mapping(self.preset, name="BundleManifest.preset")

    @classmethod
    def new(cls, name: str) -> "BundleManifest":
        return cls(
            version=BUNDLE_VERSION,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
        }
        if self.description:
            data["description"] = self.description
        if self.checksums:
            data["checksums"] = dict(self.checksums)
        if self.terrain is not None:
            data["terrain"] = self.terrain.to_dict()
        if self.camera_bookmarks:
            data["camera_bookmarks"] = [bookmark.to_dict() for bookmark in self.camera_bookmarks]
        if self.preset is not None:
            data["preset"] = _deepcopy_json(self.preset)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BundleManifest":
        return cls(
            version=int(data["version"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            description=data.get("description"),
            checksums=dict(data.get("checksums", {})),
            terrain=TerrainMeta.from_dict(data["terrain"]) if data.get("terrain") else None,
            camera_bookmarks=data.get("camera_bookmarks", []),
            preset=data.get("preset"),
        )

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BundleManifest":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        manifest = cls.from_dict(data)
        if manifest.version > BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version {manifest.version} > supported version {BUNDLE_VERSION}"
            )
        return manifest


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_scene_state(scene_state: SceneState | Mapping[str, Any] | None) -> SceneState:
    if scene_state is None:
        return SceneState()
    if isinstance(scene_state, SceneState):
        return SceneState.from_dict(scene_state.to_dict())
    if isinstance(scene_state, Mapping):
        return SceneState.from_dict(scene_state)
    raise TypeError("scene_state must be a SceneState, mapping, or None")


def _rewrite_raster_overlays_for_save(
    overlays: Iterable[RasterOverlaySpec],
    *,
    bundle_path: Path,
    checksums: dict[str, str],
    asset_cache: dict[Path, str],
) -> list[RasterOverlaySpec]:
    rewritten: list[RasterOverlaySpec] = []
    for overlay in overlays:
        rel_path = _copy_asset_into_bundle(
            Path(overlay.path),
            bundle_path=bundle_path,
            target_dir=Path("assets") / "overlays",
            checksums=checksums,
            asset_cache=asset_cache,
        )
        rewritten.append(
            RasterOverlaySpec(
                name=overlay.name,
                path=rel_path,
                extent=overlay.extent,
                opacity=overlay.opacity,
                z_order=overlay.z_order,
            )
        )
    return rewritten


def _rewrite_raster_overlays_for_load(
    overlays: Iterable[RasterOverlaySpec],
    *,
    bundle_path: Path,
) -> list[RasterOverlaySpec]:
    rewritten: list[RasterOverlaySpec] = []
    for overlay in overlays:
        rewritten.append(
            RasterOverlaySpec(
                name=overlay.name,
                path=_bundle_rel_to_abs(bundle_path, overlay.path) or overlay.path,
                extent=overlay.extent,
                opacity=overlay.opacity,
                z_order=overlay.z_order,
            )
        )
    return rewritten


def _prepare_scene_state_for_save(
    scene_state: SceneState,
    *,
    bundle_path: Path,
    checksums: dict[str, str],
    asset_cache: dict[Path, str],
) -> SceneState:
    base = SceneBaseState(
        preset=_rewrite_preset_asset_paths_for_save(
            scene_state.base.preset,
            bundle_path=bundle_path,
            checksums=checksums,
            asset_cache=asset_cache,
        )
        if scene_state.base.preset is not None
        else None,
        camera_bookmarks=scene_state.base.camera_bookmarks,
        raster_overlays=_rewrite_raster_overlays_for_save(
            scene_state.base.raster_overlays,
            bundle_path=bundle_path,
            checksums=checksums,
            asset_cache=asset_cache,
        ),
        vector_overlays=scene_state.base.vector_overlays,
        labels=scene_state.base.labels,
        scatter_batches=scene_state.base.scatter_batches,
    )

    review_layers = [
        ReviewLayer(
            id=layer.id,
            name=layer.name,
            description=layer.description,
            raster_overlays=_rewrite_raster_overlays_for_save(
                layer.raster_overlays,
                bundle_path=bundle_path,
                checksums=checksums,
                asset_cache=asset_cache,
            ),
            vector_overlays=layer.vector_overlays,
            labels=layer.labels,
            scatter_batches=layer.scatter_batches,
        )
        for layer in scene_state.review_layers
    ]
    return SceneState(
        base=base,
        review_layers=review_layers,
        variants=scene_state.variants,
        active_variant_id=scene_state.active_variant_id,
    )


def _resolve_scene_state_after_load(scene_state: SceneState, *, bundle_path: Path) -> SceneState:
    base = SceneBaseState(
        preset=_rewrite_preset_asset_paths_for_load(scene_state.base.preset, bundle_path=bundle_path)
        if scene_state.base.preset is not None
        else None,
        camera_bookmarks=scene_state.base.camera_bookmarks,
        raster_overlays=_rewrite_raster_overlays_for_load(
            scene_state.base.raster_overlays,
            bundle_path=bundle_path,
        ),
        vector_overlays=scene_state.base.vector_overlays,
        labels=scene_state.base.labels,
        scatter_batches=scene_state.base.scatter_batches,
    )
    review_layers = [
        ReviewLayer(
            id=layer.id,
            name=layer.name,
            description=layer.description,
            raster_overlays=_rewrite_raster_overlays_for_load(
                layer.raster_overlays,
                bundle_path=bundle_path,
            ),
            vector_overlays=layer.vector_overlays,
            labels=layer.labels,
            scatter_batches=layer.scatter_batches,
        )
        for layer in scene_state.review_layers
    ]
    return SceneState(
        base=base,
        review_layers=review_layers,
        variants=scene_state.variants,
        active_variant_id=scene_state.active_variant_id,
    )


def _scene_state_with_legacy_kwargs(
    scene_state: SceneState | Mapping[str, Any] | None,
    *,
    overlays: Iterable[Mapping[str, Any]] | None,
    labels: Iterable[Mapping[str, Any]] | None,
    preset: Mapping[str, Any] | None,
    camera_bookmarks: Iterable[CameraBookmark | Mapping[str, Any]] | None,
    hdr_path: str | Path | None,
) -> SceneState:
    state = _clone_scene_state(scene_state)
    base = SceneBaseState.from_dict(state.base.to_dict())

    if preset is not None:
        base.preset = _copy_json_mapping(preset, name="preset")
    if camera_bookmarks is not None:
        base.camera_bookmarks = _coerce_bookmarks(camera_bookmarks)
    if overlays is not None:
        base.vector_overlays = _collect_runtime_vector_overlays(overlays)
    if labels is not None:
        base.labels = _coerce_labels(labels)
    if hdr_path is not None:
        preset_dict = base.preset or {}
        preset_dict = dict(_deepcopy_json(preset_dict))
        preset_dict["hdr_path"] = str(Path(hdr_path))
        base.preset = preset_dict

    return SceneState(
        base=base,
        review_layers=state.review_layers,
        variants=state.variants,
        active_variant_id=state.active_variant_id,
    )


def _find_first_hdri_asset(bundle_path: Path) -> Path | None:
    hdri_dir = bundle_path / "assets" / "hdri"
    if not hdri_dir.exists():
        return None
    for ext in (".hdr", ".exr"):
        for hdr_file in sorted(hdri_dir.glob(f"*{ext}")):
            return hdr_file.resolve()
    return None


def _resolve_hdr_path_from_preset(preset: Mapping[str, Any] | None) -> Path | None:
    if not preset:
        return None
    for key in ("hdr", "hdr_path"):
        candidate = preset.get(key)
        if isinstance(candidate, str):
            resolved = Path(candidate)
            if resolved.exists():
                return resolved.resolve()
    return None


def _synthesize_v1_scene_state(
    manifest: BundleManifest,
    *,
    overlays: list[dict[str, Any]] | None,
    labels: list[dict[str, Any]] | None,
    preset: dict[str, Any] | None,
    hdr_path: Path | None,
) -> SceneState:
    base_preset = _copy_json_mapping(preset, name="preset")
    if hdr_path is not None:
        if base_preset is None:
            base_preset = {}
        if "hdr" not in base_preset and "hdr_path" not in base_preset:
            base_preset["hdr_path"] = str(hdr_path)
    return SceneState(
        base=SceneBaseState(
            preset=base_preset,
            camera_bookmarks=manifest.camera_bookmarks,
            vector_overlays=_collect_runtime_vector_overlays(overlays),
            labels=labels or [],
        )
    )


@dataclass
class LoadedBundle:
    """Result of loading a bundle."""

    path: Path
    manifest: BundleManifest
    dem_path: Path | None = None
    overlays: list[dict[str, Any]] | None = None
    labels: list[dict[str, Any]] | None = None
    preset: dict[str, Any] | None = None
    hdr_path: Path | None = None
    camera_bookmarks: list[CameraBookmark] = field(default_factory=list)
    scene_state: SceneState = field(default_factory=SceneState)
    _layer_visibility_overrides: dict[str, bool] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not isinstance(self.manifest, BundleManifest):
            self.manifest = BundleManifest.from_dict(self.manifest)
        if not isinstance(self.scene_state, SceneState):
            self.scene_state = SceneState.from_dict(self.scene_state)
        self.overlays = None if self.overlays is None else _coerce_vector_payloads(self.overlays, name="LoadedBundle.overlays")
        self.labels = None if self.labels is None else _coerce_labels(self.labels)
        self.preset = _copy_json_mapping(self.preset, name="LoadedBundle.preset")
        if self.hdr_path is not None:
            self.hdr_path = Path(self.hdr_path)
        self.camera_bookmarks = _coerce_bookmarks(
            self.camera_bookmarks or self.scene_state.base.camera_bookmarks or self.manifest.camera_bookmarks
        )

    def list_variants(self) -> list[SceneVariant]:
        return [SceneVariant.from_dict(variant.to_dict()) for variant in self.scene_state.variants]

    def list_review_layers(self) -> list[ReviewLayer]:
        return [ReviewLayer.from_dict(layer.to_dict()) for layer in self.scene_state.review_layers]

    def get_active_variant_id(self) -> str | None:
        return self.scene_state.active_variant_id

    def _variant_by_id(self, variant_id: str) -> SceneVariant:
        for variant in self.scene_state.variants:
            if variant.id == variant_id:
                return variant
        raise KeyError(f"Unknown scene variant: {variant_id}")

    def _layer_by_id(self, layer_id: str) -> ReviewLayer:
        for layer in self.scene_state.review_layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(f"Unknown review layer: {layer_id}")

    def _default_visible_layer_ids(self) -> set[str]:
        if self.scene_state.active_variant_id is None:
            return set()
        return set(self._variant_by_id(self.scene_state.active_variant_id).active_layer_ids)

    def _visible_layer_ids(self) -> set[str]:
        visible = set(self._default_visible_layer_ids())
        for layer_id, override in self._layer_visibility_overrides.items():
            if override:
                visible.add(layer_id)
            else:
                visible.discard(layer_id)
        return visible

    def apply_variant(self, variant_id: str) -> None:
        self._variant_by_id(variant_id)
        self.scene_state.active_variant_id = variant_id
        self._layer_visibility_overrides.clear()

    def set_review_layer_visible(self, layer_id: str, visible: bool) -> None:
        self._layer_by_id(layer_id)
        default_visible = layer_id in self._default_visible_layer_ids()
        if bool(visible) == default_visible:
            self._layer_visibility_overrides.pop(layer_id, None)
        else:
            self._layer_visibility_overrides[layer_id] = bool(visible)

    def effective_scene_state(self) -> SceneBaseState:
        result = SceneBaseState.from_dict(self.scene_state.base.to_dict())
        visible_layer_ids = self._visible_layer_ids()
        visible_layers = [
            layer for layer in self.scene_state.review_layers if layer.id in visible_layer_ids
        ]
        for layer in visible_layers:
            result.raster_overlays.extend(
                RasterOverlaySpec.from_dict(overlay.to_dict()) for overlay in layer.raster_overlays
            )
            result.vector_overlays.extend(_deepcopy_json(layer.vector_overlays))
            result.labels.extend(_deepcopy_json(layer.labels))
            result.scatter_batches.extend(_deepcopy_json(layer.scatter_batches))

        if self.scene_state.active_variant_id is not None:
            variant = self._variant_by_id(self.scene_state.active_variant_id)
            if variant.preset is not None:
                result.preset = _copy_json_mapping(variant.preset, name="SceneVariant.preset")
        return result


def save_bundle(
    path: str | Path,
    *,
    name: str | None = None,
    dem_path: str | Path | None = None,
    colormap_name: str | None = None,
    domain: tuple[float, float] | None = None,
    crs: str | None = None,
    overlays: list[dict[str, Any]] | None = None,
    labels: list[dict[str, Any]] | None = None,
    preset: dict[str, Any] | None = None,
    camera_bookmarks: list[CameraBookmark] | None = None,
    hdr_path: str | Path | None = None,
    scene_state: SceneState | Mapping[str, Any] | None = None,
) -> Path:
    """[Pro] Save a scene bundle to disk."""

    _check_pro_access("Scene bundle save")
    bundle_path = _canonical_bundle_path(path)

    bundle_path.mkdir(parents=True, exist_ok=True)
    for rel_dir in (
        "terrain",
        "overlays",
        "camera",
        "render",
        "scene",
        "assets",
        "assets/hdri",
        "assets/overlays",
    ):
        (bundle_path / rel_dir).mkdir(parents=True, exist_ok=True)

    manifest = BundleManifest.new(name or bundle_path.stem)

    if dem_path is not None:
        dem_src = Path(dem_path).expanduser().resolve()
        if not dem_src.exists():
            raise FileNotFoundError(f"DEM not found: {dem_src}")
        dem_dst = bundle_path / "terrain" / dem_src.name
        shutil.copy2(dem_src, dem_dst)
        rel_path = str(Path("terrain") / dem_src.name).replace("\\", "/")
        manifest.checksums[rel_path] = _compute_sha256(dem_dst)
        manifest.terrain = TerrainMeta(
            dem_path=rel_path,
            crs=crs,
            domain=domain,
            colormap=colormap_name,
        )

    state = _scene_state_with_legacy_kwargs(
        scene_state,
        overlays=overlays,
        labels=labels,
        preset=preset,
        camera_bookmarks=camera_bookmarks,
        hdr_path=hdr_path,
    )
    state = _prepare_scene_state_for_save(
        state,
        bundle_path=bundle_path,
        checksums=manifest.checksums,
        asset_cache={},
    )

    if overlays is not None:
        overlays_path = bundle_path / "overlays" / "vectors.geojson"
        with overlays_path.open("w", encoding="utf-8") as handle:
            json.dump(_deepcopy_json(overlays), handle, indent=2)
        manifest.checksums["overlays/vectors.geojson"] = _compute_sha256(overlays_path)

    if labels is not None:
        labels_path = bundle_path / "overlays" / "labels.json"
        with labels_path.open("w", encoding="utf-8") as handle:
            json.dump(_deepcopy_json(labels), handle, indent=2)
        manifest.checksums["overlays/labels.json"] = _compute_sha256(labels_path)

    if state.base.camera_bookmarks:
        bookmarks_path = bundle_path / "camera" / "bookmarks.json"
        with bookmarks_path.open("w", encoding="utf-8") as handle:
            json.dump([bookmark.to_dict() for bookmark in state.base.camera_bookmarks], handle, indent=2)
        manifest.checksums["camera/bookmarks.json"] = _compute_sha256(bookmarks_path)
    manifest.camera_bookmarks = state.base.camera_bookmarks

    if state.base.preset is not None:
        preset_path = bundle_path / "render" / "preset.json"
        with preset_path.open("w", encoding="utf-8") as handle:
            json.dump(_deepcopy_json(state.base.preset), handle, indent=2)
        manifest.checksums["render/preset.json"] = _compute_sha256(preset_path)
    manifest.preset = state.base.preset

    scene_path = bundle_path / "scene" / "state.json"
    with scene_path.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    manifest.checksums["scene/state.json"] = _compute_sha256(scene_path)

    manifest.save(bundle_path / "manifest.json")
    return bundle_path


def load_bundle(path: str | Path, verify_checksums: bool = True) -> LoadedBundle:
    """[Pro] Load a scene bundle from disk."""

    _check_pro_access("Scene bundle load")
    bundle_path = Path(path)
    if not bundle_path.exists() and not bundle_path.suffix:
        bundle_path = _canonical_bundle_path(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Not a valid bundle: {bundle_path} (no manifest.json)")

    manifest = BundleManifest.load(manifest_path)
    if verify_checksums:
        for rel_path, expected_hash in manifest.checksums.items():
            file_path = bundle_path / rel_path
            if file_path.exists() and _compute_sha256(file_path) != expected_hash:
                raise ValueError(f"Checksum mismatch for {rel_path}")

    dem_file: Path | None = None
    if manifest.terrain is not None:
        candidate = bundle_path / manifest.terrain.dem_path
        if candidate.exists():
            dem_file = candidate.resolve()

    overlays: list[dict[str, Any]] | None = None
    overlays_path = bundle_path / "overlays" / "vectors.geojson"
    if overlays_path.exists():
        with overlays_path.open(encoding="utf-8") as handle:
            overlays = json.load(handle)

    labels: list[dict[str, Any]] | None = None
    labels_path = bundle_path / "overlays" / "labels.json"
    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as handle:
            labels = json.load(handle)

    bookmark_entries: list[dict[str, Any]] | None = None
    bookmarks_path = bundle_path / "camera" / "bookmarks.json"
    if bookmarks_path.exists():
        with bookmarks_path.open(encoding="utf-8") as handle:
            bookmark_entries = json.load(handle)

    preset: dict[str, Any] | None = None
    preset_path = bundle_path / "render" / "preset.json"
    if preset_path.exists():
        with preset_path.open(encoding="utf-8") as handle:
            preset = json.load(handle)
    elif manifest.preset is not None:
        preset = _deepcopy_json(manifest.preset)

    scene_path = bundle_path / "scene" / "state.json"
    if scene_path.exists():
        with scene_path.open(encoding="utf-8") as handle:
            scene_state = SceneState.from_dict(json.load(handle))
        scene_state = _resolve_scene_state_after_load(scene_state, bundle_path=bundle_path)
        if bookmark_entries is not None:
            scene_state.base.camera_bookmarks = _coerce_bookmarks(bookmark_entries)
        if preset is not None:
            scene_state.base.preset = _rewrite_preset_asset_paths_for_load(preset, bundle_path=bundle_path)
    else:
        scene_state = _synthesize_v1_scene_state(
            manifest,
            overlays=overlays,
            labels=labels,
            preset=preset,
            hdr_path=_find_first_hdri_asset(bundle_path),
        )

    base_preset = _copy_json_mapping(scene_state.base.preset, name="SceneBaseState.preset")
    hdr_resolved = _resolve_hdr_path_from_preset(base_preset)
    if hdr_resolved is None:
        hdr_resolved = _find_first_hdri_asset(bundle_path)

    return LoadedBundle(
        path=bundle_path.resolve(),
        manifest=manifest,
        dem_path=dem_file,
        overlays=overlays if overlays is not None else (_deepcopy_json(scene_state.base.vector_overlays) or None),
        labels=labels if labels is not None else (_deepcopy_json(scene_state.base.labels) or None),
        preset=base_preset,
        hdr_path=hdr_resolved,
        camera_bookmarks=scene_state.base.camera_bookmarks,
        scene_state=scene_state,
    )


def is_bundle(path: str | Path) -> bool:
    """Check if a path is a valid bundle directory."""

    candidate = Path(path)
    if not candidate.exists() and not candidate.suffix:
        candidate = _canonical_bundle_path(candidate)
    return candidate.is_dir() and (candidate / "manifest.json").exists()
