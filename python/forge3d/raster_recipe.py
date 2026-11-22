"""A fixed normalized-difference recipe and its live native GPU residency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

import numpy as np

from ._canonical_json import canonical_json_bytes
from ._native import get_native_module
from .alignment import _affine_from_metadata, _require_rasterio
from .observation_time import ObservationRecord
from .preparation import PreparationGrid, _require_render_grid, _sha
from .raster_registration import geotiff_registration
from .raster_style import CategoricalRasterStyle

if TYPE_CHECKING:
    from .map_scene import SceneRecipe

__all__ = ["RasterRecipe", "RasterRecipeSource"]


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value, error_context="Raster recipe")


def _grid(dataset: Any) -> PreparationGrid:
    if not dataset.crs:
        raise ValueError("Raster recipe inputs require a CRS and registered grid")
    return PreparationGrid(
        str(dataset.crs), dataset.width, dataset.height, dataset.transform.to_gdal(),
        registration=geotiff_registration(dataset),
    )


def _f32(value: float, name: str) -> float:
    original = value.item() if isinstance(value, np.generic) else value
    if isinstance(original, bool) or not isinstance(original, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if (
        abs(original) > float(np.finfo(np.float32).max)
        or not math.isfinite(original)
        or float(np.float32(original)) != original
    ):
        raise ValueError(f"{name} must be exactly representable as finite f32")
    return 0.0 if original == 0 else float(original)


@dataclass(frozen=True)
class RasterRecipeSource:
    """One band of a local GeoTIFF on the declared recipe grid."""

    path: str | Path
    source_id: str
    sha256: str
    observation: ObservationRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Raster recipe source_id must be nonempty")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError(
                "Raster recipe source sha256 must be a lowercase SHA-256 digest"
            )
        if self.observation is not None and not isinstance(
            self.observation, ObservationRecord
        ):
            raise TypeError(
                "Raster recipe source observation must be an ObservationRecord"
            )
        object.__setattr__(self, "path", Path(self.path).resolve())

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        source_id: str,
        *,
        observation: ObservationRecord | None = None,
    ) -> RasterRecipeSource:
        path = Path(path).resolve()
        return cls(path, source_id, _sha(path), observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_id": self.source_id,
            "sha256": self.sha256,
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RasterRecipeSource:
        record = data["observation"]
        observation = (
            None
            if record is None
            else ObservationRecord(
                record["record_id"],
                record["declared_observed_at"],
                record["declared_valid_from"],
                record["declared_valid_until"],
                record["payload"],
            )
        )
        return cls(data["path"], data["source_id"], data["sha256"], observation)

    def _load(
        self, grid: PreparationGrid, *, mask: bool
    ) -> tuple[np.ndarray, float | None]:
        if _sha(self.path) != self.sha256:
            raise ValueError(f"Source content changed: {self.source_id}")
        rasterio = _require_rasterio()
        with (
            rasterio.Env(
                GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
            ),
            rasterio.open(self.path) as ds,
        ):
            if (
                ds.driver != "GTiff"
                or ds.count != 1
                or ds.dtypes[0] != ("uint8" if mask else "float32")
            ):
                raise ValueError(
                    f"{self.source_id} needs a single-band {'uint8' if mask else 'float32'} GeoTIFF"
                )
            if not grid.same_as(_grid(ds)):
                raise ValueError(f"Grid mismatch for {self.source_id}")
            if ds.scales != (1.0,) or ds.offsets != (0.0,):
                raise ValueError(
                    f"Decode scale/offset before supplying raster {self.source_id}"
                )
            values = ds.read(1)
            # All coverage comes from the pinned validity file and declared nodata.
            # Reject extra masks whose bytes could live in external GDAL sidecars.
            flags = set(ds.mask_flag_enums[0])
            mask_flags = rasterio.enums.MaskFlags
            if not flags <= {mask_flags.all_valid, mask_flags.nodata}:
                raise ValueError(
                    f"Prepare a separate validity mask for {self.source_id}"
                )
            if mask:
                if ds.nodata not in (None, 0) or np.any(values > 1):
                    raise ValueError(
                        "Validity mask must contain only 0 and 1 with optional nodata 0"
                    )
                return values, None
            nodata = ds.nodata
            return values, (
                _f32(nodata, "Nodata")
                if nodata is not None and math.isfinite(nodata)
                else None
            )


class RasterRecipe:
    """Registered normalized difference, fixed bins, and one native terrain overlay.

    Class 0 preserves invalid coverage. Valid bins start at 1, with unbounded
    tails and interior closure matching `classify_raster`'s `right` argument.
    Construction and JSON decoding do not allocate GPU resources.
    """

    def __init__(
        self,
        a: RasterRecipeSource,
        b: RasterRecipeSource,
        mask: RasterRecipeSource,
        grid: PreparationGrid,
        bins: Sequence[float],
        palette: Sequence[Sequence[int]],
        *,
        right: bool = False,
        quantity: str = "Normalized difference",
    ) -> None:
        if not isinstance(grid, PreparationGrid):
            raise TypeError("Raster recipe grid must be a PreparationGrid")
        _require_render_grid(grid)
        if (
            not isinstance(right, bool)
            or not isinstance(quantity, str)
            or not quantity.strip()
        ):
            raise ValueError(
                "Raster recipe requires a boolean right flag and a quantity label"
            )
        self._grid = grid
        self._quantity = quantity
        self._right = right
        self._sources = self._check_sources(a, b, mask)
        self._bins, self._style = self._classification(bins, palette)
        self._resident = None
        self._inputs_current = False
        self._classified_as = None
        self._lock = RLock()

    @property
    def grid(self) -> PreparationGrid:
        return self._grid

    @staticmethod
    def _check_sources(*sources: RasterRecipeSource) -> tuple[RasterRecipeSource, ...]:
        if not all(isinstance(src, RasterRecipeSource) for src in sources):
            raise TypeError("Raster recipe inputs must be RasterRecipeSource records")
        return sources

    def _classification(
        self, bins: Sequence[float], palette: Sequence[Sequence[int]]
    ) -> tuple:
        edges = tuple(_f32(edge, "Threshold") for edge in bins)
        if not edges or any(lo >= hi for lo, hi in zip(edges, edges[1:])):
            raise ValueError(
                "Raster recipe thresholds must be strictly increasing and nonempty"
            )
        if len(palette) != len(edges) + 1:
            raise ValueError("Raster recipe needs one colour per class")
        edges_as_text = [repr(edge) for edge in edges]
        if self._right:
            labels = [
                f"≤ {edges_as_text[0]}",
                *(f"({lo}, {hi}]" for lo, hi in zip(edges_as_text, edges_as_text[1:])),
                f"> {edges_as_text[-1]}",
            ]
        else:
            labels = [
                f"< {edges_as_text[0]}",
                *(f"[{lo}, {hi})" for lo, hi in zip(edges_as_text, edges_as_text[1:])),
                f"≥ {edges_as_text[-1]}",
            ]
        style = CategoricalRasterStyle(
            dict(enumerate(palette, 1)), dict(enumerate(labels, 1)), nodata=0, units="1"
        )
        return edges, style

    def legend(self) -> dict[str, Any]:
        with self._lock:
            return self._style.legend() | {
                "quantity": self._quantity,
                "bins": list(self._bins),
                "right_inclusive": self._right,
                "sources": [source.source_id for source in self._sources],
                "observed_at": [
                    (
                        source.observation.to_dict()["observed_at"]
                        if source.observation
                        else None
                    )
                    for source in self._sources[:2]
                ],
            }

    def set_bins(self, bins: Sequence[float]) -> None:
        with self._lock:
            self._bins, self._style = self._classification(
                bins, list(self._style.palette.values())
            )

    def replace_inputs(
        self, a: RasterRecipeSource, b: RasterRecipeSource, mask: RasterRecipeSource
    ) -> None:
        sources = self._check_sources(a, b, mask)
        with self._lock:
            if sources != self._sources:
                self._sources = sources
                self._inputs_current = False
                self._classified_as = None

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "operator": "normalized_difference_f32",
                "grid": self.grid.to_dict(),
                "quantity": self._quantity,
                "inputs": [source.to_dict() for source in self._sources],
                "bins": list(self._bins),
                "palette": [list(color) for color in self._style.palette.values()],
                "right": self._right,
            }

    def to_json(self) -> str:
        return _canonical(self.to_dict()).decode("ascii")

    @classmethod
    def from_json(cls, data: str) -> RasterRecipe:
        row = json.loads(data)
        if row["schema_version"] != 1 or row["operator"] != "normalized_difference_f32":
            raise ValueError("Unsupported raster recipe version or operator")
        a, b, mask = (RasterRecipeSource.from_dict(source) for source in row["inputs"])
        return cls(
            a,
            b,
            mask,
            PreparationGrid(**row["grid"]),
            row["bins"],
            row["palette"],
            right=row["right"],
            quantity=row["quantity"],
        )

    @property
    def identity(self) -> str:
        """ANAMNESIS-style path-free, canonical content identity."""
        row = self.to_dict()
        for source in row["inputs"]:
            source.pop("path")
        return sha256(_canonical(row)).hexdigest()

    def _execute(self) -> Any:
        if self._resident is None:
            constructor = getattr(get_native_module(), "_ResidentRasterRecipe", None)
            if constructor is None:
                raise RuntimeError(
                    "Resident raster recipe bindings unavailable; rebuild the native extension"
                )
            self._resident = constructor()
        if not self._inputs_current:
            a, b, mask = self._sources
            left, nodata_a = a._load(self.grid, mask=False)
            right, nodata_b = b._load(self.grid, mask=False)
            valid, _ = mask._load(self.grid, mask=True)
            self._resident.replace(left, right, valid, (nodata_a, nodata_b))
            self._inputs_current = True
        identity = self.identity
        if self._classified_as != identity:
            colors = list(self._style.palette.values())
            self._resident.classify(
                list(zip(self._bins, colors)),
                colors[-1],
                self._right,
                bytes.fromhex(identity),
            )
            self._classified_as = identity
        return self._resident

    def sample(self, row: int, column: int) -> tuple[float | None, int]:
        """Read one cell's binary32 result and class ID."""
        with self._lock:
            return self._execute().sample(row, column)

    def counts(self) -> dict[int, int]:
        """Read one count per class, including invalid cells under ID 0."""
        with self._lock:
            return dict(enumerate(self._execute().counts()))

    def export_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Read all values and class IDs for an explicit final raster export."""
        with self._lock:
            result = (
                self._execute()
                .export_cells()
                .reshape(self.grid.height, self.grid.width, 2)
            )
            return result[..., 0].view(np.float32), result[..., 1]

    def metrics(self) -> dict[str, Any]:
        """Return observed completed dispatch counts and analysis transfer bytes."""
        with self._lock:
            if self._resident is None:
                return {}
            return self._resident.metrics()

    def close(self) -> None:
        """Release resources. The declaration can start a new GPU session."""
        with self._lock:
            self._resident = None
            self._inputs_current = False
            self._classified_as = None

    @contextmanager
    def _for_render(self, recipe: SceneRecipe):
        from .map_scene import (
            RasterOverlay,
            _is_geotiff_path,
            _terrain_alignment_grid,
            _terrain_scene_diagonal,
        )

        with self._lock:
            terrain = recipe.terrain
            if terrain.style is not None or any(
                isinstance(layer, RasterOverlay) for layer in recipe.layers
            ):
                raise ValueError(
                    "Raster recipe requires the single native thematic terrain slot"
                )
            gt = self.grid.geotransform
            span = self.grid.width * gt[1]
            if not math.isfinite(span) or span != -self.grid.height * gt[5]:
                raise ValueError(
                    "Native thematic terrain uses one horizontal span; prepare a square extent"
                )
            for crs in (terrain.crs, recipe.target_crs):
                if crs and not self.grid.same_as(
                    PreparationGrid(crs, self.grid.width, self.grid.height, gt, self.grid.registration)
                ):
                    raise ValueError("Raster recipe CRS mismatch with terrain")
            alignment = _terrain_alignment_grid(terrain)
            if alignment is None:
                raise ValueError("Raster recipe terrain requires a registered grid")
            terrain_grid = PreparationGrid(
                alignment["crs"],
                alignment["width"],
                alignment["height"],
                _affine_from_metadata(alignment["transform"]).to_gdal(),
                registration=alignment["registration"],
            )
            if not self.grid.same_as(terrain_grid):
                raise ValueError("Raster recipe grid mismatch with terrain")
            if _terrain_scene_diagonal(terrain) != span:
                raise ValueError(
                    "Raster recipe grid conflicts with terrain resolution metadata"
                )
            if (
                terrain.data is None
                and terrain.path
                and _is_geotiff_path(Path(terrain.path))
            ):
                with _require_rasterio().open(terrain.path) as dataset:
                    if not self.grid.same_as(_grid(dataset)):
                        raise ValueError(
                            "Raster recipe grid mismatch with terrain GeoTIFF"
                        )
            resident = self._execute()
            yield resident.overlay(), self.legend(), self.identity
