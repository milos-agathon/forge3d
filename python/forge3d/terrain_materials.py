"""Geographically scaled terrain appearance recipes (C01).

Preparation is CPU raster work. Geometry and camera rendering remain owned by
MapScene's native terrain renderer. Protected presentation bakes cartographic
relief into physical layers and displays thematic colours without scene lighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np

__all__ = ["MaterialGrid", "MaterialTexture", "MaterialMask", "TerrainMaterialLayer",
           "TerrainMaterialRecipe", "PreparedTerrainMaterial", "TerrainMaterialStyle"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0 else result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _color(value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 255 for v in value):
        raise ValueError("albedo must be three sRGB uint8 channels")
    return tuple(int(v) for v in value)


def _linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def _display(rgb: np.ndarray) -> np.ndarray:
    from .legend import _linear_to_srgb
    return np.rint(np.clip(_linear_to_srgb(np.clip(rgb, 0, 1)), 0, 1) * 255).astype(np.uint8)


def _material_classes(values: np.ndarray, categorical: Any) -> np.ndarray:
    if values.dtype.kind not in "buif":
        raise ValueError("class IDs require a real numeric array; object, string and complex inputs are ambiguous")
    if np.issubdtype(values.dtype, np.integer) and any(int(v) != int(float(v)) for v in np.unique(values)):
        raise ValueError("class IDs exceed exact float64 representation in the native raster boundary")
    result = values.astype(np.float64)
    if categorical is not None and categorical.nodata is not None:
        result[result == categorical.nodata] = np.nan
    return result


def _channel_rgba(name: str, data: np.ndarray) -> np.ndarray:
    encoded = np.rint(np.clip((data + 1) / 2 if name == "normal" else data, 0, 1) * 255).astype(np.uint8)
    if encoded.ndim == 2:
        encoded = np.repeat(encoded[..., None], 3, axis=-1)
    return np.concatenate((encoded, np.full((*encoded.shape[:2], 1), 255, np.uint8)), axis=-1)


def _interpreted_legend(legend: Mapping[str, Any], presentation: str, binding: str) -> dict[str, Any]:
    interpretation = "Surface albedo; lit pixels differ" if presentation == "physical" else "Exact display swatch"
    return {**legend, "entries": [{**entry, "interpretation": interpretation, "binding": binding}
                                  for entry in legend.get("entries", ())]}


@dataclass(frozen=True)
class MaterialGrid:
    """North-up sample grid. Origin is the north-west *sample centre*.

    Horizontal coordinates, spacing and measured elevation are metres in the
    supplied projected CRS. Pixel-edge affine origins must be shifted half a
    pixel before construction. Rotated/geographic grids require preparation.
    """
    crs: str
    origin: tuple[float, float]
    spacing: tuple[float, float]
    units: str
    elevation_units: str

    def __post_init__(self) -> None:
        from .alignment import _require_rasterio
        if self.units != "m" or self.elevation_units != "m":
            raise ValueError("material grid requires horizontal and elevation units 'm'")
        crs = _require_rasterio().crs.CRS.from_user_input(self.crs)
        if not crs.is_projected or crs.linear_units_factor[1] != 1.0:
            raise ValueError("material grid CRS must be projected in metres")
        if len(self.origin) != 2 or len(self.spacing) != 2:
            raise ValueError("grid origin and spacing need two coordinates")
        object.__setattr__(self, "origin", tuple(_finite(v, "origin") for v in self.origin))
        spacing = tuple(_finite(v, "spacing") for v in self.spacing)
        if min(spacing) <= 0:
            raise ValueError("grid spacing must be positive")
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "crs", crs.to_string())

    def to_dict(self) -> dict[str, Any]:
        return {"crs": self.crs, "origin": list(self.origin), "spacing": list(self.spacing),
                "units": self.units, "elevation_units": self.elevation_units}


@dataclass(frozen=True)
class MaterialTexture:
    """Repeatable RGB/RGBA albedo; physical scale and channel role are explicit.

    Repeat phase is anchored in the recipe's projected CRS. Sampling uses
    periodic bilinear interpolation in linear light, including alpha coverage.
    Normal and roughness maps use linear data. OpenGL normals encode +X/+Y
    east and north, +Z outward. The native bridge flips G for its south bitangent.
    Roughness is a grayscale perceptual BRDF value, 0 smooth through 1 rough.
    """
    path: str
    repeat_m: tuple[float, float]
    color_space: str
    channel: str
    anchor: tuple[float, float] = (0.0, 0.0)
    _source_directory: str | None = field(default=None, repr=False, compare=False)
    normal_convention: str | None = None
    variation_seed: int | None = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("texture path must be nonempty")
        if self.channel == "albedo":
            if self.color_space != "srgb" or self.normal_convention is not None:
                raise ValueError("albedo requires sRGB and no normal convention")
        elif self.channel in {"normal", "roughness"}:
            if self.color_space != "linear":
                raise ValueError("normal and roughness channels require linear data")
            if self.channel == "normal" and self.normal_convention != "opengl":
                raise ValueError("normal channel requires explicit opengl tangent convention")
            if self.channel == "roughness" and self.normal_convention is not None:
                raise ValueError("roughness has no normal convention")
        else:
            raise ValueError("texture channel must be albedo, normal or roughness")
        if len(self.repeat_m) != 2 or len(self.anchor) != 2:
            raise ValueError("repeat_m and anchor need two coordinates")
        repeat = tuple(_finite(v, "repeat_m") for v in self.repeat_m)
        if min(repeat) <= 0:
            raise ValueError("repeat_m must be positive metres")
        object.__setattr__(self, "repeat_m", repeat)
        object.__setattr__(self, "anchor", tuple(_finite(v, "anchor") for v in self.anchor))
        if self.variation_seed is not None and (isinstance(self.variation_seed, bool) or not isinstance(self.variation_seed, int)):
            raise ValueError("variation_seed must be an integer or None for periodic baseline sampling")

    def to_dict(self) -> dict[str, Any]:
        values = {"path": self.path, "repeat_m": list(self.repeat_m), "color_space": self.color_space,
                  "channel": self.channel, "anchor": list(self.anchor)}
        if self.variation_seed is not None:
            values["variation_seed"] = self.variation_seed
        if self.normal_convention is not None:
            values["normal_convention"] = self.normal_convention
        return values

    @property
    def resolved_path(self) -> Path:
        path = Path(self.path)
        return Path(self._source_directory) / path if self._source_directory and not path.is_absolute() else path

    def sample(self, grid: MaterialGrid, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        x = grid.origin[0] + np.arange(shape[1]) * grid.spacing[0]
        y = grid.origin[1] - np.arange(shape[0]) * grid.spacing[1]
        xx, yy = np.meshgrid(x, y)
        return self.sample_points(xx, yy, footprint_m=grid.spacing)

    def sample_points(self, x: Any, y: Any, *, footprint_m: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("material texture coordinates must be finite projected metres")
        from PIL import Image
        with Image.open(self.resolved_path) as image:
            if self.channel == "roughness":
                if image.mode != "L":
                    raise ValueError("roughness texture must be an explicit grayscale L data image")
                data = np.asarray(image, dtype=np.float64)[..., None] / 255.0
                pixels = np.concatenate((data, np.ones_like(data)), axis=-1)
            else:
                if image.mode not in {"RGB", "RGBA"}:
                    raise ValueError("albedo/normal texture must be RGB or RGBA, not an ambiguous data image")
                pixels = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
                if self.channel == "normal":
                    if np.any(pixels[..., 3] != 1):
                        raise ValueError("normal data images must be opaque")
                    pixels[..., :3] = pixels[..., :3] * 2 - 1
                else:
                    pixels[..., :3] = _linear(pixels[..., :3]) * pixels[..., 3:4]
        h, w = pixels.shape[:2]
        world_u = (x - self.anchor[0]) / self.repeat_m[0]
        world_v = (self.anchor[1] - y) / self.repeat_m[1]
        integral = None
        if footprint_m is not None:
            if len(footprint_m) != 2 or any(_finite(v, "footprint_m") <= 0 for v in footprint_m):
                raise ValueError("footprint_m needs two positive metre dimensions")
            integral = np.pad(pixels.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
            du, dv = footprint_m[0] / self.repeat_m[0] * w, footprint_m[1] / self.repeat_m[1] * h

        def prefix(u, v):
            # Exact integral of piecewise-constant source texels, extended
            # periodically. Interpolating the summed-area table integrates
            # fractional edge texels as well as footprints larger than a tile.
            qx, qy = np.floor(u / w), np.floor(v / h)
            rx, ry = u - qx * w, v - qy * h
            def partial(a, b):
                ix, iy = np.floor(a).astype(np.int64), np.floor(b).astype(np.int64)
                jx, jy = np.minimum(ix + 1, w), np.minimum(iy + 1, h)
                fx, fy = (a - ix)[..., None], (b - iy)[..., None]
                return ((integral[iy, ix] * (1-fx) + integral[iy, jx] * fx) * (1-fy)
                        + (integral[jy, ix] * (1-fx) + integral[jy, jx] * fx) * fy)
            return (qx[..., None] * qy[..., None] * integral[h, w]
                    + qx[..., None] * partial(np.full_like(rx, w), ry)
                    + qy[..., None] * partial(rx, np.full_like(ry, h)) + partial(rx, ry))

        def sample_phase(u, v):
            u, v = np.remainder(u, 1) * w, np.remainder(v, 1) * h
            if integral is not None:
                return (prefix(u+du/2, v+dv/2) - prefix(u-du/2, v+dv/2)
                        - prefix(u+du/2, v-dv/2) + prefix(u-du/2, v-dv/2)) / (du*dv)
            u, v = u-.5, v-.5
            ix, iy = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            fx, fy = (u-ix)[..., None], (v-iy)[..., None]
            top = pixels[iy % h, ix % w]*(1-fx) + pixels[iy % h, (ix+1) % w]*fx
            bottom = pixels[(iy+1) % h, ix % w]*(1-fx) + pixels[(iy+1) % h, (ix+1) % w]*fx
            return top*(1-fy) + bottom*fy

        if np.all(pixels == pixels[0, 0]):
            sampled = np.broadcast_to(pixels[0, 0], (*x.shape, pixels.shape[2])).copy()
        elif self.variation_seed is None:
            sampled = sample_phase(world_u, world_v)
        else:
            # Four smoothly blended random phases remove the observed 16-cell
            # lattice. Cell identities depend only on world coordinates/seed;
            # crops and derivative halos cannot change phase or blend weights.
            cx, cy = np.floor(world_u), np.floor(world_v)
            fx, fy = world_u-cx, world_v-cy
            fx, fy = fx*fx*(3-2*fx), fy*fy*(3-2*fy)
            def phase(ix, iy, salt):
                word = (ix.astype(np.int64).astype(np.uint64)*np.uint64(0x9e3779b1)
                        + iy.astype(np.int64).astype(np.uint64)*np.uint64(0x85ebca77)
                        + np.uint64((self.variation_seed + salt) & 0xffffffff)) & np.uint64(0xffffffff)
                word = ((word ^ (word >> np.uint64(16)))*np.uint64(0x7feb352d)) & np.uint64(0xffffffff)
                word = ((word ^ (word >> np.uint64(15)))*np.uint64(0x846ca68b)) & np.uint64(0xffffffff)
                return (word ^ (word >> np.uint64(16))).astype(np.float64) / 2**32
            sampled = np.zeros((*x.shape, pixels.shape[2]), np.float64)
            squared_weights = np.zeros(x.shape, np.float64)
            for j in (0, 1):
                for i in (0, 1):
                    weight = (fx if i else 1-fx) * (fy if j else 1-fy)
                    squared_weights += weight*weight
                    sampled += sample_phase(world_u+phase(cx+i, cy+j, 0), world_v+phase(cx+i, cy+j, 1)) * weight[..., None]
            if self.channel == "albedo" and np.all(pixels[..., -1] == 1):
                # Blending independent phases reduces variance by sum(w²).
                # Restore opaque albedo contrast around the source's linear
                # mean. Coverage and nonopaque premultiplied colour retain
                # their ordinary area averages; normal vectors are normalized
                # separately rather than treated as colour distributions.
                mean = pixels[..., :-1].mean(axis=(0, 1))
                sampled[..., :-1] = np.clip(mean + (sampled[..., :-1]-mean)/np.sqrt(squared_weights)[..., None], 0, 1)
        sampled[..., -1] = np.clip(sampled[..., -1], 0, 1)
        if self.channel == "normal":
            length = np.linalg.norm(sampled[..., :3], axis=-1, keepdims=True)
            if np.any(length == 0) or np.any(sampled[..., 2] <= 0):
                raise ValueError("normal samples require a nonzero outward tangent direction")
            return sampled[..., :3] / length, sampled[..., 3]
        data = np.divide(sampled[..., :-1], sampled[..., -1:],
                         out=np.zeros_like(sampled[..., :-1]), where=sampled[..., -1:] > 0)
        return data, sampled[..., -1]


@dataclass(frozen=True)
class MaterialMask:
    """Composable range/class masks over measured geometry and aligned fields.

    Ranges include both endpoints. Aspect is compass downslope degrees; a range
    with lower > upper wraps north. Flat terrain has no aspect and never matches
    an aspect range. Field masks declare the field's units and never normalize
    to the current crop. all/any/invert combine masks without changing classes.
    """
    operation: str = "all"
    field: str | None = None
    units: str | None = None
    lower: float | None = None
    upper: float | None = None
    class_ids: tuple[int, ...] = ()
    children: tuple[MaterialMask, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in {"all", "any", "invert", "range", "ramp", "classes"}:
            raise ValueError("unknown material mask operation")
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "class_ids", tuple(self.class_ids))
        if self.operation in {"all", "any", "invert"}:
            if self.operation in {"any", "invert"} and (not self.children or self.operation == "invert" and len(self.children) != 1):
                raise ValueError("any needs children; invert needs exactly one child")
            if self.field is not None or self.units is not None or self.lower is not None or self.upper is not None or self.class_ids:
                raise ValueError("composite masks accept children only")
        elif self.children:
            raise ValueError("leaf masks cannot have children")
        elif self.operation == "classes":
            if not self.class_ids or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in self.class_ids):
                raise ValueError("class mask requires integer IDs")
            if self.field is not None or self.units is not None or self.lower is not None or self.upper is not None:
                raise ValueError("class masks accept class_ids only")
            object.__setattr__(self, "class_ids", tuple(int(v) for v in self.class_ids))
        else:
            if not self.field or not self.units or (self.lower is None and self.upper is None) or self.class_ids:
                raise ValueError("range mask requires field, units and at least one bound")
            for name in ("lower", "upper"):
                if getattr(self, name) is not None:
                    object.__setattr__(self, name, _finite(getattr(self, name), name))
            if self.operation == "range" and self.lower is not None and self.upper is not None and self.lower > self.upper and self.field != "aspect":
                raise ValueError("range lower must not exceed upper")
            if self.field == "aspect" and any(bound is not None and not 0 <= bound <= 360 for bound in (self.lower, self.upper)):
                raise ValueError("aspect range must be in [0, 360]")
            if self.operation == "ramp" and (self.lower is None or self.upper is None or self.lower >= self.upper):
                raise ValueError("ramp requires two increasing bounds in the field's units")

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "field": self.field, "units": self.units,
                "lower": self.lower, "upper": self.upper, "class_ids": list(self.class_ids),
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialMask:
        data = dict(data)
        data["children"] = tuple(cls.from_dict(c) for c in data.get("children", ()))
        return cls(**data)

    def evaluate(self, fields: Mapping[str, tuple[np.ndarray, str]], classes: np.ndarray | None, valid: np.ndarray) -> np.ndarray:
        if self.operation == "classes":
            if classes is None:
                raise ValueError("class mask needs an aligned categorical input")
            return np.isin(classes, self.class_ids) & valid
        if self.operation in {"range", "ramp"}:
            if self.field not in fields:
                raise ValueError(f"mask field {self.field!r} is missing")
            values, units = fields[self.field]
            if units != self.units:
                raise ValueError(f"mask field {self.field!r} has units {units!r}, expected {self.units!r}")
            if self.operation == "ramp":
                t = np.clip((values - self.lower) / (self.upper - self.lower), 0, 1)
                return np.where(valid & np.isfinite(values), t * t * (3 - 2 * t), 0)
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                match = (values >= self.lower) | (values <= self.upper)
            else:
                match = np.ones_like(valid)
                if self.lower is not None:
                    match &= values >= self.lower
                if self.upper is not None:
                    match &= values <= self.upper
            return match & np.isfinite(values) & valid
        matches = [c.evaluate(fields, classes, valid) for c in self.children]
        if self.operation == "invert":
            domain = self.children[0]._domain(fields, classes, valid)
            if matches[0].dtype == np.bool_:
                return ~matches[0] & domain
            return np.where(domain, 1 - matches[0], 0)
        result = valid.copy() if self.operation == "all" else np.zeros_like(valid)
        for match in matches:
            result = np.minimum(result, match) if self.operation == "all" else np.maximum(result, match)
        return result

    def _domain(self, fields: Any, classes: Any, valid: np.ndarray) -> np.ndarray:
        if self.operation in {"range", "ramp"}:
            return np.isfinite(fields[self.field][0]) & valid
        if self.operation == "classes":
            return np.isfinite(classes) & valid
        domains = [c._domain(fields, classes, valid) for c in self.children]
        result = valid.copy() if self.operation != "any" else np.zeros_like(valid)
        for domain in domains:
            result = result | domain if self.operation == "any" else result & domain
        return result


@dataclass(frozen=True)
class TerrainMaterialLayer:
    name: str
    albedo: tuple[int, int, int]
    mask: MaterialMask = field(default_factory=MaterialMask)
    texture: MaterialTexture | None = None
    opacity: float = 1.0
    normal: MaterialTexture | None = None
    roughness: float | None = None
    roughness_texture: MaterialTexture | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material layer name must be nonempty")
        object.__setattr__(self, "albedo", _color(self.albedo))
        object.__setattr__(self, "opacity", _unit(self.opacity, "opacity"))
        for name, channel in (("texture", "albedo"), ("normal", "normal"), ("roughness_texture", "roughness")):
            image = getattr(self, name)
            if image is not None and image.channel != channel:
                raise ValueError(f"{name} requires the {channel} channel")
        if self.roughness is not None:
            object.__setattr__(self, "roughness", _unit(self.roughness, "roughness"))
        if self.roughness is not None and self.roughness_texture is not None:
            raise ValueError("choose scalar roughness or a roughness texture")

    def to_dict(self) -> dict[str, Any]:
        values = {"name": self.name, "albedo": list(self.albedo), "mask": self.mask.to_dict(),
                  "texture": self.texture.to_dict() if self.texture else None, "opacity": self.opacity}
        for name in ("normal", "roughness_texture"):
            image = getattr(self, name)
            if image is not None:
                values[name] = image.to_dict()
        if self.roughness is not None:
            values["roughness"] = self.roughness
        return values

    def textures(self) -> tuple[MaterialTexture, ...]:
        return tuple(image for image in (self.texture, self.normal, self.roughness_texture) if image is not None)


@dataclass(frozen=True)
class TerrainMaterialRecipe:
    """Ordered surface layers with an optional authoritative categorical overlay.

    protected: physical lighting and declared cartographic relief affect surface layers; categorical
    swatches remain exact sRGB. physical: colours are surface albedo and may
    change under lighting; the legend describes source albedo, not lit pixels.
    No automatic exposure/default changes occur outside a bound material style.
    """
    layers: tuple[TerrainMaterialLayer, ...]
    presentation: str = "protected"
    relief_strength: float = 0.0
    relief_azimuth: float = 315.0
    relief_altitude: float = 45.0
    categorical: Any | None = None
    relief_mode: str = "single"

    def __post_init__(self) -> None:
        from .raster_style import CategoricalRasterStyle
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers or len({layer.name for layer in self.layers}) != len(self.layers):
            raise ValueError("recipe requires uniquely named material layers")
        if self.presentation not in {"protected", "physical"}:
            raise ValueError("presentation must be 'protected' or 'physical'")
        object.__setattr__(self, "relief_strength", _unit(self.relief_strength, "relief_strength"))
        for name in ("relief_azimuth", "relief_altitude"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0 <= self.relief_altitude <= 90:
            raise ValueError("relief_altitude must be in [0, 90]")
        if self.relief_mode not in {"single", "multidirectional"}:
            raise ValueError("relief_mode must be 'single' or 'multidirectional'")
        if self.categorical is not None and not isinstance(self.categorical, CategoricalRasterStyle):
            raise ValueError("categorical must be a CategoricalRasterStyle")
        if self.categorical is not None and any(rgba[3] != 255 for rgba in self.categorical.palette.values()):
            raise ValueError("authoritative categorical swatches must be opaque")
        if any(layer.roughness is not None or layer.roughness_texture is not None for layer in self.layers):
            if any(layer.roughness is None and layer.roughness_texture is None for layer in self.layers):
                raise ValueError("composed roughness requires a scalar or texture on every layer")
        if self.presentation == "physical" and self.relief_strength:
            raise ValueError("physical presentation uses renderer lighting; cartographic relief would double-shade")

    @property
    def has_surface_channels(self) -> bool:
        return any(layer.normal is not None or layer.roughness is not None or layer.roughness_texture is not None
                   for layer in self.layers)

    def resolved(self) -> TerrainMaterialRecipe:
        return replace(self, layers=tuple(replace(layer, **{
            name: replace(image, path=str(image.resolved_path.resolve()), _source_directory=None)
            for name in ("texture", "normal", "roughness_texture") if (image := getattr(layer, name)) is not None
        }) for layer in self.layers))

    @classmethod
    def land_cover(cls, categorical: Any, *, albedo: Sequence[int], texture: MaterialTexture | None = None,
                   relief_strength: float = 0.0) -> TerrainMaterialRecipe:
        """Approachable preset using the same composable recipe contract."""
        return cls((TerrainMaterialLayer("ground", tuple(albedo), texture=texture),),
                   categorical=categorical, relief_strength=relief_strength)

    @classmethod
    def rock_snow(cls, ground: TerrainMaterialLayer, rock: TerrainMaterialLayer, snow: TerrainMaterialLayer, *,
                  rock_slope_degrees: tuple[float, float], snow_elevation_m: tuple[float, float],
                  snow_slope_degrees: tuple[float, float], presentation: str = "protected",
                  categorical: Any = None, relief_strength: float = 0.0,
                  relief_mode: str = "single") -> TerrainMaterialRecipe:
        """Compose supplied surfaces with explicit, measured-unit transitions.

        Thresholds are author inputs; the preset does not guess regional snow
        observations. Existing layer masks remain part of each intersection.
        """
        rock_rule = MaterialMask("ramp", "slope", "degrees", *rock_slope_degrees)
        elevation_rule = MaterialMask("ramp", "elevation", "m", *snow_elevation_m)
        snow_rule = MaterialMask("invert", children=(MaterialMask("ramp", "slope", "degrees", *snow_slope_degrees),))
        return cls((ground,
                    replace(rock, mask=MaterialMask("all", children=(rock.mask, rock_rule))),
                    replace(snow, mask=MaterialMask("all", children=(snow.mask, elevation_rule, snow_rule)))),
                   presentation=presentation, categorical=categorical,
                   relief_strength=relief_strength, relief_mode=relief_mode)

    def with_layer(self, name: str, **overrides: Any) -> TerrainMaterialRecipe:
        if name not in {layer.name for layer in self.layers}:
            raise ValueError(f"unknown material layer {name!r}")
        return replace(self, layers=tuple(replace(layer, **overrides) if layer.name == name else layer for layer in self.layers))

    def to_dict(self) -> dict[str, Any]:
        values = {"schema_version": 1, "layers": [layer.to_dict() for layer in self.layers],
                "presentation": self.presentation, "relief_strength": self.relief_strength,
                "relief_azimuth": self.relief_azimuth, "relief_altitude": self.relief_altitude,
                "categorical": self.categorical.to_dict() if self.categorical else None}
        # Preserve the serialized default recipe; the new operation is opt-in.
        if self.relief_mode != "single":
            values["relief_mode"] = self.relief_mode
        return values

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerrainMaterialRecipe:
        from .raster_style import raster_style_from_dict
        values = dict(data)
        if values.pop("schema_version", None) != 1:
            raise ValueError("unsupported terrain material recipe schema_version")
        layers = []
        for item in values.pop("layers"):
            item = dict(item)
            item["mask"] = MaterialMask.from_dict(item["mask"])
            for name in ("texture", "normal", "roughness_texture"):
                if item.get(name) is not None:
                    # A recipe saved before stochastic sampling must retain its
                    # periodic semantics and canonical identity when replayed.
                    item[name] = MaterialTexture(**{"variation_seed": None, **item[name]})
            layers.append(TerrainMaterialLayer(**item))
        values["categorical"] = raster_style_from_dict(values.get("categorical"))
        return cls(tuple(layers), **values)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        values = self.to_dict()
        # A loaded relative reference belongs to its source directory. Saving
        # in another directory must retain the asset, rather than point at an
        # unrelated or missing destination-side file with the same name.
        for layer, serialized in zip(self.layers, values["layers"]):
            for name in ("texture", "normal", "roughness_texture"):
                texture = getattr(layer, name)
                if texture is not None and texture._source_directory and not Path(texture.path).is_absolute():
                    if Path(texture._source_directory).resolve() != destination.resolve().parent:
                        import os
                        try:
                            reference = os.path.relpath(texture.resolved_path, destination.resolve().parent)
                        except ValueError:
                            reference = str(texture.resolved_path.resolve())
                        serialized[name]["path"] = reference
        destination.write_text(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TerrainMaterialRecipe:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        recipe = cls.from_dict(data)
        return replace(recipe, layers=tuple(replace(layer, **{
            name: replace(image, _source_directory=str(path.resolve().parent))
            for name in ("texture", "normal", "roughness_texture") if (image := getattr(layer, name)) is not None
        }) for layer in recipe.layers))

    def _compose_layers(self, fields: Mapping[str, tuple[np.ndarray, str]], classes: Any,
                        valid: np.ndarray, texture_sampler: Any) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
        """The shared layer compositor for grids and irregular surface samples."""
        rgb = np.zeros((*valid.shape, 3), dtype=np.float64)
        coverage = np.zeros(valid.shape)
        masks = {}
        normals = np.zeros_like(rgb) if any(layer.normal for layer in self.layers) else None
        roughness = np.zeros_like(coverage) if any(layer.roughness is not None or layer.roughness_texture for layer in self.layers) else None
        for layer in self.layers:
            mask = layer.mask.evaluate(fields, classes, valid)
            masks[layer.name] = mask
            color = np.broadcast_to(_linear(np.asarray(layer.albedo) / 255), rgb.shape)
            alpha = mask.astype(np.float64) * layer.opacity
            if layer.texture:
                sampled, texture_alpha = texture_sampler(layer.texture)
                color = sampled * color
                alpha *= texture_alpha
            if normals is not None:
                direction = np.broadcast_to((0., 0., 1.), rgb.shape) if layer.normal is None else texture_sampler(layer.normal)[0]
                normals = direction * alpha[..., None] + normals * (1 - alpha[..., None])
            if roughness is not None:
                value = layer.roughness if layer.roughness_texture is None else texture_sampler(layer.roughness_texture)[0][..., 0]
                roughness = value * alpha + roughness * (1 - alpha)
            rgb = color * alpha[..., None] + rgb * (1 - alpha[..., None])
            coverage = alpha + coverage * (1 - alpha)
        rgb = np.divide(rgb, coverage[..., None], out=np.zeros_like(rgb), where=coverage[..., None] > 0)
        if normals is not None:
            length = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = np.divide(normals, length, out=np.broadcast_to((0., 0., 1.), rgb.shape).copy(), where=length > 0)
        if roughness is not None:
            roughness = np.divide(roughness, coverage, out=np.zeros_like(roughness), where=coverage > 0)
        return rgb, coverage, masks, normals, roughness

    def prepare(self, elevations: Any, grid: MaterialGrid, *, classes: Any = None,
                fields: Mapping[str, tuple[Any, str]] | None = None, valid_mask: Any = None,
                derivative_halo: Any = None) -> PreparedTerrainMaterial:
        heights = np.asarray(elevations, dtype=np.float64)
        if heights.ndim != 2 or min(heights.shape) < 2:
            raise ValueError("material preparation requires a 2D elevation grid with derivatives")
        valid = np.isfinite(heights)
        if valid_mask is not None:
            mask = np.asarray(valid_mask)
            if mask.shape != heights.shape or mask.dtype != np.bool_:
                raise ValueError("valid_mask must be a boolean array aligned with elevations")
            valid &= mask
        measured_heights = np.where(valid, heights, np.nan)
        derivative_source = measured_heights
        derivative_slice = (slice(None), slice(None))
        if derivative_halo is not None:
            derivative_source = np.asarray(derivative_halo, dtype=np.float64)
            if derivative_source.shape != (heights.shape[0] + 2, heights.shape[1] + 2) or not np.array_equal(
                    derivative_source[1:-1, 1:-1], heights, equal_nan=True):
                raise ValueError("derivative_halo must be the same elevation grid with one true neighbouring sample on every edge")
            derivative_source = derivative_source.copy()
            derivative_source[1:-1, 1:-1] = measured_heights
            derivative_slice = (slice(1, -1), slice(1, -1))
        dy, dx = np.gradient(derivative_source, *grid.spacing[::-1])
        dy, dx = dy[derivative_slice], dx[derivative_slice]
        slope = np.degrees(np.arctan(np.hypot(dx, dy)))
        aspect = np.remainder(np.degrees(np.arctan2(-dx, dy)), 360)
        aspect[np.hypot(dx, dy) == 0] = np.nan
        measured = {"elevation": (heights, "m"), "slope": (slope, "degrees"), "aspect": (aspect, "degrees")}
        for name, (values, units) in (fields or {}).items():
            if name in measured:
                raise ValueError(f"cannot override measured field {name!r}")
            values = np.asarray(values, dtype=np.float64)
            if values.shape != heights.shape or not isinstance(units, str) or not units:
                raise ValueError("data fields need declared units and the elevation grid shape")
            measured[name] = values, units
        class_values = None if classes is None else np.asarray(classes)
        if class_values is not None and class_values.shape != heights.shape:
            raise ValueError("categorical data must already be aligned to the material grid")
        if class_values is not None:
            class_values = _material_classes(class_values, self.categorical)
        rgb, coverage, masks, normals, roughness = self._compose_layers(measured, class_values, valid, lambda texture: texture.sample(grid, heights.shape))
        if self.relief_strength:
            # Reuse the repository's hillshade algorithm, but feed derivatives
            # in ground metres, independently of render exaggeration/crop size.
            from .widgets import _hillshade, _multidirectional_hillshade
            operation = _hillshade if self.relief_mode == "single" else _multidirectional_hillshade
            relief = operation(derivative_source, self.relief_azimuth, self.relief_altitude,
                               dx_m=grid.spacing[0], dy_m=grid.spacing[1])[derivative_slice]
            relief = np.nan_to_num(relief, nan=1.0)
            rgb *= (1 - self.relief_strength + self.relief_strength * relief)[..., None]
        rgba = np.concatenate((_display(np.nan_to_num(rgb)), np.rint(coverage[..., None] * 255).astype(np.uint8)), axis=-1)
        protected = np.zeros_like(valid)
        legend = None
        if self.categorical is not None:
            if class_values is None:
                raise ValueError("categorical presentation requires aligned class IDs")
            styled = self.categorical.apply(class_values, valid_mask=valid)
            protected = styled.valid_mask
            rgba[protected] = styled.rgba[protected]
            legend = styled.legend
        rgba[~valid] = 0
        return PreparedTerrainMaterial(rgba, valid, slope, aspect, masks, protected, legend, self, grid, normals, roughness)

    def bind(self, scene: Any, grid: MaterialGrid, *, classes: Any = None,
             fields: Mapping[str, tuple[Any, str]] | None = None, derivative_halo: Any = None) -> Any:
        """Return a SceneRecipe using the native terrain raster path.

        Inputs must already share this grid. Existing other raster sources are
        rejected by the existing single-colour-source gate. Terrain heights are
        kept as geometry; material preparation never uses render exaggeration.
        """
        from .map_scene import MapFurnitureLayer
        bound_recipe = self.resolved()
        style = TerrainMaterialStyle(bound_recipe, grid, classes, fields, derivative_halo)
        if scene.terrain.data is None:
            from .map_scene import _load_raw_heightmap
            elevations = _load_raw_heightmap(scene.terrain)
        else:
            elevations = scene.terrain.data
        prepared = style.prepare(elevations)
        if scene.terrain.crs != grid.crs:
            from .alignment import _require_rasterio
            if _require_rasterio().crs.CRS.from_user_input(scene.terrain.crs) != _require_rasterio().crs.CRS.from_user_input(grid.crs):
                raise ValueError("terrain and material grid CRS must agree")
        h, w = prepared.valid_mask.shape
        x, y = grid.origin
        sx, sy = grid.spacing
        bounds = (x - sx / 2, y - (h - .5) * sy, x + (w - .5) * sx, y + sy / 2)
        metadata = dict(scene.terrain.metadata or {})
        if metadata.get("bounds") is not None and tuple(metadata["bounds"]) != bounds:
            raise ValueError("material grid sample centres do not match terrain bounds")
        metadata.update(bounds=list(bounds), resolution=list(grid.spacing), width=w, height=h)
        terrain = replace(scene.terrain, crs=grid.crs, metadata=metadata, nodata_policy="preserve", style=style)
        furniture = scene.map_furniture or MapFurnitureLayer()
        if prepared.legend is not None:
            furniture = replace(furniture, legend=_interpreted_legend(prepared.legend, self.presentation, "terrain"))
        result = replace(scene, terrain=terrain, map_furniture=furniture)
        style.validate_scene(result)
        return result


@dataclass
class PreparedTerrainMaterial:
    rgba: np.ndarray
    valid_mask: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    masks: dict[str, np.ndarray]
    thematic_mask: np.ndarray
    legend: dict[str, Any] | None
    recipe: TerrainMaterialRecipe
    grid: MaterialGrid
    normals: np.ndarray | None = None
    roughness: np.ndarray | None = None

    def save_raster(self, path: str | Path) -> Path:
        """Save the prepared sRGB raster and its meaning/scale sidecar."""
        from .helpers.offscreen import save_png_deterministic
        path = Path(path)
        save_png_deterministic(path, self.rgba)
        sidecar = {
            "grid": self.grid.to_dict(), "recipe": self.recipe.to_dict(), "legend": self.legend,
            "display_transform": "sRGB", "presentation": self.recipe.presentation,
            "legend_interpretation": "exact display swatches" if self.recipe.presentation == "protected" else "surface albedo; lit pixels differ"
        }
        channels = {}
        for name, data in (("normal", self.normals), ("roughness", self.roughness)):
            if data is not None:
                image_path = path.with_suffix("." + name + ".png")
                save_png_deterministic(image_path, _channel_rgba(name, data))
                channels[name] = {"path": image_path.name, "color_space": "linear",
                                  "interpretation": "OpenGL tangent +X east, +G north, +Z outward; native bridge flips G to south" if name == "normal" else "perceptual BRDF roughness"}
        if channels:
            sidecar["surface_channels"] = channels
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(sidecar, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class TerrainMaterialStyle:
    """A recipe bound to one grid, serialized through the existing raster seam."""
    recipe: TerrainMaterialRecipe
    grid: MaterialGrid
    classes: Any = None
    fields: Mapping[str, tuple[Any, str]] | None = None
    derivative_halo: Any = None

    def __post_init__(self) -> None:
        from types import MappingProxyType
        if self.classes is not None:
            values = np.asarray(self.classes).copy()
            values.flags.writeable = False
            object.__setattr__(self, "classes", values)
        fields = {}
        for name, (values, units) in (self.fields or {}).items():
            values = np.asarray(values).copy()
            values.flags.writeable = False
            fields[name] = values, units
        object.__setattr__(self, "fields", MappingProxyType(fields))
        if self.derivative_halo is not None:
            halo = np.asarray(self.derivative_halo).copy()
            halo.flags.writeable = False
            object.__setattr__(self, "derivative_halo", halo)

    def prepare(self, source: Any) -> PreparedTerrainMaterial:
        return self._prepare(source)

    def _prepare(self, source: Any, valid_mask: Any = None) -> PreparedTerrainMaterial:
        from .raster_style import _single
        from ._material_preparation import prepared_material
        source = _single(source)
        return prepared_material(self, source, valid_mask, lambda: self.recipe.prepare(
            source, self.grid, classes=self.classes, fields=self.fields,
            valid_mask=valid_mask, derivative_halo=self.derivative_halo))

    def apply(self, source: Any, *, valid_mask: Any = None) -> Any:
        from .raster_style import RasterStyleResult, _single
        result = self._prepare(source, valid_mask)
        if np.any(result.rgba[result.valid_mask, 3] != 255):
            raise ValueError("native material presentation requires opaque coverage of every valid terrain sample; add a ground layer")
        return RasterStyleResult(result.rgba, result.valid_mask, result.legend or {
            "kind": "material", "entries": [], "presentation": self.recipe.presentation},
            heights=np.where(result.valid_mask, _single(source), np.nan).astype(np.float32))

    def validate_scene(self, scene: Any) -> None:
        from .map_scene import RasterOverlay, _is_3d_camera_mode, _mapscene_effective_camera_mode, _load_raw_heightmap
        rasters = [layer for layer in scene.layers if isinstance(layer, RasterOverlay)]
        if isinstance(scene.terrain.style, TerrainMaterialStyle):
            from .raster_style import CategoricalRasterStyle
            from .geospatial_materials import MaterialRasterStyle
            if any(not isinstance(layer.style, (CategoricalRasterStyle, MaterialRasterStyle)) for layer in rasters):
                raise ValueError("terrain materials combine with one aligned categorical raster colour source")
        elif not rasters:
            raise ValueError("material presentation requires the existing single raster colour source")
        if scene.target_crs not in {None, self.grid.crs} or scene.terrain.crs != self.grid.crs:
            raise ValueError("material presentation requires its prepared projected CRS")
        if _is_3d_camera_mode(_mapscene_effective_camera_mode(scene)):
            height, width = np.asarray(_load_raw_heightmap(scene.terrain)).shape
            if width * self.grid.spacing[0] != height * self.grid.spacing[1]:
                raise ValueError("native 3D material rendering uses a square ground footprint; prepare a square extent to preserve metre scale")
        for layer in scene.layers:
            if isinstance(layer, RasterOverlay) and layer.opacity != 1:
                raise ValueError("material raster presentation requires opacity=1 to preserve its palette")
        if self.recipe.presentation == "protected":
            settings = dict(scene.lighting.settings or {})
            for name in ("clouds", "screen_space", "water"):
                effect = settings.get(name) or (scene.terrain.metadata or {}).get(name)
                if effect and not isinstance(effect, Mapping):
                    raise ValueError(f"{name} settings must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        from .raster_style import raster_values_to_dict
        return {"kind": "terrain_material", "recipe": self.recipe.to_dict(), "grid": self.grid.to_dict(),
                "classes": raster_values_to_dict(self.classes), "derivative_halo": raster_values_to_dict(self.derivative_halo), "fields": {
                    name: {"values": raster_values_to_dict(values), "units": units}
                    for name, (values, units) in self.fields.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerrainMaterialStyle:
        from .raster_style import raster_values_from_dict
        return cls(TerrainMaterialRecipe.from_dict(data["recipe"]), MaterialGrid(**data["grid"]),
                   raster_values_from_dict(data.get("classes")), {
                       name: (raster_values_from_dict(item["values"]), item["units"])
                       for name, item in data.get("fields", {}).items()},
                   raster_values_from_dict(data.get("derivative_halo")))
