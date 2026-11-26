"""One C01 authoring language bound to existing terrain/raster/vector/buildings.

Bindings declare their actual capabilities. Scalar feature paint does not pretend
to be a continuously textured mesh, and unsupported presentation modes raise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields as dataclass_fields
from typing import Any, Mapping
import copy

import numpy as np

from .terrain_materials import (
    MaterialGrid, TerrainMaterialLayer,
    TerrainMaterialRecipe, TerrainMaterialStyle, _display, _material_classes,
)

MaterialLayer = TerrainMaterialLayer
__all__ = ["GeospatialMaterialRecipe", "MaterialLayer", "MaterialEvaluation", "MaterialRasterStyle"]


@dataclass
class MaterialEvaluation:
    rgba: np.ndarray
    masks: dict[str, np.ndarray]
    legend: dict[str, Any] | None
    crs: str
    normals: np.ndarray | None = None
    roughness: np.ndarray | None = None


class GeospatialMaterialRecipe(TerrainMaterialRecipe):
    """Shared layers, typed masks, parameters and categorical meaning for all families."""

    def evaluate_samples(self, coordinates_m: Any, *, crs: str, units: str,
                         fields: Mapping[str, tuple[Any, str]] | None = None,
                         classes: Any = None, normals_enu: Any = None,
                         footprint_m: tuple[float, float] | None = None) -> MaterialEvaluation:
        context = MaterialGrid(crs, (0, 0), (1, 1), units, "m")
        coords = np.asarray(coordinates_m, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] not in {2, 3} or not np.isfinite(coords).all():
            raise ValueError("material samples require finite Nx2/Nx3 projected metre coordinates")
        if self.relief_strength:
            raise ValueError("cartographic relief needs a measured elevation grid; use prepare() or prepare_raster()")
        count = len(coords)
        values = {}
        if coords.shape[1] == 3:
            values["elevation"] = coords[:, 2], "m"
        if normals_enu is not None:
            normals = np.asarray(normals_enu, dtype=np.float64)
            if normals.shape != (count, 3) or not np.isfinite(normals).all() or np.any(np.max(np.abs(normals), axis=1) == 0):
                raise ValueError("surface normals must be finite nonzero ENU Nx3 data")
            # Normals are directions: their arbitrary scale must neither
            # overflow the squared length nor underflow into a zero vector.
            normals = normals / np.max(np.abs(normals), axis=1)[:, None]
            normals = normals / np.linalg.norm(normals, axis=1)[:, None]
            values["slope"] = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1))), "degrees"
            aspect = np.remainder(np.degrees(np.arctan2(normals[:, 0], normals[:, 1])), 360)
            aspect[np.linalg.norm(normals[:, :2], axis=1) == 0] = np.nan
            values["aspect"] = aspect, "degrees"
        for name, (data, field_units) in (fields or {}).items():
            if name in values or not isinstance(field_units, str) or not field_units:
                raise ValueError("sample fields need explicit units and cannot override measured geometry")
            data = np.asarray(data, dtype=np.float64)
            if data.shape != (count,):
                raise ValueError("sample field must contain one value per coordinate")
            values[name] = data, field_units
        class_data = None if classes is None else np.asarray(classes)
        if class_data is not None:
            class_data = _material_classes(class_data, self.categorical)
        if class_data is not None and class_data.shape != (count,):
            raise ValueError("class IDs must contain one value per sample")
        valid = np.ones(count, dtype=bool)
        rgb, coverage, masks, normals, roughness = self._compose_layers(values, class_data, valid,
            lambda texture: texture.sample_points(coords[:, 0], coords[:, 1], footprint_m=footprint_m))
        rgba = np.concatenate((_display(rgb), np.rint(coverage[:, None] * 255).astype(np.uint8)), axis=-1)
        legend = None
        if self.categorical is not None:
            if class_data is None:
                raise ValueError("categorical material needs class IDs")
            styled = self.categorical.apply(class_data[None, :])
            selected = styled.valid_mask[0]
            rgba[selected] = styled.rgba[0, selected]
            legend = styled.legend
        return MaterialEvaluation(rgba, masks, legend, context.crs, normals, roughness)

    def prepare_raster(self, values: Any, grid: MaterialGrid, *, units: str,
                       classes: Any = None, fields: Mapping[str, tuple[Any, str]] | None = None,
                       elevations: Any = None, derivative_halo: Any = None) -> Any:
        from .raster_style import RasterStyleResult, _single
        source = _single(values).astype(np.float64)
        if not isinstance(units, str) or not units:
            raise ValueError("raster values require declared units")
        if classes is not None and np.asarray(classes).shape != source.shape:
            raise ValueError("raster class IDs must use the value grid shape")
        if any(np.asarray(data).shape != source.shape for data, _ in (fields or {}).values()):
            raise ValueError("raster data fields must use the value grid shape")
        if elevations is not None:
            supplied = dict(fields or {})
            if "value" in supplied:
                raise ValueError("value field is owned by the raster input")
            supplied["value"] = source, units
            prepared = self.prepare(elevations, grid, classes=classes, fields=supplied,
                                    valid_mask=np.isfinite(source), derivative_halo=derivative_halo)
            return RasterStyleResult(prepared.rgba, prepared.valid_mask, prepared.legend or {"kind": "material", "entries": []})
        if derivative_halo is not None:
            raise ValueError("raster derivative halo requires a measured elevation grid")
        yy, xx = np.indices(source.shape)
        coordinates = np.column_stack((grid.origin[0] + xx.ravel() * grid.spacing[0],
                                      grid.origin[1] - yy.ravel() * grid.spacing[1]))
        supplied = {name: (np.asarray(data).ravel(), field_units) for name, (data, field_units) in (fields or {}).items()}
        if "value" in supplied:
            raise ValueError("value field is owned by the raster input")
        supplied["value"] = source.ravel(), units
        result = self.evaluate_samples(coordinates, crs=grid.crs, units=grid.units, fields=supplied,
                                       classes=None if classes is None else np.asarray(classes).ravel(), footprint_m=grid.spacing)
        rgba = result.rgba.reshape((*source.shape, 4))
        valid = np.isfinite(source)
        rgba[~valid] = 0
        return RasterStyleResult(rgba, valid, result.legend or {"kind": "material", "entries": []})

    def bind_raster(self, scene: Any, values: Any, grid: MaterialGrid, *, units: str,
                    classes: Any = None, fields: Mapping[str, tuple[Any, str]] | None = None,
                    elevations: Any = None, derivative_halo: Any = None,
                    layer_id: str = "material-raster") -> Any:
        from .map_scene import RasterOverlay, MapFurnitureLayer, _load_raw_heightmap
        shape = np.asarray(_load_raw_heightmap(scene.terrain)).shape
        if np.asarray(values).shape != shape:
            raise ValueError("raster material values must already match the scene's terrain grid")
        bound_recipe = self.resolved()
        style = MaterialRasterStyle(bound_recipe, grid, classes, fields, derivative_halo,
                                    value_units=units, elevations=elevations)
        result = style.apply(values)
        h, w = shape
        x, y = grid.origin
        sx, sy = grid.spacing
        bounds = (x - sx / 2, y - (h - .5) * sy, x + (w - .5) * sx, y + sy / 2)
        metadata = dict(scene.terrain.metadata or {})
        if metadata.get("bounds") is not None and tuple(metadata["bounds"]) != bounds:
            raise ValueError("raster material grid must agree with terrain bounds")
        metadata.update(bounds=list(bounds), resolution=list(grid.spacing), width=w, height=h)
        if scene.terrain.data is not None:
            from .raster_style import raster_values_to_dict
            # Opt-in C01 geometry snapshot. Bare legacy TerrainSource bundles
            # keep their descriptor-only behavior; this delivery needs replay.
            metadata["forge3d_material_geometry"] = {"schema_version": 1,
                                                     "values": raster_values_to_dict(scene.terrain.data)}
        from .alignment import _require_rasterio
        if _require_rasterio().crs.CRS.from_user_input(scene.terrain.crs) != _require_rasterio().crs.CRS.from_user_input(grid.crs):
            raise ValueError("raster material CRS must agree with the terrain CRS")
        terrain = replace(scene.terrain, crs=grid.crs, metadata=metadata)
        layer = RasterOverlay(layer_id, data=np.asarray(values), crs=grid.crs, style=style,
            metadata={"material_recipe": bound_recipe.to_dict(), "material_legend": result.legend,
                      "material_presentation": self.presentation})
        furniture = scene.map_furniture or MapFurnitureLayer()
        if result.legend.get("entries"):
            from .terrain_materials import _interpreted_legend
            furniture = replace(furniture, legend=_interpreted_legend(result.legend, self.presentation, layer_id))
        bound = replace(scene, terrain=terrain, layers=(*scene.layers, layer), map_furniture=furniture)
        style.validate_scene(bound)
        return bound

    def _feature_binding(self, layer: Any, coordinates_m: Any, *, units: str,
                         classes: Any, fields: Any, presentation: str) -> tuple[list, MaterialEvaluation]:
        if self.has_surface_channels:
            raise ValueError("feature paint/scatter does not consume normal or roughness channels; use the terrain/raster surface binding")
        if any(item.texture is not None for item in self.layers):
            raise ValueError("this feature binding supports scalar albedo; repeating surface textures require a qualified mesh/UV path")
        features = copy.deepcopy(list(layer.features or ()))
        if not features:
            raise ValueError("material binding requires provided features; load the source before authoring")
        if len(coordinates_m) != len(features):
            raise ValueError("feature material anchors must correspond to every provided feature")
        recipe = replace(self, presentation=presentation)
        crs = getattr(layer, "crs", None) or (layer.metadata or {}).get("crs")
        if not crs:
            raise ValueError("feature materials require their projected source CRS")
        result = recipe.evaluate_samples(coordinates_m, crs=crs, units=units, classes=classes, fields=fields)
        if np.any(result.rgba[:, 3] != 255):
            raise ValueError("feature material needs opaque albedo coverage")
        for feature, rgba in zip(features, result.rgba):
            properties = dict(feature.get("properties") or {})
            properties["forge3d:material:color"] = "#" + "".join(f"{v:02x}" for v in rgba)
            properties["forge3d:material"] = {"rgba": rgba.tolist(), "presentation": presentation}
            feature["properties"] = properties
        return features, result

    def bind_vector(self, layer: Any, coordinates_m: Any, *, units: str,
                    classes: Any = None, fields: Any = None) -> Any:
        features, result = self._feature_binding(layer, coordinates_m, units=units,
            classes=classes, fields=fields, presentation=self.presentation)
        style = copy.deepcopy(dict(layer.style or {}))
        rows = list(style.get("layers") or [])
        explicit_outline = (layer.width_world is not None or layer.width_px is not None or
                            any(row.get("type") == "line" for row in rows))
        for kind, color_key in (("fill", "fill-color"), ("line", "line-color"), ("circle", "circle-color")):
            matching = [row for row in rows if row.get("type") == kind]
            if not matching:
                row = {"id": f"{layer.layer_id}-material-{kind}", "type": kind, "paint": {}}
                rows.append(row)
                matching = [row]
            for row in matching:
                if self.presentation == "protected" and row.get("paint", {}).get(f"{kind}-opacity", 1) != 1:
                    raise ValueError("protected feature materials require paint opacity=1")
                row.setdefault("paint", {})[color_key] = ["get", "forge3d:material:color"]
        style["layers"] = rows
        metadata = dict(layer.metadata or {})
        metadata["material_recipe"] = self.to_dict()
        metadata["material_legend"] = result.legend
        metadata["material_presentation"] = ("protected feature paint" if self.presentation == "protected"
                                             else "physical terrain albedo")
        metadata["material_polygon_outline"] = explicit_outline
        return replace(layer, features=features, style=style, metadata=metadata)

    def bind_buildings(self, layer: Any, coordinates_m: Any, *, units: str,
                       classes: Any = None, fields: Any = None,
                       presentation: str = "physical") -> Any:
        features, result = self._feature_binding(layer, coordinates_m, units=units,
            classes=classes, fields=fields, presentation=presentation)
        metadata = dict(layer.metadata or {})
        metadata.update(material_recipe=replace(self, presentation=presentation).to_dict(), material_legend=result.legend,
                        material_presentation=("physical scalar albedo; lit pixels differ" if presentation == "physical"
                                               else "protected building swatches"))
        return replace(layer, features=features, metadata=metadata, material_status="scalar_material_recipe")

    def attach(self, scene: Any, *layers: Any) -> Any:
        """Replace/append bound layers and keep the publication legend connected."""
        from .map_scene import MapFurnitureLayer
        replacements = {layer.layer_id: layer for layer in layers}
        if len(replacements) != len(layers):
            raise ValueError("material layer IDs must be unique")
        combined = tuple(replacements.pop(layer.layer_id, layer) for layer in scene.layers) + tuple(replacements.values())
        from .terrain_materials import _interpreted_legend
        legends = []
        if isinstance(scene.terrain.style, TerrainMaterialStyle) and scene.terrain.style.recipe.categorical is not None:
            legends.append(_interpreted_legend(scene.terrain.style.recipe.categorical.legend(),
                scene.terrain.style.recipe.presentation, "terrain"))
        for layer in combined:
            metadata = getattr(layer, "metadata", None) or {}
            if metadata.get("material_legend"):
                presentation = "physical" if "physical" in str(metadata.get("material_presentation", "")) else "protected"
                legends.append(_interpreted_legend(metadata["material_legend"], presentation, str(layer.layer_id)))
        entries = []
        seen = set()
        for legend in legends:
            for entry in legend.get("entries", ()):
                key = (entry.get("class_id"), entry["label"], tuple(entry["rgba"]), legend.get("units", ""),
                       entry["interpretation"], entry["binding"])
                if key not in seen:
                    entries.append(entry)
                    seen.add(key)
        furniture = scene.map_furniture or MapFurnitureLayer()
        if entries:
            furniture = replace(furniture, legend={"entries": entries,
                "interpretation": "Per-binding meaning"})
        return replace(scene, layers=combined, map_furniture=furniture)


@dataclass(frozen=True)
class MaterialRasterStyle:
    recipe: TerrainMaterialRecipe
    grid: MaterialGrid
    classes: Any = None
    fields: Mapping[str, tuple[Any, str]] | None = None
    derivative_halo: Any = None
    value_units: str = ""
    elevations: Any = None

    def __post_init__(self) -> None:
        inputs = TerrainMaterialStyle(self.recipe, self.grid, self.classes, self.fields, self.derivative_halo)
        object.__setattr__(self, "_inputs", inputs)
        for name in ("classes", "fields", "derivative_halo"):
            object.__setattr__(self, name, getattr(inputs, name))
        if not isinstance(self.value_units, str) or not self.value_units:
            raise ValueError("material raster needs value units")
        if self.elevations is not None:
            data = np.asarray(self.elevations).copy()
            data.flags.writeable = False
            object.__setattr__(self, "elevations", data)

    def apply(self, source: Any, *, valid_mask: Any = None) -> Any:
        recipe = GeospatialMaterialRecipe(**{item.name: getattr(self.recipe, item.name)
                                            for item in dataclass_fields(TerrainMaterialRecipe)})
        result = recipe.prepare_raster(source, self.grid, units=self.value_units, classes=self.classes,
                                       fields=self.fields, elevations=self.elevations,
                                       derivative_halo=self.derivative_halo)
        if valid_mask is not None:
            result.valid_mask &= np.asarray(valid_mask, dtype=bool)
            result.rgba[~result.valid_mask] = 0
        if np.any(result.rgba[result.valid_mask, 3] != 255):
            raise ValueError("native raster material requires opaque valid samples")
        return result

    def channels(self, source: Any) -> tuple[Any, Any]:
        from .raster_style import _single
        values = _single(source)
        recipe = GeospatialMaterialRecipe(**{item.name: getattr(self.recipe, item.name)
                                            for item in dataclass_fields(TerrainMaterialRecipe)})
        if self.elevations is not None:
            fields = dict(self.fields)
            fields["value"] = values, self.value_units
            result = recipe.prepare(self.elevations, self.grid, classes=self.classes, fields=fields,
                                    derivative_halo=self.derivative_halo, valid_mask=np.isfinite(values))
            return result.normals, result.roughness
        yy, xx = np.indices(values.shape)
        coordinates = np.column_stack((self.grid.origin[0] + xx.ravel() * self.grid.spacing[0],
                                       self.grid.origin[1] - yy.ravel() * self.grid.spacing[1]))
        fields = {name: (data.ravel(), units) for name, (data, units) in self.fields.items()}
        fields["value"] = values.ravel(), self.value_units
        result = recipe.evaluate_samples(coordinates, crs=self.grid.crs, units=self.grid.units, fields=fields,
                                         classes=None if self.classes is None else self.classes.ravel(), footprint_m=self.grid.spacing)
        return (None if result.normals is None else result.normals.reshape((*values.shape, 3)),
                None if result.roughness is None else result.roughness.reshape(values.shape))

    def validate_scene(self, scene: Any) -> None:
        self._inputs.validate_scene(scene)

    def to_dict(self) -> dict[str, Any]:
        from .raster_style import raster_values_to_dict
        data = self._inputs.to_dict()
        data.update(kind="geospatial_material_raster", value_units=self.value_units,
                    elevations=raster_values_to_dict(self.elevations))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialRasterStyle:
        from .raster_style import raster_values_from_dict
        base = TerrainMaterialStyle.from_dict(data)
        return cls(base.recipe, base.grid, base.classes, base.fields, base.derivative_halo,
                   data["value_units"], raster_values_from_dict(data.get("elevations")))
