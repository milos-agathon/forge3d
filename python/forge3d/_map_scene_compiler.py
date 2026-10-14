"""Own validated scene decisions and frozen plans behind MapScene's API."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ._map_scene_common import (
    _has_explicit_crs_policy,
    _layer_id,
    _metadata,
    _metadata_dict,
    _same_crs,
    _sequence,
    _json_safe,
    _stable_hash,
)
from ._map_scene_labels import (
    _atlas_glyph_set,
    _diagnostic_for_layer,
    _has_labels_or_plan,
    _label_plan_from_layer,
)
from ._map_scene_validation import (
    _BUILDING_ASSET_EXTENSIONS,
    _POINT_CLOUD_ASSET_EXTENSIONS,
    _RASTER_ASSET_EXTENSIONS,
    _TERRAIN_ASSET_EXTENSIONS,
    _asset_path_diagnostics,
    _diagnostic_codes_for_layer,
    _dimension_memory_bytes,
    _explicit_memory_bytes,
    _geometry_memory_bytes,
    _has_identity_path_or_metadata,
    _large_scene_resource_summary,
    _merge_report,
    _missing_crs_diagnostic,
    _missing_renderable_data_diagnostic,
    _missing_source_identity_diagnostic,
    _output_memory_bytes,
    _p2_building_texture_diagnostics,
    _p2_resource_availability_diagnostics,
    _point_cloud_memory_bytes,
    _source_kind,
    _source_path,
    _unsupported_feature_diagnostic,
    _unsupported_layer_type_diagnostic,
    _unsupported_output_format_diagnostic,
    _virtual_texture_report_from_metadata,
)
from .diagnostics import (
    Diagnostic,
    LayerSummary,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    experimental_feature_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unsupported_tile_feature_diagnostic,
    unsupported_tile_format_diagnostic,
    validate_label_support,
)

if TYPE_CHECKING:
    from .map_scene import CompiledScenePlan, SceneRecipe


@dataclass(frozen=True)
class _SceneCompilation:
    recipe_hash: str | None
    report: ValidationReport | None
    label_plans: dict[str, Any]
    plan: CompiledScenePlan | None = None


def _snapshot_label_value(value: Any) -> Any:
    """Normalize signed zero without rejecting in-memory nodata metadata."""
    if isinstance(value, Mapping):
        return {str(key): _snapshot_label_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_label_value(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return copy.deepcopy(value)


class _FrozenLabelPlans(Mapping[str, Any]):
    """Owned decision snapshots with detached LabelPlan values on access.

    LabelPlan remains mutable for callers that use it outside compilation.
    Private snapshots never escape through the public mapping. Nodata metadata
    may be non-finite in memory; persisted manifests retain their finite-JSON
    checks. Parse each snapshot once and copy it when read.
    """

    __slots__ = ("_snapshots",)

    def __init__(self, plans: Mapping[str, Any]) -> None:
        from .label_plan import LabelPlan

        self._snapshots = MappingProxyType({
            str(layer_id): LabelPlan.from_dict(_snapshot_label_value(plan.to_dict()))
            for layer_id, plan in plans.items()
        })

    def __getitem__(self, layer_id: str) -> Any:
        return copy.deepcopy(self._snapshots[layer_id])

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)


class _SceneCompiler:
    """Publish a complete result only after validation/compilation succeeds.

    ``label_plans`` is the historical mutable inspection view. Compilation
    never reads it: render decisions belong to the detached frozen plan.
    Restoration validates with persisted decisions, without fresh planning.
    """

    def __init__(self) -> None:
        self._state = _SceneCompilation(None, None, {})

    @property
    def report(self) -> ValidationReport | None:
        return self._state.report

    @report.setter
    def report(self, report: ValidationReport | None) -> None:
        self._state = replace(self._state, report=report)

    @property
    def plan(self) -> CompiledScenePlan | None:
        return self._state.plan

    @plan.setter
    def plan(self, plan: CompiledScenePlan | None) -> None:
        # Preserve callers attaching a native CHRONOS frame via replace().
        if plan is None:
            self._state = replace(self._state, plan=None)
        else:
            report = plan.validation_report
            previous = self.plan
            if (
                previous is not None
                and previous.recipe_hash == plan.recipe_hash
                and previous.camera_terrain_key == plan.camera_terrain_key
                and previous.label_plans is plan.label_plans
                and previous.manifest is plan.manifest
                and previous.validation_report is plan.validation_report
                and previous.render_passes_json == plan.render_passes_json
                and previous.render_pass_inputs is plan.render_pass_inputs
                and self.report is not None
            ):
                report = self.report
            self._publish(plan.recipe_hash, report, plan.label_plans, plan)

    @property
    def label_plans(self) -> dict[str, Any]:
        return self._state.label_plans

    @label_plans.setter
    def label_plans(self, plans: dict[str, Any]) -> None:
        self._state = replace(self._state, label_plans=plans)

    def _publish(
        self,
        recipe_hash: str,
        report: ValidationReport,
        label_plans: Mapping[str, Any],
        plan: CompiledScenePlan | None,
    ) -> None:
        self._state = _SceneCompilation(
            recipe_hash, report, copy.deepcopy(dict(label_plans)), plan
        )

    def _matching_plan(self, recipe_hash: str) -> CompiledScenePlan | None:
        plan = self.plan
        return plan if plan is not None and plan.recipe_hash == recipe_hash else None

    def validate(self, recipe: SceneRecipe) -> ValidationReport:
        recipe_hash = _stable_hash(recipe.to_dict(include_pass_data=False))
        plan = self._matching_plan(recipe_hash)
        report, label_plans = _validate_scene_recipe(
            recipe, frozen_label_plans=plan.label_plans if plan is not None else None
        )
        self._publish(recipe_hash, report, label_plans, plan)
        return report

    def compile(self, recipe: SceneRecipe) -> CompiledScenePlan:
        from types import MappingProxyType

        from .recipe_manifest import RecipeManifest
        from .map_scene import CompiledScenePlan

        report, label_plans = _validate_scene_recipe(recipe)
        recipe_payload = recipe.to_dict(include_pass_data=False)
        recipe_hash = _stable_hash(recipe_payload)
        camera_terrain_key = _stable_hash(
            {
                "camera": recipe.camera.to_dict(),
                "terrain": _json_safe(recipe.terrain),
                "output": recipe.output.to_dict() if recipe.output is not None else None,
            }
        )
        label_plans = copy.deepcopy(label_plans)
        compiled_label_plans = {
            str(layer_id): plan.to_dict() for layer_id, plan in sorted(label_plans.items())
        }
        depth_cull_layers: dict[str, Any] = {}
        for layer_id, plan in sorted(label_plans.items()):
            visibility: dict[str, Any] = {}
            for visible, labels_for_state in (
                (True, plan.accepted),
                (False, plan.rejected),
            ):
                for label in labels_for_state:
                    instance_id = "label-instance:" + _stable_hash(
                        {
                            "layer_id": str(layer_id),
                            "label_id": str(label.label_id),
                            "source_id": str(label.source_id),
                            "ordering_key": str(label.ordering_key or ""),
                        }
                    )
                    entry = {
                        "instance_id": instance_id,
                        "label_id": str(label.label_id),
                        "source_id": str(label.source_id),
                        "visible": visible,
                    }
                    if not visible:
                        entry["reason"] = str(label.reason)
                    visibility[instance_id] = entry
            depth_cull_layers[str(layer_id)] = {
                "accepted": [str(label.label_id) for label in plan.accepted],
                "rejected": [
                    {"label_id": str(label.label_id), "reason": str(label.reason)}
                    for label in plan.rejected
                ],
                # Public label ids are presentation metadata and need not be
                # unique.  Stable instance/source identity prevents duplicate
                # ids from overwriting each other's frozen visibility state.
                "visibility": visibility,
            }
        depth_authorities = sorted(
            {
                str(sample["depth_authority"])
                for plan in label_plans.values()
                for sample in (
                    *(
                        dict(candidate.terrain_sample or {})
                        for label in plan.accepted
                        for candidate in label.candidates
                    ),
                    *(
                        dict(label.details.get("terrain_sample") or {})
                        for label in plan.rejected
                    ),
                )
                if sample.get("depth_authority")
            }
        )
        manifest = RecipeManifest(
            recipe_family="mapscene_showcases",
            recipe_id=f"mapscene-compiled-{recipe_hash[:16]}",
            status="proven_in_forge3d",
            camera_defaults=recipe.camera.to_dict(),
            lighting_defaults=recipe.lighting.to_dict(),
            render_export_defaults=(
                recipe.output.to_dict() if recipe.output is not None else {}
            ),
            compiled_label_plans=compiled_label_plans,
            compiled_render_passes=recipe_payload.get("render_passes") or {},
            depth_cull={
                "source": "compile_phase",
                "depth_inputs": depth_authorities,
                "depth_proxy": (
                    "pre_supplied_authoritative"
                    if "pre_supplied_authoritative" in depth_authorities
                    else "deterministic_terrain_proxy"
                ),
                "camera_terrain_key": camera_terrain_key,
                "layers": depth_cull_layers,
            },
        )
        compiled = CompiledScenePlan(
            recipe_hash=recipe_hash,
            camera_terrain_key=camera_terrain_key,
            label_plans=MappingProxyType(label_plans),
            manifest=manifest,
            validation_report=report,
            render_passes_json=(json.dumps(manifest.compiled_render_passes, sort_keys=True, allow_nan=False)
                                if manifest.compiled_render_passes else None),
            render_pass_inputs=MappingProxyType(dict(recipe.pass_inputs)),
        )
        self._publish(recipe_hash, report, label_plans, compiled)
        return compiled

    def restore(self, recipe: SceneRecipe, manifest: Any) -> CompiledScenePlan:
        from types import MappingProxyType

        from .label_plan import LabelPlan
        from .map_scene import CompiledScenePlan
        from .recipe_manifest import manifest_to_json

        recipe_hash = _stable_hash(recipe.to_dict(include_pass_data=False))
        if manifest.recipe_id != f"mapscene-compiled-{recipe_hash[:16]}":
            raise ValueError("Compiled plan recipe identity does not match the current recipe")
        manifest_to_json(manifest)

        label_plans = {
            str(layer_id): LabelPlan.from_dict(payload)
            for layer_id, payload in dict(manifest.compiled_label_plans or {}).items()
        }
        # Validate the recipe and frozen plan diagnostics without making new
        # label/depth decisions while loading a v3 manifest.
        report, _ = _validate_scene_recipe(recipe, frozen_label_plans=label_plans)
        depth_cull = dict(manifest.depth_cull or {})
        pass_payload = recipe.to_dict(include_pass_data=False).get("render_passes") or {}
        if manifest.compiled_render_passes != pass_payload:
            raise ValueError("compiled render passes do not match the bundle recipe")
        compiled = CompiledScenePlan(
            recipe_hash=_stable_hash(recipe.to_dict(include_pass_data=False)),
            camera_terrain_key=str(depth_cull.get("camera_terrain_key") or ""),
            label_plans=MappingProxyType(label_plans),
            manifest=manifest,
            validation_report=report,
            render_passes_json=(json.dumps(manifest.compiled_render_passes, sort_keys=True, allow_nan=False)
                                if manifest.compiled_render_passes else None),
            render_pass_inputs=MappingProxyType(dict(recipe.pass_inputs)),
        )
        self._publish(recipe_hash, report, label_plans, compiled)
        return compiled

    def current(self, recipe: SceneRecipe) -> CompiledScenePlan:
        plan = self._matching_plan(_stable_hash(recipe.to_dict(include_pass_data=False)))
        return plan if plan is not None else self.compile(recipe)


def _validate_scene_recipe(
    recipe: SceneRecipe, *, frozen_label_plans: Mapping[str, Any] | None = None
) -> tuple[ValidationReport, dict[str, Any]]:
    """Evaluate diagnostics while owning shared projection resources."""
    projection_context: dict[str, Any] = {}
    try:
        return _validate_with_projection(recipe, projection_context, frozen_label_plans)
    finally:
        projection_context.pop("depth", None)
        reservation = projection_context.pop("depth_reservation", None)
        if reservation is not None:
            reservation.close()


def _validate_with_projection(
    recipe: SceneRecipe,
    projection_context: dict[str, Any],
    frozen_label_plans: Mapping[str, Any] | None,
) -> tuple[ValidationReport, dict[str, Any]]:
    from .map_scene import (
        BuildingLayer,
        LabelLayer,
        MapSceneNativeUnavailable,
        PointCloudLayer,
        RasterOverlay,
        Tiles3DLayer,
        VectorOverlay,
        _alignment_residual_diagnostic,
        _alignment_transform_diagnostic,
        _is_3d_camera_mode,
        _load_raw_heightmap,
        _mapscene_effective_camera_mode,
        _metadata_resolution,
        _path_to_str,
        _preset_override_diagnostic,
        _resolution_mismatch_diagnostic,
        _tiles3d_layer_has_native_geometry,
    )

    if recipe.pass_specs or recipe.pass_inputs:
        recipe._pass_payload()
    label_plans: dict[str, Any] = {}
    seen_label_layer_ids: set[str] = set()
    diagnostics: list[Diagnostic] = []
    layer_summaries: list[LayerSummary] = []
    supported_features: dict[str, str] = {
        "mapscene.recipe": "underdeveloped",
        "mapscene.validation": "supported",
    }
    unsupported_features: dict[str, str] = {}
    total_memory = 0
    memory_known = False

    preset_override = _preset_override_diagnostic(recipe.lighting)
    if preset_override is not None:
        diagnostics.append(preset_override)
    lighting_settings = _metadata_dict(recipe.lighting.settings)
    if lighting_settings.get("resolved_preset"):
        supported_features["mapscene.presets"] = "supported"
        supported_features["mapscene.lighting_preset"] = "supported"

    output = recipe.output
    if output is not None and str(output.format).lower() not in {"png", "exr"}:
        diagnostics.append(_unsupported_output_format_diagnostic(str(output.format).lower()))
        unsupported_features["output.format"] = "unsupported"
    elif output is not None:
        supported_features[f"output.format.{str(output.format).lower()}"] = "supported"
        supported_features[f"output.bit_depth.{int(output.bit_depth)}"] = "supported"
        if int(output.samples) > 1:
            supported_features["mapscene.offline_accumulation"] = "supported"
        if output.aovs:
            supported_features["mapscene.aov_export"] = "supported"
        if bool(output.hdr) or str(output.format).lower() == "exr":
            supported_features["mapscene.hdr_output"] = "supported"

    furniture = recipe.map_furniture
    if furniture is not None:
        supported_features["mapscene.furniture"] = "supported"
        if furniture.graticule is not None:
            supported_features["mapscene.graticule"] = "supported"
        if furniture.scale_bar is not None:
            supported_features["mapscene.scale_bar"] = "supported"
            scale_bar_options = _metadata_dict(furniture.scale_bar)
            if bool(scale_bar_options.get("geodesic", True)):
                supported_features["mapscene.scale_bar.geodesic"] = "supported"
        if furniture.north_arrow is not None:
            supported_features["mapscene.north_arrow"] = "supported"
        if furniture.legend is not None:
            supported_features["mapscene.legend"] = "underdeveloped"

    output_bytes = _output_memory_bytes(output)
    if output_bytes is not None:
        total_memory += output_bytes
        memory_known = True

    terrain = recipe.terrain
    scene_crs = recipe.target_crs or terrain.crs
    terrain_metadata = _metadata_dict(terrain.metadata)
    terrain_memory = _dimension_memory_bytes(terrain.metadata)
    if terrain_memory is not None:
        total_memory += terrain_memory
        memory_known = True

    vt_report = _virtual_texture_report_from_metadata(terrain.metadata)
    if vt_report is not None:
        _merge_report(
            vt_report,
            diagnostics=diagnostics,
            layer_summaries=layer_summaries,
            supported_features=supported_features,
            unsupported_features=unsupported_features,
        )

    terrain_support_level = "supported"
    terrain_details = {
        "crs": scene_crs,
        "elevation_sampling_available": bool(terrain.elevation_sampling_available),
        "path": _path_to_str(terrain.path),
    }
    if terrain.style is not None:
        try:
            styled = terrain.style.apply(_load_raw_heightmap(terrain))
            terrain_details["style"] = terrain.style.to_dict()
            terrain_details["legend"] = styled.legend
            supported_features["terrain.population_height_shade"] = "supported"
        except (ValueError, TypeError, RuntimeError) as exc:
            diagnostics.append(Diagnostic(code="invalid_raster_style", severity="error", message=str(exc),
                remediation="Correct the population values, nodata or scale.", layer_id="terrain", support_level="unsupported"))
            terrain_support_level = "unsupported"
    terrain_resource_diagnostics, terrain_resource_details, terrain_resource_support = (
        _p2_resource_availability_diagnostics(
            terrain.metadata,
            layer_id="terrain",
            layer_type="terrain_source",
        )
    )
    if terrain_resource_diagnostics:
        diagnostics.extend(terrain_resource_diagnostics)
        terrain_details.update(terrain_resource_details)
        if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in terrain_resource_diagnostics):
            unsupported_features["terrain.cache_lod_stats"] = "underdeveloped"
        if terrain_resource_support == "underdeveloped":
            terrain_support_level = "underdeveloped"
    if not _has_identity_path_or_metadata(terrain.path, terrain.metadata):
        diagnostics.append(
            _missing_source_identity_diagnostic(
                "terrain_source",
                layer_id="terrain",
                source_fields=("path", "metadata"),
            )
        )
        unsupported_features["terrain.source_identity"] = "unsupported"
        terrain_support_level = "unsupported"
    else:
        terrain_asset_diagnostics = _asset_path_diagnostics(
            "terrain_source",
            layer_id="terrain",
            path=terrain.path,
            metadata=terrain.metadata,
            supported_extensions=_TERRAIN_ASSET_EXTENSIONS,
        )
        if terrain_asset_diagnostics:
            diagnostics.extend(terrain_asset_diagnostics)
            unsupported_features["terrain.asset"] = "unsupported"
            terrain_support_level = "unsupported"
    if not scene_crs:
        diagnostics.append(_missing_crs_diagnostic(None, layer_id="terrain"))
        unsupported_features["terrain.crs"] = "unsupported"
        terrain_support_level = "unsupported"
    if terrain_metadata.get("alignment_transform_applied"):
        source_crs = str(terrain_metadata.get("source_crs") or terrain.crs or "")
        target = str(terrain_metadata.get("target_crs") or scene_crs or "")
        diagnostics.append(_alignment_transform_diagnostic("terrain", source_crs, target))
        supported_features["mapscene.alignment"] = "supported"

    layer_summaries.append(
        LayerSummary(
            layer_id="terrain",
            layer_type="terrain_source",
            support_level=terrain_support_level,
            diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, "terrain"),
            memory_estimate_bytes=terrain_memory,
            details=terrain_details,
        )
    )
    supported_features["layer.terrain"] = "supported"
    if recipe.target_crs:
        supported_features["mapscene.alignment"] = "supported"

    from ._map_scene_rasters import raster_route_blocks
    raster_blocks = {block["layer"]: block for block in raster_route_blocks(recipe)}
    for index, layer in enumerate(recipe.layers):
        layer_id = _layer_id(layer, f"layer_{index}")
        layer_diagnostics: list[Diagnostic] = []
        layer_memory: int | None = None
        support_level = "supported"
        layer_type = type(layer).__name__
        object_count: int | None = None
        details: dict[str, Any] = {}
        layer_metadata = _metadata_dict(getattr(layer, "metadata", None))
        terrain_resolution = _metadata_resolution(terrain.metadata)
        layer_resolution = _metadata_resolution(layer_metadata)
        if terrain_resolution is not None and layer_resolution is not None:
            rel_x = abs(layer_resolution[0] - terrain_resolution[0]) / max(abs(terrain_resolution[0]), 1.0e-12)
            rel_y = abs(layer_resolution[1] - terrain_resolution[1]) / max(abs(terrain_resolution[1]), 1.0e-12)
            if max(rel_x, rel_y) > 1.0e-6:
                layer_diagnostics.append(
                    _resolution_mismatch_diagnostic(layer_id, terrain_resolution, layer_resolution)
                )
        residual = layer_metadata.get("alignment_residual")
        if isinstance(residual, Mapping):
            max_residual = float(residual.get("max", residual.get("max_error", 0.0)) or 0.0)
            threshold = float(residual.get("threshold", residual.get("tolerance", 1.0)) or 1.0)
            if max_residual > threshold:
                layer_diagnostics.append(_alignment_residual_diagnostic(layer_id, residual))
        if layer_metadata.get("alignment_transform_applied"):
            source_crs = str(layer_metadata.get("source_crs"))
            target = str(layer_metadata.get("target_crs") or scene_crs)
            layer_diagnostics.append(_alignment_transform_diagnostic(layer_id, source_crs, target))
            supported_features["mapscene.alignment"] = "supported"

        layer_crs = getattr(layer, "crs", None)
        has_crs_field = hasattr(layer, "crs")
        if scene_crs and has_crs_field and layer_crs is None and not _has_explicit_crs_policy(layer):
            layer_diagnostics.append(_missing_crs_diagnostic(str(scene_crs), layer_id=layer_id))
            support_level = "unsupported"
        elif scene_crs and layer_crs and not _same_crs(scene_crs, str(layer_crs)) and not _has_explicit_crs_policy(layer):
            layer_diagnostics.append(
                crs_mismatch_diagnostic(
                    str(scene_crs),
                    str(layer_crs),
                    layer_id=layer_id,
                )
            )
            support_level = "unsupported"

        if isinstance(layer, RasterOverlay) and str(layer_id) in raster_blocks:
            block = raster_blocks[str(layer_id)]
            layer_type = "raster_overlay"
            layer_memory = _dimension_memory_bytes(layer.metadata)
            layer_diagnostics.append(Diagnostic(
                code="unsupported_raster_route", severity="fatal",
                message=block["reason"],
                remediation="Use one styled overlay or an ordered stack of unstyled overlays.",
                layer_id=layer_id, support_level="unsupported", details=block,
            ))
            unsupported_features["raster.overlay_route"] = "unsupported"
            support_level = "unsupported"
        elif isinstance(layer, RasterOverlay) and layer.style is not None:
            layer_type = "raster_overlay"
            layer_memory = _dimension_memory_bytes(layer.metadata)
            details = {"style": layer.style.to_dict(), "opacity": layer.opacity}
            try:
                styled = _styled_raster_result(layer)
                terrain_shape = _load_native_heightmap(terrain).shape
                _mapscene_thematic_surface(recipe, terrain_shape)
                details["legend"] = styled.legend
                supported_features["raster.thematic_style"] = "supported"
            except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
                layer_diagnostics.append(Diagnostic(code="invalid_raster_style", severity="error", message=str(exc),
                    remediation="Correct thematic inputs and align both raster axes to the terrain grid.", layer_id=layer_id, support_level="unsupported"))
                support_level = "unsupported"
        elif isinstance(layer, RasterOverlay):
            layer_type = "raster_overlay"
            layer_memory = _dimension_memory_bytes(layer.metadata)
            details = {"path": _path_to_str(layer.path), "crs": layer.crs, "opacity": float(layer.opacity)}
            supported_features["layer.raster_overlay"] = "supported"
            if not _has_identity_path_or_metadata(layer.path, layer.metadata):
                layer_diagnostics.append(
                    _missing_source_identity_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        source_fields=("path", "metadata"),
                    )
                )
                unsupported_features["raster.source_identity"] = "unsupported"
                support_level = "unsupported"
            else:
                layer_diagnostics.extend(
                    _asset_path_diagnostics(
                        layer_type,
                        layer_id=layer_id,
                        path=layer.path,
                        metadata=layer.metadata,
                        supported_extensions=_RASTER_ASSET_EXTENSIONS,
                    )
                )
                if any(
                    diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                    for diagnostic in layer_diagnostics
                ):
                    unsupported_features["raster.asset"] = "unsupported"
                    support_level = "unsupported"
        elif isinstance(layer, VectorOverlay):
            layer_type = "vector_overlay"
            object_count = len(layer.features or ())
            details = {
                "path": _path_to_str(layer.path),
                "crs": layer.crs,
                "feature_count": object_count,
                "width_px": layer.width_px,
                "width_world": layer.width_world,
                "line_join": layer.line_join,
                "line_cap": layer.line_cap,
                "dash_array": _sequence(layer.dash_array),
            }
            supported_features["layer.vector_overlay"] = "supported"
            supported_features["vector.stroke.joins_caps"] = "supported"
            if layer.dash_array:
                supported_features["vector.stroke.dashes"] = "supported"
            if layer.width_px is not None:
                supported_features["vector.stroke.width_px"] = "supported"
            if layer.width_world is not None:
                supported_features["vector.stroke.width_world"] = "supported"
            if not layer.path and not layer.features:
                layer_diagnostics.append(
                    _missing_renderable_data_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        required_any_of=("path", "features"),
                    )
                )
                unsupported_features["vector.renderable_data"] = "unsupported"
                support_level = "unsupported"
            elif layer.path and not layer.features:
                layer_diagnostics.append(placeholder_fallback_diagnostic("vector path loader", layer_id=layer_id))
                unsupported_features["vector.path_loader"] = "placeholder/fallback"
                support_level = "placeholder/fallback"
            if layer.features and _is_3d_camera_mode(_mapscene_effective_camera_mode(recipe)):
                from ._map_scene_projection import project_vector_recipe
                try:
                    projection_recipe = copy.copy(recipe)
                    projection_recipe.layers = (layer,)
                    project_vector_recipe(projection_recipe)
                except (MapSceneNativeUnavailable, ValueError, TypeError, RuntimeError) as exc:
                    layer_diagnostics.append(Diagnostic(
                        code="world_overlay_projection_unavailable", severity="error", message=str(exc),
                        remediation="Provide finite world coordinates inside the terrain footprint.",
                        support_level="unsupported", layer_id=layer_id))
                    unsupported_features["vector.world_projection"] = "unsupported"
                    support_level = "unsupported"
            if layer.style:
                for style_layer in (layer.style.get("layers", ()) if isinstance(layer.style, Mapping) else ()):
                    if not isinstance(style_layer, Mapping) or str(style_layer.get("type", "")).lower() != "line":
                        continue
                    paint = style_layer.get("paint") if isinstance(style_layer.get("paint"), Mapping) else {}
                    layout = style_layer.get("layout") if isinstance(style_layer.get("layout"), Mapping) else {}
                    if paint.get("line-dasharray"):
                        supported_features["vector.stroke.dashes"] = "supported"
                    if layout.get("line-cap") or layout.get("line-join"):
                        supported_features["vector.stroke.joins_caps"] = "supported"
                from .style import validate_style_support

                style_report = validate_style_support(dict(layer.style))
                _merge_report(
                    style_report,
                    diagnostics=diagnostics,
                    layer_summaries=layer_summaries,
                    supported_features=supported_features,
                    unsupported_features=unsupported_features,
                )
                if style_report.status in {"error", "fatal"}:
                    support_level = "unsupported"
                elif style_report.status == "warning" and support_level == "supported":
                    support_level = "underdeveloped"
        elif isinstance(layer, LabelLayer):
            if layer_id in seen_label_layer_ids:
                raise ValueError(f"Duplicate label layer id {layer_id!r}")
            seen_label_layer_ids.add(layer_id)
            layer_type = "label_layer"
            labels = list(layer.labels or ())
            object_count = len(labels)
            details = {"label_count": object_count, "occlusion": str(layer.occlusion)}
            supported_features["layer.label_layer"] = "supported"
            if layer.occlusion == "terrain":
                supported_features["labels.occlusion.terrain"] = "supported"
                layer_metadata = _metadata_dict(layer.metadata)
                terrain_metadata = _metadata_dict(terrain.metadata)
                if any(
                    key in layer_metadata or key in terrain_metadata
                    for key in ("depth_occlusion", "depth_aov", "depth_buffer", "depth_image")
                ):
                    supported_features["labels.occlusion.depth_aov"] = "supported"
            else:
                supported_features["labels.occlusion.none"] = "supported"
            layer_diagnostics.extend(
                diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                for diagnostic in (layer.diagnostics or ())
            )
            if not _has_labels_or_plan(layer):
                layer_diagnostics.append(
                    _missing_renderable_data_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        required_any_of=("labels", "plan"),
                    )
                )
                unsupported_features["labels.renderable_data"] = "unsupported"
                support_level = "unsupported"
            label_report = validate_label_support(
                labels,
                atlas_glyphs=_atlas_glyph_set(layer.glyph_atlas),
                layer_id=layer_id,
            )
            _merge_report(
                label_report,
                diagnostics=diagnostics,
                layer_summaries=layer_summaries,
                supported_features=supported_features,
                unsupported_features=unsupported_features,
            )
            if _has_labels_or_plan(layer):
                frozen_plans = frozen_label_plans
                if frozen_plans is not None:
                    if layer_id not in frozen_plans:
                        raise ValueError(f"Compiled plan missing label layer {layer_id!r}")
                    plan = frozen_plans[layer_id]
                else:
                    try:
                        plan = _label_plan_from_layer(layer, recipe=recipe, terrain=terrain,
                            projection_context=projection_context)
                    except (MapSceneNativeUnavailable, ValueError, TypeError, RuntimeError) as exc:
                        from .label_plan import LabelPlan
                        world_projection = _is_3d_camera_mode(_mapscene_effective_camera_mode(recipe))
                        plan = LabelPlan(accepted=(), rejected=(), diagnostics=[Diagnostic(
                            code="world_overlay_projection_unavailable" if world_projection else "label_plan_compile_unavailable",
                            severity="error", message=str(exc),
                            remediation="Provide valid label geometry, projection and depth metadata.",
                            support_level="unsupported", layer_id=layer_id,
                        )])
                label_plans[layer_id] = plan
                plan_diagnostics = [_diagnostic_for_layer(diagnostic, layer_id) for diagnostic in plan.diagnostics]
                existing_diagnostic_keys = {
                    (diagnostic.code, diagnostic.layer_id, diagnostic.object_id)
                    for diagnostic in diagnostics
                }
                plan_diagnostics = [
                    diagnostic
                    for diagnostic in plan_diagnostics
                    if (diagnostic.code, diagnostic.layer_id, diagnostic.object_id) not in existing_diagnostic_keys
                ]
                diagnostics.extend(plan_diagnostics)
                details["compiled_label_plan"] = {
                    "accepted_count": len(plan.accepted),
                    "rejected_count": len(plan.rejected),
                    "diagnostic_codes": sorted({diagnostic.code for diagnostic in plan_diagnostics}),
                    "seed": plan.seed,
                }
                if any(diagnostic.severity in {"error", "fatal"} for diagnostic in plan_diagnostics):
                    support_level = "underdeveloped"
                elif support_level == "supported":
                    support_level = "supported"
        elif isinstance(layer, PointCloudLayer):
            layer_type = "point_cloud_layer"
            object_count = layer.point_count
            layer_memory = _point_cloud_memory_bytes(layer.point_count)
            details = {"path": _path_to_str(layer.path), "crs": layer.crs, "point_count": layer.point_count}
            has_renderable_points = "positions" in _metadata_dict(layer.metadata) or bool(layer.path)
            supported_features["layer.point_cloud"] = "supported" if has_renderable_points else "underdeveloped"
            if has_renderable_points:
                supported_features["point_cloud.mapscene_render"] = "supported"
            support_level = "supported" if has_renderable_points else "underdeveloped"
            if not layer.path and layer.point_count is None and not _metadata_dict(layer.metadata):
                layer_diagnostics.append(
                    _missing_source_identity_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        source_fields=("path", "point_count", "metadata"),
                    )
                )
                unsupported_features["point_cloud.source_identity"] = "unsupported"
                support_level = "unsupported"
            else:
                layer_diagnostics.extend(
                    _asset_path_diagnostics(
                        layer_type,
                        layer_id=layer_id,
                        path=layer.path,
                        metadata=layer.metadata,
                        supported_extensions=_POINT_CLOUD_ASSET_EXTENSIONS,
                    )
                )
                if any(
                    diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                    for diagnostic in layer_diagnostics
                ):
                    unsupported_features["point_cloud.asset"] = "unsupported"
                    support_level = "unsupported"
            if support_level == "underdeveloped":
                layer_diagnostics.append(
                    placeholder_fallback_diagnostic("point cloud MapScene render path", layer_id=layer_id)
                )
                unsupported_features["point_cloud.mapscene_render"] = "placeholder/fallback"
                support_level = "placeholder/fallback"
        elif isinstance(layer, Tiles3DLayer):
            layer_type = "tiles3d_layer"
            object_count = None
            layer_memory = _explicit_memory_bytes(layer.metadata)
            source_path = _source_path(layer.source)
            source_kind = _source_kind(layer.source, layer.metadata)
            details = {
                "source_kind": source_kind,
                "cache_budget": layer.cache_budget,
                "cache_stats": _metadata(layer.cache_stats),
                "lod": _metadata(layer.lod),
            }
            supported_features["layer.tiles3d_intent"] = "underdeveloped"
            support_level = layer.support_level
            layer_diagnostics.extend(
                diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                for diagnostic in (layer.diagnostics or ())
            )
            if layer.source is None and not _metadata_dict(layer.metadata):
                layer_diagnostics.append(
                    _missing_source_identity_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        source_fields=("source", "metadata"),
                    )
                )
                unsupported_features["tiles3d.source_identity"] = "unsupported"
                support_level = "unsupported"
            else:
                if source_path:
                    if not source_path.lower().endswith(("tileset.json", ".b3dm", ".pnts")):
                        layer_diagnostics.append(
                            unsupported_tile_format_diagnostic(
                                Path(source_path).suffix.lstrip(".") or source_kind or "unknown",
                                layer_id=layer_id,
                                object_id=source_path,
                            )
                        )
                        unsupported_features["tiles3d.format"] = "unsupported"
                        support_level = "unsupported"
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=source_path,
                            metadata=layer.metadata,
                            supported_extensions=("tileset.json", ".b3dm", ".pnts"),
                        )
                    )
                for feature in _metadata_dict(layer.metadata).get("unsupported_features", ()) or ():
                    layer_diagnostics.append(
                        unsupported_tile_feature_diagnostic(str(feature), layer_id=layer_id)
                    )
                    unsupported_features["tiles3d.feature"] = "unsupported"
                    support_level = "unsupported"
                if support_level != "unsupported" and _tiles3d_layer_has_native_geometry(layer):
                    supported_features["layer.tiles3d_intent"] = "supported"
                    supported_features["tiles3d.mapscene_render"] = "supported"
                    support_level = "supported"
                elif support_level != "unsupported":
                    layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                    unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                    support_level = "underdeveloped"
        elif isinstance(layer, BuildingLayer):
            layer_type = "building_layer"
            object_count = layer.geometry_count
            layer_memory = _geometry_memory_bytes(layer.geometry_count)
            source_kind = _source_kind(layer.source, layer.metadata)
            support_level = layer.support_level
            details = {
                "geometry_count": layer.geometry_count,
                "material_status": layer.material_status,
                "source_kind": source_kind,
            }
            scalar_building = str(layer.material_status or "").lower().startswith("scalar")
            supported_features["layer.building_intent"] = "supported" if scalar_building else "underdeveloped"
            if scalar_building:
                supported_features["buildings.scalar_materials"] = "supported"
                supported_features["buildings.mapscene_render"] = "supported"
            if layer.source is None and not _metadata_dict(layer.metadata):
                layer_diagnostics.append(
                    _missing_source_identity_diagnostic(
                        layer_type,
                        layer_id=layer_id,
                        source_fields=("source", "metadata"),
                    )
                )
                unsupported_features["buildings.source_identity"] = "unsupported"
                support_level = "unsupported"
            elif (
                layer.support_level == "supported"
                or (layer.support_level == "underdeveloped" and bool(layer.features))
            ) and scalar_building:
                layer_diagnostics.extend(
                    _asset_path_diagnostics(
                        layer_type,
                        layer_id=layer_id,
                        path=_source_path(layer.source),
                        metadata=layer.metadata,
                        supported_extensions=_BUILDING_ASSET_EXTENSIONS,
                    )
                )
                if any(
                    diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                    for diagnostic in layer_diagnostics
                ):
                    unsupported_features["buildings.asset"] = "unsupported"
                    support_level = "unsupported"
                else:
                    support_level = "supported"
            if source_kind == "3dtiles":
                layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                support_level = "underdeveloped"
            if layer.support_level == "Pro-gated":
                layer_diagnostics.append(pro_gated_path_diagnostic("building layer", layer_id=layer_id))
                unsupported_features["buildings.pro_gated_path"] = "Pro-gated"
            if str(layer.material_status or "").lower() in {"textured_pbr_unsupported", "textured pbr unsupported"}:
                layer_diagnostics.append(
                    _unsupported_feature_diagnostic("building textured PBR", layer_id=layer_id)
                )
                unsupported_features["buildings.textured_pbr"] = "unsupported"
            elif layer.support_level == "placeholder/fallback" or (
                layer.geometry_count == 0 and layer.support_level not in {"experimental", "underdeveloped"}
            ):
                layer_diagnostics.append(placeholder_fallback_diagnostic("building layer", layer_id=layer_id))
                unsupported_features["buildings.placeholder_fallback"] = "placeholder/fallback"
                support_level = "placeholder/fallback"
            elif layer.support_level == "experimental":
                layer_diagnostics.append(experimental_feature_diagnostic("building layer", layer_id=layer_id))
                unsupported_features["buildings.experimental"] = "experimental"
            elif layer.support_level == "unsupported":
                layer_diagnostics.append(_unsupported_feature_diagnostic("building layer", layer_id=layer_id))
                unsupported_features["buildings.unsupported"] = "unsupported"
            texture_diagnostics, texture_details, texture_support = _p2_building_texture_diagnostics(
                layer.metadata,
                layer_id=layer_id,
            )
            if texture_details:
                details.update(texture_details)
            if texture_diagnostics:
                layer_diagnostics.extend(texture_diagnostics)
                if any(diagnostic.code == "missing_texture_path" for diagnostic in texture_diagnostics):
                    unsupported_features["buildings.missing_texture_path"] = "unsupported"
                if any(diagnostic.code == "missing_uvs" for diagnostic in texture_diagnostics):
                    unsupported_features["buildings.missing_uvs"] = "unsupported"
                if any(diagnostic.code == "unsupported_texture_format" for diagnostic in texture_diagnostics):
                    unsupported_features["buildings.unsupported_texture_format"] = "unsupported"
                if any(diagnostic.code == "placeholder_fallback" for diagnostic in texture_diagnostics):
                    unsupported_features["buildings.textured_material_fallback"] = "placeholder/fallback"
                if any(diagnostic.code == "unsupported_feature" for diagnostic in texture_diagnostics):
                    unsupported_features["buildings.textured_pbr"] = "unsupported"
                if texture_support == "unsupported":
                    support_level = "unsupported"
                elif texture_support == "placeholder/fallback" and support_level not in {"unsupported", "Pro-gated"}:
                    support_level = "placeholder/fallback"
            elif texture_support == "supported":
                supported_features["buildings.textured_pbr"] = "supported"
        else:
            layer_diagnostics.append(_unsupported_layer_type_diagnostic(layer, layer_id=layer_id))
            unsupported_features["layer.unknown"] = "unsupported"
            support_level = "unsupported"
            details = {"python_type": type(layer).__name__}

        resource_diagnostics, resource_details, resource_support = _p2_resource_availability_diagnostics(
            getattr(layer, "metadata", None),
            layer_id=layer_id,
            layer_type=layer_type,
        )
        if resource_diagnostics:
            layer_diagnostics.extend(resource_diagnostics)
            details.update(resource_details)
            if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in resource_diagnostics):
                unsupported_features[f"{layer_type}.cache_lod_stats"] = "underdeveloped"
            if any(diagnostic.code == "unsupported_instancing_path" for diagnostic in resource_diagnostics):
                unsupported_features[f"{layer_type}.instancing"] = "unsupported"
            if resource_support == "unsupported":
                support_level = "unsupported"
            elif resource_support == "underdeveloped" and support_level == "supported":
                support_level = "underdeveloped"

        if layer_memory is not None:
            total_memory += layer_memory
            memory_known = True
        if any(diagnostic.code in {"crs_mismatch", "missing_crs"} for diagnostic in layer_diagnostics):
            support_level = "unsupported"

        diagnostics.extend(layer_diagnostics)
        layer_summaries.append(
            LayerSummary(
                layer_id=layer_id,
                layer_type=layer_type,
                support_level=support_level,
                diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, layer_id),
                object_count=object_count,
                bounds=getattr(layer, "bounds", None),
                memory_estimate_bytes=layer_memory,
                details=details,
            )
        )

    estimated_memory = total_memory if memory_known else None
    budget = None
    diagnostics_policy = _metadata_dict(recipe.diagnostics_policy)
    if "gpu_memory_budget_bytes" in diagnostics_policy:
        try:
            budget = int(diagnostics_policy["gpu_memory_budget_bytes"])
        except (TypeError, ValueError):
            budget = None
    if estimated_memory is not None and budget is not None:
        diagnostics.append(
            estimated_gpu_memory_diagnostic(
                estimated_memory,
                budget,
                layer_id="scene",
            )
        )

    if bool(diagnostics_policy.get("large_scene_summary")):
        layer_summaries.append(
            _large_scene_resource_summary(
                layer_summaries,
                estimated_memory=estimated_memory,
                diagnostics=diagnostics,
            )
        )

    report = ValidationReport(
        diagnostics=diagnostics,
        layer_summaries=layer_summaries,
        estimated_gpu_memory_bytes=estimated_memory,
        supported_features=supported_features,
        unsupported_features=unsupported_features,
    )
    return report, label_plans
