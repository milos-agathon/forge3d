use crate::viewer::scene_review::{
    ViewerRasterOverlayConfig, ViewerReviewLayerConfig, ViewerSceneBaseStateConfig,
    ViewerSceneReviewStateConfig, ViewerSceneVariantConfig, ViewerVectorOverlayLayerConfig,
};
use crate::viewer::viewer_enums::ViewerCmd;

use super::super::payloads::{
    IpcRasterOverlaySpec, IpcReviewLayer, IpcSceneBaseState, IpcSceneReviewState, IpcSceneVariant,
    IpcSceneVectorOverlay,
};
use super::super::request::IpcRequest;
use super::terrain::map_terrain_scatter_batch;

pub(super) fn to_viewer_cmd(req: &IpcRequest) -> Result<Option<ViewerCmd>, String> {
    match req {
        IpcRequest::SetSceneReviewState { state } => Ok(Some(ViewerCmd::SetSceneReviewState {
            state: map_scene_review_state(state)?,
        })),
        IpcRequest::ApplySceneVariant { variant_id } => Ok(Some(ViewerCmd::ApplySceneVariant {
            variant_id: variant_id.clone(),
        })),
        IpcRequest::SetReviewLayerVisible { layer_id, visible } => {
            Ok(Some(ViewerCmd::SetReviewLayerVisible {
                layer_id: layer_id.clone(),
                visible: *visible,
            }))
        }
        _ => Ok(None),
    }
}

fn map_scene_review_state(
    config: &IpcSceneReviewState,
) -> Result<ViewerSceneReviewStateConfig, String> {
    let state = ViewerSceneReviewStateConfig {
        base_state: map_base_state(&config.base_state, "base_state")?,
        review_layers: config
            .review_layers
            .iter()
            .enumerate()
            .map(|(index, layer)| map_review_layer(layer, index))
            .collect::<Result<Vec<_>, _>>()?,
        variants: config.variants.iter().map(map_variant).collect::<Vec<_>>(),
        active_variant_id: config.active_variant_id.clone(),
    };
    state.validate()?;
    Ok(state)
}

fn map_base_state(
    config: &IpcSceneBaseState,
    context: &str,
) -> Result<ViewerSceneBaseStateConfig, String> {
    Ok(ViewerSceneBaseStateConfig {
        preset: config.preset.clone(),
        raster_overlays: config
            .raster_overlays
            .iter()
            .map(map_raster_overlay)
            .collect(),
        vector_overlays: config
            .vector_overlays
            .iter()
            .map(map_vector_overlay)
            .collect(),
        labels: config.labels.clone(),
        scatter_batches: config
            .scatter_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                map_terrain_scatter_batch(batch, index).map_err(|e| format!("{context}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn map_review_layer(
    config: &IpcReviewLayer,
    layer_index: usize,
) -> Result<ViewerReviewLayerConfig, String> {
    Ok(ViewerReviewLayerConfig {
        id: config.id.clone(),
        name: config.name.clone(),
        description: config.description.clone(),
        raster_overlays: config
            .raster_overlays
            .iter()
            .map(map_raster_overlay)
            .collect(),
        vector_overlays: config
            .vector_overlays
            .iter()
            .map(map_vector_overlay)
            .collect(),
        labels: config.labels.clone(),
        scatter_batches: config
            .scatter_batches
            .iter()
            .enumerate()
            .map(|(batch_index, batch)| {
                map_terrain_scatter_batch(batch, batch_index)
                    .map_err(|e| format!("review layer {}: {}", layer_index, e))
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn map_variant(config: &IpcSceneVariant) -> ViewerSceneVariantConfig {
    ViewerSceneVariantConfig {
        id: config.id.clone(),
        name: config.name.clone(),
        description: config.description.clone(),
        active_layer_ids: config.active_layer_ids.clone(),
        preset: config.preset.clone(),
    }
}

fn map_raster_overlay(config: &IpcRasterOverlaySpec) -> ViewerRasterOverlayConfig {
    ViewerRasterOverlayConfig {
        name: config.name.clone(),
        path: config.path.clone(),
        extent: config.extent,
        opacity: config.opacity,
        z_order: config.z_order,
    }
}

fn map_vector_overlay(config: &IpcSceneVectorOverlay) -> ViewerVectorOverlayLayerConfig {
    ViewerVectorOverlayLayerConfig {
        name: config.name.clone(),
        vertices: config.vertices.clone(),
        indices: config.indices.clone(),
        primitive: config.primitive.clone(),
        drape: config.drape,
        drape_offset: config.drape_offset,
        opacity: config.opacity,
        depth_bias: config.depth_bias,
        line_width: config.line_width,
        point_size: config.point_size,
        z_order: config.z_order,
    }
}
