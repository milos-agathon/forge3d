use serde::Serialize;

use super::payloads::{BundleRequest, TerrainVolumetricsReport, ViewerStats};
use crate::viewer::scene_review::{ReviewLayerSummary, SceneVariantSummary};

#[derive(Debug, Clone, Serialize)]
pub struct IpcResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ViewerStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick_events: Option<Vec<crate::picking::PickEvent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lasso_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle_request: Option<BundleRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_volumetrics_report: Option<TerrainVolumetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_variants: Option<Vec<SceneVariantSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_layers: Option<Vec<ReviewLayerSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_scene_variant: Option<Option<String>>,
}

impl IpcResponse {
    pub fn success() -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(msg.into()),
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_stats(stats: ViewerStats) -> Self {
        Self {
            ok: true,
            error: None,
            stats: Some(stats),
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_pick_events(events: Vec<crate::picking::PickEvent>) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: Some(events),
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_bundle_request(req: BundleRequest) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: Some(req),
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_lasso_state(state: String) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: Some(state),
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_terrain_volumetrics_report(report: TerrainVolumetricsReport) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: Some(report),
            scene_variants: None,
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_scene_variants(scene_variants: Vec<SceneVariantSummary>) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: Some(scene_variants),
            review_layers: None,
            active_scene_variant: None,
        }
    }

    pub fn with_review_layers(review_layers: Vec<ReviewLayerSummary>) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: Some(review_layers),
            active_scene_variant: None,
        }
    }

    pub fn with_active_scene_variant(active_scene_variant: Option<String>) -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
            pick_events: None,
            lasso_state: None,
            bundle_request: None,
            terrain_volumetrics_report: None,
            scene_variants: None,
            review_layers: None,
            active_scene_variant: Some(active_scene_variant),
        }
    }
}
