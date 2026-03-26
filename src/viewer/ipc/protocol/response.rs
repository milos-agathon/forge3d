use serde::Serialize;

use super::payloads::{BundleRequest, TerrainVolumetricsReport, ViewerStats};

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
        }
    }
}
