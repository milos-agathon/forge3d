use crate::gis::error::{GisError, GisResult};

use super::model::BACKEND_UNAVAILABLE;

pub(crate) fn topology_backend_available() -> bool {
    false
}

pub(crate) fn require_topology_backend(operation: &str) -> GisResult<()> {
    let _ = topology_backend_available();
    Err(GisError::BackendUnavailable(format!(
        "{BACKEND_UNAVAILABLE}: geos-topology feature required for {operation}"
    )))
}
