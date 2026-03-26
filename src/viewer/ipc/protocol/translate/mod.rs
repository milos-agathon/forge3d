mod core;
mod labels;
mod overlays;
mod terrain;

use crate::viewer::viewer_enums::ViewerCmd;

use super::request::IpcRequest;

pub fn ipc_request_to_viewer_cmd(req: &IpcRequest) -> Result<Option<ViewerCmd>, String> {
    if matches!(
        req,
        IpcRequest::GetStats
            | IpcRequest::PollPickEvents
            | IpcRequest::GetLassoState
            | IpcRequest::PollPendingBundleSave
            | IpcRequest::PollPendingBundleLoad
    ) {
        return Ok(None);
    }

    if let Some(cmd) = core::to_viewer_cmd(req) {
        return Ok(Some(cmd));
    }
    if let Some(cmd) = terrain::to_viewer_cmd(req)? {
        return Ok(Some(cmd));
    }
    if let Some(cmd) = overlays::to_viewer_cmd(req) {
        return Ok(Some(cmd));
    }
    if let Some(cmd) = labels::to_viewer_cmd(req) {
        return Ok(Some(cmd));
    }

    Err(format!("Unhandled IPC request: {req:?}"))
}
