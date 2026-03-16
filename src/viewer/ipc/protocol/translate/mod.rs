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

    let cmd = core::to_viewer_cmd(req)
        .or_else(|| terrain::to_viewer_cmd(req))
        .or_else(|| overlays::to_viewer_cmd(req))
        .or_else(|| labels::to_viewer_cmd(req));

    cmd.map(Some)
        .ok_or_else(|| format!("Unhandled IPC request: {req:?}"))
}
