use crate::viewer::viewer_enums::ViewerCmd;

use super::super::request::IpcRequest;

pub(super) fn to_viewer_cmd(req: &IpcRequest) -> Option<ViewerCmd> {
    match req {
        IpcRequest::LoadOverlay {
            name,
            path,
            extent,
            opacity,
            z_order,
        } => Some(ViewerCmd::LoadOverlay {
            name: name.clone(),
            path: path.clone(),
            extent: *extent,
            opacity: *opacity,
            z_order: *z_order,
        }),
        IpcRequest::RemoveOverlay { id } => Some(ViewerCmd::RemoveOverlay { id: *id }),
        IpcRequest::SetOverlayVisible { id, visible } => Some(ViewerCmd::SetOverlayVisible {
            id: *id,
            visible: *visible,
        }),
        IpcRequest::SetOverlayOpacity { id, opacity } => Some(ViewerCmd::SetOverlayOpacity {
            id: *id,
            opacity: *opacity,
        }),
        IpcRequest::SetGlobalOverlayOpacity { opacity } => {
            Some(ViewerCmd::SetGlobalOverlayOpacity { opacity: *opacity })
        }
        IpcRequest::SetOverlaysEnabled { enabled } => {
            Some(ViewerCmd::SetOverlaysEnabled { enabled: *enabled })
        }
        IpcRequest::SetOverlaySolid { solid } => Some(ViewerCmd::SetOverlaySolid { solid: *solid }),
        IpcRequest::ListOverlays => Some(ViewerCmd::ListOverlays),
        IpcRequest::AddVectorOverlay {
            name,
            vertices,
            indices,
            primitive,
            drape,
            drape_offset,
            opacity,
            depth_bias,
            line_width,
            point_size,
            z_order,
        } => Some(ViewerCmd::AddVectorOverlay {
            name: name.clone(),
            vertices: vertices.clone(),
            indices: indices.clone(),
            primitive: primitive.clone(),
            drape: *drape,
            drape_offset: *drape_offset,
            opacity: *opacity,
            depth_bias: *depth_bias,
            line_width: *line_width,
            point_size: *point_size,
            z_order: *z_order,
        }),
        IpcRequest::RemoveVectorOverlay { id } => Some(ViewerCmd::RemoveVectorOverlay { id: *id }),
        IpcRequest::SetVectorOverlayVisible { id, visible } => {
            Some(ViewerCmd::SetVectorOverlayVisible {
                id: *id,
                visible: *visible,
            })
        }
        IpcRequest::SetVectorOverlayOpacity { id, opacity } => {
            Some(ViewerCmd::SetVectorOverlayOpacity {
                id: *id,
                opacity: *opacity,
            })
        }
        IpcRequest::ListVectorOverlays => Some(ViewerCmd::ListVectorOverlays),
        IpcRequest::SetVectorOverlaysEnabled { enabled } => {
            Some(ViewerCmd::SetVectorOverlaysEnabled { enabled: *enabled })
        }
        IpcRequest::SetGlobalVectorOverlayOpacity { opacity } => {
            Some(ViewerCmd::SetGlobalVectorOverlayOpacity { opacity: *opacity })
        }
        IpcRequest::LoadPointCloud {
            path,
            point_size,
            max_points,
            color_mode,
        } => Some(ViewerCmd::LoadPointCloud {
            path: path.clone(),
            point_size: *point_size,
            max_points: *max_points,
            color_mode: color_mode.clone(),
        }),
        IpcRequest::ClearPointCloud => Some(ViewerCmd::ClearPointCloud),
        IpcRequest::SetPointCloudParams {
            point_size,
            visible,
            color_mode,
            phi,
            theta,
            radius,
        } => Some(ViewerCmd::SetPointCloudParams {
            point_size: *point_size,
            visible: *visible,
            color_mode: color_mode.clone(),
            phi: *phi,
            theta: *theta,
            radius: *radius,
        }),
        _ => None,
    }
}
