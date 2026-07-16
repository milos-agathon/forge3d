use glam::DVec3;

use crate::viewer::camera_controller::{validate_world_point, CoordRole};
use crate::viewer::Viewer;

pub(super) fn validate_points<I>(
    anchor: &crate::camera::Anchor,
    role: CoordRole,
    points: I,
) -> Result<(), String>
where
    I: IntoIterator<Item = DVec3>,
{
    points
        .into_iter()
        .try_for_each(|point| validate_world_point(role, point, anchor).map_err(|e| e.to_string()))
}

pub(super) fn distinct_value<T: Copy + PartialEq>(
    slot: &mut Option<T>,
    value: T,
    name: &str,
) -> Result<(), String> {
    if slot.is_some_and(|existing| existing != value) {
        return Err(format!("ambiguous prospective batch: conflicting {name}"));
    }
    *slot = Some(value);
    Ok(())
}

pub(super) fn point_repack_failpoint_blocks_publish(viewer: &Viewer, focus: DVec3) -> bool {
    if std::env::var("RUN_M06_VIEWER_CI").as_deref() != Ok("1")
        || std::env::var("FORGE3D_M06_POINT_REPACK_FAILPOINT").as_deref()
            != Ok("before_anchor_publish")
    {
        return false;
    }
    let candidate =
        crate::viewer::camera_controller::prospective_anchor(&viewer.camera_anchor, focus);
    candidate.origin() != viewer.camera_anchor.origin()
}
