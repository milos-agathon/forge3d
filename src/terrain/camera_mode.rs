//! Device-free camera-mode frame rules shared by the terrain renderer and the
//! frame compiler, so both compile without the extension-module feature.

/// True when the camera mode carries the `zup` option (e.g. `"mesh:zup"`).
///
/// Mesh-mode terrain lives in the world XY plane with heights along +Z, but
/// the legacy orbit camera is parameterized around +Y with `up = Vec3::Y` —
/// a grid axis — so oblique views render rolled and north-south inverted.
/// `zup` opts into a Z-up orbit (theta = polar angle from +Z, 0 looks straight
/// down; phi = azimuth in the terrain plane) with `up = Vec3::Z`, matching the
/// interactive viewer's terrain semantics. Kept as an opt-in suffix so
/// existing `"mesh"` callers keep byte-identical output.
pub(crate) fn is_zup_camera_mode(camera_mode: &str) -> bool {
    camera_mode
        .split(':')
        .skip(1)
        .any(|part| part.trim().eq_ignore_ascii_case("zup"))
}

/// True when a Z-up camera mode also carries the `north` option
/// (e.g. `"mesh:zup:north"`).
///
/// Mesh terrain rows run along +Y with row 0 at the north edge, so with east
/// = +X and up = +Z the world is left-handed and views render mirrored
/// north-south. `north` reads `cam_target` and `cam_phi_deg` in a
/// right-handed geographic frame (+X east, +Y north, +Z up) and folds the Y
/// reflection into the view matrix; heights, UVs and every texture stay as
/// they are. Opt-in so ORBIS and existing `zup` callers are unchanged.
pub(crate) fn is_north_up_camera_mode(camera_mode: &str) -> bool {
    is_zup_camera_mode(camera_mode)
        && camera_mode
            .split(':')
            .skip(1)
            .any(|part| part.trim().eq_ignore_ascii_case("north"))
}

/// Geographic (north-up) camera point to the mesh world frame for `north` modes.
pub(crate) fn north_up_to_world(point: [f32; 3]) -> [f32; 3] {
    [point[0], -point[1], point[2]]
}
