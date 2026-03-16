pub(super) fn default_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

pub(super) fn default_intensity() -> f32 {
    1.0
}

pub(super) fn default_phi() -> f32 {
    135.0
}

pub(super) fn default_theta() -> f32 {
    45.0
}

pub(super) fn default_radius() -> f32 {
    1000.0
}

pub(super) fn default_fov() -> f32 {
    55.0
}

pub(super) fn default_sun_azimuth() -> f32 {
    135.0
}

pub(super) fn default_sun_elevation() -> f32 {
    35.0
}

pub(super) fn default_sun_intensity() -> f32 {
    1.0
}

pub(super) fn default_primitive() -> String {
    "triangles".to_string()
}

pub(super) fn default_drape_offset() -> f32 {
    0.5
}

pub(super) fn default_opacity() -> f32 {
    1.0
}

pub(super) fn default_depth_bias() -> f32 {
    0.1
}

pub(super) fn default_line_width() -> f32 {
    2.0
}

pub(super) fn default_point_size() -> f32 {
    5.0
}

pub(super) fn default_max_points() -> u64 {
    5_000_000
}

pub(super) fn default_true() -> bool {
    true
}

pub(super) fn default_oit_mode() -> String {
    "auto".to_string()
}
