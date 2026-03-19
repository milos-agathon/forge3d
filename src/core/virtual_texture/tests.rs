use super::*;

#[test]
fn test_virtual_texture_config() {
    let config = VirtualTextureConfig::default();

    assert_eq!(config.width, 16384);
    assert_eq!(config.height, 16384);
    assert_eq!(config.tile_size, 128);
    assert!(config.use_feedback);
}

#[test]
fn test_tile_id_to_page_index() {
    let config = VirtualTextureConfig {
        width: 1024,
        height: 1024,
        tile_size: 128,
        ..Default::default()
    };

    let pages_x = (config.width + config.tile_size - 1) / config.tile_size;
    let expected_index = (2 * pages_x + 1) as usize;
    assert_eq!(expected_index, 17);
}

#[test]
fn test_page_table_entry() {
    let entry = PageTableEntry {
        atlas_u: 0.5,
        atlas_v: 0.25,
        is_resident: 1,
        mip_bias: 0.0,
    };

    assert_eq!(entry.atlas_u, 0.5);
    assert_eq!(entry.atlas_v, 0.25);
    assert_eq!(entry.is_resident, 1);
    assert_eq!(entry.mip_bias, 0.0);
}

#[test]
fn test_camera_info() {
    let camera = CameraInfo {
        position: [0.0, 100.0, 0.0],
        direction: [0.0, -1.0, 0.0],
        fov_degrees: 45.0,
        aspect_ratio: 16.0 / 9.0,
        near_plane: 0.1,
        far_plane: 1000.0,
    };

    assert_eq!(camera.position[1], 100.0);
    assert_eq!(camera.fov_degrees, 45.0);
    assert!((camera.aspect_ratio - 16.0 / 9.0).abs() < f32::EPSILON);
}
