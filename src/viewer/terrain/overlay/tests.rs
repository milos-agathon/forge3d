use super::*;

fn approx_eq(a: f32, b: f32) {
    assert!((a - b).abs() <= 1e-6, "expected {b}, got {a}");
}

#[test]
fn test_overlay_layer_default() {
    let layer = OverlayLayer::default();
    assert!(layer.name.is_empty());
    assert_eq!(layer.opacity, 1.0);
    assert!(layer.visible);
    assert_eq!(layer.z_order, 0);
    assert_eq!(layer.blend_mode, BlendMode::Normal);
}

#[test]
fn test_bilinear_sampling_round_trips_texel_centers() {
    let rgba = [
        10u8, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
    ];

    let top_left = super::sampling::sample_bilinear(&rgba, 2, 2, 0.25, 0.25, 1.0);
    approx_eq(top_left[0], 10.0 / 255.0);
    approx_eq(top_left[1], 20.0 / 255.0);
    approx_eq(top_left[2], 30.0 / 255.0);
    approx_eq(top_left[3], 1.0);

    let bottom_right = super::sampling::sample_bilinear(&rgba, 2, 2, 0.75, 0.75, 1.0);
    approx_eq(bottom_right[0], 100.0 / 255.0);
    approx_eq(bottom_right[1], 110.0 / 255.0);
    approx_eq(bottom_right[2], 120.0 / 255.0);
    approx_eq(bottom_right[3], 1.0);
}
