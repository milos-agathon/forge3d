use super::*;

#[test]
fn test_blend_mode_values() {
    assert_eq!(BlendMode::Normal.to_shader_value(), 0.0);
    assert_eq!(BlendMode::Multiply.to_shader_value(), 1.0);
    assert_eq!(BlendMode::Overlay.to_shader_value(), 2.0);
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
