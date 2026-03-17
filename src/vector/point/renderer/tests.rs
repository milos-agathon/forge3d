use super::*;
use crate::vector::api::VectorStyle;

#[test]
fn test_pack_simple_points() {
    let Some(device) = crate::core::gpu::create_device_for_test() else {
        return;
    };
    let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

    let points = vec![
        PointDef {
            position: Vec2::new(0.0, 0.0),
            style: VectorStyle {
                point_size: 4.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
        },
        PointDef {
            position: Vec2::new(1.0, 1.0),
            style: VectorStyle {
                point_size: 6.0,
                fill_color: [0.0, 1.0, 0.0, 1.0],
                ..Default::default()
            },
        },
    ];

    let instances = renderer.pack_points(&points).unwrap();
    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0].size, 4.0);
    assert_eq!(instances[0].color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(instances[1].size, 6.0);
    assert_eq!(instances[1].color, [0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_reject_invalid_point_size() {
    let Some(device) = crate::core::gpu::create_device_for_test() else {
        return;
    };
    let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

    let invalid_point = PointDef {
        position: Vec2::new(0.0, 0.0),
        style: VectorStyle {
            point_size: -1.0,
            ..Default::default()
        },
    };

    let result = renderer.pack_points(&[invalid_point]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("positive and finite"));
}

#[test]
fn test_reject_non_finite_coordinates() {
    let Some(device) = crate::core::gpu::create_device_for_test() else {
        return;
    };
    let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

    let invalid_point = PointDef {
        position: Vec2::new(f32::NAN, 0.0),
        style: VectorStyle::default(),
    };

    let result = renderer.pack_points(&[invalid_point]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("non-finite coordinates"));
}

#[test]
fn test_point_clustering() {
    let points = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(0.1, 0.1),
        Vec2::new(5.0, 5.0),
        Vec2::new(5.1, 5.1),
    ];

    let clusters = cluster_points(&points, 1.0);
    assert_eq!(clusters.len(), 2);
    assert_eq!(clusters[0].1, 2);
    assert_eq!(clusters[1].1, 2);
}

#[test]
fn test_no_clustering_when_all_far_apart() {
    let points = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(20.0, 20.0),
    ];

    let clusters = cluster_points(&points, 1.0);
    assert_eq!(clusters.len(), 3);
    assert!(clusters.iter().all(|(_, count)| *count == 1));
}
