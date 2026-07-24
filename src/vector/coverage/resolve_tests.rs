use super::math::rasterize_coverage_cpu;
use super::resolve::{resolve_coverage_cpu, resolve_shader_source};
use super::types::FillRule;
use super::CoverageGeometryBuilder;
use crate::vector::api::{PolygonDef, VectorStyle};
use glam::Vec2;

#[test]
fn resolve_shader_and_pinned_math_assemble_as_valid_wgsl() {
    let module = naga::front::wgsl::parse_str(&resolve_shader_source())
        .expect("combined LIMES resolve shader must parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("combined LIMES resolve shader must validate");
}

#[test]
fn later_layers_source_over_earlier_layers_once() {
    let mut builder = CoverageGeometryBuilder::new(1, 1).unwrap();
    builder
        .add_layer("back", FillRule::NonZero, [1.0, 0.0, 0.0, 0.5])
        .unwrap();
    builder
        .add_layer("front", FillRule::NonZero, [0.0, 0.0, 1.0, 0.5])
        .unwrap();
    let geometry = builder.finish().unwrap();
    let resolved = resolve_coverage_cpu(&geometry, &[1.0, 1.0]);
    assert_eq!(resolved, vec![[0.25, 0.0, 0.5, 0.75]]);
}

#[test]
fn same_layer_shared_edges_resolve_to_one_without_a_seam() {
    let mut builder = CoverageGeometryBuilder::new(10, 10).unwrap();
    let layer = builder
        .add_layer("mosaic", FillRule::NonZero, [1.0, 1.0, 1.0, 1.0])
        .unwrap();
    for row in 0..10 {
        for column in 0..10 {
            let position = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
                f64::from(column),
                f64::from(row),
                0.0,
            ));
            let x = position.x;
            let y = position.y;
            builder
                .push_polygon(
                    layer,
                    &PolygonDef {
                        exterior: vec![
                            Vec2::new(x, y),
                            Vec2::new(x + 1.0, y),
                            Vec2::new(x + 1.0, y + 1.0),
                            Vec2::new(x, y + 1.0),
                        ],
                        holes: Vec::new(),
                        style: VectorStyle::default(),
                    },
                )
                .unwrap();
        }
    }
    let geometry = builder.finish().unwrap();
    let coverage = rasterize_coverage_cpu(&geometry);
    let resolved = resolve_coverage_cpu(&geometry, &coverage);
    let max_deviation = resolved
        .iter()
        .map(|pixel| (1.0 - pixel[3]).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(max_deviation, 0.0);
}
