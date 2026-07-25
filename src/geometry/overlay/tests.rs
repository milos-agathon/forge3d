use super::*;

fn square(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> MultiPolygon {
    MultiPolygon(vec![Polygon {
        exterior: vec![
            Point::new(min_x, min_y),
            Point::new(max_x, min_y),
            Point::new(max_x, max_y),
            Point::new(min_x, max_y),
            Point::new(min_x, min_y),
        ],
        holes: Vec::new(),
    }])
}

fn polygonal_area(value: &MultiPolygon) -> f64 {
    value
        .0
        .iter()
        .map(|polygon| {
            let ring_area = |ring: &Ring| {
                signed_area2(
                    &ring
                        .iter()
                        .copied()
                        .map(Point::as_array)
                        .collect::<Vec<_>>(),
                )
                .abs()
                    * 0.5
            };
            ring_area(&polygon.exterior) - polygon.holes.iter().map(ring_area).sum::<f64>()
        })
        .sum()
}

fn donut() -> MultiPolygon {
    let exterior = square(0.0, 0.0, 10.0, 10.0).0[0].exterior.clone();
    let mut hole = square(3.0, 3.0, 7.0, 7.0).0[0].exterior.clone();
    hole.reverse();
    MultiPolygon(vec![Polygon {
        exterior,
        holes: vec![hole],
    }])
}

#[test]
fn overlapping_square_operations_are_valid_and_deterministic() {
    let left = square(0.0, 0.0, 2.0, 2.0);
    let right = square(1.0, 1.0, 3.0, 3.0);
    for operation in [
        BooleanOp::Union,
        BooleanOp::Intersection,
        BooleanOp::Difference,
        BooleanOp::SymmetricDifference,
    ] {
        let first = overlay(&left, &right, operation).unwrap();
        let second = overlay(&left, &right, operation).unwrap();
        assert_eq!(first, second);
        assert!(is_valid_polygonal(&first.geometry).valid);
        assert!(first.max_snap_motion <= first.snap_motion_bound);
    }
}

#[test]
fn identity_disjoint_shared_edge_and_containment_cases() {
    let outer = square(0.0, 0.0, 4.0, 4.0);
    let inner = square(1.0, 1.0, 3.0, 3.0);
    let disjoint = square(5.0, 0.0, 6.0, 1.0);
    let adjacent = square(4.0, 0.0, 5.0, 4.0);

    assert_eq!(
        overlay(&outer, &outer, BooleanOp::Union)
            .unwrap()
            .geometry
            .0
            .len(),
        1
    );
    assert!(overlay(&outer, &outer, BooleanOp::Difference)
        .unwrap()
        .geometry
        .0
        .is_empty());
    assert!(overlay(&outer, &disjoint, BooleanOp::Intersection)
        .unwrap()
        .geometry
        .0
        .is_empty());
    assert_eq!(
        overlay(&outer, &disjoint, BooleanOp::Union)
            .unwrap()
            .geometry
            .0
            .len(),
        2
    );
    let merged = overlay(&outer, &adjacent, BooleanOp::Union).unwrap();
    assert_eq!(merged.geometry.0.len(), 1);
    let cutout = overlay(&outer, &inner, BooleanOp::Difference).unwrap();
    assert_eq!(cutout.geometry.0.len(), 1);
    assert_eq!(cutout.geometry.0[0].holes.len(), 1);
    assert!(is_valid_polygonal(&cutout.geometry).valid);
}

#[test]
fn non_finite_input_is_rejected_before_sweep() {
    let invalid = square(0.0, 0.0, f64::NAN, 1.0);
    let error = overlay(&invalid, &square(0.0, 0.0, 1.0, 1.0), BooleanOp::Union).unwrap_err();
    assert!(error.0.contains("NaN or infinity"));
}

#[test]
fn deterministic_triangle_corpus_stays_valid() {
    fn triangle(x: f64, y: f64, width: f64, height: f64) -> MultiPolygon {
        MultiPolygon(vec![Polygon {
            exterior: vec![
                Point::new(x, y),
                Point::new(x + width, y),
                Point::new(x + width * 0.25, y + height),
                Point::new(x, y),
            ],
            holes: Vec::new(),
        }])
    }
    let mut state = 0x4555_434c_4944_4541u64;
    for _ in 0..100 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state & 0xff) as f64 / 64.0 - 2.0;
        let y = ((state >> 8) & 0xff) as f64 / 64.0 - 2.0;
        let left = triangle(0.0, 0.0, 2.0, 2.0);
        let right = triangle(x, y, 1.5, 2.5);
        for operation in [
            BooleanOp::Union,
            BooleanOp::Intersection,
            BooleanOp::Difference,
            BooleanOp::SymmetricDifference,
        ] {
            let result = overlay(&left, &right, operation).unwrap_or_else(|error| {
                panic!("triangle corpus failed at ({x}, {y}) {operation:?}: {error}")
            });
            assert!(is_valid_polygonal(&result.geometry).valid);
        }
    }
}

#[test]
fn input_holes_and_multipolygons_work_for_all_boolean_operations() {
    let left = donut();
    let right = square(5.0, -1.0, 11.0, 11.0);
    let mut results = Vec::new();
    for operation in [
        BooleanOp::Union,
        BooleanOp::Intersection,
        BooleanOp::Difference,
        BooleanOp::SymmetricDifference,
    ] {
        let result = overlay(&left, &right, operation);
        assert!(result.is_ok(), "{result:?}");
        let Ok(result) = result else {
            return;
        };
        assert!(is_valid_polygonal(&result.geometry).valid);
        assert!(result.max_snap_motion <= result.snap_motion_bound);
        results.push(result.geometry);
    }
    assert_eq!(polygonal_area(&results[0]), 114.0);
    assert_eq!(polygonal_area(&results[1]), 42.0);
    assert_eq!(polygonal_area(&results[2]), 42.0);
    assert_eq!(polygonal_area(&results[3]), 72.0);
    assert_eq!(
        polygonal_area(&results[0]) + polygonal_area(&results[1]),
        polygonal_area(&left) + polygonal_area(&right)
    );

    let multi = MultiPolygon(vec![
        square(-3.0, -3.0, -1.0, -1.0).0[0].clone(),
        square(12.0, 12.0, 14.0, 14.0).0[0].clone(),
    ]);
    let expected_areas = [92.0, 0.0, 8.0, 92.0];
    for (operation, expected_area) in [
        BooleanOp::Union,
        BooleanOp::Intersection,
        BooleanOp::Difference,
        BooleanOp::SymmetricDifference,
    ]
    .into_iter()
    .zip(expected_areas)
    {
        let result = overlay(&multi, &left, operation);
        assert!(result.is_ok(), "{result:?}");
        let Ok(result) = result else {
            return;
        };
        assert!(is_valid_polygonal(&result.geometry).valid);
        assert_eq!(polygonal_area(&result.geometry), expected_area);
    }
}
