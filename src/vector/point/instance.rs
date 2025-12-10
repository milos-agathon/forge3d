use crate::core::error::RenderError;
use crate::vector::api::PointDef;
use crate::vector::data::{validate_point_instances, PointInstance};
use glam::Vec2;

/// Convert point definitions to point instances
pub fn pack_points(points: &[PointDef]) -> Result<Vec<PointInstance>, RenderError> {
    let mut instances = Vec::with_capacity(points.len());

    for point in points {
        // Validate point data
        if !point.position.x.is_finite() || !point.position.y.is_finite() {
            return Err(RenderError::Upload(format!(
                "Point has non-finite coordinates: ({}, {})",
                point.position.x, point.position.y
            )));
        }

        if point.style.point_size <= 0.0 || !point.style.point_size.is_finite() {
            return Err(RenderError::Upload(format!(
                "Point size must be positive and finite, got {}",
                point.style.point_size
            )));
        }

        instances.push(PointInstance {
            position: [point.position.x, point.position.y],
            size: point.style.point_size,
            color: point.style.fill_color,
            rotation: 0.0,         // Default rotation (H21)
            uv_offset: [0.0, 0.0], // Default UV offset (H21)
            _pad: 0.0,
        });
    }

    // Validate packed instances
    let validation_result = validate_point_instances(&instances);
    if !validation_result.is_valid {
        return Err(RenderError::Upload(
            validation_result
                .error_message
                .unwrap_or_else(|| "Point instance validation failed".to_string()),
        ));
    }

    Ok(instances)
}

/// Calculate point clustering for high-density datasets (H20)
pub fn cluster_points(points: &[Vec2], cluster_radius: f32) -> Vec<(Vec2, u32)> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut used = vec![false; points.len()];

    for (i, &point) in points.iter().enumerate() {
        if used[i] {
            continue;
        }

        let mut cluster_center = point;
        let mut cluster_count = 1;
        used[i] = true;

        // Find nearby points to cluster
        for (j, &other_point) in points.iter().enumerate().skip(i + 1) {
            if used[j] {
                continue;
            }

            let distance = (other_point - point).length();
            if distance <= cluster_radius {
                cluster_center = (cluster_center * cluster_count as f32 + other_point)
                    / (cluster_count + 1) as f32;
                cluster_count += 1;
                used[j] = true;
            }
        }

        clusters.push((cluster_center, cluster_count));
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;

    #[test]
    fn test_pack_simple_points() {
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

        let instances = pack_points(&points).unwrap();

        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].size, 4.0);
        assert_eq!(instances[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(instances[1].size, 6.0);
        assert_eq!(instances[1].color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_reject_invalid_point_size() {
        let invalid_point = PointDef {
            position: Vec2::new(0.0, 0.0),
            style: VectorStyle {
                point_size: -1.0, // Invalid negative size
                ..Default::default()
            },
        };

        let result = pack_points(&[invalid_point]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("positive and finite"));
    }

    #[test]
    fn test_reject_non_finite_coordinates() {
        let invalid_point = PointDef {
            position: Vec2::new(f32::NAN, 0.0),
            style: VectorStyle::default(),
        };

        let result = pack_points(&[invalid_point]);
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
            Vec2::new(0.1, 0.1), // Close to first point
            Vec2::new(5.0, 5.0), // Far away
            Vec2::new(5.1, 5.1), // Close to third point
        ];

        let clusters = cluster_points(&points, 1.0);

        // Should create 2 clusters
        assert_eq!(clusters.len(), 2);

        // First cluster should have 2 points
        assert_eq!(clusters[0].1, 2);
        // Second cluster should have 2 points
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

        // Should create 3 clusters (no clustering)
        assert_eq!(clusters.len(), 3);
        assert!(clusters.iter().all(|(_, count)| *count == 1));
    }
}
