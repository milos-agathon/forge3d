//! Axis-aligned bounding box for 2D spatial operations.

use glam::Vec2;

/// Axis-aligned bounding box for 2D primitives.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    /// Create a new AABB with explicit min/max bounds.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Compute AABB from a set of points.
    pub fn from_points(points: &[Vec2]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min = points[0];
        let mut max = points[0];

        for &point in points.iter().skip(1) {
            min = min.min(point);
            max = max.max(point);
        }

        Some(Self { min, max })
    }

    /// Get the center point of the AABB.
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Get the dimensions of the AABB.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Compute the area of the AABB.
    pub fn area(&self) -> f32 {
        let size = self.size();
        size.x * size.y
    }

    /// Test if a point is contained within the AABB.
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Test if this AABB intersects another.
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Compute the union of two AABBs.
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_from_points() {
        let points = vec![
            Vec2::new(1.0, 2.0),
            Vec2::new(-1.0, 3.0),
            Vec2::new(2.0, -1.0),
        ];

        let aabb = AABB::from_points(&points).unwrap();
        assert_eq!(aabb.min, Vec2::new(-1.0, -1.0));
        assert_eq!(aabb.max, Vec2::new(2.0, 3.0));
        assert_eq!(aabb.center(), Vec2::new(0.5, 1.0));
        assert_eq!(aabb.size(), Vec2::new(3.0, 4.0));
        assert_eq!(aabb.area(), 12.0);
    }

    #[test]
    fn test_aabb_intersection() {
        let aabb1 = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let aabb2 = AABB::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let aabb3 = AABB::new(Vec2::new(3.0, 3.0), Vec2::new(4.0, 4.0));

        assert!(aabb1.intersects(&aabb2));
        assert!(aabb2.intersects(&aabb1));
        assert!(!aabb1.intersects(&aabb3));
        assert!(!aabb3.intersects(&aabb1));
    }
}
