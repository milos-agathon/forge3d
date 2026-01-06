//! R-tree based collision detection for labels.
//!
//! Provides faster collision detection for many labels compared to grid-based approach.

use rstar::{RTree, RTreeObject, AABB};

/// A label bounding box for R-tree storage.
#[derive(Debug, Clone, Copy)]
pub struct LabelBounds {
    /// Label ID for reference.
    pub id: u64,
    /// Bounding box [x0, y0, x1, y1].
    pub bounds: [f32; 4],
}

impl RTreeObject for LabelBounds {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.bounds[0], self.bounds[1]],
            [self.bounds[2], self.bounds[3]],
        )
    }
}

/// R-tree based collision detection for labels.
pub struct LabelRTree {
    tree: RTree<LabelBounds>,
    width: f32,
    height: f32,
}

impl LabelRTree {
    /// Create a new R-tree for the given screen dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            tree: RTree::new(),
            width: width as f32,
            height: height as f32,
        }
    }

    /// Clear all stored bounds.
    pub fn clear(&mut self) {
        self.tree = RTree::new();
    }

    /// Try to insert a bounding box. Returns true if no collision, false otherwise.
    pub fn try_insert(&mut self, id: u64, bounds: [f32; 4]) -> bool {
        let [x0, y0, x1, y1] = bounds;

        // Clamp to screen bounds
        let x0 = x0.max(0.0);
        let y0 = y0.max(0.0);
        let x1 = x1.min(self.width);
        let y1 = y1.min(self.height);

        // Check if completely off-screen
        if x0 >= x1 || y0 >= y1 {
            return false;
        }

        let clamped_bounds = [x0, y0, x1, y1];
        let envelope = AABB::from_corners([x0, y0], [x1, y1]);

        // Check for collisions with existing bounds
        let has_collision = self
            .tree
            .locate_in_envelope_intersecting(&envelope)
            .any(|existing| rects_overlap(clamped_bounds, existing.bounds));

        if has_collision {
            return false;
        }

        // No collision, insert
        self.tree.insert(LabelBounds {
            id,
            bounds: clamped_bounds,
        });

        true
    }

    /// Check if a bounding box would collide without inserting.
    pub fn check_collision(&self, bounds: [f32; 4]) -> bool {
        let [x0, y0, x1, y1] = bounds;

        let x0 = x0.max(0.0);
        let y0 = y0.max(0.0);
        let x1 = x1.min(self.width);
        let y1 = y1.min(self.height);

        if x0 >= x1 || y0 >= y1 {
            return true; // Off-screen counts as collision
        }

        let clamped_bounds = [x0, y0, x1, y1];
        let envelope = AABB::from_corners([x0, y0], [x1, y1]);

        self.tree
            .locate_in_envelope_intersecting(&envelope)
            .any(|existing| rects_overlap(clamped_bounds, existing.bounds))
    }

    /// Get all bounds that intersect with the given rectangle.
    pub fn query_intersecting(&self, bounds: [f32; 4]) -> Vec<LabelBounds> {
        let [x0, y0, x1, y1] = bounds;
        let envelope = AABB::from_corners([x0, y0], [x1, y1]);

        self.tree
            .locate_in_envelope_intersecting(&envelope)
            .copied()
            .collect()
    }

    /// Get the number of stored bounds.
    pub fn len(&self) -> usize {
        self.tree.size()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tree.size() == 0
    }

    /// Update screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as f32;
        self.height = height as f32;
        self.clear();
    }
}

/// Check if two axis-aligned rectangles overlap.
#[inline]
fn rects_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_collision() {
        let mut tree = LabelRTree::new(100, 100);
        assert!(tree.try_insert(1, [0.0, 0.0, 10.0, 10.0]));
        assert!(tree.try_insert(2, [50.0, 50.0, 60.0, 60.0]));
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_collision() {
        let mut tree = LabelRTree::new(100, 100);
        assert!(tree.try_insert(1, [0.0, 0.0, 20.0, 20.0]));
        assert!(!tree.try_insert(2, [10.0, 10.0, 30.0, 30.0])); // Overlaps
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut tree = LabelRTree::new(100, 100);
        assert!(tree.try_insert(1, [0.0, 0.0, 50.0, 50.0]));
        tree.clear();
        assert!(tree.try_insert(2, [0.0, 0.0, 50.0, 50.0])); // Should work after clear
    }

    #[test]
    fn test_query() {
        let mut tree = LabelRTree::new(100, 100);
        tree.try_insert(1, [0.0, 0.0, 20.0, 20.0]);
        tree.try_insert(2, [50.0, 50.0, 70.0, 70.0]);

        let results = tree.query_intersecting([10.0, 10.0, 60.0, 60.0]);
        assert_eq!(results.len(), 2);
    }
}
