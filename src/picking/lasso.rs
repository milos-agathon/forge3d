// src/picking/lasso.rs
// Lasso and box selection for multi-feature picking
// Part of Plan 3: Premium - Unified Picking with BVH + Python Callbacks

/// A 2D point in screen space
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Lasso selection state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LassoState {
    /// No lasso active
    Inactive,
    /// Currently drawing lasso
    Drawing,
    /// Lasso complete, ready to query
    Complete,
}

/// Configuration for lasso selection
#[derive(Debug, Clone)]
pub struct LassoConfig {
    /// Minimum points for valid lasso
    pub min_points: usize,
    /// Maximum points before auto-simplification
    pub max_points: usize,
    /// Distance threshold for point simplification
    pub simplify_threshold: f32,
    /// Lasso outline color (RGBA)
    pub outline_color: [f32; 4],
    /// Lasso fill color (RGBA)
    pub fill_color: [f32; 4],
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            min_points: 3,
            max_points: 1000,
            simplify_threshold: 5.0,
            outline_color: [1.0, 1.0, 0.0, 1.0], // Yellow
            fill_color: [1.0, 1.0, 0.0, 0.2],    // Yellow transparent
        }
    }
}

/// Lasso selection manager
#[derive(Debug)]
pub struct LassoSelection {
    config: LassoConfig,
    state: LassoState,
    points: Vec<Point2D>,
    bounding_box: Option<BoundingBox2D>,
}

/// 2D bounding box
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox2D {
    pub min: Point2D,
    pub max: Point2D,
}

impl BoundingBox2D {
    pub fn new(min: Point2D, max: Point2D) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Point2D]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Some(Self {
            min: Point2D::new(min_x, min_y),
            max: Point2D::new(max_x, max_y),
        })
    }

    pub fn contains(&self, point: Point2D) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
            && point.y >= self.min.y && point.y <= self.max.y
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
}

impl LassoSelection {
    /// Create new lasso selection
    pub fn new() -> Self {
        Self {
            config: LassoConfig::default(),
            state: LassoState::Inactive,
            points: Vec::new(),
            bounding_box: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LassoConfig) -> Self {
        Self {
            config,
            state: LassoState::Inactive,
            points: Vec::new(),
            bounding_box: None,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &LassoConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: LassoConfig) {
        self.config = config;
    }

    /// Get current state
    pub fn state(&self) -> &LassoState {
        &self.state
    }

    /// Get lasso points
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> Option<&BoundingBox2D> {
        self.bounding_box.as_ref()
    }

    /// Start drawing lasso at position
    pub fn begin(&mut self, x: f32, y: f32) {
        self.points.clear();
        self.points.push(Point2D::new(x, y));
        self.state = LassoState::Drawing;
        self.bounding_box = None;
    }

    /// Add point to lasso
    pub fn add_point(&mut self, x: f32, y: f32) {
        if self.state != LassoState::Drawing {
            return;
        }

        // Simplify if we have too many points
        if self.points.len() >= self.config.max_points {
            self.simplify();
        }

        // Skip if too close to previous point
        if let Some(last) = self.points.last() {
            let dx = x - last.x;
            let dy = y - last.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < self.config.simplify_threshold * 0.5 {
                return;
            }
        }

        self.points.push(Point2D::new(x, y));
    }

    /// Complete the lasso
    pub fn complete(&mut self) {
        if self.state != LassoState::Drawing {
            return;
        }

        if self.points.len() >= self.config.min_points {
            self.state = LassoState::Complete;
            self.bounding_box = BoundingBox2D::from_points(&self.points);
        } else {
            self.cancel();
        }
    }

    /// Cancel the lasso
    pub fn cancel(&mut self) {
        self.points.clear();
        self.state = LassoState::Inactive;
        self.bounding_box = None;
    }

    /// Simplify the lasso polygon using Douglas-Peucker algorithm
    fn simplify(&mut self) {
        if self.points.len() < 10 {
            return;
        }

        let simplified = douglas_peucker(&self.points, self.config.simplify_threshold);
        self.points = simplified;
    }

    /// Check if a screen point is inside the lasso polygon
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        if self.state != LassoState::Complete || self.points.len() < 3 {
            return false;
        }

        let point = Point2D::new(x, y);

        // Quick bounding box check first
        if let Some(bb) = &self.bounding_box {
            if !bb.contains(point) {
                return false;
            }
        }

        // Point-in-polygon test (ray casting)
        point_in_polygon(point, &self.points)
    }

    /// Get all feature IDs that are inside the lasso
    /// Takes a function that returns screen position for each feature ID
    pub fn get_selected_features<F>(&self, feature_screen_positions: F) -> Vec<u32>
    where
        F: Fn(u32) -> Option<(f32, f32)>,
    {
        if self.state != LassoState::Complete {
            return Vec::new();
        }

        let mut selected = Vec::new();

        // Closure-based iteration keeps this helper decoupled from layer storage.
        for id in 1..10000u32 {
            if let Some((x, y)) = feature_screen_positions(id) {
                if self.contains_point(x, y) {
                    selected.push(id);
                }
            } else {
                // Stop when we've exhausted valid IDs
                break;
            }
        }

        selected
    }

    /// Test multiple points against lasso, return indices of contained points
    pub fn test_points(&self, points: &[(f32, f32)]) -> Vec<usize> {
        if self.state != LassoState::Complete {
            return Vec::new();
        }

        points
            .iter()
            .enumerate()
            .filter(|(_, (x, y))| self.contains_point(*x, *y))
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for LassoSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-polygon test using ray casting algorithm
fn point_in_polygon(point: Point2D, polygon: &[Point2D]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.len();

    let mut j = n - 1;
    for i in 0..n {
        let pi = &polygon[i];
        let pj = &polygon[j];

        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Douglas-Peucker line simplification algorithm
fn douglas_peucker(points: &[Point2D], epsilon: f32) -> Vec<Point2D> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with the maximum distance from the line between first and last
    let first = &points[0];
    let last = &points[points.len() - 1];

    let mut max_dist = 0.0f32;
    let mut max_idx = 0usize;

    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let dist = perpendicular_distance(point, first, last);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon {
        let mut left = douglas_peucker(&points[..=max_idx], epsilon);
        let right = douglas_peucker(&points[max_idx..], epsilon);

        // Remove duplicate point at junction
        left.pop();
        left.extend(right);
        left
    } else {
        // Keep only endpoints
        vec![points[0], points[points.len() - 1]]
    }
}

/// Calculate perpendicular distance from point to line segment
fn perpendicular_distance(point: &Point2D, line_start: &Point2D, line_end: &Point2D) -> f32 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;

    let line_len_sq = dx * dx + dy * dy;

    if line_len_sq < 1e-10 {
        // Line segment is actually a point
        let px = point.x - line_start.x;
        let py = point.y - line_start.y;
        return (px * px + py * py).sqrt();
    }

    // Calculate perpendicular distance using cross product
    let numerator = ((line_end.y - line_start.y) * point.x
        - (line_end.x - line_start.x) * point.y
        + line_end.x * line_start.y
        - line_end.y * line_start.x)
        .abs();

    numerator / line_len_sq.sqrt()
}

/// Box selection (simpler alternative to lasso)
#[derive(Debug)]
pub struct BoxSelection {
    start: Option<Point2D>,
    end: Option<Point2D>,
    is_active: bool,
}

impl BoxSelection {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
            is_active: false,
        }
    }

    pub fn begin(&mut self, x: f32, y: f32) {
        self.start = Some(Point2D::new(x, y));
        self.end = Some(Point2D::new(x, y));
        self.is_active = true;
    }

    pub fn update(&mut self, x: f32, y: f32) {
        if self.is_active {
            self.end = Some(Point2D::new(x, y));
        }
    }

    pub fn complete(&mut self) -> Option<BoundingBox2D> {
        if !self.is_active {
            return None;
        }

        self.is_active = false;

        match (self.start, self.end) {
            (Some(s), Some(e)) => {
                let min = Point2D::new(s.x.min(e.x), s.y.min(e.y));
                let max = Point2D::new(s.x.max(e.x), s.y.max(e.y));
                Some(BoundingBox2D::new(min, max))
            }
            _ => None,
        }
    }

    pub fn cancel(&mut self) {
        self.start = None;
        self.end = None;
        self.is_active = false;
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn current_box(&self) -> Option<BoundingBox2D> {
        match (self.start, self.end) {
            (Some(s), Some(e)) => {
                let min = Point2D::new(s.x.min(e.x), s.y.min(e.y));
                let max = Point2D::new(s.x.max(e.x), s.y.max(e.y));
                Some(BoundingBox2D::new(min, max))
            }
            _ => None,
        }
    }
}

impl Default for BoxSelection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon() {
        // Simple square
        let polygon = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.0, 10.0),
        ];

        assert!(point_in_polygon(Point2D::new(5.0, 5.0), &polygon));
        assert!(!point_in_polygon(Point2D::new(15.0, 5.0), &polygon));
        assert!(!point_in_polygon(Point2D::new(-5.0, 5.0), &polygon));
    }

    #[test]
    fn test_lasso_workflow() {
        let mut lasso = LassoSelection::new();

        // Begin lasso
        lasso.begin(0.0, 0.0);
        assert_eq!(*lasso.state(), LassoState::Drawing);

        // Add points
        lasso.add_point(10.0, 0.0);
        lasso.add_point(10.0, 10.0);
        lasso.add_point(0.0, 10.0);

        // Complete
        lasso.complete();
        assert_eq!(*lasso.state(), LassoState::Complete);

        // Test containment
        assert!(lasso.contains_point(5.0, 5.0));
        assert!(!lasso.contains_point(15.0, 5.0));
    }

    #[test]
    fn test_box_selection() {
        let mut box_sel = BoxSelection::new();

        box_sel.begin(10.0, 10.0);
        box_sel.update(50.0, 50.0);

        let bbox = box_sel.complete();
        assert!(bbox.is_some());

        let bbox = bbox.unwrap();
        assert!(bbox.contains(Point2D::new(30.0, 30.0)));
        assert!(!bbox.contains(Point2D::new(5.0, 5.0)));
    }
}
