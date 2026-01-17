//! Point cloud LOD traversal

use glam::{Mat4, Vec3};
use super::octree::{OctreeKey, OctreeBounds, OctreeNode};

/// Parameters for LOD traversal
#[derive(Debug, Clone)]
pub struct TraversalParams {
    /// Maximum points to load
    pub point_budget: u64,
    /// Minimum node spacing in world units
    pub min_spacing: f32,
    /// Viewport height for SSE computation
    pub viewport_height: f32,
    /// Vertical FOV in radians
    pub fov_y: f32,
    /// Maximum traversal depth
    pub max_depth: u32,
}

impl Default for TraversalParams {
    fn default() -> Self {
        Self {
            point_budget: 5_000_000,
            min_spacing: 0.01,
            viewport_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4,
            max_depth: 20,
        }
    }
}

/// Result of traversal for a visible node
#[derive(Debug, Clone)]
pub struct VisibleNode {
    pub key: OctreeKey,
    pub bounds: OctreeBounds,
    pub point_count: u64,
    pub spacing: f32,
    pub priority: f32,
}

/// Point cloud traverser with budget enforcement
pub struct PointCloudTraverser {
    params: TraversalParams,
}

impl Default for PointCloudTraverser {
    fn default() -> Self {
        Self { params: TraversalParams::default() }
    }
}

impl PointCloudTraverser {
    pub fn new(params: TraversalParams) -> Self {
        Self { params }
    }

    pub fn set_point_budget(&mut self, budget: u64) {
        self.params.point_budget = budget;
    }

    pub fn set_viewport(&mut self, height: f32, fov_y: f32) {
        self.params.viewport_height = height;
        self.params.fov_y = fov_y;
    }

    /// Traverse and return visible nodes within budget
    pub fn visible_nodes<F>(
        &self,
        root: &OctreeNode,
        camera_pos: Vec3,
        view_proj: Option<&Mat4>,
        get_children: F,
    ) -> Vec<VisibleNode>
    where
        F: Fn(&OctreeKey) -> Vec<OctreeNode>,
    {
        let mut candidates = Vec::new();
        let mut result = Vec::new();
        let mut total_points: u64 = 0;

        // Start with root
        candidates.push((root.clone(), self.compute_priority(&root.bounds, camera_pos)));

        while let Some((node, priority)) = candidates.pop() {
            // Frustum cull
            if let Some(vp) = view_proj {
                if !node.bounds.intersects_frustum(vp) {
                    continue;
                }
            }

            // Check depth limit
            if node.key.depth > self.params.max_depth {
                continue;
            }

            // Check if we can add this node within budget
            if total_points + node.point_count > self.params.point_budget {
                // Over budget - only add if high priority
                if priority < 1.0 {
                    continue;
                }
            }

            // Check spacing threshold
            let screen_size = self.compute_screen_size(&node.bounds, camera_pos);
            let should_refine = screen_size > 1.0 && node.has_children();

            if should_refine && node.key.depth < self.params.max_depth {
                // Refine - add children to candidates
                let children = get_children(&node.key);
                for child in children {
                    let child_priority = self.compute_priority(&child.bounds, camera_pos);
                    // Insert sorted by priority (higher first)
                    let pos = candidates.iter()
                        .position(|(_, p)| *p < child_priority)
                        .unwrap_or(candidates.len());
                    candidates.insert(pos, (child, child_priority));
                }
            } else {
                // Accept this node
                total_points += node.point_count;
                result.push(VisibleNode {
                    key: node.key.clone(),
                    bounds: node.bounds,
                    point_count: node.point_count,
                    spacing: node.spacing,
                    priority,
                });

                // Check if we've reached budget
                if total_points >= self.params.point_budget {
                    break;
                }
            }
        }

        result
    }

    fn compute_priority(&self, bounds: &OctreeBounds, camera_pos: Vec3) -> f32 {
        let center = bounds.center();
        let distance = (center - camera_pos).length();
        let radius = bounds.radius();
        
        if distance < radius {
            return f32::MAX; // Camera inside bounds
        }
        
        // Priority based on projected size
        radius / distance
    }

    fn compute_screen_size(&self, bounds: &OctreeBounds, camera_pos: Vec3) -> f32 {
        let center = bounds.center();
        let distance = (center - camera_pos).length().max(0.001);
        let radius = bounds.radius();
        
        let sse_factor = self.params.viewport_height / (2.0 * (self.params.fov_y / 2.0).tan());
        (radius / distance) * sse_factor
    }

    /// Get traversal statistics
    pub fn stats<F>(
        &self,
        root: &OctreeNode,
        camera_pos: Vec3,
        get_children: F,
    ) -> TraversalStats
    where
        F: Fn(&OctreeKey) -> Vec<OctreeNode>,
    {
        let visible = self.visible_nodes(root, camera_pos, None, get_children);
        
        let total_points: u64 = visible.iter().map(|n| n.point_count).sum();
        let max_depth = visible.iter().map(|n| n.key.depth).max().unwrap_or(0);
        
        TraversalStats {
            node_count: visible.len(),
            total_points,
            max_depth,
            budget_used_percent: (total_points as f32 / self.params.point_budget as f32) * 100.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraversalStats {
    pub node_count: usize,
    pub total_points: u64,
    pub max_depth: u32,
    pub budget_used_percent: f32,
}
