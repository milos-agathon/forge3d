// src/picking/terrain_query.rs
// Terrain elevation and slope queries from depth buffer
// Part of Plan 2: Standard - GPU Ray Picking + Hover Support

use super::ray::{invert_matrix, Ray};

/// Result of a terrain query
#[derive(Debug, Clone, Copy)]
pub struct TerrainQueryResult {
    /// Elevation at the query point (world units)
    pub elevation: f32,
    /// Slope angle in degrees (0 = flat, 90 = vertical)
    pub slope: f32,
    /// Aspect angle in degrees (0 = north, 90 = east, 180 = south, 270 = west)
    pub aspect: f32,
    /// Render-relative position reconstructed by the depth/ray query.
    pub render_pos: [f32; 3],
    /// Absolute f64 world position restored with the exact producing anchor.
    pub world_pos: [f64; 3],
    /// Anchor snapshot that produced `render_pos` and `world_pos`.
    pub producing_anchor: crate::camera::Anchor,
    /// Normal vector at the query point
    pub normal: [f32; 3],
}

impl Default for TerrainQueryResult {
    fn default() -> Self {
        Self {
            elevation: 0.0,
            slope: 0.0,
            aspect: 0.0,
            render_pos: [0.0, 0.0, 0.0],
            world_pos: [0.0, 0.0, 0.0],
            producing_anchor: crate::camera::Anchor::new(),
            normal: [0.0, 1.0, 0.0],
        }
    }
}

impl TerrainQueryResult {
    fn from_render(
        render_pos: [f32; 3],
        producing_anchor: crate::camera::Anchor,
        elevation: f32,
        slope: f32,
        aspect: f32,
        normal: [f32; 3],
    ) -> Self {
        let world_pos = producing_anchor
            .to_world_from_render_f64(glam::Vec3::from(render_pos).as_dvec3())
            .to_array();
        Self {
            elevation,
            slope,
            aspect,
            render_pos,
            world_pos,
            producing_anchor,
            normal,
        }
    }
}

/// Configuration for terrain queries
#[derive(Debug, Clone)]
pub struct TerrainQueryConfig {
    /// Terrain width in world units
    pub terrain_width: f32,
    /// Terrain height in world units (elevation range)
    pub terrain_height: f32,
    /// Minimum elevation
    pub min_elevation: f32,
    /// Maximum elevation
    pub max_elevation: f32,
    /// Z-scale factor
    pub z_scale: f32,
}

impl Default for TerrainQueryConfig {
    fn default() -> Self {
        Self {
            terrain_width: 1000.0,
            terrain_height: 100.0,
            min_elevation: 0.0,
            max_elevation: 100.0,
            z_scale: 1.0,
        }
    }
}

/// Terrain query engine for elevation and slope queries
#[derive(Debug)]
pub struct TerrainQueryEngine {
    config: TerrainQueryConfig,
}

impl TerrainQueryEngine {
    /// Create a new terrain query engine
    pub fn new(config: TerrainQueryConfig) -> Self {
        Self { config }
    }

    /// Update terrain configuration
    pub fn set_config(&mut self, config: TerrainQueryConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &TerrainQueryConfig {
        &self.config
    }

    /// Reconstruct world position from screen coordinates and depth
    pub fn reconstruct_world_from_depth(
        &self,
        screen_x: u32,
        screen_y: u32,
        screen_width: u32,
        screen_height: u32,
        depth: f32,
        view_proj: [[f32; 4]; 4],
    ) -> Option<[f32; 3]> {
        let inv_view_proj = invert_matrix(view_proj)?;

        // Convert screen to NDC
        let ndc_x = (2.0 * screen_x as f32 / screen_width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y as f32 / screen_height as f32);
        let ndc_z = depth; // Assuming depth is already in [0, 1] range

        // Transform from NDC to world
        let ndc = [ndc_x, ndc_y, ndc_z, 1.0];
        let world = transform_point(ndc, inv_view_proj);

        Some(world)
    }

    /// Query terrain at a screen position using depth buffer value
    pub fn query_at_depth(
        &self,
        screen_x: u32,
        screen_y: u32,
        screen_width: u32,
        screen_height: u32,
        depth: f32,
        view_proj: [[f32; 4]; 4],
        producing_anchor: crate::camera::Anchor,
    ) -> Option<TerrainQueryResult> {
        let render_pos = self.reconstruct_world_from_depth(
            screen_x,
            screen_y,
            screen_width,
            screen_height,
            depth,
            view_proj,
        )?;

        // Extract elevation from Y coordinate
        let elevation = render_pos[1] / self.config.z_scale + self.config.min_elevation;
        Some(TerrainQueryResult::from_render(
            render_pos,
            producing_anchor,
            elevation,
            0.0, // Would need heightmap sampling for accurate slope
            0.0, // Would need heightmap sampling for accurate aspect
            [0.0, 1.0, 0.0],
        ))
    }

    /// Query terrain using ray intersection with heightfield
    /// This is a CPU-based fallback when depth buffer is unavailable
    pub fn query_ray_heightfield(
        &self,
        ray: &Ray,
        heightmap: &[f32],
        heightmap_width: u32,
        heightmap_height: u32,
        producing_anchor: crate::camera::Anchor,
    ) -> Option<TerrainQueryResult> {
        // Simple ray marching through heightfield
        let max_t = self.config.terrain_width * 2.0;
        let step = self.config.terrain_width / heightmap_width as f32;

        let mut t = 0.0;
        let mut prev_above = true;

        while t < max_t {
            let p = ray.point_at(t);

            // Check if point is within terrain bounds
            if p[0] < 0.0
                || p[0] > self.config.terrain_width
                || p[2] < 0.0
                || p[2] > self.config.terrain_width
            {
                t += step;
                continue;
            }

            // Sample heightfield at this position
            let u = (p[0] / self.config.terrain_width).clamp(0.0, 1.0);
            let v = (p[2] / self.config.terrain_width).clamp(0.0, 1.0);

            let hx = (u * (heightmap_width - 1) as f32) as u32;
            let hz = (v * (heightmap_height - 1) as f32) as u32;
            let idx = (hz * heightmap_width + hx) as usize;

            if idx >= heightmap.len() {
                t += step;
                continue;
            }

            let terrain_height = heightmap[idx] * self.config.z_scale;
            let ray_height = p[1];

            let above = ray_height > terrain_height;

            // Detect crossing
            if !above && prev_above {
                // Binary refinement for accurate intersection
                let refined_t = self.refine_intersection(
                    ray,
                    t - step,
                    t,
                    heightmap,
                    heightmap_width,
                    heightmap_height,
                );
                let hit_pos = ray.point_at(refined_t);

                let elevation = heightmap[idx] * self.config.z_scale / self.config.z_scale
                    + self.config.min_elevation;

                // Compute normal from heightfield gradient
                let normal =
                    self.compute_normal_at(u, v, heightmap, heightmap_width, heightmap_height);
                let (slope, aspect) = self.normal_to_slope_aspect(normal);

                return Some(TerrainQueryResult::from_render(
                    hit_pos,
                    producing_anchor,
                    elevation,
                    slope,
                    aspect,
                    normal,
                ));
            }

            prev_above = above;
            t += step;
        }

        None
    }

    /// Binary refinement for intersection point
    fn refine_intersection(
        &self,
        ray: &Ray,
        t_lo: f32,
        t_hi: f32,
        heightmap: &[f32],
        heightmap_width: u32,
        heightmap_height: u32,
    ) -> f32 {
        let mut lo = t_lo;
        let mut hi = t_hi;

        for _ in 0..8 {
            let mid = (lo + hi) * 0.5;
            let p = ray.point_at(mid);

            let u = (p[0] / self.config.terrain_width).clamp(0.0, 1.0);
            let v = (p[2] / self.config.terrain_width).clamp(0.0, 1.0);

            let hx = (u * (heightmap_width - 1) as f32) as u32;
            let hz = (v * (heightmap_height - 1) as f32) as u32;
            let idx = (hz * heightmap_width + hx) as usize;

            if idx >= heightmap.len() {
                break;
            }

            let terrain_height = heightmap[idx] * self.config.z_scale;

            if p[1] > terrain_height {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        (lo + hi) * 0.5
    }

    /// Compute normal from heightfield gradient
    fn compute_normal_at(
        &self,
        u: f32,
        v: f32,
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> [f32; 3] {
        let sample = |uu: f32, vv: f32| -> f32 {
            let uu = uu.clamp(0.0, 1.0);
            let vv = vv.clamp(0.0, 1.0);
            let hx = (uu * (width - 1) as f32) as u32;
            let hz = (vv * (height - 1) as f32) as u32;
            let idx = (hz * width + hx) as usize;
            if idx < heightmap.len() {
                heightmap[idx]
            } else {
                0.0
            }
        };

        let delta = 1.0 / width as f32;
        let h_l = sample(u - delta, v);
        let h_r = sample(u + delta, v);
        let h_d = sample(u, v - delta);
        let h_u = sample(u, v + delta);

        let dx = (h_r - h_l) * self.config.z_scale;
        let dz = (h_u - h_d) * self.config.z_scale;

        normalize([-dx, 2.0 * delta * self.config.terrain_width, -dz])
    }

    /// Convert normal vector to slope and aspect angles
    fn normal_to_slope_aspect(&self, normal: [f32; 3]) -> (f32, f32) {
        // Slope is the angle from vertical (Y-axis)
        let slope = (1.0 - normal[1].abs()).acos().to_degrees();

        // Aspect is the direction the slope faces (0 = north, 90 = east)
        let aspect = if normal[0].abs() < 1e-6 && normal[2].abs() < 1e-6 {
            0.0
        } else {
            let mut a = normal[0].atan2(-normal[2]).to_degrees();
            if a < 0.0 {
                a += 360.0;
            }
            a
        };

        (slope, aspect)
    }
}

/// Transform a point by a 4x4 matrix with perspective divide
fn transform_point(point: [f32; 4], matrix: [[f32; 4]; 4]) -> [f32; 3] {
    let x = matrix[0][0] * point[0]
        + matrix[1][0] * point[1]
        + matrix[2][0] * point[2]
        + matrix[3][0] * point[3];
    let y = matrix[0][1] * point[0]
        + matrix[1][1] * point[1]
        + matrix[2][1] * point[2]
        + matrix[3][1] * point[3];
    let z = matrix[0][2] * point[0]
        + matrix[1][2] * point[1]
        + matrix[2][2] * point[2]
        + matrix[3][2] * point[3];
    let w = matrix[0][3] * point[0]
        + matrix[1][3] * point[1]
        + matrix[2][3] * point[2]
        + matrix[3][3] * point[3];

    if w.abs() < 1e-10 {
        [x, y, z]
    } else {
        [x / w, y / w, z / w]
    }
}

/// Normalize a 3D vector
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_result_restores_submillimetre_render_offset_with_producing_anchor() {
        let origin = glam::DVec3::new(6_378_137.0, 500_000.0, -5_500_000.0);
        let mut anchor = crate::camera::Anchor::new();
        assert!(anchor.rebase_if_needed(origin));
        let result = TerrainQueryResult::from_render(
            [0.000_25, -0.000_5, 0.000_75],
            anchor,
            0.0,
            0.0,
            0.0,
            [0.0, 1.0, 0.0],
        );
        let expected = origin + glam::DVec3::new(0.000_25, -0.000_5, 0.000_75);
        assert!((glam::DVec3::from(result.world_pos) - expected).length() < 1.0e-9);
        assert_eq!(result.producing_anchor, anchor);
    }

    #[test]
    fn current_anchor_is_a_detectable_wrong_provenance_negative_control() {
        let origin = glam::DVec3::new(6_378_137.0, 500_000.0, -5_500_000.0);
        let mut producing = crate::camera::Anchor::new();
        assert!(producing.rebase_if_needed(origin));
        let render_pos = [0.000_25, 0.0, 0.0];
        let result =
            TerrainQueryResult::from_render(render_pos, producing, 0.0, 0.0, 0.0, [0.0, 1.0, 0.0]);

        let mut current = producing;
        assert!(current.rebase_if_needed(origin + glam::DVec3::X * 1_500.0));
        let wrong = current.to_world_from_render_f64(glam::Vec3::from(render_pos).as_dvec3());
        assert!((wrong - glam::DVec3::from(result.world_pos)).length() > 1_000.0);
        assert!((result.world_pos[0] - 6_378_137.000_25).abs() < 1.0e-9);
    }
}
