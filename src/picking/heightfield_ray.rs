// src/picking/heightfield_ray.rs
// Ray-heightfield intersection for terrain picking
// Part of Plan 3: Premium — Unified Picking with BVH + Python Callbacks

use super::ray::Ray;

/// Configuration for heightfield ray intersection
#[derive(Debug, Clone)]
pub struct HeightfieldConfig {
    /// Terrain width in world units
    pub terrain_width: f32,
    /// Terrain depth in world units  
    pub terrain_depth: f32,
    /// Minimum elevation
    pub min_elevation: f32,
    /// Maximum elevation
    pub max_elevation: f32,
    /// Z-scale factor
    pub z_scale: f32,
    /// Initial step size for ray marching
    pub initial_step: f32,
    /// Refinement iterations for binary search
    pub refinement_iterations: u32,
}

impl Default for HeightfieldConfig {
    fn default() -> Self {
        Self {
            terrain_width: 1000.0,
            terrain_depth: 1000.0,
            min_elevation: 0.0,
            max_elevation: 1000.0,
            z_scale: 1.0,
            initial_step: 1.0,
            refinement_iterations: 8,
        }
    }
}

/// Result of heightfield ray intersection
#[derive(Debug, Clone, Copy)]
pub struct HeightfieldHit {
    /// World position of hit
    pub position: [f32; 3],
    /// Distance along ray
    pub t: f32,
    /// UV coordinates on heightmap (0-1)
    pub uv: [f32; 2],
    /// Elevation at hit point
    pub elevation: f32,
    /// Surface normal at hit point
    pub normal: [f32; 3],
    /// Slope angle in degrees
    pub slope: f32,
    /// Aspect angle in degrees (0 = north, 90 = east)
    pub aspect: f32,
}

/// Heightfield ray intersection engine
pub struct HeightfieldRayEngine {
    config: HeightfieldConfig,
}

impl HeightfieldRayEngine {
    /// Create new heightfield ray engine
    pub fn new(config: HeightfieldConfig) -> Self {
        Self { config }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: HeightfieldConfig) {
        self.config = config;
    }

    /// Get configuration
    pub fn config(&self) -> &HeightfieldConfig {
        &self.config
    }

    /// Ray-heightfield intersection using ray marching with binary refinement
    /// 
    /// # Arguments
    /// * `ray` - The ray to test
    /// * `heightmap` - Heightmap data (row-major, Y values in world units)
    /// * `width` - Heightmap width in pixels
    /// * `height` - Heightmap height in pixels
    pub fn intersect(
        &self,
        ray: &Ray,
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> Option<HeightfieldHit> {
        let max_t = self.config.terrain_width * 2.0;
        let step = self.config.initial_step;

        let mut t = 0.0f32;
        let mut prev_above = true;
        let mut prev_t = 0.0f32;

        while t < max_t {
            let p = ray.point_at(t);

            // Check if point is within terrain bounds
            if !self.is_in_bounds(p) {
                // If we were in bounds before, we've exited - stop
                if t > 0.0 && self.is_in_bounds(ray.point_at(t - step)) {
                    break;
                }
                t += step;
                continue;
            }

            // Sample heightfield at this position
            let terrain_height = self.sample_heightfield(p, heightmap, width, height);
            let ray_height = p[1];

            let above = ray_height > terrain_height;

            // Detect crossing from above to below
            if !above && prev_above && t > 0.0 {
                // Binary refinement for accurate intersection
                let refined_t = self.refine_intersection(
                    ray, prev_t, t, heightmap, width, height
                );
                
                return self.compute_hit(ray, refined_t, heightmap, width, height);
            }

            prev_above = above;
            prev_t = t;
            t += step;
        }

        None
    }

    /// Check if point is within terrain bounds
    fn is_in_bounds(&self, p: [f32; 3]) -> bool {
        p[0] >= 0.0 && p[0] <= self.config.terrain_width
            && p[2] >= 0.0 && p[2] <= self.config.terrain_depth
    }

    /// Sample heightfield with bilinear interpolation
    fn sample_heightfield(
        &self,
        p: [f32; 3],
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> f32 {
        let u = (p[0] / self.config.terrain_width).clamp(0.0, 1.0);
        let v = (p[2] / self.config.terrain_depth).clamp(0.0, 1.0);

        let fx = u * (width - 1) as f32;
        let fy = v * (height - 1) as f32;

        let x0 = (fx.floor() as u32).min(width - 2);
        let y0 = (fy.floor() as u32).min(height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let idx00 = (y0 * width + x0) as usize;
        let idx10 = (y0 * width + x1) as usize;
        let idx01 = (y1 * width + x0) as usize;
        let idx11 = (y1 * width + x1) as usize;

        let h00 = heightmap.get(idx00).copied().unwrap_or(0.0);
        let h10 = heightmap.get(idx10).copied().unwrap_or(0.0);
        let h01 = heightmap.get(idx01).copied().unwrap_or(0.0);
        let h11 = heightmap.get(idx11).copied().unwrap_or(0.0);

        // Bilinear interpolation
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        let h = h0 * (1.0 - ty) + h1 * ty;

        // Scale height
        (h - self.config.min_elevation) * self.config.z_scale
    }

    /// Binary refinement for intersection point
    fn refine_intersection(
        &self,
        ray: &Ray,
        t_lo: f32,
        t_hi: f32,
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> f32 {
        let mut lo = t_lo;
        let mut hi = t_hi;

        for _ in 0..self.config.refinement_iterations {
            let mid = (lo + hi) * 0.5;
            let p = ray.point_at(mid);

            let terrain_height = self.sample_heightfield(p, heightmap, width, height);

            if p[1] > terrain_height {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        (lo + hi) * 0.5
    }

    /// Compute full hit information
    fn compute_hit(
        &self,
        ray: &Ray,
        t: f32,
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> Option<HeightfieldHit> {
        let position = ray.point_at(t);

        // Compute UV
        let u = (position[0] / self.config.terrain_width).clamp(0.0, 1.0);
        let v = (position[2] / self.config.terrain_depth).clamp(0.0, 1.0);

        // Sample elevation
        let elevation = self.sample_heightfield(position, heightmap, width, height) 
            / self.config.z_scale + self.config.min_elevation;

        // Compute normal from gradient
        let normal = self.compute_normal(u, v, heightmap, width, height);

        // Compute slope and aspect from normal
        let (slope, aspect) = self.normal_to_slope_aspect(normal);

        Some(HeightfieldHit {
            position,
            t,
            uv: [u, v],
            elevation,
            normal,
            slope,
            aspect,
        })
    }

    /// Compute normal from heightfield gradient
    fn compute_normal(
        &self,
        u: f32,
        v: f32,
        heightmap: &[f32],
        width: u32,
        height: u32,
    ) -> [f32; 3] {
        let delta_u = 1.0 / width as f32;
        let delta_v = 1.0 / height as f32;

        let sample = |uu: f32, vv: f32| -> f32 {
            let uu = uu.clamp(0.0, 1.0);
            let vv = vv.clamp(0.0, 1.0);

            let fx = uu * (width - 1) as f32;
            let fy = vv * (height - 1) as f32;

            let x = (fx as u32).min(width - 1);
            let y = (fy as u32).min(height - 1);
            let idx = (y * width + x) as usize;

            heightmap.get(idx).copied().unwrap_or(0.0)
        };

        let h_l = sample(u - delta_u, v);
        let h_r = sample(u + delta_u, v);
        let h_d = sample(u, v - delta_v);
        let h_u = sample(u, v + delta_v);

        let dx = (h_r - h_l) * self.config.z_scale;
        let dz = (h_u - h_d) * self.config.z_scale;

        let world_delta_x = delta_u * self.config.terrain_width * 2.0;
        let world_delta_z = delta_v * self.config.terrain_depth * 2.0;

        normalize([-dx / world_delta_x, 1.0, -dz / world_delta_z])
    }

    /// Convert normal vector to slope and aspect angles
    fn normal_to_slope_aspect(&self, normal: [f32; 3]) -> (f32, f32) {
        // Slope is the angle from vertical (0 = flat, 90 = cliff)
        let slope = normal[1].clamp(-1.0, 1.0).acos().to_degrees();

        // Aspect is the direction the slope faces
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
    fn test_heightfield_intersection() {
        // Create a simple flat heightfield
        let width = 10u32;
        let height = 10u32;
        let heightmap: Vec<f32> = vec![50.0; (width * height) as usize];

        let config = HeightfieldConfig {
            terrain_width: 100.0,
            terrain_depth: 100.0,
            min_elevation: 0.0,
            max_elevation: 100.0,
            z_scale: 1.0,
            initial_step: 1.0,
            refinement_iterations: 8,
        };

        let engine = HeightfieldRayEngine::new(config);

        // Ray from above, pointing down
        let ray = Ray::new([50.0, 100.0, 50.0], [0.0, -1.0, 0.0]);
        let hit = engine.intersect(&ray, &heightmap, width, height);

        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.elevation - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_slope_calculation() {
        let engine = HeightfieldRayEngine::new(HeightfieldConfig::default());
        
        // Flat normal (pointing up)
        let (slope, _) = engine.normal_to_slope_aspect([0.0, 1.0, 0.0]);
        assert!(slope.abs() < 1.0);

        // 45-degree slope
        let normal = normalize([1.0, 1.0, 0.0]);
        let (slope, _) = engine.normal_to_slope_aspect(normal);
        assert!((slope - 45.0).abs() < 1.0);
    }
}
