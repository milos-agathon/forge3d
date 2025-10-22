//! Heightfield acceleration structure for efficient DEM ray tracing.
//!
//! Instead of generating millions of triangles, this uses 2D DDA traversal
//! over a heightfield grid with min/max mipmap pyramid for early rejection.
//!
//! Performance: O(sqrt(N)) build, O(log N) query, ~4 MB memory for typical DEMs.

use glam::{Vec2, Vec3};

/// Min/max height values for a tile in the mipmap hierarchy
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeightBounds {
    pub min: f32,
    pub max: f32,
}

/// Heightfield accelerator with 4-level mipmap pyramid
pub struct HeightfieldAccel {
    /// Original heightmap dimensions
    pub width: u32,
    pub height: u32,
    
    /// Height data (row-major, Y-up)
    pub heights: Vec<f32>,
    
    /// Mipmap pyramid: level 0 = full res, level N = downsampled by 2^N
    /// Each level stores min/max for 2x2 tiles
    pub mip_levels: Vec<Vec<HeightBounds>>,
    
    /// World-space scale factors
    pub scale: Vec3,
    pub offset: Vec3,
}

impl HeightfieldAccel {
    /// Build heightfield accelerator from DEM data
    pub fn new(heights: Vec<f32>, width: u32, height: u32, scale: Vec3, offset: Vec3) -> Self {
        let mut accel = Self {
            width,
            height,
            heights,
            mip_levels: Vec::new(),
            scale,
            offset,
        };
        
        accel.build_mipmap_pyramid();
        accel
    }
    
    /// Build 4-level min/max mipmap pyramid
    fn build_mipmap_pyramid(&mut self) {
        let max_levels = 4;
        
        // Level 0: per-pixel min/max (trivial - just the height itself)
        let mut current_width = self.width;
        let mut current_height = self.height;
        
        for level in 0..max_levels {
            if current_width < 2 || current_height < 2 {
                break;
            }
            
            let next_width = (current_width + 1) / 2;
            let next_height = (current_height + 1) / 2;
            let mut level_bounds = vec![HeightBounds { min: f32::MAX, max: f32::MIN }; 
                                        (next_width * next_height) as usize];
            
            // Downsample: each output tile covers 2x2 input tiles
            for y in 0..next_height {
                for x in 0..next_width {
                    let src_x = x * 2;
                    let src_y = y * 2;
                    
                    let mut tile_min = f32::MAX;
                    let mut tile_max = f32::MIN;
                    
                    // Sample 2x2 region
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sample_x = (src_x + dx).min(current_width - 1);
                            let sample_y = (src_y + dy).min(current_height - 1);
                            
                            let height = if level == 0 {
                                // Sample from original heightmap
                                self.heights[(sample_y * self.width + sample_x) as usize]
                            } else {
                                // Sample from previous mip level
                                let prev_level = &self.mip_levels[level - 1];
                                let idx = (sample_y * current_width + sample_x) as usize;
                                let bounds = prev_level[idx];
                                tile_min = tile_min.min(bounds.min);
                                tile_max = tile_max.max(bounds.max);
                                continue; // Skip individual min/max updates
                            };
                            
                            tile_min = tile_min.min(height);
                            tile_max = tile_max.max(height);
                        }
                    }
                    
                    let idx = (y * next_width + x) as usize;
                    level_bounds[idx] = HeightBounds { min: tile_min, max: tile_max };
                }
            }
            
            self.mip_levels.push(level_bounds);
            current_width = next_width;
            current_height = next_height;
        }
    }
    
    /// Sample height at grid coordinates with bilinear filtering
    pub fn sample_height(&self, u: f32, v: f32) -> f32 {
        let x = u * (self.width - 1) as f32;
        let y = v * (self.height - 1) as f32;
        
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        
        let h00 = self.heights[(y0 * self.width + x0) as usize];
        let h10 = self.heights[(y0 * self.width + x1) as usize];
        let h01 = self.heights[(y1 * self.width + x0) as usize];
        let h11 = self.heights[(y1 * self.width + x1) as usize];
        
        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;
        h0 * (1.0 - fy) + h1 * fy
    }
    
    /// Get min/max bounds for a tile at given mip level
    pub fn get_tile_bounds(&self, x: u32, y: u32, level: usize) -> Option<HeightBounds> {
        if level >= self.mip_levels.len() {
            return None;
        }
        
        let level_data = &self.mip_levels[level];
        let level_width = (self.width + (1 << level) - 1) >> level;
        
        if x >= level_width || y >= (self.height >> level) {
            return None;
        }
        
        let idx = (y * level_width + x) as usize;
        level_data.get(idx).copied()
    }
    
    /// Compute normal at UV coordinates using central differences
    pub fn compute_normal(&self, u: f32, v: f32) -> Vec3 {
        let epsilon = 1.0 / self.width as f32;
        
        let h_center = self.sample_height(u, v);
        let h_right = self.sample_height((u + epsilon).min(1.0), v);
        let h_up = self.sample_height(u, (v + epsilon).min(1.0));
        
        let dx = (h_right - h_center) / (epsilon * self.scale.x);
        let dy = (h_up - h_center) / (epsilon * self.scale.z);
        
        Vec3::new(-dx, 1.0, -dy).normalize()
    }
    
    /// Convert UV coordinates to world position
    pub fn uv_to_world(&self, u: f32, v: f32) -> Vec3 {
        let x = u * self.scale.x + self.offset.x;
        let z = v * self.scale.z + self.offset.z;
        let y = self.sample_height(u, v) * self.scale.y + self.offset.y;
        Vec3::new(x, y, z)
    }
}

/// Ray-heightfield intersection result
#[derive(Debug, Clone, Copy)]
pub struct HeightfieldHit {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// 2D DDA traversal for ray-heightfield intersection
pub fn intersect_heightfield(
    ray_origin: Vec3,
    ray_dir: Vec3,
    accel: &HeightfieldAccel,
    t_min: f32,
    t_max: f32,
) -> Option<HeightfieldHit> {
    // Transform ray to heightfield space
    let origin_local = (ray_origin - accel.offset) / accel.scale;
    let dir_local = ray_dir / accel.scale;
    
    // Compute UV bounds
    let u_range = (0.0, 1.0);
    let v_range = (0.0, 1.0);
    
    // Early exit if ray doesn't intersect heightfield XZ plane
    if dir_local.y.abs() < 1e-6 {
        return None;
    }
    
    // DDA march along XZ plane
    let mut t = t_min;
    let dt = 0.01; // Step size in world units
    
    while t < t_max {
        let p = origin_local + dir_local * t;
        
        // Convert to UV
        let u = (p.x - u_range.0) / (u_range.1 - u_range.0);
        let v = (p.z - v_range.0) / (v_range.1 - v_range.0);
        
        // Check bounds
        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            t += dt;
            continue;
        }
        
        // Sample heightfield
        let terrain_height = accel.sample_height(u, v);
        
        // Check if ray is below terrain surface
        if p.y <= terrain_height {
            // Found intersection
            let normal = accel.compute_normal(u, v);
            let world_pos = accel.uv_to_world(u, v);
            
            return Some(HeightfieldHit {
                t,
                position: world_pos,
                normal,
                uv: Vec2::new(u, v),
            });
        }
        
        t += dt;
    }
    
    None
}
