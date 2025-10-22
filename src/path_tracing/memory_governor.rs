// src/path_tracing/memory_governor.rs
// Memory governor for ray tracing to prevent OOM on unified memory systems (macOS/Metal)
// Pre-computes memory footprint and auto-adjusts tile size to fit within budget

/// Accumulation buffer format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumFormat {
    /// RGBA16F: 8 bytes per pixel (half precision, good quality, low memory)
    Rgba16F,
    /// RGBA32F: 16 bytes per pixel (full precision, highest quality, high memory)
    Rgba32F,
}

impl AccumFormat {
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            AccumFormat::Rgba16F => 8,
            AccumFormat::Rgba32F => 16,
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rgba16f" | "f16" | "half" => Some(AccumFormat::Rgba16F),
            "rgba32f" | "f32" | "float" => Some(AccumFormat::Rgba32F),
            _ => None,
        }
    }
}

impl Default for AccumFormat {
    fn default() -> Self {
        // Default to RGBA16F on macOS for memory efficiency
        #[cfg(target_os = "macos")]
        return AccumFormat::Rgba16F;
        
        #[cfg(not(target_os = "macos"))]
        return AccumFormat::Rgba32F;
    }
}

/// Memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum host-visible memory in bytes (default 512 MiB)
    pub limit_bytes: usize,
    /// Minimum tile size (width, height)
    pub tile_min: (u32, u32),
    /// Maximum tile size (width, height)
    pub tile_max: (u32, u32),
    /// Accumulation buffer format
    pub accum_format: AccumFormat,
    /// Enable automatic tile sizing
    pub auto_tile: bool,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            limit_bytes: 512 * 1024 * 1024, // 512 MiB
            tile_min: (96, 96),
            tile_max: (512, 512),
            accum_format: AccumFormat::default(),
            auto_tile: true,
        }
    }
}

impl MemoryBudget {
    pub fn from_mib(mib: usize) -> Self {
        Self {
            limit_bytes: mib * 1024 * 1024,
            ..Default::default()
        }
    }
}

/// Memory footprint breakdown for a single tile
#[derive(Debug, Clone)]
pub struct TileMemoryFootprint {
    pub tile_width: u32,
    pub tile_height: u32,
    /// Accumulation buffer (RGBA16F or RGBA32F)
    pub accum_bytes: usize,
    /// Per-sample buffer (for progressive rendering)
    pub spp_buffer_bytes: usize,
    /// RNG state buffer
    pub rng_state_bytes: usize,
    /// Staging buffer for readback
    pub staging_bytes: usize,
    /// Output texture (RGBA8)
    pub output_bytes: usize,
    /// Per-tile BVH (estimated, depends on triangle count)
    pub bvh_bytes: usize,
    /// Per-tile vertices (quantized to 16-bit)
    pub vertex_bytes: usize,
    /// Per-tile indices (32-bit)
    pub index_bytes: usize,
    /// OIDN scratch buffer (if enabled, ~2x image size)
    pub oidn_scratch_bytes: usize,
    /// Total bytes for this tile
    pub total_bytes: usize,
}

impl TileMemoryFootprint {
    /// Compute memory footprint for a tile of given dimensions
    pub fn compute(
        tile_width: u32,
        tile_height: u32,
        accum_format: AccumFormat,
        num_triangles_per_tile: usize,
        enable_oidn: bool,
    ) -> Self {
        let pixels = (tile_width as usize) * (tile_height as usize);
        
        // Accumulation buffer (tile-local)
        let accum_bytes = pixels * accum_format.bytes_per_pixel();
        
        // SPP counter buffer (u32 per pixel)
        let spp_buffer_bytes = pixels * 4;
        
        // RNG state (2x u32 per pixel for Sobol/CMJ)
        let rng_state_bytes = pixels * 8;
        
        // Staging buffer for GPU->CPU readback (RGBA8)
        let staging_bytes = pixels * 4;
        
        // Output texture (RGBA8)
        let output_bytes = pixels * 4;
        
        // Per-tile BVH (BVH8 nodes ~32 bytes, assume 2x triangle count for nodes)
        let bvh_bytes = num_triangles_per_tile * 2 * 32;
        
        // Quantized vertices (16-bit positions + 16-bit normals = 12 bytes per vertex)
        let vertex_bytes = num_triangles_per_tile * 3 * 12;
        
        // Indices (32-bit, 3 per triangle)
        let index_bytes = num_triangles_per_tile * 3 * 4;
        
        // OIDN scratch (conservative: 2x RGB32F image size)
        let oidn_scratch_bytes = if enable_oidn {
            pixels * 3 * 4 * 2
        } else {
            0
        };
        
        let total_bytes = accum_bytes
            + spp_buffer_bytes
            + rng_state_bytes
            + staging_bytes
            + output_bytes
            + bvh_bytes
            + vertex_bytes
            + index_bytes
            + oidn_scratch_bytes;
        
        Self {
            tile_width,
            tile_height,
            accum_bytes,
            spp_buffer_bytes,
            rng_state_bytes,
            staging_bytes,
            output_bytes,
            bvh_bytes,
            vertex_bytes,
            index_bytes,
            oidn_scratch_bytes,
            total_bytes,
        }
    }
    
    /// Pretty-print memory breakdown
    pub fn print_table(&self) {
        eprintln!("┌─────────────────────────────────────────────────────┐");
        eprintln!("│ Per-Tile Memory Budget ({}x{} px)                   │", 
            self.tile_width, self.tile_height);
        eprintln!("├─────────────────────────────────────────────────────┤");
        eprintln!("│ Accumulation buffer:    {:>8} MB             │", self.accum_bytes / (1024 * 1024));
        eprintln!("│ SPP counter buffer:     {:>8} MB             │", self.spp_buffer_bytes / (1024 * 1024));
        eprintln!("│ RNG state buffer:       {:>8} MB             │", self.rng_state_bytes / (1024 * 1024));
        eprintln!("│ Staging buffer:         {:>8} MB             │", self.staging_bytes / (1024 * 1024));
        eprintln!("│ Output texture:         {:>8} MB             │", self.output_bytes / (1024 * 1024));
        eprintln!("│ BVH nodes:              {:>8} MB             │", self.bvh_bytes / (1024 * 1024));
        eprintln!("│ Vertices (quantized):   {:>8} MB             │", self.vertex_bytes / (1024 * 1024));
        eprintln!("│ Indices:                {:>8} MB             │", self.index_bytes / (1024 * 1024));
        if self.oidn_scratch_bytes > 0 {
            eprintln!("│ OIDN scratch:           {:>8} MB             │", self.oidn_scratch_bytes / (1024 * 1024));
        }
        eprintln!("├─────────────────────────────────────────────────────┤");
        eprintln!("│ TOTAL:                  {:>8} MB             │", self.total_bytes / (1024 * 1024));
        eprintln!("└─────────────────────────────────────────────────────┘");
    }
}

/// Memory governor that auto-adjusts tile size to fit budget
pub struct MemoryGovernor {
    budget: MemoryBudget,
}

impl MemoryGovernor {
    pub fn new(budget: MemoryBudget) -> Self {
        eprintln!("[MemoryGovernor] Initialized with budget: {} MB", 
            budget.limit_bytes / (1024 * 1024));
        eprintln!("[MemoryGovernor] Tile range: {}x{} to {}x{}", 
            budget.tile_min.0, budget.tile_min.1,
            budget.tile_max.0, budget.tile_max.1);
        eprintln!("[MemoryGovernor] Accumulation format: {:?}", budget.accum_format);
        Self { budget }
    }
    
    /// Compute optimal tile size for given image dimensions and constraints
    /// Returns (tile_width, tile_height, estimated_memory_per_tile)
    pub fn compute_tile_size(
        &self,
        image_width: u32,
        image_height: u32,
        total_triangles: usize,
        enable_oidn: bool,
    ) -> Result<(u32, u32, TileMemoryFootprint), String> {
        if !self.budget.auto_tile {
            // Use max tile size without auto-adjustment
            let tw = self.budget.tile_max.0;
            let th = self.budget.tile_max.1;
            let tris_per_tile = self.estimate_triangles_per_tile(
                tw, th, image_width, image_height, total_triangles
            );
            let footprint = TileMemoryFootprint::compute(
                tw, th, self.budget.accum_format, tris_per_tile, enable_oidn
            );
            return Ok((tw, th, footprint));
        }
        
        // Binary search for largest tile size that fits budget
        let mut lo = self.budget.tile_min;
        let mut hi = self.budget.tile_max;
        let mut best = lo;
        let mut best_footprint = None;
        
        for _ in 0..10 {
            // Try midpoint
            let mid_w = (lo.0 + hi.0) / 2;
            let mid_h = (lo.1 + hi.1) / 2;
            
            // Round to multiple of 8 for workgroup alignment
            let mid_w = ((mid_w + 7) / 8) * 8;
            let mid_h = ((mid_h + 7) / 8) * 8;
            
            let tris_per_tile = self.estimate_triangles_per_tile(
                mid_w, mid_h, image_width, image_height, total_triangles
            );
            
            let footprint = TileMemoryFootprint::compute(
                mid_w, mid_h, self.budget.accum_format, tris_per_tile, enable_oidn
            );
            
            if footprint.total_bytes <= self.budget.limit_bytes {
                // Fits! Try larger
                best = (mid_w, mid_h);
                best_footprint = Some(footprint);
                lo = (mid_w, mid_h);
            } else {
                // Too large, try smaller
                hi = (mid_w, mid_h);
            }
            
            // Converged?
            if hi.0 <= lo.0 + 16 && hi.1 <= lo.1 + 16 {
                break;
            }
        }
        
        if let Some(footprint) = best_footprint {
            eprintln!("[MemoryGovernor] Optimal tile size: {}x{} ({}x{} image)", 
                best.0, best.1, image_width, image_height);
            eprintln!("[MemoryGovernor] Per-tile memory: {} MB / {} MB budget", 
                footprint.total_bytes / (1024 * 1024),
                self.budget.limit_bytes / (1024 * 1024));
            Ok((best.0, best.1, footprint))
        } else {
            Err(format!(
                "Cannot fit any tile within budget {} MB. Minimum tile {}x{} requires more memory.",
                self.budget.limit_bytes / (1024 * 1024),
                self.budget.tile_min.0, self.budget.tile_min.1
            ))
        }
    }
    
    /// Estimate number of triangles that will be visible in a tile
    /// Uses conservative heuristic: tile_area / image_area * total_triangles * 1.2 (margin)
    fn estimate_triangles_per_tile(
        &self,
        tile_width: u32,
        tile_height: u32,
        image_width: u32,
        image_height: u32,
        total_triangles: usize,
    ) -> usize {
        let tile_area = (tile_width as f64) * (tile_height as f64);
        let image_area = (image_width as f64) * (image_height as f64);
        let ratio = tile_area / image_area;
        
        // Conservative estimate with 1.5x margin for overlapping geometry
        let estimated = (total_triangles as f64 * ratio * 1.5).ceil() as usize;
        
        // Clamp to reasonable range
        estimated.max(1000).min(total_triangles)
    }
    
    /// Check if a render configuration will fit in budget
    pub fn validate_config(
        &self,
        tile_width: u32,
        tile_height: u32,
        triangles_per_tile: usize,
        enable_oidn: bool,
    ) -> Result<TileMemoryFootprint, String> {
        let footprint = TileMemoryFootprint::compute(
            tile_width, tile_height, self.budget.accum_format, triangles_per_tile, enable_oidn
        );
        
        if footprint.total_bytes > self.budget.limit_bytes {
            return Err(format!(
                "Configuration exceeds budget: {} MB > {} MB limit",
                footprint.total_bytes / (1024 * 1024),
                self.budget.limit_bytes / (1024 * 1024)
            ));
        }
        
        Ok(footprint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_accum_format_bytes() {
        assert_eq!(AccumFormat::Rgba16F.bytes_per_pixel(), 8);
        assert_eq!(AccumFormat::Rgba32F.bytes_per_pixel(), 16);
    }
    
    #[test]
    fn test_memory_footprint_small_tile() {
        let footprint = TileMemoryFootprint::compute(
            256, 256, AccumFormat::Rgba16F, 50_000, false
        );
        
        // 256x256 = 65,536 pixels
        assert_eq!(footprint.accum_bytes, 65_536 * 8); // RGBA16F
        assert_eq!(footprint.output_bytes, 65_536 * 4); // RGBA8
        assert!(footprint.total_bytes < 50 * 1024 * 1024); // Should be under 50 MB
    }
    
    #[test]
    fn test_memory_governor_auto_tile() {
        let budget = MemoryBudget {
            limit_bytes: 128 * 1024 * 1024, // 128 MB
            tile_min: (96, 96),
            tile_max: (512, 512),
            accum_format: AccumFormat::Rgba16F,
            auto_tile: true,
        };
        
        let governor = MemoryGovernor::new(budget);
        let result = governor.compute_tile_size(3840, 2160, 1_000_000, false);
        
        assert!(result.is_ok());
        let (tw, th, footprint) = result.unwrap();
        assert!(tw >= 96 && tw <= 512);
        assert!(th >= 96 && th <= 512);
        assert!(footprint.total_bytes <= 128 * 1024 * 1024);
    }
    
    #[test]
    fn test_memory_governor_validate() {
        let budget = MemoryBudget::from_mib(256);
        let governor = MemoryGovernor::new(budget);
        
        // Small tile should fit
        let result = governor.validate_config(256, 256, 50_000, false);
        assert!(result.is_ok());
        
        // Huge tile should fail
        let result = governor.validate_config(2048, 2048, 500_000, true);
        assert!(result.is_err());
    }
}
