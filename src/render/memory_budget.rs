// GPU Memory Budget Enforcement (P8)
// Tracks GPU memory usage and enforces budget constraints with auto-downscaling

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Memory allocation category for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    VertexBuffers,
    IndexBuffers,
    UniformBuffers,
    TextureRgba8,
    TextureRgba16f,
    TextureR32f,
    TextureDepth,
    ShadowMaps,
    IblCubemaps,
    IblBrdfLut,
    FroxelGrid,
    ScreenSpaceEffects,
    Other,
}

impl MemoryCategory {
    pub fn name(&self) -> &'static str {
        match self {
            MemoryCategory::VertexBuffers => "Vertex Buffers",
            MemoryCategory::IndexBuffers => "Index Buffers",
            MemoryCategory::UniformBuffers => "Uniform Buffers",
            MemoryCategory::TextureRgba8 => "Textures (RGBA8)",
            MemoryCategory::TextureRgba16f => "Textures (RGBA16F)",
            MemoryCategory::TextureR32f => "Textures (R32F)",
            MemoryCategory::TextureDepth => "Depth Textures",
            MemoryCategory::ShadowMaps => "Shadow Maps",
            MemoryCategory::IblCubemaps => "IBL Cubemaps",
            MemoryCategory::IblBrdfLut => "IBL BRDF LUT",
            MemoryCategory::FroxelGrid => "Froxel Grid",
            MemoryCategory::ScreenSpaceEffects => "Screen-space Effects",
            MemoryCategory::Other => "Other",
        }
    }

    pub fn all() -> &'static [MemoryCategory] {
        use MemoryCategory::*;
        &[
            VertexBuffers,
            IndexBuffers,
            UniformBuffers,
            TextureRgba8,
            TextureRgba16f,
            TextureR32f,
            TextureDepth,
            ShadowMaps,
            IblCubemaps,
            IblBrdfLut,
            FroxelGrid,
            ScreenSpaceEffects,
            Other,
        ]
    }
}

/// GPU memory budget tracker
#[derive(Debug)]
pub struct GpuMemoryBudget {
    /// Maximum allowed GPU memory in bytes
    max_budget_bytes: usize,

    /// Current allocations by category (atomic for thread safety)
    allocations: Vec<(MemoryCategory, Arc<AtomicUsize>)>,

    /// Total allocated bytes
    total_allocated: Arc<AtomicUsize>,

    /// Peak allocation seen
    peak_allocated: Arc<AtomicUsize>,

    /// Warning threshold (percentage of budget)
    warning_threshold: f32,

    /// Whether warnings have been printed
    warned: Arc<AtomicUsize>,
}

impl GpuMemoryBudget {
    /// Create a new memory budget tracker
    ///
    /// # Arguments
    /// * `max_budget_mib` - Maximum budget in MiB (default: 512 MiB)
    pub fn new(max_budget_mib: usize) -> Self {
        let max_budget_bytes = max_budget_mib * 1024 * 1024;

        let mut allocations = Vec::new();
        for category in MemoryCategory::all() {
            allocations.push((*category, Arc::new(AtomicUsize::new(0))));
        }

        Self {
            max_budget_bytes,
            allocations,
            total_allocated: Arc::new(AtomicUsize::new(0)),
            peak_allocated: Arc::new(AtomicUsize::new(0)),
            warning_threshold: 0.9, // Warn at 90%
            warned: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Allocate memory in a category
    ///
    /// Returns true if allocation succeeded within budget
    pub fn allocate(&self, category: MemoryCategory, bytes: usize) -> bool {
        // Find category counter
        if let Some((_, counter)) = self.allocations.iter().find(|(cat, _)| *cat == category) {
            counter.fetch_add(bytes, Ordering::SeqCst);
        }

        // Update total
        let new_total = self.total_allocated.fetch_add(bytes, Ordering::SeqCst) + bytes;

        // Update peak
        let mut peak = self.peak_allocated.load(Ordering::SeqCst);
        while new_total > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                new_total,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        // Check budget
        let within_budget = new_total <= self.max_budget_bytes;

        // Check warning threshold
        if !within_budget || (new_total as f32 / self.max_budget_bytes as f32) > self.warning_threshold {
            if self.warned.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                log::warn!(
                    "GPU memory budget: {:.1} MiB / {:.1} MiB ({:.1}%)",
                    new_total as f32 / (1024.0 * 1024.0),
                    self.max_budget_bytes as f32 / (1024.0 * 1024.0),
                    (new_total as f32 / self.max_budget_bytes as f32) * 100.0
                );
            }
        }

        within_budget
    }

    /// Deallocate memory from a category
    pub fn deallocate(&self, category: MemoryCategory, bytes: usize) {
        if let Some((_, counter)) = self.allocations.iter().find(|(cat, _)| *cat == category) {
            counter.fetch_sub(bytes, Ordering::SeqCst);
        }
        self.total_allocated.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Get current allocation for a category (in bytes)
    pub fn get_category_bytes(&self, category: MemoryCategory) -> usize {
        self.allocations
            .iter()
            .find(|(cat, _)| *cat == category)
            .map(|(_, counter)| counter.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Get total allocated bytes
    pub fn get_total_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated bytes
    pub fn get_peak_bytes(&self) -> usize {
        self.peak_allocated.load(Ordering::SeqCst)
    }

    /// Get budget utilization as percentage (0.0 to 1.0+)
    pub fn get_utilization(&self) -> f32 {
        self.get_total_bytes() as f32 / self.max_budget_bytes as f32
    }

    /// Check if allocation would fit within budget
    pub fn would_fit(&self, bytes: usize) -> bool {
        self.get_total_bytes() + bytes <= self.max_budget_bytes
    }

    /// Get available budget in bytes
    pub fn get_available_bytes(&self) -> usize {
        self.max_budget_bytes.saturating_sub(self.get_total_bytes())
    }

    /// Get max budget in bytes
    pub fn get_max_bytes(&self) -> usize {
        self.max_budget_bytes
    }

    /// Get detailed breakdown of allocations
    pub fn get_breakdown(&self) -> Vec<(String, usize, f32)> {
        let total = self.get_total_bytes() as f32;

        self.allocations
            .iter()
            .map(|(cat, counter)| {
                let bytes = counter.load(Ordering::SeqCst);
                let percentage = if total > 0.0 { bytes as f32 / total } else { 0.0 };
                (cat.name().to_string(), bytes, percentage)
            })
            .filter(|(_, bytes, _)| *bytes > 0)
            .collect()
    }
}

/// Performance and memory statistics
#[derive(Debug, Clone)]
pub struct RenderStats {
    /// Total GPU memory allocated (bytes)
    pub gpu_memory_bytes: usize,

    /// Peak GPU memory allocated (bytes)
    pub gpu_memory_peak_bytes: usize,

    /// GPU memory budget (bytes)
    pub gpu_memory_budget_bytes: usize,

    /// GPU memory utilization (0.0 to 1.0+)
    pub gpu_memory_utilization: f32,

    /// Memory breakdown by category
    pub gpu_memory_breakdown: Vec<(String, usize, f32)>,

    /// Last frame time (milliseconds)
    pub frame_time_ms: f32,

    /// Enabled rendering passes
    pub passes_enabled: Vec<String>,
}

impl RenderStats {
    /// Create stats from memory budget
    pub fn from_budget(budget: &GpuMemoryBudget, frame_time_ms: f32, passes: Vec<String>) -> Self {
        Self {
            gpu_memory_bytes: budget.get_total_bytes(),
            gpu_memory_peak_bytes: budget.get_peak_bytes(),
            gpu_memory_budget_bytes: budget.get_max_bytes(),
            gpu_memory_utilization: budget.get_utilization(),
            gpu_memory_breakdown: budget.get_breakdown(),
            frame_time_ms,
            passes_enabled: passes,
        }
    }
}

/// Auto-downscale shadow map resolution to fit budget
pub fn auto_downscale_shadow_map(
    requested_res: u32,
    budget: &GpuMemoryBudget,
    format_bytes_per_pixel: usize,
) -> (u32, bool) {
    let requested_bytes = (requested_res * requested_res) as usize * format_bytes_per_pixel;

    if budget.would_fit(requested_bytes) {
        return (requested_res, false); // No downscaling needed
    }

    // Try progressively smaller resolutions
    let available = budget.get_available_bytes();
    let mut current_res = requested_res;

    while current_res > 512 {
        current_res /= 2;
        let bytes = (current_res * current_res) as usize * format_bytes_per_pixel;
        if bytes <= available {
            log::warn!(
                "Shadow map auto-downscaled: {}x{} → {}x{} to fit GPU budget",
                requested_res, requested_res, current_res, current_res
            );
            return (current_res, true);
        }
    }

    // Minimum viable resolution
    log::warn!(
        "Shadow map auto-downscaled to minimum: {}x{} → 512x512 to fit GPU budget",
        requested_res, requested_res
    );
    (512, true)
}

/// Auto-downscale IBL cubemap resolution to fit budget
pub fn auto_downscale_ibl_cubemap(
    requested_res: u32,
    budget: &GpuMemoryBudget,
) -> (u32, bool) {
    // Cubemap: 6 faces × res² × 8 bytes (RGBA16F) × mip levels
    // Approximate with 1.33x for mip chain
    let mip_multiplier = 1.33;
    let bytes_per_pixel = 8; // RGBA16F
    let requested_bytes = (6 * requested_res * requested_res) as f32 * bytes_per_pixel as f32 * mip_multiplier;

    if budget.would_fit(requested_bytes as usize) {
        return (requested_res, false);
    }

    let available = budget.get_available_bytes();
    let mut current_res = requested_res;

    while current_res > 32 {
        current_res /= 2;
        let bytes = (6 * current_res * current_res) as f32 * bytes_per_pixel as f32 * mip_multiplier;
        if bytes <= available as f32 {
            log::warn!(
                "IBL cubemap auto-downscaled: {} → {} to fit GPU budget",
                requested_res, current_res
            );
            return (current_res, true);
        }
    }

    log::warn!(
        "IBL cubemap auto-downscaled to minimum: {} → 32 to fit GPU budget",
        requested_res
    );
    (32, true)
}

/// Auto-downscale froxel grid dimensions to fit budget
pub fn auto_downscale_froxel_grid(
    requested_dims: (u32, u32, u32),
    budget: &GpuMemoryBudget,
) -> ((u32, u32, u32), bool) {
    let (x, y, z) = requested_dims;
    let bytes_per_froxel = 8; // RGBA16F
    let requested_bytes = (x * y * z) as usize * bytes_per_froxel;

    if budget.would_fit(requested_bytes) {
        return (requested_dims, false);
    }

    // Reduce Z dimension first (depth), then Y, then X
    let available = budget.get_available_bytes();
    let mut current_z = z;

    while current_z > 16 {
        current_z /= 2;
        let bytes = (x * y * current_z) as usize * bytes_per_froxel;
        if bytes <= available {
            log::warn!(
                "Froxel grid auto-downscaled: {}×{}×{} → {}×{}×{} to fit GPU budget",
                x, y, z, x, y, current_z
            );
            return ((x, y, current_z), true);
        }
    }

    // Fallback: minimal froxel grid
    let minimal = (8, 8, 16);
    log::warn!(
        "Froxel grid auto-downscaled to minimum: {}×{}×{} → {}×{}×{} to fit GPU budget",
        x, y, z, minimal.0, minimal.1, minimal.2
    );
    (minimal, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_tracking() {
        let budget = GpuMemoryBudget::new(512); // 512 MiB

        // Allocate some memory
        assert!(budget.allocate(MemoryCategory::ShadowMaps, 16 * 1024 * 1024)); // 16 MiB
        assert_eq!(budget.get_category_bytes(MemoryCategory::ShadowMaps), 16 * 1024 * 1024);
        assert_eq!(budget.get_total_bytes(), 16 * 1024 * 1024);

        // Deallocate
        budget.deallocate(MemoryCategory::ShadowMaps, 16 * 1024 * 1024);
        assert_eq!(budget.get_total_bytes(), 0);
    }

    #[test]
    fn test_budget_overflow() {
        let budget = GpuMemoryBudget::new(10); // 10 MiB budget

        // Allocate within budget
        assert!(budget.allocate(MemoryCategory::ShadowMaps, 5 * 1024 * 1024));

        // Try to exceed budget
        assert!(!budget.allocate(MemoryCategory::IblCubemaps, 10 * 1024 * 1024));

        // Should still track the allocation even if over budget
        assert!(budget.get_total_bytes() > budget.get_max_bytes());
    }

    #[test]
    fn test_auto_downscale_shadow_map() {
        let budget = GpuMemoryBudget::new(10); // 10 MiB

        // Fill most of budget
        budget.allocate(MemoryCategory::Other, 9 * 1024 * 1024);

        // Try to allocate 4096² shadow map (4 bytes/pixel = 64 MiB)
        let (res, downscaled) = auto_downscale_shadow_map(4096, &budget, 4);

        assert!(downscaled);
        assert!(res < 4096);

        // Verify it actually fits
        let bytes = (res * res) as usize * 4;
        assert!(bytes <= budget.get_available_bytes());
    }

    #[test]
    fn test_utilization() {
        let budget = GpuMemoryBudget::new(100); // 100 MiB

        budget.allocate(MemoryCategory::ShadowMaps, 50 * 1024 * 1024);

        assert!((budget.get_utilization() - 0.5).abs() < 0.01); // ~50%
    }

    #[test]
    fn test_breakdown() {
        let budget = GpuMemoryBudget::new(512);

        budget.allocate(MemoryCategory::ShadowMaps, 16 * 1024 * 1024);
        budget.allocate(MemoryCategory::IblCubemaps, 8 * 1024 * 1024);

        let breakdown = budget.get_breakdown();
        assert_eq!(breakdown.len(), 2);

        // Find shadow maps entry
        let shadows = breakdown.iter().find(|(name, _, _)| name == "Shadow Maps");
        assert!(shadows.is_some());

        let (_, bytes, pct) = shadows.unwrap();
        assert_eq!(*bytes, 16 * 1024 * 1024);
        assert!((pct - 0.666).abs() < 0.01); // 16/(16+8) ≈ 66.6%
    }
}
