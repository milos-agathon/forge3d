use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use wgpu::{BufferUsages, TextureFormat};

/// Global memory tracking registry for GPU resources
pub struct ResourceRegistry {
    // Atomic counters for thread-safe tracking
    buffer_count: AtomicU32,
    texture_count: AtomicU32,
    buffer_bytes: AtomicU64,
    texture_bytes: AtomicU64,
    host_visible_bytes: AtomicU64,
    
    // Budget limit (512 MiB)
    budget_limit: u64,
}

/// Memory metrics returned to Python
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub buffer_count: u32,
    pub texture_count: u32,
    pub buffer_bytes: u64,
    pub texture_bytes: u64,
    pub host_visible_bytes: u64,
    pub total_bytes: u64,
    pub limit_bytes: u64,
    pub within_budget: bool,
    pub utilization_ratio: f64,
}

impl ResourceRegistry {
    /// Create new registry with 512 MiB budget limit
    pub fn new() -> Self {
        Self {
            buffer_count: AtomicU32::new(0),
            texture_count: AtomicU32::new(0),
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
            host_visible_bytes: AtomicU64::new(0),
            budget_limit: 512 * 1024 * 1024, // 512 MiB
        }
    }
    
    /// Track a buffer allocation
    pub fn track_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_add(size, Ordering::Relaxed);
        
        if is_host_visible {
            self.host_visible_bytes.fetch_add(size, Ordering::Relaxed);
        }
    }
    
    /// Free a buffer allocation
    pub fn free_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_sub(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_sub(size, Ordering::Relaxed);
        
        if is_host_visible {
            self.host_visible_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }
    
    /// Track a texture allocation
    pub fn track_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        
        self.texture_count.fetch_add(1, Ordering::Relaxed);
        self.texture_bytes.fetch_add(size, Ordering::Relaxed);
        // Textures are typically not host-visible in our usage
    }
    
    /// Free a texture allocation
    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        
        self.texture_count.fetch_sub(1, Ordering::Relaxed);
        self.texture_bytes.fetch_sub(size, Ordering::Relaxed);
    }
    
    /// Get current memory metrics
    pub fn get_metrics(&self) -> MemoryMetrics {
        let buffer_count = self.buffer_count.load(Ordering::Relaxed);
        let texture_count = self.texture_count.load(Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        let host_visible_bytes = self.host_visible_bytes.load(Ordering::Relaxed);
        let total_bytes = buffer_bytes + texture_bytes;
        let within_budget = host_visible_bytes <= self.budget_limit;
        let utilization_ratio = host_visible_bytes as f64 / self.budget_limit as f64;
        
        MemoryMetrics {
            buffer_count,
            texture_count,
            buffer_bytes,
            texture_bytes,
            host_visible_bytes,
            total_bytes,
            limit_bytes: self.budget_limit,
            within_budget,
            utilization_ratio,
        }
    }
    
    /// Get budget limit
    pub fn get_budget_limit(&self) -> u64 {
        self.budget_limit
    }
    
    /// Check if allocation would exceed budget
    pub fn check_budget(&self, additional_host_visible: u64) -> Result<(), String> {
        let current = self.host_visible_bytes.load(Ordering::Relaxed);
        if current + additional_host_visible > self.budget_limit {
            return Err(format!(
                "Memory budget exceeded: current {} bytes + requested {} bytes would exceed limit of {} bytes",
                current, additional_host_visible, self.budget_limit
            ));
        }
        Ok(())
    }
}

/// Calculate texture size in bytes based on dimensions and format
fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    let bytes_per_pixel = match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb | TextureFormat::R32Float => 4,
        TextureFormat::Rg8Unorm => 2,
        TextureFormat::R8Unorm => 1,
        // Add more formats as needed
        _ => 4, // Conservative estimate
    };
    
    (width as u64) * (height as u64) * bytes_per_pixel
}

/// Check if buffer usage indicates host-visible memory
pub fn is_host_visible_usage(usage: BufferUsages) -> bool {
    usage.contains(BufferUsages::MAP_READ) || usage.contains(BufferUsages::MAP_WRITE)
}

/// Global singleton registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<ResourceRegistry> = std::sync::OnceLock::new();

/// Get reference to global memory tracker
pub fn global_tracker() -> &'static ResourceRegistry {
    GLOBAL_REGISTRY.get_or_init(|| ResourceRegistry::new())

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_basic_operations() {
        let registry = ResourceRegistry::new();
        
        // Initial state
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.buffer_bytes, 0);
        assert!(metrics.within_budget);
        
        // Track allocation
        registry.track_buffer_allocation(1024, true);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 1);
        assert_eq!(metrics.buffer_bytes, 1024);
        assert_eq!(metrics.host_visible_bytes, 1024);
        
        // Free allocation
        registry.free_buffer_allocation(1024, true);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.buffer_bytes, 0);
        assert_eq!(metrics.host_visible_bytes, 0);
    }
    
    #[test]
    fn test_budget_checking() {
        let registry = ResourceRegistry::new();
        
        // Should pass within budget
        assert!(registry.check_budget(100 * 1024 * 1024).is_ok());
        
        // Should fail when exceeding budget
        assert!(registry.check_budget(600 * 1024 * 1024).is_err());
    }
    
    #[test]
    fn test_host_visible_detection() {
        assert!(is_host_visible_usage(BufferUsages::MAP_READ));
        assert!(is_host_visible_usage(BufferUsages::MAP_WRITE));
        assert!(is_host_visible_usage(BufferUsages::COPY_DST | BufferUsages::MAP_READ));
        assert!(!is_host_visible_usage(BufferUsages::VERTEX));
        assert!(!is_host_visible_usage(BufferUsages::INDEX));
    }
}