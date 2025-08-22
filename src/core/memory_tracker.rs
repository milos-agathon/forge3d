//! Memory tracking and budget enforcement for GPU resources.
//!
//! This module provides facilities to track GPU memory allocations and enforce
//! budget limits, with particular focus on host-visible memory which is often
//! the most constrained resource.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use wgpu::{TextureFormat, BufferUsages};

/// Error type for budget enforcement violations.
#[derive(thiserror::Error, Debug)]
pub enum BudgetError {
    #[error("Memory budget exceeded: current {current_bytes} bytes + requested {requested_bytes} bytes would exceed limit of {limit_bytes} bytes (host-visible: {host_visible_bytes} bytes)")]
    BudgetExceeded {
        current_bytes: u64,
        requested_bytes: u64,
        limit_bytes: u64,
        host_visible_bytes: u64,
    },
}

/// Memory usage metrics for reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMetrics {
    /// Total number of tracked buffers
    pub buffer_count: usize,
    /// Total number of tracked textures  
    pub texture_count: usize,
    /// Total bytes in all tracked buffers
    pub buffer_bytes: u64,
    /// Total bytes in all tracked textures
    pub texture_bytes: u64,
    /// Total bytes in host-visible resources
    pub host_visible_bytes: u64,
    /// Current memory budget limit in bytes
    pub limit_bytes: u64,
    /// Whether current usage is within budget
    pub within_budget: bool,
}

impl MemoryMetrics {
    /// Get total tracked bytes across all resource types.
    pub fn total_bytes(&self) -> u64 {
        self.buffer_bytes + self.texture_bytes
    }
    
    /// Get utilization as a percentage of the budget (0.0 to 1.0+).
    pub fn utilization_ratio(&self) -> f64 {
        if self.limit_bytes == 0 {
            0.0
        } else {
            self.host_visible_bytes as f64 / self.limit_bytes as f64
        }
    }
}

/// Central registry for tracking GPU resource memory usage.
/// 
/// Uses atomic operations for thread-safe tracking without locks.
/// Focuses on host-visible memory as the primary budget constraint.
#[derive(Debug)]
pub struct ResourceRegistry {
    // Counters for different resource types
    buffer_count: AtomicUsize,
    texture_count: AtomicUsize,
    
    // Byte counters by category
    buffer_bytes: AtomicU64,
    texture_bytes: AtomicU64,
    host_visible_bytes: AtomicU64,
    
    // Budget limit (0 = no limit)
    limit_bytes: AtomicU64,
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceRegistry {
    /// Create a new empty resource registry.
    pub fn new() -> Self {
        Self {
            buffer_count: AtomicUsize::new(0),
            texture_count: AtomicUsize::new(0),
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
            host_visible_bytes: AtomicU64::new(0),
            limit_bytes: AtomicU64::new(0),
        }
    }
    
    /// Set the memory budget limit in bytes.
    /// Set to 0 to disable budget enforcement.
    pub fn set_budget_limit(&self, limit_bytes: u64) {
        self.limit_bytes.store(limit_bytes, Ordering::Relaxed);
    }
    
    /// Get the current budget limit in bytes.
    pub fn get_budget_limit(&self) -> u64 {
        self.limit_bytes.load(Ordering::Relaxed)
    }
    
    /// Track allocation of a buffer.
    /// 
    /// # Arguments
    /// * `size` - Size of the buffer in bytes
    /// * `host_visible` - Whether the buffer is host-visible (CPU accessible)
    pub fn track_buffer_allocation(&self, size: u64, host_visible: bool) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_add(size, Ordering::Relaxed);
        
        if host_visible {
            self.host_visible_bytes.fetch_add(size, Ordering::Relaxed);
        }
    }
    
    /// Track deallocation of a buffer.
    pub fn free_buffer_allocation(&self, size: u64, host_visible: bool) {
        self.buffer_count.fetch_sub(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_sub(size, Ordering::Relaxed);
        
        if host_visible {
            self.host_visible_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }
    
    /// Track allocation of a texture.
    /// 
    /// # Arguments  
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `format` - Texture format (determines bytes per pixel)
    pub fn track_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let bytes_per_pixel = texture_format_bytes_per_pixel(format);
        let size = (width as u64) * (height as u64) * (bytes_per_pixel as u64);
        
        self.texture_count.fetch_add(1, Ordering::Relaxed);
        self.texture_bytes.fetch_add(size, Ordering::Relaxed);
        
        // Textures are generally not host-visible, but readback textures might be
        // For now, we don't track textures as host-visible unless explicitly specified
    }
    
    /// Track deallocation of a texture.
    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let bytes_per_pixel = texture_format_bytes_per_pixel(format);
        let size = (width as u64) * (height as u64) * (bytes_per_pixel as u64);
        
        self.texture_count.fetch_sub(1, Ordering::Relaxed);
        self.texture_bytes.fetch_sub(size, Ordering::Relaxed);
    }
    
    /// Check if allocating additional memory would exceed the budget.
    /// 
    /// Returns Ok(()) if within budget, Err(BudgetError) if it would exceed.
    pub fn check_budget_limits(&self, additional_bytes: u64) -> Result<(), BudgetError> {
        let limit = self.limit_bytes.load(Ordering::Relaxed);
        
        // No limit set
        if limit == 0 {
            return Ok(());
        }
        
        let current_host_visible = self.host_visible_bytes.load(Ordering::Relaxed);
        let would_be_total = current_host_visible + additional_bytes;
        
        if would_be_total > limit {
            return Err(BudgetError::BudgetExceeded {
                current_bytes: current_host_visible,
                requested_bytes: additional_bytes,
                limit_bytes: limit,
                host_visible_bytes: current_host_visible,
            });
        }
        
        Ok(())
    }
    
    /// Get current memory usage metrics.
    pub fn get_metrics(&self) -> MemoryMetrics {
        let buffer_count = self.buffer_count.load(Ordering::Relaxed);
        let texture_count = self.texture_count.load(Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        let host_visible_bytes = self.host_visible_bytes.load(Ordering::Relaxed);
        let limit_bytes = self.limit_bytes.load(Ordering::Relaxed);
        
        let within_budget = limit_bytes == 0 || host_visible_bytes <= limit_bytes;
        
        MemoryMetrics {
            buffer_count,
            texture_count,
            buffer_bytes,
            texture_bytes,
            host_visible_bytes,
            limit_bytes,
            within_budget,
        }
    }
    
    /// Reset all counters to zero (for testing).
    #[cfg(test)]
    pub fn reset(&self) {
        self.buffer_count.store(0, Ordering::Relaxed);
        self.texture_count.store(0, Ordering::Relaxed);
        self.buffer_bytes.store(0, Ordering::Relaxed);
        self.texture_bytes.store(0, Ordering::Relaxed);
        self.host_visible_bytes.store(0, Ordering::Relaxed);
        self.limit_bytes.store(0, Ordering::Relaxed);
    }
}

/// Determine if a buffer usage pattern indicates host visibility.
pub fn is_host_visible_usage(usage: BufferUsages) -> bool {
    usage.contains(BufferUsages::MAP_READ) || usage.contains(BufferUsages::MAP_WRITE)
}

/// Get bytes per pixel for common texture formats.
fn texture_format_bytes_per_pixel(format: TextureFormat) -> u32 {
    match format {
        // 8-bit formats
        TextureFormat::R8Unorm | TextureFormat::R8Snorm | TextureFormat::R8Uint | TextureFormat::R8Sint => 1,
        
        // 16-bit formats
        TextureFormat::R16Uint | TextureFormat::R16Sint | TextureFormat::R16Float => 2,
        TextureFormat::Rg8Unorm | TextureFormat::Rg8Snorm | TextureFormat::Rg8Uint | TextureFormat::Rg8Sint => 2,
        
        // 32-bit formats
        TextureFormat::R32Uint | TextureFormat::R32Sint | TextureFormat::R32Float => 4,
        TextureFormat::Rg16Uint | TextureFormat::Rg16Sint | TextureFormat::Rg16Float => 4,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Snorm | 
        TextureFormat::Rgba8Uint | TextureFormat::Rgba8Sint => 4,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => 4,
        
        // 64-bit formats
        TextureFormat::Rg32Uint | TextureFormat::Rg32Sint | TextureFormat::Rg32Float => 8,
        TextureFormat::Rgba16Uint | TextureFormat::Rgba16Sint | TextureFormat::Rgba16Float => 8,
        
        // 128-bit formats
        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,
        
        // Depth/stencil formats
        TextureFormat::Depth16Unorm => 2,
        TextureFormat::Depth24Plus => 4, // Implementation dependent, estimate 4
        TextureFormat::Depth24PlusStencil8 => 4,
        TextureFormat::Depth32Float => 4,
        TextureFormat::Depth32FloatStencil8 => 8,
        
        // Compressed formats - return approximate bytes per pixel
        // These are estimates since compression ratios vary
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 1, // 4:1 compression ratio estimate
        TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb => 1,
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => 1,
        TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => 1,
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 1,
        TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => 1,
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => 1,
        
        // ETC2/EAC formats
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => 1,
        TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb => 1,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 1,
        TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm => 1,
        TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => 1,
        
        // ASTC formats - highly variable compression
        TextureFormat::Astc { .. } => 1, // Very rough estimate
        
        // Fallback for any new formats
        _ => 4, // Conservative estimate
    }
}

/// Global memory tracker instance.
static GLOBAL_TRACKER: once_cell::sync::Lazy<Arc<ResourceRegistry>> = 
    once_cell::sync::Lazy::new(|| Arc::new(ResourceRegistry::new()));

/// Get a reference to the global memory tracker.
pub fn global_tracker() -> Arc<ResourceRegistry> {
    GLOBAL_TRACKER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resource_registry_basic() {
        let registry = ResourceRegistry::new();
        
        // Initially empty
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.texture_count, 0);
        assert_eq!(metrics.total_bytes(), 0);
        assert_eq!(metrics.host_visible_bytes, 0);
        assert!(metrics.within_budget);
        
        // Track a buffer
        registry.track_buffer_allocation(1024, true);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 1);
        assert_eq!(metrics.buffer_bytes, 1024);
        assert_eq!(metrics.host_visible_bytes, 1024);
        
        // Track a texture
        registry.track_texture_allocation(256, 256, TextureFormat::Rgba8Unorm);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.texture_count, 1);
        assert_eq!(metrics.texture_bytes, 256 * 256 * 4);
        
        // Free resources
        registry.free_buffer_allocation(1024, true);
        registry.free_texture_allocation(256, 256, TextureFormat::Rgba8Unorm);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.texture_count, 0);
        assert_eq!(metrics.total_bytes(), 0);
    }
    
    #[test]
    fn test_budget_enforcement() {
        let registry = ResourceRegistry::new();
        registry.set_budget_limit(1024);
        
        // Should pass when under budget
        assert!(registry.check_budget_limits(512).is_ok());
        
        // Track allocation that uses half the budget
        registry.track_buffer_allocation(512, true);
        assert!(registry.check_budget_limits(512).is_ok());
        
        // Should fail when would exceed budget
        let result = registry.check_budget_limits(513);
        assert!(result.is_err());
        
        if let Err(BudgetError::BudgetExceeded { current_bytes, requested_bytes, limit_bytes, .. }) = result {
            assert_eq!(current_bytes, 512);
            assert_eq!(requested_bytes, 513);
            assert_eq!(limit_bytes, 1024);
        }
    }
    
    #[test]
    fn test_host_visible_usage_detection() {
        assert!(is_host_visible_usage(BufferUsages::MAP_READ));
        assert!(is_host_visible_usage(BufferUsages::MAP_WRITE));
        assert!(is_host_visible_usage(BufferUsages::MAP_READ | BufferUsages::COPY_DST));
        assert!(!is_host_visible_usage(BufferUsages::VERTEX));
        assert!(!is_host_visible_usage(BufferUsages::UNIFORM));
    }
    
    #[test]
    fn test_texture_format_bytes_calculation() {
        assert_eq!(texture_format_bytes_per_pixel(TextureFormat::R8Unorm), 1);
        assert_eq!(texture_format_bytes_per_pixel(TextureFormat::Rgba8Unorm), 4);
        assert_eq!(texture_format_bytes_per_pixel(TextureFormat::R32Float), 4);
        assert_eq!(texture_format_bytes_per_pixel(TextureFormat::Rgba32Float), 16);
    }
    
    #[test]
    fn test_metrics_utilization() {
        let registry = ResourceRegistry::new();
        registry.set_budget_limit(1000);
        registry.track_buffer_allocation(250, true);
        
        let metrics = registry.get_metrics();
        assert_eq!(metrics.utilization_ratio(), 0.25);
        assert!(metrics.within_budget);
    }
}