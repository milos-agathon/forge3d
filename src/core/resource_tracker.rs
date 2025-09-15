use crate::core::memory_tracker::{global_tracker, is_host_visible_usage};
use wgpu::{BufferUsages, TextureFormat};

/// Resource handle that automatically unregisters on drop
#[derive(Debug)]
pub enum ResourceHandle {
    Buffer {
        size: u64,
        is_host_visible: bool,
    },
    Texture {
        width: u32,
        height: u32,
        format: TextureFormat,
    },
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        let tracker = global_tracker();
        match self {
            ResourceHandle::Buffer {
                size,
                is_host_visible,
            } => {
                tracker.free_buffer_allocation(*size, *is_host_visible);
            }
            ResourceHandle::Texture {
                width,
                height,
                format,
            } => {
                tracker.free_texture_allocation(*width, *height, *format);
            }
        }
    }
}

/// Register a buffer allocation and return a handle that will unregister on drop
pub fn register_buffer(size: u64, usage: BufferUsages) -> ResourceHandle {
    let is_host_visible = is_host_visible_usage(usage);
    let tracker = global_tracker();
    tracker.track_buffer_allocation(size, is_host_visible);
    ResourceHandle::Buffer {
        size,
        is_host_visible,
    }
}

/// Register a texture allocation and return a handle that will unregister on drop  
pub fn register_texture(width: u32, height: u32, format: TextureFormat) -> ResourceHandle {
    let tracker = global_tracker();
    tracker.track_texture_allocation(width, height, format);
    ResourceHandle::Texture {
        width,
        height,
        format,
    }
}

/// Register a buffer allocation with explicit host-visible flag
pub fn register_buffer_explicit(size: u64, is_host_visible: bool) -> ResourceHandle {
    let tracker = global_tracker();
    tracker.track_buffer_allocation(size, is_host_visible);
    ResourceHandle::Buffer {
        size,
        is_host_visible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_tracker::ResourceRegistry;

    #[test]
    fn test_resource_handle_cleanup() {
        // Test with isolated registry (can't easily test global one)
        let registry = ResourceRegistry::new();

        // Test buffer handle
        {
            let handle = ResourceHandle::Buffer {
                size: 1024,
                is_host_visible: true,
            };

            // Manually track allocation to simulate what register_buffer does
            registry.track_buffer_allocation(1024, true);

            let metrics = registry.get_metrics();
            assert_eq!(metrics.buffer_count, 1);
            assert_eq!(metrics.buffer_bytes, 1024);
            assert_eq!(metrics.host_visible_bytes, 1024);

            // Now drop the handle (but it will call global_tracker, not our local registry)
            drop(handle);
        }
    }

    #[test]
    fn test_register_buffer_helper() {
        let usage = BufferUsages::COPY_DST | BufferUsages::MAP_READ;
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_buffer(2048, usage);
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation
            assert_eq!(
                after_alloc_metrics.buffer_count,
                initial_metrics.buffer_count + 1
            );
            assert_eq!(
                after_alloc_metrics.buffer_bytes,
                initial_metrics.buffer_bytes + 2048
            );
            assert_eq!(
                after_alloc_metrics.host_visible_bytes,
                initial_metrics.host_visible_bytes + 2048
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.buffer_count, initial_metrics.buffer_count);
        assert_eq!(final_metrics.buffer_bytes, initial_metrics.buffer_bytes);
        assert_eq!(
            final_metrics.host_visible_bytes,
            initial_metrics.host_visible_bytes
        );
    }

    #[test]
    fn test_register_texture_helper() {
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_texture(512, 512, TextureFormat::Rgba8Unorm);
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation (512*512*4 = 1,048,576 bytes)
            assert_eq!(
                after_alloc_metrics.texture_count,
                initial_metrics.texture_count + 1
            );
            assert_eq!(
                after_alloc_metrics.texture_bytes,
                initial_metrics.texture_bytes + 1_048_576
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.texture_count, initial_metrics.texture_count);
        assert_eq!(final_metrics.texture_bytes, initial_metrics.texture_bytes);
    }
}
