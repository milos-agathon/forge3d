//! GPU feedback buffer system for virtual texture streaming
//!
//! This module provides GPU → CPU communication for tile visibility feedback,
//! allowing the virtual texture system to know which tiles are actually being used.

use crate::core::tile_cache::TileId;
use std::collections::HashSet;
use wgpu::{
    BindGroup, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource,
};

/// GPU feedback buffer for collecting tile visibility information
pub struct FeedbackBuffer {
    /// GPU buffer for collecting feedback data from shaders
    feedback_buffer: Buffer,
    /// CPU-readable staging buffer for feedback readback  
    readback_buffer: Buffer,
    /// Maximum number of feedback entries
    max_entries: u32,
    /// Feedback compute pipeline
    compute_pipeline: Option<ComputePipeline>,
    /// Bind group for feedback computation
    bind_group: Option<BindGroup>,
}

/// Feedback entry structure (matches GPU layout)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct FeedbackEntry {
    /// Tile X coordinate
    pub tile_x: u32,
    /// Tile Y coordinate  
    pub tile_y: u32,
    /// Mip level
    pub mip_level: u32,
    /// Frame number when accessed
    pub frame_number: u32,
}

impl FeedbackBuffer {
    /// Create new feedback buffer
    pub fn new(device: &Device, max_tiles: u32) -> Result<Self, String> {
        let entry_size = std::mem::size_of::<FeedbackEntry>() as u64;
        let buffer_size = entry_size * max_tiles as u64;

        // Create GPU feedback buffer
        let feedback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("FeedbackBuffer_GPU"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create CPU readback buffer
        let readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("FeedbackBuffer_Readback"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            feedback_buffer,
            readback_buffer,
            max_entries: max_tiles,
            compute_pipeline: None,
            bind_group: None,
        })
    }

    /// Initialize feedback compute pipeline
    pub fn init_compute_pipeline(
        &mut self,
        device: &Device,
        page_table_buffer: &Buffer,
    ) -> Result<(), String> {
        // Create shader for feedback processing
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("FeedbackBuffer_Shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/virtual_texture_feedback.wgsl").into(),
            ),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FeedbackBuffer_BindGroupLayout"),
            entries: &[
                // Feedback buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Page table buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("FeedbackBuffer_PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FeedbackBuffer_Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_feedback",
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FeedbackBuffer_BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.feedback_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: page_table_buffer.as_entire_binding(),
                },
            ],
        });

        self.compute_pipeline = Some(compute_pipeline);
        self.bind_group = Some(bind_group);

        Ok(())
    }

    /// Clear feedback buffer for new frame
    pub fn clear(&self, encoder: &mut CommandEncoder) {
        // Clear feedback buffer by writing zeros
        encoder.clear_buffer(&self.feedback_buffer, 0, None);
    }

    /// Process feedback data on GPU
    pub fn process_feedback(&self, encoder: &mut CommandEncoder) -> Result<(), String> {
        if let (Some(pipeline), Some(bind_group)) = (&self.compute_pipeline, &self.bind_group) {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FeedbackBuffer_Process"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            // Dispatch threads to process feedback entries
            let workgroup_size = 64;
            let num_workgroups = (self.max_entries + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        } else {
            return Err("Feedback compute pipeline not initialized".to_string());
        }

        Ok(())
    }

    /// Copy feedback data to readback buffer
    pub fn prepare_readback(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.feedback_buffer,
            0,
            &self.readback_buffer,
            0,
            self.feedback_buffer.size(),
        );
    }

    /// Read feedback data from GPU (async)
    pub async fn read_feedback_async(&self, device: &Device) -> Result<Vec<TileId>, String> {
        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive feedback data: {}", e))?
            .map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_data(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Read feedback data from GPU (blocking)
    pub fn read_feedback(&self, device: &Device, _queue: &Queue) -> Result<Vec<TileId>, String> {
        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive feedback data: {}", e))?
            .map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_data(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Parse raw feedback data into tile IDs
    fn parse_feedback_data(&self, data: &[u8]) -> Vec<TileId> {
        let entry_size = std::mem::size_of::<FeedbackEntry>();
        let mut tile_ids = HashSet::new();

        for chunk in data.chunks_exact(entry_size) {
            if chunk.len() == entry_size {
                // Safety: We know the chunk is exactly the size of FeedbackEntry
                let entry: FeedbackEntry =
                    unsafe { std::ptr::read(chunk.as_ptr() as *const FeedbackEntry) };

                // Filter out invalid/empty entries
                if entry.frame_number > 0 && entry.tile_x != u32::MAX && entry.tile_y != u32::MAX {
                    tile_ids.insert(TileId {
                        x: entry.tile_x,
                        y: entry.tile_y,
                        mip_level: entry.mip_level,
                    });
                }
            }
        }

        tile_ids.into_iter().collect()
    }

    /// Get bind group for shader binding
    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get feedback buffer for direct shader access
    pub fn buffer(&self) -> &Buffer {
        &self.feedback_buffer
    }

    /// Get maximum number of feedback entries
    pub fn max_entries(&self) -> u32 {
        self.max_entries
    }
}

/// Feedback statistics and performance metrics
#[derive(Debug, Clone, Default)]
pub struct FeedbackStats {
    /// Number of feedback entries processed this frame
    pub entries_processed: u32,
    /// Number of unique tiles in feedback
    pub unique_tiles: u32,
    /// Time spent processing feedback in milliseconds
    pub process_time_ms: f32,
    /// Time spent reading back feedback in milliseconds
    pub readback_time_ms: f32,
}

/// High-level feedback manager
pub struct FeedbackManager {
    /// Feedback buffer
    buffer: FeedbackBuffer,
    /// Current frame number
    frame_number: u32,
    /// Statistics
    stats: FeedbackStats,
}

impl FeedbackManager {
    /// Create new feedback manager
    pub fn new(device: &Device, max_tiles: u32) -> Result<Self, String> {
        let buffer = FeedbackBuffer::new(device, max_tiles)?;

        Ok(Self {
            buffer,
            frame_number: 0,
            stats: FeedbackStats::default(),
        })
    }

    /// Initialize with page table buffer
    pub fn initialize(
        &mut self,
        device: &Device,
        page_table_buffer: &Buffer,
    ) -> Result<(), String> {
        self.buffer.init_compute_pipeline(device, page_table_buffer)
    }

    /// Begin new frame
    pub fn begin_frame(&mut self, encoder: &mut CommandEncoder) {
        self.frame_number += 1;
        self.buffer.clear(encoder);
    }

    /// Process feedback for current frame
    pub fn process_frame(&mut self, encoder: &mut CommandEncoder) -> Result<(), String> {
        let start_time = std::time::Instant::now();

        self.buffer.process_feedback(encoder)?;
        self.buffer.prepare_readback(encoder);

        self.stats.process_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        Ok(())
    }

    /// Read feedback results
    pub fn read_results(&mut self, device: &Device, queue: &Queue) -> Result<Vec<TileId>, String> {
        let start_time = std::time::Instant::now();

        let tiles = self.buffer.read_feedback(device, queue)?;

        self.stats.readback_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.stats.entries_processed = tiles.len() as u32;
        self.stats.unique_tiles = tiles.len() as u32; // Already deduplicated

        Ok(tiles)
    }

    /// Get current statistics
    pub fn stats(&self) -> &FeedbackStats {
        &self.stats
    }

    /// Get current frame number
    pub fn frame_number(&self) -> u32 {
        self.frame_number
    }

    /// Get underlying buffer
    pub fn buffer(&self) -> &FeedbackBuffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_entry_size() {
        // Ensure FeedbackEntry has expected size for GPU compatibility
        assert_eq!(std::mem::size_of::<FeedbackEntry>(), 16); // 4 u32s = 16 bytes
    }

    #[test]
    fn test_feedback_entry_creation() {
        let entry = FeedbackEntry {
            tile_x: 10,
            tile_y: 20,
            mip_level: 2,
            frame_number: 100,
        };

        assert_eq!(entry.tile_x, 10);
        assert_eq!(entry.tile_y, 20);
        assert_eq!(entry.mip_level, 2);
        assert_eq!(entry.frame_number, 100);
    }

    #[test]
    fn test_feedback_stats_default() {
        let stats = FeedbackStats::default();

        assert_eq!(stats.entries_processed, 0);
        assert_eq!(stats.unique_tiles, 0);
        assert_eq!(stats.process_time_ms, 0.0);
        assert_eq!(stats.readback_time_ms, 0.0);
    }

    #[test]
    fn test_parse_empty_feedback_data() {
        // Create a minimal feedback buffer for testing parsing logic
        let device = pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            let (device, _) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .unwrap();
            device
        });

        let buffer = FeedbackBuffer::new(&device, 10).unwrap();

        // Test with empty data
        let empty_data = vec![0u8; 0];
        let tiles = buffer.parse_feedback_data(&empty_data);
        assert!(tiles.is_empty());

        // Test with invalid data (all zeros)
        let zero_data = vec![0u8; std::mem::size_of::<FeedbackEntry>()];
        let tiles = buffer.parse_feedback_data(&zero_data);
        assert!(tiles.is_empty()); // Should filter out invalid entries
    }
}
