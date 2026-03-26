//! GPU feedback buffer system for virtual texture streaming
//!
//! This module provides GPU -> CPU communication for tile visibility feedback.
//! Terrain material VT writes feedback entries directly from the render shader,
//! then this buffer stages the data back to the CPU for residency updates.

use crate::core::tile_cache::TileId;
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue};

/// GPU feedback buffer for collecting tile visibility information
pub struct FeedbackBuffer {
    /// GPU buffer for collecting feedback data from shaders
    feedback_buffer: Buffer,
    /// CPU-readable staging buffer for feedback readback  
    readback_buffer: Buffer,
}

/// Feedback entry structure (matches GPU layout)
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct FeedbackEntry {
    /// Tile X coordinate
    pub tile_x: u32,
    /// Tile Y coordinate  
    pub tile_y: u32,
    /// Mip level
    pub mip_level: u32,
    /// Caller-defined payload. Terrain VT uses this as `material_index + 1`.
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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
        })
    }

    /// Clear feedback buffer for new frame
    pub fn clear(&self, encoder: &mut CommandEncoder) {
        // Clear feedback buffer by writing zeros
        encoder.clear_buffer(&self.feedback_buffer, 0, None);
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
        let entries = self.parse_feedback_tile_ids(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Read feedback data from GPU (blocking)
    pub fn read_feedback(&self, device: &Device, _queue: &Queue) -> Result<Vec<TileId>, String> {
        self.read_feedback_entries(device, _queue).map(|entries| {
            entries
                .into_iter()
                .map(|entry| TileId {
                    x: entry.tile_x,
                    y: entry.tile_y,
                    mip_level: entry.mip_level,
                })
                .collect()
        })
    }

    /// Read raw feedback entries from GPU (blocking)
    pub fn read_feedback_entries(
        &self,
        device: &Device,
        _queue: &Queue,
    ) -> Result<Vec<FeedbackEntry>, String> {
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
        let entries = self.parse_feedback_entries(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Parse raw feedback data into deduplicated feedback entries.
    fn parse_feedback_entries(&self, data: &[u8]) -> Vec<FeedbackEntry> {
        let entry_size = std::mem::size_of::<FeedbackEntry>();
        let mut unique_entries = HashSet::new();

        let mut chunks = data.chunks_exact(entry_size);
        for chunk in &mut chunks {
            let entry_bytes = match bytemuck::try_from_bytes::<FeedbackEntry>(chunk) {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let entry = *entry_bytes;

            if entry.frame_number > 0 && entry.tile_x != u32::MAX && entry.tile_y != u32::MAX {
                unique_entries.insert((
                    entry.tile_x,
                    entry.tile_y,
                    entry.mip_level,
                    entry.frame_number,
                ));
            }
        }

        if !chunks.remainder().is_empty() {
            log::warn!(
                "feedback_buffer: discarded {} trailing bytes from GPU feedback stream",
                chunks.remainder().len()
            );
        }

        unique_entries
            .into_iter()
            .map(|(tile_x, tile_y, mip_level, frame_number)| FeedbackEntry {
                tile_x,
                tile_y,
                mip_level,
                frame_number,
            })
            .collect()
    }

    fn parse_feedback_tile_ids(&self, data: &[u8]) -> Vec<TileId> {
        self.parse_feedback_entries(data)
            .into_iter()
            .map(|entry| TileId {
                x: entry.tile_x,
                y: entry.tile_y,
                mip_level: entry.mip_level,
            })
            .collect()
    }

    /// Get feedback buffer for direct shader access
    pub fn buffer(&self) -> &Buffer {
        &self.feedback_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_entry_size() {
        assert_eq!(std::mem::size_of::<FeedbackEntry>(), 16);
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
    fn test_parse_empty_feedback_data() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            return;
        };

        let buffer = FeedbackBuffer::new(&device, 10).unwrap();

        let empty_data = vec![0u8; 0];
        let tiles = buffer.parse_feedback_tile_ids(&empty_data);
        assert!(tiles.is_empty());

        let zero_data = vec![0u8; std::mem::size_of::<FeedbackEntry>()];
        let tiles = buffer.parse_feedback_tile_ids(&zero_data);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_parse_feedback_trailing_bytes() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            return;
        };

        let buffer = FeedbackBuffer::new(&device, 4).unwrap();
        let entry = FeedbackEntry {
            tile_x: 3,
            tile_y: 9,
            mip_level: 1,
            frame_number: 77,
        };
        let mut bytes = bytemuck::bytes_of(&entry).to_vec();
        bytes.extend_from_slice(&[0xAA, 0xBB]);

        let tiles = buffer.parse_feedback_tile_ids(&bytes);
        assert_eq!(tiles.len(), 1);
        assert!(tiles
            .iter()
            .any(|id| id.x == 3 && id.y == 9 && id.mip_level == 1));
    }
}
