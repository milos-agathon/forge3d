use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Queue};

use crate::terrain::stream::HeightMosaic;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PageTableEntry {
    // tile id
    pub lod: u32,
    pub x: u32,
    pub y: u32,
    pub _pad0: u32,
    // atlas slot coordinates and linearized index
    pub sx: u32,
    pub sy: u32,
    pub slot: u32,
    pub _pad1: u32,
}

pub struct PageTable {
    pub buffer: Buffer,
    pub capacity: usize,
}

impl PageTable {
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let entry_size = std::mem::size_of::<PageTableEntry>() as u64;
        let size = (capacity as u64).saturating_mul(entry_size).max(entry_size);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("terrain-page-table"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, capacity }
    }

    pub fn sync_from_mosaic(&mut self, queue: &Queue, mosaic: &HeightMosaic) {
        // Build compact list (truncate if needed to fit capacity)
        let mut out: Vec<PageTableEntry> = Vec::with_capacity(self.capacity);
        let tiles_x = mosaic.config.tiles_x;
        for (id, (sx, sy)) in mosaic.entries().into_iter() {
            if out.len() >= self.capacity {
                break;
            }
            let slot = sy * tiles_x + sx;
            out.push(PageTableEntry {
                lod: id.lod,
                x: id.x,
                y: id.y,
                _pad0: 0,
                sx,
                sy,
                slot,
                _pad1: 0,
            });
        }
        if out.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice(&out);
        queue.write_buffer(&self.buffer, 0, bytes);
    }
}
