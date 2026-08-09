use bytemuck::{Pod, Zeroable};
use wgpu::{BufferDescriptor, BufferUsages, Queue};

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use crate::terrain::stream::HeightMosaic;
use crate::terrain::tiling::TileId;

pub const EMPTY_PAGE_KEY: u32 = u32::MAX;

/// The first 32 bytes of the GPU page-table buffer. The buckets immediately
/// follow this header, which lets one storage binding carry the entire lookup
/// contract used by every terrain render path.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PageTableHeader {
    pub enabled: u32,
    pub root_ready: u32,
    pub table_mask: u32,
    pub max_probe_count: u32,
    pub target_lod: u32,
    pub tile_resolution: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct PageTableEntry {
    pub lod: u32,
    pub x: u32,
    pub y: u32,
    pub _pad0: u32,
    pub sx: u32,
    pub sy: u32,
    pub slot: u32,
    pub _pad1: u32,
}

impl PageTableEntry {
    fn empty() -> Self {
        Self {
            lod: EMPTY_PAGE_KEY,
            x: 0,
            y: 0,
            _pad0: 0,
            sx: 0,
            sy: 0,
            slot: 0,
            _pad1: 0,
        }
    }

    pub fn tile_id(&self) -> Option<TileId> {
        (self.lod != EMPTY_PAGE_KEY).then(|| TileId::new(self.lod, self.x, self.y))
    }
}

fn hash_tile(id: TileId) -> u32 {
    // Keep this bit-for-bit aligned with height_page_hash in WGSL.
    let mut h = id.lod.wrapping_mul(0x9e37_79b9);
    h ^= id.x.wrapping_mul(0x85eb_ca6b);
    h = h.rotate_left(13);
    h ^= id.y.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn capacity_for_entries(max_entries: usize) -> usize {
    max_entries.max(1).saturating_mul(2).next_power_of_two()
}

#[derive(Clone, Debug)]
pub struct SerializedPageTable {
    pub header: PageTableHeader,
    pub buckets: Vec<PageTableEntry>,
}

impl SerializedPageTable {
    /// Build the exact bounded sparse table consumed by the renderer from an
    /// already-finalized id-to-slot mapping. Keeping construction independent
    /// of `wgpu::Device` lets browser integration exercise the production
    /// hash/ancestor contract before a WebGPU upload.
    pub fn from_entries(
        entries: &[(TileId, (u32, u32))],
        capacity: usize,
        target_lod: u32,
        tile_resolution: u32,
        atlas_dimensions: (u32, u32),
        tiles_x: u32,
    ) -> Result<Self, String> {
        if entries.len() > capacity {
            return Err(format!(
                "{} resident height pages exceed page-table capacity {}",
                entries.len(),
                capacity
            ));
        }
        let bucket_count = capacity_for_entries(capacity.max(1));
        let root_ready = entries.iter().any(|(id, _)| *id == TileId::new(0, 0, 0));
        let has_entries = !entries.is_empty();
        let mut buckets = vec![PageTableEntry::empty(); bucket_count];
        let mask = (bucket_count - 1) as u32;
        let mut longest_probe = 1u32;
        for &(id, (sx, sy)) in entries {
            let mut bucket = hash_tile(id) & mask;
            let mut inserted = false;
            for probe in 1..=bucket_count as u32 {
                if buckets[bucket as usize].lod == EMPTY_PAGE_KEY {
                    buckets[bucket as usize] = PageTable::entry_for(id, sx, sy, tiles_x);
                    longest_probe = longest_probe.max(probe);
                    inserted = true;
                    break;
                }
                bucket = (bucket + 1) & mask;
            }
            if !inserted {
                return Err("bounded sparse height page table is full".to_string());
            }
        }
        Ok(Self {
            header: PageTableHeader {
                enabled: u32::from(has_entries),
                root_ready: u32::from(root_ready),
                table_mask: mask,
                max_probe_count: longest_probe,
                target_lod,
                tile_resolution,
                atlas_width: atlas_dimensions.0,
                atlas_height: atlas_dimensions.1,
            },
            buckets,
        })
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            std::mem::size_of::<PageTableHeader>()
                + self.buckets.len() * std::mem::size_of::<PageTableEntry>(),
        );
        bytes.extend_from_slice(bytemuck::bytes_of(&self.header));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.buckets));
        bytes
    }

    pub fn lookup_exact(&self, id: TileId) -> Option<PageTableEntry> {
        if self.header.enabled == 0 || self.buckets.is_empty() {
            return None;
        }
        let mut bucket = hash_tile(id) & self.header.table_mask;
        for _ in 0..self.header.max_probe_count {
            let entry = self.buckets[bucket as usize];
            if entry.lod == EMPTY_PAGE_KEY {
                return None;
            }
            if entry.lod == id.lod && entry.x == id.x && entry.y == id.y {
                return Some(entry);
            }
            bucket = (bucket + 1) & self.header.table_mask;
        }
        None
    }

    pub fn resolve_nearest_resident_ancestor(&self, requested: TileId) -> Option<PageTableEntry> {
        let mut candidate = Some(requested);
        while let Some(id) = candidate {
            if let Some(entry) = self.lookup_exact(id) {
                return Some(entry);
            }
            candidate = id.parent();
        }
        None
    }
}

pub struct PageTable {
    pub buffer: TrackedBuffer,
    /// Maximum resident mappings; the hash bucket count is at least twice this
    /// value so lookups never exceed 50% load.
    pub capacity: usize,
    bucket_count: usize,
    target_lod: u32,
    tile_resolution: u32,
    atlas_dimensions: (u32, u32),
}

impl PageTable {
    pub fn allocation_bytes_for_capacity(capacity: usize) -> u64 {
        (std::mem::size_of::<PageTableHeader>()
            + capacity_for_entries(capacity.max(1)) * std::mem::size_of::<PageTableEntry>())
            as u64
    }

    pub fn new(device: &wgpu::Device, capacity: usize) -> RenderResult<Self> {
        Self::new_sparse(device, capacity, 0, 1, (1, 1))
    }

    pub fn new_sparse(
        device: &wgpu::Device,
        capacity: usize,
        target_lod: u32,
        tile_resolution: u32,
        atlas_dimensions: (u32, u32),
    ) -> RenderResult<Self> {
        let capacity = capacity.max(1);
        let bucket_count = capacity_for_entries(capacity);
        let size = Self::allocation_bytes_for_capacity(capacity);
        let buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("orbis.height.page-table"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        Ok(Self {
            buffer,
            capacity,
            bucket_count,
            target_lod,
            tile_resolution,
            atlas_dimensions,
        })
    }

    pub fn gpu_visible_bytes(&self) -> u64 {
        (std::mem::size_of::<PageTableHeader>()
            + self.bucket_count * std::mem::size_of::<PageTableEntry>()) as u64
    }

    fn entry_for(id: TileId, sx: u32, sy: u32, tiles_x: u32) -> PageTableEntry {
        PageTableEntry {
            lod: id.lod,
            x: id.x,
            y: id.y,
            _pad0: 0,
            sx,
            sy,
            slot: sy * tiles_x + sx,
            _pad1: 0,
        }
    }

    pub fn serialize(&self, mosaic: &HeightMosaic) -> Result<SerializedPageTable, String> {
        let entries = mosaic.entries();
        SerializedPageTable::from_entries(
            &entries,
            self.capacity,
            self.target_lod,
            self.tile_resolution,
            self.atlas_dimensions,
            mosaic.config.tiles_x,
        )
    }

    pub fn resolve_nearest_resident_ancestor(
        &self,
        mosaic: &HeightMosaic,
        requested: TileId,
    ) -> Option<PageTableEntry> {
        self.serialize(mosaic)
            .ok()?
            .resolve_nearest_resident_ancestor(requested)
    }

    /// Legacy upload helper for non-ORBIS callers. The ORBIS renderer uploads
    /// the same serialized bytes through its tracked, bounded staging path.
    pub fn sync_from_mosaic(&mut self, queue: &Queue, mosaic: &HeightMosaic) {
        if let Ok(table) = self.serialize(mosaic) {
            queue.write_buffer(&self.buffer, 0, &table.bytes());
        }
    }
}

/// Disabled fallback bound whenever dynamic height streaming is absent or its
/// root is not ready. Its header makes the shader use the caller's overview.
pub fn create_disabled_page_table(device: &wgpu::Device) -> RenderResult<TrackedBuffer> {
    let size =
        (std::mem::size_of::<PageTableHeader>() + std::mem::size_of::<PageTableEntry>()) as u64;
    let buffer = tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some("terrain.height-page-table.disabled"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        },
    )?;
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.fill(0);
        mapped[std::mem::size_of::<PageTableHeader>()
            ..std::mem::size_of::<PageTableHeader>() + std::mem::size_of::<u32>()]
            .copy_from_slice(&EMPTY_PAGE_KEY.to_le_bytes());
    }
    buffer.unmap();
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_for(entries: &[(TileId, (u32, u32))], bucket_count: usize) -> SerializedPageTable {
        let mut buckets = vec![PageTableEntry::empty(); bucket_count];
        let mask = (bucket_count - 1) as u32;
        let mut longest = 1;
        for &(id, (sx, sy)) in entries {
            let mut bucket = hash_tile(id) & mask;
            for probe in 1..=bucket_count as u32 {
                if buckets[bucket as usize].lod == EMPTY_PAGE_KEY {
                    buckets[bucket as usize] = PageTable::entry_for(id, sx, sy, 4);
                    longest = longest.max(probe);
                    break;
                }
                bucket = (bucket + 1) & mask;
            }
        }
        SerializedPageTable {
            header: PageTableHeader {
                enabled: 1,
                root_ready: 1,
                table_mask: mask,
                max_probe_count: longest,
                target_lod: 3,
                tile_resolution: 16,
                atlas_width: 64,
                atlas_height: 64,
            },
            buckets,
        }
    }

    #[test]
    fn sparse_hash_handles_collisions_and_exact_hits_with_bounded_probes() {
        let mask = 7;
        let mut colliders = Vec::new();
        for y in 0..64 {
            let id = TileId::new(6, 3, y);
            if hash_tile(id) & mask == 2 {
                colliders.push(id);
                if colliders.len() == 3 {
                    break;
                }
            }
        }
        assert_eq!(colliders.len(), 3);
        let entries: Vec<_> = colliders
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, (index as u32, 0)))
            .collect();
        let table = table_for(&entries, 8);
        assert_eq!(table.header.max_probe_count, 3);
        for (index, id) in colliders.iter().enumerate() {
            assert_eq!(table.lookup_exact(*id).unwrap().sx, index as u32);
        }
        assert!(table.lookup_exact(TileId::new(6, 63, 63)).is_none());
    }

    #[test]
    fn parent_walk_reaches_root_and_stale_snapshot_entries_disappear() {
        let root = TileId::new(0, 0, 0);
        let parent = TileId::new(2, 1, 2);
        let leaf = TileId::new(4, 6, 9);
        let first = table_for(&[(root, (0, 0)), (parent, (1, 0))], 8);
        assert_eq!(
            first
                .resolve_nearest_resident_ancestor(leaf)
                .unwrap()
                .tile_id(),
            Some(parent)
        );
        let after_eviction = table_for(&[(root, (0, 0))], 8);
        assert!(after_eviction.lookup_exact(parent).is_none());
        assert_eq!(
            after_eviction
                .resolve_nearest_resident_ancestor(leaf)
                .unwrap()
                .tile_id(),
            Some(root)
        );
    }

    #[test]
    fn hash_capacity_is_power_of_two_and_never_above_half_load() {
        for entries in 1..100 {
            let capacity = capacity_for_entries(entries);
            assert!(capacity.is_power_of_two());
            assert!(entries * 2 <= capacity);
        }
    }
}
