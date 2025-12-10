use half::f16;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use wgpu::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
};
use crate::terrain::tiling::TileId;
use super::config::MosaicConfig;
use super::util::{copy_rows_with_padding, padded_bytes_per_row};

#[derive(Debug)]
pub struct HeightMosaic {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub config: MosaicConfig,
    pub format: TextureFormat,
    // Mapping: TileId -> atlas slot index (sx, sy)
    slot_map: HashMap<TileId, (u32, u32)>,
    lru: VecDeque<TileId>,
}

impl HeightMosaic {
    pub fn new(device: &wgpu::Device, config: MosaicConfig, filter_linear: bool) -> Self {
        let (w, h) = config.texture_size();
        // E6: Choose format — prefer R32Float; if linear filtering requested but unsupported, fall back to RG16Float
        let has_f32_filter = device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let want_filter = filter_linear;
        let use_rg16f = want_filter && !has_f32_filter;
        let format = if use_rg16f {
            TextureFormat::Rg16Float
        } else {
            TextureFormat::R32Float
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(if use_rg16f {
                "terrain-height-mosaic-rg16f"
            } else {
                "terrain-height-mosaic-r32f"
            }),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain-height-mosaic-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: if want_filter {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            min_filter: if want_filter {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            texture,
            view,
            sampler,
            config,
            format,
            slot_map: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    fn evict_if_needed(&mut self) {
        let capacity = (self.config.tiles_x * self.config.tiles_y) as usize;
        while self.slot_map.len() > capacity {
            if let Some(old) = self.lru.pop_front() {
                self.slot_map.remove(&old);
            } else {
                break;
            }
        }
    }

    fn find_free_slot(&self) -> Option<(u32, u32)> {
        let capacity = (self.config.tiles_x * self.config.tiles_y) as usize;
        if self.slot_map.len() >= capacity {
            return None;
        }
        for sy in 0..self.config.tiles_y {
            for sx in 0..self.config.tiles_x {
                let occupied = self.slot_map.values().any(|&(x, y)| x == sx && y == sy);
                if !occupied {
                    return Some((sx, sy));
                }
            }
        }
        None
    }

    pub fn slot_of(&self, id: &TileId) -> Option<(u32, u32)> {
        self.slot_map.get(id).copied()
    }

    /// Snapshot current TileId -> (sx, sy) mappings for page-table sync
    pub fn entries(&self) -> Vec<(TileId, (u32, u32))> {
        self.slot_map.iter().map(|(k, v)| (*k, *v)).collect()
    }

    pub fn upload_tile(
        &mut self,
        queue: &Queue,
        id: TileId,
        height_data: &[f32],
    ) -> Result<(u32, u32), String> {
        let sz = (self.config.tile_size_px * self.config.tile_size_px) as usize;
        if height_data.len() != sz {
            return Err(format!(
                "height_data length mismatch: got {}, expected {}",
                height_data.len(),
                sz
            ));
        }
        // Determine slot
        let (sx, sy) = if let Some(lod) = self.config.fixed_lod {
            if id.lod != lod {
                return Err(format!(
                    "fixed_lod={} mismatch for tile id.lod={}",
                    lod, id.lod
                ));
            }
            if id.x >= self.config.tiles_x || id.y >= self.config.tiles_y {
                return Err("tile id out of mosaic bounds".into());
            }
            (id.x, id.y)
        } else {
            if let Some(slot) = self.slot_map.get(&id).copied() {
                slot
            } else if let Some(slot) = self.find_free_slot() {
                self.slot_map.insert(id, slot);
                self.lru.push_back(id);
                slot
            } else {
                // Evict the least recently used; be robust to inconsistent LRU entries
                if let Some(evicted) = self.lru.pop_front() {
                    let victim_slot = if let Some((ex, ey)) = self.slot_map.remove(&evicted) {
                        (ex, ey)
                    } else if let Some((any_id, &(ex, ey))) = self.slot_map.iter().next() {
                        // Fallback: evict an arbitrary entry
                        let any_id = *any_id;
                        let _ = self.slot_map.remove(&any_id);
                        (ex, ey)
                    } else {
                        return Err("No slots to evict".into());
                    };
                    self.slot_map.insert(id, victim_slot);
                    self.lru.push_back(id);
                    victim_slot
                } else if let Some((any_id, &(ex, ey))) = self.slot_map.iter().next() {
                    // LRU empty but slot_map full; evict arbitrary
                    let any_id = *any_id;
                    let _ = self.slot_map.remove(&any_id);
                    self.slot_map.insert(id, (ex, ey));
                    self.lru.push_back(id);
                    (ex, ey)
                } else {
                    return Err("No slots and empty LRU".into());
                }
            }
        };

        let offset_x = sx * self.config.tile_size_px;
        let offset_y = sy * self.config.tile_size_px;
        let rows_per_image = self.config.tile_size_px;
        let unpadded_bpr = if self.format == TextureFormat::Rg16Float {
            4 * self.config.tile_size_px // 2 channels * 2 bytes
        } else {
            4 * self.config.tile_size_px // 4 bytes per f32
        };
        let padded_bpr = padded_bytes_per_row(unpadded_bpr);

        let bytes_ref: Cow<[u8]> = if self.format == TextureFormat::Rg16Float {
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct Rg16 {
                r: f16,
                g: f16,
            }
            unsafe impl bytemuck::Zeroable for Rg16 {}
            unsafe impl bytemuck::Pod for Rg16 {}

            let mut tmp: Vec<Rg16> = Vec::with_capacity(sz);
            for &h in height_data.iter() {
                tmp.push(Rg16 {
                    r: f16::from_f32(h),
                    g: f16::from_f32(0.0),
                });
            }
            let mut vec_u8 = bytemuck::cast_slice(&tmp).to_vec();
            if padded_bpr != unpadded_bpr {
                vec_u8 = copy_rows_with_padding(
                    &vec_u8,
                    unpadded_bpr as usize,
                    padded_bpr as usize,
                    rows_per_image as usize,
                );
            }
            Cow::Owned(vec_u8)
        } else {
            let src_bytes = bytemuck::cast_slice(height_data);
            if padded_bpr != unpadded_bpr {
                Cow::Owned(copy_rows_with_padding(
                    src_bytes,
                    unpadded_bpr as usize,
                    padded_bpr as usize,
                    rows_per_image as usize,
                ))
            } else {
                Cow::Borrowed(src_bytes)
            }
        };

        queue.write_texture(
            ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: offset_x,
                    y: offset_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes_ref.as_ref(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr.try_into().unwrap()),
                rows_per_image: Some(rows_per_image.try_into().unwrap()),
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
        Ok((sx, sy))
    }

    pub fn mark_used(&mut self, id: TileId) {
        // Update LRU order
        self.lru.retain(|&t| t != id);
        self.lru.push_back(id);
    }
}
