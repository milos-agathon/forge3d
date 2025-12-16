use super::config::MosaicConfig;
use super::util::{copy_rows_with_padding, padded_bytes_per_row};
use crate::terrain::tiling::TileId;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use wgpu::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
};

// E3: Color mosaic for RGBA8 overlays/basemaps
#[derive(Debug)]
pub struct ColorMosaic {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub config: MosaicConfig,
    slot_map: HashMap<TileId, (u32, u32)>,
    lru: VecDeque<TileId>,
}

impl ColorMosaic {
    pub fn new(
        device: &wgpu::Device,
        config: MosaicConfig,
        srgb: bool,
        filter_linear: bool,
    ) -> Self {
        let (w, h) = config.texture_size();
        let format = if srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("terrain-color-mosaic-rgba8"),
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
            label: Some("terrain-color-mosaic-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: if filter_linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            min_filter: if filter_linear {
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
            slot_map: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn upload_tile(
        &mut self,
        queue: &Queue,
        id: TileId,
        rgba_data: &[u8],
    ) -> Result<(u32, u32), String> {
        let px = self.config.tile_size_px;
        let expected = (px * px * 4) as usize;
        if rgba_data.len() != expected {
            return Err(format!(
                "rgba_data length mismatch: got {}, expected {}",
                rgba_data.len(),
                expected
            ));
        }
        // Similar slot management as HeightMosaic
        let (sx, sy) = if let Some(slot) = self.slot_map.get(&id).copied() {
            slot
        } else {
            // find first free slot
            let cap = (self.config.tiles_x * self.config.tiles_y) as usize;
            if self.slot_map.len() >= cap {
                if let Some(evicted) = self.lru.pop_front() {
                    let victim_slot = if let Some((ex, ey)) = self.slot_map.remove(&evicted) {
                        (ex, ey)
                    } else if let Some((any_id, &(ex, ey))) = self.slot_map.iter().next() {
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
                    let any_id = *any_id;
                    let _ = self.slot_map.remove(&any_id);
                    self.slot_map.insert(id, (ex, ey));
                    self.lru.push_back(id);
                    (ex, ey)
                } else {
                    return Err("No slots and empty LRU".into());
                }
            } else {
                let mut chosen: Option<(u32, u32)> = None;
                'outer: for y in 0..self.config.tiles_y {
                    for x in 0..self.config.tiles_x {
                        let occ = self.slot_map.values().any(|&(ax, ay)| ax == x && ay == y);
                        if !occ {
                            chosen = Some((x, y));
                            break 'outer;
                        }
                    }
                }
                let (x, y) = chosen.ok_or_else(|| "No free slot found".to_string())?;
                self.slot_map.insert(id, (x, y));
                self.lru.push_back(id);
                (x, y)
            }
        };
        let offset_x = sx * self.config.tile_size_px;
        let offset_y = sy * self.config.tile_size_px;
        let row_bytes = 4 * self.config.tile_size_px; // RGBA8 bytes per row
        let padded_bpr = padded_bytes_per_row(row_bytes);
        let bytes_ref: Cow<[u8]> = if padded_bpr != row_bytes {
            Cow::Owned(copy_rows_with_padding(
                rgba_data,
                row_bytes as usize,
                padded_bpr as usize,
                self.config.tile_size_px as usize,
            ))
        } else {
            Cow::Borrowed(rgba_data)
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
                rows_per_image: Some(self.config.tile_size_px.try_into().unwrap()),
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
        Ok((sx, sy))
    }

    pub fn slot_of(&self, id: &TileId) -> Option<(u32, u32)> {
        self.slot_map.get(id).copied()
    }

    pub fn mark_used(&mut self, id: TileId) {
        // Simple LRU update similar to HeightMosaic
        self.lru.retain(|&t| t != id);
        self.lru.push_back(id);
    }
}
