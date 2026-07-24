use super::config::MosaicConfig;
use super::util::{copy_rows_with_padding, padded_bytes_per_row};
use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::terrain::tiling::TileId;
use half::f16;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use wgpu::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
};

#[derive(Debug)]
pub struct HeightMosaic {
    pub texture: TrackedTexture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub config: MosaicConfig,
    pub format: TextureFormat,
    // Mapping: TileId -> atlas slot index (sx, sy)
    slot_map: HashMap<TileId, (u32, u32)>,
    lru: VecDeque<TileId>,
    f3dz_sources: HashMap<TileId, F3dzUsage>,
}

#[derive(Clone, Copy, Debug)]
struct F3dzUsage {
    epsilon: f32,
    page_count: u32,
    base_quality: bool,
}

impl HeightMosaic {
    pub fn new(
        device: &wgpu::Device,
        config: MosaicConfig,
        filter_linear: bool,
    ) -> RenderResult<Self> {
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
        let texture = tracked_create_texture(
            device,
            &TextureDescriptor {
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
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | if use_rg16f {
                        TextureUsages::empty()
                    } else {
                        // F3DZ compute decode writes R32Float pages straight
                        // into this atlas, without a CPU height array.
                        TextureUsages::STORAGE_BINDING
                    },
                view_formats: &[],
            },
        )?;
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
        Ok(Self {
            texture,
            view,
            sampler,
            config,
            format,
            slot_map: HashMap::new(),
            lru: VecDeque::new(),
            f3dz_sources: HashMap::new(),
        })
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
        let (sx, sy) = self.allocate_slot(id)?;

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
        self.f3dz_sources.remove(&id);
        Ok((sx, sy))
    }

    /// Decode an F3DZ stream directly into this height atlas. A failure removes
    /// the new dynamic mapping, so callers can never observe stale tile bytes
    /// under the requested id.
    pub fn upload_f3dz(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        id: TileId,
        compressed: &[u8],
    ) -> Result<(u32, u32), String> {
        if self.format != TextureFormat::R32Float {
            return Err("f3dz direct atlas decode requires an R32Float height mosaic".to_string());
        }
        let header = crate::codec::f3dz::gpu::validate_stream(compressed)
            .map_err(|error| error.to_string())?;
        if header.width != self.config.tile_size_px || header.height != self.config.tile_size_px {
            return Err(format!(
                "f3dz tile dimensions {}x{} do not match mosaic tile_size_px={}",
                header.width, header.height, self.config.tile_size_px
            ));
        }
        let decoder = crate::codec::f3dz::gpu::F3dzGpuDecoder::new(device)
            .map_err(|error| error.to_string())?;
        let (sx, sy) = self.allocate_slot(id)?;
        let origin = (sx * self.config.tile_size_px, sy * self.config.tile_size_px);
        if let Err(error) =
            decoder.decode_into_atlas(device, queue, compressed, &self.texture, origin)
        {
            if self.config.fixed_lod.is_none() {
                self.slot_map.remove(&id);
                self.lru.retain(|candidate| *candidate != id);
            }
            self.f3dz_sources.remove(&id);
            return Err(error.to_string());
        }
        self.f3dz_sources.insert(
            id,
            F3dzUsage {
                epsilon: header.epsilon,
                page_count: header.page_count,
                base_quality: header.base_only(),
            },
        );
        Ok((sx, sy))
    }

    fn allocate_slot(&mut self, id: TileId) -> Result<(u32, u32), String> {
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
                        self.f3dz_sources.remove(&evicted);
                        (ex, ey)
                    } else if let Some((any_id, &(ex, ey))) = self.slot_map.iter().next() {
                        // Fallback: evict an arbitrary entry
                        let any_id = *any_id;
                        let _ = self.slot_map.remove(&any_id);
                        self.f3dz_sources.remove(&any_id);
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
                    self.f3dz_sources.remove(&any_id);
                    self.slot_map.insert(id, (ex, ey));
                    self.lru.push_back(id);
                    (ex, ey)
                } else {
                    return Err("No slots and empty LRU".into());
                }
            }
        };
        Ok((sx, sy))
    }

    /// Seed the active render capture with evidence for every resident F3DZ
    /// tile. This keeps certificates honest even when streaming decode
    /// completed before the render capture began.
    pub fn record_certificate_usage(&self) {
        for usage in self.f3dz_sources.values() {
            crate::core::certificate::record_f3dz_pages(
                usage.epsilon,
                usage.page_count,
                usage.base_quality,
            );
        }
    }

    pub fn mark_used(&mut self, id: TileId) {
        // Update LRU order
        self.lru.retain(|&t| t != id);
        self.lru.push_back(id);
    }
}
