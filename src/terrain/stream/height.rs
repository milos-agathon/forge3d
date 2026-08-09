use super::config::MosaicConfig;
use super::residency::{EvictionPolicy, HeightPageResidency};
use super::util::{copy_rows_with_padding, padded_bytes_per_row};
use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::terrain::tiling::TileId;
use half::f16;
use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{
    CommandEncoder, Extent3d, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, Origin3d, Queue,
    Sampler, SamplerDescriptor, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView,
};

pub struct PreparedHeightUpload {
    pub id: TileId,
    pub sx: u32,
    pub sy: u32,
    pub bytes: Vec<u8>,
    pub coverage_bytes: Vec<u8>,
    pub padded_bytes_per_row: u32,
    pub coverage_padded_bytes_per_row: u32,
    pub rows: u32,
}

#[derive(Debug)]
pub struct HeightMosaic {
    pub texture: TrackedTexture,
    pub view: TextureView,
    pub coverage_texture: TrackedTexture,
    pub coverage_view: TextureView,
    pub sampler: Sampler,
    pub config: MosaicConfig,
    pub format: TextureFormat,
    residency: HeightPageResidency,
    f3dz_sources: HashMap<TileId, F3dzUsage>,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
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
        let coverage_texture = tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("terrain-height-coverage-mosaic-r8"),
                size: Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let coverage_view = coverage_texture.create_view(&wgpu::TextureViewDescriptor::default());
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
            coverage_texture,
            coverage_view,
            sampler,
            config,
            format,
            residency: HeightPageResidency::new(config.tiles_x, config.tiles_y, config.fixed_lod)
                .map_err(crate::core::error::RenderError::upload)?,
            f3dz_sources: HashMap::new(),
        })
    }

    pub fn slot_of(&self, id: &TileId) -> Option<(u32, u32)> {
        self.residency.slot_of(*id)
    }

    /// Snapshot current TileId -> (sx, sy) mappings for page-table sync
    pub fn entries(&self) -> Vec<(TileId, (u32, u32))> {
        self.residency.entries()
    }

    /// Resolve a selected virtual tile to the nearest resident ancestor. The
    /// caller can therefore keep drawing coarse coverage while a leaf request
    /// remains in flight instead of binding an empty mapping.
    pub fn nearest_resident_ancestor(&self, requested: TileId) -> Option<(TileId, (u32, u32))> {
        self.residency.resolve(requested)
    }

    /// Touch the physical page actually used for a virtual request. This is
    /// the runtime LRU seam: exact hits touch themselves and misses touch the
    /// ancestor the shader will sample.
    pub fn touch_nearest_resident_ancestor(&mut self, requested: TileId) -> Option<TileId> {
        self.residency.touch_resolved(requested)
    }

    pub fn resident_tile_count(&self) -> usize {
        self.residency.len()
    }

    pub fn remove_mapping(&mut self, id: TileId) {
        self.residency.remove(id);
        self.f3dz_sources.remove(&id);
    }

    /// Evict exactly the least-recently-used resident tile. Callers that have
    /// a wider GPU budget (page table, staging, atlas) use this before upload
    /// so a new leaf never causes an unbounded allocation.
    pub fn evict_least_recently_used(&mut self) -> Option<TileId> {
        let (evicted, _) = self.residency.evict_lru(EvictionPolicy::Any)?;
        self.f3dz_sources.remove(&evicted);
        Some(evicted)
    }

    /// Evict an LRU refinement while preserving the root ancestor that gives
    /// every not-yet-resident virtual tile coarse coverage.
    pub fn evict_least_recently_used_leaf(&mut self) -> Option<TileId> {
        let (evicted, _) = self.residency.evict_lru(EvictionPolicy::PreserveRoot)?;
        self.f3dz_sources.remove(&evicted);
        Some(evicted)
    }

    /// Physical, tracked atlas bytes. This is deliberately separate from
    /// logical resident-tile bytes: eviction changes the latter, not this
    /// allocation's high-water mark.
    pub fn gpu_visible_bytes(&self) -> u64 {
        let (width, height) = self.config.texture_size();
        u64::from(width) * u64::from(height) * 5
    }

    pub fn upload_tile(
        &mut self,
        queue: &Queue,
        id: TileId,
        height_data: &[f32],
    ) -> Result<(u32, u32), String> {
        let prepared = self.prepare_tile_upload(id, height_data)?;
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: prepared.sx * self.config.tile_size_px,
                    y: prepared.sy * self.config.tile_size_px,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &prepared.bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(prepared.padded_bytes_per_row),
                rows_per_image: Some(prepared.rows),
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.coverage_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: prepared.sx * self.config.tile_size_px,
                    y: prepared.sy * self.config.tile_size_px,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &prepared.coverage_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(prepared.coverage_padded_bytes_per_row),
                rows_per_image: Some(prepared.rows),
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
        Ok((prepared.sx, prepared.sy))
    }

    pub fn prepare_tile_upload(
        &mut self,
        id: TileId,
        height_data: &[f32],
    ) -> Result<PreparedHeightUpload, String> {
        let coverage = vec![u8::MAX; height_data.len()];
        self.prepare_covered_tile_upload(id, height_data, &coverage)
    }

    pub fn prepare_covered_tile_upload(
        &mut self,
        id: TileId,
        height_data: &[f32],
        coverage_data: &[u8],
    ) -> Result<PreparedHeightUpload, String> {
        let sz = (self.config.tile_size_px * self.config.tile_size_px) as usize;
        if height_data.len() != sz {
            return Err(format!(
                "height_data length mismatch: got {}, expected {}",
                height_data.len(),
                sz
            ));
        }
        if coverage_data.len() != sz {
            return Err(format!(
                "coverage_data length mismatch: got {}, expected {}",
                coverage_data.len(),
                sz
            ));
        }
        let (sx, sy) = self.allocate_slot(id)?;

        let rows_per_image = self.config.tile_size_px;
        let unpadded_bpr = if self.format == TextureFormat::Rg16Float {
            4 * self.config.tile_size_px // 2 channels * 2 bytes
        } else {
            4 * self.config.tile_size_px // 4 bytes per f32
        };
        let padded_bpr = padded_bytes_per_row(unpadded_bpr);
        let coverage_padded_bpr = padded_bytes_per_row(self.config.tile_size_px);

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

        self.f3dz_sources.remove(&id);
        let coverage_bytes = copy_rows_with_padding(
            coverage_data,
            self.config.tile_size_px as usize,
            coverage_padded_bpr as usize,
            rows_per_image as usize,
        );
        Ok(PreparedHeightUpload {
            id,
            sx,
            sy,
            bytes: bytes_ref.into_owned(),
            coverage_bytes,
            padded_bytes_per_row: padded_bpr,
            coverage_padded_bytes_per_row: coverage_padded_bpr,
            rows: rows_per_image,
        })
    }

    pub fn encode_prepared_upload(
        &self,
        encoder: &mut CommandEncoder,
        staging: &wgpu::Buffer,
        staging_offset: u64,
        coverage_staging_offset: u64,
        prepared: &PreparedHeightUpload,
    ) {
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: staging,
                layout: ImageDataLayout {
                    offset: staging_offset,
                    bytes_per_row: Some(prepared.padded_bytes_per_row),
                    rows_per_image: Some(prepared.rows),
                },
            },
            ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: prepared.sx * self.config.tile_size_px,
                    y: prepared.sy * self.config.tile_size_px,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: staging,
                layout: ImageDataLayout {
                    offset: coverage_staging_offset,
                    bytes_per_row: Some(prepared.coverage_padded_bytes_per_row),
                    rows_per_image: Some(prepared.rows),
                },
            },
            ImageCopyTexture {
                texture: &self.coverage_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: prepared.sx * self.config.tile_size_px,
                    y: prepared.sy * self.config.tile_size_px,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Decode an F3DZ stream directly into this height atlas. A failure removes
    /// the new dynamic mapping, so callers can never observe stale tile bytes
    /// under the requested id.
    #[cfg(not(target_arch = "wasm32"))]
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
                self.residency.remove(id);
            }
            self.f3dz_sources.remove(&id);
            return Err(error.to_string());
        }
        let coverage_row = padded_bytes_per_row(self.config.tile_size_px);
        let coverage = copy_rows_with_padding(
            &vec![u8::MAX; (self.config.tile_size_px * self.config.tile_size_px) as usize],
            self.config.tile_size_px as usize,
            coverage_row as usize,
            self.config.tile_size_px as usize,
        );
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.coverage_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.0,
                    y: origin.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &coverage,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(coverage_row),
                rows_per_image: Some(self.config.tile_size_px),
            },
            Extent3d {
                width: self.config.tile_size_px,
                height: self.config.tile_size_px,
                depth_or_array_layers: 1,
            },
        );
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
        let placement = self.residency.place(id, EvictionPolicy::Any)?;
        if let Some(evicted) = placement.evicted {
            self.f3dz_sources.remove(&evicted);
        }
        Ok(placement.slot)
    }

    /// Seed the active render capture with evidence for every resident F3DZ
    /// tile. This keeps certificates honest even when streaming decode
    /// completed before the render capture began.
    #[cfg(not(target_arch = "wasm32"))]
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
        let _ = self.residency.touch_resolved(id);
    }
}
