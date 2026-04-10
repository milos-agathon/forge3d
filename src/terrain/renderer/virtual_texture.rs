use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[cfg(feature = "extension-module")]
use super::*;

#[cfg(feature = "extension-module")]
use crate::core::feedback_buffer::FeedbackBuffer;
#[cfg(feature = "extension-module")]
use crate::core::tile_cache::{TileCache, TileData, TileId};
#[cfg(feature = "extension-module")]
use crate::core::virtual_texture::PageTableEntry;

#[cfg(feature = "extension-module")]
// Terrain VT v1 pages only the albedo family. The Python contract already
// reserves normal/mask entries so the public API does not need to change later.
const TERRAIN_VT_SUPPORTED_FAMILY: &str = "albedo";
#[cfg(feature = "extension-module")]
const TERRAIN_VT_BYTES_PER_PIXEL: usize = 4;

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub(super) struct VTSource {
    pub virtual_size: (u32, u32),
    pub data: Vec<u8>,
    pub fallback_color: [f32; 4],
}

#[cfg(feature = "extension-module")]
pub(super) struct TerrainVTBindingResources<'a> {
    pub atlas_view: &'a wgpu::TextureView,
    pub page_table_view: &'a wgpu::TextureView,
    pub feedback_buffer: Option<&'a wgpu::Buffer>,
}

#[cfg(feature = "extension-module")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainVTUniformsGpu {
    config0: [u32; 4],
    config1: [u32; 4],
    config2: [u32; 4],
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileKey {
    material_index: u32,
    x: u32,
    y: u32,
    mip_level: u32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
struct MipImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
struct PreparedVTSource {
    fallback_color: [f32; 4],
    mips: Vec<MipImage>,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, Default)]
struct TerrainMaterialVTStats {
    resident_pages: u32,
    total_pages: u32,
    cache_budget_pages: u32,
    cache_budget_mb: f32,
    cache_hits: u32,
    cache_misses: u32,
    tiles_streamed: u32,
    evictions: u32,
    avg_upload_ms: f32,
    last_upload_ms: f32,
    resident_megabytes: f32,
    source_count: u32,
    feedback_requests: u32,
}

#[cfg(feature = "extension-module")]
struct TerrainMaterialVTRuntime {
    virtual_size: (u32, u32),
    tile_size: u32,
    tile_border: u32,
    slot_size: u32,
    atlas_size: u32,
    material_count: u32,
    max_mip_levels: u32,
    pages_x0: u32,
    pages_y0: u32,
    atlas_texture: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    page_table_texture: wgpu::Texture,
    page_table_view: wgpu::TextureView,
    page_tables: Vec<Vec<PageTableEntry>>,
    sources: HashMap<u32, PreparedVTSource>,
    tile_cache: TileCache,
    feedback_buffer: Option<FeedbackBuffer>,
    pending_feedback: Vec<TileKey>,
    feedback_staged: bool,
    budget_pages: u32,
    source_generation: u64,
    use_feedback: bool,
    stats: TerrainMaterialVTStats,
}

#[cfg(feature = "extension-module")]
pub(super) struct TerrainMaterialVT {
    pub sources: HashMap<(u32, String), VTSource>,
    runtime: Option<TerrainMaterialVTRuntime>,
    source_generation: u64,
    last_stats: TerrainMaterialVTStats,
}

#[cfg(feature = "extension-module")]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            runtime: None,
            source_generation: 0,
            last_stats: TerrainMaterialVTStats::default(),
        }
    }

    pub fn register_source(
        &mut self,
        material_index: u32,
        family: String,
        virtual_size_px: (u32, u32),
        data: Vec<u8>,
        fallback_color: [f32; 4],
    ) -> Result<(), String> {
        if virtual_size_px.0 == 0 || virtual_size_px.1 == 0 {
            return Err("virtual_size_px must be > 0 in both dimensions".to_string());
        }
        if family != TERRAIN_VT_SUPPORTED_FAMILY {
            log::warn!(
                "terrain material VT currently pages only '{supported}' sources; storing '{family}' for forward compatibility but the native runtime will ignore it",
                supported = TERRAIN_VT_SUPPORTED_FAMILY,
                family = family,
            );
        }
        if family == TERRAIN_VT_SUPPORTED_FAMILY {
            let expected_len = virtual_size_px.0 as usize
                * virtual_size_px.1 as usize
                * TERRAIN_VT_BYTES_PER_PIXEL;
            if data.len() != expected_len {
                return Err(format!(
                    "VT source data size mismatch for {family}: expected {expected_len} RGBA8 bytes, got {}",
                    data.len()
                ));
            }
        } else if data.is_empty() {
            return Err("VT source data must not be empty".to_string());
        }

        if let Some(existing) = self.sources.get(&(material_index, family.clone())) {
            if existing.virtual_size != virtual_size_px {
                return Err(format!(
                    "Virtual size mismatch: existing {:?}, new {:?}",
                    existing.virtual_size, virtual_size_px
                ));
            }
        }

        self.sources.insert(
            (material_index, family),
            VTSource {
                virtual_size: virtual_size_px,
                data,
                fallback_color,
            },
        );
        self.source_generation = self.source_generation.wrapping_add(1);
        self.runtime = None;
        Ok(())
    }

    pub fn clear_sources(&mut self) {
        self.sources.clear();
        self.runtime = None;
        self.source_generation = self.source_generation.wrapping_add(1);
        self.last_stats = TerrainMaterialVTStats::default();
    }

    pub fn get_stats(&self) -> HashMap<String, f32> {
        let stats = if let Some(runtime) = self.runtime.as_ref() {
            runtime.stats
        } else {
            self.last_stats
        };
        let mut out = HashMap::new();
        out.insert("resident_pages".to_string(), stats.resident_pages as f32);
        out.insert("total_pages".to_string(), stats.total_pages as f32);
        out.insert(
            "cache_budget_pages".to_string(),
            stats.cache_budget_pages as f32,
        );
        out.insert("cache_budget_mb".to_string(), stats.cache_budget_mb);
        out.insert("cache_hits".to_string(), stats.cache_hits as f32);
        out.insert("cache_misses".to_string(), stats.cache_misses as f32);
        out.insert("miss_rate".to_string(), Self::miss_rate(stats));
        out.insert("tiles_streamed".to_string(), stats.tiles_streamed as f32);
        out.insert("evictions".to_string(), stats.evictions as f32);
        out.insert("avg_upload_ms".to_string(), stats.avg_upload_ms);
        out.insert("last_upload_ms".to_string(), stats.last_upload_ms);
        out.insert("resident_megabytes".to_string(), stats.resident_megabytes);
        out.insert("source_count".to_string(), stats.source_count as f32);
        out.insert(
            "feedback_requests".to_string(),
            stats.feedback_requests as f32,
        );
        out
    }

    fn miss_rate(stats: TerrainMaterialVTStats) -> f32 {
        let total_requests = stats.cache_hits + stats.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            stats.cache_misses as f32 / total_requests as f32
        }
    }

    pub fn binding_resources(&self) -> Option<TerrainVTBindingResources<'_>> {
        self.runtime
            .as_ref()
            .map(|runtime| TerrainVTBindingResources {
                atlas_view: &runtime.atlas_view,
                page_table_view: &runtime.page_table_view,
                feedback_buffer: runtime
                    .feedback_buffer
                    .as_ref()
                    .map(|buffer| buffer.buffer()),
            })
    }

    fn selected_layer(
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Option<&crate::terrain::render_params::VTLayerFamilyNative> {
        if !vt.enabled {
            return None;
        }
        // The shipped terrain VT path currently samples only the albedo family.
        vt.layers
            .iter()
            .find(|layer| layer.family == TERRAIN_VT_SUPPORTED_FAMILY)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        material_count: u32,
        render_width: u32,
        render_height: u32,
        vt_uniform_buffer: &wgpu::Buffer,
        vt_fallback_uniform_buffer: &wgpu::Buffer,
    ) -> Result<bool, String> {
        let layer = match Self::selected_layer(&decoded.vt) {
            Some(layer) => layer,
            None => {
                self.runtime = None;
                self.last_stats = TerrainMaterialVTStats::default();
                Self::write_disabled_uniforms(queue, vt_uniform_buffer, vt_fallback_uniform_buffer);
                return Ok(false);
            }
        };

        let effective_material_count =
            material_count.clamp(1, super::core::MATERIAL_LAYER_CAPACITY as u32);
        self.ensure_runtime(device, layer, effective_material_count, &decoded.vt)?;
        let runtime = self.runtime.as_mut().unwrap();
        runtime.reset_frame_stats(decoded.vt.residency_budget_mb);

        let fallback_colors = runtime.fallback_colors(layer);
        Self::write_uniforms(queue, vt_uniform_buffer, runtime, true);
        queue.write_buffer(
            vt_fallback_uniform_buffer,
            0,
            bytemuck::cast_slice(&fallback_colors),
        );

        let requests =
            runtime.collect_requests(params, render_width, render_height, decoded.vt.use_feedback);
        for key in requests {
            runtime.ensure_tile_resident(device, queue, key)?;
        }
        runtime.upload_page_tables(queue);
        runtime.refresh_stats();
        self.last_stats = runtime.stats;

        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            feedback_buffer.clear(encoder);
            runtime.feedback_staged = false;
        }

        Ok(true)
    }
    pub fn stage_feedback_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(());
        };
        let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() else {
            return Ok(());
        };
        feedback_buffer.prepare_readback(encoder);
        runtime.feedback_staged = true;
        Ok(())
    }

    pub fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(());
        };
        if !runtime.feedback_staged {
            return Ok(());
        }

        runtime.pending_feedback.clear();
        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            let entries = feedback_buffer.read_feedback_entries(device, queue)?;
            for entry in entries {
                let material_index = entry.frame_number.saturating_sub(1);
                if !runtime.sources.contains_key(&material_index) {
                    continue;
                }
                if entry.mip_level >= runtime.max_mip_levels {
                    continue;
                }
                let (pages_x, pages_y) = runtime.pages_at_mip(entry.mip_level);
                if entry.tile_x >= pages_x || entry.tile_y >= pages_y {
                    continue;
                }
                runtime.pending_feedback.push(TileKey {
                    material_index,
                    x: entry.tile_x,
                    y: entry.tile_y,
                    mip_level: entry.mip_level,
                });
            }
            runtime.stats.feedback_requests = runtime.pending_feedback.len() as u32;
        }
        runtime.feedback_staged = false;
        self.last_stats = runtime.stats;
        Ok(())
    }
    fn write_disabled_uniforms(
        queue: &wgpu::Queue,
        vt_uniform_buffer: &wgpu::Buffer,
        vt_fallback_uniform_buffer: &wgpu::Buffer,
    ) {
        let uniforms = TerrainVTUniformsGpu {
            config0: [0, 0, 0, 0],
            config1: [0, 0, 0, 0],
            config2: [0, 0, 0, 0],
        };
        let fallback_colors = [[0.5, 0.5, 0.5, 1.0]; super::core::MATERIAL_LAYER_CAPACITY];
        queue.write_buffer(vt_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            vt_fallback_uniform_buffer,
            0,
            bytemuck::cast_slice(&fallback_colors),
        );
    }

    fn write_uniforms(
        queue: &wgpu::Queue,
        vt_uniform_buffer: &wgpu::Buffer,
        runtime: &TerrainMaterialVTRuntime,
        enabled: bool,
    ) {
        let uniforms = TerrainVTUniformsGpu {
            config0: [
                if enabled { 1 } else { 0 },
                runtime.tile_size,
                runtime.tile_border,
                runtime.atlas_size,
            ],
            config1: [
                runtime.virtual_size.0,
                runtime.virtual_size.1,
                runtime.pages_x0,
                runtime.pages_y0,
            ],
            config2: [
                runtime.max_mip_levels,
                runtime.material_count,
                runtime.slot_size,
                if runtime.use_feedback { 1 } else { 0 },
            ],
        };
        queue.write_buffer(vt_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn ensure_runtime(
        &mut self,
        device: &wgpu::Device,
        layer: &crate::terrain::render_params::VTLayerFamilyNative,
        material_count: u32,
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Result<(), String> {
        let full_levels = TerrainMaterialVTRuntime::full_pyramid_levels(
            layer.virtual_size.0,
            layer.virtual_size.1,
            layer.tile_size,
        );
        let max_mip_levels = vt.max_mip_levels.min(full_levels).max(1);

        let runtime_matches = self.runtime.as_ref().is_some_and(|runtime| {
            runtime.virtual_size == layer.virtual_size
                && runtime.tile_size == layer.tile_size
                && runtime.tile_border == layer.tile_border
                && runtime.atlas_size == vt.atlas_size
                && runtime.material_count == material_count
                && runtime.max_mip_levels == max_mip_levels
                && runtime.source_generation == self.source_generation
                && runtime.use_feedback == vt.use_feedback
        });
        if runtime_matches {
            return Ok(());
        }

        let runtime = TerrainMaterialVTRuntime::new(
            device,
            &self.sources,
            self.source_generation,
            layer,
            material_count,
            vt.atlas_size,
            vt.residency_budget_mb,
            max_mip_levels,
            vt.use_feedback,
        )?;
        self.last_stats = runtime.stats;
        self.runtime = Some(runtime);
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
impl TerrainMaterialVTRuntime {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        sources: &HashMap<(u32, String), VTSource>,
        source_generation: u64,
        layer: &crate::terrain::render_params::VTLayerFamilyNative,
        material_count: u32,
        atlas_size: u32,
        residency_budget_mb: f32,
        max_mip_levels: u32,
        use_feedback: bool,
    ) -> Result<Self, String> {
        let slot_size = layer.tile_size + 2 * layer.tile_border;
        let pages_x0 = ceil_div(layer.virtual_size.0, layer.tile_size);
        let pages_y0 = ceil_div(layer.virtual_size.1, layer.tile_size);
        let max_mip_levels = max_mip_levels
            .min(Self::page_table_mip_levels(pages_x0, pages_y0))
            .max(1);

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.material_vt.atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.material_vt.atlas.view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let page_table_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.material_vt.page_table"),
            size: wgpu::Extent3d {
                width: pages_x0,
                height: pages_y0,
                depth_or_array_layers: material_count * max_mip_levels,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let page_table_view = page_table_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.material_vt.page_table.view"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(material_count * max_mip_levels),
            ..Default::default()
        });

        let mut prepared_sources = HashMap::new();
        for ((material_index, family), source) in sources {
            if family != TERRAIN_VT_SUPPORTED_FAMILY || *material_index >= material_count {
                continue;
            }
            if source.virtual_size != layer.virtual_size {
                return Err(format!(
                    "VT source {:?} virtual size {:?} does not match layer contract {:?}",
                    (material_index, family),
                    source.virtual_size,
                    layer.virtual_size
                ));
            }
            prepared_sources.insert(
                *material_index,
                PreparedVTSource {
                    fallback_color: source.fallback_color,
                    mips: build_rgba_mip_chain(&source.data, source.virtual_size, max_mip_levels),
                },
            );
        }

        let total_pages =
            Self::total_pages_for(layer.virtual_size, layer.tile_size, max_mip_levels)
                .saturating_mul(prepared_sources.len() as u32);

        let atlas_slots_total = (atlas_size / slot_size) * (atlas_size / slot_size);
        let slot_bytes = slot_size as usize * slot_size as usize * TERRAIN_VT_BYTES_PER_PIXEL;
        let budget_bytes = (residency_budget_mb * 1024.0 * 1024.0).floor() as usize;
        let budget_pages = if slot_bytes == 0 {
            0
        } else {
            (budget_bytes / slot_bytes).max(1) as u32
        }
        .min(atlas_slots_total)
        .max(1);

        let mut tile_cache = TileCache::new(budget_pages as usize);
        tile_cache.configure_atlas(atlas_size, atlas_size, slot_size);

        let feedback_capacity = material_count
            .saturating_mul(max_mip_levels)
            .saturating_mul(pages_x0)
            .saturating_mul(pages_y0)
            .max(1);
        let feedback_buffer = if use_feedback {
            Some(FeedbackBuffer::new(device, feedback_capacity)?)
        } else {
            None
        };

        let mut page_tables = Vec::with_capacity((material_count * max_mip_levels) as usize);
        for _material_index in 0..material_count {
            for mip_level in 0..max_mip_levels {
                let (pages_x, pages_y) = pages_for_mip_counts(pages_x0, pages_y0, mip_level);
                page_tables.push(vec![
                    PageTableEntry::default();
                    (pages_x * pages_y) as usize
                ]);
            }
        }

        let mut runtime = Self {
            virtual_size: layer.virtual_size,
            tile_size: layer.tile_size,
            tile_border: layer.tile_border,
            slot_size,
            atlas_size,
            material_count,
            max_mip_levels,
            pages_x0,
            pages_y0,
            atlas_texture,
            atlas_view,
            page_table_texture,
            page_table_view,
            page_tables,
            sources: prepared_sources,
            tile_cache,
            feedback_buffer,
            pending_feedback: Vec::new(),
            feedback_staged: false,
            budget_pages,
            source_generation,
            use_feedback,
            stats: TerrainMaterialVTStats::default(),
        };
        runtime.stats.total_pages = total_pages;
        runtime.stats.cache_budget_pages = budget_pages;
        runtime.stats.cache_budget_mb = residency_budget_mb;
        runtime.stats.source_count = runtime.sources.len() as u32;
        Ok(runtime)
    }

    fn fallback_colors(
        &self,
        layer: &crate::terrain::render_params::VTLayerFamilyNative,
    ) -> [[f32; 4]; super::core::MATERIAL_LAYER_CAPACITY] {
        let mut colors = [layer.fallback; super::core::MATERIAL_LAYER_CAPACITY];
        for (material_index, source) in &self.sources {
            if (*material_index as usize) < colors.len() {
                colors[*material_index as usize] = source.fallback_color;
            }
        }
        colors
    }

    fn reset_frame_stats(&mut self, residency_budget_mb: f32) {
        self.stats.cache_hits = 0;
        self.stats.cache_misses = 0;
        self.stats.tiles_streamed = 0;
        self.stats.evictions = 0;
        self.stats.last_upload_ms = 0.0;
        self.stats.avg_upload_ms = 0.0;
        self.stats.cache_budget_pages = self.budget_pages;
        self.stats.cache_budget_mb = residency_budget_mb;
        self.stats.source_count = self.sources.len() as u32;
    }

    fn collect_requests(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_width: u32,
        render_height: u32,
        use_feedback: bool,
    ) -> Vec<TileKey> {
        let desired_mip = self.target_mip_level(params, render_width, render_height);
        let (uv_min, uv_max) = self.visible_uv_rect(params);
        let (pages_x, pages_y) = self.pages_at_mip(desired_mip);
        let start_x = ((uv_min[0] * pages_x as f32).floor() as i32).clamp(0, pages_x as i32 - 1);
        let start_y = ((uv_min[1] * pages_y as f32).floor() as i32).clamp(0, pages_y as i32 - 1);
        let end_x = ((uv_max[0] * pages_x as f32).ceil() as i32 - 1).clamp(0, pages_x as i32 - 1);
        let end_y = ((uv_max[1] * pages_y as f32).ceil() as i32 - 1).clamp(0, pages_y as i32 - 1);

        let mut requests = HashSet::new();
        for material_index in self.sources.keys().copied() {
            for y in start_y..=end_y {
                for x in start_x..=end_x {
                    self.insert_tile_with_ancestors(
                        &mut requests,
                        TileKey {
                            material_index,
                            x: x as u32,
                            y: y as u32,
                            mip_level: desired_mip,
                        },
                    );
                }
            }
        }

        if use_feedback {
            for feedback in &self.pending_feedback {
                if self.sources.contains_key(&feedback.material_index) {
                    self.insert_tile_with_ancestors(&mut requests, *feedback);
                }
            }
        }

        let mut ordered = requests.into_iter().collect::<Vec<_>>();
        ordered.sort_by_key(|key| (key.mip_level, key.material_index, key.y, key.x));
        ordered
    }

    fn ensure_tile_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
    ) -> Result<(), String> {
        let cache_tile = self.encode_cache_tile(key);
        if self.tile_cache.is_resident(&cache_tile) {
            self.tile_cache.access_tile(&cache_tile);
            self.stats.cache_hits += 1;
            return Ok(());
        }

        let Some(source) = self.sources.get(&key.material_index).cloned() else {
            return Ok(());
        };

        self.stats.cache_misses += 1;
        let Some((atlas_slot, evicted)) = self.tile_cache.allocate_tile_with_evicted(cache_tile)
        else {
            return Ok(());
        };
        for evicted_tile in evicted {
            self.clear_page_entry(self.decode_cache_tile(evicted_tile));
        }

        let tile_data = self.build_tile_data(&source, key);
        let upload_start = Instant::now();
        self.upload_tile_to_atlas(queue, &tile_data, atlas_slot);
        let upload_ms = upload_start.elapsed().as_secs_f32() * 1000.0;
        self.stats.tiles_streamed += 1;
        self.stats.last_upload_ms = upload_ms;
        let stream_count = self.stats.tiles_streamed.max(1) as f32;
        self.stats.avg_upload_ms =
            ((self.stats.avg_upload_ms * (stream_count - 1.0)) + upload_ms) / stream_count;
        self.stats.evictions = self.tile_cache.stats().evictions as u32;
        self.set_page_entry(key, atlas_slot);
        let _ = device;
        Ok(())
    }

    fn refresh_stats(&mut self) {
        self.stats.resident_pages = self.tile_cache.resident_count() as u32;
        let resident_bytes = self.stats.resident_pages as usize
            * self.slot_size as usize
            * self.slot_size as usize
            * TERRAIN_VT_BYTES_PER_PIXEL;
        self.stats.resident_megabytes = resident_bytes as f32 / (1024.0 * 1024.0);
    }

    fn upload_page_tables(&self, queue: &wgpu::Queue) {
        for material_index in 0..self.material_count {
            for mip_level in 0..self.max_mip_levels {
                let layer_index = self.layer_mip_index(material_index, mip_level);
                let entries = &self.page_tables[layer_index];
                let (pages_x, pages_y) = self.pages_at_mip(mip_level);
                let packed_entries = entries
                    .iter()
                    .map(|entry| {
                        [
                            entry.atlas_u,
                            entry.atlas_v,
                            if entry.is_resident > 0 { 1.0 } else { 0.0 },
                            entry.mip_bias,
                        ]
                    })
                    .collect::<Vec<_>>();
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.page_table_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_index as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&packed_entries),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(pages_x * 16),
                        rows_per_image: Some(pages_y),
                    },
                    wgpu::Extent3d {
                        width: pages_x,
                        height: pages_y,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }

    fn build_tile_data(&self, source: &PreparedVTSource, key: TileKey) -> TileData {
        let mip = &source.mips[key.mip_level as usize];
        let slot_size = self.slot_size as usize;
        let tile_size = self.tile_size as i32;
        let tile_border = self.tile_border as i32;
        let mut data = vec![0u8; slot_size * slot_size * TERRAIN_VT_BYTES_PER_PIXEL];

        for slot_y in 0..slot_size {
            for slot_x in 0..slot_size {
                let src_x = (key.x as i32 * tile_size + slot_x as i32 - tile_border)
                    .clamp(0, mip.width as i32 - 1) as usize;
                let src_y = (key.y as i32 * tile_size + slot_y as i32 - tile_border)
                    .clamp(0, mip.height as i32 - 1) as usize;
                let src_index = (src_y * mip.width as usize + src_x) * TERRAIN_VT_BYTES_PER_PIXEL;
                let dst_index = (slot_y * slot_size + slot_x) * TERRAIN_VT_BYTES_PER_PIXEL;
                data[dst_index..dst_index + TERRAIN_VT_BYTES_PER_PIXEL]
                    .copy_from_slice(&mip.data[src_index..src_index + TERRAIN_VT_BYTES_PER_PIXEL]);
            }
        }

        TileData {
            id: self.encode_cache_tile(key),
            data,
            width: self.slot_size,
            height: self.slot_size,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    fn upload_tile_to_atlas(
        &self,
        queue: &wgpu::Queue,
        tile_data: &TileData,
        atlas_slot: crate::core::tile_cache::AtlasSlot,
    ) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: atlas_slot.atlas_x,
                    y: atlas_slot.atlas_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &tile_data.data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tile_data.width * TERRAIN_VT_BYTES_PER_PIXEL as u32),
                rows_per_image: Some(tile_data.height),
            },
            wgpu::Extent3d {
                width: tile_data.width,
                height: tile_data.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn set_page_entry(&mut self, key: TileKey, atlas_slot: crate::core::tile_cache::AtlasSlot) {
        let layer_index = self.layer_mip_index(key.material_index, key.mip_level);
        let (pages_x, _pages_y) = self.pages_at_mip(key.mip_level);
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            entry.atlas_u = atlas_slot.atlas_u;
            entry.atlas_v = atlas_slot.atlas_v;
            entry.is_resident = 1;
            entry.mip_bias = 0.0;
        }
    }

    fn clear_page_entry(&mut self, key: TileKey) {
        if key.material_index >= self.material_count || key.mip_level >= self.max_mip_levels {
            return;
        }
        let layer_index = self.layer_mip_index(key.material_index, key.mip_level);
        let (pages_x, pages_y) = self.pages_at_mip(key.mip_level);
        if key.x >= pages_x || key.y >= pages_y {
            return;
        }
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            *entry = PageTableEntry::default();
        }
    }

    fn insert_tile_with_ancestors(&self, requests: &mut HashSet<TileKey>, mut key: TileKey) {
        loop {
            if !requests.insert(key) {
                break;
            }
            if key.mip_level + 1 >= self.max_mip_levels {
                break;
            }
            key = TileKey {
                material_index: key.material_index,
                x: key.x / 2,
                y: key.y / 2,
                mip_level: key.mip_level + 1,
            };
        }
    }

    fn visible_uv_rect(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> ([f32; 2], [f32; 2]) {
        if params.camera_mode.eq_ignore_ascii_case("mesh") {
            let aspect = params.size_px.0 as f32 / params.size_px.1.max(1) as f32;
            let center = [
                (params.cam_target[0] / params.terrain_span.max(1e-3)) + 0.5,
                (params.cam_target[1] / params.terrain_span.max(1e-3)) + 0.5,
            ];
            let half_height =
                params.cam_radius.max(1.0) * (params.fov_y_deg.to_radians() * 0.5).tan();
            let half_width = half_height * aspect;
            let span_u = ((half_width * 2.5) / params.terrain_span.max(1e-3)).clamp(0.05, 1.0);
            let span_v = ((half_height * 2.5) / params.terrain_span.max(1e-3)).clamp(0.05, 1.0);
            let min = [
                (center[0] - span_u * 0.5).clamp(0.0, 1.0),
                (center[1] - span_v * 0.5).clamp(0.0, 1.0),
            ];
            let max = [
                (center[0] + span_u * 0.5).clamp(0.0, 1.0),
                (center[1] + span_v * 0.5).clamp(0.0, 1.0),
            ];
            (min, max)
        } else {
            ([0.0, 0.0], [1.0, 1.0])
        }
    }

    fn target_mip_level(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_width: u32,
        render_height: u32,
    ) -> u32 {
        let (uv_min, uv_max) = self.visible_uv_rect(params);
        let uv_span_x = (uv_max[0] - uv_min[0]).max(1.0 / render_width.max(1) as f32);
        let uv_span_y = (uv_max[1] - uv_min[1]).max(1.0 / render_height.max(1) as f32);
        let texels_per_pixel_x =
            self.virtual_size.0 as f32 * uv_span_x / render_width.max(1) as f32;
        let texels_per_pixel_y =
            self.virtual_size.1 as f32 * uv_span_y / render_height.max(1) as f32;
        let texels_per_pixel = texels_per_pixel_x.max(texels_per_pixel_y).max(1.0);
        let desired = texels_per_pixel.log2().floor().max(0.0) as u32;
        desired.min(self.max_mip_levels.saturating_sub(1))
    }

    fn pages_at_mip(&self, mip_level: u32) -> (u32, u32) {
        pages_for_mip_counts(self.pages_x0, self.pages_y0, mip_level)
    }

    fn layer_mip_index(&self, material_index: u32, mip_level: u32) -> usize {
        (material_index * self.max_mip_levels + mip_level) as usize
    }

    fn encode_cache_tile(&self, key: TileKey) -> TileId {
        TileId {
            x: key.material_index * self.pages_x0.max(1) + key.x,
            y: key.y,
            mip_level: key.mip_level,
        }
    }

    fn decode_cache_tile(&self, tile: TileId) -> TileKey {
        TileKey {
            material_index: tile.x / self.pages_x0.max(1),
            x: tile.x % self.pages_x0.max(1),
            y: tile.y,
            mip_level: tile.mip_level,
        }
    }

    fn total_pages_for(virtual_size: (u32, u32), tile_size: u32, max_mip_levels: u32) -> u32 {
        let pages_x0 = ceil_div(virtual_size.0, tile_size);
        let pages_y0 = ceil_div(virtual_size.1, tile_size);
        let mut total = 0u32;
        for mip_level in 0..max_mip_levels {
            let (pages_x, pages_y) = pages_for_mip_counts(pages_x0, pages_y0, mip_level);
            total = total.saturating_add(pages_x.saturating_mul(pages_y));
        }
        total
    }

    fn full_pyramid_levels(width: u32, height: u32, tile_size: u32) -> u32 {
        let pages_x = ceil_div(width, tile_size).max(1);
        let pages_y = ceil_div(height, tile_size).max(1);
        Self::page_table_mip_levels(pages_x, pages_y)
    }

    fn page_table_mip_levels(pages_x0: u32, pages_y0: u32) -> u32 {
        let max_dim = pages_x0.max(pages_y0).max(1);
        u32::BITS - max_dim.leading_zeros()
    }
}

#[cfg(feature = "extension-module")]
fn ceil_div(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor.max(1)
}

#[cfg(feature = "extension-module")]
fn pages_for_mip_counts(pages_x0: u32, pages_y0: u32, mip_level: u32) -> (u32, u32) {
    let div = 1u32.checked_shl(mip_level).unwrap_or(u32::MAX).max(1);
    (
        ceil_div(pages_x0.max(1), div).max(1),
        ceil_div(pages_y0.max(1), div).max(1),
    )
}

#[cfg(feature = "extension-module")]
fn build_rgba_mip_chain(data: &[u8], size: (u32, u32), max_mip_levels: u32) -> Vec<MipImage> {
    let mut chain = Vec::with_capacity(max_mip_levels as usize);
    chain.push(MipImage {
        width: size.0,
        height: size.1,
        data: data.to_vec(),
    });

    while chain.len() < max_mip_levels as usize {
        let previous = chain.last().unwrap().clone();
        if previous.width == 1 && previous.height == 1 {
            chain.push(previous);
            continue;
        }

        let next_width = previous.width.max(1).div_ceil(2);
        let next_height = previous.height.max(1).div_ceil(2);
        let mut next_data =
            vec![0u8; next_width as usize * next_height as usize * TERRAIN_VT_BYTES_PER_PIXEL];

        for y in 0..next_height {
            for x in 0..next_width {
                let mut accum = [0u32; TERRAIN_VT_BYTES_PER_PIXEL];
                let mut sample_count = 0u32;
                for src_y in (y * 2)..((y * 2 + 2).min(previous.height)) {
                    for src_x in (x * 2)..((x * 2 + 2).min(previous.width)) {
                        let src_index = (src_y as usize * previous.width as usize + src_x as usize)
                            * TERRAIN_VT_BYTES_PER_PIXEL;
                        for channel in 0..TERRAIN_VT_BYTES_PER_PIXEL {
                            accum[channel] += previous.data[src_index + channel] as u32;
                        }
                        sample_count += 1;
                    }
                }

                let dst_index =
                    (y as usize * next_width as usize + x as usize) * TERRAIN_VT_BYTES_PER_PIXEL;
                for channel in 0..TERRAIN_VT_BYTES_PER_PIXEL {
                    next_data[dst_index + channel] = (accum[channel] / sample_count.max(1)) as u8;
                }
            }
        }

        chain.push(MipImage {
            width: next_width,
            height: next_height,
            data: next_data,
        });
    }

    chain
}

#[cfg(feature = "extension-module")]
impl TerrainScene {
    pub(super) fn prepare_material_vt_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        material_count: u32,
        render_width: u32,
        render_height: u32,
    ) -> Result<bool> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .prepare_frame(
                encoder,
                self.device.as_ref(),
                self.queue.as_ref(),
                params,
                decoded,
                material_count,
                render_width,
                render_height,
                &self.vt_uniform_buffer,
                &self.vt_fallback_uniform_buffer,
            )
            .map_err(anyhow::Error::msg)
    }
    pub(super) fn stage_material_vt_feedback_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .stage_feedback_readback(encoder)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn finish_material_vt_frame(&self) -> Result<()> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .finish_frame(self.device.as_ref(), self.queue.as_ref())
            .map_err(anyhow::Error::msg)
    }
}

#[cfg(not(feature = "extension-module"))]
pub(super) struct TerrainMaterialVT;

#[cfg(not(feature = "extension-module"))]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self
    }
}
