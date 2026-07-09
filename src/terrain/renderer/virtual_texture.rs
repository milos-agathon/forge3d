use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "extension-module")]
use super::*;

#[cfg(feature = "extension-module")]
use crate::core::feedback_buffer::FeedbackBuffer;
#[cfg(feature = "extension-module")]
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
#[cfg(feature = "enable-staging-rings")]
use crate::core::staging_rings::StagingRing;
#[cfg(feature = "extension-module")]
use crate::core::tile_cache::{TileCache, TileData, TileId};
#[cfg(feature = "extension-module")]
use crate::terrain::vt_family_residency::{
    decode_feedback_payload, FamilyResidency, FamilyResidencyTracker, TileKey, VT_FAMILY_COUNT,
};

#[cfg(feature = "extension-module")]
const TERRAIN_VT_SUPPORTED_FAMILIES: &[&str] = &["albedo", "normal", "mask"];
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_COUNT: u32 = 3;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_ALBEDO: u32 = 0;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_NORMAL: u32 = 1;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_MASK: u32 = 2;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_BYTES_PER_PIXEL: usize = 4;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FALLBACK_COUNT: usize =
    super::core::MATERIAL_LAYER_CAPACITY * TERRAIN_VT_FAMILY_COUNT as usize;

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub(super) struct VTSource {
    pub virtual_size: (u32, u32),
    pub data: Vec<u8>,
    pub fallback_color: [f32; 4],
    /// VERITAS: stable, device-independent source id
    /// (`family_slot * 4 + material_index + 1`; 0 == SOURCE_ID_NONE).
    pub source_id: u32,
    /// VERITAS: SHA256 of `data`, computed once at ingest.
    pub content_hash: [u8; 32],
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
    /// Per-family info (`TerrainVtFamilyInfo`): the single source of truth the
    /// shader reads per family. x = enabled (0/1), y = page-table layer
    /// offset, z = atlas layer (0 while all families share one atlas layer),
    /// w = registered source count. Matches `family_info` in
    /// `terrain_pbr_pom.wgsl`; refreshed every `prepare_frame`.
    family_info: [[u32; 4]; TERRAIN_VT_FAMILY_COUNT as usize],
}

#[cfg(feature = "extension-module")]
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct PageTableEntry {
    atlas_u: f32,
    atlas_v: f32,
    is_resident: u32,
    mip_bias: f32,
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
    /// VERITAS: stable source id + SHA256 of the source payload (both
    /// copied from `VTSource`, assigned at ingest).
    source_id: u32,
    content_hash: [u8; 32],
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
    families: [FamilyResidency; VT_FAMILY_COUNT],
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
    atlas_texture: TrackedTexture,
    atlas_view: wgpu::TextureView,
    #[cfg(feature = "enable-staging-rings")]
    staging_ring: StagingRing,
    page_table_texture: TrackedTexture,
    page_table_view: wgpu::TextureView,
    page_tables: Vec<Vec<PageTableEntry>>,
    dirty_page_table_layers: HashSet<usize>,
    sources: HashMap<(u32, u32), PreparedVTSource>,
    tile_cache: TileCache,
    family_residency: FamilyResidencyTracker,
    feedback_buffer: Option<FeedbackBuffer>,
    pending_feedback: [Vec<TileKey>; VT_FAMILY_COUNT],
    feedback_staged: bool,
    budget_pages: u32,
    residency_budget_mb: f32,
    source_generation: u64,
    use_feedback: bool,
    family_mask: u32,
    layer_fallbacks: [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize],
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
        let family_supported = TERRAIN_VT_SUPPORTED_FAMILIES.contains(&family.as_str());
        if !family_supported {
            log::warn!(
                "terrain material VT received unsupported family '{family}'; storing it for diagnostics but the native runtime will ignore it",
                family = family,
            );
        }
        if family_supported {
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

        // VERITAS provenance identity: derive the stable source id from the
        // (family, material) slot so it is reproducible across devices and
        // registration orders; hash the payload once at ingest.
        let source_id = Self::family_slot(&family)
            .map_or(crate::core::provenance::SOURCE_ID_NONE, |family_slot| {
                crate::core::provenance::source_id_for(family_slot, material_index)
            });
        let content_hash = crate::core::provenance::sha256(&data);
        self.sources.insert(
            (material_index, family),
            VTSource {
                virtual_size: virtual_size_px,
                data,
                fallback_color,
                source_id,
                content_hash,
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
        let mut resident_bytes_total = 0u64;
        for (slot, name) in TERRAIN_VT_SUPPORTED_FAMILIES.iter().enumerate() {
            let family = stats.families[slot];
            out.insert(
                format!("resident_tiles_{name}"),
                family.resident_tiles as f32,
            );
            out.insert(
                format!("resident_bytes_{name}"),
                family.resident_bytes as f32,
            );
            out.insert(format!("budget_bytes_{name}"), family.budget_bytes as f32);
            resident_bytes_total += family.resident_bytes;
        }
        out.insert(
            "resident_bytes_total".to_string(),
            resident_bytes_total as f32,
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

    fn family_slot(family: &str) -> Option<u32> {
        match family {
            "albedo" => Some(TERRAIN_VT_FAMILY_ALBEDO),
            "normal" => Some(TERRAIN_VT_FAMILY_NORMAL),
            "mask" => Some(TERRAIN_VT_FAMILY_MASK),
            _ => None,
        }
    }

    fn active_layers(
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Vec<&crate::terrain::render_params::VTLayerFamilyNative> {
        if !vt.enabled {
            return Vec::new();
        }
        vt.layers
            .iter()
            .filter(|layer| TERRAIN_VT_SUPPORTED_FAMILIES.contains(&layer.family.as_str()))
            .collect()
    }

    fn compatible_layout<'a>(
        layers: &'a [&crate::terrain::render_params::VTLayerFamilyNative],
    ) -> Result<&'a crate::terrain::render_params::VTLayerFamilyNative, String> {
        let Some(first) = layers.first().copied() else {
            return Err("terrain VT requires at least one supported family".to_string());
        };
        for layer in layers.iter().copied().skip(1) {
            if layer.virtual_size != first.virtual_size
                || layer.tile_size != first.tile_size
                || layer.tile_border != first.tile_border
            {
                return Err(format!(
                    "terrain VT families must share virtual_size_px/tile_size/tile_border; '{}' has {:?}/{}.{}, '{}' has {:?}/{}.{}",
                    first.family,
                    first.virtual_size,
                    first.tile_size,
                    first.tile_border,
                    layer.family,
                    layer.virtual_size,
                    layer.tile_size,
                    layer.tile_border,
                ));
            }
        }
        Ok(first)
    }

    fn family_mask(layers: &[&crate::terrain::render_params::VTLayerFamilyNative]) -> u32 {
        layers.iter().fold(0u32, |mask, layer| {
            mask | Self::family_slot(&layer.family).map_or(0u32, |slot| 1u32 << slot)
        })
    }

    fn default_family_fallbacks() -> [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize] {
        let mut fallbacks = [[0.5, 0.5, 0.5, 1.0]; TERRAIN_VT_FAMILY_COUNT as usize];
        fallbacks[TERRAIN_VT_FAMILY_NORMAL as usize] = [0.5, 0.5, 1.0, 1.0];
        fallbacks[TERRAIN_VT_FAMILY_MASK as usize] = [1.0, 1.0, 1.0, 1.0];
        fallbacks
    }

    fn layer_fallbacks(
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
    ) -> [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize] {
        let mut fallbacks = Self::default_family_fallbacks();
        for layer in layers {
            if let Some(slot) = Self::family_slot(&layer.family) {
                fallbacks[slot as usize] = layer.fallback;
            }
        }
        fallbacks
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        material_count: u32,
        render_width: u32,
        render_height: u32,
        vt_uniform_buffer: &wgpu::Buffer,
        vt_fallback_uniform_buffer: &wgpu::Buffer,
    ) -> Result<bool, String> {
        let layers = Self::active_layers(&decoded.vt);
        if layers.is_empty() {
            self.runtime = None;
            self.last_stats = TerrainMaterialVTStats::default();
            Self::write_disabled_uniforms(
                queue.as_ref(),
                vt_uniform_buffer,
                vt_fallback_uniform_buffer,
            );
            return Ok(false);
        }

        // A requested family with no registered source is a fatal diagnostic:
        // rendering would silently degrade the PBR result (e.g. normal-mapped
        // lighting collapsing to fallback colors), so refuse instead.
        for layer in &layers {
            let has_source = self
                .sources
                .keys()
                .any(|(_, family)| family == &layer.family);
            if !has_source {
                return Err(format!(
                    "terrain VT: family '{}' requested but no source registered; refusing to render with corrupted PBR",
                    layer.family
                ));
            }
        }

        let effective_material_count =
            material_count.clamp(1, super::core::MATERIAL_LAYER_CAPACITY as u32);
        self.ensure_runtime(
            device,
            queue,
            &layers,
            effective_material_count,
            &decoded.vt,
        )?;
        let runtime = self.runtime.as_mut().unwrap();
        runtime.reset_frame_stats(decoded.vt.residency_budget_mb);

        let fallback_colors = runtime.fallback_colors();
        Self::write_uniforms(queue.as_ref(), vt_uniform_buffer, runtime, true);
        queue.write_buffer(
            vt_fallback_uniform_buffer,
            0,
            bytemuck::cast_slice(&fallback_colors),
        );

        let requests =
            runtime.collect_requests(params, render_width, render_height, decoded.vt.use_feedback);
        for key in requests {
            runtime.ensure_tile_resident(encoder, device.as_ref(), queue.as_ref(), key)?;
        }
        runtime.upload_page_tables(queue.as_ref());
        runtime.refresh_stats();
        self.last_stats = runtime.stats;

        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            feedback_buffer.clear(encoder);
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
        if feedback_buffer.has_pending_readback() {
            return Ok(());
        }
        feedback_buffer.prepare_readback(encoder);
        runtime.feedback_staged = true;
        Ok(())
    }

    pub fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(());
        };
        if !runtime.feedback_staged {
            return Ok(());
        }

        for bucket in runtime.pending_feedback.iter_mut() {
            bucket.clear();
        }
        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            let Some(entries) = feedback_buffer.try_read_feedback_entries(device)? else {
                return Ok(());
            };
            // Demux decoded entries by family so each family drives its own
            // requested tile/mip set; a family with no feedback this frame
            // simply contributes an empty bucket.
            for entry in entries {
                let Some((family_slot, material_index)) =
                    decode_feedback_payload(entry.frame_number, runtime.material_count)
                else {
                    continue;
                };
                if !runtime.sources.contains_key(&(family_slot, material_index)) {
                    continue;
                }
                if entry.mip_level >= runtime.max_mip_levels {
                    continue;
                }
                let (pages_x, pages_y) = runtime.pages_at_mip(entry.mip_level);
                if entry.tile_x >= pages_x || entry.tile_y >= pages_y {
                    continue;
                }
                runtime.pending_feedback[family_slot as usize].push(TileKey {
                    family_slot,
                    material_index,
                    x: entry.tile_x,
                    y: entry.tile_y,
                    mip_level: entry.mip_level,
                });
            }
            runtime.stats.feedback_requests = runtime
                .pending_feedback
                .iter()
                .map(|bucket| bucket.len() as u32)
                .sum();
        }
        runtime.feedback_staged = false;
        self.last_stats = runtime.stats;
        Ok(())
    }

    /// VERITAS: drain the feedback buffer (blocking) and resolve each sampled
    /// tile to the resident mip the shader actually landed on this frame.
    ///
    /// The GPU walk starts at the desired mip and climbs coarser until a
    /// page-table entry is resident; this replays the identical walk against
    /// the CPU page-table mirror (which was uploaded before the pass and is
    /// unchanged since), so the leaf set describes exactly the tiles the
    /// composite sampled. Feedback chains with no resident ancestor sampled
    /// the fallback color and contribute no leaf (SOURCE_ID_NONE pixels).
    pub fn read_contributing_tiles(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Vec<crate::core::provenance::ContributingTile>, String> {
        use crate::core::provenance::ContributingTile;

        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(Vec::new());
        };
        let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() else {
            return Ok(Vec::new());
        };

        let entries = feedback_buffer.read_feedback_entries_blocking(device)?;
        runtime.feedback_staged = false;

        let mut resolved = HashSet::new();
        for entry in entries {
            let Some((family_slot, material_index)) =
                decode_feedback_payload(entry.frame_number, runtime.material_count)
            else {
                continue;
            };
            if entry.mip_level >= runtime.max_mip_levels {
                continue;
            }
            let (pages_x, pages_y) = runtime.pages_at_mip(entry.mip_level);
            if entry.tile_x >= pages_x || entry.tile_y >= pages_y {
                continue;
            }
            let key = TileKey {
                family_slot,
                material_index,
                x: entry.tile_x,
                y: entry.tile_y,
                mip_level: entry.mip_level,
            };
            if let Some(resident) = runtime.resolve_resident_mip(key) {
                resolved.insert(resident);
            }
        }

        let mut tiles = Vec::with_capacity(resolved.len());
        for key in resolved {
            let Some(source) = runtime.sources.get(&(key.family_slot, key.material_index)) else {
                continue;
            };
            tiles.push(ContributingTile {
                family_slot: key.family_slot,
                source_id: source.source_id,
                tile_x: key.x,
                tile_y: key.y,
                mip_level: key.mip_level,
                content_hash: source.content_hash,
            });
        }
        tiles.sort_by_key(|tile| {
            (
                tile.family_slot,
                tile.source_id,
                tile.mip_level,
                tile.tile_y,
                tile.tile_x,
            )
        });
        Ok(tiles)
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
            family_info: [[0, 0, 0, 0]; TERRAIN_VT_FAMILY_COUNT as usize],
        };
        let fallback_colors = TerrainMaterialVTRuntime::default_fallback_colors();
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
        let mut family_info = [[0u32; 4]; TERRAIN_VT_FAMILY_COUNT as usize];
        for (slot, info) in family_info.iter_mut().enumerate() {
            let slot_u32 = slot as u32;
            let family_enabled = enabled && (runtime.family_mask & (1u32 << slot_u32)) != 0;
            let source_count = runtime
                .sources
                .keys()
                .filter(|(family_slot, _)| *family_slot == slot_u32)
                .count() as u32;
            *info = [
                if family_enabled && source_count > 0 {
                    1
                } else {
                    0
                },
                slot_u32 * runtime.material_count * runtime.max_mip_levels,
                0,
                source_count,
            ];
        }
        let uniforms = TerrainVTUniformsGpu {
            config0: [
                if enabled { runtime.family_mask } else { 0 },
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
            family_info,
        };
        queue.write_buffer(vt_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn ensure_runtime(
        &mut self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
        material_count: u32,
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Result<(), String> {
        let layer = Self::compatible_layout(layers)?;
        let family_mask = Self::family_mask(layers);
        let layer_fallbacks = Self::layer_fallbacks(layers);
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
                && runtime.family_mask == family_mask
                && runtime.layer_fallbacks == layer_fallbacks
                // A budget change must rebuild so the shared tile-cache
                // capacity and the per-family budgets both pick it up.
                && runtime.residency_budget_mb == vt.residency_budget_mb
        });
        if runtime_matches {
            return Ok(());
        }

        let runtime = TerrainMaterialVTRuntime::new(
            device,
            queue,
            &self.sources,
            self.source_generation,
            layers,
            layer,
            family_mask,
            layer_fallbacks,
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
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        sources: &HashMap<(u32, String), VTSource>,
        source_generation: u64,
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
        layer: &crate::terrain::render_params::VTLayerFamilyNative,
        family_mask: u32,
        layer_fallbacks: [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize],
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

        let atlas_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
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
            },
        )
        .map_err(|e| e.to_string())?;
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.material_vt.atlas.view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        #[cfg(feature = "enable-staging-rings")]
        let staging_ring = {
            let max_tile_bytes =
                slot_size as u64 * slot_size as u64 * TERRAIN_VT_BYTES_PER_PIXEL as u64;
            let buffer_size = max_tile_bytes.max(8 * 1024 * 1024);
            StagingRing::new(device.clone(), queue.clone(), 3, buffer_size)
                .map_err(|e| e.to_string())?
        };

        let page_table_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.material_vt.page_table"),
                size: wgpu::Extent3d {
                    width: pages_x0,
                    height: pages_y0,
                    depth_or_array_layers: TERRAIN_VT_FAMILY_COUNT
                        * material_count
                        * max_mip_levels,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
        .map_err(|e| e.to_string())?;
        let page_table_view = page_table_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.material_vt.page_table.view"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(TERRAIN_VT_FAMILY_COUNT * material_count * max_mip_levels),
            ..Default::default()
        });

        let mut prepared_sources = HashMap::new();
        for ((material_index, family), source) in sources {
            let Some(family_slot) = TerrainMaterialVT::family_slot(family) else {
                continue;
            };
            if family_mask & (1u32 << family_slot) == 0 || *material_index >= material_count {
                continue;
            };
            let Some(layer_for_family) =
                layers.iter().find(|candidate| candidate.family == *family)
            else {
                continue;
            };
            if source.virtual_size != layer_for_family.virtual_size {
                return Err(format!(
                    "VT source {:?} virtual size {:?} does not match layer contract {:?}",
                    (material_index, family),
                    source.virtual_size,
                    layer_for_family.virtual_size
                ));
            }
            prepared_sources.insert(
                (family_slot, *material_index),
                PreparedVTSource {
                    fallback_color: source.fallback_color,
                    mips: build_rgba_mip_chain(&source.data, source.virtual_size, max_mip_levels),
                    source_id: source.source_id,
                    content_hash: source.content_hash,
                },
            );
        }

        let total_pages =
            Self::total_pages_for(layer.virtual_size, layer.tile_size, max_mip_levels)
                .saturating_mul(prepared_sources.len() as u32);

        let atlas_slots_total = (atlas_size / slot_size) * (atlas_size / slot_size);
        let slot_bytes = slot_size as usize * slot_size as usize * TERRAIN_VT_BYTES_PER_PIXEL;
        let budget_bytes = (residency_budget_mb * 1024.0 * 1024.0).floor() as usize;
        let budget_pages = budget_bytes.checked_div(slot_bytes).unwrap_or(0).max(1) as u32;
        let budget_pages = budget_pages.min(atlas_slots_total).max(1);

        // Per-family budgets: even split of the VT residency budget across the
        // enabled families; within-family LRU eviction keeps each family under
        // its own share before the shared tile cache evicts globally.
        let family_residency =
            FamilyResidencyTracker::new(budget_bytes as u64, family_mask, slot_bytes as u64);

        let mut tile_cache = TileCache::new(budget_pages as usize);
        tile_cache.configure_atlas(atlas_size, atlas_size, slot_size);

        let feedback_capacity = material_count
            .saturating_mul(TERRAIN_VT_FAMILY_COUNT)
            .saturating_mul(max_mip_levels)
            .saturating_mul(pages_x0)
            .saturating_mul(pages_y0)
            .max(1);
        let feedback_buffer = if use_feedback {
            Some(FeedbackBuffer::new(device, feedback_capacity)?)
        } else {
            None
        };

        // Route the VT footprint through the 512 MiB resource registry so the
        // budget tracker sees the atlas, page table, and feedback buffers.
        let memory_tracker = crate::core::memory_tracker::global_tracker();
        let page_table_layers = TERRAIN_VT_FAMILY_COUNT * material_count * max_mip_levels;
        memory_tracker.track_texture_allocation(
            atlas_size,
            atlas_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        memory_tracker.track_texture_allocation(
            pages_x0,
            pages_y0.saturating_mul(page_table_layers),
            wgpu::TextureFormat::Rgba32Float,
        );
        if let Some(feedback) = feedback_buffer.as_ref() {
            let feedback_bytes = feedback.buffer().size();
            memory_tracker.track_buffer_allocation(feedback_bytes, false);
            // The readback staging buffer is host-visible (MAP_READ).
            memory_tracker.track_buffer_allocation(feedback_bytes, true);
        }

        let mut page_tables = Vec::with_capacity(
            (TERRAIN_VT_FAMILY_COUNT * material_count * max_mip_levels) as usize,
        );
        for _family_slot in 0..TERRAIN_VT_FAMILY_COUNT {
            for _material_index in 0..material_count {
                for mip_level in 0..max_mip_levels {
                    let (pages_x, pages_y) = pages_for_mip_counts(pages_x0, pages_y0, mip_level);
                    page_tables.push(vec![
                        PageTableEntry::default();
                        (pages_x * pages_y) as usize
                    ]);
                }
            }
        }

        let dirty_page_table_layers = (0..page_tables.len()).collect();

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
            #[cfg(feature = "enable-staging-rings")]
            staging_ring,
            page_table_texture,
            page_table_view,
            page_tables,
            dirty_page_table_layers,
            sources: prepared_sources,
            tile_cache,
            family_residency,
            feedback_buffer,
            pending_feedback: Default::default(),
            feedback_staged: false,
            budget_pages,
            residency_budget_mb,
            source_generation,
            use_feedback,
            family_mask,
            layer_fallbacks,
            stats: TerrainMaterialVTStats::default(),
        };
        runtime.stats.total_pages = total_pages;
        runtime.stats.cache_budget_pages = budget_pages;
        runtime.stats.cache_budget_mb = residency_budget_mb;
        runtime.stats.source_count = runtime.sources.len() as u32;
        Ok(runtime)
    }

    fn default_fallback_colors() -> [[f32; 4]; TERRAIN_VT_FALLBACK_COUNT] {
        let mut colors = [[0.5, 0.5, 0.5, 1.0]; TERRAIN_VT_FALLBACK_COUNT];
        for material_index in 0..super::core::MATERIAL_LAYER_CAPACITY {
            colors[TERRAIN_VT_FAMILY_NORMAL as usize * super::core::MATERIAL_LAYER_CAPACITY
                + material_index] = [0.5, 0.5, 1.0, 1.0];
            colors[TERRAIN_VT_FAMILY_MASK as usize * super::core::MATERIAL_LAYER_CAPACITY
                + material_index] = [1.0, 1.0, 1.0, 1.0];
        }
        colors
    }

    fn fallback_colors(&self) -> [[f32; 4]; TERRAIN_VT_FALLBACK_COUNT] {
        let mut colors = Self::default_fallback_colors();
        for family_slot in 0..TERRAIN_VT_FAMILY_COUNT {
            for material_index in 0..super::core::MATERIAL_LAYER_CAPACITY {
                colors[family_slot as usize * super::core::MATERIAL_LAYER_CAPACITY
                    + material_index] = self.layer_fallbacks[family_slot as usize];
            }
        }
        for ((family_slot, material_index), source) in &self.sources {
            if *family_slot < TERRAIN_VT_FAMILY_COUNT
                && (*material_index as usize) < super::core::MATERIAL_LAYER_CAPACITY
            {
                colors[*family_slot as usize * super::core::MATERIAL_LAYER_CAPACITY
                    + *material_index as usize] = source.fallback_color;
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
        for (family_slot, material_index) in self.sources.keys().copied() {
            for y in start_y..=end_y {
                for x in start_x..=end_x {
                    self.insert_tile_with_ancestors(
                        &mut requests,
                        TileKey {
                            family_slot,
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
            for feedback in self.pending_feedback.iter().flatten() {
                if self
                    .sources
                    .contains_key(&(feedback.family_slot, feedback.material_index))
                {
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
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
    ) -> Result<(), String> {
        let cache_tile = self.encode_cache_tile(key);
        if self.tile_cache.is_resident(&cache_tile) {
            self.tile_cache.access_tile(&cache_tile);
            self.family_residency.note_access(key);
            self.stats.cache_hits += 1;
            return Ok(());
        }

        let Some(source) = self
            .sources
            .get(&(key.family_slot, key.material_index))
            .cloned()
        else {
            return Ok(());
        };

        self.stats.cache_misses += 1;
        // Enforce the family's own residency budget first: evict within-family
        // LRU tiles before touching the shared pool, so one family's paging
        // pressure never drains another family's resident set.
        while self.family_residency.needs_eviction(key.family_slot) {
            let Some(victim) = self.family_residency.lru_tile(key.family_slot) else {
                break;
            };
            let victim_tile = self.encode_cache_tile(victim);
            self.tile_cache.evict_tile(&victim_tile);
            self.family_residency.on_evict(&victim);
            self.clear_page_entry(victim);
        }
        let Some((atlas_slot, evicted)) = self.tile_cache.allocate_tile_with_evicted(cache_tile)
        else {
            return Ok(());
        };
        for evicted_tile in evicted {
            let victim = self.decode_cache_tile(evicted_tile);
            self.family_residency.on_evict(&victim);
            self.clear_page_entry(victim);
        }

        let tile_data = self.build_tile_data(&source, key);
        let upload_start = Instant::now();
        self.upload_tile_to_atlas(encoder, queue, &tile_data, atlas_slot);
        let upload_ms = upload_start.elapsed().as_secs_f32() * 1000.0;
        self.stats.tiles_streamed += 1;
        self.stats.last_upload_ms = upload_ms;
        let stream_count = self.stats.tiles_streamed.max(1) as f32;
        self.stats.avg_upload_ms =
            ((self.stats.avg_upload_ms * (stream_count - 1.0)) + upload_ms) / stream_count;
        self.stats.evictions = self.tile_cache.stats().evictions as u32;
        self.set_page_entry(key, atlas_slot);
        self.family_residency.on_insert(key);
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
        for slot in 0..VT_FAMILY_COUNT {
            self.stats.families[slot] = self.family_residency.family(slot as u32);
        }
        crate::core::memory_tracker::global_tracker().set_resident_tiles(
            self.family_residency.total_resident_tiles(),
            self.family_residency.total_resident_bytes(),
        );
    }

    fn upload_page_tables(&mut self, queue: &wgpu::Queue) {
        let mut dirty_layers = self.dirty_page_table_layers.drain().collect::<Vec<_>>();
        dirty_layers.sort_unstable();

        for layer_index in dirty_layers {
            let mip_level = (layer_index as u32) % self.max_mip_levels;
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
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        tile_data: &TileData,
        atlas_slot: crate::core::tile_cache::AtlasSlot,
    ) {
        let origin = wgpu::Origin3d {
            x: atlas_slot.atlas_x,
            y: atlas_slot.atlas_y,
            z: 0,
        };
        #[cfg(feature = "enable-staging-rings")]
        {
            if self.staging_ring.upload_texture_region(
                encoder,
                queue,
                &self.atlas_texture,
                origin,
                &tile_data.data,
                tile_data.width,
                tile_data.height,
                TERRAIN_VT_BYTES_PER_PIXEL as u32,
            ) {
                return;
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin,
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
        let layer_index = self.layer_mip_index(key.family_slot, key.material_index, key.mip_level);
        let (pages_x, _pages_y) = self.pages_at_mip(key.mip_level);
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            entry.atlas_u = atlas_slot.atlas_u;
            entry.atlas_v = atlas_slot.atlas_v;
            entry.is_resident = 1;
            entry.mip_bias = 0.0;
            self.dirty_page_table_layers.insert(layer_index);
        }
    }

    fn clear_page_entry(&mut self, key: TileKey) {
        if key.family_slot >= TERRAIN_VT_FAMILY_COUNT
            || key.material_index >= self.material_count
            || key.mip_level >= self.max_mip_levels
        {
            return;
        }
        let layer_index = self.layer_mip_index(key.family_slot, key.material_index, key.mip_level);
        let (pages_x, pages_y) = self.pages_at_mip(key.mip_level);
        if key.x >= pages_x || key.y >= pages_y {
            return;
        }
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            *entry = PageTableEntry::default();
            self.dirty_page_table_layers.insert(layer_index);
        }
    }

    /// VERITAS: replay the shader's residency walk on the CPU page-table
    /// mirror — climb from `key.mip_level` toward coarser mips and return the
    /// first resident tile, or `None` when the whole chain is non-resident.
    fn resolve_resident_mip(&self, key: TileKey) -> Option<TileKey> {
        if key.family_slot >= TERRAIN_VT_FAMILY_COUNT || key.material_index >= self.material_count {
            return None;
        }
        let mut mip_level = key.mip_level;
        loop {
            let (pages_x, pages_y) = self.pages_at_mip(mip_level);
            let shift = mip_level - key.mip_level;
            let x = (key.x >> shift).min(pages_x.saturating_sub(1));
            let y = (key.y >> shift).min(pages_y.saturating_sub(1));
            let layer_index = self.layer_mip_index(key.family_slot, key.material_index, mip_level);
            let page_index = (y * pages_x + x) as usize;
            if let Some(entry) = self
                .page_tables
                .get(layer_index)
                .and_then(|table| table.get(page_index))
            {
                if entry.is_resident > 0 {
                    return Some(TileKey {
                        family_slot: key.family_slot,
                        material_index: key.material_index,
                        x,
                        y,
                        mip_level,
                    });
                }
            }
            if mip_level + 1 >= self.max_mip_levels {
                return None;
            }
            mip_level += 1;
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
                family_slot: key.family_slot,
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

    fn layer_mip_index(&self, family_slot: u32, material_index: u32, mip_level: u32) -> usize {
        ((family_slot * self.material_count + material_index) * self.max_mip_levels + mip_level)
            as usize
    }

    fn encode_cache_tile(&self, key: TileKey) -> TileId {
        let logical_material = key.family_slot * self.material_count.max(1) + key.material_index;
        TileId {
            x: logical_material * self.pages_x0.max(1) + key.x,
            y: key.y,
            mip_level: key.mip_level,
        }
    }

    fn decode_cache_tile(&self, tile: TileId) -> TileKey {
        let logical_material = tile.x / self.pages_x0.max(1);
        TileKey {
            family_slot: logical_material / self.material_count.max(1),
            material_index: logical_material % self.material_count.max(1),
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
impl Drop for TerrainMaterialVTRuntime {
    fn drop(&mut self) {
        // Release the footprint reported to the 512 MiB resource registry in
        // `TerrainMaterialVTRuntime::new`.
        let memory_tracker = crate::core::memory_tracker::global_tracker();
        let page_table_layers = TERRAIN_VT_FAMILY_COUNT * self.material_count * self.max_mip_levels;
        memory_tracker.free_texture_allocation(
            self.atlas_size,
            self.atlas_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        memory_tracker.free_texture_allocation(
            self.pages_x0,
            self.pages_y0.saturating_mul(page_table_layers),
            wgpu::TextureFormat::Rgba32Float,
        );
        if let Some(feedback) = self.feedback_buffer.as_ref() {
            let feedback_bytes = feedback.buffer().size();
            memory_tracker.free_buffer_allocation(feedback_bytes, false);
            memory_tracker.free_buffer_allocation(feedback_bytes, true);
        }
        memory_tracker.clear_resident_tiles();
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
                &self.device,
                &self.queue,
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

    /// VERITAS: blocking drain of the VT feedback stream resolved to the
    /// resident tiles the last frame actually sampled.
    pub(super) fn read_material_vt_contributing_tiles(
        &self,
    ) -> Result<Vec<crate::core::provenance::ContributingTile>> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .read_contributing_tiles(self.device.as_ref())
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
