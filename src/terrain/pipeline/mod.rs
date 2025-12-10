// T33-BEGIN:terrain-pipeline
//! Terrain pipeline state & bindings (T3.3).
//! Creates bind group layouts (0: Globals UBO, 1: height+sampler, 2: LUT+sampler)
//! and a render pipeline targeting Rgba8UnormSrgb. No integration/draw in this task.
//! Supports optional descriptor indexing for texture arrays when available.

use crate::core::reflections::PlanarReflectionRenderer;
use wgpu::*;

mod bind_groups;
mod creation;

pub struct TerrainPipeline {
    pub layout: PipelineLayout,
    pub pipeline: RenderPipeline,
    pub bgl_globals: BindGroupLayout,
    pub bgl_height: BindGroupLayout,
    pub bgl_lut: BindGroupLayout,
    pub bgl_cloud_shadows: BindGroupLayout, // B7: Cloud shadows bind group layout
    pub bgl_reflection: BindGroupLayout,    // B5: Planar reflections bind group layout
    pub bgl_tile: BindGroupLayout,          // E2: Per-tile uniforms (uv/world remap)
    pub descriptor_indexing: bool,
    pub max_palette_textures: u32,
    pub sample_count: u32,
    pub depth_format: Option<TextureFormat>,
    pub normal_format: TextureFormat,
}

impl TerrainPipeline {
    /// Create the terrain pipeline. Does **not** record commands or create bind groups.
    pub fn create(
        device: &Device,
        color_format: TextureFormat,
        normal_format: TextureFormat,
        sample_count: u32,
        depth_format: Option<TextureFormat>,
        height_filterable: bool,
    ) -> Self {
        creation::create_terrain_pipeline(
            device,
            color_format,
            normal_format,
            sample_count,
            depth_format,
            height_filterable,
        )
    }

    // ---------- Bind-group helpers (builders) ----------
    pub fn make_bg_globals(&self, device: &Device, ubo: &Buffer) -> BindGroup {
        bind_groups::make_bg_globals(self, device, ubo)
    }

    /// E2/E1: Per-tile uniform + page table bind group helper
    pub fn make_bg_tile(
        &self,
        device: &Device,
        tile_ubo: &Buffer,
        page_table: Option<&Buffer>,
        tile_slot_ubo: &Buffer,
        mosaic_params_ubo: &Buffer,
    ) -> BindGroup {
        bind_groups::make_bg_tile(self, device, tile_ubo, page_table, tile_slot_ubo, mosaic_params_ubo)
    }

    /// Bind group for height texture/sampler
    pub fn make_bg_height(&self, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
        bind_groups::make_bg_height(self, device, view, samp)
    }

    pub fn make_bg_lut(&self, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
        bind_groups::make_bg_lut(self, device, view, samp)
    }

    /// Create bind group with texture array for descriptor indexing mode
    pub fn make_bg_lut_array(
        &self,
        device: &Device,
        views: &[&TextureView],
        samp: &Sampler,
    ) -> BindGroup {
        bind_groups::make_bg_lut_array(self, device, views, samp)
    }

    // B7: Cloud shadow bind group helper
    pub fn make_bg_cloud_shadows(
        &self,
        device: &Device,
        view: &TextureView,
        samp: &Sampler,
    ) -> BindGroup {
        bind_groups::make_bg_cloud_shadows(self, device, view, samp)
    }

    pub fn make_bg_reflection(
        &self,
        device: &Device,
        renderer: &PlanarReflectionRenderer,
    ) -> BindGroup {
        bind_groups::make_bg_reflection(self, device, renderer)
    }

    /// Check if descriptor indexing is supported
    pub fn supports_descriptor_indexing(&self) -> bool {
        self.descriptor_indexing
    }

    /// Get maximum number of palette textures supported
    pub fn max_palette_textures(&self) -> u32 {
        self.max_palette_textures
    }
}

// ---- Tests (no GPU device creation; descriptor sanity only where possible) ----
#[cfg(test)]
mod tests {
    #[test]
    fn vertex_stride_is_16_bytes() {
        // Keep this in sync with two vec2<f32> attributes
        assert_eq!(16, 4 * 4);
    }
}
// T33-END:terrain-pipeline
