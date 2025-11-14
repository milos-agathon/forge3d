// src/render/pbr_pass.rs
// PBR render pass orchestration with shadow and IBL wiring
// Exists to drive PbrPipelineWithShadows inside renderer surfaces
// RELEVANT FILES: src/pipeline/pbr.rs, src/render/params.rs, python/forge3d/config.py, tests/test_renderer_config.py

#![cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]

use crate::core::material::{PbrLighting, PbrMaterial};
use crate::pipeline::pbr::{create_pbr_sampler, PbrPipelineWithShadows, PbrSceneUniforms};
use crate::render::params::{BrdfModel as CfgBrdfModel, RendererConfig};
use wgpu::{Device, Queue, RenderPass, Sampler, TextureFormat};

/// Helper that owns a PBR pipeline and records bind groups for render passes.
pub struct PbrRenderPass {
    pipeline: PbrPipelineWithShadows,
    material_sampler: Sampler,
    surface_format: Option<TextureFormat>,
}

impl PbrRenderPass {
    /// Construct a render pass wrapper with default shadow/IBL resources.
    pub fn new(
        device: &Device,
        queue: &Queue,
        material: PbrMaterial,
        enable_shadows: bool,
    ) -> Self {
        let mut pipeline = PbrPipelineWithShadows::new(device, queue, material, enable_shadows);
        let material_sampler = create_pbr_sampler(device);
        pipeline.ensure_material_bind_group(device, queue, &material_sampler);

        Self {
            pipeline,
            material_sampler,
            surface_format: None,
        }
    }

    /// Access the underlying pipeline.
    pub fn pipeline(&self) -> &PbrPipelineWithShadows {
        &self.pipeline
    }

    /// Mutable access to the underlying pipeline.
    pub fn pipeline_mut(&mut self) -> &mut PbrPipelineWithShadows {
        &mut self.pipeline
    }

    /// Mutable access to the material for configuration.
    pub fn material_mut(&mut self) -> &mut PbrMaterial {
        &mut self.pipeline.material.material
    }

    /// Upload the current material values to the GPU.
    pub fn sync_material(&self, queue: &Queue) {
        self.pipeline.material.update_uniforms(queue);
    }

    fn ensure_material_bind_group(&mut self, device: &Device, queue: &Queue) {
        self.pipeline
            .ensure_material_bind_group(device, queue, &self.material_sampler);
    }

    /// Prepare per-frame uniforms and ensure the pipeline is ready for the target format.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        surface_format: TextureFormat,
        scene_uniforms: &PbrSceneUniforms,
        lighting: &PbrLighting,
    ) {
        self.pipeline.update_scene_uniforms(queue, scene_uniforms);
        self.pipeline.update_lighting_uniforms(queue, lighting);
        self.pipeline.ensure_pipeline(device, surface_format);
        self.surface_format = Some(surface_format);
        self.ensure_material_bind_group(device, queue);
    }

    /// Bind the pipeline and all dependent bind groups for a render pass.
    pub fn begin<'a>(&'a mut self, device: &Device, pass: &mut RenderPass<'a>) {
        let surface_format = self
            .surface_format
            .expect("prepare must be called before begin");
        self.pipeline.begin_render(device, surface_format, pass);
    }

    /// Apply renderer configuration to the PBR pipeline (M2-03)
    ///
    /// Selects the BRDF model from either `brdf_override` (if present) or
    /// from `shading.brdf` in the provided `RendererConfig` and uploads it
    /// to the GPU via `ShadingParamsGpu`.
    pub fn apply_renderer_config(&mut self, queue: &Queue, cfg: &RendererConfig) {
        let model = cfg.brdf_override.unwrap_or(cfg.shading.brdf);
        self.set_brdf_model(queue, model);
    }

    /// Set BRDF model using config enum and upload to GPU (M2-03, P2-04)
    pub fn set_brdf_model(&mut self, queue: &Queue, model: CfgBrdfModel) {
        // Map to shader indices matching src/shaders/lighting.wgsl constants
        let idx = match model {
            CfgBrdfModel::Lambert => 0u32,              // BRDF_LAMBERT
            CfgBrdfModel::Phong => 1u32,                // BRDF_PHONG
            CfgBrdfModel::BlinnPhong => 2u32,           // BRDF_BLINN_PHONG
            CfgBrdfModel::OrenNayar => 3u32,            // BRDF_OREN_NAYAR (P2-04)
            CfgBrdfModel::CookTorranceGGX => 4u32,      // BRDF_COOK_TORRANCE_GGX
            CfgBrdfModel::CookTorranceBeckmann => 5u32, // BRDF_COOK_TORRANCE_BECKMANN
            CfgBrdfModel::DisneyPrincipled => 6u32,     // BRDF_DISNEY_PRINCIPLED
            CfgBrdfModel::AshikhminShirley => 7u32,     // BRDF_ASHIKHMIN_SHIRLEY (P2-04)
            CfgBrdfModel::Ward => 8u32,                 // BRDF_WARD (P2-04)
            CfgBrdfModel::Toon => 9u32,                 // BRDF_TOON (P2-04)
            CfgBrdfModel::Minnaert => 10u32,            // BRDF_MINNAERT (P2-04)
            CfgBrdfModel::Subsurface => 11u32,          // BRDF_SUBSURFACE
            CfgBrdfModel::Hair => 12u32,                // BRDF_HAIR
        };
        self.pipeline.set_brdf_index(queue, idx);
    }
}
