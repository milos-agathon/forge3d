// src/render/pbr_pass.rs
// PBR render pass orchestration with shadow and IBL wiring
// Exists to drive PbrPipelineWithShadows inside renderer surfaces
// RELEVANT FILES: src/pipeline/pbr.rs, src/render/params.rs, python/forge3d/config.py, tests/test_renderer_config.py

#![cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]

use crate::core::material::{PbrLighting, PbrMaterial};
use crate::pipeline::pbr::{create_pbr_sampler, PbrPipelineWithShadows, PbrSceneUniforms};
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
        self.pipeline
            .update_scene_uniforms(queue, scene_uniforms);
        self.pipeline
            .update_lighting_uniforms(queue, lighting);
        self.pipeline.ensure_pipeline(device, surface_format);
        self.surface_format = Some(surface_format);
        self.ensure_material_bind_group(device, queue);
    }

    /// Bind the pipeline and all dependent bind groups for a render pass.
    pub fn begin<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut RenderPass<'a>,
    ) {
        let surface_format = self
            .surface_format
            .expect("prepare must be called before begin");
        self.pipeline
            .begin_render(device, surface_format, pass);
    }
}
