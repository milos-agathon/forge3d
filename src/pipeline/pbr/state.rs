use super::*;

impl PbrPipelineWithShadows {
    pub fn set_tone_mapping(&mut self, config: ToneMappingConfig) {
        self.tone_mapping = config;
    }

    pub fn update_scene_uniforms(&mut self, queue: &Queue, uniforms: &PbrSceneUniforms) {
        self.scene_uniforms = *uniforms;
        queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.scene_uniforms),
        );
    }

    pub fn update_scene_from_matrices(
        &mut self,
        queue: &Queue,
        model: Mat4,
        view: Mat4,
        projection: Mat4,
    ) {
        let uniforms = PbrSceneUniforms::from_matrices(model, view, projection);
        self.update_scene_uniforms(queue, &uniforms);
    }

    pub fn update_lighting_uniforms(&mut self, queue: &Queue, lighting: &PbrLighting) {
        self.lighting_uniforms = *lighting;
        queue.write_buffer(
            &self.lighting_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );
    }

    pub fn advance_light_frame(&mut self, _device: &Device) {
        // Advance to next triple-buffered index
        self.light_buffer.next_frame();

        // Invalidate bind group to force recreation with new buffers
        self.globals_bind_group = None;
    }

    pub fn update_lights(
        &mut self,
        device: &Device,
        queue: &Queue,
        lights: &[crate::lighting::types::Light],
    ) -> Result<(), String> {
        self.light_buffer.update(device, queue, lights)?;

        // Invalidate bind group to pick up new light data
        self.globals_bind_group = None;

        Ok(())
    }

    pub fn light_count(&self) -> usize {
        self.light_buffer.last_uploaded_lights().len()
    }

    pub fn light_debug_info(&self) -> String {
        self.light_buffer.debug_info()
    }

    pub fn ensure_material_bind_group(
        &mut self,
        device: &Device,
        queue: &Queue,
        sampler: &Sampler,
    ) {
        if self.material.bind_group.is_none() {
            self.material.create_bind_group(
                device,
                queue,
                &self.material_bind_group_layout,
                sampler,
            );
        }
    }

    pub fn globals_layout(&self) -> &BindGroupLayout {
        &self.globals_bind_group_layout
    }

    pub fn material_layout(&self) -> &BindGroupLayout {
        &self.material_bind_group_layout
    }

    pub fn ibl_layout(&self) -> &BindGroupLayout {
        &self.ibl_bind_group_layout
    }

    pub fn set_brdf_index(&mut self, queue: &Queue, brdf_index: u32) {
        self.shading_uniforms.brdf = brdf_index;
        queue.write_buffer(
            &self.shading_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.shading_uniforms),
        );
    }

    pub fn update_shading_uniforms(&mut self, queue: &Queue, shading: &MaterialShading) {
        self.shading_uniforms = *shading;
        queue.write_buffer(
            &self.shading_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.shading_uniforms),
        );
    }
}
