impl SsaoResources {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        _color: &wgpu::Texture,
        _normal: &wgpu::Texture,
    ) -> Result<Self, String> {
        let shader = create_ssao_shader(device);
        let layouts = create_ssao_layouts(device);
        let pipelines = create_ssao_pipelines(device, &shader, &layouts);
        let buffers = create_ssao_buffers(device);
        let (ao_texture, ao_view) = create_ssao_texture(device, width, height, "scene-ssao");
        let (blur_texture, blur_view) =
            create_ssao_texture(device, width, height, "scene-ssao-blur");
        let noise = create_ssao_noise_resources(device, queue);
        let depth = create_ssao_depth_resources(device, width, height);

        let resources = Self {
            radius: 1.0,
            intensity: 1.0,
            bias: 0.025,
            width,
            height,
            sampler: buffers.sampler,
            blur_sampler: buffers.blur_sampler,
            settings_buffer: buffers.settings_buffer,
            blur_settings_buffer: buffers.blur_settings_buffer,
            view_buffer: buffers.view_buffer,
            ao_texture,
            ao_view,
            blur_texture,
            blur_view,
            noise_texture: noise.texture,
            noise_view: noise.view,
            noise_sampler: noise.sampler,
            depth_texture: depth.texture,
            depth_view: depth.view,
            ssao_bind_group_layout: layouts.ssao_bind_group_layout,
            ssao_output_bind_group_layout: layouts.ssao_output_bind_group_layout,
            blur_bind_group_layout: layouts.blur_bind_group_layout,
            composite_bind_group_layout: layouts.composite_bind_group_layout,
            ssao_pipeline: pipelines.ssao_pipeline,
            blur_pipeline: pipelines.blur_pipeline,
            composite_pipeline: pipelines.composite_pipeline,
        };
        resources.update_inv_resolution(queue);
        Ok(resources)
    }
}

