use super::*;

impl WavefrontScheduler {
    pub fn set_restir_enabled(&mut self, enabled: bool) {
        self.restir_enabled = enabled;
    }

    pub fn set_restir_spatial_enabled(&mut self, enabled: bool) {
        self.restir_spatial_enabled = enabled;
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.width = width;
        self.height = height;
        let queue_capacity = width * height * QUEUE_CAPACITY_SCALE;
        self.queue_buffers = QueueBuffers::new(&self.device, queue_capacity)?;
        self.restir_reservoirs =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_light_samples = empty_light_samples_buffer(&self.device);
        self.restir_alias_entries = empty_alias_entries_buffer(&self.device);
        self.restir_light_probs =
            crate::path_tracing::restir::empty_light_probs_buffer(&self.device);
        self.restir_prev =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_out =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_diag_flags =
            create_diag_flags_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_debug_aov =
            create_debug_aov_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_gbuffer =
            create_restir_gbuffer(&self.device, (width as usize) * (height as usize));
        self.restir_gbuffer_pos =
            create_restir_gbuffer_pos(&self.device, (width as usize) * (height as usize));
        let mat_zero: Vec<u32> = vec![0u32; (width as usize) * (height as usize)];
        self.restir_gbuffer_mat =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("restir-gbuffer-mat"),
                    contents: bytemuck::cast_slice(&mat_zero),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
        let px_count = (width as usize) * (height as usize);
        let aov_bytes: u64 = (px_count * std::mem::size_of::<[f32; 4]>()) as u64;
        self.aov_albedo = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-albedo"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.aov_depth = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-depth"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.aov_normal = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-normal"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let medium_init: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        self.medium_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("medium-params-uniform"),
                contents: bytemuck::cast_slice(&medium_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        Ok(())
    }

    pub fn set_restir_light_data(&mut self, light_samples: Buffer, alias_entries: Buffer) {
        self.restir_light_samples = light_samples;
        self.restir_alias_entries = alias_entries;
    }

    pub fn set_restir_light_probs(&mut self, light_probs: Buffer) {
        self.restir_light_probs = light_probs;
    }

    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    pub fn reset_frame_index(&mut self) {
        self.frame_index = 0;
    }

    pub fn set_medium_params(&self, g: f32, sigma_t: f32, density: f32, enable: bool) {
        let val: [f32; 4] = [g, sigma_t, density, if enable { 1.0 } else { 0.0 }];
        self.queue
            .write_buffer(&self.medium_params, 0, bytemuck::cast_slice(&val));
    }

    pub fn set_hair_segments_buffer(&mut self, buffer: Buffer) {
        self.hair_segments = buffer;
    }

    pub fn set_restir_debug_aov_mode(&self, enabled: bool) {
        let val: [u32; 4] = [if enabled { 1 } else { 0 }, 0, 0, 0];
        self.queue
            .write_buffer(&self.restir_settings, 0, bytemuck::cast_slice(&val));
    }

    pub fn set_qmc_mode(&self, mode: u32) {
        self.queue
            .write_buffer(&self.restir_settings, 4, bytemuck::cast_slice(&[mode]));
    }

    pub fn set_adaptive_rr_threshold(&self, threshold: f32) {
        self.queue.write_buffer(
            &self.restir_settings,
            8,
            bytemuck::cast_slice(&[threshold.to_bits()]),
        );
    }

    pub fn set_adaptive_spp_limit(&self, limit: u32) {
        self.queue
            .write_buffer(&self.restir_settings, 8, bytemuck::cast_slice(&[limit]));
    }

    pub fn set_instances_buffer(&mut self, buffer: Buffer) {
        self.instances_buffer = buffer;
    }

    pub fn set_blas_descs_buffer(&mut self, buffer: Buffer) {
        self.blas_descs = buffer;
    }
}
