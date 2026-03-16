use super::*;

impl WavefrontScheduler {
    pub fn aov_pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn copy_aov_depth_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_depth, 0, dst, 0, bytes);
    }

    pub fn copy_aov_albedo_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_albedo, 0, dst, 0, bytes);
    }

    pub fn copy_aov_normal_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_normal, 0, dst, 0, bytes);
    }

    pub fn aov_depth_buffer(&self) -> &Buffer {
        &self.aov_depth
    }

    pub fn aov_normal_buffer(&self) -> &Buffer {
        &self.aov_normal
    }

    pub fn aov_albedo_buffer(&self) -> &Buffer {
        &self.aov_albedo
    }

    pub fn dispatch_ao_from_aovs(
        &self,
        encoder: &mut CommandEncoder,
        samples: u32,
        intensity: f32,
        bias: f32,
        seed: u32,
        ao_out: &Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AOUniforms {
            width: u32,
            height: u32,
            samples: u32,
            intensity: f32,
            bias: f32,
            seed: u32,
            _pad0: u32,
        }
        let u = AOUniforms {
            width: self.width,
            height: self.height,
            samples,
            intensity,
            bias,
            seed,
            _pad0: 0,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ao-uniforms"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao-bind-group"),
            layout: &self.pipelines.ao_bind_group_layout,
            entries: &[
                entry(0, &self.aov_depth),
                entry(1, &self.aov_normal),
                entry(2, ao_out),
                entry(3, &ubuf),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ao-from-aovs-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.ao_compute);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }
}

fn entry<'a>(binding: u32, buffer: &'a Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
