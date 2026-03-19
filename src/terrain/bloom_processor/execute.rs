use super::uniforms::{BloomBlurUniforms, BloomBrightPassUniforms, BloomCompositeUniforms};
use super::{TerrainBloomConfig, TerrainBloomProcessor};
use anyhow::{anyhow, Result};

impl TerrainBloomProcessor {
    /// Execute bloom pipeline
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        config: &TerrainBloomConfig,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if !config.enabled {
            return Ok(());
        }

        self.ensure_textures(device, width, height);

        let bright_view = self
            .bright_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom bright view not initialized"))?;
        let blur_temp_view = self
            .blur_temp_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom blur temp view not initialized"))?;
        let blur_result_view = self
            .blur_result_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom blur result view not initialized"))?;

        let brightpass_uniforms = BloomBrightPassUniforms {
            threshold: config.threshold,
            softness: config.softness,
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            &self.brightpass_uniform_buffer,
            0,
            bytemuck::bytes_of(&brightpass_uniforms),
        );

        let blur_uniforms = BloomBlurUniforms {
            radius: config.radius,
            strength: 1.0,
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            &self.blur_uniform_buffer,
            0,
            bytemuck::bytes_of(&blur_uniforms),
        );

        let composite_uniforms = BloomCompositeUniforms {
            intensity: config.intensity,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );

        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.brightpass_bind_group"),
                layout: &self.brightpass_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(bright_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.brightpass_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.brightpass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brightpass_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.blur_h_bind_group"),
                layout: &self.blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(bright_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.blur_h"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.blur_v_bind_group"),
                layout: &self.blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(blur_temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_result_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.blur_v"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.composite_bind_group"),
                layout: &self.composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_result_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.composite_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        log::debug!(
            target: "terrain.bloom",
            "M2: Bloom executed: threshold={:.2}, intensity={:.2}, radius={:.1}",
            config.threshold,
            config.intensity,
            config.radius
        );

        Ok(())
    }
}
