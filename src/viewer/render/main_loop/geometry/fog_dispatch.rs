use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::{FogUpsampleParamsStd140, Viewer};

impl Viewer {
    pub(super) fn dispatch_raymarch_fog(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gi: &mut ScreenSpaceEffectsManager,
        bg0: &wgpu::BindGroup,
        bg1: &wgpu::BindGroup,
        bg2: &wgpu::BindGroup,
    ) {
        if self.fog_half_res_enabled {
            let bg2_half = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.fog.bg2.half"),
                layout: &self.fog_bgl2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.fog_output_half_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.fog_history_half_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                    },
                ],
            });
            let gx = ((self.config.width / 2) + 7) / 8;
            let gy = ((self.config.height / 2) + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.fog.raymarch.half"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.fog_pipeline);
                cpass.set_bind_group(0, bg0, &[]);
                cpass.set_bind_group(1, bg1, &[]);
                cpass.set_bind_group(2, &bg2_half, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.fog_output_half,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.fog_history_half,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.config.width.max(1) / 2,
                    height: self.config.height.max(1) / 2,
                    depth_or_array_layers: 1,
                },
            );

            let upsampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("viewer.fog.upsampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            let params = FogUpsampleParamsStd140 {
                sigma: self.fog_upsigma.max(0.0),
                use_bilateral: if self.fog_bilateral { 1 } else { 0 },
                _pad: [0.0; 2],
            };
            self.queue
                .write_buffer(&self.fog_upsample_params, 0, bytemuck::bytes_of(&params));
            let up_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.fog.upsample.bg"),
                layout: &self.fog_upsample_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.fog_output_half_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&upsampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.fog_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.fog_depth_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.fog_upsample_params.as_entire_binding(),
                    },
                ],
            });
            let ugx = (self.config.width + 7) / 8;
            let ugy = (self.config.height + 7) / 8;
            let mut up_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.fog.upsample"),
                timestamp_writes: None,
            });
            up_pass.set_pipeline(&self.fog_upsample_pipeline);
            up_pass.set_bind_group(0, &up_bg, &[]);
            up_pass.dispatch_workgroups(ugx, ugy, 1);
        } else {
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.fog.raymarch"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.fog_pipeline);
                cpass.set_bind_group(0, bg0, &[]);
                cpass.set_bind_group(1, bg1, &[]);
                cpass.set_bind_group(2, bg2, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.fog_output,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.fog_history,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub(super) fn dispatch_froxel_fog(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bg0: &wgpu::BindGroup,
        bg1: &wgpu::BindGroup,
        bg2: &wgpu::BindGroup,
    ) {
        let bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.fog.bg3"),
            layout: &self.fog_bgl3,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.froxel_sampler),
                },
            ],
        });
        let gx3d = (16u32 + 3) / 4;
        let gy3d = (8u32 + 3) / 4;
        let gz3d = (64u32 + 3) / 4;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.fog.froxel.build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.froxel_build_pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.set_bind_group(1, bg1, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(gx3d, gy3d, gz3d);
        }

        let gx2d = (self.config.width + 7) / 8;
        let gy2d = (self.config.height + 7) / 8;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.fog.froxel.apply"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.froxel_apply_pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.set_bind_group(2, bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(gx2d, gy2d, 1);
        }

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fog_output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.fog_history,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
    }
}
