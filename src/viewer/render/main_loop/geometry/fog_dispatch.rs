use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::{FogUpsampleParamsStd140, Viewer};

impl Viewer {
    pub(super) fn dispatch_raymarch_fog(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gi: &mut ScreenSpaceEffectsManager,
        bg0: &wgpu::BindGroup,
        bg1: &wgpu::BindGroup,
        bg2: &wgpu::BindGroup,
    ) {
        if self.fog_half_res_enabled {
            self.ensure_half_res_fog_bind_groups(gi);
            let bg2_half = self.fog_bg2_half_cache.borrow();
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
                cpass.set_bind_group(2, bg2_half.as_ref().unwrap(), &[]);
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

            let params = FogUpsampleParamsStd140 {
                sigma: self.fog_upsigma.max(0.0),
                use_bilateral: if self.fog_bilateral { 1 } else { 0 },
                _pad: [0.0; 2],
            };
            self.queue
                .write_buffer(&self.fog_upsample_params, 0, bytemuck::bytes_of(&params));
            let up_bg = self.fog_upsample_bg_cache.borrow();
            let ugx = (self.config.width + 7) / 8;
            let ugy = (self.config.height + 7) / 8;
            let mut up_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.fog.upsample"),
                timestamp_writes: None,
            });
            up_pass.set_pipeline(&self.fog_upsample_pipeline);
            up_pass.set_bind_group(0, up_bg.as_ref().unwrap(), &[]);
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
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg0: &wgpu::BindGroup,
        bg1: &wgpu::BindGroup,
        bg2: &wgpu::BindGroup,
    ) {
        self.ensure_froxel_bind_group();
        let bg3 = self.fog_bg3_cache.borrow();
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
            pass.set_bind_group(3, bg3.as_ref().unwrap(), &[]);
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
            pass.set_bind_group(3, bg3.as_ref().unwrap(), &[]);
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
