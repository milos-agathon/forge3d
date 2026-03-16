use super::*;

impl SsaoRenderer {
    pub fn encode_temporal(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;

        if self.temporal_enabled {
            if !self.history_valid {
                let src_tex = if self.blur_enabled {
                    &self.ssao_blurred
                } else {
                    &self.ssao_texture
                };
                copy_ao_texture(
                    encoder,
                    src_tex,
                    &self.ssao_resolved,
                    self.width,
                    self.height,
                );
                copy_ao_texture(
                    encoder,
                    &self.ssao_resolved,
                    &self.ssao_history,
                    self.width,
                    self.height,
                );
                self.history_valid = true;
                self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
                return Ok(());
            }

            let input_view = if self.blur_enabled {
                &self.ssao_blurred_view
            } else {
                &self.ssao_view
            };
            let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
                label: Some("ssao_temporal_bg"),
                layout: &self.temporal_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.ssao_history_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&self.ssao_resolved_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.temporal_params_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut tpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ssao_temporal"),
                timestamp_writes: None,
            });
            tpass.set_pipeline(&self.temporal_pipeline);
            tpass.set_bind_group(0, &temporal_bg, &[]);
            tpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            drop(tpass);

            copy_ao_texture(
                encoder,
                &self.ssao_resolved,
                &self.ssao_history,
                self.width,
                self.height,
            );
            self.history_valid = true;
            self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
        } else {
            let src_tex = if self.blur_enabled {
                &self.ssao_blurred
            } else {
                &self.ssao_texture
            };
            copy_ao_texture(
                encoder,
                src_tex,
                &self.ssao_resolved,
                self.width,
                self.height,
            );
            self.history_valid = false;
            self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }

        Ok(())
    }
}

fn copy_ao_texture(
    encoder: &mut CommandEncoder,
    src: &Texture,
    dst: &Texture,
    width: u32,
    height: u32,
) {
    encoder.copy_texture_to_texture(
        ImageCopyTexture {
            texture: src,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        ImageCopyTexture {
            texture: dst,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
