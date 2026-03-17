use super::*;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindingResource, IndexFormat};

impl CloudRenderer {
    pub fn ensure_resources(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        let desired_resolution = self.desired_noise_resolution();
        if self.noise_view.is_none() || self.noise_resolution != desired_resolution {
            self.recreate_noise_texture(device, queue)?;
        }
        if self.shape_view.is_none() {
            self.recreate_shape_texture(device, queue)?;
        }
        if self.ibl_irradiance_view.is_none() || self.ibl_prefilter_view.is_none() {
            self.recreate_default_ibl(device, queue)?;
        }

        if self.bind_group_textures.is_none() {
            let noise_view = self
                .noise_view
                .as_ref()
                .ok_or_else(|| "Noise texture missing".to_string())?;
            let shape_view = self
                .shape_view
                .as_ref()
                .ok_or_else(|| "Shape texture missing".to_string())?;

            self.bind_group_textures = Some(device.create_bind_group(&BindGroupDescriptor {
                label: Some("cloud_bind_group_textures"),
                layout: &self.bind_group_layout_textures,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(noise_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.cloud_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(shape_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.shape_sampler),
                    },
                ],
            }));
        }

        if self.bind_group_ibl.is_none() {
            let irradiance_view = self
                .ibl_irradiance_view
                .as_ref()
                .ok_or_else(|| "IBL irradiance view missing".to_string())?;
            let prefilter_view = self
                .ibl_prefilter_view
                .as_ref()
                .ok_or_else(|| "IBL prefilter view missing".to_string())?;

            self.bind_group_ibl = Some(device.create_bind_group(&BindGroupDescriptor {
                label: Some("cloud_bind_group_ibl"),
                layout: &self.bind_group_layout_ibl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(irradiance_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.ibl_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(prefilter_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.ibl_sampler),
                    },
                ],
            }));
        }

        Ok(())
    }

    pub fn prepare_frame(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        self.ensure_resources(device, queue)
    }

    pub(super) fn desired_noise_resolution(&self) -> u32 {
        self.params.quality.noise_resolution().min(128)
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if !self.enabled {
            return;
        }
        let Some(ref textures_bg) = self.bind_group_textures else {
            return;
        };
        let Some(ref ibl_bg) = self.bind_group_ibl else {
            return;
        };

        pass.set_pipeline(&self.cloud_pipeline);
        pass.set_bind_group(0, &self.bind_group_uniforms, &[]);
        pass.set_bind_group(1, textures_bg, &[]);
        pass.set_bind_group(2, ibl_bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
