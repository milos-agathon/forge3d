use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::Viewer;

impl Viewer {
    pub(super) fn ensure_lit_bind_group(&self, gi: &ScreenSpaceEffectsManager) {
        if self.lit_bind_group_cache.borrow().is_some() {
            return;
        }
        let env_view = self
            .ibl_env_view
            .as_ref()
            .expect("viewer IBL environment view must be initialized");
        let env_sampler = self
            .ibl_sampler
            .as_ref()
            .expect("viewer IBL sampler must be initialized");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.lit.bg.cached"),
            layout: &self.lit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.lit_uniform.as_entire_binding(),
                },
            ],
        });
        *self.lit_bind_group_cache.borrow_mut() = Some(bind_group);
    }

    pub(super) fn ensure_composite_bind_group(
        &self,
        depth_view: &wgpu::TextureView,
        fog_view: &wgpu::TextureView,
        params: &wgpu::Buffer,
        color_view: &wgpu::TextureView,
    ) -> Option<[usize; 5]> {
        let layout = self.comp_bind_group_layout.as_ref()?;
        let key = [
            &self.sky_output_view as *const wgpu::TextureView as usize,
            depth_view as *const wgpu::TextureView as usize,
            fog_view as *const wgpu::TextureView as usize,
            params as *const wgpu::Buffer as usize,
            color_view as *const wgpu::TextureView as usize,
        ];
        if self.comp_bind_group_cache.borrow().contains_key(&key) {
            return Some(key);
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.comp.bg.cached"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(fog_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
            ],
        });
        self.comp_bind_group_cache
            .borrow_mut()
            .insert(key, bind_group);
        Some(key)
    }
}
