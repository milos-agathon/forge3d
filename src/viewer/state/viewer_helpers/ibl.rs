use crate::core::ibl::{IBLQuality, IBLRenderer};
use crate::viewer::Viewer;
use anyhow::anyhow;

impl Viewer {
    pub(crate) fn load_ibl(&mut self, path: &str) -> anyhow::Result<()> {
        let hdr_img = crate::formats::hdr::load_hdr(path)
            .map_err(|e| anyhow!("failed to load HDR '{}': {}", path, e))?;

        let mut ibl = IBLRenderer::new(&self.device, IBLQuality::Low);

        if let Some(res) = self.ibl_base_resolution {
            ibl.set_base_resolution(res);
        } else {
            ibl.set_base_resolution(IBLQuality::Low.base_environment_size());
        }

        if let Some(ref cache_dir) = self.ibl_cache_dir {
            ibl.configure_cache(cache_dir, std::path::Path::new(path))
                .map_err(|e| anyhow!("failed to configure IBL cache: {}", e))?;
        }

        ibl.load_environment_map(
            &self.device,
            &self.queue,
            &hdr_img.data,
            hdr_img.width,
            hdr_img.height,
        )
        .map_err(|e| anyhow!("failed to upload environment: {}", e))?;

        ibl.initialize(&self.device, &self.queue)
            .map_err(|e| anyhow!("failed to initialize IBL: {}", e))?;

        let (irr_tex_opt, spec_tex_opt, _) = ibl.textures();
        if let Some(ref mut gi) = self.gi {
            if let Some(irr_tex) = irr_tex_opt {
                gi.set_ssgi_env(&self.device, irr_tex);
            }
            if let Some(spec_tex) = spec_tex_opt {
                gi.set_ssr_env(&self.device, spec_tex);
                let cube_view = spec_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("viewer.ibl.specular.cube.view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                });
                let env_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.ibl.env.sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                self.ibl_env_view = Some(cube_view);
                self.ibl_sampler = Some(env_sampler);
            }
        }

        self.ibl_renderer = Some(ibl);
        self.ibl_hdr_path = Some(path.to_string());
        Ok(())
    }
}
