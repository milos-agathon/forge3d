impl Scene {
    pub(super) fn set_height_from_r32f_impl(&mut self, height_r32f: &PyAny) -> PyResult<()> {
        let arr: PyReadonlyArray2<f32> = height_r32f.extract()?;
        let (h, w) = (arr.shape()[0] as u32, arr.shape()[1] as u32);
        let data = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("height must be C-contiguous float32[H,W]")
        })?;

        let g = crate::core::gpu::ctx();
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-height-r32f"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row_bytes = w * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let src_bytes: &[u8] = bytemuck::cast_slice::<f32, u8>(data);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize]
                .copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
        }
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&Default::default());
        let samp = g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene-height-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        self.height_view = Some(view);
        self.height_sampler = Some(samp);
        self.bg1_height = self.tp.make_bg_height(
            &g.device,
            self.height_view.as_ref().unwrap(),
            self.height_sampler.as_ref().unwrap(),
        );

        let height_ref = self.height_view.as_ref();
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.recreate_bind_group(&g.device, None, height_ref, None, None);
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }
}

