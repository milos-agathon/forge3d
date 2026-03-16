fn create_grid_buffers(device: &wgpu::Device, grid: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = grid.max(2) as usize;
    let (w, h) = (n, n);
    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);
    let mut verts = Vec::<f32>::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            let x = -scale + i as f32 * step_x;
            let z = -scale + j as f32 * step_z;
            let u = i as f32 / (w as f32 - 1.0);
            let v = j as f32 / (h as f32 - 1.0);
            verts.extend_from_slice(&[x, z, u, v]);
        }
    }
    let mut idx = Vec::<u32>::with_capacity((w - 1) * (h - 1) * 6);
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let a = (j * w + i) as u32;
            let b = (j * w + i + 1) as u32;
            let c = ((j + 1) * w + i) as u32;
            let d = ((j + 1) * w + i + 1) as u32;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene-xyuv-vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene-xyuv-ibuf"),
        contents: bytemuck::cast_slice(&idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, idx.len() as u32)
}

fn create_dummy_height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let w = 2u32;
    let h = 2u32;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-dummy-height"),
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
    let src_vals: [f32; 4] = [0.00, 0.25, 0.50, 0.75];
    let src_bytes: &[u8] = bytemuck::cast_slice(&src_vals);
    let mut padded = vec![0u8; (padded_bpr * h) as usize];
    for y in 0..h as usize {
        let s = y * row_bytes as usize;
        let d = y * padded_bpr as usize;
        padded[d..d + row_bytes as usize].copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
    }
    queue.write_texture(
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
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("scene-height-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

fn create_dummy_cloud_shadow_bind_group(
    tp: &crate::terrain::pipeline::TerrainPipeline,
    device: &wgpu::Device,
) -> wgpu::BindGroup {
    let dummy_cloud_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene.dummy_cloud_shadow"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let dummy_cloud_view = dummy_cloud_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let dummy_cloud_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("scene.dummy_cloud_sampler"),
        ..Default::default()
    });
    tp.make_bg_cloud_shadows(device, &dummy_cloud_view, &dummy_cloud_sampler)
}

