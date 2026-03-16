impl Scene {
    fn fast_softlight_only(&self) -> bool {
        self.width >= 1920
            && self.height >= 1080
            && self.soft_light_radius_enabled
            && !self.point_spot_lights_enabled
            && !self.ibl_enabled
            && !self.clouds_enabled
            && !self.cloud_shadows_enabled
            && !self.reflections_enabled
            && !self.ssao_enabled
            && !self.ssgi_enabled
            && !self.ssr_enabled
            && !self.bloom_enabled
            && !self.dof_enabled
    }

    fn readback_color_pixels(&self, readback_label: &str, copy_label: &str) -> PyResult<Vec<u8>> {
        let g = crate::core::gpu::ctx();
        let bpp = 4u32;
        let unpadded = self.width * bpp;
        let padded = crate::core::gpu::align_copy_bpr(unpadded);
        let size = (padded * self.height) as wgpu::BufferAddress;
        let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(readback_label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(copy_label),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded) as usize;
            let end = start + unpadded as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        readback.unmap();
        Ok(pixels)
    }
}
