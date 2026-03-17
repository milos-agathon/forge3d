use super::*;

impl HybridPathTracer {
    pub fn render(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
    ) -> Result<Vec<u8>, RenderError> {
        let device = &ctx().device;
        let queue = &ctx().queue;

        let hybrid_uniforms = HybridUniforms {
            sdf_primitive_count: hybrid_scene.sdf_scene.primitive_count() as u32,
            sdf_node_count: hybrid_scene.sdf_scene.node_count() as u32,
            mesh_vertex_count: hybrid_scene.vertices.len() as u32,
            mesh_index_count: hybrid_scene.indices.len() as u32,
            mesh_bvh_node_count: match &hybrid_scene.bvh {
                Some(bvh) => match &bvh.backend {
                    crate::accel::BvhBackend::Cpu(cpu_data) => cpu_data.nodes.len() as u32,
                    crate::accel::BvhBackend::Gpu(gpu_data) => gpu_data.node_count,
                },
                None => 0,
            },
            traversal_mode: params.traversal_mode as u32,
            _pad: [0; 2],
        };

        let base_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-base-ubo"),
            contents: bytemuck::bytes_of(&params.base_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let hybrid_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-hybrid-ubo"),
            contents: bytemuck::bytes_of(&hybrid_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lighting_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-lighting-ubo"),
            contents: bytemuck::bytes_of(&params.lighting_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_bytes = if spheres.is_empty() {
            std::borrow::Cow::Owned(vec![0u8; std::mem::size_of::<Sphere>()])
        } else {
            std::borrow::Cow::Borrowed(bytemuck::cast_slice(spheres))
        };
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-scene"),
            contents: scene_bytes.as_ref(),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let accum_size = (width as u64) * (height as u64) * 16;
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-accum"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hybrid-pt-out-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        let aov_frames = AovFrames::new(device, width, height, &aovs_all);
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg0"),
            layout: &self.layouts.uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: base_ubo.as_entire_binding(),
            }],
        });

        let mut bg1_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hybrid_ubo.as_entire_binding(),
            },
        ];
        bg1_entries.extend(
            hybrid_scene
                .get_mesh_bind_entries()
                .into_iter()
                .enumerate()
                .map(|(i, mut entry)| {
                    entry.binding = (i + 2) as u32;
                    entry
                }),
        );
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg1"),
            layout: &self.layouts.scene,
            entries: &bg1_entries,
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg2"),
            layout: &self.layouts.accum,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: accum_buf.as_entire_binding(),
            }],
        });

        let mut bg3_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }];
        for (i, view) in aov_views.iter().enumerate() {
            bg3_entries.push(wgpu::BindGroupEntry {
                binding: (i as u32) + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg3"),
            layout: &self.layouts.output,
            entries: &bg3_entries,
        });
        let bg4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg4"),
            layout: &self.layouts.lighting,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lighting_ubo.as_entire_binding(),
            }],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hybrid-pt-encoder"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hybrid-pt-cpass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);
            cpass.set_bind_group(4, &bg4, &[]);
            cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }

        let row_bytes = width * 8;
        let padded_bpr = align_copy_bpr(row_bytes);
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-read"),
            size: (padded_bpr as u64) * (height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit([enc.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| RenderError::Readback("map_async channel closed".into()))?
            .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;

        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        let src_stride = padded_bpr as usize;
        let dst_stride = (width as usize) * 4;

        for y in 0..(height as usize) {
            let row = &data[y * src_stride..y * src_stride + (width as usize) * 8];
            for x in 0..(width as usize) {
                let o = x * 8;
                let r = f16::from_bits(u16::from_le_bytes([row[o], row[o + 1]])).to_f32();
                let g = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();

                let ix = y * dst_stride + x * 4;
                out[ix] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 3] = 255;
            }
        }

        drop(data);
        read_buf.unmap();
        Ok(out)
    }
}
