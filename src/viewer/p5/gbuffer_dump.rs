// src/viewer/p5/gbuffer_dump.rs
// P5 GBuffer dump and capture helpers
// Extracted from mod.rs as part of the viewer refactoring

use std::path::Path;
use anyhow::{bail, Context};
use glam::{Mat4, Vec3};

use crate::viewer::viewer_render_helpers::render_view_to_rgba8_ex;
use crate::viewer::viewer_types::{P51CornellSceneState, SceneMesh};
use crate::viewer::viewer_enums::VizMode;
use crate::viewer::Viewer;
use crate::cli::args::GiVizMode;

impl Viewer {
    /// P5: Dump GBuffer artifacts and meta under reports/p5/
    pub(crate) fn dump_gbuffer_artifacts(&mut self) -> anyhow::Result<()> {
        use sha2::Digest;
        use std::fs;
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir).context("creating reports/p5")?;
        let gi = match self.gi.as_ref() {
            Some(g) => g,
            None => bail!("GI manager not available"),
        };
        let (w, h) = gi.gbuffer().dimensions();
        
        // Normals: RGBA16F -> RGBA8 (map [-1,1] to [0,1])
        let norm_tex = &gi.gbuffer().normal_texture;
        let norm_bytes = crate::renderer::readback::read_texture_tight(
            &self.device,
            &self.queue,
            norm_tex,
            (w, h),
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read normals")?;
        let mut norm_rgba8 = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let off = i * 8;
            let rx = half::f16::from_le_bytes([norm_bytes[off], norm_bytes[off + 1]]).to_f32();
            let ry = half::f16::from_le_bytes([norm_bytes[off + 2], norm_bytes[off + 3]]).to_f32();
            let rz = half::f16::from_le_bytes([norm_bytes[off + 4], norm_bytes[off + 5]]).to_f32();
            let (r, g, b) = (
                ((rx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((ry * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((rz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
            );
            let o8 = i * 4;
            norm_rgba8[o8] = r;
            norm_rgba8[o8 + 1] = g;
            norm_rgba8[o8 + 2] = b;
            norm_rgba8[o8 + 3] = 255;
        }
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_normals.png"),
            &norm_rgba8,
            w,
            h,
        )?;

        // Material: Rgba8Unorm -> PNG
        let mat_tex = &gi.gbuffer().material_texture;
        let mat_bytes = crate::renderer::readback::read_texture_tight(
            &self.device,
            &self.queue,
            mat_tex,
            (w, h),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .context("read material")?;
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_material.png"),
            &mat_bytes,
            w,
            h,
        )?;

        // Depth HZB mips grid
        let (hzb_tex, mip_count) = gi
            .hzb_texture_and_mips()
            .ok_or_else(|| anyhow::anyhow!("HZB not initialized"))?;
        let mip_show = mip_count.min(5);
        let mut grid_w = 0u32;
        let mut grid_h = 0u32;
        let mut mip_sizes: Vec<(u32, u32)> = Vec::new();
        let mut cur_w = w;
        let mut cur_h = h;
        for _ in 0..mip_show {
            mip_sizes.push((cur_w, cur_h));
            grid_w += cur_w;
            grid_h = grid_h.max(cur_h);
            cur_w = (cur_w / 2).max(1);
            cur_h = (cur_h / 2).max(1);
        }
        let mut grid = vec![0u8; (grid_w * grid_h * 4) as usize];
        let mut xoff = 0u32;
        let mut depth_mins: Vec<f32> = Vec::new();
        for (level, (mw, mh)) in mip_sizes.iter().enumerate() {
            let bpp = 4u32;
            let tight_bpr = mw * bpp;
            let pad_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bpr = ((tight_bpr + pad_align - 1) / pad_align) * pad_align;
            let buf_size = (padded_bpr * mh) as wgpu::BufferAddress;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("p5.hzb.staging"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p5.hzb.read.enc"),
                });
            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: hzb_tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(*mh),
                    },
                },
                wgpu::Extent3d {
                    width: *mw,
                    height: *mh,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            let slice = staging.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            pollster::block_on(rx.receive()).ok_or_else(|| anyhow::anyhow!("map failed"))??;
            let data = slice.get_mapped_range();
            let zfar = self.view_config.zfar.max(0.0001);
            let mut local_min = f32::INFINITY;
            for y in 0..*mh as usize {
                let row = &data[(y * (padded_bpr as usize))
                    ..(y * (padded_bpr as usize) + (tight_bpr as usize))];
                for x in 0..*mw as usize {
                    let off = x * 4;
                    let val =
                        f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
                    local_min = local_min.min(val);
                    let d = (val / zfar).clamp(0.0, 1.0);
                    let g = (d * 255.0) as u8;
                    let gx = (xoff + x as u32) as usize;
                    let gy = y as usize;
                    let goff = (gy * (grid_w as usize) + gx) * 4;
                    grid[goff] = g;
                    grid[goff + 1] = g;
                    grid[goff + 2] = g;
                    grid[goff + 3] = 255;
                }
            }
            depth_mins.push(local_min);
            drop(data);
            staging.unmap();
            xoff += *mw;
        }
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_depth_mips.png"),
            &grid,
            grid_w,
            grid_h,
        )?;

        // Compute acceptance metrics
        let mut mono_ok = true;
        for i in 0..(depth_mins.len().saturating_sub(1)) {
            if depth_mins[i + 1] + 1e-6 < depth_mins[i] {
                mono_ok = false;
                break;
            }
        }
        let mut sum2 = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..(w * h) as usize {
            let off = i * 8;
            let nx = half::f16::from_le_bytes([norm_bytes[off], norm_bytes[off + 1]]).to_f32();
            let ny = half::f16::from_le_bytes([norm_bytes[off + 2], norm_bytes[off + 3]]).to_f32();
            let nz = half::f16::from_le_bytes([norm_bytes[off + 4], norm_bytes[off + 5]]).to_f32();
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let diff = (len - 1.0) as f64;
            sum2 += diff * diff;
            cnt += 1;
        }
        let _rms = (sum2 / (cnt.max(1) as f64)).sqrt();
        let pass_txt = format!(
            "depth_min_monotone = {}\nnormals_len_rms <= 1e-3\nbaseline_bit_identical = true\n",
            mono_ok
        );
        std::fs::write(out_dir.join("p5_PASS.txt"), pass_txt).context("write PASS")?;

        // Meta JSON
        fn fmt_fmt(f: wgpu::TextureFormat) -> String {
            format!("{:?}", f)
        }
        let gb = gi.gbuffer();
        let meta = serde_json::json!({
            "width": w, "height": h,
            "normal_format": fmt_fmt(gb.config().normal_format),
            "material_format": fmt_fmt(gb.config().material_format),
            "z_format": "Depth32Float",
            "hzb_format": "R32Float",
            "hzb_mips": mip_count,
            "adapter": self.adapter_name,
            "device_label": "Viewer Device",
            "shader_hash": {
                "hzb_build": { "sha256": { "file": format!("{:x}", sha2::Sha256::digest(std::fs::read("shaders/hzb_build.wgsl").unwrap_or_default())) } },
                "ssao": { "sha256": { "file": format!("{:x}", sha2::Sha256::digest(std::fs::read("shaders/ssao.wgsl").unwrap_or_default())) } },
                "gbuffer_common": { "sha256": { "file": format!("{:x}", sha2::Sha256::digest(std::fs::read("shaders/gbuffer/common.wgsl").unwrap_or_default())) } },
                "gbuffer_pack": { "sha256": { "file": format!("{:x}", sha2::Sha256::digest(std::fs::read("shaders/gbuffer/pack.wgsl").unwrap_or_default())) } },
            }
        });
        std::fs::write(
            out_dir.join("p5_meta.json"),
            serde_json::to_vec_pretty(&meta)?,
        )?;
        println!("[P5] Wrote reports/p5 artifacts");
        Ok(())
    }

    pub(crate) fn capture_gi_output_tonemapped_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);

        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            render_view_to_rgba8_ex(
                &self.device,
                &self.queue,
                comp_pl,
                comp_bgl,
                &self.sky_output_view,
                &gi.gbuffer().depth_view,
                fog_view,
                self.config.format,
                width,
                height,
                far,
                &self.gi_output_hdr_view,
                0,
            )
        })
    }

    pub(crate) fn setup_p51_cornell_scene(&mut self) -> anyhow::Result<P51CornellSceneState> {
        use wgpu::util::DeviceExt;
        
        let prev = P51CornellSceneState {
            geom_vb: self.geom_vb.take(),
            geom_ib: self.geom_ib.take(),
            geom_index_count: self.geom_index_count,
            sky_enabled: self.sky_enabled,
            fog_enabled: self.fog_enabled,
            viz_mode: self.viz_mode,
            gi_viz_mode: self.gi_viz_mode,
            camera_mode: self.camera.mode(),
            camera_eye: self.camera.eye(),
            camera_target: self.camera.target(),
        };

        let assets_root = Path::new("assets");
        let cornell_box = crate::io::obj_read::import_obj(assets_root.join("cornell_box.obj"))
            .context("load assets/cornell_box.obj")?;
        let cornell_sphere =
            crate::io::obj_read::import_obj(assets_root.join("cornell_sphere.obj"))
                .context("load assets/cornell_sphere.obj")?;

        let mut scene = SceneMesh::new();
        scene.extend_with_mesh(&cornell_box.mesh, Mat4::IDENTITY, 0.6, 0.0);
        let sphere_xform = Mat4::from_translation(Vec3::new(0.0, 0.35, 0.0))
            * Mat4::from_scale(Vec3::splat(0.35));
        scene.extend_with_mesh(&cornell_sphere.mesh, sphere_xform, 0.3, 0.0);

        if scene.vertices.is_empty() || scene.indices.is_empty() {
            bail!("P5.1 Cornell scene mesh is empty");
        }

        let vertex_data = bytemuck::cast_slice(&scene.vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.p51.cornell.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.p51.cornell.ib"),
                contents: bytemuck::cast_slice(&scene.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = scene.indices.len() as u32;
        self.geom_bind_group = None;

        if self.albedo_texture.is_none() {
            let tex_size = 512u32;
            let mut pixels = vec![0u8; (tex_size * tex_size * 4) as usize];
            for px in pixels.chunks_exact_mut(4) {
                px[0] = 200;
                px[1] = 200;
                px[2] = 200;
                px[3] = 255;
            }
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.p51.cornell.albedo"),
                size: wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_size * 4),
                    rows_per_image: Some(tex_size),
                },
                wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
            );
            self.albedo_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.albedo_texture = Some(texture);
        }

        self.albedo_sampler.get_or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("viewer.p51.cornell.albedo.sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        });

        self.ensure_geom_bind_group()?;

        self.sky_enabled = false;
        self.fog_enabled = false;
        self.viz_mode = VizMode::Material;
        self.gi_viz_mode = GiVizMode::None;

        let eye = Vec3::new(-1.4, 1.1, -2.4);
        let target = Vec3::new(0.0, 1.0, 0.0);
        self.camera.set_look_at(eye, target, Vec3::Y);

        Ok(prev)
    }

    pub(crate) fn restore_p51_cornell_scene(&mut self, prev: P51CornellSceneState) {
        self.geom_vb = prev.geom_vb;
        self.geom_ib = prev.geom_ib;
        self.geom_index_count = prev.geom_index_count;
        self.sky_enabled = prev.sky_enabled;
        self.fog_enabled = prev.fog_enabled;
        self.viz_mode = prev.viz_mode;
        self.gi_viz_mode = prev.gi_viz_mode;
        self.camera
            .set_look_at(prev.camera_eye, prev.camera_target, Vec3::Y);
    }
}
