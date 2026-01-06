// src/viewer/state/resize.rs
// Resize handling for the Viewer
// Extracted from mod.rs as part of the viewer refactoring

use winit::dpi::PhysicalSize;
use crate::passes::gi::GiPass;
use crate::viewer::Viewer;

impl Viewer {
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.resize_render_targets(new_size.width, new_size.height);
        }
    }

    /// Resize all internal render targets (G-Buffer, GI, etc.) to the given dimensions.
    /// This is separate from window resize to support high-res offscreen snapshots.
    pub fn resize_render_targets(&mut self, width: u32, height: u32) {
        if let Some(ref mut gi) = self.gi {
            gi.gbuffer_mut()
                .resize(&self.device, width, height)
                .ok();
            gi.set_ssr_params(&self.queue, &self.ssr_params);
        }
        // Recreate lit output
        self.lit_output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.lit.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.lit_output_view = self
            .lit_output
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Recreate GI HDR baseline and output targets
        self.gi_baseline_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.baseline.hdr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.gi_baseline_hdr_view = self
            .gi_baseline_hdr
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gi_baseline_diffuse_hdr =
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gi.baseline.diffuse.hdr"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        self.gi_baseline_diffuse_hdr_view = self
            .gi_baseline_diffuse_hdr
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gi_baseline_spec_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.baseline.spec.hdr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.gi_baseline_spec_hdr_view = self
            .gi_baseline_spec_hdr
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gi_output_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.output.hdr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.gi_output_hdr_view = self
            .gi_output_hdr
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gi_debug = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.debug"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.gi_debug_view = self
            .gi_debug
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.gi_pass.is_some() {
            match GiPass::new(&self.device, width, height) {
                Ok(pass) => {
                    self.gi_pass = Some(pass);
                }
                Err(e) => {
                    eprintln!("Failed to recreate GiPass after resize: {}", e);
                    self.gi_pass = None;
                }
            }
        }
        // Recreate sky output
        self.sky_output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.sky.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.sky_output_view = self
            .sky_output
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Recreate depth buffer for geometry pass
        if self.geom_pipeline.is_some() {
            let z_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gbuf.z"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.z_texture = Some(z_texture);
            self.z_view = Some(z_view);
        }
        // Recreate fog textures
        self.fog_output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.fog_output_view = self
            .fog_output
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.fog_history = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.history"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.fog_history_view = self
            .fog_history
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Recreate half-resolution fog targets
        let half_w = (width.max(1)) / 2;
        let half_h = (height.max(1)) / 2;
        self.fog_output_half = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.output.half"),
            size: wgpu::Extent3d {
                width: half_w.max(1),
                height: half_h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.fog_output_half_view = self
            .fog_output_half
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.fog_history_half = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.history.half"),
            size: wgpu::Extent3d {
                width: half_w.max(1),
                height: half_h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.fog_history_half_view = self
            .fog_history_half
            .create_view(&wgpu::TextureViewDescriptor::default());
        // HUD resolution
        self.hud.set_resolution(width, height);
    }
}
