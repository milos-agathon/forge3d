// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes
#![allow(dead_code)]

pub mod camera_controller;
mod cmd;
mod event_loop;
pub mod hud;
pub mod image_analysis;
mod init;
mod input;
pub mod ipc;
mod p5;
mod render;
mod state;
mod terrain;
mod util;
mod viewer_analysis;
mod viewer_cmd_parse;
pub mod viewer_config;
mod viewer_constants;
pub mod viewer_enums;
mod viewer_image_utils;
mod viewer_render_helpers;
mod viewer_ssr_scene;
pub mod viewer_struct;
mod viewer_types;

// Re-export public items
pub use viewer_config::{set_initial_commands, ViewerConfig};
#[cfg(feature = "extension-module")]
pub use viewer_config::set_initial_terrain_config;
pub use ipc::IpcServerConfig;

use viewer_config::INITIAL_CMDS;
use viewer_constants::VIEWER_SNAPSHOT_MAX_MEGAPIXELS;
#[cfg(feature = "extension-module")]
use viewer_config::INITIAL_TERRAIN_CONFIG;
use viewer_types::{
    FogCameraUniforms, FogUpsampleParamsStd140, SkyUniforms, VolumetricUniformsStd140,
};

// HUD push functions available for future use
#[allow(unused_imports)]
use hud::{push_number, push_text_3x5};
use crate::core::shadows::CameraFrustum;
#[cfg(feature = "extension-module")]
use std::sync::Arc;

// Constants moved to viewer_constants.rs


// build_ssr_albedo_texture moved to viewer_ssr_scene.rs

impl Viewer {
    /// Override fog shadow map with a constant depth value.
    /// This is primarily intended for tests and debug paths until a
    /// full directional shadow pass is wired into the viewer.
    pub fn set_fog_shadow_constant_depth(&mut self, depth: f32) {
        let _ = depth;
    }

    /// Override the fog shadow matrix used by the volumetric shader.
    /// This allows tests or higher-level systems to provide a
    /// light-space transform without changing the default identity.
    pub fn set_fog_shadow_matrix(&mut self, mat: [[f32; 4]; 4]) {
        self.queue
            .write_buffer(&self.fog_shadow_matrix, 0, bytemuck::bytes_of(&mat));
    }

    /// P1.3: Enable or disable Temporal Anti-Aliasing
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        // P1.4: If terrain viewer is active, delegate TAA to terrain viewer
        let terrain_active = self.terrain_viewer.as_ref().map(|tv| tv.has_terrain()).unwrap_or(false);
        if terrain_active {
            if let Some(ref mut tv) = self.terrain_viewer {
                tv.set_taa_enabled(enabled);
            }
            // Don't enable main viewer's jitter when terrain is active
            return;
        }

        // Initialize TAA renderer if not already done
        if enabled && self.taa_renderer.is_none() {
            match crate::core::taa::TaaRenderer::new(
                &self.device,
                self.config.width,
                self.config.height,
            ) {
                Ok(renderer) => {
                    self.taa_renderer = Some(renderer);
                }
                Err(e) => {
                    eprintln!("[taa] Failed to create TAA renderer: {}", e);
                    return;
                }
            }
        }

        // Enable/disable TAA and jitter together
        if let Some(ref mut taa) = self.taa_renderer {
            taa.set_enabled(enabled);
        }
        
        // P1.2: Enable jitter when TAA is enabled (only for main viewer, not terrain)
        self.taa_jitter.enabled = enabled;
        if enabled {
            // Use enabled() to get proper initial offset
            self.taa_jitter = crate::core::jitter::JitterState::enabled();
        } else {
            self.taa_jitter = crate::core::jitter::JitterState::new();
        }
    }

    /// P1.3: Check if TAA is enabled
    pub fn is_taa_enabled(&self) -> bool {
        self.taa_renderer.as_ref().map(|t| t.is_enabled()).unwrap_or(false)
    }
}

// Types moved to viewer_types.rs (including P51CornellSceneState)

// build_ssr_scene_mesh moved to viewer_ssr_scene.rs

// ViewerConfig and statics moved to viewer_config.rs
pub use viewer_struct::Viewer;

// FpsCounter moved to viewer_config.rs

// FpsCounter moved to viewer_config.rs

impl Viewer {
    #[cfg(feature = "extension-module")]
    pub fn load_terrain_from_config(
        &mut self,
        cfg: &crate::render::params::RendererConfig,
    ) -> anyhow::Result<()> {
        // TerrainScene currently owns its own configuration; we accept cfg so
        // that future milestones can thread it through without changing the
        // Viewer API again.
        let _ = cfg;

        let scene = crate::terrain::TerrainScene::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            Arc::clone(&self.adapter),
        )?;
        self.terrain_scene = Some(scene);
        Ok(())
    }

    fn ensure_geom_bind_group(&mut self) -> anyhow::Result<()> {
        if self.geom_bind_group.is_some() {
            return Ok(());
        }
        let cam_buf = match self.geom_camera_buffer.as_ref() {
            Some(buf) => buf,
            None => return Ok(()),
        };
        let sampler = self.albedo_sampler.get_or_insert_with(|| {
            self.device
                .create_sampler(&wgpu::SamplerDescriptor::default())
        });
        let tex = self.albedo_texture.get_or_insert_with(|| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.empty"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let view = self
            .albedo_view
            .get_or_insert_with(|| tex.create_view(&wgpu::TextureViewDescriptor::default()));
        if let Some(ref layout) = self.geom_bind_group_layout {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg.runtime"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cam_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.geom_bind_group = Some(bg);
        }
        Ok(())
    }
}


// SSR scene helpers, GI methods, and reexecute_gi moved to p5/ssr_scene_impl.rs and other modules

// P5 capture helpers moved to p5/gbuffer_dump.rs

// Enums and ViewerCmd moved to viewer_enums.rs
// handle_cmd method moved to cmd/handler.rs

// upload_mesh and load_albedo_texture moved to state/mesh_upload.rs
// Event loop runner functions extracted to event_loop/runner.rs
pub use event_loop::{run_viewer, run_viewer_with_ipc};

// IPC state functions extracted to event_loop/ipc_state.rs
