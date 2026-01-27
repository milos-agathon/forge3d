// src/viewer/terrain/render.rs
// Terrain rendering for the interactive viewer

use super::scene::ViewerTerrainScene;
use crate::lighting::shadow::ShadowTechnique;
use crate::shadows::CsmUniforms;

/// Shader for accumulating frames (additive blend)
const ACCUMULATE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_tex, samp, in.uv);
}
"#;

/// Terrain uniforms for the simple terrain shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
}

/// Shadow pass uniforms for depth-only terrain rendering (per cascade)
/// Must match ShadowPassUniforms in terrain_shadow_depth.wgsl exactly (112 bytes)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowPassUniforms {
    pub light_view_proj: [[f32; 4]; 4],  // 64 bytes
    pub terrain_params: [f32; 4],         // 16 bytes: spacing, height_exag, height_min, height_max
    pub grid_params: [f32; 4],            // 16 bytes: grid_resolution, _pad, _pad, _pad
    pub height_curve: [f32; 4],           // 16 bytes: mode, strength, power, _pad
}

/// Extended uniforms for PBR terrain shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainPbrUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
    pub pbr_params: [f32; 4],    // exposure, normal_strength, ibl_intensity, _
    pub camera_pos: [f32; 4],   // camera world position
    pub lens_params: [f32; 4],  // vignette_strength, vignette_radius, vignette_softness, _
    pub screen_dims: [f32; 4],  // width, height, _, _
    pub overlay_params: [f32; 4], // enabled (>0.5), opacity, blend_mode (0=normal, 1=multiply, 2=overlay), solid (>0.5)
}

impl ViewerTerrainScene {
    pub(super) fn ensure_depth(&mut self, width: u32, height: u32) {
        if self.depth_size != (width, height) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            // Create depth view with explicit DepthOnly aspect for sampling
            self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain_viewer.depth_view"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            }));
            self.depth_texture = Some(tex);
            self.depth_size = (width, height);
        }
    }

    /// Render shadow depth passes for all cascades
    /// Must be called before the main terrain render pass
    pub fn render_shadow_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_view: glam::Mat4,
        camera_proj: glam::Mat4,
        sun_direction: glam::Vec3,
    ) {
        // Early exit if shadow infrastructure not ready
        let csm = match self.csm_renderer.as_mut() {
            Some(c) => c,
            None => return,
        };
        
        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };
        
        if self.shadow_pipeline.is_none() || self.shadow_bind_groups.is_empty() {
            return;
        }
        
        let cascade_count = csm.config.cascade_count;
        let shadow_map_size = csm.config.shadow_map_size;
        
        // Mark that shadow passes are running (will be copied to csm_uniforms later)
        csm.uniforms.technique_reserved[0] = 1.0; // Flag: shadow passes executed
        
        // Update CSM cascade matrices based on camera and light
        let near_plane = 1.0;
        let far_plane = csm.config.max_shadow_distance;
        csm.update_cascades(camera_view, camera_proj, sun_direction, near_plane, far_plane);
        
        // Get terrain parameters for shadow pass uniforms
        let (min_h, max_h) = terrain.domain;
        let terrain_span = terrain.dimensions.0.max(terrain.dimensions.1) as f32;
        
        let z_scale = terrain.z_scale;
        let grid_res = 512u32; // Shadow pass grid resolution
        
        // Height curve params: use linear (mode=0) with no curve transformation
        // This matches the default terrain rendering behavior
        let height_curve_mode = 0.0_f32;    // 0=linear
        let height_curve_strength = 0.0_f32; // No curve applied
        let height_curve_power = 1.0_f32;    // Default power
        
        // Compute a terrain-covering light view-projection matrix
        // The cascade's light_view_proj is based on camera frustum which doesn't cover full terrain
        // We need a projection that covers the entire terrain from light's perspective
        let terrain_height = (max_h - min_h) * z_scale;
        let terrain_center = glam::Vec3::new(terrain_span * 0.5, terrain_height * 0.5, terrain_span * 0.5);
        let light_distance = terrain_span * 2.0;  // Place light far enough to see entire terrain
        let light_pos = terrain_center - sun_direction * light_distance;
        let light_view = glam::Mat4::look_at_rh(light_pos, terrain_center, glam::Vec3::Y);
        
        // Orthographic projection covering terrain bounds with margin
        let half_extent = terrain_span * 0.75;  // Cover terrain with some margin
        let terrain_light_proj = glam::Mat4::orthographic_rh(
            -half_extent, half_extent,
            -half_extent, half_extent,
            0.1, light_distance * 2.0 + terrain_span,
        );
        let terrain_light_view_proj = terrain_light_proj * light_view;
        let terrain_light_view_proj_arr = terrain_light_view_proj.to_cols_array_2d();
        
        // Render each cascade
        for cascade_idx in 0..cascade_count as usize {
            if cascade_idx >= self.shadow_bind_groups.len() || cascade_idx >= self.shadow_uniform_buffers.len() {
                break;
            }
            
            // Use terrain-covering projection for shadow depth pass
            // This ensures the entire terrain is rendered to shadow map, not just camera frustum portion
            let light_view_proj = terrain_light_view_proj_arr;
            
            // Build shadow pass uniforms
            // Match main shader terrain_params layout: [min_h, h_range, terrain_width, z_scale]
            let shadow_uniforms = ShadowPassUniforms {
                light_view_proj,
                terrain_params: [min_h, max_h - min_h, terrain_span, z_scale],
                grid_params: [grid_res as f32, 0.0, 0.0, 0.0],
                height_curve: [height_curve_mode, height_curve_strength, height_curve_power, 0.0],
            };
            
            // Upload uniforms
            self.queue.write_buffer(
                &self.shadow_uniform_buffers[cascade_idx],
                0,
                bytemuck::cast_slice(&[shadow_uniforms]),
            );
            
            // Get shadow map view for this cascade
            let shadow_map_view = &csm.shadow_map_views[cascade_idx];
            
            // Begin depth-only render pass for this cascade
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_depth_pass_cascade_{}", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.shadow_bind_groups[cascade_idx], &[]);
            
            // Draw terrain grid (6 vertices per quad, (grid_res-1)^2 quads)
            let vertex_count = 6 * (grid_res - 1) * (grid_res - 1);
            render_pass.draw(0..vertex_count, 0..1);
        }
        
        // Execute moment generation pass for VSM/EVSM/MSM techniques
        // This converts the depth maps into moment statistics
        let technique = match self.pbr_config.shadow_technique.to_lowercase().as_str() {
            "vsm" => crate::lighting::shadow::ShadowTechnique::VSM,
            "evsm" => crate::lighting::shadow::ShadowTechnique::EVSM,
            "msm" => crate::lighting::shadow::ShadowTechnique::MSM,
            _ => return, // No moment generation needed for HARD/PCF/PCSS
        };
        
        // Prepare and execute moment pass if we have the resources
        if let (Some(ref mut moment_pass), Some(ref csm)) = (&mut self.moment_pass, &self.csm_renderer) {
            if let Some(ref moment_texture) = csm.evsm_maps {
                let depth_view = csm.shadow_texture_view();
                let moment_view = crate::shadows::create_moment_storage_view(moment_texture, cascade_count);
                
                moment_pass.prepare_bind_group(&self.device, &depth_view, &moment_view);
                moment_pass.execute(
                    &self.queue,
                    encoder,
                    technique,
                    cascade_count,
                    shadow_map_size,
                    csm.config.evsm_positive_exp,
                    csm.config.evsm_negative_exp,
                );
            }
        }
    }

    /// Render terrain to the given view
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        selected_feature_id: u32,
    ) -> bool {
        if self.terrain.is_none() {
            return false;
        }

        self.ensure_depth(width, height);

        // Initialize PBR pipeline if enabled but not yet created
        // This ensures overlays work in interactive mode, not just snapshots
        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(self.surface_format) {
                eprintln!("[render] Failed to initialize PBR pipeline: {}", e);
            }
        }

        // Pre-compute values needed for PBR bind group before borrowing terrain
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        
        // Check if DoF is enabled
        let needs_dof = self.pbr_config.dof.enabled;
        
        // Check if post-process effects need a separate pass (distortion, CA, or vignette)
        let needs_post_process = self.pbr_config.lens_effects.enabled && 
            (self.pbr_config.lens_effects.distortion.abs() > 0.001 || 
             self.pbr_config.lens_effects.chromatic_aberration > 0.001 ||
             self.pbr_config.lens_effects.vignette_strength > 0.001);
        
        // P5: Volumetrics pass
        let needs_volumetrics = self.pbr_config.volumetrics.enabled && 
            self.pbr_config.volumetrics.density > 0.0001;
            
        // M5: Denoise pass
        let needs_denoise = self.pbr_config.denoise.enabled;
        
        // We need PostProcess pass (for intermediate texture) if PP is active OR Volumetrics is active (since Vol reads from PP intermediate)
        if (needs_post_process || needs_volumetrics) && self.post_process.is_none() {
            self.init_post_process();
        }

        // We need DoF pass if DoF is active OR (Volumetrics AND PP are active) - for scratch buffer
        // Case: Vol+PP (no DoF) needs a scratch buffer for Vol output before PP reads it. We reuse DoF input buffer.
        let needs_dof_scratch = needs_volumetrics && needs_post_process && !needs_dof;
        
        if (needs_dof || needs_dof_scratch) && self.dof_pass.is_none() {
            self.init_dof_pass();
        }
        
        // Initialize volumetrics pass if needed
        if needs_volumetrics && self.volumetrics_pass.is_none() {
            self.init_volumetrics_pass();
        }
        
        if needs_denoise && self.denoise_pass.is_none() {
            self.init_denoise_pass();
        }

        // Ensure textures exist for active passes
        if needs_denoise {
             if let Some(ref mut denoise) = self.denoise_pass {
                 // Denoise pass manages its own ping-pong resources
                 let _ = denoise.get_input_view(width, height);
             }
        }
        if needs_dof || needs_dof_scratch {
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, self.surface_format);
            }
        }
        
        // P0.1/M1: Check if OIT is needed and initialize WBOIT resources early
        let has_vector_overlays_early = self.vector_overlay_stack.as_ref()
            .map(|s| s.visible_layer_count() > 0)
            .unwrap_or(false);
        if has_vector_overlays_early && self.oit_enabled {
            // Ensure WBOIT is fully initialized including textures and bind groups
            if self.wboit_compose_bind_group.is_none() || self.wboit_size != (width, height) {
                self.init_wboit(width, height);
            }
        }
        
        if needs_post_process || needs_volumetrics {
            if let Some(ref mut pp) = self.post_process {
                let _ = pp.get_intermediate_view(width, height, self.surface_format);
            }
        }
        
        // Extract all terrain values we need before any mutable operations
        let (phi, theta, r, tw, th, terrain_z_scale, domain, fov_deg, sun_azimuth_deg, sun_elevation_deg) = {
            let terrain = self.terrain.as_ref().unwrap();
            (
                terrain.cam_phi_deg.to_radians(),
                terrain.cam_theta_deg.to_radians(),
                terrain.cam_radius,
                terrain.dimensions.0,
                terrain.dimensions.1,
                terrain.z_scale,
                terrain.domain,
                terrain.cam_fov_deg,
                terrain.sun_azimuth_deg,
                terrain.sun_elevation_deg,
            )
        };
        
        let terrain_width = tw.max(th) as f32;
        let h_range = domain.1 - domain.0;
        let legacy_z_scale = terrain_z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr { terrain_z_scale } else { legacy_z_scale };
        let center_y = if use_pbr {
            h_range * terrain_z_scale * 0.5
        } else {
            terrain_width * legacy_z_scale * 0.001 * 0.5
        };
        let center = glam::Vec3::new(
            terrain_width * 0.5,
            center_y,
            terrain_width * 0.5,
        );

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        let proj_base = glam::Mat4::perspective_rh(
            fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        // P1.4: Apply TAA jitter to projection matrix
        let proj = if self.taa_jitter.enabled {
            crate::core::jitter::apply_jitter(
                proj_base,
                self.taa_jitter.offset.0,
                self.taa_jitter.offset.1,
                width,
                height,
            )
        } else {
            proj_base
        };
        let view_proj = proj * view_mat;

        let sun_az = sun_azimuth_deg.to_radians();
        let sun_el = sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        // Initialize shadow depth pipeline if PBR mode with shadows enabled
        if use_pbr && self.shadow_pipeline.is_none() {
            self.init_shadow_depth_pipeline();
            self.update_shadow_bind_groups();
        }
        
        // Render shadow depth passes before main terrain render
        if use_pbr && self.shadow_pipeline.is_some() {
            self.render_shadow_passes(encoder, view_mat, proj, -sun_dir);
        } else if use_pbr {
            eprintln!("[render] Skipping shadow passes: pipeline={}", self.shadow_pipeline.is_some());
        }
        
        // Debug: print uniform values on first render
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!("[render] terrain_params: min_h={:.1}, h_range={:.1}, width={:.1}, z_scale={:.2}",
                domain.0, h_range, terrain_width, shader_z_scale);
            let max_y = if use_pbr {
                h_range * terrain_z_scale
            } else {
                terrain_width * legacy_z_scale * 0.001
            };
            println!("[render] Expected Y range: 0 to {:.1}", max_y);
            println!("[render] Camera center: ({:.1}, {:.1}, {:.1})", center.x, center.y, center.z);
        });
        
        // Re-borrow terrain for the remaining operations
        let terrain = self.terrain.as_ref().unwrap();

        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [
                terrain.domain.0,
                terrain.domain.1 - terrain.domain.0,
                terrain_width,
                shader_z_scale,
            ],
            lighting: [
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
            ],
            background: [
                terrain.background_color[0],
                terrain.background_color[1],
                terrain.background_color[2],
                0.0,
            ],
            water_color: [
                terrain.water_color[0],
                terrain.water_color[1],
                terrain.water_color[2],
                0.0,
            ],
        };

        self.queue.write_buffer(
            &terrain.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Prepare PBR bind group if PBR mode is enabled (must be done before render pass)
        // Extract values needed for PBR uniforms before mutable borrow
        let pbr_uniforms_data = if use_pbr {
            Some((
                terrain.domain,
                terrain.z_scale,
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
                terrain.background_color,
                terrain.water_color,
            ))
        } else {
            None
        };
        
        // Release the immutable borrow of terrain before mutable operations
        let _ = terrain;
        
        if let Some((domain, z_scale, sun_intensity, ambient, shadow_intensity, water_level, background_color, water_color)) = pbr_uniforms_data {
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [
                    domain.0,
                    domain.1 - domain.0,
                    terrain_width,
                    z_scale,
                ],
                lighting: [
                    sun_intensity,
                    ambient,
                    shadow_intensity,
                    water_level,
                ],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    0.0,
                ],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    0.0,
                ],
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                lens_params: [
                    self.pbr_config.lens_effects.vignette_strength,
                    self.pbr_config.lens_effects.vignette_radius,
                    self.pbr_config.lens_effects.vignette_softness,
                    0.0,
                ],
                screen_dims: [width as f32, height as f32, 0.0, 0.0],
                overlay_params: [
                    if self.pbr_config.overlay.enabled { 1.0 } else { 0.0 },
                    self.pbr_config.overlay.global_opacity,
                    0.0,  // Blend mode: 0 = Normal
                    if self.pbr_config.overlay.solid { 1.0 } else { 0.0 },
                ],
            };
            
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes for heightfield AO and sun visibility before render
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);
        
        // Re-borrow terrain after mutable operations
        // Determine primary render target based on active effects
        // Logic:
        // - If Volumetrics active: Scene -> PP Intermediate (so Vol can read it)
        // - Else if DoF active: Scene -> DoF Input
        // - Else if PP active: Scene -> PP Intermediate
        // - Else: Scene -> Final View
        
        // - If Denoise active: Scene -> Denoise Input
        // - Else if Volumetrics active: Scene -> PP Intermediate (so Vol can read it)
        // - Else if DoF active: Scene -> DoF Input
        // - Else if PP active: Scene -> PP Intermediate
        // - Else: Scene -> Final View
        
        let render_target: &wgpu::TextureView = if needs_denoise {
            self.denoise_pass.as_mut().unwrap().get_input_view(width, height)
        } else if needs_volumetrics {
            self.post_process.as_ref().unwrap().intermediate_view.as_ref().unwrap()
        } else if needs_dof {
            self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
        } else if needs_post_process {
            self.post_process.as_ref().unwrap().intermediate_view.as_ref().unwrap()
        } else {
            view
        };
        
        // Store camera params for DoF
        let cam_radius = self.terrain.as_ref().unwrap().cam_radius;
        
        let terrain = self.terrain.as_ref().unwrap();
        let depth_view = self.depth_view.as_ref().unwrap();
        let bg = &terrain.background_color;
        
        // Option B: Prepare vector overlay stack if it has visible layers
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            stack.is_enabled() && stack.visible_layer_count() > 0
        } else {
            false
        };
        
        if has_vector_overlays {
            // Ensure we have a fallback texture for when sun visibility isn't enabled
            if self.fallback_texture.is_none() {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vector_overlay_fallback_texture"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                // Initialize to 1.0 (full visibility / no shadow)
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&[1.0f32]),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
                self.fallback_texture_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                self.fallback_texture = Some(texture);
            }
            
            // Initialize vector overlay pipelines if not yet done
            if let Some(ref mut stack) = self.vector_overlay_stack {
                // P0.1/M1: For OIT mode, ensure OIT pipelines are initialized
                if !stack.pipelines_ready() || (self.oit_enabled && !stack.oit_pipelines_ready()) {
                    stack.init_pipelines(self.surface_format);
                }
                
                // Prepare bind group with sun visibility texture or fallback
                let texture_view = self.sun_vis_view.as_ref()
                    .or(self.fallback_texture_view.as_ref())
                    .unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }
        
        // Store values needed for vector overlay rendering
        let vo_view_proj = view_proj.to_cols_array_2d();
        let vo_sun_dir = [sun_dir.x, sun_dir.y, sun_dir.z];
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Choose pipeline based on PBR mode
            if use_pbr {
                if let Some(ref pbr_bind_group) = self.pbr_bind_group {
                    pass.set_pipeline(self.pbr_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, pbr_bind_group, &[]);
                } else {
                    // Fallback to legacy if bind group not ready
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &terrain.bind_group, &[]);
                }
            } else {
                // Legacy rendering path
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &terrain.bind_group, &[]);
            }
            
            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
            
            // Option B: Render vector overlays after terrain
            // P0.1/M1: Use OIT rendering if enabled, otherwise standard alpha blending
            if has_vector_overlays && !self.oit_enabled {
                // Standard alpha blending path
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_with_highlight(
                                &mut pass, 
                                super::vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                }
                            );
                        }
                    }
                }
            }
        }
        
        // P0.1/M1: OIT rendering path - render overlays to WBOIT accumulation buffers
        if has_vector_overlays && self.oit_enabled {
            let depth_view = self.depth_view.as_ref().unwrap();
            
            // OIT accumulation pass
            if let (Some(color_view), Some(reveal_view)) = 
                (self.wboit_color_view.as_ref(), self.wboit_reveal_view.as_ref()) 
            {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.accumulation_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve terrain depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.oit_pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_oit(
                                &mut oit_pass,
                                super::vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                }
                            );
                        }
                    }
                }
            }
            
            // OIT compose pass - blend accumulated transparency onto scene
            if let (Some(pipeline), Some(bind_group)) = 
                (self.wboit_compose_pipeline.as_ref(), self.wboit_compose_bind_group.as_ref())
            {
                let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.compose_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve scene
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                compose_pass.set_pipeline(pipeline);
                compose_pass.set_bind_group(0, bind_group, &[]);
                compose_pass.draw(0..3, 0..1); // Fullscreen triangle
            }
            
            static OIT_LOG_ONCE: std::sync::Once = std::sync::Once::new();
            OIT_LOG_ONCE.call_once(|| {
                println!("[render] WBOIT active: mode={}", self.oit_mode);
            });
        }
        
        // M5: Apply Denoise pass if enabled (after scene render, before effects)
        if needs_denoise {
            let (iterations, sigma_color) = {
                let config = &self.pbr_config.denoise;
                (config.iterations, config.sigma_color)
            };

            let ViewerTerrainScene {
                denoise_pass,
                post_process,
                dof_pass,
                depth_view,
                queue,
                device,
                surface_format,
                ..
            } = self;

            if let Some(denoise) = denoise_pass.as_mut() {
                let depth_view = depth_view.as_ref().unwrap();
                denoise.apply(encoder, depth_view, iterations, sigma_color);

                let denoise_result = denoise
                    .get_last_result_view(iterations)
                    .unwrap_or(denoise.view_a.as_ref().unwrap());

                if post_process.is_none() {
                    *post_process = Some(super::post_process::PostProcessPass::new(
                        device.clone(),
                        *surface_format,
                    ));
                }

                let post_process = post_process.as_mut().unwrap();
                let mut intermediate_view = None;
                let next_target = if needs_volumetrics {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else if needs_dof {
                    dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else if needs_post_process {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else {
                    view
                };

                post_process.apply_from_input(
                    encoder,
                    queue,
                    denoise_result,
                    next_target,
                    width,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );

                if let Some(view) = intermediate_view {
                    post_process.intermediate_view = Some(view);
                }
            }
        }
        
        // P5: Apply volumetrics pass if enabled (after main render, before DoF)
        if needs_volumetrics {
            if let Some(ref vol_pass) = self.volumetrics_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                let color_input = self.post_process.as_ref().unwrap().intermediate_view.as_ref().unwrap();
                
                // Determine volumetrics output target with robust chaining:
                // - If DoF enabled: output to DoF input
                // - If DoF disabled but PP enabled: output to DoF input (as scratch buffer)
                // - Otherwise: output to Final View
                let vol_output = if needs_dof || needs_post_process {
                    self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else {
                    view
                };
                
                // Calculate inverse view-projection matrix
                let inv_view_proj = (proj * view_mat).inverse();
                
                vol_pass.apply(
                    encoder,
                    &self.queue,
                    color_input,
                    depth_view,
                    &terrain.heightmap_view,
                    vol_output,
                    width,
                    height,
                    inv_view_proj.to_cols_array_2d(),
                    [eye.x, eye.y, eye.z],
                    1.0,  // near
                    cam_radius * 10.0,  // far
                    [sun_dir.x, sun_dir.y, sun_dir.z],
                    terrain.sun_intensity,
                    [terrain_width, terrain.domain.0, shader_z_scale, h_range],
                    &self.pbr_config.volumetrics,
                );
            }
        }
        
        // Apply DoF pass if enabled (before other post-process effects)
        if needs_dof {
            // DoF input source:
            // - If Volumetrics enabled: Read from DoF Input (Volumetrics wrote here)
            // - Else: Read from DoF Input (Scene wrote here)
            // Note: In both cases, data is already in dof.input_view
            
            // DoF output target:
            // - If PP enabled: Output to PP Intermediate (so PP can read it)
            // - Else: Output to Final View
            let dof_output = if needs_post_process {
                self.post_process.as_ref().unwrap().intermediate_view.as_ref().unwrap()
            } else {
                view
            };
            
            if let Some(ref mut dof) = self.dof_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                
                let dof_cfg = super::dof::DofConfig {
                    enabled: true,
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };
                
                // We use apply_from_input to explicit chaining from dof.input_view
                dof.apply_from_input(
                    encoder,
                    &self.queue,
                    depth_view,
                    dof_output,
                    width,
                    height,
                    self.surface_format,
                    &dof_cfg,
                    1.0,           // near plane
                    cam_radius * 10.0, // far plane
                );
            }
        }
        
        // Apply post-process pass if needed (distortion, CA, vignette)
        if needs_post_process {
            // Determine if we need to read from an external source (scratch buffer)
            // This happens when Volumetrics + PP are active but DoF is not.
            // In that case, Volumetrics wrote to DoF Input (as scratch), so we read from there.
            // In all other cases (including DoF+PP), the input is in our own intermediate_view.
            let external_input = if !needs_dof && needs_volumetrics {
                self.dof_pass.as_ref().and_then(|dof| dof.input_view.as_ref())
            } else {
                None
            };

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
                
                if let Some(input_view) = external_input {
                    // Read from external texture (DoF scratch)
                    pp.apply_from_input(
                        encoder,
                        &self.queue,
                        input_view,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                } else {
                    // Read from internal intermediate texture (Standard path)
                    pp.apply(
                        encoder,
                        &self.queue,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                }
            }
        }

        true
    }

    /// Render terrain to an offscreen texture at the specified resolution.
    /// This allows high-resolution snapshots independent of window size.
    /// Returns the texture if successful.
    pub fn render_to_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        selected_feature_id: u32,
    ) -> Option<wgpu::Texture> {
        eprintln!("[DEBUG render_to_texture ENTRY] {}x{}", width, height);
        if self.terrain.is_none() {
            eprintln!("[DEBUG render_to_texture] No terrain, returning None");
            return None;
        }
        
        // Ensure critical resources are initialized (needed when called without prior interactive rendering)
        self.ensure_fallback_texture();
        
        // Ensure shared depth resources exist (used by compute shaders and PBR)
        // Note: render_to_texture creates its own depth texture, but some systems need
        // the shared depth_view to exist even if not directly used
        if self.depth_view.is_none() {
            self.ensure_depth(width, height);
        }

        // Create offscreen color texture at requested resolution
        // Include TEXTURE_BINDING for DoF shader to sample from
        let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.snapshot_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture at requested resolution
        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.snapshot_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate all the same values as render() but with custom dimensions
        // Ensure PBR pipeline is initialized if PBR mode is enabled
        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(target_format) {
                eprintln!("[snapshot] Failed to initialize PBR pipeline: {}", e);
            }
        }
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        let terrain = self.terrain.as_ref().unwrap();

        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let terrain_width = tw.max(th) as f32;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let legacy_z_scale = terrain.z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr { terrain.z_scale } else { legacy_z_scale };
        let center_y = if use_pbr {
            h_range * terrain.z_scale * 0.5
        } else {
            terrain_width * legacy_z_scale * 0.001 * 0.5
        };
        let center = glam::Vec3::new(
            terrain_width * 0.5,
            center_y,
            terrain_width * 0.5,
        );

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        // Use requested width/height for aspect ratio
        let proj_base = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        // P1.4: Apply TAA jitter to projection matrix (snapshot path)
        let proj = if self.taa_jitter.enabled {
            crate::core::jitter::apply_jitter(
                proj_base,
                self.taa_jitter.offset.0,
                self.taa_jitter.offset.1,
                width,
                height,
            )
        } else {
            proj_base
        };
        let view_proj = proj * view_mat;

        let sun_az = terrain.sun_azimuth_deg.to_radians();
        let sun_el = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [
                terrain.domain.0,
                terrain.domain.1 - terrain.domain.0,
                terrain_width,
                shader_z_scale,
            ],
            lighting: [
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
            ],
            background: [
                terrain.background_color[0],
                terrain.background_color[1],
                terrain.background_color[2],
                0.0,
            ],
            water_color: [
                terrain.water_color[0],
                terrain.water_color[1],
                terrain.water_color[2],
                0.0,
            ],
        };

        self.queue.write_buffer(
            &terrain.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Prepare PBR bind group if enabled
        let pbr_uniforms_data = if use_pbr {
            Some((
                terrain.domain,
                terrain.z_scale,
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
                terrain.background_color,
                terrain.water_color,
            ))
        } else {
            None
        };

        let bg = terrain.background_color;
        let _ = terrain; // Release borrow

        if let Some((domain, z_scale, sun_intensity, ambient, shadow_intensity, water_level, background_color, water_color)) = pbr_uniforms_data {
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [
                    domain.0,
                    domain.1 - domain.0,
                    terrain_width,
                    z_scale,
                ],
                lighting: [
                    sun_intensity,
                    ambient,
                    shadow_intensity,
                    water_level,
                ],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    0.0,
                ],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    0.0,
                ],
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                lens_params: [
                    self.pbr_config.lens_effects.vignette_strength,
                    self.pbr_config.lens_effects.vignette_radius,
                    self.pbr_config.lens_effects.vignette_softness,
                    0.0,
                ],
                screen_dims: [width as f32, height as f32, 0.0, 0.0],
                overlay_params: [
                    if self.pbr_config.overlay.enabled { 1.0 } else { 0.0 },
                    self.pbr_config.overlay.global_opacity,
                    0.0,  // Blend mode: 0 = Normal
                    if self.pbr_config.overlay.solid { 1.0 } else { 0.0 },
                ],
            };
            // DEBUG: Print overlay_params values
            eprintln!("[DEBUG render_to_texture] overlay_params: enabled={}, opacity={}, blend={}, solid={}",
                pbr_uniforms.overlay_params[0],
                pbr_uniforms.overlay_params[1],
                pbr_uniforms.overlay_params[2],
                pbr_uniforms.overlay_params[3]);
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);

        // Option B: Prepare vector overlay stack if it has visible layers (for snapshot path)
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            let enabled = stack.is_enabled();
            let count = stack.visible_layer_count();
            // println!("[snapshot] Checking overlays: enabled={}, visible={}", enabled, count);
            enabled && count > 0
        } else {
            false
        };
        
        if has_vector_overlays {
            // Ensure we have a fallback texture for when sun visibility isn't enabled
            if self.fallback_texture.is_none() {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vector_overlay_fallback_texture_snapshot"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                // Initialize to 1.0 (full visibility / no shadow)
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&[1.0f32]),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
                self.fallback_texture_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                self.fallback_texture = Some(texture);
            }
            
            // Initialize vector overlay pipelines if not yet done
            if let Some(ref mut stack) = self.vector_overlay_stack {
                // P0.1/M1: For OIT snapshots, ensure OIT pipelines are initialized
                if !stack.pipelines_ready() || (self.oit_enabled && !stack.oit_pipelines_ready()) {
                    stack.init_pipelines(self.surface_format);
                }
                
                // Prepare bind group with sun visibility texture or fallback
                // For snapshots, force fallback texture (fully lit) to avoid potential issues with
                // uninitialized or black sun visibility texture which would hide the overlays.
                let texture_view = self.fallback_texture_view.as_ref().unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }
        
        // Store values needed for vector overlay rendering
        let vo_view_proj = view_proj.to_cols_array_2d();
        let vo_sun_dir = [sun_dir.x, sun_dir.y, sun_dir.z];
        let terrain = self.terrain.as_ref().unwrap();
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];
        let _ = terrain; // Release borrow for render pass

        // Render to offscreen texture
        {
            let terrain = self.terrain.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.snapshot_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if use_pbr {
                if let Some(ref pbr_bind_group) = self.pbr_bind_group {
                    pass.set_pipeline(self.pbr_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, pbr_bind_group, &[]);
                } else {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &terrain.bind_group, &[]);
                }
            } else {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &terrain.bind_group, &[]);
            }

            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
            
            // Option B: Render vector overlays after terrain (snapshot path)
            // P0.1/M1: Use standard alpha blending if OIT is disabled
            if has_vector_overlays && !self.oit_enabled {
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5]; // Yellow highlight
                        for i in 0..layer_count {
                            stack.render_layer_with_highlight(
                                &mut pass, 
                                super::vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                }
                            );
                        }
                    }
                }
            }
        }
        
        // P0.1/M1: OIT rendering path for snapshots - render overlays to temporary WBOIT accumulation buffers
        if has_vector_overlays && self.oit_enabled {
            // Ensure OIT compose pipeline is initialized (size-independent resources)
            // This is safe to call from snapshot path as it doesn't corrupt interactive viewer state
            self.init_wboit_pipeline();
            
            // Create temporary OIT accumulation textures at snapshot resolution
            let oit_color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_color"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let oit_color_view = oit_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            
            let oit_reveal_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_reveal"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let oit_reveal_view = oit_reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());
            
            // OIT accumulation pass
            {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.snapshot_oit_accumulation"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &oit_color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &oit_reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve terrain depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.oit_pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_oit(
                                &mut oit_pass,
                                super::vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                }
                            );
                        }
                    } else {
                        println!("[snapshot] OIT skip: pipelines_ready={} bind_group={}", stack.oit_pipelines_ready(), stack.bind_group.is_some());
                    }
                }
            }
            
            // Create temporary OIT compose bind group
            // Note: compose pipeline is initialized by render() method when OIT is first enabled
            if let (Some(ref pipeline), Some(ref sampler)) = 
                (self.wboit_compose_pipeline.as_ref(), self.wboit_sampler.as_ref()) 
            {
                println!("[snapshot] Compositing OIT result");
                let layout = &pipeline.get_bind_group_layout(0);
                let compose_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.snapshot_oit_compose_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&oit_color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&oit_reveal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });
                
                // OIT compose pass - blend accumulated transparency onto scene
                let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.snapshot_oit_compose"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve scene
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                compose_pass.set_pipeline(pipeline);
                compose_pass.set_bind_group(0, &compose_bind_group, &[]);
                compose_pass.draw(0..3, 0..1); // Fullscreen triangle
            }
        }

        let mut out_tex = color_tex;
        let mut out_view = color_view;

        // P1.4: TAA is disabled for render_to_texture snapshots
        // TAA requires temporal history between consecutive frames with small camera deltas.
        // For animation exports, each frame is rendered independently at potentially large
        // camera deltas, so TAA temporal accumulation doesn't apply correctly.
        // TAA is only used for the interactive render() path where frames are consecutive.

        // P5: Apply volumetrics pass if enabled (after main render, before DoF)
        let needs_volumetrics = self.pbr_config.volumetrics.enabled && 
            self.pbr_config.volumetrics.density > 0.0001;
        if needs_volumetrics {
            // Initialize volumetrics pass if needed
            if self.volumetrics_pass.is_none() {
                self.init_volumetrics_pass();
            }

            if let Some(ref vol_pass) = self.volumetrics_pass {
                // Create output texture for volumetrics
                let vol_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_vol_output"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let vol_output_view = vol_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                // Calculate inverse view-projection matrix
                let inv_view_proj = (proj * view_mat).inverse();
                let terrain = self.terrain.as_ref().unwrap();
                let cam_radius = terrain.cam_radius;
                let terrain_sun_intensity = terrain.sun_intensity;

                vol_pass.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    &depth_view,
                    &terrain.heightmap_view,
                    &vol_output_view,
                    width,
                    height,
                    inv_view_proj.to_cols_array_2d(),
                    [eye.x, eye.y, eye.z],
                    1.0,  // near
                    cam_radius * 10.0,  // far
                    [sun_dir.x, sun_dir.y, sun_dir.z],
                    terrain_sun_intensity,
                    [terrain_width, terrain.domain.0, shader_z_scale, h_range],
                    &self.pbr_config.volumetrics,
                );

                out_tex = vol_output_tex;
                out_view = vol_output_view;
            }
        }

        // Apply DoF if enabled
        let needs_dof = self.pbr_config.dof.enabled;
        if needs_dof {
            // Initialize DoF pass if needed
            if self.dof_pass.is_none() {
                self.init_dof_pass();
            }
            
            if let Some(ref mut dof) = self.dof_pass {
                // Get DoF input view (allocates textures if needed)
                let _ = dof.get_input_view(width, height, target_format);
                
                // We need to copy the rendered scene to DoF input first
                // Then apply DoF from input to a new output texture
                let dof_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_dof_output"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let dof_output_view = dof_output_tex.create_view(&wgpu::TextureViewDescriptor::default());
                
                let cam_radius = self.terrain.as_ref().map(|t| t.cam_radius).unwrap_or(2000.0);
                
                let dof_cfg = super::dof::DofConfig {
                    enabled: true,
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };
                
                // Apply DoF: reads from color_view, writes to dof_output_view
                dof.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    &depth_view,
                    &dof_output_view,
                    width,
                    height,
                    target_format,
                    &dof_cfg,
                    1.0,
                    cam_radius * 10.0,
                );
                
                out_tex = dof_output_tex;
                out_view = dof_output_view;
            }
        }

        // Apply lens effects post-process in snapshot path to match on-screen output
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        if needs_post_process {
            if self.post_process.is_none() {
                self.init_post_process();
            }

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
                let lens_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_lens_output"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let lens_output_view =
                    lens_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                pp.apply_from_input(
                    encoder,
                    &self.queue,
                    &out_view,
                    &lens_output_view,
                    width,
                    height,
                    lens.distortion,
                    lens.chromatic_aberration,
                    lens.vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
                return Some(lens_output_tex);
            }
        }

        Some(out_tex)
    }

    /// Render terrain with motion blur via temporal accumulation.
    /// Returns the final resolved texture if successful.
    pub fn render_with_motion_blur(
        &mut self,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<wgpu::Texture> {
        if self.terrain.is_none() {
            return None;
        }

        let config = self.pbr_config.motion_blur.clone();
        if !config.enabled || config.samples <= 1 {
            // No motion blur needed, fall back to regular render
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain_viewer.motion_blur_fallback"),
            });
            let result = self.render_to_texture(&mut encoder, target_format, width, height, 0);
            self.queue.submit(std::iter::once(encoder.finish()));
            return result;
        }

        // Initialize motion blur pass if needed
        if self.motion_blur_pass.is_none() {
            self.init_motion_blur_pass();
        }

        // Store original camera params
        let terrain = self.terrain.as_ref().unwrap();
        let base_phi = terrain.cam_phi_deg;
        let base_theta = terrain.cam_theta_deg;
        let base_radius = terrain.cam_radius;
        let _ = terrain;

        // Create accumulation texture (Rgba32Float for HDR)
        let accum_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.motion_blur_accum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear accumulation buffer
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain_viewer.motion_blur_clear"),
            });
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.motion_blur_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accum_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Render N sub-frames with interpolated camera
        let orig_distortion = self.pbr_config.lens_effects.distortion;
        let orig_chromatic_aberration = self.pbr_config.lens_effects.chromatic_aberration;
        self.pbr_config.lens_effects.distortion = 0.0;
        self.pbr_config.lens_effects.chromatic_aberration = 0.0;

        let samples = config.samples.max(1);

        for i in 0..samples {
            // Calculate interpolation factor using the shutter timing
            // sample_t spans [shutter_open, shutter_close] across the sample set
            // This allows the cam_*_delta values to represent motion per full frame,
            // with the shutter timing determining how much of that motion is captured
            let shutter_range = config.shutter_close - config.shutter_open;
            let relative_t = (i as f32 + 0.5) / samples as f32;  // 0..1
            let sample_t = config.shutter_open + shutter_range * relative_t;

            // Interpolate camera position across the shutter interval
            // cam_*_delta represents motion per full frame (frame time = 1.0)
            // The shutter timing naturally scales the effective motion captured
            let phi = base_phi + config.cam_phi_delta * sample_t;
            let theta = base_theta + config.cam_theta_delta * sample_t;
            let radius = base_radius + config.cam_radius_delta * sample_t;

            // Temporarily set camera params
            if let Some(ref mut terrain) = self.terrain {
                terrain.cam_phi_deg = phi;
                terrain.cam_theta_deg = theta;
                terrain.cam_radius = radius;
            }

            // Render frame to temporary texture
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain_viewer.motion_blur_sample"),
            });
            
            let frame_tex = self.render_to_texture(&mut encoder, target_format, width, height, 0);
            self.queue.submit(std::iter::once(encoder.finish()));

            // Add to accumulation (additive blend)
            if let Some(ref frame) = frame_tex {
                let frame_view = frame.create_view(&wgpu::TextureViewDescriptor::default());
                self.accumulate_frame(&frame_view, &accum_view, width, height);
            }
        }

        self.pbr_config.lens_effects.distortion = orig_distortion;
        self.pbr_config.lens_effects.chromatic_aberration = orig_chromatic_aberration;

        // Restore original camera params
        if let Some(ref mut terrain) = self.terrain {
            terrain.cam_phi_deg = base_phi;
            terrain.cam_theta_deg = base_theta;
            terrain.cam_radius = base_radius;
        }

        // Resolve: create final output and divide by sample count
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.motion_blur_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Use motion blur pass to resolve
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);

        let mut final_tex = output_tex;
        if let Some(ref motion_blur) = self.motion_blur_pass {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain_viewer.motion_blur_resolve"),
            });
            motion_blur.resolve(
                &mut encoder,
                &self.queue,
                &accum_view,
                &output_view,
                width,
                height,
                samples,
            );

            if needs_post_process {
                if self.post_process.is_none() {
                    self.init_post_process();
                }
                if let Some(ref mut pp) = self.post_process {
                    let lens = &self.pbr_config.lens_effects;
                    let lens_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("terrain_viewer.motion_blur_lens_output"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: target_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let lens_output_view = lens_output_tex
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    pp.apply_from_input(
                        &mut encoder,
                        &self.queue,
                        &output_view,
                        &lens_output_view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                    final_tex = lens_output_tex;
                }
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        println!("[terrain] Motion blur: {} samples rendered", samples);
        Some(final_tex)
    }

    /// Accumulate a frame into the accumulation buffer (additive blend)
    fn accumulate_frame(
        &self,
        frame_view: &wgpu::TextureView,
        accum_view: &wgpu::TextureView,
        _width: u32,
        _height: u32,
    ) {
        // Create a simple additive blit pipeline if needed
        // Use a simple additive pass until a dedicated accumulation pipeline is wired.
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur.accumulate_shader"),
            source: wgpu::ShaderSource::Wgsl(ACCUMULATE_SHADER.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion_blur.accumulate_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion_blur.accumulate_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion_blur.accumulate_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motion_blur.accumulate_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_blur.accumulate_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion_blur.accumulate"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur.accumulate_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: accum_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Blit an arbitrary texture view to a target view using the existing post-process pipeline with neutral parameters.
    pub fn blit_texture_to_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        if self.post_process.is_none() {
            self.post_process = Some(super::post_process::PostProcessPass::new(
                self.device.clone(),
                format,
            ));
        }
        if let Some(ref mut pp) = self.post_process {
            pp.apply_from_input(
                encoder,
                &self.queue,
                input_view,
                output_view,
                width,
                height,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
        }
    }

    /// Prepare PBR bind group with current uniforms (called before render pass)
    /// Gets heightmap_view internally from self.terrain to avoid borrow issues
    fn prepare_pbr_bind_group_internal(&mut self, uniforms: &TerrainPbrUniforms) {
        // Ensure fallback texture exists first (before any borrows)
        self.ensure_fallback_texture();
        
        // Early return checks
        if self.pbr_bind_group_layout.is_none() || self.terrain.is_none() {
            return;
        }

        // Create or update uniform buffer
        if self.pbr_uniform_buffer.is_none() {
            self.pbr_uniform_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain_viewer_pbr.uniform_buffer"),
                size: std::mem::size_of::<TerrainPbrUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Write uniforms
        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&[*uniforms]));
        }

        // Recreate bind group
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer_pbr.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Ensure overlay stack exists with fallback texture BEFORE borrowing other fields
        if self.overlay_stack.is_none() {
            self.overlay_stack = Some(super::overlay::OverlayStack::new(
                self.device.clone(),
                self.queue.clone(),
            ));
        }
        // Rebuild overlay composite if dirty, then ensure fallback exists
        if let Some(ref mut stack) = self.overlay_stack {
            if stack.is_dirty() {
                if let Some(ref terrain) = self.terrain {
                    stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
                }
            }
            stack.ensure_fallback_texture();
        }

        // Now borrow everything we need
        let layout = self.pbr_bind_group_layout.as_ref().unwrap();
        let terrain = self.terrain.as_ref().unwrap();
        let fallback_view = self.fallback_texture_view.as_ref().unwrap();
        let ao_view = self.height_ao_view.as_ref().unwrap_or(fallback_view);
        let sv_view = self.sun_vis_view.as_ref().unwrap_or(fallback_view);
        
        // Get overlay view and sampler from stack
        // ensure_fallback_texture() guarantees composite_view is Some (either actual composite or RGBA fallback)
        let overlay_stack = self.overlay_stack.as_ref().unwrap();
        let overlay_view = overlay_stack
            .composite_view()
            .expect("overlay composite_view should exist after ensure_fallback_texture");
        let overlay_sampler = overlay_stack.sampler();

        // Get CSM shadow resources - create fallbacks if they don't exist
        let (shadow_view, moment_view, shadow_sampler) = if let Some(csm) = self.csm_renderer.as_ref() {
            let shadow_view = csm.shadow_texture_view();
            if let Some(moment_view) = csm.moment_texture_view() {
                (shadow_view, moment_view, &csm.shadow_sampler)
            } else {
                eprintln!("[WARN] CSM moment maps not created - using fallback");
                // Create fallback moment texture
                let fallback = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("csm_moment_fallback"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 4 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let moment_view = fallback.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                });
                (shadow_view, moment_view, &csm.shadow_sampler)
            }
        } else {
            eprintln!("[ERROR] CSM renderer not initialized - cannot create PBR bind group");
            return;
        };
        
        // Moment sampler (Filtering)
        // We can use the existing pbr sampler (linear) or create a new one. 
        // csm_renderer doesn't expose a dedicated moment sampler, but it uses Linear/Linear.
        // Let's use the one we created above 'sampler' which is Linear/Nearest/Clamp.
        // Or better, creating a dedicated one matching CSM requirements.
        
        let moment_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("csm.moment.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let csm_buffer = if let Some(buf) = &self.csm_uniform_buffer {
            buf
        } else {
            return;
        };
        
        // P6.2: Write CSM uniforms with technique value from pbr_config
        // Map shadow technique string to enum value
        let technique = match self.pbr_config.shadow_technique.to_lowercase().as_str() {
            "hard" => ShadowTechnique::Hard,
            "pcf" => ShadowTechnique::PCF,
            "pcss" => ShadowTechnique::PCSS,
            "vsm" => ShadowTechnique::VSM,
            "evsm" => ShadowTechnique::EVSM,
            "msm" => ShadowTechnique::MSM,
            _ => ShadowTechnique::PCF, // default
        };
        
        // Build CSM uniforms with current technique
        // Use cascade data from CSM renderer if shadow depth passes have been rendered
        let debug_mode = self.pbr_config.debug_mode;
        let csm_uniforms = if let Some(ref csm) = self.csm_renderer {
            // Copy uniforms from CSM renderer (populated by render_shadow_passes)
            let mut u = csm.uniforms;
            u.technique = technique.as_u32();
            u.pcf_kernel_size = match technique {
                ShadowTechnique::Hard => 1,
                ShadowTechnique::PCSS => 5,
                _ => 3,
            };
            u.technique_params = [0.0, 0.0, 0.0005, 1.0]; // moment_bias, light_size
            u.debug_mode = debug_mode; // P6.2: Debug visualization from pbr_config
            u
        } else {
            // Fallback: no shadow depth passes, cascade_count=0 triggers soft shadow fallback
            let mut u = CsmUniforms::default();
            u.technique = technique.as_u32();
            u.cascade_count = 0;
            u.shadow_map_size = self.pbr_config.shadow_map_res as f32;
            u.pcf_kernel_size = match technique {
                ShadowTechnique::Hard => 1,
                ShadowTechnique::PCSS => 5,
                _ => 3,
            };
            u.depth_bias = 0.005;
            u.slope_bias = 0.01;
            u.peter_panning_offset = 0.001;
            u.evsm_positive_exp = 40.0;
            u.evsm_negative_exp = 5.0;
            u.technique_params = [0.0, 0.0, 0.0005, 1.0];
            u.debug_mode = debug_mode; // P6.2: Debug visualization from pbr_config
            
            // Set up default cascade far distances for cascade selection
            let terrain_scale = terrain.dimensions.0.max(terrain.dimensions.1) as f32;
            let base_distance = terrain_scale * 0.1;
            for (i, cascade) in u.cascades.iter_mut().enumerate() {
                cascade.far_distance = base_distance * (2.0_f32).powi(i as i32 + 1);
            }
            u
        };
        
        // Write CSM uniforms to buffer
        // Debug: log uniform values
        static CSM_ONCE: std::sync::Once = std::sync::Once::new();
        CSM_ONCE.call_once(|| {
            println!("[csm_uniforms] cascade_count={}, technique={}, shadow_map_size={}", 
                csm_uniforms.cascade_count, csm_uniforms.technique, csm_uniforms.shadow_map_size);
            for (i, c) in csm_uniforms.cascades.iter().enumerate().take(csm_uniforms.cascade_count as usize) {
                println!("[csm_uniforms] cascade[{}] far_distance={:.1}", i, c.far_distance);
            }
        });
        self.queue.write_buffer(csm_buffer, 0, bytemuck::cast_slice(&[csm_uniforms]));
        
        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.pbr_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain_viewer_pbr.bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(sv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(overlay_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(overlay_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&moment_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&moment_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: csm_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
    }
    
    /// Ensure fallback 1x1 white texture exists for when AO/sun_vis are disabled
    fn ensure_fallback_texture(&mut self) {
        if self.fallback_texture.is_some() {
            return;
        }
        
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.fallback_texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Write 1.0 (fully lit / no occlusion)
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&[1.0f32]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        
        self.fallback_texture_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.fallback_texture = Some(texture);
    }

    /// Dispatch compute passes for heightfield AO and sun visibility
    fn dispatch_heightfield_compute(&mut self, encoder: &mut wgpu::CommandEncoder, terrain_width: f32, sun_dir: glam::Vec3) {
        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };
        let (width, height) = terrain.dimensions;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let z_scale = terrain.z_scale;
        
        // Height AO compute pass
        if self.pbr_config.height_ao.enabled {
            if let (Some(ref pipeline), Some(ref layout), Some(ref uniform_buf), Some(ref ao_view), Some(ref sampler)) = (
                &self.height_ao_pipeline,
                &self.height_ao_bind_group_layout,
                &self.height_ao_uniform_buffer,
                &self.height_ao_view,
                &self.sampler_nearest,
            ) {
                let ao_width = (width as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
                let ao_height = (height as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
                
                // Update uniforms
                let uniforms: [f32; 16] = [
                    self.pbr_config.height_ao.directions as f32,
                    self.pbr_config.height_ao.steps as f32,
                    self.pbr_config.height_ao.max_distance,
                    self.pbr_config.height_ao.strength,
                    terrain_width / width as f32,
                    terrain_width / height as f32,
                    h_range * z_scale,
                    terrain.domain.0,
                    ao_width as f32,
                    ao_height as f32,
                    width as f32,
                    height as f32,
                    0.0, 0.0, 0.0, 0.0,
                ];
                self.queue.write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));
                
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.height_ao_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ao_view) },
                    ],
                });
                
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.height_ao_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((ao_width + 7) / 8, (ao_height + 7) / 8, 1);
            }
        }
        
        // Sun visibility compute pass
        if self.pbr_config.sun_visibility.enabled {
            if let (Some(ref pipeline), Some(ref layout), Some(ref uniform_buf), Some(ref sv_view), Some(ref sampler)) = (
                &self.sun_vis_pipeline,
                &self.sun_vis_bind_group_layout,
                &self.sun_vis_uniform_buffer,
                &self.sun_vis_view,
                &self.sampler_nearest,
            ) {
                let sv_width = (width as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                let sv_height = (height as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                
                // Update uniforms - sun_dir should point toward sun (negate light direction)
                let uniforms: [f32; 16] = [
                    self.pbr_config.sun_visibility.samples as f32,
                    self.pbr_config.sun_visibility.steps as f32,
                    self.pbr_config.sun_visibility.max_distance,
                    self.pbr_config.sun_visibility.softness,
                    terrain_width / width as f32,
                    terrain_width / height as f32,
                    h_range * z_scale,
                    terrain.domain.0,
                    sv_width as f32,
                    sv_height as f32,
                    width as f32,
                    height as f32,
                    sun_dir.x,
                    sun_dir.y,
                    sun_dir.z,
                    self.pbr_config.sun_visibility.bias,
                ];
                self.queue.write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));
                
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.sun_vis_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sv_view) },
                    ],
                });
                
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.sun_vis_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((sv_width + 7) / 8, (sv_height + 7) / 8, 1);
            }
        }
    }
}
