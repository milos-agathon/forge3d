// src/viewer/terrain/render.rs
// Terrain rendering for the interactive viewer

use super::scene::ViewerTerrainScene;

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
        
        // Initialize volumetrics pass if needed
        if needs_volumetrics && self.volumetrics_pass.is_none() {
            self.init_volumetrics_pass();
        }
        if needs_dof && self.dof_pass.is_none() {
            self.init_dof_pass();
        }
        if needs_dof {
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, self.surface_format);
            }
        }
        
        if (needs_post_process || needs_volumetrics) && self.post_process.is_none() {
            self.init_post_process();
        }
        if needs_post_process || needs_volumetrics {
            if let Some(ref mut pp) = self.post_process {
                let _ = pp.get_intermediate_view(width, height, self.surface_format);
            }
        }
        
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
        let proj = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        let view_proj = proj * view_mat;

        let sun_az = terrain.sun_azimuth_deg.to_radians();
        let sun_el = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        // Debug: print uniform values on first render
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!("[render] terrain_params: min_h={:.1}, h_range={:.1}, width={:.1}, z_scale={:.2}",
                terrain.domain.0, h_range, terrain_width, shader_z_scale);
            let max_y = if use_pbr {
                h_range * terrain.z_scale
            } else {
                terrain_width * legacy_z_scale * 0.001
            };
            println!("[render] Expected Y range: 0 to {:.1}", max_y);
            println!("[render] Camera center: ({:.1}, {:.1}, {:.1})", center.x, center.y, center.z);
        });

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
        
        // Ensure DoF textures are allocated before scene render if DoF is enabled
        if needs_dof {
            if let Some(ref mut dof) = self.dof_pass {
                // This allocates textures if they don't exist or size changed
                let _ = dof.get_input_view(width, height, self.surface_format);
            }
        }
        
        // Re-borrow terrain after mutable operations
        // Get render target after mutable borrows complete
        // Priority: DoF needs its own input -> post-process intermediate -> final view
        // P5: Volumetrics will be applied as a post-process step after main render
        let render_target: &wgpu::TextureView = if needs_dof {
            // When DoF enabled, render to DoF input texture first
            self.dof_pass.as_ref()
                .and_then(|dof| dof.input_view.as_ref())
                .unwrap_or(view)
        } else if needs_post_process || needs_volumetrics {
            // Need intermediate for post-process or volumetrics
            self.post_process.as_ref()
                .and_then(|pp| pp.intermediate_view.as_ref())
                .unwrap_or(view)
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
                if !stack.pipelines_ready() {
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
            if has_vector_overlays {
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
        
        // P5: Apply volumetrics pass if enabled (after main render, before DoF)
        if needs_volumetrics {
            if let Some(ref vol_pass) = self.volumetrics_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                let color_input = match self.post_process.as_ref().and_then(|pp| pp.intermediate_view.as_ref()) {
                    Some(v) => v,
                    None => {
                        eprintln!("[volumetrics] ERROR: intermediate_view is None, skipping volumetrics pass");
                        return false;
                    }
                };
                
                // Determine volumetrics output target:
                // - If DoF enabled: output to DoF input texture
                // - If lens effects enabled (but no DoF): output to final view (lens will read from intermediate)
                // - Otherwise: output to final view
                let vol_output = if needs_dof {
                    // Output to DoF input
                    self.dof_pass.as_ref()
                        .and_then(|dof| dof.input_view.as_ref())
                        .unwrap_or(view)
                } else {
                    // Output directly to final view
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
            // DoF uses its input_view as source (scene rendered there)
            // and outputs to either post-process intermediate or final view
            let dof_output = if needs_post_process {
                self.post_process.as_ref()
                    .and_then(|pp| pp.intermediate_view.as_ref())
                    .unwrap_or(view)
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
            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
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
        if self.terrain.is_none() {
            return None;
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
        let proj = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
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
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);

        // Option B: Prepare vector overlay stack if it has visible layers (for snapshot path)
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            stack.is_enabled() && stack.visible_layer_count() > 0
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
                if !stack.pipelines_ready() {
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
            if has_vector_overlays {
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

        let mut out_tex = color_tex;
        let mut out_view = color_view;

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
        let overlay_view = overlay_stack.composite_view()
            .expect("overlay composite_view should exist after ensure_fallback_texture");
        let overlay_sampler = overlay_stack.sampler();
        
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
