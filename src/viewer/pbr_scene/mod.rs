// src/viewer/pbr_scene/mod.rs
// PBR scene subsystem for the interactive viewer.
//
// A generic, viewer-owned scene of `PbrSceneObject`s rendered through the
// shared `Viewer::render` frame pipeline with the production instanced PBR
// renderer (`render::mesh_instanced::MeshInstancedRenderer` + `pipeline::pbr`
// + CSM + IBL). Frame flow per `Viewer::render_pbr_scene_stage`:
//   per-object instances/uniforms -> tightened CSM cascade depth passes ->
//   forward pass into the SSAA HDR target -> `viewer.pbr_scene.resolve`
//   (tent-filter compute) -> `viewer.pbr_scene.present` (Reinhard + sRGB)
//   into the frame output (and the snapshot target when requested).
// RELEVANT FILES: src/viewer/pbr_scene/reference.rs, src/render/mesh_instanced.rs,
// src/pipeline/pbr/constructor.rs, src/shaders/viewer_pbr_scene_resolve.wgsl,
// src/shaders/viewer_pbr_scene_present.wgsl

pub(crate) mod reference;

use std::sync::Arc;

use glam::{DVec3, Mat4, Quat, Vec3};
use wgpu::{Device, Queue};

use crate::core::error::{RenderError, RenderResult};
use crate::core::material::PbrMaterial;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedTexture,
};
use crate::render::mesh_instanced::VertexPN;
use crate::viewer::viewer_types::FrameCamera;
use crate::viewer::Viewer;

/// Route identity recorded in adjudication metadata for this render path.
#[cfg(feature = "extension-module")]
pub const PBR_SCENE_ROUTE: &str = "viewer.pbr_scene";

/// A baked texture map (row-major RGBA8) uploaded into a material slot.
#[derive(Clone, Debug)]
pub struct BakedMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Baked plane exit-radiance feed for an object's emissive slot.
/// `gi_ambient` is the constant ambient radiance (miss + occluder sky
/// terms), `gi_sph` the scene spheres used for hemisphere occlusion and
/// `gi_sph_albedo` their Lambertian albedo — exactly the lighting-uniform
/// contract `pipeline::pbr` consumes.
#[derive(Clone, Debug)]
pub struct GroundRadiance {
    pub map: BakedMap,
    pub gi_ambient: [f32; 3],
    pub gi_sph: [[f32; 4]; 3],
    pub gi_sph_albedo: [[f32; 4]; 3],
}

/// Optional baked texture feeds for one object.
#[derive(Clone, Debug, Default)]
pub struct PbrObjectMaps {
    pub occlusion: Option<BakedMap>,
    pub emissive: Option<BakedMap>,
    /// When present, the map is bound to the emissive texture slot but the
    /// material is flagged GROUND_RADIANCE (not EMISSIVE) and the lighting
    /// uniforms carry the bounce baseline + scene spheres.
    pub ground_radiance: Option<GroundRadiance>,
}

/// One renderable object: mesh geometry plus a world-space transform,
/// a PBR material, and optional baked maps.
#[derive(Clone)]
pub struct PbrSceneObject {
    pub vertices: Vec<VertexPN>,
    pub indices: Vec<u32>,
    /// World-space object origin; per-frame model matrices are
    /// `scale/rotation` around `frame.anchor.model_offset(translation)`.
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub material: PbrMaterial,
    pub maps: PbrObjectMaps,
}

/// World-space camera the scene was configured with. The viewer's frame
/// camera is set from these values by the load command; they are kept here
/// as the authoritative f64 pose the scene consumed.
#[derive(Clone, Copy, Debug)]
pub struct PbrSceneCamera {
    pub eye: DVec3,
    pub target: DVec3,
    pub up: Vec3,
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

/// Generic PBR scene description: objects, lighting, environment, SSAA and
/// the world-space points bounding every shadow caster+receiver.
#[derive(Clone)]
pub struct PbrSceneDesc {
    pub objects: Vec<PbrSceneObject>,
    /// Direction the sun light TRAVELS (from light toward scene).
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    /// Constant ambient/environment radiance.
    pub ambient: [f32; 3],
    /// Background clear color (miss radiance).
    pub sky: [f32; 3],
    pub exposure: f32,
    /// Supersample factor for the HDR target (per axis).
    pub ssaa: u32,
    /// World points bounding every shadow caster and receiver (sphere AABBs,
    /// their shadow-pool footprints, and the visible plane patch); each
    /// cascade's light projection is rebuilt around their light-space extent.
    pub shadow_focus: Vec<DVec3>,
    /// Declared scene camera; consumed by the extension-module cache key.
    /// The render itself always uses the live frame camera.
    #[cfg(feature = "extension-module")]
    pub camera: PbrSceneCamera,
}

#[cfg(feature = "extension-module")]
fn segment(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

/// A `PbrSceneDesc` realized on the GPU: one `MeshInstancedRenderer` per
/// object, the SSAA HDR target, the resolved HDR output, and the
/// resolve/present pipelines.
pub(crate) struct ViewerPbrScene {
    desc: PbrSceneDesc,
    meshes: Vec<crate::render::mesh_instanced::MeshInstancedRenderer>,
    width: u32,
    height: u32,
    sun_dir: Vec3,
    // Ownership anchors for hdr_ss_view/depth_ss_view (wgpu views do not
    // retain their texture); the views are the fields the passes use.
    _hdr_ss: TrackedTexture,
    hdr_ss_view: wgpu::TextureView,
    _depth_ss: TrackedTexture,
    depth_ss_view: wgpu::TextureView,
    hdr_resolved: Arc<TrackedTexture>,
    resolve_pipeline: wgpu::ComputePipeline,
    resolve_bind_group: wgpu::BindGroup,
    present_pipeline: wgpu::RenderPipeline,
    present_bind_group: wgpu::BindGroup,
    /// Scene values recorded at the last rendered frame (same keys/order as
    /// `ReferenceSceneDesc::metadata_fields` minus `spp`).
    consumed: Vec<(&'static str, f64)>,
}

#[cfg(all(
    feature = "enable-gpu-instancing",
    feature = "enable-pbr",
    feature = "enable-tbn"
))]
impl ViewerPbrScene {
    pub(crate) fn load(
        device: &Device,
        queue: &Queue,
        desc: PbrSceneDesc,
        width: u32,
        height: u32,
        present_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        if desc.ssaa == 0 {
            return Err(RenderError::Render(
                "pbr scene requires a non-zero SSAA factor".into(),
            ));
        }
        let valid_size = |size: u32| {
            size.checked_mul(desc.ssaa)
                .filter(|size| *size > 0 && *size <= device.limits().max_texture_dimension_2d)
        };
        let (Some(rw), Some(rh)) = (valid_size(width), valid_size(height)) else {
            return Err(RenderError::Render(
                "pbr scene size exceeds the device texture contract".into(),
            ));
        };
        for object in &desc.objects {
            if !object.translation.is_finite()
                || !object.rotation.is_finite()
                || !object.scale.is_finite()
                || object.scale.min_element() <= 0.0
            {
                return Err(RenderError::Render(
                    "pbr scene objects require finite transforms with positive scale".into(),
                ));
            }
        }

        let sun_dir = Vec3::from(desc.sun_direction).normalize();
        let mut meshes = Vec::with_capacity(desc.objects.len());
        for object in &desc.objects {
            let mut mesh = crate::render::mesh_instanced::MeshInstancedRenderer::new(
                device,
                wgpu::TextureFormat::Rgba32Float,
                Some(wgpu::TextureFormat::Depth32Float),
            )?;
            mesh.set_mesh(device, queue, &object.vertices, &object.indices)?;
            mesh.set_light(desc.sun_direction, desc.sun_intensity);
            mesh.enable_pbr(device, queue, object.material, desc.ambient, desc.sun_color)?;
            {
                let pbr = mesh.pbr_pipeline_mut().unwrap();
                if let Some(gr) = &object.maps.ground_radiance {
                    // The emissive slot carries the baked ground radiance
                    // field: GROUND_RADIANCE (not EMISSIVE) makes the shader
                    // sample it at the fragment's ground-projected point.
                    pbr.material.set_emissive_texture(
                        device,
                        queue,
                        &gr.map.data,
                        gr.map.width,
                        gr.map.height,
                    )?;
                    pbr.material.material.texture_flags &=
                        !crate::core::material::texture_flags::EMISSIVE;
                    pbr.material.material.texture_flags |=
                        crate::core::material::texture_flags::GROUND_RADIANCE;
                    pbr.lighting_uniforms.gi_ambient = gr.gi_ambient;
                    pbr.lighting_uniforms.gi_sph = gr.gi_sph;
                    pbr.lighting_uniforms.gi_sph_albedo = gr.gi_sph_albedo;
                }
                if let Some(occlusion) = &object.maps.occlusion {
                    pbr.material.set_occlusion_texture(
                        device,
                        queue,
                        &occlusion.data,
                        occlusion.width,
                        occlusion.height,
                    )?;
                }
                if let Some(emissive) = &object.maps.emissive {
                    pbr.material.set_emissive_texture(
                        device,
                        queue,
                        &emissive.data,
                        emissive.width,
                        emissive.height,
                    )?;
                }
                pbr.material.update_uniforms(queue);
                pbr.material.bind_group = None;
                let sampler = crate::pipeline::pbr::create_pbr_sampler(device);
                pbr.ensure_material_bind_group(device, queue, &sampler)?;
            }
            let pbr = mesh.pbr_pipeline_mut().unwrap();
            let manager = pbr.shadow_manager.as_mut().unwrap();
            manager.renderer_mut().uniforms.cascade_count = manager.config().csm.cascade_count;
            meshes.push(mesh);
        }

        let extent_ss = wgpu::Extent3d {
            width: rw,
            height: rh,
            depth_or_array_layers: 1,
        };
        let hdr_ss = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("viewer.pbr_scene.hdr_ss"),
                size: extent_ss,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let hdr_ss_view = hdr_ss.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_ss = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("viewer.pbr_scene.depth_ss"),
                size: extent_ss,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        )?;
        let depth_ss_view = depth_ss.create_view(&wgpu::TextureViewDescriptor::default());
        let hdr_resolved = Arc::new(tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("viewer.pbr_scene.hdr_resolved"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?);
        let hdr_resolved_view = hdr_resolved.create_view(&wgpu::TextureViewDescriptor::default());

        // --- viewer.pbr_scene.resolve (SSAA tent-filter compute) ---
        let resolve_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "viewer.pbr_scene.resolve.shader",
            include_str!("../../shaders/viewer_pbr_scene_resolve.wgsl"),
        );
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.pbr_scene.resolve.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.pbr_scene.resolve.pl"),
            bind_group_layouts: &[&resolve_bgl],
            push_constant_ranges: &[],
        });
        let resolve_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("viewer.pbr_scene.resolve"),
                layout: Some(&resolve_pl),
                module: &resolve_shader,
                entry_point: "cs_main",
            },
        );
        let resolve_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("viewer.pbr_scene.resolve.params"),
                contents: bytemuck::cast_slice(&[desc.ssaa, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.pbr_scene.resolve.bg"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_ss_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hdr_resolved_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resolve_params.as_entire_binding(),
                },
            ],
        });

        // --- viewer.pbr_scene.present (fullscreen Reinhard + sRGB) ---
        let present_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "viewer.pbr_scene.present.shader",
            include_str!("../../shaders/viewer_pbr_scene_present.wgsl"),
        );
        let present_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.pbr_scene.present.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let present_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.pbr_scene.present.pl"),
            bind_group_layouts: &[&present_bgl],
            push_constant_ranges: &[],
        });
        let present_pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("viewer.pbr_scene.present"),
                layout: Some(&present_pl),
                vertex: wgpu::VertexState {
                    module: &present_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &present_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: present_format,
                        blend: None,
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
            },
        );
        // sRGB encodes on write for *Srgb formats; manual encode otherwise.
        let present_params_data: [f32; 4] = [
            desc.exposure,
            if present_format.is_srgb() { 0.0 } else { 1.0 },
            0.0,
            0.0,
        ];
        let present_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("viewer.pbr_scene.present.params"),
                contents: bytemuck::cast_slice(&present_params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let present_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.pbr_scene.present.bg"),
            layout: &present_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_resolved_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: present_params.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            desc,
            meshes,
            width,
            height,
            sun_dir,
            _hdr_ss: hdr_ss,
            hdr_ss_view,
            _depth_ss: depth_ss,
            depth_ss_view,
            hdr_resolved,
            resolve_pipeline,
            resolve_bind_group,
            present_pipeline,
            present_bind_group,
            consumed: Vec::new(),
        })
    }

    /// The default CSM ortho expansion is terrain-scale, which quantizes a
    /// ~5u scene's shadow pools to a few texels. Rebuild each cascade's light
    /// projection around the actual scene extents in light space (the
    /// `shadow_focus` points) and shift the bias into the slope term.
    /// Identical numerics to the former offscreen adapter.
    fn tighten_shadow_bounds(
        pbr: &mut crate::pipeline::pbr::PbrPipelineWithShadows,
        queue: &Queue,
        focus: &[DVec3],
        sun_dir: Vec3,
        anchor: &crate::camera::Anchor,
    ) {
        let light_view = Mat4::look_to_rh(Vec3::ZERO, sun_dir, Vec3::Y);
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in focus {
            let q = light_view.transform_point3(anchor.to_render_vec3(*p));
            lo = lo.min(q);
            hi = hi.max(q);
        }
        if !lo.is_finite() || !hi.is_finite() {
            return;
        }
        let pad_xy = 1.0;
        let pad_z = 2.0;
        let manager = pbr.shadow_manager.as_mut().unwrap();
        {
            let renderer = manager.renderer_mut();
            let map_size = renderer.uniforms.shadow_map_size.max(1.0);
            for cascade in renderer.uniforms.cascades.iter_mut() {
                let proj = Mat4::orthographic_rh(
                    lo.x - pad_xy,
                    hi.x + pad_xy,
                    lo.y - pad_xy,
                    hi.y + pad_xy,
                    -hi.z - pad_z,
                    -lo.z + pad_z,
                );
                cascade.light_projection = proj.to_cols_array();
                cascade.light_view_proj = (proj * light_view).to_cols_array_2d();
                cascade.texel_size = (hi.x - lo.x + 2.0 * pad_xy) / map_size;
            }
            let depth_texel_ndc = 1.0 / map_size;
            renderer.uniforms.depth_bias = 1.0 * depth_texel_ndc;
            renderer.uniforms.slope_bias = 4.0 * depth_texel_ndc;
        }
        manager.upload_uniforms(queue);
    }

    /// Encode the scene's per-frame GPU work on the shared frame encoder:
    /// CSM cascade depth passes, the forward pass into `hdr_ss`, the tent
    /// resolve into `hdr_resolved`, and the present pass into `view`
    /// (plus the snapshot texture when requested). Returns the snapshot
    /// texture when one was rendered.
    #[allow(clippy::too_many_arguments)]
    fn render_stage(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: FrameCamera,
        want_snapshot: bool,
        out_format: wgpu::TextureFormat,
        mut timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    ) -> Result<Option<TrackedTexture>, RenderError> {
        let view_mat = frame.view();
        let proj = frame.projection(self.width, self.height);
        let near = frame.near;
        let far = frame.far;

        // Per-object model matrices + uniforms + tightened shadow bounds.
        for (object, mesh) in self.desc.objects.iter().zip(self.meshes.iter_mut()) {
            let model = Mat4::from_scale_rotation_translation(
                object.scale,
                object.rotation,
                frame.anchor.model_offset(object.translation),
            );
            mesh.set_view_proj(view_mat, proj);
            mesh.upload_instances_from_mat4(device, queue, &[model])?;
            let pbr = mesh.pbr_pipeline_mut().unwrap();
            pbr.update_shadows(queue, view_mat, proj, self.sun_dir, near, far);
            Self::tighten_shadow_bounds(
                pbr,
                queue,
                &self.desc.shadow_focus,
                self.sun_dir,
                &frame.anchor,
            );
            mesh.reset_shadow_draw_batch_uniforms();
        }

        let timing_scope = timing
            .as_mut()
            .and_then(|(t, label)| t.begin(encoder, label));

        // Shadow cascade passes: every caster into every receiver's map.
        for receiver in &self.meshes {
            let csm = receiver
                .pbr_pipeline()
                .unwrap()
                .shadow_manager
                .as_ref()
                .unwrap()
                .renderer();
            for (index, depth_view) in csm
                .shadow_map_views
                .iter()
                .take(csm.uniforms.cascade_count as usize)
                .enumerate()
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.pbr_scene.shadow_cascade"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                let light_matrix =
                    Mat4::from_cols_array_2d(&csm.uniforms.cascades[index].light_view_proj);
                for caster in &self.meshes {
                    caster.render_shadow(&mut pass, queue, light_matrix, 1);
                }
            }
        }

        // Forward pass into the SSAA HDR target, cleared to the scene sky.
        let sky = self.desc.sky;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.pbr_scene.forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_ss_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: sky[0] as f64,
                            g: sky[1] as f64,
                            b: sky[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_ss_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for mesh in &self.meshes {
                mesh.render(&mut pass, queue, 1);
            }
        }

        // SSAA tent-filter resolve into hdr_resolved.
        {
            let gx = self.width.div_ceil(8);
            let gy = self.height.div_ceil(8);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.pbr_scene.resolve.pass"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("viewer.pbr_scene.resolve.shader");
            cpass.set_pipeline(&self.resolve_pipeline);
            cpass.set_bind_group(0, &self.resolve_bind_group, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Present into the frame output.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.pbr_scene.present.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            crate::core::shader_registry::record_shader_use("viewer.pbr_scene.present.shader");
            pass.set_pipeline(&self.present_pipeline);
            pass.set_bind_group(0, &self.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Mirror postfx.rs: also present into an offscreen snapshot texture.
        let snapshot_tex = if want_snapshot {
            let snap_tex = tracked_create_texture(
                device,
                &wgpu::TextureDescriptor {
                    label: Some("viewer.pbr_scene.snapshot"),
                    size: wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: out_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )?;
            let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.pbr_scene.present.snapshot"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &snap_view,
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
                pass.set_pipeline(&self.present_pipeline);
                pass.set_bind_group(0, &self.present_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            Some(snap_tex)
        } else {
            None
        };

        if let Some((t, _)) = timing.as_mut() {
            t.end(encoder, timing_scope, self.meshes.len() as u32);
            t.resolve(encoder);
        }

        self.record_consumed(frame);
        Ok(snapshot_tex)
    }
}

#[cfg(not(all(
    feature = "enable-gpu-instancing",
    feature = "enable-pbr",
    feature = "enable-tbn"
)))]
impl ViewerPbrScene {
    pub(crate) fn load(
        _device: &Device,
        _queue: &Queue,
        _desc: PbrSceneDesc,
        _width: u32,
        _height: u32,
        _present_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        Err(RenderError::Render(
            "viewer pbr scene requires enable-gpu-instancing, enable-pbr and enable-tbn".into(),
        ))
    }
}

impl ViewerPbrScene {
    /// Rebuild the scene's GPU resources at a new output size from the
    /// retained desc (meshes, baked maps, lighting are unchanged).
    /// On error `self` keeps the old-size scene.
    pub(crate) fn resize(
        &mut self,
        device: &Device,
        queue: &Queue,
        width: u32,
        height: u32,
        present_format: wgpu::TextureFormat,
    ) -> Result<(), RenderError> {
        *self = Self::load(
            device,
            queue,
            self.desc.clone(),
            width,
            height,
            present_format,
        )?;
        Ok(())
    }

    /// Resolved HDR output texture (linear RGBA f32, `width*height`).
    pub(crate) fn hdr_target(&self) -> Arc<TrackedTexture> {
        Arc::clone(&self.hdr_resolved)
    }

    /// Scene values consumed by the last rendered frame, in the same
    /// keys/order as `ReferenceSceneDesc::metadata_fields` minus `spp`.
    /// The camera pose is the frame camera the frame was actually rendered
    /// with, not the declared scene camera.
    pub(crate) fn consumed_metadata(&self) -> &[(&'static str, f64)] {
        &self.consumed
    }

    fn record_consumed(&mut self, frame: FrameCamera) {
        let d = &self.desc;
        self.consumed = vec![
            ("cam_origin_x", frame.eye_world.x),
            ("cam_origin_y", frame.eye_world.y),
            ("cam_origin_z", frame.eye_world.z),
            ("cam_look_at_x", frame.target_world.x),
            ("cam_look_at_y", frame.target_world.y),
            ("cam_look_at_z", frame.target_world.z),
            ("fov_y_deg", frame.fov_deg as f64),
            ("exposure", d.exposure as f64),
            ("sun_dir_x", self.sun_dir.x as f64),
            ("sun_dir_y", self.sun_dir.y as f64),
            ("sun_dir_z", self.sun_dir.z as f64),
            ("sun_intensity", d.sun_intensity as f64),
            ("sun_color_r", d.sun_color[0] as f64),
            ("sun_color_g", d.sun_color[1] as f64),
            ("sun_color_b", d.sun_color[2] as f64),
            ("ambient_r", d.ambient[0] as f64),
            ("ambient_g", d.ambient[1] as f64),
            ("ambient_b", d.ambient[2] as f64),
            ("sky_r", d.sky[0] as f64),
            ("sky_g", d.sky[1] as f64),
            ("sky_b", d.sky[2] as f64),
            ("width", self.width as f64),
            ("height", self.height as f64),
        ];
    }

    /// Canonical ANAMNESIS key material:
    /// `(pipeline_descriptor_bytes, uniform_bytes, external_input_bytes)`.
    ///
    /// Descriptor: segment-encoded sources of every shader/pipeline module
    /// this scene executes. Uniform: canonical bytes of the desc scalars plus
    /// the frame camera and output shape. External: every object's vertices,
    /// indices, transform, material and baked maps.
    #[cfg(feature = "extension-module")]
    pub(crate) fn cache_key_parts(&self, frame: FrameCamera) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut descriptor =
            b"forge3d.production-instanced-pbr/v1;rgba32float;depth32float;ccw;back;less_equal;sample_count=1".to_vec();
        for source in [
            crate::shader_sources::pbr(),
            include_str!("../../shaders/mesh_instanced.wgsl").to_string(),
            include_str!("../../shaders/ibl_equirect.wgsl").to_string(),
            include_str!("../../shaders/ibl_prefilter.wgsl").to_string(),
            include_str!("../../shaders/ibl_brdf.wgsl").to_string(),
            include_str!("../../core/ibl.rs").to_string(),
            include_str!("../../core/ibl/irradiance.rs").to_string(),
            include_str!("../../core/ibl/prefilter.rs").to_string(),
            include_str!("../../core/ibl/brdf_lut.rs").to_string(),
            include_str!("../../core/ibl/environment.rs").to_string(),
            include_str!("../../core/ibl/constructor.rs").to_string(),
            include_str!("../../pipeline/pbr/constructor.rs").to_string(),
            include_str!("../../pipeline/pbr/material.rs").to_string(),
            include_str!("../../pipeline/pbr/rendering.rs").to_string(),
            include_str!("../../pipeline/pbr/ibl.rs").to_string(),
            include_str!("../../render/mesh_instanced.rs").to_string(),
            include_str!("mod.rs").to_string(),
            include_str!("reference.rs").to_string(),
            include_str!("../../shaders/viewer_pbr_scene_resolve.wgsl").to_string(),
            include_str!("../../shaders/viewer_pbr_scene_present.wgsl").to_string(),
        ] {
            segment(&mut descriptor, source.as_bytes());
        }

        let mut uniform_bytes = Vec::new();
        let d = &self.desc;
        for value in [
            d.sun_direction[0] as f64,
            d.sun_direction[1] as f64,
            d.sun_direction[2] as f64,
            d.sun_intensity as f64,
            d.sun_color[0] as f64,
            d.sun_color[1] as f64,
            d.sun_color[2] as f64,
            d.ambient[0] as f64,
            d.ambient[1] as f64,
            d.ambient[2] as f64,
            d.sky[0] as f64,
            d.sky[1] as f64,
            d.sky[2] as f64,
            d.exposure as f64,
            d.camera.eye.x,
            d.camera.eye.y,
            d.camera.eye.z,
            d.camera.target.x,
            d.camera.target.y,
            d.camera.target.z,
            frame.eye_world.x,
            frame.eye_world.y,
            frame.eye_world.z,
            frame.target_world.x,
            frame.target_world.y,
            frame.target_world.z,
            frame.up.x as f64,
            frame.up.y as f64,
            frame.up.z as f64,
            frame.fov_deg as f64,
            frame.near as f64,
            frame.far as f64,
            self.width as f64,
            self.height as f64,
            d.ssaa as f64,
        ] {
            segment(&mut uniform_bytes, &value.to_le_bytes());
        }

        let mut input_bytes = Vec::new();
        for object in &self.desc.objects {
            segment(&mut input_bytes, bytemuck::cast_slice(&object.vertices));
            segment(&mut input_bytes, bytemuck::cast_slice(&object.indices));
            segment(
                &mut input_bytes,
                bytemuck::cast_slice(&[
                    object.translation.x,
                    object.translation.y,
                    object.translation.z,
                ]),
            );
            segment(
                &mut input_bytes,
                bytemuck::cast_slice(&object.rotation.to_array()),
            );
            segment(
                &mut input_bytes,
                bytemuck::cast_slice(&object.scale.to_array()),
            );
            segment(&mut input_bytes, bytemuck::bytes_of(&object.material));
            if let Some(occlusion) = &object.maps.occlusion {
                segment(&mut input_bytes, &occlusion.width.to_le_bytes());
                segment(&mut input_bytes, &occlusion.height.to_le_bytes());
                segment(&mut input_bytes, &occlusion.data);
            }
            if let Some(emissive) = &object.maps.emissive {
                segment(&mut input_bytes, &emissive.width.to_le_bytes());
                segment(&mut input_bytes, &emissive.height.to_le_bytes());
                segment(&mut input_bytes, &emissive.data);
            }
            if let Some(gr) = &object.maps.ground_radiance {
                segment(&mut input_bytes, &gr.map.width.to_le_bytes());
                segment(&mut input_bytes, &gr.map.height.to_le_bytes());
                segment(&mut input_bytes, &gr.map.data);
                segment(&mut input_bytes, bytemuck::cast_slice(&gr.gi_ambient));
                segment(&mut input_bytes, bytemuck::cast_slice(&gr.gi_sph));
                segment(&mut input_bytes, bytemuck::cast_slice(&gr.gi_sph_albedo));
            }
        }
        (descriptor, uniform_bytes, input_bytes)
    }
}

impl Viewer {
    /// PBR scene stage: replaces the geometry + secondary paths when a scene
    /// is loaded (see `Viewer::render_frame`).
    pub(super) fn render_pbr_scene_stage(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    ) -> RenderResult<()> {
        let frame = self.current_frame_camera();
        let want_snapshot = self.snapshot_request.is_some();
        let out_format = self.config.format;
        let Some(scene) = self.pbr_scene.as_mut() else {
            return Ok(());
        };
        let snapshot_tex = scene.render_stage(
            &self.device,
            &self.queue,
            encoder,
            view,
            frame,
            want_snapshot,
            out_format,
            timing,
        )?;
        if let Some(tex) = snapshot_tex {
            self.pending_snapshot_tex = Some(tex);
        }
        Ok(())
    }

    /// Resolved linear-HDR pixels of the loaded PBR scene (row-major,
    /// `width * height * 4` f32), read back through `core::hdr`.
    pub fn read_pbr_scene_hdr(&self) -> RenderResult<Vec<f32>> {
        let Some(scene) = self.pbr_scene.as_ref() else {
            return Err(RenderError::Render(
                "no pbr scene is loaded (LoadReferenceScene first)".into(),
            ));
        };
        crate::core::hdr::read_hdr_texture(
            &self.device,
            &self.queue,
            scene.hdr_resolved.as_ref(),
            scene.width,
            scene.height,
            wgpu::TextureFormat::Rgba32Float,
        )
        .map_err(RenderError::Readback)
    }

    /// The scene's resolved HDR target (for ANAMNESIS restore).
    pub fn pbr_scene_hdr_target(&self) -> Option<Arc<TrackedTexture>> {
        self.pbr_scene.as_ref().map(|scene| scene.hdr_target())
    }

    /// Scene values consumed by the last rendered PBR-scene frame.
    pub fn pbr_scene_consumed_metadata(&self) -> Vec<(&'static str, f64)> {
        self.pbr_scene
            .as_ref()
            .map(|scene| scene.consumed_metadata().to_vec())
            .unwrap_or_default()
    }

    /// Record the consumed metadata for a frame that was NOT rendered
    /// (ANAMNESIS cache hit): the restored image's cache key describes
    /// exactly the scene desc plus this frame camera, so report those
    /// values as what the image consumed.
    #[cfg(feature = "extension-module")]
    pub(crate) fn pbr_scene_record_keyed_metadata(&mut self, frame: FrameCamera) {
        if let Some(scene) = self.pbr_scene.as_mut() {
            scene.record_consumed(frame);
        }
    }

    /// ANAMNESIS cache key parts for the loaded PBR scene.
    #[cfg(feature = "extension-module")]
    pub(crate) fn pbr_scene_cache_key_parts(
        &self,
        frame: FrameCamera,
    ) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        self.pbr_scene
            .as_ref()
            .map(|scene| scene.cache_key_parts(frame))
    }
}
