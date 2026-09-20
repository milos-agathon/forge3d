//! Physical ORBIS metric capture staged in the final terrain AOV submission.

use super::{TerrainGeometryProvider, TerrainRenderer, TerrainScene};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use anyhow::{anyhow, ensure, Result};
use glam::{DMat4, DVec2, DVec3, Vec2, Vec3};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

const VERTEX_WORDS: u64 = 9;
const SAMPLE_BYTES: u64 = 48;
const ORBIS_MICRO_STEP_M: f64 = 0.002;

pub(super) struct OrbisCaptureRequest {
    frame_anchor_ecef: DVec3,
    camera_ecef: DVec3,
    local_to_ecef: DMat4,
}

pub(super) struct OrbisPendingCapture {
    frames: Vec<OrbisFrameCapture>,
    validation_scopes: u32,
}

struct OrbisFrameCapture {
    // Keep every resource referenced by the encoded compute pass alive until
    // the enclosing render command buffer has been submitted and read back.
    // wgpu command encoders do not make dropping the last public handle before
    // submission a portable resource-lifetime contract across backends.
    _capture_buffers: Vec<TrackedBuffer>,
    _pipeline: wgpu::ComputePipeline,
    _bind_groups: Vec<wgpu::BindGroup>,
    projection_readback: TrackedBuffer,
    depth_readback: TrackedBuffer,
    coverage_readback: TrackedBuffer,
    selection_readback: TrackedBuffer,
    vertex_count: usize,
    width: u32,
    height: u32,
    depth_bytes_per_row: u32,
    coverage_bytes_per_row: u32,
    expected_view_proj: DMat4,
    frame_anchor_ecef: DVec3,
    local_to_ecef: DMat4,
    clip_near: f64,
    clip_far: f64,
    terrain_span: f64,
    topology_keys: Vec<(u32, u32, bool)>,
    topology_uvs: Vec<DVec2>,
    mesh_vertices: Vec<crate::terrain::clipmap::ClipmapVertex>,
    mesh_indices: Vec<u32>,
    draw_templates: Vec<crate::terrain::clipmap::gpu_lod::IndirectDrawTemplate>,
    variant_count: u32,
    selection_max_draw_count: u32,
    selection_provenance: crate::terrain::clipmap::gpu_lod::LodSelectionProvenance,
    submitted: bool,
    submission_serial: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct OrbisPhysicalCapture {
    pub max_vertex_jitter_px: f64,
    pub naive_max_vertex_jitter_px: f64,
    pub jitter_sample_count: u64,
    pub lod_crack_pixels: u64,
    pub crack_boundary_samples: u64,
    pub crack_depth_variance: f64,
}

impl TerrainRenderer {
    pub(crate) fn begin_orbis_descent(&mut self) {
        self.scene.orbis_descent_active = true;
    }

    pub(crate) fn end_orbis_descent(&mut self) {
        self.scene.orbis_descent_active = false;
    }

    pub(crate) fn orbis_descent_active(&self) -> bool {
        self.scene.orbis_descent_active
    }

    pub(crate) fn begin_orbis_physical_capture(&mut self, camera_ecef: DVec3) -> Result<()> {
        ensure!(camera_ecef.is_finite(), "ORBIS capture camera must be finite");
        ensure!(
            self.scene.orbis_capture_request.is_none(),
            "an ORBIS physical capture request is already pending"
        );
        let pending_frames = self
            .scene
            .orbis_pending_capture
            .as_ref()
            .map_or(0, |pending| pending.frames.len());
        ensure!(pending_frames < 2, "ORBIS physical capture already has two frames");
        let (frame_anchor_ecef, _) = self
            .scene
            .orbis_reanchor_state()
            .ok_or_else(|| anyhow!("ORBIS capture requires globe streaming reanchor state"))?;
        let local_to_ecef = crate::terrain::clipmap::globe::GlobeFrame::tangent_to_ecef(
            frame_anchor_ecef,
        )
        .ok_or_else(|| anyhow!("ORBIS capture could not build its ENU-to-ECEF basis"))?;
        self.scene.orbis_capture_request = Some(OrbisCaptureRequest {
            frame_anchor_ecef,
            camera_ecef,
            local_to_ecef,
        });
        Ok(())
    }

    pub(crate) fn finish_orbis_physical_capture(&mut self) -> Result<OrbisPhysicalCapture> {
        let pending = self
            .scene
            .orbis_pending_capture
            .take()
            .ok_or_else(|| anyhow!("ORBIS physical capture did not reach GPU submission"))?;
        pending.finish(self.scene.device.as_ref())
    }

    pub(crate) fn abort_orbis_physical_capture(&mut self) {
        self.scene.orbis_capture_request = None;
        if let Some(pending) = self.scene.orbis_pending_capture.take() {
            pending.drain_validation_scopes(self.scene.device.as_ref());
        }
    }

    pub(crate) fn allocation_owner(&self) -> crate::core::resource_tracker::AllocationOwner {
        self.scene.allocation_owner.clone()
    }

    pub(crate) fn adapter_evidence(&self) -> Result<(wgpu::AdapterInfo, bool)> {
        let info = self.scene.adapter.get_info();
        let (_, software_fallback) = crate::core::gpu::active_adapter_info()
            .filter(|(active, _)| active.name == info.name && active.backend == info.backend)
            .ok_or_else(|| anyhow!("active adapter identity does not match terrain renderer"))?;
        Ok((info, software_fallback))
    }

    pub(crate) fn orbis_reanchor_state(&self) -> Result<(DVec3, f64)> {
        self.scene
            .orbis_reanchor_state()
            .ok_or_else(|| anyhow!("ORBIS globe reanchor state is unavailable"))
    }
}

fn orbis_camera_match_quantization_bound(
    local_eye: Vec3,
    camera_radius: f32,
    camera_target: [f32; 3],
) -> Result<f64> {
    let operand_scale = camera_target
        .into_iter()
        .chain(local_eye.to_array())
        .chain(std::iter::once(camera_radius))
        .map(|value| f64::from(value.abs()))
        .fold(1.0_f64, f64::max);
    // The z-up probe subtracts two already-rounded f32 operands of similar
    // magnitude. Sterbenz makes that subtraction exact; the two conversions
    // contribute at most one f32 epsilon at the larger operand scale.
    let bound = f64::from(f32::EPSILON) * operand_scale + 1.0e-6;
    ensure!(bound.is_finite(), "ORBIS camera quantization bound is non-finite");
    ensure!(
        bound < ORBIS_MICRO_STEP_M * 0.5,
        "ORBIS render-camera quantization bound {bound:.9} m is not below half the 2 mm probe"
    );
    Ok(bound)
}

fn validate_orbis_camera_match(
    measured_camera: DVec3,
    requested_camera: DVec3,
    quantization_bound: f64,
) -> Result<()> {
    let mismatch = (measured_camera - requested_camera).length();
    ensure!(
        mismatch.is_finite() && mismatch <= quantization_bound,
        "ORBIS render camera mismatch {mismatch:.9} m exceeds derived f32 bound {quantization_bound:.9} m"
    );
    Ok(())
}

impl TerrainScene {
    pub(super) fn mark_orbis_capture_submitted(&mut self) {
        let Some(pending) = self.orbis_pending_capture.as_mut() else { return };
        let next = pending.frames.iter().map(|frame| frame.submission_serial).max().unwrap_or(0) + 1;
        if let Some(frame) = pending.frames.last_mut().filter(|frame| !frame.submitted) {
            frame.submitted = true;
            frame.submission_serial = next;
        }
    }

    pub(super) fn encode_orbis_capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_texture: &wgpu::Texture,
        coverage_texture: &wgpu::Texture,
        width: u32,
        height: u32,
        params: &crate::terrain::render_params::TerrainRenderParams,
        request: OrbisCaptureRequest,
        terrain_bind_group: &wgpu::BindGroup,
        selection: super::geometry::StagedLodSelection,
    ) -> Result<OrbisPendingCapture> {
        let TerrainGeometryProvider::Clipmap {
            vertex_buffer,
            index_buffer,
            cpu_mesh,
            lod_resources,
            draw_templates,
            variant_count,
            ..
        } = self.geometry_provider()?
        else {
            return Err(anyhow!("ORBIS capture requires submitted clipmap geometry"));
        };
        ensure!(!cpu_mesh.vertices.is_empty(), "ORBIS clipmap has no vertices");
        ensure!(width > 0 && height > 0, "ORBIS capture viewport is empty");

        let vertex_bytes = cpu_mesh.vertices.len() as u64 * VERTEX_WORDS * 4;
        let copied_vertices = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.submitted_vertices"),
                size: vertex_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let index_bytes = cpu_mesh.indices.len() as u64 * 4;
        let copied_indices = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.submitted_indices"),
                size: index_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let (local_eye, _, _) = Self::build_camera_matrices(params);
        let measured_camera = request.frame_anchor_ecef
            + request
                .local_to_ecef
                .transform_vector3(DVec3::from(local_eye));
        let camera_match_bound = orbis_camera_match_quantization_bound(
            local_eye,
            params.cam_radius,
            params.cam_target,
        )?;
        validate_orbis_camera_match(measured_camera, request.camera_ecef, camera_match_bound)?;
        let (expected_view, expected_projection) = build_camera_matrices_f64(params);
        let ecef_to_local = request.local_to_ecef.transpose();
        let absolute_view_proj = expected_projection
            * expected_view
            * ecef_to_local
            * DMat4::from_translation(-request.frame_anchor_ecef);
        let mut uniforms = Vec::with_capacity(40);
        uniforms.extend_from_slice(&absolute_view_proj.as_mat4().to_cols_array());
        uniforms.extend_from_slice(&request.local_to_ecef.as_mat4().to_cols_array());
        uniforms.extend_from_slice(&request.frame_anchor_ecef.as_vec3().extend(0.0).to_array());
        uniforms.extend_from_slice(&[width as f32, height as f32, cpu_mesh.vertices.len() as f32, 0.0]);
        debug_assert_eq!(uniforms.len(), 40);
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("orbis.metric.uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let output_size = cpu_mesh.vertices.len() as u64 * SAMPLE_BYTES;
        let output = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.projection_output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let projection_readback = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.projection_readback"),
                size: output_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;

        let depth_bytes_per_row = (width * 4).div_ceil(256) * 256;
        let depth_size = u64::from(depth_bytes_per_row) * u64::from(height);
        let depth_readback = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.depth_coverage_readback"),
                size: depth_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let coverage_bytes_per_row = width.div_ceil(256) * 256;
        let coverage_size = u64::from(coverage_bytes_per_row) * u64::from(height);
        let coverage_readback = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.coverage_readback"),
                size: coverage_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;

        let header_bytes = std::mem::size_of::<crate::terrain::clipmap::gpu_lod::OutputHeader>() as u64;
        let tile_bytes = u64::from(lod_resources.max_draw_count)
            * std::mem::size_of::<crate::terrain::clipmap::gpu_lod::TileInfo>() as u64;
        let indirect_bytes = u64::from(lod_resources.max_draw_count)
            * std::mem::size_of::<crate::terrain::clipmap::gpu_lod::DrawIndexedIndirectArgs>() as u64;
        let instance_bytes = u64::from(lod_resources.max_draw_count)
            * std::mem::size_of::<crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance>() as u64;
        let selection_size = header_bytes + tile_bytes + indirect_bytes + instance_bytes;
        let selection_readback = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("orbis.metric.exact_indirect_selection"),
                size: selection_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let topology_keys = cpu_mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (index as u32, vertex.ring_index(), vertex.is_skirt()))
            .collect();
        let topology_uvs = cpu_mesh
            .vertices
            .iter()
            .map(|vertex| DVec2::new(f64::from(vertex.uv[0]), f64::from(vertex.uv[1])))
            .collect();
        let selected_template_indices = draw_templates.iter().any(|template| template.index_count > 0);
        ensure!(
            selected_template_indices,
            "ORBIS selected templates contain no indices"
        );

        // Everything that can return an ordinary Rust error is complete before
        // opening this scope.  It remains live through submission/readback, so
        // a validation failure cannot be silently converted into zero metrics.
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("orbis.metric.probe.shader"),
            source: wgpu::ShaderSource::Wgsl(Self::preprocess_terrain_shader(self.device.as_ref(), false).into()),
        });
        let probe_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("orbis.metric.probe.layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_layout_entry(3, true),
                storage_layout_entry(4, true),
                storage_layout_entry(5, true),
            ],
        });
        let empty_layouts = (0..6)
            .map(|_| self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("orbis.metric.empty.layout"),
                entries: &[],
            }))
            .collect::<Vec<_>>();
        let empty_bind_groups = empty_layouts
            .iter()
            .map(|layout| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("orbis.metric.empty.bind_group"),
                    layout,
                    entries: &[],
                })
            })
            .collect::<Vec<_>>();
        let mut layouts = Vec::with_capacity(8);
        layouts.push(&self.bind_group_layout);
        layouts.extend(empty_layouts.iter());
        layouts.push(&probe_layout);
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("orbis.metric.probe.pipeline_layout"),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });
        let pipeline = match crate::core::shader_registry::try_create_compute_pipeline_scoped(
            self.device.as_ref(),
            &wgpu::ComputePipelineDescriptor {
                label: Some("orbis.metric.probe.pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "orbis_metric_probe",
            },
        ) {
            Ok(pipeline) => pipeline,
            Err(message) => {
                // Balance the outer validation scope opened above so the next
                // capture's error accounting is not misattributed.
                let _ = pollster::block_on(self.device.pop_error_scope());
                return Err(anyhow!(
                    "ORBIS metric probe pipeline validation failed: {message}"
                ));
            }
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orbis.metric.probe.bind_group"),
            layout: &probe_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: copied_vertices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: lod_resources.output_header.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: lod_resources.indirect_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: copied_indices.as_entire_binding() },
            ],
        });
        encoder.copy_buffer_to_buffer(vertex_buffer, 0, &copied_vertices, 0, vertex_bytes);
        encoder.copy_buffer_to_buffer(index_buffer, 0, &copied_indices, 0, index_bytes);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orbis.metric.probe"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("orbis.metric.probe.shader");
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, terrain_bind_group, &[]);
            for (index, empty) in empty_bind_groups.iter().enumerate() {
                pass.set_bind_group(index as u32 + 1, empty, &[]);
            }
            pass.set_bind_group(7, &bind_group, &[]);
            // One invocation owns one output vertex slot. The shader scans the
            // exact selected indirect spans for membership, avoiding duplicate
            // indexed-triangle writers to the same storage element.
            pass.dispatch_workgroups((cpu_mesh.vertices.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &projection_readback, 0, output_size);
        encoder.copy_buffer_to_buffer(
            &lod_resources.output_header,
            0,
            &selection_readback,
            0,
            header_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &lod_resources.output_tiles,
            0,
            &selection_readback,
            header_bytes,
            tile_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &lod_resources.indirect_buffer,
            0,
            &selection_readback,
            header_bytes + tile_bytes,
            indirect_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &lod_resources.instance_buffer,
            0,
            &selection_readback,
            header_bytes + tile_bytes + indirect_bytes,
            instance_bytes,
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::ImageCopyBuffer {
                buffer: &depth_readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(depth_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: coverage_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &coverage_readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(coverage_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let frame = OrbisFrameCapture {
            _capture_buffers: vec![copied_vertices, copied_indices, output, uniform_buffer],
            _pipeline: pipeline,
            _bind_groups: empty_bind_groups
                .into_iter()
                .chain(std::iter::once(bind_group))
                .collect(),
            projection_readback,
            depth_readback,
            coverage_readback,
            selection_readback,
            vertex_count: cpu_mesh.vertices.len(),
            width,
            height,
            depth_bytes_per_row,
            coverage_bytes_per_row,
            expected_view_proj: expected_projection * expected_view,
            frame_anchor_ecef: request.frame_anchor_ecef,
            local_to_ecef: request.local_to_ecef,
            clip_near: f64::from(params.clip.0),
            clip_far: f64::from(params.clip.1),
            terrain_span: f64::from(params.terrain_span),
            topology_keys,
            topology_uvs,
            mesh_vertices: cpu_mesh.vertices.clone(),
            mesh_indices: cpu_mesh.indices.clone(),
            draw_templates: draw_templates.clone(),
            variant_count: *variant_count,
            selection_max_draw_count: lod_resources.max_draw_count,
            selection_provenance: selection.provenance,
            submitted: false,
            submission_serial: 0,
        };
        let mut pending = self.orbis_pending_capture.take().unwrap_or(OrbisPendingCapture {
            frames: Vec::with_capacity(2),
            validation_scopes: 0,
        });
        pending.frames.push(frame);
        pending.validation_scopes += 1;
        Ok(pending)
    }
}

impl OrbisPendingCapture {
    fn finish(mut self, device: &wgpu::Device) -> Result<OrbisPhysicalCapture> {
        ensure!(self.frames.len() == 2, "ORBIS jitter capture requires exactly two submitted frames");
        ensure!(
            self.frames[0].submitted && self.frames[1].submitted
                && self.frames[0].submission_serial > 0
                && self.frames[1].submission_serial > self.frames[0].submission_serial,
            "ORBIS capture lacks two distinct successful render submissions"
        );
        device.poll(wgpu::Maintain::Wait);
        let validation = self.pop_validation_errors(device);
        if !validation.is_empty() {
            return Err(anyhow!("ORBIS GPU validation failed: {}", validation.join("; ")));
        }
        let second = self.frames.pop().unwrap();
        let first = self.frames.pop().unwrap();
        ensure!(
            (second.frame_anchor_ecef - first.frame_anchor_ecef).length() > 0.0,
            "ORBIS two-frame capture did not cross a physical reanchor"
        );
        let first_projection = read_projection(device, &first)?;
        let second_projection = read_projection(device, &second)?;
        let first_selection = read_exact_selection(device, &first)?;
        let second_selection = read_exact_selection(device, &second)?;
        ensure!(
            first.selection_provenance != second.selection_provenance,
            "ORBIS two submitted selections reused one provenance"
        );
        ensure!(first_selection == second_selection, "ORBIS selected indirect LOD set changed across 2 mm probe");
        let first_selected = exact_selected_vertex_indices(
            &first.mesh_indices,
            first.vertex_count,
            &first_selection,
        )?;
        let second_selected = exact_selected_vertex_indices(
            &second.mesh_indices,
            second.vertex_count,
            &second_selection,
        )?;
        ensure!(
            first_selected == second_selected,
            "ORBIS exact selected vertex identities changed across the 2 mm probe"
        );
        require_selected_jitter_samples(&first_selected, 32)?;
        for &index in &first_selected {
            ensure!(
                first.topology_keys.get(index) == second.topology_keys.get(index),
                "ORBIS selected topology key changed at vertex {index}"
            );
            let (Some(a), Some(b)) =
                (first.topology_uvs.get(index), second.topology_uvs.get(index))
            else {
                return Err(anyhow!("ORBIS selected UV identity {index} is out of range"));
            };
            ensure!(
                (*a - *b).length() <= 1.0e-7,
                "ORBIS selected UV identity changed at vertex {index}"
            );
        }

        let viewport = DVec2::new(second.width as f64 * 0.5, second.height as f64 * 0.5);
        let mut max_jitter = 0.0_f64;
        let mut max_naive = 0.0_f64;
        let mut sample_count = 0_u64;
        let mut rejected_skirt = 0_u64;
        let mut rejected_nonpositive_w = 0_u64;
        let mut rejected_nonfinite = 0_u64;
        let mut rejected_edge = 0_u64;
        let mut rejected_expected_projection = 0_u64;
        let mut projected_vertices = vec![None; second.vertex_count];
        for &index in &second_selected {
            let a = first_projection[index];
            let b = second_projection[index];
            if a.skirt || b.skirt {
                rejected_skirt += 1;
                continue;
            }
            if a.production_w <= 0.0 || b.production_w <= 0.0 {
                rejected_nonpositive_w += 1;
                continue;
            }
            if ![a.production, b.production, a.naive, b.naive].iter().all(|p| p.is_finite()) {
                rejected_nonfinite += 1;
                continue;
            }
            projected_vertices[index] = Some(Vec2::new(
                (b.production.x as f32 * 0.5 + 0.5) * second.width as f32,
                (0.5 - b.production.y as f32 * 0.5) * second.height as f32,
            ));
            if a.production.abs().max_element() > 0.98 || b.production.abs().max_element() > 0.98 {
                rejected_edge += 1;
                continue;
            }
            let world_a =
                first.frame_anchor_ecef + first.local_to_ecef.transform_vector3(a.local);
            let world_b =
                second.frame_anchor_ecef + second.local_to_ecef.transform_vector3(b.local);
            let cpu_a = first
                .mesh_vertices
                .get(index)
                .ok_or_else(|| anyhow!("ORBIS first CPU vertex {index} is out of range"))?;
            let cpu_b = second
                .mesh_vertices
                .get(index)
                .ok_or_else(|| anyhow!("ORBIS second CPU vertex {index} is out of range"))?;
            let base_a = Vec3::from(cpu_a.position).as_dvec3();
            let base_b = Vec3::from(cpu_b.position).as_dvec3();
            let base_world_a =
                first.frame_anchor_ecef + first.local_to_ecef.transform_vector3(base_a);
            let base_world_b =
                second.frame_anchor_ecef + second.local_to_ecef.transform_vector3(base_b);
            let canonical = canonical_selected_world_identity(
                world_a,
                world_b,
                a.local,
                b.local,
                base_a,
                base_b,
                base_world_a,
                base_world_b,
            )
            .map_err(|error| {
                anyhow!(
                    "ORBIS tracked ground vertex {index}: {error}; cpu_a={cpu_a:?}, cpu_b={cpu_b:?}",
                )
            })?;
            let expected_local_0 = first
                .local_to_ecef
                .transpose()
                .transform_vector3(canonical - first.frame_anchor_ecef);
            let expected_local_1 = second
                .local_to_ecef
                .transpose()
                .transform_vector3(canonical - second.frame_anchor_ecef);
            let Some(expected_0) = project_d(first.expected_view_proj, expected_local_0) else {
                rejected_expected_projection += 1;
                continue;
            };
            let Some(expected_1) = project_d(second.expected_view_proj, expected_local_1) else {
                rejected_expected_projection += 1;
                continue;
            };
            let expected_delta = (expected_1 - expected_0) * viewport;
            let production_delta = (b.production - a.production) * viewport;
            let naive_delta = (b.naive - a.naive) * viewport;
            max_jitter = max_jitter.max((production_delta - expected_delta).length());
            max_naive = max_naive.max((naive_delta - expected_delta).length());
            sample_count += 1;
        }
        ensure!(
            sample_count >= 32,
            "ORBIS jitter probe had only {sample_count} stable in-frustum non-skirt samples (selected={}, skirt={rejected_skirt}, nonpositive_w={rejected_nonpositive_w}, nonfinite={rejected_nonfinite}, edge={rejected_edge}, expected_projection_missing={rejected_expected_projection}, accepted={sample_count})",
            second_selected.len(),
        );

        let seam_pairs = selected_mating_boundaries(
            &second,
            &second_selection,
            &second_projection,
            &projected_vertices,
        )?;

        let depth = map_readback(device, &second.depth_readback)?;
        let coverage = map_readback(device, &second.coverage_readback)?;
        if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
            let mut depth_vis = format!("P2\n{} {}\n65535\n", second.width, second.height);
            for y in 0..second.height {
                for x in 0..second.width {
                    let offset =
                        y as usize * second.depth_bytes_per_row as usize + x as usize * 4;
                    let d = f32::from_le_bytes(
                        depth[offset..offset + 4].try_into().unwrap(),
                    );
                    let v = (d.clamp(0.0, 1.0) * 65535.0) as u32;
                    depth_vis.push_str(&format!("{v} "));
                }
                depth_vis.push('\n');
            }
            let _ = std::fs::write("orbis-debug-depth.pgm", depth_vis);
            let mut depth_raw = Vec::with_capacity(
                second.height as usize * second.width as usize * 4,
            );
            for y in 0..second.height {
                for x in 0..second.width {
                    let offset =
                        y as usize * second.depth_bytes_per_row as usize + x as usize * 4;
                    depth_raw.extend_from_slice(&depth[offset..offset + 4]);
                }
            }
            let _ = std::fs::write("orbis-debug-depth.f32", depth_raw);
            let mut cov_vis = format!("P2\n{} {}\n255\n", second.width, second.height);
            for y in 0..second.height {
                for x in 0..second.width {
                    let offset =
                        y as usize * second.coverage_bytes_per_row as usize + x as usize;
                    cov_vis.push_str(&format!("{} ", coverage[offset]));
                }
                cov_vis.push('\n');
            }
            let _ = std::fs::write("orbis-debug-coverage.pgm", cov_vis);
        }
        let world_units_per_pixel = second.terrain_span
            / f64::from(second.width.min(second.height).max(1));
        let (cracks, boundary_samples, variance) = analyze_paired_cracks(
            &depth, &coverage, second.width, second.height,
            second.depth_bytes_per_row, second.coverage_bytes_per_row,
            second.clip_near, second.clip_far, world_units_per_pixel,
            &seam_pairs,
        )?;
        Ok(OrbisPhysicalCapture {
            max_vertex_jitter_px: max_jitter,
            naive_max_vertex_jitter_px: max_naive,
            jitter_sample_count: sample_count,
            lod_crack_pixels: cracks,
            crack_boundary_samples: boundary_samples,
            crack_depth_variance: variance,
        })
    }

    fn pop_validation_errors(&mut self, device: &wgpu::Device) -> Vec<String> {
        let mut errors = Vec::new();
        while self.validation_scopes > 0 {
            if let Some(error) = pollster::block_on(device.pop_error_scope()) {
                errors.push(error.to_string());
            }
            self.validation_scopes -= 1;
        }
        errors
    }

    fn drain_validation_scopes(mut self, device: &wgpu::Device) {
        device.poll(wgpu::Maintain::Wait);
        let _ = self.pop_validation_errors(device);
    }
}

#[derive(Clone, Copy)]
struct ProjectionSample {
    production: DVec2,
    production_w: f32,
    naive: DVec2,
    ndc_depth: f32,
    local: DVec3,
    skirt: bool,
}

fn read_projection(device: &wgpu::Device, frame: &OrbisFrameCapture) -> Result<Vec<ProjectionSample>> {
    let bytes = map_readback(device, &frame.projection_readback)?;
    ensure!(bytes.len() == frame.vertex_count * SAMPLE_BYTES as usize, "ORBIS projection readback length mismatch");
    Ok(bytes.chunks_exact(SAMPLE_BYTES as usize).map(|sample| {
        let value = |index: usize| f32::from_le_bytes(sample[index * 4..index * 4 + 4].try_into().unwrap());
        ProjectionSample {
            production: DVec2::new(f64::from(value(0)), f64::from(value(1))),
            production_w: value(2),
            skirt: value(3) != 0.0,
            naive: DVec2::new(f64::from(value(4)), f64::from(value(5))),
            ndc_depth: value(7),
            local: DVec3::new(f64::from(value(8)), f64::from(value(9)), f64::from(value(10))),
        }
    }).collect())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectedDraw {
    tile_id: u32,
    selected_lod: u32,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
}

fn canonicalize_selected_draws(mut selected: Vec<SelectedDraw>) -> Result<Vec<SelectedDraw>> {
    selected.sort_unstable_by_key(|draw| {
        (
            draw.tile_id,
            draw.selected_lod,
            draw.first_index,
            draw.index_count,
            draw.base_vertex,
        )
    });
    for pair in selected.windows(2) {
        ensure!(
            (pair[0].tile_id, pair[0].selected_lod)
                != (pair[1].tile_id, pair[1].selected_lod),
            "ORBIS exact indirect selection contains duplicate tile/LOD identity ({}, {})",
            pair[0].tile_id,
            pair[0].selected_lod,
        );
    }
    Ok(selected)
}

fn read_exact_selection(device: &wgpu::Device, frame: &OrbisFrameCapture) -> Result<Vec<SelectedDraw>> {
    use crate::terrain::clipmap::gpu_lod::{ClipmapDrawInstance, DrawIndexedIndirectArgs, OutputHeader, TileInfo};
    let bytes = map_readback(device, &frame.selection_readback)?;
    let header_size = std::mem::size_of::<OutputHeader>();
    let tile_size = std::mem::size_of::<TileInfo>();
    let arg_size = std::mem::size_of::<DrawIndexedIndirectArgs>();
    let instance_size = std::mem::size_of::<ClipmapDrawInstance>();
    let header = bytemuck::pod_read_unaligned::<OutputHeader>(&bytes[..header_size]);
    ensure!(header.visible_count > 0 && header.visible_count <= frame.selection_max_draw_count, "ORBIS exact indirect selection count is empty/out of range");
    let tile_base = header_size;
    let arg_base = tile_base + frame.selection_max_draw_count as usize * tile_size;
    let instance_base = arg_base + frame.selection_max_draw_count as usize * arg_size;
    let mut selected = Vec::with_capacity(header.visible_count as usize);
    for index in 0..header.visible_count as usize {
        let tile = bytemuck::pod_read_unaligned::<TileInfo>(&bytes[tile_base + index * tile_size..tile_base + (index + 1) * tile_size]);
        let args = bytemuck::pod_read_unaligned::<DrawIndexedIndirectArgs>(&bytes[arg_base + index * arg_size..arg_base + (index + 1) * arg_size]);
        let instance = bytemuck::pod_read_unaligned::<ClipmapDrawInstance>(&bytes[instance_base + index * instance_size..instance_base + (index + 1) * instance_size]);
        ensure!(tile.tile_id < frame.selection_max_draw_count && tile.selected_lod < frame.variant_count, "ORBIS exact indirect tile/LOD is out of range");
        let template = frame.draw_templates[(tile.tile_id * frame.variant_count + tile.selected_lod) as usize];
        ensure!(args.index_count == template.index_count && args.first_index == template.first_index && args.base_vertex == template.base_vertex && args.instance_count == 1, "ORBIS indirect arguments do not match selected template");
        ensure!(args.first_instance == 0 || args.first_instance == index as u32, "ORBIS indirect first-instance provenance is invalid");
        validate_selected_instance(&instance, tile.tile_id, tile.selected_lod)?;
        selected.push(SelectedDraw {
            tile_id: tile.tile_id,
            selected_lod: tile.selected_lod,
            first_index: template.first_index,
            index_count: template.index_count,
            base_vertex: template.base_vertex,
        });
    }
    canonicalize_selected_draws(selected)
}

fn validate_selected_instance(
    instance: &crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance,
    tile_id: u32,
    selected_lod: u32,
) -> Result<()> {
    ensure!(
        instance.tile_id_lod == [tile_id, selected_lod],
        "ORBIS draw instance does not match selected tile/LOD"
    );
    ensure!(
        instance.transform == glam::Mat4::IDENTITY.to_cols_array_2d(),
        "ORBIS metric probe requires bitwise-identity selected instance transforms"
    );
    Ok(())
}

fn exact_selected_vertex_indices(
    mesh_indices: &[u32],
    vertex_count: usize,
    selected: &[SelectedDraw],
) -> Result<BTreeSet<usize>> {
    let mut vertices = BTreeSet::new();
    for draw in selected {
        let start = draw.first_index as usize;
        let end = start
            .checked_add(draw.index_count as usize)
            .ok_or_else(|| anyhow!("ORBIS selected index span overflow"))?;
        ensure!(end <= mesh_indices.len(), "ORBIS selected index span is out of range");
        for &raw in &mesh_indices[start..end] {
            let resolved = i64::from(raw) + i64::from(draw.base_vertex);
            ensure!(
                resolved >= 0 && resolved < vertex_count as i64,
                "ORBIS selected index resolves outside the submitted vertex buffer"
            );
            vertices.insert(resolved as usize);
        }
    }
    ensure!(!vertices.is_empty(), "ORBIS exact selected vertex set is empty");
    Ok(vertices)
}

fn require_selected_jitter_samples(selected: &BTreeSet<usize>, minimum: usize) -> Result<()> {
    ensure!(
        selected.len() >= minimum,
        "ORBIS exact selected set has only {} vertices; at least {minimum} required",
        selected.len()
    );
    Ok(())
}

fn canonical_selected_world_identity(
    world_a: DVec3,
    world_b: DVec3,
    local_a: DVec3,
    local_b: DVec3,
    base_a: DVec3,
    base_b: DVec3,
    base_world_a: DVec3,
    base_world_b: DVec3,
) -> Result<DVec3> {
    ensure!(
        world_a.is_finite()
            && world_b.is_finite()
            && local_a.is_finite()
            && local_b.is_finite()
            && base_a.is_finite()
            && base_b.is_finite()
            && base_world_a.is_finite()
            && base_world_b.is_finite(),
        "selected world identity is non-finite"
    );
    let quantization_bound = local_vector_quantization_bound(
        local_a,
        local_b,
        base_a,
        base_b,
        base_world_a,
        base_world_b,
    )?;
    ensure!(
        quantization_bound < 0.001,
        "selected local-coordinate quantization bound {quantization_bound:.9} m is not below half the 2 mm probe"
    );
    let drift = (world_b - world_a).length();
    ensure!(
        drift <= quantization_bound,
        "selected ECEF identity drift {drift:.9} m exceeds quantization bound {quantization_bound:.9} m; local_a={local_a:?}, local_b={local_b:?}, world_a={world_a:?}, world_b={world_b:?}"
    );
    Ok((world_a + world_b) * 0.5)
}

fn validate_mating_world_identity(
    world_a: DVec3,
    world_b: DVec3,
    local_a: DVec3,
    local_b: DVec3,
    base_a: DVec3,
    base_b: DVec3,
    base_world_a: DVec3,
    base_world_b: DVec3,
) -> Result<()> {
    ensure!(
        world_a.is_finite()
            && world_b.is_finite()
            && local_a.is_finite()
            && local_b.is_finite()
            && base_a.is_finite()
            && base_b.is_finite()
            && base_world_a.is_finite()
            && base_world_b.is_finite(),
        "mating world identity is non-finite"
    );
    let quantization_bound = local_vector_quantization_bound(
        local_a,
        local_b,
        base_a,
        base_b,
        base_world_a,
        base_world_b,
    )?;
    let drift = (world_b - world_a).length();
    ensure!(
        drift <= quantization_bound,
        "mating ECEF identity drift {drift:.9} m exceeds quantization bound {quantization_bound:.9} m"
    );
    Ok(())
}

fn f32_rounding_ulp(value: f64) -> f64 {
    let magnitude = value.abs();
    if magnitude < f64::from(f32::MIN_POSITIVE) {
        f64::from(f32::from_bits(1))
    } else {
        magnitude.log2().floor().exp2() * f64::from(f32::EPSILON)
    }
}

fn f32_pair_rounding_bound(a: DVec3, b: DVec3) -> f64 {
    let component = |x, y| (f32_rounding_ulp(x) + f32_rounding_ulp(y)) * 0.5;
    DVec3::new(
        component(a.x, b.x),
        component(a.y, b.y),
        component(a.z, b.z),
    )
    .length()
}

fn local_vector_quantization_bound(
    local_a: DVec3,
    local_b: DVec3,
    base_a: DVec3,
    base_b: DVec3,
    base_world_a: DVec3,
    base_world_b: DVec3,
) -> Result<f64> {
    // Validate the first stored-f32 stage from observed ECEF drift, then add
    // the final shader displacement's independent half-ULP endpoint bound.
    let base_drift = (base_world_b - base_world_a).length();
    let base_bound = f32_pair_rounding_bound(base_a, base_b) + 1.0e-6;
    ensure!(
        base_drift.is_finite() && base_drift <= base_bound,
        "stored base ECEF drift {base_drift:.9} m exceeds quantization bound {base_bound:.9} m"
    );
    Ok(base_drift + f32_pair_rounding_bound(local_a, local_b) + 1.0e-6)
}

#[derive(Debug)]
struct SelectedBoundaryComponent {
    tile_id: u32,
    vertices: Vec<usize>,
    radius: f32,
    uv_min: DVec2,
    uv_max: DVec2,
}

fn selected_boundary_components(
    frame: &OrbisFrameCapture,
    selected: &[SelectedDraw],
) -> Result<Vec<SelectedBoundaryComponent>> {
    let mut out = Vec::new();
    for draw in selected {
        let start = draw.first_index as usize;
        let end = start + draw.index_count as usize;
        ensure!(end <= frame.mesh_indices.len(), "ORBIS selected template index span is out of range");
        let mut counts = BTreeMap::<(u32, u32), u32>::new();
        for triangle in frame.mesh_indices[start..end].chunks_exact(3) {
            let resolved = [triangle[0], triangle[1], triangle[2]].map(|index| {
                let index = i64::from(index) + i64::from(draw.base_vertex);
                usize::try_from(index).ok()
            });
            ensure!(
                resolved.iter().all(|index| index.is_some_and(|value| value < frame.mesh_vertices.len())),
                "ORBIS selected template references an invalid vertex"
            );
            let resolved = resolved.map(Option::unwrap);
            if resolved
                .iter()
                .any(|&index| frame.mesh_vertices[index].is_skirt())
            {
                continue;
            }
            for (a, b) in [
                (resolved[0] as u32, resolved[1] as u32),
                (resolved[1] as u32, resolved[2] as u32),
                (resolved[2] as u32, resolved[0] as u32),
            ] {
                *counts.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let boundary = counts.into_iter().filter_map(|(edge, count)| (count == 1).then_some(edge)).collect::<Vec<_>>();
        for component in edge_components(&boundary) {
            let members = component.iter().copied().collect::<HashSet<_>>();
            let edges = boundary
                .iter()
                .copied()
                .filter(|(a, b)| members.contains(a) && members.contains(b))
                .collect::<Vec<_>>();
            let vertices = ordered_boundary_loop(frame, &edges)?;
            let radius = component_radius_vertices(&frame.mesh_vertices, &component);
            let (uv_min, uv_max) = vertices.iter().fold(
                (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
                |(min, max), &index| {
                    let uv = frame.topology_uvs[index];
                    (min.min(uv), max.max(uv))
                },
            );
            out.push(SelectedBoundaryComponent {
                tile_id: draw.tile_id,
                vertices,
                radius,
                uv_min,
                uv_max,
            });
        }
    }
    ensure!(out.len() >= 3, "ORBIS exact selected draws contain no mating ring boundaries");
    Ok(out)
}

fn ordered_boundary_loop(
    frame: &OrbisFrameCapture,
    edges: &[(u32, u32)],
) -> Result<Vec<usize>> {
    let mut adjacency = BTreeMap::<u32, Vec<u32>>::new();
    for &(a, b) in edges {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    ensure!(
        !adjacency.is_empty() && adjacency.values().all(|neighbors| neighbors.len() == 2),
        "ORBIS selected boundary is not one closed manifold loop"
    );
    let start = *adjacency
        .keys()
        .min_by(|&&a, &&b| {
            let a_uv = frame.topology_uvs[a as usize];
            let b_uv = frame.topology_uvs[b as usize];
            a_uv.x
                .total_cmp(&b_uv.x)
                .then_with(|| a_uv.y.total_cmp(&b_uv.y))
        })
        .unwrap();
    for neighbors in adjacency.values_mut() {
        neighbors.sort_by(|&a, &b| {
            let a_uv = frame.topology_uvs[a as usize];
            let b_uv = frame.topology_uvs[b as usize];
            a_uv.x
                .total_cmp(&b_uv.x)
                .then_with(|| a_uv.y.total_cmp(&b_uv.y))
        });
    }
    let mut ordered = vec![start as usize];
    let mut previous = None;
    let mut current = start;
    loop {
        let neighbors = &adjacency[&current];
        let next = neighbors
            .iter()
            .copied()
            .find(|candidate| Some(*candidate) != previous)
            .unwrap_or(neighbors[0]);
        if next == start {
            ordered.push(start as usize);
            break;
        }
        ensure!(
            !ordered.contains(&(next as usize)),
            "ORBIS selected boundary loop is self-intersecting/ambiguous"
        );
        ordered.push(next as usize);
        previous = Some(current);
        current = next;
        ensure!(
            ordered.len() <= adjacency.len() + 1,
            "ORBIS selected boundary loop did not close"
        );
    }
    ensure!(
        ordered.len() == adjacency.len() + 1,
        "ORBIS selected boundary traversal omitted vertices"
    );
    Ok(ordered)
}

fn selected_mating_boundaries(
    frame: &OrbisFrameCapture,
    selected: &[SelectedDraw],
    projections: &[ProjectionSample],
    projected: &[Option<Vec2>],
) -> Result<Vec<ProjectedSeamPair>> {
    let components = selected_boundary_components(frame, selected)?;
    let max_radius = components
        .iter()
        .map(|component| component.radius)
        .fold(f32::NEG_INFINITY, f32::max);
    let uv_tolerance = 2.0e-6;
    let mut used = HashSet::new();
    let mut pairs = Vec::new();
    for (index, component) in components.iter().enumerate() {
        if used.contains(&index) {
            continue;
        }
        let candidates = components
            .iter()
            .enumerate()
            .filter(|(other_index, other)| {
                *other_index != index
                    && !used.contains(other_index)
                    && other.tile_id != component.tile_id
                    && (other.uv_min - component.uv_min).abs().max_element() <= uv_tolerance
                    && (other.uv_max - component.uv_max).abs().max_element() <= uv_tolerance
            })
            .map(|(other_index, _)| other_index)
            .collect::<Vec<_>>();
        if candidates.is_empty()
            && (component.radius - max_radius).abs() <= max_radius.abs().max(1.0) * 1.0e-5
        {
            // Only the unmatched outermost component is the terrain
            // silhouette. It is explicitly excluded from seam evidence.
            used.insert(index);
            continue;
        }
        ensure!(
            candidates.len() == 1,
            "ORBIS selected boundary has {} mating candidates instead of exactly one",
            candidates.len()
        );
        let other_index = candidates[0];
        used.insert(index);
        used.insert(other_index);
        let other = &components[other_index];
        validate_boundary_identity(frame, projections, component, other)?;
        let other_vertices = aligned_boundary_vertices(frame, component, other)?;
        let inner = component
            .vertices
            .iter()
            .map(|&vertex| {
                projected
                    .get(vertex)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow!("ORBIS mating boundary projection is missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        let outer = other_vertices
            .iter()
            .map(|&vertex| {
                projected
                    .get(vertex)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow!("ORBIS mating boundary projection is missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner_depth = component
            .vertices
            .iter()
            .map(|&vertex| {
                projections
                    .get(vertex)
                    .map(|sample| sample.ndc_depth)
                    .ok_or_else(|| anyhow!("ORBIS inner mating depth is missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        let outer_depth = other_vertices
            .iter()
            .map(|&vertex| {
                projections
                    .get(vertex)
                    .map(|sample| sample.ndc_depth)
                    .ok_or_else(|| anyhow!("ORBIS outer mating depth is missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        let pair = ProjectedSeamPair {
            inner,
            outer,
            inner_depth,
            outer_depth,
            silhouette: false,
        };
        validate_projected_seam_pair(&pair)?;
        pairs.push(pair);
    }
    ensure!(!pairs.is_empty(), "ORBIS exact selection has no paired interior seam");
    Ok(pairs)
}

fn aligned_boundary_vertices(
    frame: &OrbisFrameCapture,
    first: &SelectedBoundaryComponent,
    second: &SelectedBoundaryComponent,
) -> Result<Vec<usize>> {
    ensure!(
        first.vertices.len() >= 3 && second.vertices.len() >= 3,
        "ORBIS mating boundary has no stable traversal direction"
    );
    let first_start = frame.topology_uvs[first.vertices[0]];
    let first_direction = frame.topology_uvs[first.vertices[1]] - first_start;
    let second_start = frame.topology_uvs[second.vertices[0]];
    ensure!(
        (second_start - first_start).length() <= 1.0e-7,
        "ORBIS mating boundary traversal starts at different stable UV identities"
    );
    let forward = frame.topology_uvs[second.vertices[1]] - second_start;
    let reverse = frame.topology_uvs[second.vertices[second.vertices.len() - 2]] - second_start;
    ensure!(
        first_direction.length_squared() > 0.0
            && forward.length_squared() > 0.0
            && reverse.length_squared() > 0.0,
        "ORBIS mating boundary traversal is degenerate"
    );
    if first_direction.dot(forward) >= first_direction.dot(reverse) {
        Ok(second.vertices.clone())
    } else {
        let mut aligned = Vec::with_capacity(second.vertices.len());
        aligned.push(second.vertices[0]);
        aligned.extend(
            second.vertices[1..second.vertices.len() - 1]
                .iter()
                .rev()
                .copied(),
        );
        aligned.push(second.vertices[0]);
        Ok(aligned)
    }
}

fn validate_boundary_identity(
    frame: &OrbisFrameCapture,
    projections: &[ProjectionSample],
    first: &SelectedBoundaryComponent,
    second: &SelectedBoundaryComponent,
) -> Result<()> {
    let mut matches = 0_usize;
    for &a_index in first.vertices.iter().take(first.vertices.len().saturating_sub(1)) {
        let a_uv = frame.topology_uvs[a_index];
        let Some((b_index, uv_distance)) = second
            .vertices
            .iter()
            .take(second.vertices.len().saturating_sub(1))
            .map(|b_index| (*b_index, (frame.topology_uvs[*b_index] - a_uv).length()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        if uv_distance > 1.0e-7 {
            continue;
        }
        let a = projections
            .get(a_index)
            .ok_or_else(|| anyhow!("ORBIS first mating endpoint projection is missing"))?;
        let b = projections
            .get(b_index)
            .ok_or_else(|| anyhow!("ORBIS second mating endpoint projection is missing"))?;
        let world_a = frame.frame_anchor_ecef + frame.local_to_ecef.transform_vector3(a.local);
        let world_b = frame.frame_anchor_ecef + frame.local_to_ecef.transform_vector3(b.local);
        let cpu_a = frame
            .mesh_vertices
            .get(a_index)
            .ok_or_else(|| anyhow!("ORBIS first mating CPU vertex is out of range"))?;
        let cpu_b = frame
            .mesh_vertices
            .get(b_index)
            .ok_or_else(|| anyhow!("ORBIS second mating CPU vertex is out of range"))?;
        let base_a = Vec3::from(cpu_a.position).as_dvec3();
        let base_b = Vec3::from(cpu_b.position).as_dvec3();
        let base_world_a =
            frame.frame_anchor_ecef + frame.local_to_ecef.transform_vector3(base_a);
        let base_world_b =
            frame.frame_anchor_ecef + frame.local_to_ecef.transform_vector3(base_b);
        validate_mating_world_identity(
            world_a,
            world_b,
            a.local,
            b.local,
            base_a,
            base_b,
            base_world_a,
            base_world_b,
        )
        .map_err(|error| {
            anyhow!(
                "ORBIS paired boundary identity {a_index}->{b_index}: {error}; cpu_a={cpu_a:?}, cpu_b={cpu_b:?}",
            )
        })?;
        ensure!(
            a.ndc_depth.is_finite()
                && b.ndc_depth.is_finite()
                && (a.ndc_depth - b.ndc_depth).abs() <= 2.0e-5,
            "ORBIS paired boundary endpoint depth disagrees"
        );
        matches += 1;
    }
    ensure!(
        matches >= 4,
        "ORBIS paired boundaries have only {matches} stable shared UV/world identities"
    );
    Ok(())
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn map_readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().map_err(|_| anyhow!("ORBIS readback callback dropped"))??;
    let bytes = slice.get_mapped_range().to_vec();
    buffer.unmap();
    Ok(bytes)
}

fn build_camera_matrices_f64(
    params: &crate::terrain::render_params::TerrainRenderParams,
) -> (DMat4, DMat4) {
    let phi = f64::from(params.cam_phi_deg).to_radians();
    let theta = f64::from(params.cam_theta_deg).to_radians();
    let radius = f64::from(params.cam_radius);
    let target = DVec3::from_array(params.cam_target.map(f64::from));
    let z_up = params.camera_mode.to_ascii_lowercase().contains(":zup");
    let (offset, up) = if z_up {
        let offset = DVec3::new(
            radius * theta.sin() * phi.cos(),
            radius * theta.sin() * phi.sin(),
            radius * theta.cos(),
        );
        let up = if theta.sin().abs() < 1.0e-4 {
            DVec3::Y
        } else {
            DVec3::Z
        };
        (offset, up)
    } else {
        (
            DVec3::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.cos(),
                radius * theta.sin() * phi.sin(),
            ),
            DVec3::Y,
        )
    };
    let view = DMat4::look_at_rh(target + offset, target, up);
    let projection = DMat4::perspective_rh(
        f64::from(params.fov_y_deg).to_radians(),
        f64::from(params.size_px.0) / f64::from(params.size_px.1),
        f64::from(params.clip.0),
        f64::from(params.clip.1),
    );
    (view, projection)
}

fn project_d(matrix: DMat4, position: DVec3) -> Option<DVec2> {
    let clip = matrix * position.extend(1.0);
    (clip.is_finite() && clip.w > 0.0).then(|| clip.truncate().truncate() / clip.w)
}

fn edge_components(edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut adjacency = HashMap::<u32, Vec<u32>>::new();
    for &(a, b) in edges {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    let mut unseen = adjacency.keys().copied().collect::<HashSet<_>>();
    let mut out = Vec::new();
    while let Some(&start) = unseen.iter().next() {
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        unseen.remove(&start);
        while let Some(vertex) = queue.pop_front() {
            component.push(vertex);
            for &next in adjacency.get(&vertex).into_iter().flatten() {
                if unseen.remove(&next) {
                    queue.push_back(next);
                }
            }
        }
        out.push(component);
    }
    out
}

fn component_radius_vertices(
    vertices: &[crate::terrain::clipmap::ClipmapVertex],
    component: &[u32],
) -> f32 {
    component
        .iter()
        .map(|&index| Vec3::from(vertices[index as usize].position).length())
        .sum::<f32>()
        / component.len().max(1) as f32
}

#[derive(Clone, Debug)]
struct ProjectedSeamPair {
    inner: Vec<Vec2>,
    outer: Vec<Vec2>,
    inner_depth: Vec<f32>,
    outer_depth: Vec<f32>,
    silhouette: bool,
}

fn validate_projected_seam_pair(pair: &ProjectedSeamPair) -> Result<()> {
    ensure!(
        pair.inner.len() >= 2
            && pair.outer.len() >= 2
            && pair.inner.len() == pair.inner_depth.len()
            && pair.outer.len() == pair.outer_depth.len(),
        "ORBIS selected seam is missing one mating boundary"
    );
    ensure!(
        pair.inner
            .iter()
            .chain(&pair.outer)
            .all(|point| point.is_finite())
            && pair
                .inner_depth
                .iter()
                .chain(&pair.outer_depth)
                .all(|depth| depth.is_finite()),
        "ORBIS selected seam contains non-finite projected evidence"
    );
    Ok(())
}

fn polyline_lengths(points: &[Vec2]) -> (Vec<f32>, f32) {
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0);
    for edge in points.windows(2) {
        let next = cumulative.last().copied().unwrap_or(0.0) + edge[0].distance(edge[1]);
        cumulative.push(next);
    }
    let total = cumulative.last().copied().unwrap_or(0.0);
    (cumulative, total)
}

fn sample_polyline(points: &[Vec2], cumulative: &[f32], total: f32, t: f32) -> Vec2 {
    if total <= f32::EPSILON {
        return points[0];
    }
    let distance = t.clamp(0.0, 1.0) * total;
    let upper = cumulative
        .partition_point(|value| *value < distance)
        .clamp(1, points.len() - 1);
    let lower = upper - 1;
    let span = (cumulative[upper] - cumulative[lower]).max(f32::EPSILON);
    points[lower].lerp(points[upper], (distance - cumulative[lower]) / span)
}

fn sample_polyline_scalar(values: &[f32], cumulative: &[f32], total: f32, t: f32) -> f32 {
    if total <= f32::EPSILON {
        return values[0];
    }
    let distance = t.clamp(0.0, 1.0) * total;
    let upper = cumulative
        .partition_point(|value| *value < distance)
        .clamp(1, values.len() - 1);
    let lower = upper - 1;
    let span = (cumulative[upper] - cumulative[lower]).max(f32::EPSILON);
    values[lower] + (values[upper] - values[lower]) * ((distance - cumulative[lower]) / span)
}

fn linearize_depth(depth: f32, clip_near: f64, clip_far: f64) -> Option<f64> {
    if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
        return None;
    }
    let denominator = clip_far - f64::from(depth) * (clip_far - clip_near);
    let linear = clip_near * clip_far / denominator;
    (linear.is_finite() && linear > 0.0).then_some(linear)
}

fn linear_depth_quantization_bound(linear_depth: f64, clip_near: f64, clip_far: f64) -> f64 {
    // Propagate one Depth32Float ULP through the perspective depth inverse.
    let derivative = linear_depth.powi(2) * (clip_far - clip_near)
        / (clip_near * clip_far);
    derivative.abs() * f64::from(f32::EPSILON)
}

#[derive(Clone, Copy, Debug)]
struct SeamDepthSample {
    distance: f64,
    linear_depth: f64,
    quantization: f64,
}

fn extrapolate_seam_depth(samples: &[SeamDepthSample]) -> Option<(f64, f64)> {
    let order = samples.len().min(4);
    let samples = samples.get(..order).filter(|_| order >= 2)?;
    let mut estimate = 0.0_f64;
    let mut quantization = 0.0_f64;
    for (index, sample) in samples.iter().enumerate() {
        let mut coefficient = 1.0_f64;
        for (other_index, other) in samples.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let denominator = sample.distance - other.distance;
            if denominator.abs() <= f64::EPSILON {
                return None;
            }
            coefficient *= -other.distance / denominator;
        }
        estimate += coefficient * sample.linear_depth;
        quantization += coefficient.abs() * sample.quantization;
    }
    (estimate.is_finite() && quantization.is_finite()).then_some((estimate, quantization))
}

#[allow(clippy::too_many_arguments)]
fn analyze_paired_cracks(
    bytes: &[u8],
    coverage_bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_row: u32,
    coverage_bytes_per_row: u32,
    clip_near: f64,
    clip_far: f64,
    world_units_per_pixel: f64,
    pairs: &[ProjectedSeamPair],
) -> Result<(u64, u64, f64)> {
    ensure!(width > 0 && height > 0, "ORBIS crack viewport is empty");
    ensure!(
        bytes.len() >= bytes_per_row as usize * height as usize
            && coverage_bytes.len() >= coverage_bytes_per_row as usize * height as usize,
        "ORBIS crack readback is incomplete"
    );
    ensure!(
        clip_near.is_finite()
            && clip_far.is_finite()
            && clip_near > 0.0
            && clip_far > clip_near
            && world_units_per_pixel.is_finite()
            && world_units_per_pixel > 0.0,
        "ORBIS crack projection/depth scale is invalid"
    );
    let mut sampled = HashSet::<(u32, u32)>::new();
    let mut crack_pixels = HashSet::<(u32, u32)>::new();
    let mut depth_values = Vec::new();
    let mut active_pairs = 0_usize;
    for (pair_index, pair) in pairs.iter().enumerate() {
        if pair.silhouette {
            continue;
        }
        active_pairs += 1;
        validate_projected_seam_pair(pair)?;
        let (inner_cumulative, inner_total) = polyline_lengths(&pair.inner);
        let (outer_cumulative, outer_total) = polyline_lengths(&pair.outer);
        ensure!(
            inner_total >= 2.0 && outer_total >= 2.0,
            "ORBIS selected seam has insufficient projected length"
        );
        let longitudinal_steps = inner_total.max(outer_total).ceil().max(2.0) as u32;
        for step in 0..=longitudinal_steps {
            let t = step as f32 / longitudinal_steps as f32;
            let inner = sample_polyline(&pair.inner, &inner_cumulative, inner_total, t);
            let outer = sample_polyline(&pair.outer, &outer_cumulative, outer_total, t);
            let midpoint = (inner + outer) * 0.5;
            let dt = 1.0 / longitudinal_steps as f32;
            let prev_t = (t - dt).max(0.0);
            let next_t = (t + dt).min(1.0);
            let prev_mid = (sample_polyline(
                &pair.inner,
                &inner_cumulative,
                inner_total,
                prev_t,
            ) + sample_polyline(
                &pair.outer,
                &outer_cumulative,
                outer_total,
                prev_t,
            )) * 0.5;
            let next_mid = (sample_polyline(
                &pair.inner,
                &inner_cumulative,
                inner_total,
                next_t,
            ) + sample_polyline(
                &pair.outer,
                &outer_cumulative,
                outer_total,
                next_t,
            )) * 0.5;
            let tangent = (next_mid - prev_mid).normalize_or_zero();
            let cross = outer - inner;
            let normal = if cross.length_squared() > 1.0e-6 {
                cross.normalize()
            } else {
                Vec2::new(-tangent.y, tangent.x).normalize_or_zero()
            };
            ensure!(normal != Vec2::ZERO, "ORBIS selected seam normal is degenerate");

            // Boundary attestation: the linear depth each submitted mating
            // edge claims to render at this seam position. Corridor pixels can
            // only be judged against it; without it no depth classification is
            // possible.
            let inner_linear = linearize_depth(
                sample_polyline_scalar(
                    &pair.inner_depth,
                    &inner_cumulative,
                    inner_total,
                    t,
                ),
                clip_near,
                clip_far,
            );
            let outer_linear = linearize_depth(
                sample_polyline_scalar(
                    &pair.outer_depth,
                    &outer_cumulative,
                    outer_total,
                    t,
                ),
                clip_near,
                clip_far,
            );
            let attested = match (inner_linear, outer_linear) {
                (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
                (Some(v), None) | (None, Some(v)) => Some((v, v)),
                _ => None,
            };
            let separation = f64::from((outer - inner).dot(normal)).max(0.0);
            // Without attestation the corridor cannot be classified; preserve
            // the strictest behaviour and let the side comparison run.
            let mut seam_visible = attested.is_none();
            // Flanked-hole evidence for seams too narrow for strictly interior
            // pixel centres: content behind both submitted surfaces on the
            // seam band is a hole when attested surface renders on both sides
            // of it and nothing behind appears further out. A silhouette always
            // leaves its far side behind the attestation, so it never flanks.
            let mut seam_band_behind = Vec::new();
            let mut inner_flank_attested = false;
            let mut outer_flank_attested = false;
            let mut inner_flank_behind = false;
            let mut outer_flank_behind = false;

            // Rasterize the closed corridor between the two submitted mating
            // boundaries, dilated by two pixels on both sides. Sampling both
            // longitudinal endpoints is deliberate: endpoint holes are cracks.
            let band_start = inner - normal * 2.0;
            let band_end = outer + normal * 2.0;
            let cross_steps = band_start.distance(band_end).ceil().max(1.0) as u32;
            for cross_step in 0..=cross_steps {
                let cross_t = cross_step as f32 / cross_steps as f32;
                let point = band_start.lerp(band_end, cross_t);
                let x = point.x.round() as i32;
                let y = point.y.round() as i32;
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }
                let pixel = (x as u32, y as u32);
                sampled.insert(pixel);
                let coverage_offset =
                    y as usize * coverage_bytes_per_row as usize + x as usize;
                if coverage_bytes[coverage_offset] <= 127 {
                    if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                        eprintln!("ORBIS coverage crack pair={pair_index} pixel={pixel:?}");
                    }
                    crack_pixels.insert(pixel);
                    continue;
                }
                let depth_offset =
                    y as usize * bytes_per_row as usize + x as usize * 4;
                let depth = f32::from_le_bytes(
                    bytes[depth_offset..depth_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let Some(linear) = linearize_depth(depth, clip_near, clip_far) else {
                    continue;
                };
                depth_values.push(linear);
                // Depth classification applies only inside the strict band
                // between the polylines (plus a half-pixel rasterization
                // allowance): the two-pixel dilation is slop margin where
                // legitimately different content may abut a surface edge.
                let Some((attested_lo, attested_hi)) = attested else {
                    continue;
                };
                let along = f64::from((point.round() - inner).dot(normal));
                let bound = linear_depth_quantization_bound(linear, clip_near, clip_far)
                    + linear_depth_quantization_bound(attested_lo, clip_near, clip_far)
                    + linear_depth_quantization_bound(attested_hi, clip_near, clip_far)
                    + world_units_per_pixel * 2.0;
                let consistent = linear >= attested_lo - bound && linear <= attested_hi + bound;
                // A pixel rendering at the claimed depth anywhere in the
                // dilated band proves the submitted seam surface is visible;
                // only then is the side depth-step comparison meaningful.
                if consistent {
                    seam_visible = true;
                }
                // Only pixel centers strictly interior to the sliver between
                // the two projected edges are reliable hole evidence. A
                // center within half a pixel of either claimed edge is edge
                // quantization: it may legitimately lie just past the
                // rasterized surface edge at a silhouette.
                let interior = along > 0.5 && along < separation - 0.5;
                let behind = linear > attested_hi + bound;
                if along < -0.5 {
                    inner_flank_attested |= consistent;
                    inner_flank_behind |= behind;
                } else if along > separation + 0.5 {
                    outer_flank_attested |= consistent;
                    outer_flank_behind |= behind;
                } else if behind && !interior {
                    seam_band_behind.push(pixel);
                }
                if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                    eprintln!(
                        "ORBIS strict pixel pair={pair_index} step={step} pixel={pixel:?} rendered={linear:.2} attested=({attested_lo:.2},{attested_hi:.2}) along={along:.3} sep={separation:.3} interior={interior} class={}",
                        if linear > attested_hi + bound { "behind" } else if consistent { "consistent" } else { "occluder" },
                    );
                }
                if interior && linear > attested_hi + bound {
                    // Covered by content behind both submitted surfaces: the
                    // claimed seam did not render here, so this is a hole.
                    if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                        eprintln!(
                            "ORBIS depth crack pair={pair_index} pixel={pixel:?} rendered={linear:.6} attested=({attested_lo:.6}, {attested_hi:.6}) bound={bound:.6}"
                        );
                    }
                    crack_pixels.insert(pixel);
                }
                // A pixel nearer than both boundaries is legitimate occlusion
                // of the seam by foreground terrain, not crack evidence.
            }
            if inner_flank_attested
                && outer_flank_attested
                && !inner_flank_behind
                && !outer_flank_behind
            {
                for pixel in seam_band_behind {
                    if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                        eprintln!("ORBIS flanked hole pair={pair_index} step={step} pixel={pixel:?}");
                    }
                    crack_pixels.insert(pixel);
                }
            }

            let sample_side = |direction: f32| -> Vec<SeamDepthSample> {
                let mut side = Vec::with_capacity(4);
                let mut pixels = HashSet::new();
                for magnitude in 1..=10 {
                    let point = midpoint + normal * direction * magnitude as f32;
                    let x = point.x.round() as i32;
                    let y = point.y.round() as i32;
                    if x < 0
                        || y < 0
                        || x >= width as i32
                        || y >= height as i32
                        || !pixels.insert((x, y))
                    {
                        continue;
                    }
                    let pixel_position = Vec2::new(x as f32, y as f32);
                    let distance = f64::from((pixel_position - midpoint).dot(normal));
                    if distance * f64::from(direction) <= 0.0 {
                        continue;
                    }
                    let coverage_offset =
                        y as usize * coverage_bytes_per_row as usize + x as usize;
                    if coverage_bytes[coverage_offset] <= 127 {
                        continue;
                    }
                    let depth_offset = y as usize * bytes_per_row as usize + x as usize * 4;
                    let depth = f32::from_le_bytes(
                        bytes[depth_offset..depth_offset + 4]
                            .try_into()
                            .expect("validated depth readback stride"),
                    );
                    let Some(linear_depth) = linearize_depth(depth, clip_near, clip_far) else {
                        continue;
                    };
                    side.push(SeamDepthSample {
                        distance,
                        linear_depth,
                        quantization: linear_depth_quantization_bound(
                            linear_depth,
                            clip_near,
                            clip_far,
                        ),
                    });
                    if side.len() == 4 {
                        break;
                    }
                }
                side
            };
            // Restrict each side's sample run to the contiguous surface that
            // abuts the seam: the first sample must lie within the attested
            // interval, and each later sample must continue the previous one
            // within a quantization-plus-slope bound. A run that crosses an
            // occlusion edge (a silhouette or a foreign surface) ends there —
            // extrapolating through it invents depths that were never
            // submitted. Boundary-only evidence cannot separate a mating
            // surface rendered behind its attestation from a local terrain
            // silhouette running along the seam (the physical Rainier probe
            // has both seams crossing a ridge against 6 km terrain), so an
            // unflanked far side is not crack evidence; the flanked-hole rule
            // above covers holes the surface resumes past.
            let seam_run = |mut side: Vec<SeamDepthSample>| -> Vec<SeamDepthSample> {
                if let Some((attested_lo, attested_hi)) = attested {
                    if let Some(first) = side.first() {
                        let bound = first.quantization
                            + linear_depth_quantization_bound(
                                attested_lo,
                                clip_near,
                                clip_far,
                            )
                            + linear_depth_quantization_bound(
                                attested_hi,
                                clip_near,
                                clip_far,
                            )
                            + first.distance.abs() * world_units_per_pixel;
                        if first.linear_depth < attested_lo - bound
                            || first.linear_depth > attested_hi + bound
                        {
                            side.clear();
                        }
                    }
                }
                let mut end = side.len();
                for index in 1..side.len() {
                    let previous = side[index - 1];
                    let current = side[index];
                    let jump = (current.linear_depth - previous.linear_depth).abs();
                    let run = (current.distance - previous.distance).abs();
                    if jump
                        > previous.quantization
                            + current.quantization
                            + run * world_units_per_pixel
                    {
                        end = index;
                        break;
                    }
                }
                side.truncate(end);
                side
            };
            let left = seam_run(sample_side(-1.0));
            let right = seam_run(sample_side(1.0));
            if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                eprintln!(
                    "ORBIS seam samples pair={pair_index} step={step}/{longitudinal_steps} mid={midpoint:?} inner_linear={inner_linear:?} outer_linear={outer_linear:?} visible={seam_visible} left={:?} right={:?}",
                    left.iter().map(|s| (s.distance, s.linear_depth)).collect::<Vec<_>>(),
                    right.iter().map(|s| (s.distance, s.linear_depth)).collect::<Vec<_>>(),
                );
            }
            // The depth-step comparison only runs where the strict corridor
            // proved the submitted seam surface is actually visible; an
            // occluded seam cannot show a crack.
            if seam_visible {
                if let (
                    Some((left_at_seam, left_quantization)),
                    Some((right_at_seam, right_quantization)),
                ) = (extrapolate_seam_depth(&left), extrapolate_seam_depth(&right))
                {
                    depth_values.extend(
                        left.iter().chain(&right).map(|sample| sample.linear_depth),
                    );
                    // A covered corridor whose two submitted boundaries render
                    // at different depths is legitimate neighbor occlusion,
                    // not a hole: only a step beyond the boundary-attested
                    // separation is unexplained.
                    let (boundary_step, boundary_quantization) = match attested {
                        Some((attested_lo, attested_hi)) => (
                            attested_hi - attested_lo,
                            linear_depth_quantization_bound(attested_lo, clip_near, clip_far)
                                + linear_depth_quantization_bound(
                                    attested_hi,
                                    clip_near,
                                    clip_far,
                                ),
                        ),
                        _ => (0.0, 0.0),
                    };
                    let tolerance = (left_quantization + right_quantization) * 2.0
                        + boundary_quantization
                        + world_units_per_pixel * 0.01;
                    let excess =
                        (left_at_seam - right_at_seam).abs() - boundary_step.abs();
                    if excess > tolerance {
                        let pixel = midpoint.round().as_ivec2();
                        if pixel.x >= 0
                            && pixel.y >= 0
                            && pixel.x < width as i32
                            && pixel.y < height as i32
                        {
                            let pixel = (pixel.x as u32, pixel.y as u32);
                            if std::env::var_os("FORGE3D_ORBIS_DEBUG_CRACKS").is_some() {
                                eprintln!(
                                    "ORBIS depth crack pair={pair_index} pixel={pixel:?} left={left_at_seam:.6} right={right_at_seam:.6} excess={excess:.6} boundary_step={boundary_step:.6} tolerance={tolerance:.6}",
                                );
                            }
                            crack_pixels.insert(pixel);
                        }
                    }
                }
            }
        }
    }
    ensure!(active_pairs > 0, "ORBIS crack probe has no non-silhouette mating seam");
    ensure!(sampled.len() >= 64, "ORBIS crack probe had only {} boundary samples", sampled.len());
    ensure!(depth_values.len() >= 32, "ORBIS crack probe had insufficient valid depth evidence");
    let mean = depth_values.iter().sum::<f64>() / depth_values.len() as f64;
    let variance = depth_values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / depth_values.len() as f64;
    let (min_depth, max_depth) = depth_values.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_value, max_value), &value| (min_value.min(value), max_value.max(value)),
    );
    ensure!(
        variance.is_finite() && variance > 1.0e-8 && max_depth - min_depth > 1.0e-3,
        "ORBIS crack depth evidence lacks meaningful linear-depth variance"
    );
    Ok((crack_pixels.len() as u64, sampled.len() as u64, variance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crack_analysis_fails_closed_on_empty_boundary_evidence() {
        assert!(analyze_paired_cracks(
            &[], &[], 1, 1, 256, 256, 0.1, 1000.0, 1.0, &[]
        )
        .is_err());
    }

    fn encode_depth(linear: f64, near: f64, far: f64) -> f32 {
        ((far - near * far / linear) / (far - near)) as f32
    }

    fn seam_fixture(step: bool, holes: &[(u32, u32)]) -> (Vec<u8>, Vec<u8>, u32, u32) {
        let (width, height) = (64_u32, 64_u32);
        let row = 256_u32;
        let mut depth = vec![0_u8; row as usize * height as usize];
        let mut coverage = vec![u8::MAX; row as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let linear = 100.0
                    + f64::from(y) * 0.2
                    + f64::from(x) * 0.1
                    + if step && x >= 32 { 5.0 } else { 0.0 };
                let encoded = encode_depth(linear, 0.1, 1000.0).to_le_bytes();
                let offset = y as usize * row as usize + x as usize * 4;
                depth[offset..offset + 4].copy_from_slice(&encoded);
            }
        }
        for &(x, y) in holes {
            coverage[y as usize * row as usize + x as usize] = 0;
        }
        (depth, coverage, row, row)
    }

    fn smooth_cubic_seam_fixture() -> (Vec<u8>, Vec<u8>, u32, u32) {
        let (width, height) = (64_u32, 64_u32);
        let row = 256_u32;
        let mut depth = vec![0_u8; row as usize * height as usize];
        let coverage = vec![u8::MAX; row as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) - 32.0;
                let linear = 100.0 + f64::from(y) * 0.2 + dx * dx * dx * 0.02;
                let encoded = encode_depth(linear, 0.1, 1000.0).to_le_bytes();
                let offset = y as usize * row as usize + x as usize * 4;
                depth[offset..offset + 4].copy_from_slice(&encoded);
            }
        }
        (depth, coverage, row, row)
    }

    fn vertical_pair(silhouette: bool) -> ProjectedSeamPair {
        // Attestation tracks the submitted surface at each boundary vertex —
        // matching `seam_fixture` — but deliberately without the +5 step so a
        // rendered step is an unattested crack.
        let vertex_depth = |x: f32, y: f32| {
            encode_depth(100.0 + f64::from(y) * 0.2 + f64::from(x) * 0.1, 0.1, 1000.0)
        };
        ProjectedSeamPair {
            inner: vec![Vec2::new(31.0, 8.0), Vec2::new(31.0, 55.0)],
            outer: vec![Vec2::new(32.0, 8.0), Vec2::new(32.0, 55.0)],
            inner_depth: vec![vertex_depth(31.0, 8.0), vertex_depth(31.0, 55.0)],
            outer_depth: vec![vertex_depth(32.0, 8.0), vertex_depth(32.0, 55.0)],
            silhouette,
        }
    }

    #[test]
    fn paired_seam_oracle_detects_corridor_endpoints_and_depth_step() {
        let pair = vertical_pair(false);
        let (depth, coverage, drow, crow) = seam_fixture(false, &[]);
        let green = analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(green.0, 0);

        let corridor = (8..=55).flat_map(|y| [(31, y), (32, y)]).collect::<Vec<_>>();
        let (depth, coverage, drow, crow) = seam_fixture(false, &corridor);
        assert!(analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, std::slice::from_ref(&pair),
        )
        .unwrap()
        .0 > 0);

        let vertex_depth = |x: f32, y: f32| {
            encode_depth(100.0 + f64::from(y) * 0.2 + f64::from(x) * 0.1, 0.1, 1000.0)
        };
        let separated = ProjectedSeamPair {
            inner_depth: vec![vertex_depth(28.0, 8.0), vertex_depth(28.0, 55.0)],
            outer_depth: vec![vertex_depth(35.0, 8.0), vertex_depth(35.0, 55.0)],
            inner: vec![Vec2::new(28.0, 8.0), Vec2::new(28.0, 55.0)],
            outer: vec![Vec2::new(35.0, 8.0), Vec2::new(35.0, 55.0)],
            silhouette: false,
        };
        let gap = (8..=55)
            .flat_map(|y| (30..=33).map(move |x| (x, y)))
            .collect::<Vec<_>>();
        let (depth, coverage, drow, crow) = seam_fixture(false, &gap);
        assert!(analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, &[separated],
        )
        .unwrap()
        .0 > 0);

        let endpoints = (8..=10)
            .chain(53..=55)
            .flat_map(|y| [(31, y), (32, y)])
            .collect::<Vec<_>>();
        let (depth, coverage, drow, crow) = seam_fixture(false, &endpoints);
        assert!(analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, std::slice::from_ref(&pair),
        )
        .unwrap()
        .0 > 0);

        // A gap between the projected edges whose interior pixels render
        // content deeper than both submitted surfaces is a crack: the
        // claimed surfaces do not cover their own seam.
        let gapped = ProjectedSeamPair {
            inner: vec![Vec2::new(30.0, 8.0), Vec2::new(30.0, 55.0)],
            outer: vec![Vec2::new(34.0, 8.0), Vec2::new(34.0, 55.0)],
            inner_depth: vec![vertex_depth(30.0, 8.0), vertex_depth(30.0, 55.0)],
            outer_depth: vec![vertex_depth(34.0, 8.0), vertex_depth(34.0, 55.0)],
            silhouette: false,
        };
        let (depth, coverage, drow, crow) = seam_fixture(true, &[]);
        assert!(analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, &[gapped],
        )
        .unwrap()
        .0 > 0);
    }

    #[test]
    fn paired_seam_oracle_ignores_neighbor_occlusion_when_corridor_is_covered() {
        // The mating boundaries themselves render at the 5 m step: the
        // neighbor surface legitimately occludes the covered corridor, so the
        // observed discontinuity is boundary-attested, not a crack.
        let encode = |linear: f64| encode_depth(linear, 0.1, 1000.0);
        let mut pair = vertical_pair(false);
        pair.inner_depth = pair
            .inner
            .iter()
            .map(|point| encode(100.0 + f64::from(point.y) * 0.2 + f64::from(point.x) * 0.1))
            .collect();
        pair.outer_depth = pair
            .outer
            .iter()
            .map(|point| {
                encode(100.0 + f64::from(point.y) * 0.2 + f64::from(point.x) * 0.1 + 5.0)
            })
            .collect();
        let (depth, coverage, drow, crow) = seam_fixture(true, &[]);
        let result = analyze_paired_cracks(
            &depth,
            &coverage,
            64,
            64,
            drow,
            crow,
            0.1,
            1000.0,
            1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(result.0, 0);
    }

    #[test]
    fn paired_seam_oracle_skips_occluded_and_silhouette_seams() {
        // The submitted boundaries attest the far surface while foreground
        // terrain renders in front of the projected seam: an occluded seam
        // cannot show a crack and produces no depth evidence.
        let encode = |linear: f64| encode_depth(linear, 0.1, 1000.0);
        let mut pair = vertical_pair(false);
        pair.inner_depth = pair.inner.iter().map(|_| encode(900.0)).collect();
        pair.outer_depth = pair.outer.iter().map(|_| encode(900.0)).collect();
        let (depth, coverage, drow, crow) = seam_fixture(false, &[]);
        let result = analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(result.0, 0, "an occluded seam is not crack evidence");

        // The seam surface is visible and attested-consistent, but one side's
        // run crosses a real terrain silhouette four pixels out. The foreign
        // surface must not poison the seam extrapolation.
        let (width, height, row) = (64_u32, 64_u32, 256_u32);
        let mut depth = vec![0_u8; row as usize * height as usize];
        let coverage = vec![u8::MAX; row as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let mut linear = 100.0 + f64::from(y) * 0.2 + f64::from(x) * 0.1;
                if x >= 32 {
                    linear += 5.0;
                }
                if x <= 28 {
                    linear += 700.0;
                }
                let encoded = encode_depth(linear, 0.1, 1000.0).to_le_bytes();
                let offset = y as usize * row as usize + x as usize * 4;
                depth[offset..offset + 4].copy_from_slice(&encoded);
            }
        }
        let mut pair = vertical_pair(false);
        pair.outer_depth = pair
            .outer
            .iter()
            .map(|point| {
                encode(100.0 + f64::from(point.y) * 0.2 + f64::from(point.x) * 0.1 + 5.0)
            })
            .collect();
        let result = analyze_paired_cracks(
            &depth, &coverage, 64, 64, row, row, 0.1, 1000.0, 1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(result.0, 0, "a foreign surface past the silhouette is not a crack");
    }

    fn seam_fixture_with_far_columns(far: impl Fn(u32, u32) -> bool) -> (Vec<u8>, Vec<u8>, u32) {
        let (width, height, row) = (64_u32, 64_u32, 256_u32);
        let mut depth = vec![0_u8; row as usize * height as usize];
        let coverage = vec![u8::MAX; row as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let mut linear = 100.0 + f64::from(y) * 0.2 + f64::from(x) * 0.1;
                if far(x, y) {
                    linear += 600.0;
                }
                let encoded = encode_depth(linear, 0.1, 1000.0).to_le_bytes();
                let offset = y as usize * row as usize + x as usize * 4;
                depth[offset..offset + 4].copy_from_slice(&encoded);
            }
        }
        (depth, coverage, row)
    }

    #[test]
    fn paired_seam_oracle_detects_covered_sliver_on_narrow_seam() {
        // A one-pixel seam has no strictly interior pixel centres. Farther
        // terrain showing through the seam band while the attested surface
        // renders on both sides is a hole even though coverage is full.
        let pair = vertical_pair(false);
        let (depth, coverage, row) =
            seam_fixture_with_far_columns(|x, y| (31..=32).contains(&x) && (20..=24).contains(&y));
        let result = analyze_paired_cracks(
            &depth, &coverage, 64, 64, row, row, 0.1, 1000.0, 1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert!(result.0 >= 5, "covered sliver on a narrow seam was not flagged: {}", result.0);
    }

    #[test]
    fn paired_seam_oracle_ignores_far_silhouette_along_narrow_seam() {
        // Boundary-only evidence cannot separate this from a mating surface
        // rendered behind its attestation; the far side never resumes, so it
        // is treated as a terrain silhouette (as on the physical Rainier probe).
        let pair = vertical_pair(false);
        let (depth, coverage, row) = seam_fixture_with_far_columns(|x, _| x >= 32);
        let result = analyze_paired_cracks(
            &depth, &coverage, 64, 64, row, row, 0.1, 1000.0, 1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(result.0, 0, "a silhouette along the seam is not a crack");
    }

    #[test]
    fn paired_seam_oracle_accepts_smooth_high_curvature_depth() {
        let pair = vertical_pair(false);
        let (depth, coverage, drow, crow) = smooth_cubic_seam_fixture();
        let result = analyze_paired_cracks(
            &depth,
            &coverage,
            64,
            64,
            drow,
            crow,
            0.1,
            1000.0,
            1.0,
            std::slice::from_ref(&pair),
        )
        .unwrap();
        assert_eq!(result.0, 0, "a continuous curved surface is not a seam crack");
    }

    #[test]
    fn silhouette_pair_is_excluded_but_missing_mate_fails_closed() {
        let (depth, coverage, drow, crow) = seam_fixture(false, &[(31, 20)]);
        let vertex_depth = |x: f32, y: f32| {
            encode_depth(100.0 + f64::from(y) * 0.2 + f64::from(x) * 0.1, 0.1, 1000.0)
        };
        let pairs = [vertical_pair(true), ProjectedSeamPair {
            inner: vec![Vec2::new(20.0, 8.0), Vec2::new(20.0, 55.0)],
            outer: vec![Vec2::new(21.0, 8.0), Vec2::new(21.0, 55.0)],
            inner_depth: vec![vertex_depth(20.0, 8.0), vertex_depth(20.0, 55.0)],
            outer_depth: vec![vertex_depth(21.0, 8.0), vertex_depth(21.0, 55.0)],
            silhouette: false,
        }];
        assert_eq!(analyze_paired_cracks(
            &depth, &coverage, 64, 64, drow, crow, 0.1, 1000.0, 1.0, &pairs,
        ).unwrap().0, 0);
        assert!(validate_projected_seam_pair(&ProjectedSeamPair {
            inner: vec![Vec2::ZERO],
            outer: Vec::new(),
            inner_depth: vec![0.5],
            outer_depth: Vec::new(),
            silhouette: false,
        })
        .is_err());
    }

    #[test]
    fn exact_selected_index_set_excludes_unsubmitted_vertices_and_rejects_drift() {
        let indices = (0..1032_u32).collect::<Vec<_>>();
        let selected = [SelectedDraw {
            tile_id: 7,
            selected_lod: 2,
            first_index: 0,
            index_count: 32,
            base_vertex: 0,
        }];
        let exact = exact_selected_vertex_indices(&indices, 1032, &selected).unwrap();
        assert_eq!(exact.len(), 32);
        assert!(!exact.contains(&32));
        let only_31 = [SelectedDraw { index_count: 31, ..selected[0].clone() }];
        assert!(require_selected_jitter_samples(&exact_selected_vertex_indices(
            &indices, 1032, &only_31,
        ).unwrap(), 32).is_err());

        let anchor = DVec3::new(1_000_000.0, 2_000_000.0, 3_000_000.0);
        let local = DVec3::new(1_250.0, -700.0, 20.0);
        assert!(canonical_selected_world_identity(
            anchor + local,
            anchor + local,
            local,
            local,
            local,
            local,
            anchor + local,
            anchor + local,
        )
        .is_ok());
        assert!(canonical_selected_world_identity(
            anchor + local,
            anchor + local + DVec3::X * 0.003,
            local,
            local + DVec3::X * 0.003,
            local,
            local + DVec3::X * 0.003,
            anchor + local,
            anchor + local + DVec3::X * 0.003,
        )
        .is_err());

        let far_ring = DVec3::new(12_500.0, -12_500.0, -6_000.0);
        assert!(validate_mating_world_identity(
            anchor + far_ring,
            anchor + far_ring + DVec3::X * 0.001,
            far_ring,
            far_ring + DVec3::X * 0.001,
            far_ring,
            far_ring + DVec3::X * 0.001,
            anchor + far_ring,
            anchor + far_ring + DVec3::X * 0.001,
        )
        .is_ok());
        assert!(validate_mating_world_identity(
            anchor + far_ring,
            anchor + far_ring + DVec3::X * 0.003,
            far_ring,
            far_ring + DVec3::X * 0.003,
            far_ring,
            far_ring + DVec3::X * 0.003,
            anchor + far_ring,
            anchor + far_ring + DVec3::X * 0.003,
        )
        .is_err());

        let identity =
            crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance::identity(7, 2);
        assert!(validate_selected_instance(&identity, 7, 2).is_ok());
        let mut translated = identity;
        translated.transform[3][0] = 1.0;
        assert!(validate_selected_instance(&translated, 7, 2).is_err());
    }

    #[test]
    fn six_kilometre_reanchor_quantization_bound_stays_below_half_probe() {
        let first = Vec3::new(-835.4459, -835.4459, -6000.1094).as_dvec3();
        let second = Vec3::new(-835.4459, -835.4459, -6026.2183).as_dvec3();
        assert!(
            local_vector_quantization_bound(
                first,
                second,
                first,
                second,
                DVec3::ZERO,
                DVec3::ZERO,
            )
            .unwrap()
                < ORBIS_MICRO_STEP_M * 0.5
        );
    }

    #[test]
    fn metric_probe_has_one_writer_per_vertex_and_gates_exact_selection() {
        let shader = crate::shader_sources::terrain();
        let entry = shader
            .split("fn orbis_metric_probe")
            .nth(1)
            .expect("ORBIS metric probe entry point");
        assert!(entry.contains("let index = invocation.x;"));
        assert!(entry.contains("if (!orbis_vertex_is_selected(index))"));
        assert!(!entry.contains("let draw_index = invocation.y;"));
    }

    #[test]
    fn selection_is_canonical_and_rejects_duplicate_stable_identity() {
        let a = SelectedDraw {
            tile_id: 2,
            selected_lod: 1,
            first_index: 12,
            index_count: 6,
            base_vertex: 0,
        };
        let b = SelectedDraw {
            tile_id: 1,
            selected_lod: 3,
            first_index: 30,
            index_count: 9,
            base_vertex: 4,
        };
        assert_eq!(
            canonicalize_selected_draws(vec![a.clone(), b.clone()]).unwrap(),
            canonicalize_selected_draws(vec![b.clone(), a.clone()]).unwrap()
        );
        assert!(canonicalize_selected_draws(vec![a.clone(), a]).is_err());
    }

    #[test]
    fn camera_match_uses_sub_millimetre_derived_quantization_bound() {
        let eye = glam::Vec3::new(0.0, 0.0, 0.5);
        let bound = orbis_camera_match_quantization_bound(eye, 4_400.5, [0.0, 0.0, -4_400.0])
            .unwrap();
        assert!(bound < 0.001);
        let expected = DVec3::new(1.0, 2.0, 3.0);
        assert!(validate_orbis_camera_match(expected + DVec3::X * (bound * 0.5), expected, bound)
            .is_ok());
        assert!(validate_orbis_camera_match(expected + DVec3::X * (bound * 1.01), expected, bound)
            .is_err());
    }
}
