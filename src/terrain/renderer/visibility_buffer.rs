//! TESSELLA terrain visibility buffer.
//!
//! Zero is reserved for background. Visible primitives are encoded as
//! `1 + ((tile_lod_id & 0xffff) << 16) | (triangle_id & 0xffff)`. The extra one
//! keeps tile zero / triangle zero distinct from background. That packing is
//! defined by `src/shaders/terrain_visbuffer_write.wgsl` (pass 1) and decoded
//! by `src/shaders/terrain_visibility_fullscreen.wgsl` (pass 2); the CPU oracle
//! below is the third mirror of it. The material resolve is a full-screen
//! fragment pass. It reads primitive identity and depth, reconstructs the
//! visible surface, and invokes POM/material/feedback exactly once for every
//! non-background visibility pixel. The runtime resolve replays the original
//! clipmap geometry against the pass-1 depth/identity buffer; the fullscreen
//! helper remains in the module for shader-contract coverage. `terrain_visbuffer_resolve.wgsl`
//! is the compute pass that reads those counters back, not the resolve itself.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use std::sync::{Mutex, OnceLock};

pub(in crate::terrain::renderer) struct CpuVisibilityOracle {
    mesh: crate::accel::cpu_bvh::MeshCPU,
    bvh: crate::accel::cpu_bvh::BvhCPU,
    identities: Vec<(u32, u32)>,
    inv_view_proj: [[f32; 4]; 4],
    viewport: (u32, u32),
}

impl CpuVisibilityOracle {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn build(
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: &[f32],
        height_dims: (u32, u32),
        clipmap: &crate::terrain::clipmap::ClipmapMesh,
        tiles: &[crate::terrain::clipmap::gpu_lod::TileInfo],
        templates: &[crate::terrain::clipmap::gpu_lod::IndirectDrawTemplate],
        variant_count: u32,
        lod_config: &crate::terrain::clipmap::gpu_lod::GpuLodConfig,
    ) -> anyhow::Result<Self> {
        use crate::terrain::clipmap::gpu_lod::cpu_lod_select;

        let decoded = params.decoded();
        let (h_min, h_max) = decoded.clamp.height_range;
        let h_center = (h_min + h_max) * 0.5;
        let skirt = super::core::clipmap_camera_config(&params.camera_mode)
            .map(|config| config.ring_resolution as f32 * 0.001 * params.z_scale.abs())
            .unwrap_or(0.0);
        let vertices = clipmap
            .vertices
            .iter()
            .map(|vertex| {
                let height = sample_height_bilinear(heightmap, height_dims, vertex.uv);
                let skirt_offset = if vertex.is_skirt() { skirt } else { 0.0 };
                [
                    vertex.position[0],
                    vertex.position[1],
                    (height - h_center) * params.z_scale - skirt_offset,
                ]
            })
            .collect::<Vec<_>>();
        let (eye, view, proj) = super::TerrainScene::build_camera_matrices(params);
        let selected = cpu_lod_select(
            tiles,
            proj * view,
            eye,
            lod_config,
            (
                (h_min - h_center) * params.z_scale - skirt,
                (h_max - h_center) * params.z_scale,
            ),
        );
        let tile_indices = tiles
            .iter()
            .enumerate()
            .map(|(index, tile)| (tile.tile_id, index))
            .collect::<std::collections::HashMap<_, _>>();
        let mut indices = Vec::new();
        let mut identities = Vec::new();
        for visible in selected.visible_tiles {
            let Some(&tile_index) = tile_indices.get(&visible.tile_id) else {
                continue;
            };
            let template_index =
                tile_index * variant_count as usize + visible.selected_lod as usize;
            let Some(template) = templates.get(template_index) else {
                continue;
            };
            let source = &clipmap.indices[template.first_index as usize
                ..(template.first_index + template.index_count) as usize];
            for (primitive, triangle) in source.chunks_exact(3).enumerate() {
                indices.push([triangle[0], triangle[1], triangle[2]]);
                identities.push((
                    ((visible.selected_lod & 0xf) << 12) | (template.tile_id & 0xfff),
                    primitive as u32 & 0xffff,
                ));
            }
        }
        let mesh = crate::accel::cpu_bvh::MeshCPU::new(vertices, indices);
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(
            &mesh,
            &crate::accel::cpu_bvh::BuildOptions::default(),
        )?;
        Ok(Self {
            mesh,
            bvh,
            identities,
            inv_view_proj: (proj * view).inverse().to_cols_array_2d(),
            viewport: params.size_px,
        })
    }

    fn pick(&self, pixels: &[(u32, u32)]) -> Vec<Option<(u32, u32)>> {
        pixels
            .iter()
            .map(|&(x, y)| {
                let ray = crate::picking::unproject_cursor(
                    x,
                    y,
                    self.viewport.0,
                    self.viewport.1,
                    self.inv_view_proj,
                );
                self.intersect(&ray)
            })
            .collect()
    }

    fn intersect(&self, ray: &crate::picking::Ray) -> Option<(u32, u32)> {
        let mut closest = f32::INFINITY;
        let mut identity = None;
        // `build_recursive` appends each parent after its children, so the
        // root is the final node rather than node zero. Starting at node zero
        // only traverses the first leaf and made the CPU oracle report misses
        // for every pixel outside that leaf.
        let root = self.bvh.nodes.len().checked_sub(1)? as u32;
        let mut stack = vec![root];
        while let Some(node_index) = stack.pop() {
            let node = self.bvh.nodes.get(node_index as usize)?;
            if !ray_aabb(ray, node.aabb_min, node.aabb_max, closest) {
                continue;
            }
            if node.is_leaf() {
                for offset in 0..node.right {
                    let reordered = *self.bvh.tri_indices.get((node.left + offset) as usize)?;
                    let (v0, v1, v2) = self.mesh.get_triangle(reordered as usize)?;
                    if let Some(distance) = ray_triangle(ray, v0, v1, v2) {
                        if distance < closest {
                            closest = distance;
                            identity = self.identities.get(reordered as usize).copied();
                        }
                    }
                }
            } else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        identity
    }
}

fn sample_height_bilinear(data: &[f32], dims: (u32, u32), uv: [f32; 2]) -> f32 {
    let fx = uv[0].clamp(0.0, 1.0) * dims.0.saturating_sub(1) as f32;
    let fy = uv[1].clamp(0.0, 1.0) * dims.1.saturating_sub(1) as f32;
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(dims.0.saturating_sub(1) as usize);
    let y1 = (y0 + 1).min(dims.1.saturating_sub(1) as usize);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let width = dims.0 as usize;
    let top = data[y0 * width + x0] * (1.0 - tx) + data[y0 * width + x1] * tx;
    let bottom = data[y1 * width + x0] * (1.0 - tx) + data[y1 * width + x1] * tx;
    top * (1.0 - ty) + bottom * ty
}

fn ray_aabb(ray: &crate::picking::Ray, min: [f32; 3], max: [f32; 3], limit: f32) -> bool {
    let mut near: f32 = 0.0;
    let mut far = limit;
    for axis in 0..3 {
        let inv = 1.0 / ray.direction[axis];
        let mut a = (min[axis] - ray.origin[axis]) * inv;
        let mut b = (max[axis] - ray.origin[axis]) * inv;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        near = near.max(a);
        far = far.min(b);
        if near > far {
            return false;
        }
    }
    true
}

fn ray_triangle(
    ray: &crate::picking::Ray,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<f32> {
    let origin = glam::Vec3::from_array(ray.origin);
    let direction = glam::Vec3::from_array(ray.direction);
    let a = glam::Vec3::from_array(v0);
    let edge1 = glam::Vec3::from_array(v1) - a;
    let edge2 = glam::Vec3::from_array(v2) - a;
    let p = direction.cross(edge2);
    let det = edge1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = det.recip();
    let t = origin - a;
    let u = t.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(edge1);
    let v = direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(q) * inv_det;
    (distance > 0.0).then_some(distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_oracle_traverses_the_postorder_bvh_root() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..6u32 {
            let x = triangle as f32 * 2.0;
            let first = vertices.len() as u32;
            vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]]);
            indices.push([first, first + 1, first + 2]);
        }
        let mesh = crate::accel::cpu_bvh::MeshCPU::new(vertices, indices);
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(
            &mesh,
            &crate::accel::cpu_bvh::BuildOptions::default(),
        )
        .expect("build test BVH");
        assert!(bvh.nodes.len() > 1);
        assert!(bvh.nodes[0].is_leaf());
        let oracle = CpuVisibilityOracle {
            mesh,
            bvh,
            identities: (0..6).map(|triangle| (triangle, 0)).collect(),
            inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            viewport: (1, 1),
        };

        assert_eq!(
            oracle.intersect(&crate::picking::Ray::new(
                [10.25, 0.25, 1.0],
                [0.0, 0.0, -1.0],
            )),
            Some((5, 0)),
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VisibilityCounters {
    visible_pixels: u32,
    feedback_records: u32,
    material_invocations: u32,
    background_pixels: u32,
    fallback_texels: u32,
    forward_material_invocations: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibilityStats {
    pub visible_pixels: u32,
    /// Feedback records emitted by the most recent frame.
    pub feedback_records: u32,
    /// Feedback records emitted by the visibility-resolve path.
    pub visibility_feedback_records: u32,
    /// Feedback records emitted by the forward path, including overdraw.
    pub forward_feedback_records: u32,
    pub material_invocations: u32,
    pub background_pixels: u32,
    pub fallback_texels: u32,
    pub forward_material_invocations: u32,
}

static LAST_STATS: OnceLock<Mutex<VisibilityStats>> = OnceLock::new();

pub fn publish_stats(stats: VisibilityStats) {
    if let Ok(mut current) = LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
    {
        let mut stats = stats;
        if stats.forward_material_invocations > 0 {
            stats.forward_feedback_records = stats.feedback_records;
            stats.visibility_feedback_records = current.visibility_feedback_records;
        } else if stats.material_invocations > 0 {
            stats.visibility_feedback_records = stats.feedback_records;
            stats.forward_feedback_records = current.forward_feedback_records;
            stats.forward_material_invocations = current.forward_material_invocations;
        }
        *current = stats;
    }
}

pub fn latest_stats() -> VisibilityStats {
    LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
        .map(|stats| *stats)
        .unwrap_or_default()
}

pub struct TerrainVisibilityBuffer {
    width: u32,
    height: u32,
    _texture: TrackedTexture,
    view: wgpu::TextureView,
    stats_buffer: TrackedBuffer,
    stats_readback: TrackedBuffer,
    stats_bind_group: wgpu::BindGroup,
    stats_pipeline: wgpu::ComputePipeline,
    staged: bool,
}

impl TerrainVisibilityBuffer {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        frame_counters: &wgpu::Buffer,
    ) -> RenderResult<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.visibility.ids"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let stats_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let stats_readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats_readback"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.visibility.stats.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let stats_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.visibility.stats.bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_counters.as_entire_binding(),
                },
            ],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_visbuffer_resolve",
            include_str!("../../shaders/terrain_visbuffer_resolve.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.visibility.stats.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let stats_pipeline = crate::core::shader_registry::with_error_scope(
            device,
            "terrain.visibility.stats.pipeline",
            || {
                crate::core::shader_registry::create_compute_pipeline_scoped(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("terrain.visibility.stats.pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: "cs_main",
                    },
                )
            },
        );
        Ok(Self {
            width,
            height,
            _texture: texture,
            view,
            stats_buffer,
            stats_readback,
            stats_bind_group,
            stats_pipeline,
            staged: false,
        })
    }

    pub fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width.max(1) && self.height == height.max(1)
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self._texture
    }

    pub fn stage_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.stats_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.visibility.stats.pass"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("terrain_visbuffer_resolve");
            pass.set_pipeline(&self.stats_pipeline);
            pass.set_bind_group(0, &self.stats_bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_readback,
            0,
            std::mem::size_of::<VisibilityCounters>() as u64,
        );
        self.staged = true;
    }

    pub fn finish_frame(&mut self, device: &wgpu::Device) -> RenderResult<VisibilityStats> {
        if !self.staged {
            return Ok(VisibilityStats::default());
        }
        let slice = self.stats_readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::render("visibility stats callback dropped"))?
            .map_err(|error| {
                RenderError::render(format!("visibility stats map failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range();
        let counters = bytemuck::pod_read_unaligned::<VisibilityCounters>(&mapped);
        drop(mapped);
        self.stats_readback.unmap();
        self.staged = false;
        Ok(VisibilityStats {
            visible_pixels: counters.visible_pixels,
            feedback_records: counters.feedback_records,
            visibility_feedback_records: 0,
            forward_feedback_records: 0,
            material_invocations: counters.material_invocations,
            background_pixels: counters.background_pixels,
            fallback_texels: counters.fallback_texels,
            forward_material_invocations: counters.forward_material_invocations,
        })
    }
}

#[cfg(feature = "extension-module")]
impl super::TerrainScene {
    pub(super) fn create_visibility_resolve_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        // The runtime resolve entry point is `fs_visibility_geometry`. It only
        // reads the visibility ID texture from group 7; vertices, indices,
        // templates, meta, and sampled depth belong exclusively to the static
        // fullscreen reconstruction helper. Do not put those resources in the
        // geometry bind group: binding the live depth attachment as a sampled
        // texture in this same depth-equal pass is a WebGPU usage conflict,
        // even on backends that prune unused entry-point resources.
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.visibility.geometry_resolve.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
    }

    pub(super) fn visibility_resolve_bind_group(&self) -> anyhow::Result<wgpu::BindGroup> {
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("terrain visibility buffer not initialized"))?;
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.visibility.geometry_resolve.bind_group"),
            layout: &self.visibility_resolve_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(buffer.view()),
            }],
        }))
    }

    pub(super) fn ensure_visibility_buffer(&self, width: u32, height: u32) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        if visibility
            .as_ref()
            .is_none_or(|buffer| !buffer.matches(width, height))
        {
            *visibility = Some(
                TerrainVisibilityBuffer::new(
                    self.device.as_ref(),
                    width,
                    height,
                    &self.vt_frame_counters_buffer,
                )
                .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(())
    }

    pub(super) fn stage_visibility_stats(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain visibility buffer not initialized"))?;
        buffer.stage_stats(encoder);
        Ok(())
    }

    pub(super) fn finish_visibility_frame(&self) -> anyhow::Result<VisibilityStats> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let Some(buffer) = visibility.as_mut() else {
            let stats = VisibilityStats::default();
            publish_stats(stats);
            return Ok(stats);
        };
        let stats = buffer
            .finish_frame(self.device.as_ref())
            .map_err(anyhow::Error::msg)?;
        publish_stats(stats);
        Ok(stats)
    }

    pub(super) fn pick_visibility_pixels(
        &self,
        pixels: &[(u32, u32)],
    ) -> anyhow::Result<Vec<Option<(u32, u32)>>> {
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no completed visibility render is available"))?;
        let picking =
            crate::picking::UnifiedPickingSystem::new(self.device.clone(), self.queue.clone());
        picking
            .pick_visibility_pixels(buffer.texture(), buffer.width, buffer.height, pixels)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn pick_visibility_pixels_cpu(
        &self,
        pixels: &[(u32, u32)],
    ) -> anyhow::Result<Vec<Option<(u32, u32)>>> {
        let oracle = self
            .cpu_visibility_oracle
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain CPU visibility oracle mutex poisoned"))?;
        let oracle = oracle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no CPU BVH visibility oracle is available"))?;
        Ok(oracle.pick(pixels))
    }
}
