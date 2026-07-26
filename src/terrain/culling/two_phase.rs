use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use crate::core::screen_space_effects::HzbPyramid;
use crate::terrain::clipmap::gpu_lod::{
    ClipmapDrawInstance, DrawIndexedIndirectArgs, GpuLodDrawResources,
};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::sync::{Mutex, OnceLock};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    view_proj: [[f32; 4]; 4],
    height_bounds: [f32; 4],
    viewport: [f32; 4],
    control: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct CullHeader {
    draw_count: u32,
    rejected_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullingStats {
    pub frustum_passing: u32,
    pub phase1_drawn: u32,
    pub phase1_rejected: u32,
    pub phase2_recovered: u32,
    pub final_drawn: u32,
    pub culled: u32,
    pub projection_bypassed: u32,
    pub background_bypassed: u32,
}

static LAST_STATS: OnceLock<Mutex<CullingStats>> = OnceLock::new();

pub fn publish_stats(stats: CullingStats) {
    if let Ok(mut current) = LAST_STATS
        .get_or_init(|| Mutex::new(CullingStats::default()))
        .lock()
    {
        *current = stats;
    }
}

pub fn latest_stats() -> CullingStats {
    LAST_STATS
        .get_or_init(|| Mutex::new(CullingStats::default()))
        .lock()
        .map(|stats| *stats)
        .unwrap_or_default()
}

pub struct CullDrawResources {
    pub indirect_buffer: TrackedBuffer,
    pub instance_buffer: TrackedBuffer,
    pub max_draw_count: u32,
    header: TrackedBuffer,
}

pub struct TwoPhaseTerrainCuller {
    pyramids: [HzbPyramid; 2],
    previous_index: usize,
    previous_valid: bool,
    previous_view_proj: Mat4,
    pipeline: wgpu::ComputePipeline,
    phase1_params: TrackedBuffer,
    phase2_params: TrackedBuffer,
    phase1_bind_groups: [wgpu::BindGroup; 2],
    phase2_bind_groups: [wgpu::BindGroup; 2],
    phase1: CullDrawResources,
    phase2: CullDrawResources,
    rejected_indices: TrackedBuffer,
    stats_readback: TrackedBuffer,
    width: u32,
    height: u32,
    max_mip: u32,
    stats: CullingStats,
}

impl TwoPhaseTerrainCuller {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        input: &GpuLodDrawResources,
    ) -> RenderResult<Self> {
        let pyramids = [
            HzbPyramid::new(device, width, height)?,
            HzbPyramid::new(device, width, height)?,
        ];
        let max_mip = pyramids[0].mip_count.saturating_sub(1);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "hzb_cull",
            include_str!("../../shaders/hzb_cull.wgsl"),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.hzb_cull.layout"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(9, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.hzb_cull.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("terrain.hzb_cull.pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_main",
            },
        );
        let zero_params = CullParams::zeroed();
        let phase1_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.hzb_cull.phase1_params"),
                contents: bytemuck::bytes_of(&zero_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let phase2_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.hzb_cull.phase2_params"),
                contents: bytemuck::bytes_of(&zero_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let phase1 = create_draw_resources(device, input.max_draw_count, "phase1")?;
        let phase2 = create_draw_resources(device, input.max_draw_count, "phase2")?;
        let rejected_indices = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.hzb_cull.rejected_indices"),
                size: u64::from(input.max_draw_count) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let stats_readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.hzb_cull.stats_readback"),
                size: 48,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let pyramid_views = [pyramids[0].texture_view(), pyramids[1].texture_view()];
        let phase1_bind_groups = [
            create_bind_group(
                device,
                &layout,
                &phase1_params,
                &pyramid_views[0],
                &input.output_header,
                input,
                &phase1,
                &rejected_indices,
                "phase1.previous0",
            ),
            create_bind_group(
                device,
                &layout,
                &phase1_params,
                &pyramid_views[1],
                &input.output_header,
                input,
                &phase1,
                &rejected_indices,
                "phase1.previous1",
            ),
        ];
        let phase2_bind_groups = [
            create_bind_group(
                device,
                &layout,
                &phase2_params,
                &pyramid_views[0],
                &phase1.header,
                input,
                &phase2,
                &rejected_indices,
                "phase2.fresh0",
            ),
            create_bind_group(
                device,
                &layout,
                &phase2_params,
                &pyramid_views[1],
                &phase1.header,
                input,
                &phase2,
                &rejected_indices,
                "phase2.fresh1",
            ),
        ];
        Ok(Self {
            pyramids,
            previous_index: 0,
            previous_valid: false,
            previous_view_proj: Mat4::IDENTITY,
            pipeline,
            phase1_params,
            phase2_params,
            phase1_bind_groups,
            phase2_bind_groups,
            phase1,
            phase2,
            rejected_indices,
            stats_readback,
            width,
            height,
            max_mip,
            stats: CullingStats::default(),
        })
    }

    pub fn phase1(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_view_proj: Mat4,
        height_bounds: (f32, f32),
        first_instance: bool,
    ) -> &CullDrawResources {
        let view_proj = if self.previous_valid {
            self.previous_view_proj
        } else {
            current_view_proj
        };
        let params = self.params(
            view_proj,
            height_bounds,
            1,
            self.previous_valid,
            first_instance,
        );
        queue.write_buffer(&self.phase1_params, 0, bytemuck::bytes_of(&params));
        clear_outputs(encoder, &self.phase1, Some(&self.rejected_indices));
        self.dispatch(encoder, &self.phase1_bind_groups[self.previous_index]);
        &self.phase1
    }

    pub fn build_phase2_hzb(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) -> RenderResult<()> {
        self.pyramids[1 - self.previous_index].build(device, encoder, depth, true)
    }

    pub fn build_previous_hzb(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) -> RenderResult<()> {
        self.pyramids[1 - self.previous_index].build(device, encoder, depth, false)
    }

    pub fn phase2(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_view_proj: Mat4,
        height_bounds: (f32, f32),
        first_instance: bool,
    ) -> &CullDrawResources {
        let fresh_index = 1 - self.previous_index;
        let params = self.params(current_view_proj, height_bounds, 2, true, first_instance);
        queue.write_buffer(&self.phase2_params, 0, bytemuck::bytes_of(&params));
        clear_outputs(encoder, &self.phase2, None);
        self.dispatch(encoder, &self.phase2_bind_groups[fresh_index]);
        &self.phase2
    }

    pub fn stage_stats(&self, encoder: &mut wgpu::CommandEncoder, input: &GpuLodDrawResources) {
        encoder.copy_buffer_to_buffer(&input.output_header, 0, &self.stats_readback, 0, 16);
        encoder.copy_buffer_to_buffer(&self.phase1.header, 0, &self.stats_readback, 16, 16);
        encoder.copy_buffer_to_buffer(&self.phase2.header, 0, &self.stats_readback, 32, 16);
    }

    pub fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        current_view_proj: Mat4,
    ) -> RenderResult<CullingStats> {
        let slice = self.stats_readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::render("HZB culling stats callback dropped"))?
            .map_err(|error| {
                RenderError::render(format!("HZB culling stats map failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range();
        let input = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[0..16]);
        let phase1 = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[16..32]);
        let phase2 = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[32..48]);
        drop(mapped);
        self.stats_readback.unmap();
        self.stats = CullingStats {
            frustum_passing: input.draw_count,
            phase1_drawn: phase1.draw_count,
            phase1_rejected: phase1.rejected_count,
            phase2_recovered: phase2.draw_count,
            final_drawn: phase1.draw_count + phase2.draw_count,
            culled: phase1.rejected_count.saturating_sub(phase2.draw_count),
            projection_bypassed: phase1._pad0,
            background_bypassed: phase1._pad1,
        };
        self.previous_view_proj = current_view_proj;
        self.previous_valid = true;
        self.previous_index = 1 - self.previous_index;
        Ok(self.stats)
    }

    pub fn stats(&self) -> CullingStats {
        self.stats
    }

    pub fn phase1_resources(&self) -> &CullDrawResources {
        &self.phase1
    }

    pub fn phase2_resources(&self) -> &CullDrawResources {
        &self.phase2
    }

    fn params(
        &self,
        view_proj: Mat4,
        height_bounds: (f32, f32),
        phase: u32,
        valid: bool,
        first_instance: bool,
    ) -> CullParams {
        CullParams {
            view_proj: view_proj.to_cols_array_2d(),
            height_bounds: [height_bounds.0, height_bounds.1, 0.0, 0.0],
            viewport: [
                self.width as f32,
                self.height as f32,
                self.max_mip as f32,
                u32::from(valid) as f32,
            ],
            control: [phase, u32::from(first_instance), 0, 0],
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup) {
        crate::core::shader_registry::record_shader_use("hzb_cull");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.hzb_cull.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(self.phase1.max_draw_count.div_ceil(64), 1, 1);
    }
}

impl CullDrawResources {
    pub fn count_buffer(&self) -> &TrackedBuffer {
        &self.header
    }
}

fn create_draw_resources(
    device: &wgpu::Device,
    max_draw_count: u32,
    phase: &str,
) -> RenderResult<CullDrawResources> {
    let indirect_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.indirect")),
            size: u64::from(max_draw_count) * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let instance_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.instances")),
            size: u64::from(max_draw_count) * std::mem::size_of::<ClipmapDrawInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let header = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.header")),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    Ok(CullDrawResources {
        indirect_buffer,
        instance_buffer,
        max_draw_count,
        header,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &TrackedBuffer,
    hzb: &wgpu::TextureView,
    input_header: &TrackedBuffer,
    input: &GpuLodDrawResources,
    output: &CullDrawResources,
    rejected: &TrackedBuffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            entry(0, params),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hzb),
            },
            entry(2, input_header),
            entry(3, &input.output_tiles),
            entry(4, &input.indirect_buffer),
            entry(5, &input.instance_buffer),
            entry(6, &output.indirect_buffer),
            entry(7, &output.instance_buffer),
            entry(8, rejected),
            entry(9, &output.header),
        ],
    })
}

fn entry(binding: u32, buffer: &TrackedBuffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn clear_outputs(
    encoder: &mut wgpu::CommandEncoder,
    output: &CullDrawResources,
    rejected: Option<&TrackedBuffer>,
) {
    encoder.clear_buffer(&output.header, 0, None);
    encoder.clear_buffer(&output.indirect_buffer, 0, None);
    encoder.clear_buffer(&output.instance_buffer, 0, None);
    if let Some(rejected) = rejected {
        encoder.clear_buffer(rejected, 0, None);
    }
}

#[cfg(test)]
mod tests {
    fn phase1_rejects(nearest_tile_depth: f32, covered_depths: &[f32]) -> bool {
        let closest_occluder = covered_depths.iter().copied().fold(1.0, f32::min);
        closest_occluder < 1.0 && nearest_tile_depth > closest_occluder
    }

    fn phase2_occluded(nearest_tile_depth: f32, covered_depths: &[f32]) -> bool {
        let farthest_occluder = covered_depths.iter().copied().fold(0.0, f32::max);
        farthest_occluder < 1.0 && nearest_tile_depth > farthest_occluder
    }

    #[test]
    fn fresh_hzb_recovers_previous_frame_false_negative() {
        let previous = [0.2, 0.2, 0.2, 0.2];
        let fresh = [0.2, 0.2, 1.0, 1.0];
        assert!(phase1_rejects(0.8, &previous));
        assert!(!phase2_occluded(0.8, &fresh));
    }

    #[test]
    fn max_depth_test_only_culls_when_every_covered_pixel_is_closer() {
        assert!(phase2_occluded(0.8, &[0.2, 0.4, 0.6]));
        assert!(!phase2_occluded(0.8, &[0.2, 1.0, 0.6]));
    }
}
