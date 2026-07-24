use super::binning::CoverageBins;
use super::math::{primitive_active, primitive_x};
use super::types::{CoverageGeometry, FillRule, COVERAGE_TILE_SIZE};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::BufferInitDescriptor;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;
const MAX_ACTIVE_PRIMITIVES: u32 = 96;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterParams {
    extent_tiles: [u32; 4],
    layers_capacity: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerRuleRecord {
    values: [u32; 4],
}

pub struct CoverageRasterResources {
    pub coverage: TrackedBuffer,
    pub coverage_bytes: u64,
    pub allocation_bytes: u64,
    _baselines: TrackedBuffer,
    _rules: TrackedBuffer,
    _params: TrackedBuffer,
    bind_group: wgpu::BindGroup,
}

pub struct CoverageRasterizer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CoverageRasterizer {
    pub fn new(device: &wgpu::Device) -> Self {
        let source = raster_shader_source();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "vector_coverage_raster.wgsl",
            &source,
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Raster.BindGroupLayout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                uniform_entry(5),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Raster.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vf.Vector.Coverage.Raster.Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        );
        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        geometry: &CoverageGeometry,
        bins: &CoverageBins,
    ) -> Result<CoverageRasterResources, RenderError> {
        if bins.layout.layer_count as usize != geometry.layers.len() {
            return Err(RenderError::Upload(
                "vector_coverage_raster_layout: bin and geometry layer counts differ".into(),
            ));
        }
        let baselines = build_tile_baselines(geometry, bins.layout.tile_columns);
        let rules: Vec<_> = geometry
            .layers
            .iter()
            .map(|layer| LayerRuleRecord {
                values: [layer.fill_rule as u32, 0, 0, 0],
            })
            .collect();
        let pixel_count = u64::from(geometry.width)
            .checked_mul(u64::from(geometry.height))
            .and_then(|count| count.checked_mul(u64::from(bins.layout.layer_count)))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: pixel-count overflow".into())
            })?;
        let coverage_bytes = pixel_count.checked_mul(4).ok_or_else(|| {
            RenderError::Budget("vector_coverage_raster_budget: coverage-byte overflow".into())
        })?;
        let baseline_bytes = (baselines.len() as u64).checked_mul(4).ok_or_else(|| {
            RenderError::Budget("vector_coverage_raster_budget: baseline-byte overflow".into())
        })?;
        let rule_bytes = (rules.len() as u64)
            .checked_mul(std::mem::size_of::<LayerRuleRecord>() as u64)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: rule-byte overflow".into())
            })?;
        let total_bytes = bins
            .layout
            .allocation_bytes
            .checked_add(coverage_bytes)
            .and_then(|bytes| bytes.checked_add(baseline_bytes))
            .and_then(|bytes| bytes.checked_add(rule_bytes))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<RasterParams>() as u64))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: total-byte overflow".into())
            })?;
        if total_bytes > COVERAGE_MEMORY_BUDGET {
            return Err(RenderError::Budget(format!(
                "vector_coverage_raster_budget: required={total_bytes} \
                 budget={COVERAGE_MEMORY_BUDGET}; refusing to truncate"
            )));
        }

        let baseline_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.TileBaselines"),
                contents: bytemuck::cast_slice(&baselines),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let rule_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.LayerRules"),
                contents: bytemuck::cast_slice(&rules),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let params_value = RasterParams {
            extent_tiles: [
                geometry.width,
                geometry.height,
                bins.layout.tile_columns,
                bins.layout.tile_rows,
            ],
            layers_capacity: [
                bins.layout.layer_count,
                bins.layout.tile_capacity,
                COVERAGE_TILE_SIZE,
                MAX_ACTIVE_PRIMITIVES,
            ],
        };
        let params = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.RasterParams"),
                contents: bytemuck::bytes_of(&params_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let coverage = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.Output"),
                size: coverage_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Coverage.Raster.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                entry(0, &bins.primitive_buffer),
                entry(1, &bins.tile_counts),
                entry(2, &bins.tile_indices),
                entry(3, &baseline_buffer),
                entry(4, &rule_buffer),
                entry(5, &params),
                entry(6, &coverage),
                entry(7, &bins.overflow),
            ],
        });
        Ok(CoverageRasterResources {
            coverage,
            coverage_bytes,
            allocation_bytes: total_bytes,
            _baselines: baseline_buffer,
            _rules: rule_buffer,
            _params: params,
            bind_group,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &CoverageRasterResources,
        geometry: &CoverageGeometry,
    ) {
        encoder.clear_buffer(&resources.coverage, 0, None);
        crate::core::shader_registry::record_shader_use("vector_coverage_raster.wgsl");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vf.Vector.Coverage.Raster.Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(
            geometry.width.div_ceil(8),
            geometry.height.div_ceil(8),
            geometry.layers.len() as u32,
        );
    }
}

fn build_tile_baselines(geometry: &CoverageGeometry, tile_columns: u32) -> Vec<i32> {
    let row_count = geometry.layers.len() * geometry.height as usize;
    let mut rows: Vec<Vec<(f64, i32, u32)>> = vec![Vec::new(); row_count];
    let height = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
        f64::from(geometry.height),
        0.0,
        0.0,
    ))
    .x;
    for primitive in &geometry.primitives {
        let min_row = primitive.bounds[1].floor().max(0.0) as u32;
        let max_row = primitive.bounds[3].ceil().max(0.0).min(height) as u32;
        for row in min_row..max_row {
            let y = f64::from(row) + 0.5;
            if primitive_active(primitive, y) {
                rows[primitive.layer() as usize * geometry.height as usize + row as usize].push((
                    primitive_x(primitive, y),
                    primitive.winding(),
                    primitive.stable_id(),
                ));
            }
        }
    }

    let mut result =
        vec![0_i32; geometry.layers.len() * geometry.height as usize * tile_columns as usize];
    for (row_index, crossings) in rows.iter_mut().enumerate() {
        crossings.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.2.cmp(&right.2))
        });
        let layer = row_index / geometry.height as usize;
        let rule = geometry.layers[layer].fill_rule;
        let row = row_index % geometry.height as usize;
        let mut cursor = 0;
        let mut state = 0_i32;
        for tile_x in 0..tile_columns {
            let boundary_x = f64::from(tile_x * COVERAGE_TILE_SIZE);
            while cursor < crossings.len() && crossings[cursor].0 < boundary_x {
                state += match rule {
                    FillRule::NonZero => crossings[cursor].1,
                    FillRule::EvenOdd => 1,
                };
                cursor += 1;
            }
            result[(layer * geometry.height as usize + row) * tile_columns as usize
                + tile_x as usize] = state;
        }
    }
    result
}

fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
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

pub(super) fn raster_shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../../shaders/includes/determinism.wgsl"),
        include_str!("../../shaders/vector_coverage_raster.wgsl")
    )
}
