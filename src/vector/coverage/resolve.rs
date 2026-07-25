use super::binning::CoverageBins;
use super::raster::CoverageRasterResources;
use super::types::CoverageGeometry;
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::BufferInitDescriptor;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    extent_layers: [u32; 4],
    dispatch: [u32; 4],
}

pub struct CoverageResolveResources {
    /// Premultiplied linear RGBA, one `vec4<f32>` per pixel.
    pub output: TrackedBuffer,
    pub output_bytes: u64,
    pub allocation_bytes: u64,
    pub active_pixel_count: u32,
    _colors: TrackedBuffer,
    _params: TrackedBuffer,
    bind_group: wgpu::BindGroup,
}

pub struct CoverageResolver {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CoverageResolver {
    pub fn new(device: &wgpu::Device) -> Self {
        let source = resolve_shader_source();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "vector_coverage_resolve.wgsl",
            &source,
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Resolve.BindGroupLayout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                uniform_entry(2),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Resolve.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vf.Vector.Coverage.Resolve.Pipeline"),
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
        raster: &CoverageRasterResources,
    ) -> Result<CoverageResolveResources, RenderError> {
        if bins.layout.layer_count as usize != geometry.layers.len() {
            return Err(RenderError::Upload(
                "vector_coverage_resolve_layout: bin and geometry layer counts differ".into(),
            ));
        }
        let pixel_count = u64::from(geometry.width)
            .checked_mul(u64::from(geometry.height))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_resolve_budget: pixel-count overflow".into())
            })?;
        let output_bytes = pixel_count.checked_mul(16).ok_or_else(|| {
            RenderError::Budget("vector_coverage_resolve_budget: output-byte overflow".into())
        })?;
        let color_bytes = (geometry.layers.len() as u64)
            .checked_mul(16)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_resolve_budget: color-byte overflow".into())
            })?;
        let allocation_bytes = raster
            .allocation_bytes
            .checked_add(output_bytes)
            .and_then(|bytes| bytes.checked_add(color_bytes))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<ResolveParams>() as u64))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_resolve_budget: total-byte overflow".into())
            })?;
        if allocation_bytes > COVERAGE_MEMORY_BUDGET {
            return Err(RenderError::Budget(format!(
                "vector_coverage_resolve_budget: required={allocation_bytes} \
                 budget={COVERAGE_MEMORY_BUDGET}; refusing to truncate"
            )));
        }

        let colors: Vec<[f32; 4]> = geometry.layers.iter().map(|layer| layer.color).collect();
        let color_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.LayerColors"),
                contents: bytemuck::cast_slice(&colors),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let params_value = ResolveParams {
            extent_layers: [
                geometry.width,
                geometry.height,
                bins.layout.layer_count,
                u32::try_from(pixel_count).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_resolve_budget: pixel count exceeds u32".into(),
                    )
                })?,
            ],
            dispatch: [raster.resolve_pixel_count, 0, 0, 0],
        };
        let params = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.ResolveParams"),
                contents: bytemuck::bytes_of(&params_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let output = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.ResolvedLinearRgba"),
                size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Coverage.Resolve.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                entry(0, &raster.coverage),
                entry(1, &color_buffer),
                entry(2, &params),
                entry(3, &output),
                entry(4, &bins.overflow),
                entry(5, &raster.resolve_pixels),
            ],
        });
        Ok(CoverageResolveResources {
            output,
            output_bytes,
            allocation_bytes,
            active_pixel_count: raster.resolve_pixel_count,
            _colors: color_buffer,
            _params: params,
            bind_group,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, resources: &CoverageResolveResources) {
        encoder.clear_buffer(&resources.output, 0, None);
        if resources.active_pixel_count == 0 {
            return;
        }
        crate::core::shader_registry::record_shader_use("vector_coverage_resolve.wgsl");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vf.Vector.Coverage.Resolve.Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(resources.active_pixel_count.div_ceil(64), 1, 1);
    }
}

/// Fixed-order CPU mirror of the resolve kernel.
///
/// Coverage is already the union/conflation result for a whole layer, so each
/// layer is composited exactly once. Layer order is ascending input order,
/// matching the existing vector submission convention: later layers are over.
#[cfg(test)]
pub(super) fn resolve_coverage_cpu(geometry: &CoverageGeometry, coverage: &[f32]) -> Vec<[f32; 4]> {
    let pixel_count = geometry.width as usize * geometry.height as usize;
    assert_eq!(coverage.len(), pixel_count * geometry.layers.len());
    let mut output = vec![[0.0_f32; 4]; pixel_count];
    for pixel in 0..pixel_count {
        let mut accumulated = [0.0_f32; 4];
        for (layer_index, layer) in geometry.layers.iter().enumerate() {
            let alpha =
                (layer.color[3] * coverage[layer_index * pixel_count + pixel]).clamp(0.0, 1.0);
            let remaining = 1.0 - alpha;
            accumulated[0] = layer.color[0] * alpha + accumulated[0] * remaining;
            accumulated[1] = layer.color[1] * alpha + accumulated[1] * remaining;
            accumulated[2] = layer.color[2] * alpha + accumulated[2] * remaining;
            accumulated[3] = alpha + accumulated[3] * remaining;
        }
        output[pixel] = accumulated;
    }
    output
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

pub(super) fn resolve_shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../../shaders/includes/determinism.wgsl"),
        include_str!("../../shaders/vector_coverage_resolve.wgsl")
    )
}
