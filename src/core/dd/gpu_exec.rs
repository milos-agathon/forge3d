use super::gpu_report::TwoProdVariant;
use super::DD;
use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::tracked_create_buffer;
use crate::core::shader_registry::{create_compute_pipeline_scoped, create_labeled_shader_module};
use bytemuck::{Pod, Zeroable};

const DETERMINISM: &str = include_str!("../../shaders/includes/determinism.wgsl");
const DD_SHADER: &str = include_str!("../../shaders/includes/dd.wgsl");
const HARNESS: &str = include_str!("../../shaders/dd_harness.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    offset: u32,
    count: u32,
    phase: u32,
    op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct HarnessOutput {
    pub primary: DD,
    pub product: DD,
}

pub(super) struct HarnessPipeline {
    pipeline: wgpu::ComputePipeline,
    label: String,
}

fn assembled_source(variant: TwoProdVariant) -> String {
    let selected = match variant {
        TwoProdVariant::Fma => "two_prod_fma",
        TwoProdVariant::Split => "two_prod_split",
    };
    // Bare bitcasts and constant identities are optimized through, while a
    // storage/atomic round-trips cause DX12 to discard the dispatch. Rebuilding
    // the sign and magnitude prevents DXC from reassociating across the
    // correctly-rounded f32 steps without changing any finite value.
    let dd = DD_SHADER
        .replace("__DD_TWO_PROD_CALL__", selected)
        .replace(
            "__DD_BARRIER_BODY__",
            "let bits = bitcast<u32>(value);\n    let magnitude = bitcast<f32>(bits & 0x7fffffffu);\n    return select(magnitude, -magnitude, (bits & 0x80000000u) != 0u);",
        );
    format!("{DETERMINISM}\n{}\n{}", dd, HARNESS)
}

impl HarnessPipeline {
    pub(super) fn new(device: &wgpu::Device, variant: TwoProdVariant) -> RenderResult<Self> {
        let label = format!("dupla.dd.two_prod.{}", variant.as_str());
        let source = assembled_source(variant);
        let parsed = naga::front::wgsl::parse_str(&source)
            .map_err(|error| RenderError::render(format!("DD WGSL parse failed: {error}")))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .map_err(|error| RenderError::render(format!("DD WGSL validation failed: {error}")))?;
        let module = create_labeled_shader_module(device, &label, &source);
        let pipeline = create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: None,
                module: &module,
                entry_point: "main",
            },
        );
        if let Some(validation) = crate::core::degradation::degradations_snapshot()
            .into_iter()
            .find(|item| item.kind == "validation_error" && item.name == label)
        {
            return Err(RenderError::render(format!(
                "DD harness pipeline validation failed: {}",
                validation.consequence
            )));
        }
        Ok(Self { pipeline, label })
    }

    pub(super) fn label(&self) -> &str {
        &self.label
    }

    pub(super) fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        offset: u32,
        count: u32,
        phase: u32,
        op: u32,
        certify: bool,
    ) -> RenderResult<Vec<HarnessOutput>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let params = Params {
            offset,
            count,
            phase,
            op,
        };
        let params_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("dupla.dd.params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let size = count as u64 * std::mem::size_of::<HarnessOutput>() as u64;
        let output = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("dupla.dd.output"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("dupla.dd.readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dupla.dd.bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dupla.dd.encoder"),
        });
        let mut timing = certify.then(crate::core::gpu_timing::OneShotTiming::for_current_device);
        let scope = timing
            .as_mut()
            .and_then(|timer| timer.begin(&mut encoder, "dupla.dd_harness"));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dupla.dd.harness"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        if let Some(timer) = timing.as_mut() {
            timer.end(&mut encoder, scope, 1);
            timer.resolve(&mut encoder);
            crate::core::shader_registry::record_shader_use(&self.label);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        if let Some(timer) = timing {
            if !timer.record_into_certificate() {
                crate::core::certificate::record_pass("dupla.dd_harness", 0.0, 1);
            }
        }

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| RenderError::readback(format!("DD map callback failed: {error}")))?
            .map_err(|error| RenderError::readback(format!("DD readback failed: {error}")))?;
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, HarnessOutput>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Ok(values)
    }
}
