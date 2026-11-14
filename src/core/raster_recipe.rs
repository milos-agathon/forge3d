use crate::core::gpu::{try_ctx, wait_for_device_idle, GpuContext};
use crate::core::overlay_layer::OverlayLayer;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer, TrackedTexture,
};
use crate::core::shader_registry::{
    create_labeled_shader_module, try_create_compute_pipeline_scoped,
};
use crate::terrain::ColormapLUT;
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use wgpu::{BufferUsages as BufferUse, TextureUsages as TextureUse};

const INVALID: u32 = 0x7fc0_0000;

fn gpu_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn buffer(ctx: &GpuContext, label: &str, size: u64, usage: BufferUse) -> PyResult<TrackedBuffer> {
    tracked_create_buffer(
        &ctx.device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        },
    )
    .map_err(gpu_error)
}

fn initialized(
    ctx: &GpuContext,
    label: &str,
    data: &[u8],
    usage: BufferUse,
) -> PyResult<TrackedBuffer> {
    tracked_create_buffer_init(
        &ctx.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        },
    )
    .map_err(gpu_error)
}

fn scoped<T>(ctx: &GpuContext, operation: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = operation();
    let validation = pollster::block_on(ctx.device.pop_error_scope());
    let memory = pollster::block_on(ctx.device.pop_error_scope());
    if let Some(error) = validation.or(memory) {
        return Err(gpu_error(error));
    }
    try_ctx()?;
    result
}

#[pyclass(module = "forge3d._forge3d", name = "_ResidentRasterRecipe")]
pub(crate) struct ResidentRasterRecipe {
    normalize: wgpu::ComputePipeline,
    classify: wgpu::ComputePipeline,
    inputs: Option<Inputs>,
    normalization_dispatches: u64,
    classification_dispatches: u64,
    input_upload_bytes: u64,
    parameter_upload_bytes: u64,
    readback_bytes: u64,
}

struct Inputs {
    _inputs: [TrackedBuffer; 3],
    settings: [u32; 6],
    settings_buffer: TrackedBuffer,
    cells: TrackedBuffer,
    texture: Arc<TrackedTexture>,
    counts: Option<TrackedBuffer>,
    hash: Option<[u8; 32]>,
}

impl Inputs {
    fn dispatch(&self) -> (u32, u32) {
        // Matches the shader's 8x8 workgroup.
        (self.settings[0].div_ceil(8), self.settings[1].div_ceil(8))
    }

    fn ready(&self) -> PyResult<&TrackedBuffer> {
        match (&self.hash, &self.counts) {
            (Some(_), Some(counts)) => Ok(counts),
            _ => Err(PyRuntimeError::new_err(
                "Raster recipe has no current classified result",
            )),
        }
    }
}

fn submit(
    ctx: &GpuContext,
    pipe: &wgpu::ComputePipeline,
    bindings: &[(u32, &TrackedBuffer)],
    texture: Option<&wgpu::TextureView>,
    groups: (u32, u32),
) {
    let mut entries: Vec<_> = bindings
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    if let Some(view) = texture {
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raster_recipe.bindings"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raster_recipe.compute"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raster_recipe.compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    // Keep tracked upload and binding resources alive until the dispatch completes.
    wait_for_device_idle(&ctx.device);
}

#[pymethods]
impl ResidentRasterRecipe {
    #[new]
    fn new() -> PyResult<Self> {
        let ctx = try_ctx()?;
        if !crate::core::gpu::is_physical_proof_adapter(
            &ctx.adapter.get_info(),
            ctx.software_fallback,
        ) {
            return Err(PyRuntimeError::new_err(
                "Resident raster recipes require a physical GPU adapter",
            ));
        }
        let (normalize, classify) = scoped(ctx, || {
            let shader = create_labeled_shader_module(
                &ctx.device,
                "raster_recipe.shader",
                include_str!("../shaders/raster_recipe.wgsl"),
            );
            let pipeline = |label, entry_point| {
                try_create_compute_pipeline_scoped(
                    &ctx.device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &shader,
                        entry_point,
                    },
                )
                .map_err(gpu_error)
            };
            Ok((
                pipeline("raster_recipe.normalize", "normalize")?,
                pipeline("raster_recipe.classify", "classify")?,
            ))
        })?;
        Ok(Self {
            normalize,
            classify,
            inputs: None,
            normalization_dispatches: 0,
            classification_dispatches: 0,
            input_upload_bytes: 0,
            parameter_upload_bytes: 0,
            readback_bytes: 0,
        })
    }

    fn replace(
        &mut self,
        a: PyReadonlyArray2<'_, f32>,
        b: PyReadonlyArray2<'_, f32>,
        mask: PyReadonlyArray2<'_, u8>,
        nodata: (Option<f32>, Option<f32>),
    ) -> PyResult<()> {
        if a.shape() != b.shape() || a.shape() != mask.shape() {
            return Err(PyValueError::new_err("Raster input shapes must match"));
        }
        let [height, width] = [a.shape()[0], a.shape()[1]].map(|n| u32::try_from(n).unwrap_or(0));
        let pixels = u64::from(width) * u64::from(height);
        let ctx = try_ctx()?;
        let limits = ctx.device.limits();
        if pixels == 0
            || pixels > u64::from(u32::MAX)
            || width.max(height) > limits.max_texture_dimension_2d
            || width.max(height).div_ceil(8) > limits.max_compute_workgroups_per_dimension
            || pixels * 8
                > u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
        {
            return Err(PyValueError::new_err(
                "Raster grid exceeds negotiated GPU resource limits",
            ));
        }
        let a = bytemuck::cast_slice(a.as_slice()?);
        let b = bytemuck::cast_slice(b.as_slice()?);
        let mask = mask.as_slice()?;
        let mask_tail = mask.len() % 4;
        let mask_upload_size = pixels.div_ceil(4) * 4;
        let new = scoped(ctx, || {
            use crate::core::resource_tracker::tracked_create_texture;
            let settings = [
                width,
                height,
                0,
                0,
                nodata.0.map_or(INVALID, f32::to_bits),
                nodata.1.map_or(INVALID, f32::to_bits),
            ];
            let settings_buffer = initialized(
                ctx,
                "raster_recipe.settings",
                bytemuck::bytes_of(&settings),
                BufferUse::UNIFORM | BufferUse::COPY_DST,
            )?;
            let cell_buffer = buffer(
                ctx,
                "raster_recipe.cells",
                pixels * 8,
                BufferUse::STORAGE | BufferUse::COPY_SRC,
            )?;
            let texture = tracked_create_texture(
                &ctx.device,
                &wgpu::TextureDescriptor {
                    label: Some("raster_recipe.rgba"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: TextureUse::STORAGE_BINDING | TextureUse::TEXTURE_BINDING,
                    view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
                },
            )
            .map_err(gpu_error)?;
            let left = initialized(ctx, "raster_recipe.a", a, BufferUse::STORAGE)?;
            let right = initialized(ctx, "raster_recipe.b", b, BufferUse::STORAGE)?;
            let validity = buffer(
                ctx,
                "raster_recipe.mask",
                mask_upload_size,
                BufferUse::STORAGE | BufferUse::COPY_DST,
            )?;
            let full = mask.len() - mask_tail;
            if full != 0 {
                ctx.queue.write_buffer(&validity, 0, &mask[..full]);
            }
            if mask_tail != 0 {
                let mut tail = [0u8; 4];
                tail[..mask_tail].copy_from_slice(&mask[full..]);
                ctx.queue.write_buffer(&validity, full as u64, &tail);
            }
            let next = Inputs {
                _inputs: [left, right, validity],
                settings,
                settings_buffer,
                cells: cell_buffer,
                texture: Arc::new(texture),
                counts: None,
                hash: None,
            };
            submit(
                ctx,
                &self.normalize,
                &[
                    (0, &next.settings_buffer),
                    (1, &next._inputs[0]),
                    (2, &next._inputs[1]),
                    (3, &next._inputs[2]),
                    (4, &next.cells),
                ],
                None,
                next.dispatch(),
            );
            Ok(next)
        })?;
        self.parameter_upload_bytes += std::mem::size_of_val(&new.settings) as u64;
        self.inputs = Some(new);
        self.input_upload_bytes += pixels * 8 + mask_upload_size;
        self.normalization_dispatches += 1;
        Ok(())
    }

    fn classify(
        &mut self,
        bins: Vec<(f32, [u8; 4])>,
        last_color: [u8; 4],
        right: bool,
        digest: [u8; 32],
    ) -> PyResult<()> {
        let ctx = try_ctx()?;
        let input = self
            .inputs
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Raster recipe has no inputs"))?;
        if input.hash == Some(digest) {
            return Ok(());
        }
        let bin_count = bins.len() as u64 + 1;
        let max_size = ctx.device.limits().max_storage_buffer_binding_size as u64;
        if bin_count * 8 > max_size {
            return Err(PyValueError::new_err(
                "Raster bins exceed negotiated GPU resource limits",
            ));
        }
        if Arc::strong_count(&input.texture) != 1 {
            return Err(PyRuntimeError::new_err(
                "Previous raster frame still owns the resident texture",
            ));
        }
        let entries: Vec<_> = bins
            .into_iter()
            .map(|(edge, color)| [edge.to_bits(), u32::from_le_bytes(color)])
            .chain([[0, u32::from_le_bytes(last_color)]])
            .collect();
        input.hash = None;
        let counts = scoped(ctx, || {
            let bins = initialized(
                ctx,
                "raster_recipe.bins",
                bytemuck::cast_slice(&entries),
                BufferUse::STORAGE,
            )?;
            let counts = buffer(
                ctx,
                "raster_recipe.counts",
                (bin_count + 1) * 4,
                BufferUse::STORAGE | BufferUse::COPY_SRC,
            )?;
            input.settings[2] = bin_count as u32 - 1;
            input.settings[3] = u32::from(right);
            ctx.queue.write_buffer(
                &input.settings_buffer,
                0,
                bytemuck::bytes_of(&input.settings),
            );
            let view = input
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            submit(
                ctx,
                &self.classify,
                &[
                    (0, &input.settings_buffer),
                    (4, &input.cells),
                    (5, &bins),
                    (6, &counts),
                ],
                Some(&view),
                input.dispatch(),
            );
            Ok(counts)
        })?;
        input.counts = Some(counts);
        input.hash = Some(digest);
        self.classification_dispatches += 1;
        self.parameter_upload_bytes += (std::mem::size_of_val(&input.settings)
            + std::mem::size_of_val(entries.as_slice()))
            as u64;
        Ok(())
    }

    fn overlay(&self) -> PyResult<OverlayLayer> {
        let input = self.input()?;
        input.ready()?;
        let ctx = try_ctx()?;
        let hash = input.hash.expect("checked by ready");
        let lut = ColormapLUT::from_raster_texture(&ctx.device, Arc::clone(&input.texture), hash);
        Ok(OverlayLayer::from_raster_lut(lut, 1.0))
    }

    fn sample(&mut self, row: u32, col: u32) -> PyResult<(Option<f32>, u32)> {
        let input = self.input()?;
        input.ready()?;
        let [width, height, ..] = input.settings;
        if row >= height || col >= width {
            return Err(PyValueError::new_err("Sample is outside the raster grid"));
        }
        let result = readback(
            &input.cells,
            (u64::from(row) * u64::from(width) + u64::from(col)) * 8,
            8,
        )?;
        self.readback_bytes += 8;
        Ok((
            (result[1] != 0).then(|| f32::from_bits(result[0])),
            result[1],
        ))
    }

    fn counts(&mut self) -> PyResult<Vec<u32>> {
        let source = self.input()?.ready()?;
        let bytes = source.size();
        let values = readback(source, 0, bytes)?;
        self.readback_bytes += bytes;
        Ok(values)
    }

    fn export_cells<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u32>>> {
        let input = self.input()?;
        input.ready()?;
        let bytes = input.cells.size();
        let values = readback(&input.cells, 0, bytes)?;
        self.readback_bytes += bytes;
        Ok(values.into_pyarray_bound(py))
    }

    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new_bound(py);
        result.set_item(
            "backend",
            format!("{:?}", try_ctx()?.adapter.get_info().backend),
        )?;
        result.set_item("adapter", try_ctx()?.adapter.get_info().name)?;
        result.set_item("normalization_dispatches", self.normalization_dispatches)?;
        result.set_item("classification_dispatches", self.classification_dispatches)?;
        result.set_item("input_upload_bytes", self.input_upload_bytes)?;
        result.set_item("parameter_upload_bytes", self.parameter_upload_bytes)?;
        result.set_item("readback_bytes", self.readback_bytes)?;
        Ok(result)
    }
}

impl ResidentRasterRecipe {
    fn input(&self) -> PyResult<&Inputs> {
        self.inputs
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Raster recipe has no inputs"))
    }
}

fn readback(source: &TrackedBuffer, offset: u64, size: u64) -> PyResult<Vec<u32>> {
    let ctx = try_ctx()?;
    scoped(ctx, || {
        let staging = buffer(
            ctx,
            "raster_recipe.readback",
            size,
            BufferUse::COPY_DST | BufferUse::MAP_READ,
        )?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raster_recipe.readback"),
            });
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        wait_for_device_idle(&ctx.device);
        receive.try_recv().map_err(gpu_error)?.map_err(gpu_error)?;
        let data = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(data)
    })
}
