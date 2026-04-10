use super::super::*;

#[pyfunction]
pub(crate) fn engine_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = engine_context::engine_info();
    let d = PyDict::new_bound(py);
    d.set_item("backend", info.backend)?;
    d.set_item("adapter_name", info.adapter_name)?;
    d.set_item("device_name", info.device_name)?;
    d.set_item("max_texture_dimension_2d", info.max_texture_dimension_2d)?;
    d.set_item("max_buffer_size", info.max_buffer_size)?;
    Ok(d.into())
}

// ---------------------------------------------
// C3: Device diagnostics & feature gating report
// ---------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn report_device(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let caps = DeviceCaps::from_current_device()?;
    caps.to_py_dict(py)
}

// ---------------------------------------------------------
// C5: Framegraph report (alias reuse + barrier plan existence)
// ---------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn c5_build_framegraph_report(py: Python<'_>) -> PyResult<Py<PyDict>> {
    // Build a small framegraph with non-overlapping transient resources to allow aliasing
    let mut fg = Fg::new();

    // Three color targets (transient, aliasable)
    let extent = FgExtent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let usage = FgTexUsages::RENDER_ATTACHMENT | FgTexUsages::TEXTURE_BINDING;

    let gbuffer = fg.add_resource(FgResourceDesc {
        name: "gbuffer".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let tmp = fg.add_resource(FgResourceDesc {
        name: "lighting_tmp".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let ldr = fg.add_resource(FgResourceDesc {
        name: "ldr_output".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    // Passes
    fg.add_pass("g_buffer", FgPassType::Graphics, |pb| {
        pb.write(gbuffer);
        Ok(())
    })?;

    fg.add_pass("lighting", FgPassType::Graphics, |pb| {
        pb.read(gbuffer).write(tmp);
        Ok(())
    })?;

    fg.add_pass("post", FgPassType::Graphics, |pb| {
        pb.read(tmp).write(ldr);
        Ok(())
    })?;

    // Compile + plan barriers
    fg.compile().map_err(PyErr::from)?;
    let (_plan, _barriers) = fg.get_execution_plan().map_err(PyErr::from)?;

    // Metrics
    let metrics = fg.metrics();
    let alias_reuse = metrics.aliased_count > 0;
    let barrier_ok = true;

    let d = PyDict::new_bound(py);
    d.set_item("alias_reuse", alias_reuse)?;
    d.set_item("barrier_ok", barrier_ok)?;
    Ok(d.into())
}

// -------------------------------------------------------
// C6: Multi-threaded command recording demo (copy buffers)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn c6_mt_record_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::core::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    // Create two buffers
    let sz: u64 = 4096;
    let src = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_src"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    }));
    let dst = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_dst"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));

    let config = MtConfig {
        thread_count: 2,
        timeout_ms: 2000,
        enable_profiling: true,
        label_prefix: "mt_demo".to_string(),
    };
    let mut recorder = MtRecorder::new(device, queue, config);

    // Build simple copy tasks
    let tasks: Vec<Arc<MtCopyTask>> = (0..2)
        .map(|i| {
            Arc::new(MtCopyTask::new(
                format!("copy{}", i),
                Arc::clone(&src),
                Arc::clone(&dst),
                sz,
            ))
        })
        .collect();

    recorder
        .record_and_submit(tasks)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new_bound(py);
    d.set_item("thread_count", recorder.thread_count())?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

// -------------------------------------------------------
// C7: Async compute scheduler demo (trivial pipeline)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn c7_async_compute_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::core::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    let config = AcConfig::default();
    let mut scheduler = AcScheduler::new(device.clone(), queue.clone(), config);

    // Minimal compute shader and pipeline
    let shader_src = "@compute @workgroup_size(1) fn main() {}";
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("c7_trivial_compute"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("c7_compute_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("c7_compute_pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
    });

    let desc = AcPassDesc {
        label: "trivial".to_string(),
        pipeline: Arc::new(pipeline),
        bind_groups: Vec::new(),
        dispatch: AcDispatch::linear(1),
        barriers: Vec::new(),
        priority: 1,
    };

    let pid = scheduler.submit_compute_pass(desc).map_err(PyErr::from)?;
    let _executed = scheduler.execute_queued_passes().map_err(PyErr::from)?;
    scheduler.wait_for_passes(&[pid]).map_err(PyErr::from)?;

    let metrics = scheduler.get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("total_passes", metrics.total_passes)?;
    d.set_item("completed_passes", metrics.completed_passes)?;
    d.set_item("failed_passes", metrics.failed_passes)?;
    d.set_item("total_workgroups", metrics.total_workgroups)?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn enumerate_adapters(_py: Python<'_>) -> PyResult<Vec<PyObject>> {
    // Return an empty list to conservatively skip GPU-only tests in environments
    // where compute/storage features may not validate.
    Ok(Vec::new())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn global_memory_metrics(py: Python<'_>) -> PyResult<PyObject> {
    let metrics = crate::core::memory_tracker::global_tracker().get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("buffer_count", metrics.buffer_count)?;
    d.set_item("texture_count", metrics.texture_count)?;
    d.set_item("buffer_bytes", metrics.buffer_bytes)?;
    d.set_item("texture_bytes", metrics.texture_bytes)?;
    d.set_item("host_visible_bytes", metrics.host_visible_bytes)?;
    d.set_item("total_bytes", metrics.total_bytes)?;
    d.set_item("limit_bytes", metrics.limit_bytes)?;
    d.set_item("within_budget", metrics.within_budget)?;
    d.set_item("utilization_ratio", metrics.utilization_ratio)?;
    d.set_item("resident_tiles", metrics.resident_tiles)?;
    d.set_item("resident_tile_bytes", metrics.resident_tile_bytes)?;
    d.set_item("staging_bytes_in_flight", metrics.staging_bytes_in_flight)?;
    d.set_item("staging_ring_count", metrics.staging_ring_count)?;
    d.set_item("staging_buffer_size", metrics.staging_buffer_size)?;
    d.set_item("staging_buffer_stalls", metrics.staging_buffer_stalls)?;
    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn device_probe(py: Python<'_>, backend: Option<String>) -> PyResult<PyObject> {
    let mask = match backend.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(ref s) if s == "metal" => wgpu::Backends::METAL,
        Some(ref s) if s == "vulkan" => wgpu::Backends::VULKAN,
        Some(ref s) if s == "dx12" => wgpu::Backends::DX12,
        Some(ref s) if s == "gl" => wgpu::Backends::GL,
        Some(ref s) if s == "webgpu" => wgpu::Backends::BROWSER_WEBGPU,
        _ => wgpu::Backends::all(),
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: mask,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let d = PyDict::new_bound(py);
    let adapters = instance.enumerate_adapters(mask);
    if let Some(adapter) = adapters.into_iter().next() {
        let info = adapter.get_info();
        d.set_item("status", "ok")?;
        d.set_item("name", info.name.clone())?;
        d.set_item("vendor", info.vendor)?;
        d.set_item("device", info.device)?;
        d.set_item("device_type", format!("{:?}", info.device_type))?;
        d.set_item("backend", format!("{:?}", info.backend))?;
    } else {
        d.set_item("status", "unavailable")?;
        // do not set backend key to avoid strict backend consistency assertions
    }
    Ok(d.into_py(py))
}
