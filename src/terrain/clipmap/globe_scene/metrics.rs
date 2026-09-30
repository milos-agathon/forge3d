use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Debug, Default)]
pub(super) struct StreamingEvidence {
    pub frames: u64,
    pub bounded_poll_frames: u64,
    pub progress_frames: u64,
    pub pending_frames: u64,
    pub coarse_fallback_frames: u64,
    pub max_uploads_per_frame: u64,
    previous: Option<StreamingProgressSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StreamingProgressSnapshot {
    pub bounded_step: u64,
    pub tiles_requested: u64,
    pub loader_completed: u64,
    pub tiles_uploaded: u64,
    pub pending_ring_tiles: u64,
    pub loader_pending: u64,
    pub resident_fine_tiles: u64,
    pub resident_ancestor_tiles: u64,
    pub loaded_ring_tiles: u64,
    pub page_table_updates: u64,
    pub coarse_prefilled: u64,
    pub ancestor_fallbacks: u64,
    pub required_leaf_tiles: u64,
}

impl StreamingProgressSnapshot {
    #[cfg(test)]
    fn with_bounded_step(mut self, bounded_step: u64) -> Self {
        self.bounded_step = bounded_step;
        self
    }

    fn progress_state(self) -> [u64; 12] {
        [
            self.tiles_requested,
            self.loader_completed,
            self.tiles_uploaded,
            self.pending_ring_tiles,
            self.loader_pending,
            self.resident_fine_tiles,
            self.resident_ancestor_tiles,
            self.loaded_ring_tiles,
            self.page_table_updates,
            self.coarse_prefilled,
            self.ancestor_fallbacks,
            self.required_leaf_tiles,
        ]
    }
}

impl StreamingEvidence {
    pub fn seed(&mut self, snapshot: StreamingProgressSnapshot) {
        self.previous = Some(snapshot);
    }

    pub fn observe(&mut self, snapshot: StreamingProgressSnapshot) -> Result<(), String> {
        let previous = self
            .previous
            .ok_or_else(|| "ORBIS streaming evidence has no pre-frame baseline".to_string())?;
        if snapshot.bounded_step != previous.bounded_step.saturating_add(1) {
            return Err(format!(
                "ORBIS streaming bounded-step sequence did not advance exactly once: previous={}, current={}",
                previous.bounded_step, snapshot.bounded_step
            ));
        }
        self.frames += 1;
        self.bounded_poll_frames += 1;
        self.progress_frames +=
            u64::from(snapshot.progress_state() != previous.progress_state());
        self.max_uploads_per_frame = self
            .max_uploads_per_frame
            .max(snapshot.tiles_uploaded.saturating_sub(previous.tiles_uploaded));
        self.previous = Some(snapshot);
        Ok(())
    }
}

/// Physical measurements for a completed ORBIS descent. There is no public
/// constructor and no optional/default value: a scene publishes this object
/// only after the submitted GPU projection and depth readbacks complete.
#[pyclass(module = "forge3d._forge3d", name = "GlobeMetrics")]
#[derive(Clone)]
pub struct GlobeMetrics {
    max_vertex_jitter_px: f64,
    naive_max_vertex_jitter_px: f64,
    jitter_sample_count: u64,
    peak_gpu_visible_bytes: u64,
    lod_crack_pixels: u64,
    crack_boundary_samples: u64,
    crack_depth_variance: f64,
    rendered_frames: u64,
    bounded_poll_frames: u64,
    streaming_progress_frames: u64,
    pending_streaming_frames: u64,
    coarse_fallback_frames: u64,
    max_stream_uploads_per_frame: u64,
    adapter_name: String,
    adapter_backend: String,
    adapter_vendor: u32,
    adapter_device_type: String,
    software_fallback: bool,
}

impl GlobeMetrics {
    pub(super) fn measured(
        physical: crate::terrain::renderer::orbis_capture::OrbisPhysicalCapture,
        peak_gpu_visible_bytes: u64,
        streaming: &StreamingEvidence,
        adapter: wgpu::AdapterInfo,
        software_fallback: bool,
    ) -> Result<Self, String> {
        if !physical.max_vertex_jitter_px.is_finite()
            || !physical.naive_max_vertex_jitter_px.is_finite()
            || physical.jitter_sample_count < 32
        {
            return Err("ORBIS jitter evidence is incomplete or non-finite".into());
        }
        if peak_gpu_visible_bytes == 0 {
            return Err("ORBIS allocation capture reported no GPU-visible bytes".into());
        }
        if physical.max_vertex_jitter_px >= 0.5 {
            return Err(format!(
                "ORBIS camera-relative jitter {:.9} px is not below 0.5 px",
                physical.max_vertex_jitter_px
            ));
        }
        if physical.naive_max_vertex_jitter_px
            <= physical.max_vertex_jitter_px * 10.0
        {
            return Err(format!(
                "ORBIS absolute-ECEF-f32 control was not materially worse: production={:.9} px, naive={:.9} px",
                physical.max_vertex_jitter_px, physical.naive_max_vertex_jitter_px
            ));
        }
        if peak_gpu_visible_bytes >= 512 * 1024 * 1024 {
            return Err(format!(
                "ORBIS measured GPU-visible peak {peak_gpu_visible_bytes} bytes is not below 512 MiB"
            ));
        }
        if physical.crack_boundary_samples < 64
            || !physical.crack_depth_variance.is_finite()
            || physical.crack_depth_variance <= 0.0
        {
            return Err("ORBIS depth/coverage boundary evidence is incomplete".into());
        }
        if physical.lod_crack_pixels != 0 {
            return Err(format!(
                "ORBIS measured {} uncovered projected ring-boundary pixels",
                physical.lod_crack_pixels
            ));
        }
        if streaming.frames == 0 || streaming.bounded_poll_frames != streaming.frames {
            return Err("ORBIS streaming did not execute one bounded poll on every waypoint frame".into());
        }
        if streaming.progress_frames > streaming.frames {
            return Err("ORBIS streaming progress count exceeds rendered frames".into());
        }
        // A deliberately short custom path can complete before an ancestor is
        // needed. The counts remain measured (including an honest zero); the
        // dedicated default-descent acceptance test requires both pending and
        // coarse-fallback frames.
        if software_fallback || adapter.device_type == wgpu::DeviceType::Cpu {
            return Err("ORBIS metrics require a physical non-software GPU adapter".into());
        }
        Ok(Self {
            max_vertex_jitter_px: physical.max_vertex_jitter_px,
            naive_max_vertex_jitter_px: physical.naive_max_vertex_jitter_px,
            jitter_sample_count: physical.jitter_sample_count,
            peak_gpu_visible_bytes,
            lod_crack_pixels: physical.lod_crack_pixels,
            crack_boundary_samples: physical.crack_boundary_samples,
            crack_depth_variance: physical.crack_depth_variance,
            rendered_frames: streaming.frames,
            bounded_poll_frames: streaming.bounded_poll_frames,
            streaming_progress_frames: streaming.progress_frames,
            pending_streaming_frames: streaming.pending_frames,
            coarse_fallback_frames: streaming.coarse_fallback_frames,
            max_stream_uploads_per_frame: streaming.max_uploads_per_frame,
            adapter_name: adapter.name,
            adapter_backend: format!("{:?}", adapter.backend),
            adapter_vendor: adapter.vendor,
            adapter_device_type: format!("{:?}", adapter.device_type),
            software_fallback,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_poll_is_not_fabricated_streaming_progress() {
        let baseline = StreamingProgressSnapshot {
            bounded_step: 10,
            tiles_requested: 4,
            loader_completed: 2,
            tiles_uploaded: 2,
            pending_ring_tiles: 3,
            loader_pending: 2,
            resident_fine_tiles: 2,
            resident_ancestor_tiles: 1,
            loaded_ring_tiles: 1,
            page_table_updates: 2,
            coarse_prefilled: 1,
            ancestor_fallbacks: 1,
            required_leaf_tiles: 8,
        };
        let mut evidence = StreamingEvidence::default();
        evidence.seed(baseline);
        evidence.observe(baseline.with_bounded_step(11)).unwrap();
        assert_eq!(evidence.bounded_poll_frames, 1);
        assert_eq!(evidence.progress_frames, 0);

        let progressed = StreamingProgressSnapshot {
            bounded_step: 12,
            loader_completed: 3,
            ..baseline
        };
        evidence.observe(progressed).unwrap();
        assert_eq!(evidence.bounded_poll_frames, 2);
        assert_eq!(evidence.progress_frames, 1);
    }

    #[test]
    fn request_terminal_upload_residency_and_demand_deltas_are_real_progress() {
        let baseline = StreamingProgressSnapshot {
            bounded_step: 20,
            tiles_requested: 10,
            loader_completed: 8,
            tiles_uploaded: 7,
            pending_ring_tiles: 4,
            loader_pending: 2,
            resident_fine_tiles: 6,
            resident_ancestor_tiles: 3,
            loaded_ring_tiles: 2,
            page_table_updates: 5,
            coarse_prefilled: 1,
            ancestor_fallbacks: 2,
            required_leaf_tiles: 12,
        };
        let mut deltas = Vec::new();
        let mut request = baseline;
        request.tiles_requested += 1;
        deltas.push(request);
        let mut terminal = baseline;
        terminal.loader_completed += 1;
        deltas.push(terminal);
        let mut upload = baseline;
        upload.tiles_uploaded += 1;
        deltas.push(upload);
        let mut residency = baseline;
        residency.resident_fine_tiles += 1;
        deltas.push(residency);
        let mut page_table = baseline;
        page_table.page_table_updates += 1;
        deltas.push(page_table);
        let mut pending = baseline;
        pending.pending_ring_tiles -= 1;
        deltas.push(pending);

        for mut changed in deltas {
            changed.bounded_step += 1;
            let mut evidence = StreamingEvidence::default();
            evidence.seed(baseline);
            evidence.observe(changed).unwrap();
            assert_eq!(evidence.bounded_poll_frames, 1);
            assert_eq!(evidence.progress_frames, 1, "delta was not counted: {changed:?}");
        }
    }
}

#[pymethods]
impl GlobeMetrics {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "GlobeMetrics objects are produced by a completed physical GlobeScene probe",
        ))
    }

    #[getter]
    fn max_vertex_jitter_px(&self) -> f64 { self.max_vertex_jitter_px }
    #[getter]
    fn naive_max_vertex_jitter_px(&self) -> f64 { self.naive_max_vertex_jitter_px }
    #[getter]
    fn jitter_sample_count(&self) -> u64 { self.jitter_sample_count }
    #[getter]
    fn peak_gpu_visible_bytes(&self) -> u64 { self.peak_gpu_visible_bytes }
    #[getter]
    fn lod_crack_pixels(&self) -> u64 { self.lod_crack_pixels }
    #[getter]
    fn crack_boundary_samples(&self) -> u64 { self.crack_boundary_samples }
    #[getter]
    fn crack_depth_variance(&self) -> f64 { self.crack_depth_variance }
    #[getter]
    fn rendered_frames(&self) -> u64 { self.rendered_frames }
    #[getter]
    fn bounded_poll_frames(&self) -> u64 { self.bounded_poll_frames }
    #[getter]
    fn streaming_progress_frames(&self) -> u64 { self.streaming_progress_frames }
    #[getter]
    fn pending_streaming_frames(&self) -> u64 { self.pending_streaming_frames }
    #[getter]
    fn coarse_fallback_frames(&self) -> u64 { self.coarse_fallback_frames }
    #[getter]
    fn max_stream_uploads_per_frame(&self) -> u64 { self.max_stream_uploads_per_frame }
    #[getter]
    fn adapter_name(&self) -> &str { &self.adapter_name }
    #[getter]
    fn adapter_backend(&self) -> &str { &self.adapter_backend }
    #[getter]
    fn adapter_vendor(&self) -> u32 { self.adapter_vendor }
    #[getter]
    fn adapter_device_type(&self) -> &str { &self.adapter_device_type }
    #[getter]
    fn software_fallback(&self) -> bool { self.software_fallback }

    fn __getitem__(&self, key: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| match key {
            "max_vertex_jitter_px" => Ok(self.max_vertex_jitter_px.into_py(py)),
            "naive_max_vertex_jitter_px" => Ok(self.naive_max_vertex_jitter_px.into_py(py)),
            "jitter_sample_count" => Ok(self.jitter_sample_count.into_py(py)),
            "peak_gpu_visible_bytes" => Ok(self.peak_gpu_visible_bytes.into_py(py)),
            "lod_crack_pixels" => Ok(self.lod_crack_pixels.into_py(py)),
            "crack_boundary_samples" => Ok(self.crack_boundary_samples.into_py(py)),
            "crack_depth_variance" => Ok(self.crack_depth_variance.into_py(py)),
            "rendered_frames" => Ok(self.rendered_frames.into_py(py)),
            "bounded_poll_frames" => Ok(self.bounded_poll_frames.into_py(py)),
            "streaming_progress_frames" => Ok(self.streaming_progress_frames.into_py(py)),
            "pending_streaming_frames" => Ok(self.pending_streaming_frames.into_py(py)),
            "coarse_fallback_frames" => Ok(self.coarse_fallback_frames.into_py(py)),
            "max_stream_uploads_per_frame" => Ok(self.max_stream_uploads_per_frame.into_py(py)),
            "adapter_name" => Ok(self.adapter_name.clone().into_py(py)),
            "adapter_backend" => Ok(self.adapter_backend.clone().into_py(py)),
            "adapter_vendor" => Ok(self.adapter_vendor.into_py(py)),
            "adapter_device_type" => Ok(self.adapter_device_type.clone().into_py(py)),
            "software_fallback" => Ok(self.software_fallback.into_py(py)),
            _ => Err(PyKeyError::new_err(key.to_string())),
        })
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        for key in [
            "max_vertex_jitter_px", "naive_max_vertex_jitter_px", "jitter_sample_count",
            "peak_gpu_visible_bytes", "lod_crack_pixels", "crack_boundary_samples",
            "crack_depth_variance", "rendered_frames", "bounded_poll_frames",
            "streaming_progress_frames",
            "pending_streaming_frames", "coarse_fallback_frames",
            "max_stream_uploads_per_frame", "adapter_name", "adapter_backend",
            "adapter_vendor", "adapter_device_type", "software_fallback",
        ] {
            out.set_item(key, self.__getitem__(key)?)?;
        }
        Ok(out.into())
    }
}
