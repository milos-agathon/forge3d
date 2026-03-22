//! TV12: Offline terrain render quality — batch accumulation methods.
//!
//! Implements the 7 offline #[pymethods] on TerrainRenderer:
//! begin_offline_accumulation, accumulate_batch, read_accumulation_metrics,
//! resolve_offline_hdr, upload_hdr_frame, tonemap_offline_hdr, end_offline_accumulation.

use super::*;
use crate::terrain::accumulation::OfflineAccumulationState;
use numpy::PyArrayMethods;

/// Cached state for an active offline session.
/// Stores owned PyObject references to Python inputs so accumulate_batch
/// does not need them re-passed. Uses PyObject (not Py<T>) to avoid
/// Frozen trait constraints on MaterialSet/IBL.
struct OfflineSessionInputs {
    material_set: PyObject,
    env_maps: PyObject,
    heightmap: PyObject,
    params: PyObject,
    /// Original color format to restore after the session ends
    original_color_format: wgpu::TextureFormat,
}

/// Result of accumulate_batch, returned to Python.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "OfflineBatchResult")]
pub struct OfflineBatchResult {
    #[pyo3(get)]
    pub total_samples: u32,
    #[pyo3(get)]
    pub batch_time_ms: f64,
}

/// Result of read_accumulation_metrics, returned to Python.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "OfflineMetrics")]
pub struct OfflineMetrics {
    #[pyo3(get)]
    pub total_samples: u32,
    #[pyo3(get)]
    pub mean_delta: f32,
    #[pyo3(get)]
    pub p95_delta: f32,
    #[pyo3(get)]
    pub max_tile_delta: f32,
    #[pyo3(get)]
    pub converged_tile_ratio: f32,
}

use std::cell::RefCell;
thread_local! {
    static OFFLINE_INPUTS: RefCell<Option<OfflineSessionInputs>> = RefCell::new(None);
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl TerrainRenderer {
    /// TV12: Begin an offline accumulation session.
    #[pyo3(signature = (params, heightmap, material_set, env_maps))]
    pub fn begin_offline_accumulation<'py>(
        &mut self,
        py: Python<'py>,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: numpy::PyReadonlyArray2<'py, f32>,
        material_set: &Bound<'py, PyAny>,
        env_maps: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        // Check no session is already active
        {
            let state_guard = self.scene.offline_state.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
            })?;
            if state_guard.is_some() {
                return Err(PyRuntimeError::new_err(
                    "An offline accumulation session is already active. \
                     Call end_offline_accumulation() or tonemap_offline_hdr() first.",
                ));
            }
        }

        let (width, height) = params.size_px;
        let aa_samples = params.aa_samples.max(1);
        let aa_seed = params.aa_seed;

        // Create accumulation state
        let accum_state = OfflineAccumulationState::new(
            &self.scene.device,
            width as u32,
            height as u32,
            aa_samples.max(64),
            aa_seed.map(|s| s as u64),
        );

        {
            let mut state_guard = self.scene.offline_state.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
            })?;
            *state_guard = Some(accum_state);
        }

        // Cache Python inputs and swap color format to Rgba16Float
        let original_format = self.scene.color_format;
        self.scene.color_format = wgpu::TextureFormat::Rgba16Float;

        // Force pipeline recompilation for Rgba16Float
        {
            let mut pipeline_cache = self.scene.pipeline.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("pipeline mutex poisoned: {e}"))
            })?;
            pipeline_cache.sample_count = 0;
        }

        // Copy heightmap as a new numpy array (owned)
        let heightmap_copied =
            numpy::PyArray2::from_array_bound(py, &heightmap.as_array()).into_any();

        // Clone params and store as PyObject
        let params_cloned = params.clone();
        let params_obj: PyObject = Py::new(py, params_cloned)?.into_any();

        OFFLINE_INPUTS.with(|cell| {
            *cell.borrow_mut() = Some(OfflineSessionInputs {
                material_set: material_set.clone().unbind(),
                env_maps: env_maps.clone().unbind(),
                heightmap: heightmap_copied.unbind(),
                params: params_obj,
                original_color_format: original_format,
            });
        });

        Ok(())
    }

    /// TV12: Accumulate a batch of jittered samples.
    #[pyo3(signature = (sample_count))]
    pub fn accumulate_batch(
        &mut self,
        py: Python,
        sample_count: u32,
    ) -> PyResult<Py<OfflineBatchResult>> {
        let start_time = std::time::Instant::now();

        let inputs_available = OFFLINE_INPUTS.with(|cell| cell.borrow().is_some());
        if !inputs_available {
            return Err(PyRuntimeError::new_err(
                "No offline accumulation session is active.",
            ));
        }

        for _sample_idx in 0..sample_count {
            let sample_index = {
                let state_guard = self.scene.offline_state.lock().map_err(|e| {
                    PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
                })?;
                let state = state_guard.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("Offline session ended unexpectedly")
                })?;
                state.total_samples
            };

            // Render one sample using the existing render_internal path
            let frame = OFFLINE_INPUTS.with(|cell| -> PyResult<crate::Frame> {
                let borrow = cell.borrow();
                let inputs = borrow.as_ref().unwrap();

                // Downcast stored PyObjects to their concrete types
                let params_bound = inputs.params.bind(py);
                let params_cell = params_bound
                    .downcast::<crate::terrain::render_params::TerrainRenderParams>()?;
                let mut params_ref = params_cell.borrow_mut();
                params_ref.offline_hdr_output = true;

                let heightmap_bound = inputs.heightmap.bind(py);
                let heightmap_cell = heightmap_bound.downcast::<numpy::PyArray2<f32>>()?;
                let heightmap_ro = heightmap_cell.readonly();

                let material_bound = inputs.material_set.bind(py);
                let material_cell = material_bound
                    .downcast::<crate::render::material_set::MaterialSet>()?;
                let material_ref = material_cell.borrow();

                let env_bound = inputs.env_maps.bind(py);
                let env_cell = env_bound
                    .downcast::<crate::lighting::ibl_wrapper::IBL>()?;
                let env_ref = env_cell.borrow();

                self.scene
                    .render_internal(&material_ref, &env_ref, &params_ref, heightmap_ro, None)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Offline render sample {} failed: {:#}",
                            sample_index, e
                        ))
                    })
            })?;

            // Accumulate this sample into the buffer
            self.accumulate_sample_into_buffer(sample_index, &frame)?;

            // Update total samples
            {
                let mut state_guard = self.scene.offline_state.lock().map_err(|e| {
                    PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
                })?;
                if let Some(state) = state_guard.as_mut() {
                    state.total_samples += 1;
                }
            }
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let total_samples = {
            let state_guard = self.scene.offline_state.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
            })?;
            state_guard.as_ref().map(|s| s.total_samples).unwrap_or(0)
        };

        Ok(Py::new(
            py,
            OfflineBatchResult {
                total_samples,
                batch_time_ms: elapsed_ms,
            },
        )?)
    }

    /// TV12: Read accumulation convergence metrics (temporal delta).
    #[pyo3(signature = (target_variance))]
    pub fn read_accumulation_metrics(
        &self,
        py: Python,
        target_variance: f32,
    ) -> PyResult<Py<OfflineMetrics>> {
        let mut state_guard = self.scene.offline_state.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
        })?;
        let state = state_guard.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("No offline accumulation session is active.")
        })?;

        let tile_size = 16u32;
        let tiles_x = (state.luminance_width + tile_size - 1) / tile_size;
        let tiles_y = (state.luminance_height + tile_size - 1) / tile_size;
        let num_tiles = (tiles_x * tiles_y) as usize;

        let current_tile_means = self.compute_tile_means(state)?;

        let (deltas, converged_count) = if state.prev_tile_means.is_empty() {
            (vec![1.0f32; num_tiles], 0usize)
        } else {
            let mut deltas = Vec::with_capacity(num_tiles);
            let mut converged = 0usize;
            for (i, &current) in current_tile_means.iter().enumerate() {
                let prev = state.prev_tile_means.get(i).copied().unwrap_or(0.0);
                let delta = (current - prev).abs() / current.max(1e-6);
                if delta < target_variance {
                    converged += 1;
                }
                deltas.push(delta);
            }
            (deltas, converged)
        };

        state.prev_tile_means = current_tile_means;

        let mut sorted = deltas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_delta = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().sum::<f32>() / sorted.len() as f32
        };
        let p95_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len().saturating_sub(1));
        let p95_delta = sorted.get(p95_idx).copied().unwrap_or(0.0);
        let max_tile_delta = sorted.last().copied().unwrap_or(0.0);
        let converged_ratio = if num_tiles > 0 {
            converged_count as f32 / num_tiles as f32
        } else {
            1.0
        };

        Ok(Py::new(
            py,
            OfflineMetrics {
                total_samples: state.total_samples,
                mean_delta,
                p95_delta,
                max_tile_delta,
                converged_tile_ratio: converged_ratio,
            },
        )?)
    }

    /// TV12: Resolve accumulated HDR buffer into HdrFrame + AovFrame.
    #[pyo3(signature = ())]
    pub fn resolve_offline_hdr(
        &self,
        py: Python,
    ) -> PyResult<(Py<crate::py_types::hdr_frame::HdrFrame>, Py<crate::AovFrame>)> {
        let state_guard = self.scene.offline_state.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
        })?;
        let state = state_guard.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("No offline accumulation session is active.")
        })?;

        if state.total_samples == 0 {
            return Err(PyRuntimeError::new_err("No samples accumulated."));
        }

        let total = state.total_samples;
        let read_idx = state.read_buffer_idx();

        // Read the accumulated sum (Rgba32Float)
        let accum_data = crate::core::hdr::read_hdr_texture(
            &self.scene.device,
            &self.scene.queue,
            &state.beauty_accum[read_idx],
            state.width,
            state.height,
            wgpu::TextureFormat::Rgba32Float,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Accum readback failed: {e}")))?;

        // Divide by N on CPU, convert to f16 bytes for upload
        let n = total as f32;
        let pixel_count = (state.width * state.height) as usize;
        let mut f16_bytes: Vec<u8> = Vec::with_capacity(pixel_count * 8); // 4 channels * 2 bytes
        for pixel in accum_data.chunks_exact(4) {
            for &channel in pixel {
                let val = half::f16::from_f32(channel / n);
                f16_bytes.extend_from_slice(&val.to_le_bytes());
            }
        }

        // Create resolved Rgba16Float texture
        let beauty_resolved = self.scene.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tv12.beauty_resolved"),
            size: wgpu::Extent3d {
                width: state.width,
                height: state.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.scene.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &beauty_resolved,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &f16_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(state.width * 8),
                rows_per_image: Some(state.height),
            },
            wgpu::Extent3d {
                width: state.width,
                height: state.height,
                depth_or_array_layers: 1,
            },
        );

        let hdr_frame = crate::py_types::hdr_frame::HdrFrame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            beauty_resolved,
            state.width,
            state.height,
        );

        let aov_frame = crate::AovFrame::new_empty(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            state.width,
            state.height,
        );

        Ok((Py::new(py, hdr_frame)?, Py::new(py, aov_frame)?))
    }

    /// TV12: Upload denoised HDR numpy data back to GPU as HdrFrame.
    #[pyo3(signature = (data, size))]
    pub fn upload_hdr_frame(
        &self,
        py: Python,
        data: numpy::PyReadonlyArray3<f32>,
        size: (u32, u32),
    ) -> PyResult<Py<crate::py_types::hdr_frame::HdrFrame>> {
        let (width, height) = size;
        let arr = data.as_array();
        let shape = arr.shape();

        if shape.len() != 3 || shape[2] < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data must be (H, W, 3) or (H, W, 4) float32",
            ));
        }

        let channels = shape[2];
        let mut f16_bytes: Vec<u8> = Vec::with_capacity((width * height * 4) as usize * 2);
        for y in 0..height as usize {
            for x in 0..width as usize {
                for c in 0..3 {
                    let val = half::f16::from_f32(arr[[y, x, c]]);
                    f16_bytes.extend_from_slice(&val.to_le_bytes());
                }
                let alpha = if channels > 3 {
                    half::f16::from_f32(arr[[y, x, 3]])
                } else {
                    half::f16::from_f32(1.0)
                };
                f16_bytes.extend_from_slice(&alpha.to_le_bytes());
            }
        }

        let texture = self.scene.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tv12.uploaded_hdr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.scene.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &f16_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let hdr_frame = crate::py_types::hdr_frame::HdrFrame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            texture,
            width,
            height,
        );

        Ok(Py::new(py, hdr_frame)?)
    }

    /// TV12: Tonemap an HdrFrame using the terrain filmic curve.
    #[pyo3(signature = (hdr_frame))]
    pub fn tonemap_offline_hdr(
        &mut self,
        py: Python,
        hdr_frame: &crate::py_types::hdr_frame::HdrFrame,
    ) -> PyResult<Py<crate::Frame>> {
        let (width, height) = (hdr_frame.width, hdr_frame.height);

        // Read HDR data
        let hdr_data = crate::core::hdr::read_hdr_texture(
            &hdr_frame.device,
            &hdr_frame.queue,
            &hdr_frame.texture,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("HDR readback failed: {e}")))?;

        // Apply terrain filmic tonemap + sRGB on CPU (matches terrain_pbr_pom.wgsl)
        let mut rgba8 = Vec::with_capacity((width * height * 4) as usize);
        for pixel in hdr_data.chunks_exact(4) {
            let r = tonemap_filmic_terrain(pixel[0]);
            let g = tonemap_filmic_terrain(pixel[1]);
            let b = tonemap_filmic_terrain(pixel[2]);
            rgba8.push((linear_to_srgb(r).clamp(0.0, 1.0) * 255.0) as u8);
            rgba8.push((linear_to_srgb(g).clamp(0.0, 1.0) * 255.0) as u8);
            rgba8.push((linear_to_srgb(b).clamp(0.0, 1.0) * 255.0) as u8);
            rgba8.push(255u8);
        }

        let output_texture = self.scene.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tv12.tonemapped_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.scene.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba8,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let frame = crate::Frame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            output_texture,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // Clean up offline session
        self.end_offline_accumulation_impl()?;

        Ok(Py::new(py, frame)?)
    }

    /// TV12: End the offline accumulation session.
    /// Idempotent — safe to call when no session is active.
    #[pyo3(signature = ())]
    pub fn end_offline_accumulation(&mut self) -> PyResult<()> {
        self.end_offline_accumulation_impl()
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

impl TerrainRenderer {
    fn end_offline_accumulation_impl(&mut self) -> PyResult<()> {
        OFFLINE_INPUTS.with(|cell| {
            if let Some(inputs) = cell.borrow_mut().take() {
                self.scene.color_format = inputs.original_color_format;
                if let Ok(mut pipeline_cache) = self.scene.pipeline.lock() {
                    pipeline_cache.sample_count = 0;
                }
            }
        });

        if let Ok(mut state_guard) = self.scene.offline_state.lock() {
            *state_guard = None;
        }

        Ok(())
    }

    fn accumulate_sample_into_buffer(
        &self,
        sample_index: u32,
        frame: &crate::Frame,
    ) -> PyResult<()> {
        let mut state_guard = self.scene.offline_state.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("offline state mutex poisoned: {e}"))
        })?;
        let state = state_guard.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Offline session ended unexpectedly")
        })?;

        // Read the rendered frame's HDR data
        let frame_data = frame.read_rgba_f32().map_err(|e| {
            PyRuntimeError::new_err(format!("Frame readback failed: {e}"))
        })?;

        let write_idx = state.write_buffer_idx();

        if sample_index == 0 {
            // First sample: store directly as f32 RGBA → Rgba32Float
            self.scene.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &state.beauty_accum[write_idx],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&frame_data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(state.width * 16),
                    rows_per_image: Some(state.height),
                },
                wgpu::Extent3d {
                    width: state.width,
                    height: state.height,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            let read_idx = state.read_buffer_idx();
            let prev_data = crate::core::hdr::read_hdr_texture(
                &self.scene.device,
                &self.scene.queue,
                &state.beauty_accum[read_idx],
                state.width,
                state.height,
                wgpu::TextureFormat::Rgba32Float,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Accum readback failed: {e}")))?;

            let summed: Vec<f32> = prev_data
                .iter()
                .zip(frame_data.iter())
                .map(|(a, b)| a + b)
                .collect();

            self.scene.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &state.beauty_accum[write_idx],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&summed),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(state.width * 16),
                    rows_per_image: Some(state.height),
                },
                wgpu::Extent3d {
                    width: state.width,
                    height: state.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        state.swap_buffers();
        Ok(())
    }

    fn compute_tile_means(
        &self,
        state: &OfflineAccumulationState,
    ) -> PyResult<Vec<f32>> {
        let read_idx = state.read_buffer_idx();
        let total = state.total_samples;

        let accum_data = crate::core::hdr::read_hdr_texture(
            &self.scene.device,
            &self.scene.queue,
            &state.beauty_accum[read_idx],
            state.width,
            state.height,
            wgpu::TextureFormat::Rgba32Float,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Metrics readback failed: {e}")))?;

        let n = total.max(1) as f32;
        let tile_size = 16u32;
        let tiles_x = (state.width + tile_size - 1) / tile_size;
        let tiles_y = (state.height + tile_size - 1) / tile_size;
        let num_tiles = (tiles_x * tiles_y) as usize;

        let mut tile_sums = vec![0.0f32; num_tiles];
        let mut tile_counts = vec![0u32; num_tiles];

        for y in 0..state.height {
            for x in 0..state.width {
                let pixel_idx = ((y * state.width + x) * 4) as usize;
                let r = accum_data[pixel_idx] / n;
                let g = accum_data[pixel_idx + 1] / n;
                let b = accum_data[pixel_idx + 2] / n;
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                let tx = x / tile_size;
                let ty = y / tile_size;
                let tile_idx = (ty * tiles_x + tx) as usize;
                tile_sums[tile_idx] += lum;
                tile_counts[tile_idx] += 1;
            }
        }

        Ok(tile_sums
            .iter()
            .zip(tile_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect())
    }
}

// ── CPU-side tonemap functions (matching terrain_pbr_pom.wgsl) ───────────────

/// Terrain filmic tonemap — matches tonemap_filmic_terrain() in terrain_pbr_pom.wgsl
fn tonemap_filmic_terrain(x: f32) -> f32 {
    let x = x.max(0.0);
    let a = 0.22f32;
    let b = 0.30f32;
    let c = 0.10f32;
    let d = 0.20f32;
    let e = 0.01f32;
    let f = 0.30f32;

    let curve = |v: f32| -> f32 {
        ((v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f)) - e / f
    };

    let white = 11.2f32;
    curve(x) / curve(white)
}

/// Linear to sRGB — matches linear_to_srgb() in terrain_pbr_pom.wgsl
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
