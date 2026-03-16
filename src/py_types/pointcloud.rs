use super::super::*;

// P2.1: Point Cloud GPU rendering path – PyO3 bindings

/// Python-visible point buffer with GPU interleaving support.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "PointBuffer")]
pub struct PyPointBuffer {
    inner: crate::pointcloud::PointBuffer,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyPointBuffer {
    /// Construct a PointBuffer from flat position and optional color arrays.
    ///
    /// * `positions` – flat f32 array [x,y,z, x,y,z, ...]
    /// * `colors`    – optional flat u8 array [r,g,b, r,g,b, ...] (same point count)
    #[new]
    #[pyo3(signature = (positions, colors = None))]
    fn new(positions: Vec<f32>, colors: Option<Vec<u8>>) -> PyResult<Self> {
        if positions.len() % 3 != 0 {
            return Err(PyValueError::new_err(
                "positions length must be a multiple of 3",
            ));
        }
        let point_count = positions.len() / 3;
        if let Some(ref c) = colors {
            if c.len() != point_count * 3 {
                return Err(PyValueError::new_err(format!(
                    "colors length {} does not match point_count*3 = {}",
                    c.len(),
                    point_count * 3
                )));
            }
        }
        Ok(Self {
            inner: crate::pointcloud::PointBuffer {
                positions,
                colors,
                point_count,
            },
        })
    }

    /// Number of points stored in this buffer.
    #[getter]
    fn point_count(&self) -> usize {
        self.inner.point_count
    }

    /// Raw CPU byte size (positions + colors).
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }

    /// Byte size of the interleaved GPU buffer that `create_gpu_buffer` produces.
    fn gpu_byte_size(&self) -> usize {
        self.inner.gpu_byte_size()
    }

    /// Create an interleaved GPU vertex buffer as a numpy float32 array.
    ///
    /// Layout per point: [x, y, z, r, g, b] (6 floats).
    /// Colors are normalised from u8 to 0.0..1.0; white if absent.
    fn create_gpu_buffer<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let data = self.inner.create_gpu_buffer();
        PyArray1::from_vec_bound(py, data)
    }

    /// Create a viewer-compatible GPU buffer as a numpy float32 array.
    ///
    /// Layout per point: `[x, y, z, elevation_norm, r, g, b, intensity, size, pad, pad, pad]`
    /// (12 floats = 48 bytes), matching the viewer's `PointInstance3D` struct.
    ///
    /// * `bounds_min` – `[min_x, min_y, min_z]` used for elevation normalisation
    /// * `bounds_max` – `[max_x, max_y, max_z]` used for elevation normalisation
    #[pyo3(signature = (bounds_min, bounds_max))]
    fn create_viewer_gpu_buffer<'py>(
        &self,
        py: Python<'py>,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Bound<'py, PyArray1<f32>> {
        let data = self.inner.create_viewer_gpu_buffer(bounds_min, bounds_max);
        PyArray1::from_vec_bound(py, data)
    }

    fn __repr__(&self) -> String {
        format!(
            "PointBuffer(point_count={}, cpu_bytes={}, gpu_bytes={})",
            self.inner.point_count,
            self.inner.byte_size(),
            self.inner.gpu_byte_size(),
        )
    }
}
