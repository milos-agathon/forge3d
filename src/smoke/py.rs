use ndarray::Array3;
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use super::{SmokeDomainConfig, SmokeEmitter, SmokeRenderSettings, SmokeStepSettings, SmokeVolume};

#[pyclass(module = "forge3d._forge3d", name = "SmokeEmitter")]
#[derive(Clone)]
pub struct PySmokeEmitter {
    pub inner: SmokeEmitter,
}

#[pymethods]
impl PySmokeEmitter {
    #[new]
    #[pyo3(signature = (
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        density_rate=1.0,
        temperature_rate=1.0,
        fuel_rate=0.0,
        soot_rate=0.2,
        humidity_rate=0.0,
        emission_rate=1.0,
        velocity=(0.0, 1.0, 0.0),
        start_time=0.0,
        end_time=3.4028235e38
    ))]
    fn new(
        center: (f32, f32, f32),
        radius: f32,
        density_rate: f32,
        temperature_rate: f32,
        fuel_rate: f32,
        soot_rate: f32,
        humidity_rate: f32,
        emission_rate: f32,
        velocity: (f32, f32, f32),
        start_time: f32,
        end_time: f32,
    ) -> PyResult<Self> {
        let inner = SmokeEmitter {
            center: tuple3(center),
            radius,
            density_rate,
            temperature_rate,
            fuel_rate,
            soot_rate,
            humidity_rate,
            emission_rate,
            velocity: tuple3(velocity),
            start_time,
            end_time,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn center(&self) -> (f32, f32, f32) {
        array3(self.inner.center)
    }

    #[setter]
    fn set_center(&mut self, value: (f32, f32, f32)) -> PyResult<()> {
        self.inner.center = tuple3(value);
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn radius(&self) -> f32 {
        self.inner.radius
    }

    #[setter]
    fn set_radius(&mut self, value: f32) -> PyResult<()> {
        self.inner.radius = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn velocity(&self) -> (f32, f32, f32) {
        array3(self.inner.velocity)
    }

    #[setter]
    fn set_velocity(&mut self, value: (f32, f32, f32)) -> PyResult<()> {
        self.inner.velocity = tuple3(value);
        self.inner.validate().map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "SmokeEmitter(center={:?}, radius={}, density_rate={})",
            self.inner.center, self.inner.radius, self.inner.density_rate
        )
    }
}

#[pymethods]
impl PySmokeEmitter {
    #[getter]
    fn density_rate(&self) -> f32 {
        self.inner.density_rate
    }

    #[setter]
    fn set_density_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.density_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn temperature_rate(&self) -> f32 {
        self.inner.temperature_rate
    }

    #[setter]
    fn set_temperature_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.temperature_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn fuel_rate(&self) -> f32 {
        self.inner.fuel_rate
    }

    #[setter]
    fn set_fuel_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.fuel_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn soot_rate(&self) -> f32 {
        self.inner.soot_rate
    }

    #[setter]
    fn set_soot_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.soot_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn humidity_rate(&self) -> f32 {
        self.inner.humidity_rate
    }

    #[setter]
    fn set_humidity_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.humidity_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn emission_rate(&self) -> f32 {
        self.inner.emission_rate
    }

    #[setter]
    fn set_emission_rate(&mut self, value: f32) -> PyResult<()> {
        self.inner.emission_rate = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn start_time(&self) -> f32 {
        self.inner.start_time
    }

    #[setter]
    fn set_start_time(&mut self, value: f32) -> PyResult<()> {
        self.inner.start_time = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }

    #[getter]
    fn end_time(&self) -> f32 {
        self.inner.end_time
    }

    #[setter]
    fn set_end_time(&mut self, value: f32) -> PyResult<()> {
        self.inner.end_time = value;
        self.inner.validate().map_err(PyValueError::new_err)
    }
}

#[pyclass(module = "forge3d._forge3d", name = "SmokeStepSettings")]
#[derive(Clone)]
pub struct PySmokeStepSettings {
    pub inner: SmokeStepSettings,
}

#[pymethods]
impl PySmokeStepSettings {
    #[new]
    #[pyo3(signature = (
        dt=0.033333335,
        density_decay=0.015,
        temperature_decay=0.08,
        velocity_damping=0.01,
        diffusion=0.0005,
        buoyancy=0.7,
        vorticity=0.12,
        pressure_iterations=20,
        turbulence_strength=0.0,
        turbulence_seed=0,
        mac_cormack=false,
        mass_conservation=true,
        terrain_collision=true,
        boundary_damping=0.0,
        wind=(0.0, 0.0, 0.0)
    ))]
    fn new(
        dt: f32,
        density_decay: f32,
        temperature_decay: f32,
        velocity_damping: f32,
        diffusion: f32,
        buoyancy: f32,
        vorticity: f32,
        pressure_iterations: u32,
        turbulence_strength: f32,
        turbulence_seed: u32,
        mac_cormack: bool,
        mass_conservation: bool,
        terrain_collision: bool,
        boundary_damping: f32,
        wind: (f32, f32, f32),
    ) -> PyResult<Self> {
        let inner = SmokeStepSettings {
            dt,
            density_decay,
            temperature_decay,
            velocity_damping,
            diffusion,
            buoyancy,
            vorticity,
            pressure_iterations,
            turbulence_strength,
            turbulence_seed,
            mac_cormack,
            mass_conservation,
            terrain_collision,
            boundary_damping,
            wind: tuple3(wind),
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SmokeStepSettings(dt={}, pressure_iterations={})",
            self.inner.dt, self.inner.pressure_iterations
        )
    }
}

#[pyclass(module = "forge3d._forge3d", name = "SmokeRenderSettings")]
#[derive(Clone)]
pub struct PySmokeRenderSettings {
    pub inner: SmokeRenderSettings,
}

#[pymethods]
impl PySmokeRenderSettings {
    #[new]
    #[pyo3(signature = (
        density_scale=1.0,
        extinction=2.6,
        scattering=0.85,
        absorption=0.45,
        phase_g=0.24,
        step_size=0.0,
        max_steps=256,
        self_shadow=true,
        shadow_steps=20,
        shadow_step_size=0.0,
        jitter_strength=0.5,
        exposure=1.0,
        thin_color=(0.50, 0.54, 0.58),
        dense_color=(0.93, 0.91, 0.82),
        soot_absorption=0.22,
        fire_glow=0.35
    ))]
    fn new(
        density_scale: f32,
        extinction: f32,
        scattering: f32,
        absorption: f32,
        phase_g: f32,
        step_size: f32,
        max_steps: u32,
        self_shadow: bool,
        shadow_steps: u32,
        shadow_step_size: f32,
        jitter_strength: f32,
        exposure: f32,
        thin_color: (f32, f32, f32),
        dense_color: (f32, f32, f32),
        soot_absorption: f32,
        fire_glow: f32,
    ) -> PyResult<Self> {
        let inner = SmokeRenderSettings {
            density_scale,
            extinction,
            scattering,
            absorption,
            phase_g,
            step_size,
            max_steps,
            self_shadow,
            shadow_steps,
            shadow_step_size,
            jitter_strength,
            exposure,
            thin_color: tuple3(thin_color),
            dense_color: tuple3(dense_color),
            soot_absorption,
            fire_glow,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SmokeRenderSettings(extinction={}, phase_g={}, max_steps={})",
            self.inner.extinction, self.inner.phase_g, self.inner.max_steps
        )
    }
}

#[pyclass(module = "forge3d._forge3d", name = "SmokeDomain")]
pub struct PySmokeDomain {
    pub inner: SmokeVolume,
}

#[pymethods]
impl PySmokeDomain {
    #[new]
    #[pyo3(signature = (
        dims,
        voxel_size=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        brick_size=(16, 16, 16),
        sparse_threshold=1.0e-5
    ))]
    fn new(
        dims: (usize, usize, usize),
        voxel_size: (f32, f32, f32),
        origin: (f32, f32, f32),
        brick_size: (usize, usize, usize),
        sparse_threshold: f32,
    ) -> PyResult<Self> {
        let mut config =
            SmokeDomainConfig::new(tuple3_usize(dims), tuple3(voxel_size), tuple3(origin))
                .map_err(PyValueError::new_err)?;
        config.brick_size = tuple3_usize(brick_size);
        config.sparse_threshold = sparse_threshold;
        config.validate().map_err(PyValueError::new_err)?;
        let inner = SmokeVolume::new(config).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (density, voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)))]
    fn from_density(
        density: PyReadonlyArray3<'_, f32>,
        voxel_size: (f32, f32, f32),
        origin: (f32, f32, f32),
    ) -> PyResult<Self> {
        let shape = density.shape();
        let dims = [shape[2], shape[1], shape[0]];
        let mut domain = Self::new(
            (dims[0], dims[1], dims[2]),
            voxel_size,
            origin,
            (16, 16, 16),
            1.0e-5,
        )?;
        let data = density
            .as_slice()
            .map_err(|_| PyValueError::new_err("density must be contiguous float32"))?
            .to_vec();
        domain
            .inner
            .set_density(data)
            .map_err(PyValueError::new_err)?;
        Ok(domain)
    }

    #[getter]
    fn dims(&self) -> (usize, usize, usize) {
        let d = self.inner.dims();
        (d[0], d[1], d[2])
    }

    #[getter]
    fn voxel_size(&self) -> (f32, f32, f32) {
        array3(self.inner.config.voxel_size)
    }

    #[getter]
    fn origin(&self) -> (f32, f32, f32) {
        array3(self.inner.config.origin)
    }

    #[getter]
    fn time_seconds(&self) -> f32 {
        self.inner.time_seconds
    }

    #[getter]
    fn frame_index(&self) -> u64 {
        self.inner.frame_index
    }

    fn set_density(&mut self, density: PyReadonlyArray3<'_, f32>) -> PyResult<()> {
        let shape = density.shape();
        let dims = self.inner.dims();
        if [shape[2], shape[1], shape[0]] != dims {
            return Err(PyValueError::new_err(format!(
                "density shape must be (z={}, y={}, x={})",
                dims[2], dims[1], dims[0]
            )));
        }
        let data = density
            .as_slice()
            .map_err(|_| PyValueError::new_err("density must be contiguous float32"))?
            .to_vec();
        self.inner.set_density(data).map_err(PyValueError::new_err)
    }

    fn set_velocity(&mut self, velocity: PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
        let shape = velocity.shape();
        let dims = self.inner.dims();
        if shape != [dims[2], dims[1], dims[0], 3] {
            return Err(PyValueError::new_err(format!(
                "velocity shape must be (z={}, y={}, x={}, 3)",
                dims[2], dims[1], dims[0]
            )));
        }
        let data = velocity
            .as_slice()
            .map_err(|_| PyValueError::new_err("velocity must be contiguous float32"))?
            .to_vec();
        self.inner.set_velocity(data).map_err(PyValueError::new_err)
    }

    fn add_emitter(&mut self, emitter: PyRef<'_, PySmokeEmitter>, dt: f32) -> PyResult<()> {
        self.inner
            .add_emitter(&emitter.inner, dt)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (settings, emitters=None))]
    fn step(
        &mut self,
        settings: PyRef<'_, PySmokeStepSettings>,
        emitters: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let parsed = parse_emitters(emitters)?;
        self.inner
            .step(&settings.inner, &parsed)
            .map_err(PyRuntimeError::new_err)
    }

    fn to_density_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f32>> {
        let dims = self.inner.dims();
        let arr = Array3::from_shape_vec((dims[2], dims[1], dims[0]), self.inner.density.clone())
            .map_err(|_| PyRuntimeError::new_err("failed to reshape density"))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn to_velocity_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dims = self.inner.dims();
        let arr = PyArray1::from_vec_bound(py, self.inner.velocity.clone());
        let reshaped = arr.call_method1("reshape", (dims[2], dims[1], dims[0], 3))?;
        Ok(reshaped.into_py(py))
    }

    fn to_particle_age_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f32>> {
        let dims = self.inner.dims();
        let arr =
            Array3::from_shape_vec((dims[2], dims[1], dims[0]), self.inner.particle_age.clone())
                .map_err(|_| PyRuntimeError::new_err("failed to reshape particle age"))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn sample_density(&self, position: (f32, f32, f32)) -> f32 {
        self.inner.sample_density_world(tuple3(position))
    }

    fn sample_temperature(&self, position: (f32, f32, f32)) -> f32 {
        self.inner.sample_temperature_world(tuple3(position))
    }

    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = self.inner.memory_report();
        let dict = PyDict::new_bound(py);
        dict.set_item("voxel_count", report.voxel_count)?;
        dict.set_item("dense_bytes", report.dense_bytes)?;
        dict.set_item("active_bricks", report.active_bricks)?;
        dict.set_item("total_bricks", report.total_bricks)?;
        dict.set_item("sparse_bytes_estimate", report.sparse_bytes_estimate)?;
        dict.set_item("utilization", report.utilization)?;
        dict.set_item("time_seconds", report.time_seconds)?;
        dict.set_item("frame_index", report.frame_index)?;
        Ok(dict)
    }

    fn physics_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("mass", self.inner.mass())?;
        dict.set_item("max_density", self.inner.max_density())?;
        dict.set_item("divergence_l2", self.inner.divergence_l2())?;
        dict.set_item("time_seconds", self.inner.time_seconds)?;
        dict.set_item("frame_index", self.inner.frame_index)?;
        Ok(dict)
    }

    #[pyo3(signature = (
        width,
        height,
        camera_pos,
        target,
        up=(0.0, 1.0, 0.0),
        fovy_deg=45.0,
        sun_direction=(0.4, 0.8, -0.2),
        settings=None,
        certificate=None,
        cache=None
    ))]
    fn render_rgba<'py>(
        &self,
        py: Python<'py>,
        width: u32,
        height: u32,
        camera_pos: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        sun_direction: (f32, f32, f32),
        settings: Option<PyRef<'_, PySmokeRenderSettings>>,
        certificate: Option<Bound<'_, PyAny>>,
        cache: Option<Bound<'_, PyAny>>,
    ) -> PyResult<&'py PyArray3<u8>> {
        let _ = cache;
        let certificate_capture =
            crate::core::certificate::begin_cpu_render_capture("smoke.render_rgba");
        let native_settings = settings
            .as_ref()
            .map(|settings| settings.inner.clone())
            .unwrap_or_default();
        let rgba = self
            .inner
            .raymarch_rgba(
                width,
                height,
                tuple3(camera_pos),
                tuple3(target),
                tuple3(up),
                fovy_deg,
                tuple3(sun_direction),
                &native_settings,
            )
            .map_err(PyRuntimeError::new_err)?;
        let arr = Array3::from_shape_vec((height as usize, width as usize, 4), rgba)
            .map_err(|_| PyRuntimeError::new_err("failed to reshape rendered RGBA"))?;
        let arr = arr.into_pyarray_bound(py);
        crate::core::certificate::record_pass("smoke.cpu_raymarch", 0.0, 1);
        certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok(arr.into_gil_ref())
    }

    #[pyo3(signature = (
        width,
        height,
        view_direction=(0.0, -1.0, 0.0),
        sun_direction=(0.4, 0.8, -0.2),
        settings=None,
        certificate=None,
        cache=None
    ))]
    fn render_projection_rgba<'py>(
        &self,
        py: Python<'py>,
        width: u32,
        height: u32,
        view_direction: (f32, f32, f32),
        sun_direction: (f32, f32, f32),
        settings: Option<PyRef<'_, PySmokeRenderSettings>>,
        certificate: Option<Bound<'_, PyAny>>,
        cache: Option<Bound<'_, PyAny>>,
    ) -> PyResult<&'py PyArray3<u8>> {
        let _ = cache;
        let certificate_capture =
            crate::core::certificate::begin_cpu_render_capture("smoke.render_projection_rgba");
        let native_settings = settings
            .as_ref()
            .map(|settings| settings.inner.clone())
            .unwrap_or_default();
        let rgba = self
            .inner
            .raymarch_projection_rgba(
                width,
                height,
                tuple3(view_direction),
                tuple3(sun_direction),
                &native_settings,
            )
            .map_err(PyRuntimeError::new_err)?;
        let arr = Array3::from_shape_vec((height as usize, width as usize, 4), rgba)
            .map_err(|_| PyRuntimeError::new_err("failed to reshape rendered RGBA"))?;
        let arr = arr.into_pyarray_bound(py);
        crate::core::certificate::record_pass("smoke.cpu_projection", 0.0, 1);
        certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok(arr.into_gil_ref())
    }

    fn __repr__(&self) -> String {
        format!(
            "SmokeDomain(dims={:?}, time_seconds={:.3}, frame_index={})",
            self.inner.dims(),
            self.inner.time_seconds,
            self.inner.frame_index
        )
    }
}

fn parse_emitters(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<SmokeEmitter>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let seq = obj.downcast::<PySequence>()?;
    let mut out = Vec::with_capacity(seq.len()? as usize);
    for item in seq.iter()? {
        let item = item?;
        let emitter: PyRef<'_, PySmokeEmitter> = item.extract()?;
        out.push(emitter.inner.clone());
    }
    Ok(out)
}

fn tuple3(value: (f32, f32, f32)) -> [f32; 3] {
    [value.0, value.1, value.2]
}

fn tuple3_usize(value: (usize, usize, usize)) -> [usize; 3] {
    [value.0, value.1, value.2]
}

fn array3(value: [f32; 3]) -> (f32, f32, f32) {
    (value[0], value[1], value[2])
}
