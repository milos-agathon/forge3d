use glam::Vec3;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use super::{SdfPrimitive, SdfScene, SdfSceneBuilder};

#[cfg_attr(
    feature = "extension-module",
    pyclass(name = "SdfPrimitive", module = "forge3d._forge3d")
)]
#[derive(Clone)]
pub struct PySdfPrimitive {
    pub primitive: SdfPrimitive,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySdfPrimitive {
    #[classmethod]
    pub fn sphere(
        _cls: &Bound<'_, PyType>,
        center: (f32, f32, f32),
        radius: f32,
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::sphere(center.into(), radius, material_id),
        }
    }

    #[classmethod]
    pub fn r#box(
        _cls: &Bound<'_, PyType>,
        center: (f32, f32, f32),
        extents: (f32, f32, f32),
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::box_primitive(center.into(), extents.into(), material_id),
        }
    }

    #[classmethod]
    pub fn cylinder(
        _cls: &Bound<'_, PyType>,
        center: (f32, f32, f32),
        radius: f32,
        height: f32,
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::cylinder(center.into(), radius, height, material_id),
        }
    }

    #[classmethod]
    pub fn plane(
        _cls: &Bound<'_, PyType>,
        normal: (f32, f32, f32),
        distance: f32,
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::plane(normal.into(), distance, material_id),
        }
    }

    #[classmethod]
    pub fn torus(
        _cls: &Bound<'_, PyType>,
        center: (f32, f32, f32),
        major_radius: f32,
        minor_radius: f32,
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::torus(center.into(), major_radius, minor_radius, material_id),
        }
    }

    #[classmethod]
    pub fn capsule(
        _cls: &Bound<'_, PyType>,
        point_a: (f32, f32, f32),
        point_b: (f32, f32, f32),
        radius: f32,
        material_id: u32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::capsule(point_a.into(), point_b.into(), radius, material_id),
        }
    }

    #[getter]
    pub fn primitive_type(&self) -> u32 {
        self.primitive.primitive_type
    }

    #[getter]
    pub fn material_id(&self) -> u32 {
        self.primitive.material_id
    }
}

#[cfg_attr(
    feature = "extension-module",
    pyclass(name = "SdfScene", module = "forge3d._forge3d")
)]
#[derive(Clone)]
pub struct PySdfScene(pub SdfScene);

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySdfScene {
    #[new]
    pub fn new() -> Self {
        Self(SdfScene::new())
    }

    #[classmethod]
    pub fn single_primitive(_cls: &Bound<'_, PyType>, primitive: &PySdfPrimitive) -> Self {
        Self(SdfScene::single_primitive(primitive.primitive.clone()))
    }

    pub fn set_bounds(&mut self, min_bounds: (f32, f32, f32), max_bounds: (f32, f32, f32)) {
        self.0
            .set_bounds(Vec3::from(min_bounds), Vec3::from(max_bounds));
    }

    pub fn primitive_count(&self) -> usize {
        self.0.primitive_count()
    }

    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    pub fn evaluate(&self, point: (f32, f32, f32)) -> (f32, u32) {
        let result = self.0.evaluate(Vec3::from(point));
        (result.distance, result.material_id)
    }

    pub fn in_bounds(&self, point: (f32, f32, f32)) -> bool {
        self.0.in_bounds(Vec3::from(point))
    }

    pub fn clone_scene(&self) -> PySdfScene {
        PySdfScene(self.0.clone())
    }
}

#[cfg_attr(
    feature = "extension-module",
    pyclass(name = "SdfSceneBuilder", module = "forge3d._forge3d")
)]
pub struct PySdfSceneBuilder {
    builder: Option<SdfSceneBuilder>,
}

impl PySdfSceneBuilder {
    fn take_builder(&mut self) -> PyResult<SdfSceneBuilder> {
        self.builder
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("builder consumed"))
    }

    fn put_builder(&mut self, builder: SdfSceneBuilder) {
        self.builder = Some(builder);
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySdfSceneBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            builder: Some(SdfSceneBuilder::new()),
        }
    }

    pub fn add_sphere(
        &mut self,
        center: (f32, f32, f32),
        radius: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.add_sphere(center.into(), radius, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn smooth_intersection(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.smooth_intersection(left, right, smoothing, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn smooth_subtraction(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.smooth_subtraction(left, right, smoothing, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn add_box(
        &mut self,
        center: (f32, f32, f32),
        extents: (f32, f32, f32),
        material_id: u32,
    ) -> PyResult<u32> {
        let mut builder = self.take_builder()?;
        let node = builder.add_box_mut(center.into(), extents.into(), material_id);
        let result = Ok(node);
        self.put_builder(builder);
        result
    }

    pub fn add_cylinder(
        &mut self,
        center: (f32, f32, f32),
        radius: f32,
        height: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let mut builder = self.take_builder()?;
        let node = builder.add_cylinder_mut(center.into(), radius, height, material_id);
        let result = Ok(node);
        self.put_builder(builder);
        result
    }

    pub fn add_plane(
        &mut self,
        normal: (f32, f32, f32),
        distance: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let mut builder = self.take_builder()?;
        let node = builder.add_plane_mut(normal.into(), distance, material_id);
        let result = Ok(node);
        self.put_builder(builder);
        result
    }

    pub fn add_torus(
        &mut self,
        center: (f32, f32, f32),
        major_radius: f32,
        minor_radius: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let mut builder = self.take_builder()?;
        let node = builder.add_torus_mut(center.into(), major_radius, minor_radius, material_id);
        let result = Ok(node);
        self.put_builder(builder);
        result
    }

    pub fn add_capsule(
        &mut self,
        point_a: (f32, f32, f32),
        point_b: (f32, f32, f32),
        radius: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let mut builder = self.take_builder()?;
        let node = builder.add_capsule_mut(point_a.into(), point_b.into(), radius, material_id);
        let result = Ok(node);
        self.put_builder(builder);
        result
    }

    pub fn union(&mut self, left: u32, right: u32, material_id: u32) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.union(left, right, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn smooth_union(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.smooth_union(left, right, smoothing, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn subtract(&mut self, left: u32, right: u32, material_id: u32) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.subtract(left, right, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn intersect(&mut self, left: u32, right: u32, material_id: u32) -> PyResult<u32> {
        let builder = self.take_builder()?;
        let (builder, node) = builder.intersect(left, right, material_id);
        self.put_builder(builder);
        Ok(node)
    }

    pub fn build(&mut self) -> PyResult<PySdfScene> {
        let builder = self.take_builder()?;
        let scene = builder.build();
        Ok(PySdfScene(scene))
    }

    pub fn reset(&mut self) {
        self.put_builder(SdfSceneBuilder::new());
    }
}
