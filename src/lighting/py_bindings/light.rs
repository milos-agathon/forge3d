use crate::lighting::types::Light;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn extract_f32_array<const N: usize>(obj: &PyAny, name: &str) -> PyResult<[f32; N]> {
    let list: &PyList = obj
        .downcast()
        .map_err(|_| PyValueError::new_err(format!("{} must be a list", name)))?;

    if list.len() != N {
        return Err(PyValueError::new_err(format!(
            "{} must have {} elements",
            name, N
        )));
    }

    let mut arr = [0.0f32; N];
    for (i, item) in list.iter().enumerate() {
        arr[i] = item
            .extract::<f32>()
            .map_err(|_| PyValueError::new_err(format!("{} elements must be floats", name)))?;
    }
    Ok(arr)
}

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Light")]
#[derive(Clone)]
pub struct PyLight {
    #[pyo3(get, set)]
    pub light_type: String,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub color: Vec<f32>,
    #[pyo3(get, set)]
    pub azimuth: f32,
    #[pyo3(get, set)]
    pub elevation: f32,
    #[pyo3(get, set)]
    pub position: Vec<f32>,
    #[pyo3(get, set)]
    pub direction: Vec<f32>,
    #[pyo3(get, set)]
    pub range: f32,
    #[pyo3(get, set)]
    pub spot_inner: f32,
    #[pyo3(get, set)]
    pub spot_outer: f32,
    #[pyo3(get, set)]
    pub env_texture_index: u32,
    #[pyo3(get, set)]
    pub area_width: f32,
    #[pyo3(get, set)]
    pub area_height: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLight {
    #[new]
    #[pyo3(signature = (light_type="Directional", intensity=3.0, color=None, azimuth=135.0, elevation=35.0, position=None, direction=None, range=2000.0, spot_inner=20.0, spot_outer=35.0, env_texture_index=0, area_width=1.0, area_height=1.0))]
    pub fn new(
        light_type: &str,
        intensity: f32,
        color: Option<Vec<f32>>,
        azimuth: f32,
        elevation: f32,
        position: Option<Vec<f32>>,
        direction: Option<Vec<f32>>,
        range: f32,
        spot_inner: f32,
        spot_outer: f32,
        env_texture_index: u32,
        area_width: f32,
        area_height: f32,
    ) -> PyResult<Self> {
        let color = color.unwrap_or(vec![1.0, 1.0, 1.0]);
        let position = position.unwrap_or(vec![0.0, 0.0, 0.0]);
        let direction = direction.unwrap_or(vec![0.0, -1.0, 0.0]);

        if color.len() != 3 {
            return Err(PyValueError::new_err("color must have 3 elements (RGB)"));
        }
        if position.len() != 3 {
            return Err(PyValueError::new_err("position must have 3 elements (XYZ)"));
        }
        if direction.len() != 3 {
            return Err(PyValueError::new_err(
                "direction must have 3 elements (XYZ)",
            ));
        }

        Ok(Self {
            light_type: light_type.to_string(),
            intensity,
            color,
            azimuth,
            elevation,
            position,
            direction,
            range,
            spot_inner,
            spot_outer,
            env_texture_index,
            area_width,
            area_height,
        })
    }
}

impl PyLight {
    pub fn to_native(&self) -> PyResult<Light> {
        let color = [self.color[0], self.color[1], self.color[2]];
        let position = [self.position[0], self.position[1], self.position[2]];
        let direction = [self.direction[0], self.direction[1], self.direction[2]];

        let light = match self.light_type.as_str() {
            "Directional" => {
                Light::directional(self.azimuth, self.elevation, self.intensity, color)
            }
            "Point" => Light::point(position, self.range, self.intensity, color),
            "Spot" => Light::spot(
                position,
                direction,
                self.range,
                self.spot_inner,
                self.spot_outer,
                self.intensity,
                color,
            ),
            "Environment" => Light::environment(self.intensity, self.env_texture_index),
            "AreaRect" => Light::area_rect(
                position,
                direction,
                self.area_width,
                self.area_height,
                self.intensity,
                color,
            ),
            "AreaDisk" => {
                Light::area_disk(position, direction, self.area_width, self.intensity, color)
            }
            "AreaSphere" => Light::area_sphere(position, self.area_width, self.intensity, color),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown light type: {}",
                    self.light_type
                )))
            }
        };

        Ok(light)
    }
}

#[cfg(feature = "extension-module")]
pub fn parse_light_dict(_py: Python, dict: &PyAny) -> PyResult<Light> {
    let dict = dict
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Light spec must be a dict"))?;

    let light_type: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("Light dict must have 'type' key"))?
        .extract()?;

    let intensity: f32 = dict
        .get_item("intensity")?
        .or_else(|| dict.get_item("power").ok().flatten())
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1.0);

    let color: [f32; 3] = dict
        .get_item("color")?
        .or_else(|| dict.get_item("rgb").ok().flatten())
        .map(|v| extract_f32_array::<3>(v, "color"))
        .transpose()?
        .unwrap_or([1.0, 1.0, 1.0]);

    match light_type.to_lowercase().as_str() {
        "directional" => {
            let azimuth: f32 = dict
                .get_item("azimuth")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(135.0);
            let elevation: f32 = dict
                .get_item("elevation")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(35.0);
            Ok(Light::directional(azimuth, elevation, intensity, color))
        }
        "point" => {
            let position = dict
                .get_item("position")?
                .or_else(|| dict.get_item("pos").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Point light requires 'position'"))?
                .extract::<[f32; 3]>()?;
            let range: f32 = dict
                .get_item("range")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(100.0);
            Ok(Light::point(position, range, intensity, color))
        }
        "spot" => {
            let position = dict
                .get_item("position")?
                .or_else(|| dict.get_item("pos").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Spot light requires 'position'"))?
                .extract::<[f32; 3]>()?;
            let direction = dict
                .get_item("direction")?
                .or_else(|| dict.get_item("dir").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Spot light requires 'direction'"))?
                .extract::<[f32; 3]>()?;
            let range: f32 = dict
                .get_item("range")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(100.0);
            let outer_angle: f32 = dict
                .get_item("cone_angle")?
                .or_else(|| dict.get_item("angle").ok().flatten())
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(35.0);
            let inner_angle: f32 = dict
                .get_item("inner_angle")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(outer_angle * 0.6);
            Ok(Light::spot(
                position,
                direction,
                range,
                inner_angle,
                outer_angle,
                intensity,
                color,
            ))
        }
        "environment" | "env" => {
            let texture_index: u32 = dict
                .get_item("texture_index")?
                .or_else(|| dict.get_item("env_texture_index").ok().flatten())
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0);
            Ok(Light::environment(intensity, texture_index))
        }
        "area_rect" | "arearect" => {
            let position = dict
                .get_item("position")?
                .or_else(|| dict.get_item("pos").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Area rect light requires 'position'"))?
                .extract::<[f32; 3]>()?;
            let direction = dict
                .get_item("direction")?
                .or_else(|| dict.get_item("dir").ok().flatten())
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or([0.0, -1.0, 0.0]);
            let area_extent = dict
                .get_item("area_extent")?
                .ok_or_else(|| PyValueError::new_err("Area rect light requires 'area_extent'"))?;
            let [width, height] = extract_f32_array::<2>(area_extent, "area_extent")?;
            Ok(Light::area_rect(
                position, direction, width, height, intensity, color,
            ))
        }
        "area_disk" | "areadisk" => {
            let position = dict
                .get_item("position")?
                .or_else(|| dict.get_item("pos").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Area disk light requires 'position'"))?
                .extract::<[f32; 3]>()?;
            let direction = dict
                .get_item("direction")?
                .or_else(|| dict.get_item("dir").ok().flatten())
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or([0.0, -1.0, 0.0]);
            let radius: f32 = dict
                .get_item("radius")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(1.0);
            Ok(Light::area_disk(
                position, direction, radius, intensity, color,
            ))
        }
        "area_sphere" | "areasphere" => {
            let position = dict
                .get_item("position")?
                .or_else(|| dict.get_item("pos").ok().flatten())
                .ok_or_else(|| PyValueError::new_err("Area sphere light requires 'position'"))?
                .extract::<[f32; 3]>()?;
            let radius: f32 = dict
                .get_item("radius")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(1.0);
            Ok(Light::area_sphere(position, radius, intensity, color))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown light type: {}",
            light_type
        ))),
    }
}
