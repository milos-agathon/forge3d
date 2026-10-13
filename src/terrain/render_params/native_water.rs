use super::parse::{list_to_vec3, to_finite_f32};
use super::*;

/// Immutable decoded surface inputs. Heights precede z_scale; normals are Z-up world-space.
#[derive(Clone, Default)]
pub struct WaterOpticsSettingsNative {
    pub physical: bool,
    pub surface_height: Option<Vec<Vec<f32>>>,
    pub surface_normals: Option<Vec<Vec<[f32; 3]>>>,
    pub color: Option<[f32; 3]>,
    pub roughness: f32,
}

pub(super) fn parse_water_optics(params: &Bound<'_, PyAny>) -> PyResult<WaterOpticsSettingsNative> {
    let Ok(water) = params.getattr("water") else {
        return Ok(WaterOpticsSettingsNative::default());
    };
    let Ok(optics) = water.getattr("optics") else {
        return Ok(WaterOpticsSettingsNative::default());
    };
    if optics.is_none() {
        return Ok(WaterOpticsSettingsNative::default());
    }
    let mode: String = optics.getattr("mode")?.extract()?;
    let physical = match mode.as_str() {
        "physical" => true,
        "procedural" => false,
        _ => {
            return Err(PyValueError::new_err(
                "water optics mode must be physical or procedural",
            ))
        }
    };
    let roughness = to_finite_f32(optics.getattr("roughness")?.as_gil_ref(), "water roughness")?;
    if !(0.0..=1.0).contains(&roughness) {
        return Err(PyValueError::new_err("water roughness must be in [0, 1]"));
    }
    let color_obj = optics.getattr("color")?;
    let color = if color_obj.is_none() {
        None
    } else {
        let rgb = list_to_vec3(color_obj.as_gil_ref(), "water color")?;
        if rgb.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(PyValueError::new_err("water color must be in [0, 1]"));
        }
        Some(rgb)
    };
    let heights_obj = optics.getattr("surface_height")?;
    let surface_height = if heights_obj.is_none() {
        None
    } else {
        let array_obj = heights_obj
            .getattr("_array")
            .unwrap_or_else(|_| heights_obj.clone());
        let rows: Vec<Vec<f32>> =
            if let Ok(array) = array_obj.extract::<numpy::PyReadonlyArray2<'_, f32>>() {
                array
                    .as_array()
                    .outer_iter()
                    .map(|row| row.iter().copied().collect())
                    .collect()
            } else {
                heights_obj.extract()?
            };
        if rows.len() < 2
            || rows[0].len() < 2
            || rows
                .iter()
                .any(|row| row.len() != rows[0].len() || row.iter().any(|v| !v.is_finite()))
        {
            return Err(PyValueError::new_err(
                "water height must be a finite rectangular grid",
            ));
        }
        Some(rows)
    };
    let normals_obj = optics.getattr("surface_normals")?;
    let surface_normals = if normals_obj.is_none() {
        None
    } else {
        let array_obj = normals_obj
            .getattr("_array")
            .unwrap_or_else(|_| normals_obj.clone());
        let rows: Vec<Vec<[f32; 3]>> =
            if let Ok(array) = array_obj.extract::<numpy::PyReadonlyArray3<'_, f32>>() {
                let array = array.as_array();
                if array.shape()[2] != 3 {
                    return Err(PyValueError::new_err(
                        "water normals must have three components",
                    ));
                }
                array
                    .outer_iter()
                    .map(|row| row.outer_iter().map(|n| [n[0], n[1], n[2]]).collect())
                    .collect()
            } else {
                normals_obj.extract()?
            };
        if rows.len() < 2
            || rows[0].len() < 2
            || rows.iter().any(|row| {
                row.len() != rows[0].len()
                    || row
                        .iter()
                        .any(|n| n.iter().any(|v| !v.is_finite()) || n[2] <= 0.0)
            })
        {
            return Err(PyValueError::new_err(
                "water normals must be a finite rectangular grid with positive Z for a heightfield",
            ));
        }
        Some(rows)
    };
    Ok(WaterOpticsSettingsNative {
        physical,
        surface_height,
        surface_normals,
        color,
        roughness,
    })
}
