use super::*;
use crate::vector::api::{PointDef, PolylineDef, VectorStyle};

pub(super) fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
    if let Some(obj) = list {
        Ok(obj
            .extract::<Vec<(f32, f32)>>()?
            .into_iter()
            .map(|(x, y)| glam::Vec2::new(x, y))
            .collect())
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn extract_rgba_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f32; 4]>> {
    if let Some(obj) = list {
        Ok(obj
            .extract::<Vec<(f32, f32, f32, f32)>>()?
            .into_iter()
            .map(|(r, g, b, a)| [r, g, b, a])
            .collect())
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn extract_f32_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
    if let Some(obj) = list {
        Ok(obj.extract()?)
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
    if let Some(obj) = list {
        Ok(obj
            .extract::<Vec<Vec<(f32, f32)>>>()?
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .map(|(x, y)| glam::Vec2::new(x, y))
                    .collect()
            })
            .collect())
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn build_point_defs(
    points: &[glam::Vec2],
    colors: &[[f32; 4]],
    sizes: &[f32],
) -> Vec<PointDef> {
    points
        .iter()
        .enumerate()
        .map(|(index, position)| PointDef {
            position: *position,
            style: VectorStyle {
                fill_color: *colors.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]),
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                point_size: *sizes.get(index).unwrap_or(&8.0),
            },
        })
        .collect()
}

pub(super) fn build_poly_defs(
    lines: &[Vec<glam::Vec2>],
    colors: &[[f32; 4]],
    widths: &[f32],
) -> Vec<PolylineDef> {
    lines
        .iter()
        .enumerate()
        .map(|(index, path)| PolylineDef {
            path: path.clone(),
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: *colors.get(index).unwrap_or(&[0.2, 0.8, 0.2, 0.6]),
                stroke_width: *widths.get(index).unwrap_or(&2.0),
                point_size: 4.0,
            },
        })
        .collect()
}
