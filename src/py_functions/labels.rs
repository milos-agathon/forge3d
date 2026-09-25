use super::super::*;

/// CARTOGRAPHER-PRIME: bounded-optimal label declutter solve.
///
/// `candidates` is a list of `(label_id, candidate_index, (min_x, min_y,
/// max_x, max_y), weight, visible)` tuples. Returns `(placements,
/// optimality_gap, rationale)` where `placements` is a list of
/// `(label_id, candidate_index)` pairs sorted by label id, `optimality_gap`
/// is the certified gap versus the objective upper bound (honest — larger —
/// when the node budget was exhausted), and `rationale` is a
/// `LabelRationale` of typed decision records.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "declutter_optimal")]
#[pyo3(signature = (candidates, gap_tolerance = 0.02, node_budget = 200_000, margin = 0.0))]
pub(crate) fn declutter_optimal_py(
    candidates: Vec<(u64, u32, (f32, f32, f32, f32), f64, bool)>,
    gap_tolerance: f64,
    node_budget: u64,
    margin: f32,
) -> PyResult<(
    Vec<(u64, u32)>,
    f64,
    crate::labels::py_bindings::PyLabelRationale,
)> {
    if !(0.0..=1.0).contains(&gap_tolerance) {
        return Err(PyValueError::new_err(
            "gap_tolerance must be within [0.0, 1.0]",
        ));
    }
    if !margin.is_finite() || margin < 0.0 {
        return Err(PyValueError::new_err("margin must be finite and >= 0"));
    }
    let solver_candidates: Vec<crate::labels::SolverCandidate> = candidates
        .iter()
        .map(|(label_id, candidate_index, bounds, weight, visible)| {
            crate::labels::SolverCandidate::new(
                *label_id,
                *candidate_index,
                [bounds.0, bounds.1, bounds.2, bounds.3],
                *weight,
                *visible,
            )
        })
        .collect();
    let config = crate::labels::DeclutterConfig {
        gap_tolerance,
        node_budget,
        margin,
        ..crate::labels::DeclutterConfig::default()
    };
    let outcome = crate::labels::declutter_optimal(&solver_candidates, &config);
    Ok((
        outcome.placements,
        outcome.gap,
        crate::labels::py_bindings::PyLabelRationale {
            records: outcome.rationale,
        },
    ))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "layout_label_candidate")]
#[pyo3(signature = (
    kind,
    label_id,
    candidate_index,
    path,
    text,
    font_size,
    priority,
    viewport,
    glyph_advances = None,
    target_anchor = None,
    tracking = 0.0,
))]
pub(crate) fn layout_label_candidate_py(
    py: Python<'_>,
    kind: &str,
    label_id: u64,
    candidate_index: u32,
    path: Vec<(f32, f32, f32)>,
    text: &str,
    font_size: f32,
    priority: i32,
    viewport: (f32, f32),
    glyph_advances: Option<Vec<f32>>,
    target_anchor: Option<(f32, f32)>,
    tracking: f32,
) -> PyResult<Option<Py<PyDict>>> {
    if kind != "line" && kind != "curved" {
        return Err(PyValueError::new_err(format!(
            "layout_label_candidate kind must be 'line' or 'curved', got {kind:?}"
        )));
    }
    if !font_size.is_finite() || font_size <= 0.0 {
        return Err(PyValueError::new_err("font_size must be finite and > 0"));
    }
    if !viewport.0.is_finite() || !viewport.1.is_finite() || viewport.0 <= 0.0 || viewport.1 <= 0.0
    {
        return Err(PyValueError::new_err(
            "viewport dimensions must be finite and > 0",
        ));
    }
    if !tracking.is_finite() {
        return Err(PyValueError::new_err("tracking must be finite"));
    }
    for (x, y, z) in &path {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(PyValueError::new_err("path coordinates must be finite"));
        }
    }
    if let Some(advances) = &glyph_advances {
        if advances.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("glyph advances must be finite"));
        }
    }
    if let Some((x, y)) = target_anchor {
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("target_anchor must be finite"));
        }
    }
    if text.is_empty() || path.len() < 2 {
        return Ok(None);
    }
    let (width, height) = viewport;
    let char_count = text.chars().count();
    let unit_advances: Vec<f32> = match glyph_advances {
        Some(advances) if advances.len() == char_count => advances,
        _ => crate::labels::compute_glyph_advances(text, 1.0),
    };
    let vertices: Vec<Vec3> = path.iter().map(|(x, y, z)| Vec3::new(*x, *y, *z)).collect();

    let result = PyDict::new_bound(py);
    if kind == "line" {
        let view_proj = glam::Mat4::from_cols(
            glam::Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, -2.0 / height, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(-1.0, 1.0, 0.0, 1.0),
        );
        let pixel_advances: Vec<f32> = unit_advances
            .iter()
            .map(|advance| advance * font_size)
            .collect();
        let mut placements = crate::labels::compute_line_label_placement(
            &vertices,
            text,
            &pixel_advances,
            view_proj,
            width,
            height,
            crate::labels::LineLabelPlacement::Along,
            font_size,
        );
        let Some(initial) = crate::labels::optimal::candidate_from_line_placements(
            label_id,
            candidate_index,
            &placements,
            font_size,
            priority,
        ) else {
            return Ok(None);
        };
        let mut candidate = initial;
        if let Some((target_x, target_y)) = target_anchor {
            let dx = target_x - candidate.position[0];
            let dy = target_y - candidate.position[1];
            if dx != 0.0 || dy != 0.0 {
                for placement in placements.iter_mut() {
                    placement.screen_pos[0] += dx;
                    placement.screen_pos[1] += dy;
                }
                if let Some(moved) = crate::labels::optimal::candidate_from_line_placements(
                    label_id,
                    candidate_index,
                    &placements,
                    font_size,
                    priority,
                ) {
                    candidate = moved;
                }
            }
        }
        result.set_item("position", (candidate.position[0], candidate.position[1]))?;
        result.set_item(
            "bounds",
            (
                candidate.bounds[0],
                candidate.bounds[1],
                candidate.bounds[2],
                candidate.bounds[3],
            ),
        )?;
        result.set_item("geometry_authority", "compute_line_label_placement")?;
        let glyphs = pyo3::types::PyList::empty_bound(py);
        for placement in &placements {
            let glyph = PyDict::new_bound(py);
            glyph.set_item(
                "position",
                (placement.screen_pos[0], placement.screen_pos[1]),
            )?;
            glyph.set_item("rotation", placement.rotation)?;
            glyph.set_item("scale", placement.scale)?;
            glyphs.append(glyph)?;
        }
        result.set_item("glyph_placements", glyphs)?;
    } else {
        let sampled_path = crate::labels::curved::SampledPath::from_polyline(&vertices);
        let mut layout = crate::labels::curved::layout_curved_text(
            text,
            &sampled_path,
            &unit_advances,
            font_size,
            [1.0, 1.0, 1.0, 1.0],
            tracking,
            true,
        );
        let Some(initial) = crate::labels::optimal::candidate_from_curved_layout(
            label_id,
            candidate_index,
            &layout,
            font_size,
            priority,
        ) else {
            return Ok(None);
        };
        let mut candidate = initial;
        if let Some((target_x, target_y)) = target_anchor {
            let dx = target_x - candidate.position[0];
            let dy = target_y - candidate.position[1];
            if dx != 0.0 || dy != 0.0 {
                for glyph in layout.glyphs.iter_mut() {
                    glyph.world_pos.x += dx;
                    glyph.world_pos.y += dy;
                }
                if let Some(moved) = crate::labels::optimal::candidate_from_curved_layout(
                    label_id,
                    candidate_index,
                    &layout,
                    font_size,
                    priority,
                ) {
                    candidate = moved;
                }
            }
        }
        result.set_item("position", (candidate.position[0], candidate.position[1]))?;
        result.set_item(
            "bounds",
            (
                candidate.bounds[0],
                candidate.bounds[1],
                candidate.bounds[2],
                candidate.bounds[3],
            ),
        )?;
        result.set_item("geometry_authority", "layout_curved_text")?;
        let glyphs = pyo3::types::PyList::empty_bound(py);
        for glyph in &layout.glyphs {
            let entry = PyDict::new_bound(py);
            entry.set_item(
                "position",
                (glyph.world_pos.x, glyph.world_pos.y, glyph.world_pos.z),
            )?;
            entry.set_item("rotation", glyph.rotation)?;
            entry.set_item("scale", glyph.scale)?;
            entry.set_item("path_offset", glyph.path_offset)?;
            entry.set_item("character", glyph.character.to_string())?;
            glyphs.append(entry)?;
        }
        result.set_item("glyph_placements", glyphs)?;
    }
    Ok(Some(result.unbind()))
}

pub(crate) use crate::labels::py_text::{
    bake_msdf_atlas_py, bake_msdf_atlas_shaped_py, rasterize_shaped_run_py, text_shape_py,
};
