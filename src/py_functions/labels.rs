use super::super::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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
fn first_char(value: &str) -> Option<char> {
    value.chars().next()
}

#[cfg(feature = "extension-module")]
fn char_at_cluster(text: &str, cluster: u32) -> Option<char> {
    text.get(cluster as usize..)
        .and_then(|suffix| suffix.chars().next())
}

#[cfg(feature = "extension-module")]
fn atlas_reverse_glyph_map(
    face: &rustybuzz::Face<'_>,
    atlas_glyphs: &[String],
) -> HashMap<u32, String> {
    let mut map = HashMap::new();
    for glyph in atlas_glyphs {
        if let Some(ch) = first_char(glyph) {
            if let Some(id) = face.glyph_index(ch) {
                map.entry(u32::from(id.0)).or_insert_with(|| glyph.clone());
            }
        }
    }
    map
}

/// BOP-P3-03: dependency-backed complex text shaping.
///
/// Returns rustybuzz glyph IDs and bidi direction, plus a render glyph sequence
/// mapped back to atlas characters when the provided atlas contains codepoints
/// for the shaped glyph IDs.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "shape_text")]
#[pyo3(signature = (text, font_path, atlas_glyphs = Vec::new()))]
pub(crate) fn shape_text_py(
    py: Python<'_>,
    text: &str,
    font_path: &str,
    atlas_glyphs: Vec<String>,
) -> PyResult<PyObject> {
    if text.is_empty() {
        return Err(PyValueError::new_err("text must not be empty"));
    }
    let path = Path::new(font_path);
    if !path.exists() {
        return Err(PyValueError::new_err(format!(
            "font_path does not exist: {font_path}"
        )));
    }
    let bytes = fs::read(path).map_err(|err| {
        PyValueError::new_err(format!("failed to read font_path {font_path}: {err}"))
    })?;
    let face = rustybuzz::Face::from_slice(&bytes, 0)
        .ok_or_else(|| PyValueError::new_err("font_path is not a valid TrueType/OpenType font"))?;

    let bidi = unicode_bidi::BidiInfo::new(text, None);
    let rtl = bidi
        .paragraphs
        .first()
        .map(|paragraph| paragraph.level.is_rtl())
        .unwrap_or(false);

    let mut buffer = rustybuzz::UnicodeBuffer::new();
    buffer.push_str(text);
    buffer.set_direction(if rtl {
        rustybuzz::Direction::RightToLeft
    } else {
        rustybuzz::Direction::LeftToRight
    });

    let glyph_buffer = rustybuzz::shape(&face, &[], buffer);
    let infos = glyph_buffer.glyph_infos();
    let positions = glyph_buffer.glyph_positions();
    let reverse_map = atlas_reverse_glyph_map(&face, &atlas_glyphs);

    let mut glyph_ids = Vec::with_capacity(infos.len());
    let mut clusters = Vec::with_capacity(infos.len());
    let mut advances = Vec::with_capacity(infos.len());
    let mut offsets = Vec::with_capacity(infos.len());
    let mut glyphs = Vec::with_capacity(infos.len());

    for (info, position) in infos.iter().zip(positions.iter()) {
        let glyph_id = info.glyph_id;
        glyph_ids.push(glyph_id);
        clusters.push(info.cluster);
        advances.push(position.x_advance as f32 / 64.0);
        offsets.push((
            position.x_offset as f32 / 64.0,
            position.y_offset as f32 / 64.0,
        ));
        if let Some(mapped) = reverse_map.get(&glyph_id) {
            glyphs.push(mapped.clone());
        } else if let Some(ch) = char_at_cluster(text, info.cluster) {
            glyphs.push(ch.to_string());
        }
    }

    let dict = PyDict::new_bound(py);
    dict.set_item("engine", "rustybuzz")?;
    dict.set_item("shaping", "rustybuzz")?;
    dict.set_item("direction", if rtl { "rtl" } else { "ltr" })?;
    dict.set_item("glyphs", glyphs)?;
    dict.set_item("glyph_ids", glyph_ids)?;
    dict.set_item("clusters", clusters)?;
    dict.set_item("advances", advances)?;
    dict.set_item("offsets", offsets)?;
    Ok(dict.into())
}
