use super::super::*;

/// Internal scoped host-visible reservation for compile-time label depth
/// arrays. The returned object holds the authoritative tracker entry until
/// `close()` or garbage collection.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "_reserve_label_depth_host_allocation")]
pub(crate) fn reserve_label_depth_host_allocation_py(
    bytes: u64,
    label: &str,
) -> PyResult<crate::labels::py_bindings::PyLabelDepthHostAllocation> {
    let reservation = crate::labels::DepthHostAllocationReservation::reserve(bytes, label)?;
    Ok(crate::labels::py_bindings::PyLabelDepthHostAllocation { reservation })
}

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
#[pyo3(signature = (candidates, gap_tolerance = 0.02, node_budget = None, margin = 0.0))]
pub(crate) fn declutter_optimal_py(
    candidates: Vec<(u64, u32, (f32, f32, f32, f32), f64, bool)>,
    gap_tolerance: f64,
    node_budget: Option<u64>,
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
            crate::labels::SolverCandidate::try_new(
                *label_id,
                *candidate_index,
                [bounds.0, bounds.1, bounds.2, bounds.3],
                *weight,
                *visible,
            )
        })
        .collect::<Result<_, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let config = crate::labels::DeclutterConfig {
        gap_tolerance,
        node_budget,
        margin,
        ..crate::labels::DeclutterConfig::default()
    };
    let outcome = crate::labels::declutter_optimal(&solver_candidates, &config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        outcome.placements,
        outcome.gap,
        crate::labels::py_bindings::PyLabelRationale {
            records: outcome.rationale,
        },
    ))
}

/// Generic native declutter entry point. The optimal algorithm is the typed,
/// certified surface; the dedicated `declutter_optimal` name remains as a
/// compatibility alias with the same inputs and outputs.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "declutter")]
#[pyo3(signature = (candidates, algorithm = "optimal", gap_tolerance = 0.02, node_budget = None, margin = 0.0))]
pub(crate) fn declutter_py(
    candidates: Vec<(u64, u32, (f32, f32, f32, f32), f64, bool)>,
    algorithm: &str,
    gap_tolerance: f64,
    node_budget: Option<u64>,
    margin: f32,
) -> PyResult<(
    Vec<(u64, u32)>,
    f64,
    crate::labels::py_bindings::PyLabelRationale,
)> {
    if algorithm != "optimal" {
        return Err(PyValueError::new_err(format!(
            "unsupported typed declutter algorithm {algorithm:?}; expected 'optimal'"
        )));
    }
    declutter_optimal_py(candidates, gap_tolerance, node_budget, margin)
}

/// Arc-length point on a screen-space polyline `(x, y, depth)`.
/// Returns `(x, y, z)` with `z` linearly interpolated along the segment.
fn screen_path_sample(path: &[(f32, f32, f32)], distance: f32) -> Option<(f32, f32, f32)> {
    let mut traversed = 0.0f64;
    for segment in path.windows(2) {
        let (ax, ay, az) = segment[0];
        let (bx, by, bz) = segment[1];
        let length = ((bx - ax) as f64).hypot((by - ay) as f64);
        if length > 0.0 && traversed + length >= distance as f64 {
            let t = ((distance as f64 - traversed) / length) as f32;
            return Some((ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t));
        }
        traversed += length;
    }
    path.last().copied()
}

fn screen_path_length(path: &[(f32, f32, f32)]) -> f64 {
    path.windows(2)
        .map(|segment| {
            ((segment[1].0 - segment[0].0) as f64).hypot((segment[1].1 - segment[0].1) as f64)
        })
        .sum()
}

/// CARTOGRAPHER-PRIME: geometry-authority producer for line and curved
/// labels.
///
/// Lays the already-shaped `positioned_glyphs` stream (as emitted by
/// `text_shape().to_dict()["positioned_glyphs"]`) along a screen-space
/// `screen_path` polyline using the same arc-length placement math as
/// `compute_line_label_placement` / `layout_curved_text`, and returns a
/// `geometry_authority` payload in the exact shape `LabelPlan.compile`
/// validates: `{source, projection_authority, positioned_glyphs,
/// candidates}` where each candidate carries its own anchor, bounds, and
/// re-origined glyph stream. `arc_fractions` selects the candidate
/// positions along the path (defaults to center, quarter, three-quarter).
/// Returns `None` when the text cannot fit the path; raises `ValueError`
/// on malformed input. Pure function: no clock, no RNG, no global state.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "layout_label_candidates")]
#[pyo3(signature = (kind, label_id, text, screen_path, positioned_glyphs, font_size, arc_fractions = None, tracking = 0.0))]
pub(crate) fn layout_label_candidates_py(
    py: Python<'_>,
    kind: &str,
    label_id: &str,
    text: &str,
    screen_path: Vec<(f32, f32, f32)>,
    positioned_glyphs: Vec<Py<PyDict>>,
    font_size: f32,
    arc_fractions: Option<Vec<f32>>,
    tracking: f32,
) -> PyResult<Option<Py<PyDict>>> {
    if kind != "line" && kind != "curved" {
        return Err(PyValueError::new_err(format!(
            "layout_label_candidates kind must be 'line' or 'curved', got {kind:?}"
        )));
    }
    if !font_size.is_finite() || font_size <= 0.0 {
        return Err(PyValueError::new_err("font_size must be finite and > 0"));
    }
    if !tracking.is_finite() {
        return Err(PyValueError::new_err("tracking must be finite"));
    }
    for (x, y, z) in &screen_path {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(PyValueError::new_err(
                "screen_path coordinates must be finite",
            ));
        }
    }
    if screen_path.len() < 2 || text.is_empty() || positioned_glyphs.is_empty() {
        return Ok(None);
    }

    // Extract the required identity fields from the shaped glyph stream.
    struct GlyphIn {
        glyph_id: i64,
        font_index: i64,
        cluster: Option<i64>,
        line_index: Option<i64>,
        advance: Option<(f64, f64)>,
        has_outline: Option<bool>,
        unit_advance: f32,
    }
    let mut glyphs_in = Vec::with_capacity(positioned_glyphs.len());
    for item in &positioned_glyphs {
        let required_int = |key: &str| -> PyResult<i64> {
            let value = item.bind(py).get_item(key)?.ok_or_else(|| {
                PyValueError::new_err(format!("positioned_glyphs entries require integer {key}"))
            })?;
            value.extract::<i64>().map_err(|_| {
                PyValueError::new_err(format!(
                    "positioned_glyphs {key} must be a non-negative integer"
                ))
            })
        };
        let optional_int = |key: &str| -> PyResult<Option<i64>> {
            match item.bind(py).get_item(key)? {
                Some(value) => value.extract::<i64>().map(Some).map_err(|_| {
                    PyValueError::new_err(format!(
                        "positioned_glyphs {key} must be a non-negative integer"
                    ))
                }),
                None => Ok(None),
            }
        };
        let glyph_id = required_int("glyph_id")?;
        let font_index = required_int("font_index")?;
        let cluster = optional_int("cluster")?;
        let line_index = optional_int("line_index")?;
        if glyph_id < 0
            || font_index < 0
            || cluster.is_some_and(|v| v < 0)
            || line_index.is_some_and(|v| v < 0)
        {
            return Err(PyValueError::new_err(
                "positioned_glyphs ids must be non-negative",
            ));
        }
        let advance = match item.bind(py).get_item("advance")? {
            Some(value) => {
                let pair = value.extract::<Vec<f64>>().map_err(|_| {
                    PyValueError::new_err("positioned_glyphs advance must be a 2-sequence")
                })?;
                if pair.len() < 2 || !pair[0].is_finite() || !pair[1].is_finite() {
                    return Err(PyValueError::new_err(
                        "positioned_glyphs advance must be a finite 2-sequence",
                    ));
                }
                Some((pair[0], pair[1]))
            }
            None => None,
        };
        let has_outline = match item.bind(py).get_item("has_outline")? {
            Some(value) => Some(value.extract::<bool>().map_err(|_| {
                PyValueError::new_err("positioned_glyphs has_outline must be a bool")
            })?),
            None => None,
        };
        // The shaped stream is produced at size 1.0, so `advance` is the
        // unit-em advance; the device-pixel advance is `advance * font_size`.
        let unit_advance = match advance {
            Some((ax, _)) => ax as f32,
            None => 0.6,
        };
        if !unit_advance.is_finite() || unit_advance <= 0.0 {
            return Err(PyValueError::new_err(
                "positioned_glyphs advance must be positive",
            ));
        }
        glyphs_in.push(GlyphIn {
            glyph_id,
            font_index,
            cluster,
            line_index,
            advance,
            has_outline,
            unit_advance,
        });
    }

    let glyph_count = glyphs_in.len();
    let path_length = screen_path_length(&screen_path);
    let text_width = glyphs_in
        .iter()
        .map(|glyph| f64::from(glyph.unit_advance * font_size))
        .sum::<f64>()
        + f64::from(tracking * font_size) * glyph_count.saturating_sub(1) as f64;
    if !text_width.is_finite() || text_width <= 0.0 || path_length < text_width {
        return Ok(None);
    }

    let fractions = arc_fractions.unwrap_or_else(|| vec![0.5, 0.25, 0.75]);
    for fraction in &fractions {
        if !fraction.is_finite() || !(0.0..=1.0).contains(fraction) {
            return Err(PyValueError::new_err(
                "arc_fractions must be finite values within [0.0, 1.0]",
            ));
        }
    }

    // Curved authority works in the horizontal world plane; the screen
    // polyline maps x -> x and screen y -> z so the authority's
    // `atan2(tangent.x, tangent.z)` convention is preserved verbatim.
    let curved_path = if kind == "curved" {
        let vertices: Vec<glam::Vec3> = screen_path
            .iter()
            .map(|(x, y, _z)| glam::Vec3::new(*x, 0.0, *y))
            .collect();
        Some(crate::labels::curved::SampledPath::from_polyline(&vertices))
    } else {
        None
    };
    let points2d: Vec<[f32; 2]> = screen_path.iter().map(|(x, y, _z)| [*x, *y]).collect();
    let pixel_advances: Vec<f32> = glyphs_in
        .iter()
        .map(|glyph| glyph.unit_advance * font_size)
        .collect();
    let tracking_px = tracking * font_size;
    let chars: Vec<char> = text.chars().collect();

    let mut seen_offsets = std::collections::HashSet::new();
    let mut candidates_out: Vec<(f32, Vec<(f32, f32, f32)>)> = Vec::new();
    for fraction in &fractions {
        let max_start = (path_length - text_width).max(0.0) as f32;
        let start_offset =
            ((fraction * path_length as f32) - (text_width as f32) * 0.5).clamp(0.0, max_start);
        if !seen_offsets.insert(start_offset.to_bits()) {
            continue;
        }
        // Per-glyph `(x, y, rotation)` laid out by the authority math.
        let placed: Vec<(f32, f32, f32)> = if kind == "line" {
            // The line authority has no tracking parameter; fold it into the
            // inter-glyph spacing exactly as the curved authority does.
            let mut advances = pixel_advances.clone();
            for advance in advances.iter_mut().take(glyph_count.saturating_sub(1)) {
                *advance += tracking_px;
            }
            crate::labels::line_label::place_glyphs_along_path(
                &points2d,
                &advances,
                font_size,
                start_offset,
            )
            .iter()
            .map(|placement| {
                (
                    placement.screen_pos[0],
                    placement.screen_pos[1],
                    placement.rotation,
                )
            })
            .collect()
        } else {
            if chars.len() != glyph_count {
                // The curved authority indexes advances per char; refuse to
                // guess a mapping for ligatured/merged glyph streams.
                return Ok(None);
            }
            let path = curved_path.as_ref().expect("curved path built");
            let layout = crate::labels::curved::layout_curved_text_at_offset(
                &chars,
                path,
                &glyphs_in
                    .iter()
                    .map(|glyph| glyph.unit_advance)
                    .collect::<Vec<_>>(),
                font_size,
                [1.0, 1.0, 1.0, 1.0],
                tracking,
                start_offset,
                text_width as f32,
            );
            if !layout.success || layout.glyphs.len() != glyph_count {
                continue;
            }
            layout
                .glyphs
                .iter()
                .map(|glyph| (glyph.world_pos.x, glyph.world_pos.z, glyph.rotation))
                .collect()
        };
        if placed.len() != glyph_count {
            continue;
        }
        candidates_out.push((*fraction, placed));
    }
    if candidates_out.is_empty() {
        return Ok(None);
    }

    let emit_glyphs = |placed: &[(f32, f32, f32)]| -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty_bound(py);
        for (glyph, (x, y, rotation)) in glyphs_in.iter().zip(placed.iter()) {
            let item = PyDict::new_bound(py);
            item.set_item("glyph_id", glyph.glyph_id)?;
            item.set_item("font_index", glyph.font_index)?;
            if let Some(cluster) = glyph.cluster {
                item.set_item("cluster", cluster)?;
            }
            if let Some(line_index) = glyph.line_index {
                item.set_item("line_index", line_index)?;
            }
            item.set_item("origin", (*x, *y))?;
            if let Some((ax, ay)) = glyph.advance {
                item.set_item("advance", (ax, ay))?;
            }
            item.set_item("rotation", rotation)?;
            item.set_item("scale", font_size)?;
            if let Some(has_outline) = glyph.has_outline {
                item.set_item("has_outline", has_outline)?;
            }
            list.append(item)?;
        }
        Ok(list.unbind())
    };

    let pad = font_size * 0.5;
    let candidates = pyo3::types::PyList::empty_bound(py);
    let mut shared_glyphs: Option<Py<pyo3::types::PyList>> = None;
    for (index, (fraction, placed)) in candidates_out.iter().enumerate() {
        let glyphs = emit_glyphs(placed)?;
        if shared_glyphs.is_none() {
            shared_glyphs = Some(glyphs.clone_ref(py));
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for (x, y, _rotation) in placed {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }
        let anchor_distance = fraction * path_length as f32;
        let (anchor_x, anchor_y, anchor_z) = screen_path_sample(&screen_path, anchor_distance)
            .ok_or_else(|| PyValueError::new_err("screen_path produced no anchor sample"))?;
        let candidate = PyDict::new_bound(py);
        candidate.set_item("candidate_id", format!("{label_id}:authority-{index}"))?;
        candidate.set_item("candidate_type", "geometry_authority")?;
        candidate.set_item("anchor", (anchor_x, anchor_y, anchor_z))?;
        candidate.set_item(
            "bounds",
            (min_x - pad, min_y - pad, max_x + pad, max_y + pad),
        )?;
        let details = PyDict::new_bound(py);
        details.set_item("arc_fraction", fraction)?;
        candidate.set_item("details", details)?;
        candidate.set_item("positioned_glyphs", glyphs)?;
        candidates.append(candidate)?;
    }

    let payload = PyDict::new_bound(py);
    payload.set_item(
        "source",
        if kind == "curved" {
            "layout_curved_text"
        } else {
            "compute_line_label_placement"
        },
    )?;
    payload.set_item("projection_authority", "deterministic")?;
    payload.set_item(
        "positioned_glyphs",
        shared_glyphs.expect("at least one candidate emitted"),
    )?;
    payload.set_item("candidates", candidates)?;
    Ok(Some(payload.unbind()))
}

pub(crate) use crate::labels::py_text::{
    bake_msdf_atlas_py, bake_msdf_atlas_shaped_py, rasterize_shaped_run_py, text_shape_py,
};
