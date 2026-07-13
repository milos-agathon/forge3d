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

pub(crate) use crate::labels::py_text::{
    bake_msdf_atlas_py, rasterize_shaped_run_py, text_shape_py,
};
