//! Bounded-optimal label placement via deterministic branch-and-bound.
//!
//! CARTOGRAPHER-PRIME: maximizes total placed priority weight subject to
//! (a) at most one placed candidate per label, (b) no two placed boxes
//! overlapping, and (c) `visible` candidates only. All arithmetic that
//! affects branching is integer (fixed-precision quantized), the search
//! order is a pure function of the candidate total order
//! `(label_id, candidate_index)`, and every decision is recorded as a
//! typed [`RationaleRecord`] so the emitted rationale is grounded in the
//! actual geometric conflicts the solver resolved — never a post-hoc
//! narrative.

use std::collections::BTreeMap;

use super::curved::CurvedTextLayout;
use super::declutter::{DeclutterConfig, DeclutterResult, PlacementCandidate};
use super::types::GlyphPlacement;

/// Fixed grid for box coordinates: 1/16 px, so floating-point drift cannot
/// change branch decisions across devices.
pub const COORD_SCALE: f64 = 16.0;
/// Fixed grid for priority weights: 1/1024 weight units.
pub const WEIGHT_SCALE: f64 = 1024.0;

/// Quantize a screen coordinate to the deterministic integer grid.
pub fn quantize_coord(value: f32) -> i64 {
    (value as f64 * COORD_SCALE).round() as i64
}

/// Quantize a priority weight to the deterministic integer grid.
pub fn quantize_weight(weight: f64) -> i64 {
    (weight * WEIGHT_SCALE).round() as i64
}

/// A quantized candidate as seen by the optimal solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverCandidate {
    /// Label identifier.
    pub label_id: u64,
    /// Position index within the label's candidate set (total order key).
    pub candidate_index: u32,
    /// Quantized bounds `[min_x, min_y, max_x, max_y]` on the 1/16 px grid.
    pub bounds_q: [i64; 4],
    /// Quantized priority weight (1/1024 units).
    pub weight_q: i64,
    /// Visibility gate; occluded anchors contribute zero placements.
    pub visible: bool,
}

impl SolverCandidate {
    /// Build a solver candidate from float bounds/weight, normalizing the
    /// box so `min <= max` on both axes.
    pub fn new(
        label_id: u64,
        candidate_index: u32,
        bounds: [f32; 4],
        weight: f64,
        visible: bool,
    ) -> Self {
        let x0 = quantize_coord(bounds[0].min(bounds[2]));
        let y0 = quantize_coord(bounds[1].min(bounds[3]));
        let x1 = quantize_coord(bounds[0].max(bounds[2]));
        let y1 = quantize_coord(bounds[1].max(bounds[3]));
        Self {
            label_id,
            candidate_index,
            bounds_q: [x0, y0, x1, y1],
            weight_q: quantize_weight(weight),
            visible,
        }
    }

    /// Adapt an existing [`PlacementCandidate`] (the geometry authorities'
    /// output) into the solver's quantized model.
    pub fn from_placement(candidate: &PlacementCandidate) -> Self {
        Self::new(
            candidate.label_id,
            candidate.anchor_index,
            candidate.bounds,
            candidate.priority as f64,
            candidate.visible,
        )
    }

    fn total_order_key(&self) -> (u64, u32) {
        (self.label_id, self.candidate_index)
    }
}

/// Inclusive AABB intersection on the quantized grid (touching counts, to
/// match the compile-time Python `_rects_intersect` semantics).
fn boxes_conflict_q(a: &[i64; 4], b: &[i64; 4], margin_q: i64) -> bool {
    a[0] - margin_q <= b[2]
        && a[2] + margin_q >= b[0]
        && a[1] - margin_q <= b[3]
        && a[3] + margin_q >= b[1]
}

/// Overlap area between two quantized boxes, in quantized-square units.
fn overlap_area_q(a: &[i64; 4], b: &[i64; 4]) -> i64 {
    let dx = (a[2].min(b[2]) - a[0].max(b[0])).max(0);
    let dy = (a[3].min(b[3]) - a[1].max(b[1])).max(0);
    dx * dy
}

/// A typed, reproducible record of one solver decision.
#[derive(Debug, Clone, PartialEq)]
pub enum RationaleRecord {
    /// A candidate was placed; `displaced` lists conflicting candidates of
    /// other labels that were not placed, as `(label_id, candidate_index,
    /// overlap_area_q)`.
    Placed {
        label_id: u64,
        candidate_index: u32,
        weight_q: i64,
        displaced: Vec<(u64, u32, i64)>,
    },
    /// A label with visible candidates was not placed; `blocking` lists the
    /// placed candidates conflicting with its best candidate.
    Dropped {
        label_id: u64,
        candidate_index: u32,
        weight_q: i64,
        priority_lost: bool,
        blocking: Vec<(u64, u32, i64)>,
    },
    /// A candidate was excluded because its anchor is occluded
    /// (silhouette/depth visibility gate).
    OccludedCandidate { label_id: u64, candidate_index: u32 },
    /// Solve summary: node count, certification, and achieved gap.
    Solver {
        nodes_explored: u64,
        certified: bool,
        gap: f64,
        gap_tolerance: f64,
    },
}

/// Result of a bounded-optimal solve.
#[derive(Debug, Clone)]
pub struct OptimalOutcome {
    /// Chosen `(label_id, candidate_index)` pairs, sorted by label id.
    pub placements: Vec<(u64, u32)>,
    /// Achieved objective (sum of effective quantized weights).
    pub objective_q: i64,
    /// Certified upper bound on the optimum objective.
    pub upper_bound_q: i64,
    /// Certified optimality gap: `(upper_bound - objective) / upper_bound`.
    pub gap: f64,
    /// True when the search space was exhausted within the node budget.
    /// False means the budget was hit and the gap is an honest bound, not
    /// an optimality claim.
    pub certified: bool,
    /// Branch-and-bound nodes explored.
    pub nodes_explored: u64,
    /// Grounded decision records, in deterministic order.
    pub rationale: Vec<RationaleRecord>,
}

/// Effective weight used in the objective: quantized weight plus one, so
/// every placement is strictly worth taking when it conflicts with nothing
/// (matching the greedy fallback's "place everything that fits" behavior)
/// while preserving strict order between weights ≥ 1/1024 apart.
fn effective_weight(weight_q: i64) -> i64 {
    weight_q.saturating_add(1)
}

struct LabelGroup {
    label_id: u64,
    candidates: Vec<SolverCandidate>,
    max_weight: i64,
}

/// Bounded-optimal branch-and-bound solve over the candidate set.
pub fn declutter_optimal(
    candidates: &[SolverCandidate],
    config: &DeclutterConfig,
) -> OptimalOutcome {
    // Deterministic total order over all candidates.
    let mut ordered: Vec<SolverCandidate> = candidates.to_vec();
    ordered.sort_by_key(SolverCandidate::total_order_key);

    let mut rationale: Vec<RationaleRecord> = Vec::new();
    for candidate in ordered.iter().filter(|candidate| !candidate.visible) {
        rationale.push(RationaleRecord::OccludedCandidate {
            label_id: candidate.label_id,
            candidate_index: candidate.candidate_index,
        });
    }

    // Group visible candidates per label (BTreeMap: deterministic order).
    let mut grouped: BTreeMap<u64, Vec<SolverCandidate>> = BTreeMap::new();
    for candidate in ordered.iter().filter(|candidate| candidate.visible) {
        grouped
            .entry(candidate.label_id)
            .or_default()
            .push(candidate.clone());
    }

    let mut groups: Vec<LabelGroup> = grouped
        .into_iter()
        .map(|(label_id, candidates)| {
            let max_weight = candidates
                .iter()
                .map(|candidate| effective_weight(candidate.weight_q).max(0))
                .max()
                .unwrap_or(0);
            LabelGroup {
                label_id,
                candidates,
                max_weight,
            }
        })
        .collect();
    // Branch order: strongest label first, then label id (deterministic).
    groups.sort_by(|a, b| {
        b.max_weight
            .cmp(&a.max_weight)
            .then(a.label_id.cmp(&b.label_id))
    });

    let n = groups.len();
    if n == 0 {
        rationale.push(RationaleRecord::Solver {
            nodes_explored: 0,
            certified: true,
            gap: 0.0,
            gap_tolerance: config.gap_tolerance,
        });
        return OptimalOutcome {
            placements: Vec::new(),
            objective_q: 0,
            upper_bound_q: 0,
            gap: 0.0,
            certified: true,
            nodes_explored: 0,
            rationale,
        };
    }

    // suffix_max[k] = best possible remaining contribution from labels k..n.
    let mut suffix_max = vec![0i64; n + 1];
    for k in (0..n).rev() {
        suffix_max[k] = suffix_max[k + 1] + groups[k].max_weight;
    }
    let root_bound = suffix_max[0];
    let tol_abs = ((config.gap_tolerance.max(0.0)) * root_bound as f64).floor() as i64;
    let margin_q = quantize_coord(config.margin.max(0.0));

    // Greedy incumbent: weight desc, then (label_id, candidate_index).
    let mut greedy_order: Vec<(usize, usize)> = Vec::new();
    for (group_pos, group) in groups.iter().enumerate() {
        for candidate_pos in 0..group.candidates.len() {
            greedy_order.push((group_pos, candidate_pos));
        }
    }
    greedy_order.sort_by(|&(ga, ca), &(gb, cb)| {
        let a = &groups[ga].candidates[ca];
        let b = &groups[gb].candidates[cb];
        effective_weight(b.weight_q)
            .cmp(&effective_weight(a.weight_q))
            .then(a.total_order_key().cmp(&b.total_order_key()))
    });
    let mut best_selection: Vec<Option<usize>> = vec![None; n];
    let mut best_objective: i64 = 0;
    {
        let mut placed_boxes: Vec<[i64; 4]> = Vec::new();
        for (group_pos, candidate_pos) in greedy_order {
            if best_selection[group_pos].is_some() {
                continue;
            }
            let candidate = &groups[group_pos].candidates[candidate_pos];
            if effective_weight(candidate.weight_q) <= 0 {
                continue;
            }
            if placed_boxes
                .iter()
                .any(|placed| boxes_conflict_q(&candidate.bounds_q, placed, margin_q))
            {
                continue;
            }
            best_selection[group_pos] = Some(candidate_pos);
            best_objective += effective_weight(candidate.weight_q);
            placed_boxes.push(candidate.bounds_q);
        }
    }

    // Depth-first branch-and-bound with an explicit stack (no recursion, no
    // RNG, no wall clock). Frame cursor c: 0..len = candidate index, len =
    // skip, len+1 = exhausted.
    struct Frame {
        cursor: usize,
        chosen: Option<usize>,
    }
    let mut stack: Vec<Frame> = vec![Frame {
        cursor: 0,
        chosen: None,
    }];
    let mut committed: i64 = 0;
    let mut current: Vec<Option<usize>> = vec![None; n];
    let mut nodes_explored: u64 = 0;
    let mut max_pruned_bound: i64 = 0;
    let mut budget_exceeded = false;

    while !stack.is_empty() {
        let depth = stack.len() - 1;
        // Undo the previous choice at this frame, if any.
        if let Some(prev) = stack[depth].chosen.take() {
            committed -= effective_weight(groups[depth].candidates[prev].weight_q);
            current[depth] = None;
        }
        let options = groups[depth].candidates.len();
        if stack[depth].cursor > options {
            stack.pop();
            continue;
        }
        if nodes_explored >= config.node_budget {
            budget_exceeded = true;
            break;
        }
        let cursor = stack[depth].cursor;
        stack[depth].cursor += 1;
        nodes_explored += 1;

        let (gain, feasible) = if cursor < options {
            let candidate = &groups[depth].candidates[cursor];
            let conflict = current[..depth]
                .iter()
                .enumerate()
                .any(|(prior_depth, chosen)| match chosen {
                    Some(pos) => boxes_conflict_q(
                        &candidate.bounds_q,
                        &groups[prior_depth].candidates[*pos].bounds_q,
                        margin_q,
                    ),
                    None => false,
                });
            (effective_weight(candidate.weight_q), !conflict)
        } else {
            (0, true) // skip
        };
        if !feasible {
            continue;
        }
        let bound = committed + gain + suffix_max[depth + 1];
        if bound <= best_objective + tol_abs {
            max_pruned_bound = max_pruned_bound.max(bound);
            continue;
        }
        // Commit this option.
        if cursor < options {
            committed += gain;
            current[depth] = Some(cursor);
            stack[depth].chosen = Some(cursor);
        }
        if depth + 1 == n {
            if committed > best_objective {
                best_objective = committed;
                best_selection.copy_from_slice(&current);
            }
        } else {
            stack.push(Frame {
                cursor: 0,
                chosen: None,
            });
        }
    }

    let certified = !budget_exceeded;
    let final_ub = if certified {
        best_objective.max(max_pruned_bound)
    } else {
        root_bound
    };
    let gap = if final_ub <= 0 {
        0.0
    } else {
        (final_ub - best_objective) as f64 / final_ub as f64
    };

    // Materialize placements sorted by label id.
    let mut placements: Vec<(u64, u32)> = groups
        .iter()
        .enumerate()
        .filter_map(|(group_pos, group)| {
            best_selection[group_pos]
                .map(|pos| (group.label_id, group.candidates[pos].candidate_index))
        })
        .collect();
    placements.sort_unstable();

    // Grounded rationale, derived solely from the final solution geometry.
    let placed_boxes: Vec<(u64, u32, [i64; 4], i64)> = groups
        .iter()
        .enumerate()
        .filter_map(|(group_pos, group)| {
            best_selection[group_pos].map(|pos| {
                let candidate = &group.candidates[pos];
                (
                    group.label_id,
                    candidate.candidate_index,
                    candidate.bounds_q,
                    candidate.weight_q,
                )
            })
        })
        .collect();
    let placed_keys: Vec<(u64, u32)> = placements.clone();
    let mut group_records: Vec<RationaleRecord> = Vec::new();
    let mut sorted_groups: Vec<&LabelGroup> = groups.iter().collect();
    sorted_groups.sort_by_key(|group| group.label_id);
    for group in sorted_groups {
        let placed = placed_keys
            .iter()
            .find(|(label_id, _)| *label_id == group.label_id);
        match placed {
            Some(&(_, candidate_index)) => {
                let chosen = group
                    .candidates
                    .iter()
                    .find(|candidate| candidate.candidate_index == candidate_index)
                    .expect("placed candidate exists");
                let mut displaced: Vec<(u64, u32, i64)> = Vec::new();
                for other in ordered.iter().filter(|other| {
                    other.visible
                        && other.label_id != group.label_id
                        && !placed_keys.contains(&(other.label_id, other.candidate_index))
                }) {
                    if boxes_conflict_q(&chosen.bounds_q, &other.bounds_q, margin_q) {
                        displaced.push((
                            other.label_id,
                            other.candidate_index,
                            overlap_area_q(&chosen.bounds_q, &other.bounds_q),
                        ));
                    }
                }
                displaced.sort_unstable();
                group_records.push(RationaleRecord::Placed {
                    label_id: group.label_id,
                    candidate_index,
                    weight_q: chosen.weight_q,
                    displaced,
                });
            }
            None => {
                // Best candidate = highest weight, then lowest index.
                let best = group
                    .candidates
                    .iter()
                    .max_by(|a, b| {
                        a.weight_q
                            .cmp(&b.weight_q)
                            .then(b.candidate_index.cmp(&a.candidate_index))
                    })
                    .expect("group is non-empty");
                let mut blocking: Vec<(u64, u32, i64)> = Vec::new();
                let mut priority_lost = false;
                for (label_id, candidate_index, bounds_q, weight_q) in &placed_boxes {
                    if boxes_conflict_q(&best.bounds_q, bounds_q, margin_q) {
                        blocking.push((
                            *label_id,
                            *candidate_index,
                            overlap_area_q(&best.bounds_q, bounds_q),
                        ));
                        if *weight_q > best.weight_q {
                            priority_lost = true;
                        }
                    }
                }
                blocking.sort_unstable();
                group_records.push(RationaleRecord::Dropped {
                    label_id: group.label_id,
                    candidate_index: best.candidate_index,
                    weight_q: best.weight_q,
                    priority_lost,
                    blocking,
                });
            }
        }
    }
    rationale.extend(group_records);
    rationale.push(RationaleRecord::Solver {
        nodes_explored,
        certified,
        gap,
        gap_tolerance: config.gap_tolerance,
    });

    OptimalOutcome {
        placements,
        objective_q: best_objective,
        upper_bound_q: final_ub,
        gap,
        certified,
        nodes_explored,
        rationale,
    }
}

/// Adapter for the [`super::declutter::DeclutterAlgorithm::Optimal`] arm:
/// runs the bounded-optimal solve over [`PlacementCandidate`]s and shapes
/// the answer as a [`DeclutterResult`].
pub fn declutter_optimal_result(
    candidates: Vec<PlacementCandidate>,
    config: &DeclutterConfig,
) -> DeclutterResult {
    let solver_candidates: Vec<SolverCandidate> = candidates
        .iter()
        .map(SolverCandidate::from_placement)
        .collect();
    let outcome = declutter_optimal(&solver_candidates, config);
    let mut visible_labels = Vec::with_capacity(outcome.placements.len());
    let mut positions = Vec::with_capacity(outcome.placements.len());
    for &(label_id, candidate_index) in &outcome.placements {
        visible_labels.push(label_id);
        if let Some(candidate) = candidates
            .iter()
            .find(|c| c.label_id == label_id && c.anchor_index == candidate_index)
        {
            positions.push((label_id, candidate.position));
        }
    }
    DeclutterResult {
        visible_labels,
        positions,
        total_energy: -(outcome.objective_q as f32) / WEIGHT_SCALE as f32,
        iterations: outcome.nodes_explored as usize,
    }
}

/// The 8-position cartographic ladder around an anchor, in preference
/// order: NE, NW, SE, SW, E, W, N, S (`anchor_index` 0..=7).
pub fn ladder_candidates(
    label_id: u64,
    anchor: [f32; 2],
    half_extent: [f32; 2],
    offset: f32,
    priority: i32,
) -> Vec<PlacementCandidate> {
    const DIRECTIONS: [[f32; 2]; 8] = [
        [1.0, -1.0],  // NE (screen y grows downward)
        [-1.0, -1.0], // NW
        [1.0, 1.0],   // SE
        [-1.0, 1.0],  // SW
        [1.0, 0.0],   // E
        [-1.0, 0.0],  // W
        [0.0, -1.0],  // N
        [0.0, 1.0],   // S
    ];
    DIRECTIONS
        .iter()
        .enumerate()
        .map(|(index, direction)| {
            let position = [
                anchor[0] + direction[0] * offset,
                anchor[1] + direction[1] * offset,
            ];
            PlacementCandidate {
                label_id,
                anchor_index: index as u32,
                position,
                bounds: [
                    position[0] - half_extent[0],
                    position[1] - half_extent[1],
                    position[0] + half_extent[0],
                    position[1] + half_extent[1],
                ],
                priority,
                cost: index as f32 * 0.001,
                selected: false,
                visible: true,
            }
        })
        .collect()
}

/// Candidate box from a curved-text layout produced by
/// [`super::curved::layout_curved_text`] — the geometry authority; glyphs
/// are never re-laid-out, only their bounding box is taken.
pub fn candidate_from_curved_layout(
    label_id: u64,
    anchor_index: u32,
    layout: &CurvedTextLayout,
    font_size: f32,
    priority: i32,
) -> Option<PlacementCandidate> {
    if !layout.success || layout.glyphs.is_empty() {
        return None;
    }
    let pad = font_size * 0.5;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for glyph in &layout.glyphs {
        min_x = min_x.min(glyph.world_pos.x);
        min_y = min_y.min(glyph.world_pos.y);
        max_x = max_x.max(glyph.world_pos.x);
        max_y = max_y.max(glyph.world_pos.y);
    }
    let position = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5];
    Some(PlacementCandidate {
        label_id,
        anchor_index,
        position,
        bounds: [min_x - pad, min_y - pad, max_x + pad, max_y + pad],
        priority,
        cost: 0.0,
        selected: false,
        visible: true,
    })
}

/// Candidate box from line-label glyph placements produced by
/// [`super::line_label::compute_line_label_placement`] — the geometry
/// authority; placements are consumed as-is.
pub fn candidate_from_line_placements(
    label_id: u64,
    anchor_index: u32,
    placements: &[GlyphPlacement],
    font_size: f32,
    priority: i32,
) -> Option<PlacementCandidate> {
    if placements.is_empty() {
        return None;
    }
    let pad = font_size * 0.5;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for placement in placements {
        min_x = min_x.min(placement.screen_pos[0]);
        min_y = min_y.min(placement.screen_pos[1]);
        max_x = max_x.max(placement.screen_pos[0]);
        max_y = max_y.max(placement.screen_pos[1]);
    }
    let position = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5];
    Some(PlacementCandidate {
        label_id,
        anchor_index,
        position,
        bounds: [min_x - pad, min_y - pad, max_x + pad, max_y + pad],
        priority,
        cost: 0.0,
        selected: false,
        visible: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(label: u64, index: u32, bounds: [f32; 4], weight: f64) -> SolverCandidate {
        SolverCandidate::new(label, index, bounds, weight, true)
    }

    fn config() -> DeclutterConfig {
        DeclutterConfig {
            margin: 0.0,
            ..DeclutterConfig::default()
        }
    }

    /// Exhaustive brute force over the per-label choice product using the
    /// same effective-weight objective as the solver.
    fn brute_force_optimum(candidates: &[SolverCandidate], margin_q: i64) -> i64 {
        let mut grouped: BTreeMap<u64, Vec<&SolverCandidate>> = BTreeMap::new();
        for candidate in candidates.iter().filter(|candidate| candidate.visible) {
            grouped
                .entry(candidate.label_id)
                .or_default()
                .push(candidate);
        }
        let groups: Vec<Vec<&SolverCandidate>> = grouped.into_values().collect();
        let mut best = 0i64;
        let mut choice = vec![0usize; groups.len()];
        loop {
            // Evaluate: index == len(group) means skip.
            let mut objective = 0i64;
            let mut boxes: Vec<[i64; 4]> = Vec::new();
            let mut feasible = true;
            for (group, &pick) in groups.iter().zip(choice.iter()) {
                if pick == group.len() {
                    continue;
                }
                let candidate = group[pick];
                if boxes
                    .iter()
                    .any(|other| boxes_conflict_q(&candidate.bounds_q, other, margin_q))
                {
                    feasible = false;
                    break;
                }
                boxes.push(candidate.bounds_q);
                objective += effective_weight(candidate.weight_q);
            }
            if feasible {
                best = best.max(objective);
            }
            // Advance mixed-radix counter.
            let mut done = true;
            for (slot, group) in choice.iter_mut().zip(groups.iter()) {
                if *slot < group.len() {
                    *slot += 1;
                    done = false;
                    break;
                }
                *slot = 0;
            }
            if done {
                break;
            }
        }
        best
    }

    fn assert_within_gap(candidates: &[SolverCandidate]) {
        let cfg = config();
        let outcome = declutter_optimal(candidates, &cfg);
        let optimum = brute_force_optimum(candidates, 0);
        assert!(
            outcome.objective_q as f64 >= 0.98 * optimum as f64,
            "objective {} below 98% of brute-force optimum {}",
            outcome.objective_q,
            optimum
        );
        assert!(
            outcome.gap <= cfg.gap_tolerance + 1e-12,
            "reported gap {} exceeds tolerance",
            outcome.gap
        );
        assert!(outcome.certified, "small instance must certify");
    }

    #[test]
    fn test_optimal_beats_greedy_on_chain() {
        // Chain A-B-C: B overlaps both, w(B)=10 > w(A)=6, w(C)=6.
        // Greedy places B (10); optimal places A+C (12).
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 6.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 10.0),
            cand(3, 0, [12.0, 0.0, 22.0, 10.0], 6.0),
        ];
        let outcome = declutter_optimal(&candidates, &config());
        let placed: Vec<u64> = outcome.placements.iter().map(|(id, _)| *id).collect();
        assert_eq!(placed, vec![1, 3]);
        assert_within_gap(&candidates);
    }

    #[test]
    fn test_optimal_matches_bruteforce_on_hand_instances() {
        let instances: Vec<Vec<SolverCandidate>> = vec![
            // Non-overlapping: place everything.
            vec![
                cand(1, 0, [0.0, 0.0, 5.0, 5.0], 1.0),
                cand(2, 0, [10.0, 0.0, 15.0, 5.0], 2.0),
                cand(3, 0, [20.0, 0.0, 25.0, 5.0], 3.0),
            ],
            // Same-position pair: keep the heavier one.
            vec![
                cand(1, 0, [50.0, 50.0, 50.0, 50.0], 5.0),
                cand(2, 0, [50.0, 50.0, 50.0, 50.0], 9.0),
            ],
            // Two candidates per label with cross conflicts.
            vec![
                cand(1, 0, [0.0, 0.0, 10.0, 10.0], 8.0),
                cand(1, 1, [20.0, 0.0, 30.0, 10.0], 7.0),
                cand(2, 0, [5.0, 5.0, 15.0, 15.0], 6.0),
                cand(2, 1, [40.0, 0.0, 50.0, 10.0], 5.0),
                cand(3, 0, [25.0, 5.0, 35.0, 15.0], 4.0),
            ],
            // Star: center conflicts with four satellites.
            vec![
                cand(1, 0, [10.0, 10.0, 30.0, 30.0], 11.0),
                cand(2, 0, [5.0, 5.0, 15.0, 15.0], 4.0),
                cand(3, 0, [25.0, 5.0, 35.0, 15.0], 4.0),
                cand(4, 0, [5.0, 25.0, 15.0, 35.0], 4.0),
                cand(5, 0, [25.0, 25.0, 35.0, 35.0], 4.0),
            ],
            // Zero-weight labels still get placed when they fit.
            vec![
                cand(1, 0, [0.0, 0.0, 5.0, 5.0], 0.0),
                cand(2, 0, [10.0, 0.0, 15.0, 5.0], 0.0),
            ],
            // Fractional weights on the quantized grid.
            vec![
                cand(1, 0, [0.0, 0.0, 8.0, 8.0], 1.5),
                cand(2, 0, [4.0, 4.0, 12.0, 12.0], 1.25),
                cand(3, 0, [9.0, 0.0, 17.0, 7.0], 1.75),
            ],
        ];
        for candidates in &instances {
            assert_within_gap(candidates);
        }
    }

    #[test]
    fn test_deterministic_across_repeated_calls() {
        let candidates = vec![
            cand(3, 1, [0.0, 0.0, 10.0, 10.0], 5.0),
            cand(1, 0, [5.0, 5.0, 15.0, 15.0], 5.0),
            cand(2, 0, [8.0, 0.0, 18.0, 10.0], 5.0),
            cand(3, 0, [2.0, 2.0, 12.0, 12.0], 5.0),
        ];
        let cfg = config();
        let first = declutter_optimal(&candidates, &cfg);
        for _ in 0..5 {
            let again = declutter_optimal(&candidates, &cfg);
            assert_eq!(again.placements, first.placements);
            assert_eq!(again.objective_q, first.objective_q);
            assert_eq!(again.rationale, first.rationale);
            assert_eq!(again.nodes_explored, first.nodes_explored);
        }
    }

    #[test]
    fn test_occluded_candidates_never_chosen() {
        let candidates = vec![
            SolverCandidate::new(1, 0, [0.0, 0.0, 10.0, 10.0], 100.0, false),
            SolverCandidate::new(1, 1, [20.0, 0.0, 30.0, 10.0], 1.0, true),
            SolverCandidate::new(2, 0, [50.0, 0.0, 60.0, 10.0], 50.0, false),
        ];
        let outcome = declutter_optimal(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 1)]);
        let occluded: Vec<(u64, u32)> = outcome
            .rationale
            .iter()
            .filter_map(|record| match record {
                RationaleRecord::OccludedCandidate {
                    label_id,
                    candidate_index,
                } => Some((*label_id, *candidate_index)),
                _ => None,
            })
            .collect();
        assert_eq!(occluded, vec![(1, 0), (2, 0)]);
    }

    #[test]
    fn test_budget_exceeded_returns_honest_gap() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 6.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 10.0),
            cand(3, 0, [12.0, 0.0, 22.0, 10.0], 6.0),
        ];
        let cfg = DeclutterConfig {
            margin: 0.0,
            node_budget: 1,
            ..DeclutterConfig::default()
        };
        let outcome = declutter_optimal(&candidates, &cfg);
        assert!(!outcome.certified, "budget-hit solve must not certify");
        assert!(outcome.gap > 0.0, "budget-hit gap must be honest (> 0)");
        // Incumbent is at least the greedy solution.
        assert!(!outcome.placements.is_empty());
    }

    #[test]
    fn test_rationale_records_ground_the_solution() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 9.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 2.0),
        ];
        let outcome = declutter_optimal(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 0)]);
        let placed = outcome.rationale.iter().find_map(|record| match record {
            RationaleRecord::Placed {
                label_id,
                displaced,
                ..
            } if *label_id == 1 => Some(displaced.clone()),
            _ => None,
        });
        // Placed label 1 displaced label 2's candidate; overlap 5x10 px =
        // (5*16)*(10*16) quantized units.
        assert_eq!(placed, Some(vec![(2u64, 0u32, 80i64 * 160i64)]));
        let dropped = outcome.rationale.iter().find_map(|record| match record {
            RationaleRecord::Dropped {
                label_id,
                priority_lost,
                blocking,
                ..
            } if *label_id == 2 => Some((*priority_lost, blocking.clone())),
            _ => None,
        });
        let (priority_lost, blocking) = dropped.expect("label 2 dropped");
        assert!(priority_lost);
        assert_eq!(blocking, vec![(1u64, 0u32, 80i64 * 160i64)]);
    }

    #[test]
    fn test_ladder_candidates_shape() {
        let ladder = ladder_candidates(7, [100.0, 100.0], [20.0, 8.0], 12.0, 3);
        assert_eq!(ladder.len(), 8);
        for (index, candidate) in ladder.iter().enumerate() {
            assert_eq!(candidate.label_id, 7);
            assert_eq!(candidate.anchor_index, index as u32);
            assert!(candidate.visible);
            assert!((candidate.bounds[2] - candidate.bounds[0] - 40.0).abs() < 1e-5);
            assert!((candidate.bounds[3] - candidate.bounds[1] - 16.0).abs() < 1e-5);
        }
        // NE candidate sits up-right of the anchor.
        assert!(ladder[0].position[0] > 100.0 && ladder[0].position[1] < 100.0);
    }

    #[test]
    fn test_line_placement_candidate_bbox() {
        let placements = vec![
            GlyphPlacement {
                screen_pos: [10.0, 20.0],
                rotation: 0.0,
                scale: 1.0,
            },
            GlyphPlacement {
                screen_pos: [40.0, 26.0],
                rotation: 0.1,
                scale: 1.0,
            },
        ];
        let candidate =
            candidate_from_line_placements(9, 0, &placements, 12.0, 5).expect("candidate");
        assert_eq!(candidate.bounds, [4.0, 14.0, 46.0, 32.0]);
        assert_eq!(candidate.position, [25.0, 23.0]);
        assert!(candidate_from_line_placements(9, 0, &[], 12.0, 5).is_none());
    }
}
