# tests/test_label_optimal_solver.py
# CARTOGRAPHER-PRIME: bounded-optimal, silhouette-aware, explainable label solver.
# Gated DoD: optimality gap <= 2% vs brute force, zero occluded-anchor
# placements, and byte-identical plan hashes against the committed
# cross-device golden triplet.

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

_native = pytest.importorskip("forge3d._forge3d")

pytestmark = pytest.mark.skipif(
    not hasattr(_native, "declutter_optimal"),
    reason="native declutter_optimal not available in this build",
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "labels" / "optimal_plan_hash.json"
GAP_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Brute-force reference on the solver's exact quantized model
# ---------------------------------------------------------------------------

def _quantize_coord(value: float) -> int:
    return round(value * 16.0)


def _effective_weight(weight: float) -> int:
    return round(weight * 1024.0) + 1


def _quantized_box(bounds) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bounds
    return (
        _quantize_coord(min(x0, x1)),
        _quantize_coord(min(y0, y1)),
        _quantize_coord(max(x0, x1)),
        _quantize_coord(max(y0, y1)),
    )


def _conflict(a, b) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def brute_force_optimum(candidates) -> int:
    """Exhaustive per-label choice product on the quantized objective."""
    groups: dict[int, list[tuple[tuple[int, int, int, int], int]]] = {}
    for label_id, _index, bounds, weight, visible in candidates:
        if not visible:
            continue
        groups.setdefault(label_id, []).append(
            (_quantized_box(bounds), _effective_weight(weight))
        )
    keys = sorted(groups)
    best = 0
    for choice in itertools.product(*[range(len(groups[key]) + 1) for key in keys]):
        total = 0
        boxes = []
        feasible = True
        for key, pick in zip(keys, choice):
            if pick == len(groups[key]):
                continue
            box, weight = groups[key][pick]
            if any(_conflict(box, other) for other in boxes):
                feasible = False
                break
            boxes.append(box)
            total += weight
        if feasible:
            best = max(best, total)
    return best


def solver_objective(candidates, placements) -> int:
    weights = {
        (label_id, index): weight
        for label_id, index, _bounds, weight, _visible in candidates
    }
    return sum(_effective_weight(weights[(label_id, index)]) for label_id, index in placements)


# Hand-built instances with known/brute-forceable optima. The chain instance
# is greedy-suboptimal by construction: greedy keeps B (10), the optimum is
# A+C (12).
CHAIN_INSTANCE = [
    (1, 0, (0.0, 0.0, 10.0, 10.0), 6.0, True),
    (2, 0, (5.0, 0.0, 15.0, 10.0), 10.0, True),
    (3, 0, (12.0, 0.0, 22.0, 10.0), 6.0, True),
]

SOLVER_INSTANCES = [
    CHAIN_INSTANCE,
    # Same-position pair: keep the heavier label.
    [
        (1, 0, (50.0, 50.0, 50.0, 50.0), 5.0, True),
        (2, 0, (50.0, 50.0, 50.0, 50.0), 9.0, True),
    ],
    # Multi-candidate labels with cross conflicts.
    [
        (1, 0, (0.0, 0.0, 10.0, 10.0), 8.0, True),
        (1, 1, (20.0, 0.0, 30.0, 10.0), 7.0, True),
        (2, 0, (5.0, 5.0, 15.0, 15.0), 6.0, True),
        (2, 1, (40.0, 0.0, 50.0, 10.0), 5.0, True),
        (3, 0, (25.0, 5.0, 35.0, 15.0), 4.0, True),
    ],
    # Star: heavy center conflicts with four satellites worth more together.
    [
        (1, 0, (10.0, 10.0, 30.0, 30.0), 11.0, True),
        (2, 0, (5.0, 5.0, 15.0, 15.0), 4.0, True),
        (3, 0, (25.0, 5.0, 35.0, 15.0), 4.0, True),
        (4, 0, (5.0, 25.0, 15.0, 35.0), 4.0, True),
        (5, 0, (25.0, 25.0, 35.0, 35.0), 4.0, True),
    ],
    # Occlusion-gated: the highest-weight candidate is behind a silhouette.
    [
        (1, 0, (0.0, 0.0, 10.0, 10.0), 100.0, False),
        (1, 1, (20.0, 0.0, 30.0, 10.0), 1.0, True),
        (2, 0, (5.0, 0.0, 15.0, 10.0), 3.0, True),
    ],
]


# ---------------------------------------------------------------------------
# Synthetic ridgeline occlusion fixture (compile-time depth visibility)
# ---------------------------------------------------------------------------

class RidgelineDepthSampler:
    """Synthetic terrain silhouette depth buffer over a 100x100 viewport.

    Rows with y < 45 sit behind a near ridge (scene depth 6); rows with
    y >= 45 see far terrain (scene depth 10). An anchor with projected depth
    8 is therefore occluded above the ridgeline and visible below it.
    """

    NEAR_RIDGE_DEPTH = 6.0
    FAR_TERRAIN_DEPTH = 10.0

    def scene_depth(self, y: float) -> float:
        return self.NEAR_RIDGE_DEPTH if y < 45.0 else self.FAR_TERRAIN_DEPTH

    def sample_label(self, coords, *, record=None, label_id=None):
        del label_id
        y = float(coords[1])
        label_depth = float(coords[2]) if len(coords) > 2 else 0.0
        scene_depth = self.scene_depth(y)
        return {
            "scene_depth": scene_depth,
            "label_depth": label_depth,
            "visible": bool(label_depth <= scene_depth),
            "occlusion": "depth_silhouette",
            "source": "synthetic_ridgeline",
        }


def _ridgeline_labels():
    def point(label_id, x, y, z, priority):
        return {
            "id": label_id,
            "text": label_id.upper(),
            "geometry": {"type": "Point", "coordinates": [x, y, z]},
            "priority": priority,
            "requires_terrain": True,
            "terrain_mode": "terrain",
            "candidate_policy": {"radial_count": 0},
        }

    return [
        # Behind the ridgeline (y < 45 -> scene depth 6 < anchor depth 8).
        point("behind-ridge-a", 20.0, 30.0, 8.0, 9),
        point("behind-ridge-b", 70.0, 20.0, 8.0, 7),
        # Visible terrain (y >= 45 -> scene depth 10 >= anchor depth 8).
        point("front-a", 20.0, 60.0, 8.0, 5),
        point("front-b", 70.0, 80.0, 8.0, 3),
        # Visible anchor whose 'above' ladder candidate (y - 12 = 38) falls
        # behind the ridgeline: the candidate must be gated visible=False.
        point("edge-of-ridge", 50.0, 50.0, 8.0, 1),
    ]


def _compile_ridgeline_plan():
    from forge3d.label_plan import LabelPlan

    return LabelPlan.compile(
        labels=_ridgeline_labels(),
        camera={"name": "fixed"},
        viewport={"width": 100, "height": 100},
        terrain=RidgelineDepthSampler(),
        seed=7,
    )


# ---------------------------------------------------------------------------
# Solver-surface tests (native declutter_optimal)
# ---------------------------------------------------------------------------

class TestOptimalSolverGap:
    @pytest.mark.parametrize("index", range(len(SOLVER_INSTANCES)))
    def test_gap_within_2pct_of_bruteforce(self, index: int):
        candidates = SOLVER_INSTANCES[index]
        placements, gap, _rationale = _native.declutter_optimal(candidates)
        optimum = brute_force_optimum(candidates)
        achieved = solver_objective(candidates, placements)
        assert achieved >= 0.98 * optimum, (
            f"instance {index}: achieved {achieved} < 98% of optimum {optimum}"
        )
        assert gap <= GAP_TOLERANCE, f"instance {index}: certified gap {gap} > 2%"

    def test_chain_beats_greedy(self):
        placements, gap, _rationale = _native.declutter_optimal(CHAIN_INSTANCE)
        assert sorted(label for label, _ in placements) == [1, 3]
        assert gap == 0.0

    def test_occluded_candidates_never_placed(self):
        candidates = SOLVER_INSTANCES[-1]
        placements, _gap, rationale = _native.declutter_optimal(candidates)
        hidden = {
            (label, index)
            for label, index, _bounds, _weight, visible in candidates
            if not visible
        }
        assert not hidden.intersection(set(placements))
        occluded_records = [
            record
            for record in rationale.records()
            if record["kind"] == "occluded_candidate"
        ]
        assert {(r["label_id"], r["candidate_index"]) for r in occluded_records} == hidden

    def test_budget_exceeded_returns_honest_gap(self):
        placements, gap, rationale = _native.declutter_optimal(
            CHAIN_INSTANCE, node_budget=1
        )
        assert gap > 0.0, "budget-exhausted solve must report an honest non-zero gap"
        solver_records = [
            record for record in rationale.records() if record["kind"] == "solver"
        ]
        assert len(solver_records) == 1
        assert solver_records[0]["certified"] is False
        assert placements, "incumbent (greedy) solution must still be returned"

    def test_deterministic_across_repeated_calls(self):
        first = _native.declutter_optimal(SOLVER_INSTANCES[2])
        for _ in range(5):
            again = _native.declutter_optimal(SOLVER_INSTANCES[2])
            assert again[0] == first[0]
            assert again[1] == first[1]
            assert again[2].records() == first[2].records()
            assert again[2].render() == first[2].render()


# ---------------------------------------------------------------------------
# Compile-time occlusion, determinism, and rationale tests
# ---------------------------------------------------------------------------

class TestCompileTimeOcclusion:
    def test_zero_occluded_anchor_placements(self):
        plan = _compile_ridgeline_plan()
        accepted_ids = {label.label_id for label in plan.accepted}
        assert accepted_ids == {"front-a", "front-b", "edge-of-ridge"}
        for label in plan.accepted:
            assert label.candidate.terrain_sample.get("visible") is not False
        rejected = {label.label_id: label.reason for label in plan.rejected}
        assert rejected == {
            "behind-ridge-a": "terrain_occluded",
            "behind-ridge-b": "terrain_occluded",
        }

    def test_occluded_ladder_candidate_is_gated(self):
        plan = _compile_ridgeline_plan()
        edge = next(label for label in plan.accepted if label.label_id == "edge-of-ridge")
        above = next(
            candidate
            for candidate in edge.candidates
            if candidate.candidate_id == "edge-of-ridge:above"
        )
        assert above.details.get("visible") is False, (
            "'above' anchor (y=38) sits behind the ridgeline and must be gated"
        )
        # The primary (center) anchor stays ungated.
        center = next(
            candidate
            for candidate in edge.candidates
            if candidate.candidate_id == "edge-of-ridge:center"
        )
        assert center.details.get("visible") is not False

    def test_rationale_cites_occluded_anchors_with_depths(self):
        plan = _compile_ridgeline_plan()
        occluded = [
            record for record in plan.rationale if record["kind"] == "occluded_anchor"
        ]
        cited = {record["label_id"] for record in occluded}
        assert {"behind-ridge-a", "behind-ridge-b", "edge-of-ridge"} <= cited
        for record in occluded:
            sample = record["terrain_sample"]
            assert sample["label_depth"] > sample["scene_depth"], (
                "every occlusion record must cite the ridgeline depth vs anchor depth"
            )
        rendered = "\n".join(plan.render_rationale())
        assert "terrain depth 6.000 nearer than anchor depth 8.000" in rendered

    def test_compile_is_byte_identical(self):
        first = _compile_ridgeline_plan()
        second = _compile_ridgeline_plan()
        first_bytes = json.dumps(first.to_dict(), sort_keys=True).encode("utf-8")
        second_bytes = json.dumps(second.to_dict(), sort_keys=True).encode("utf-8")
        assert first_bytes == second_bytes
        from forge3d._map_scene_common import _stable_hash

        assert _stable_hash(first.to_dict()) == _stable_hash(second.to_dict())


class TestRationaleGrounding:
    def _colliding_plan(self):
        from forge3d.label_plan import LabelPlan

        labels = [
            {
                "id": "winner",
                "text": "Winner",
                "geometry": {"type": "Point", "coordinates": [40.0, 40.0]},
                "priority": 9,
                "candidate_policy": {"radial_count": 0},
            },
            {
                "id": "loser",
                "text": "Loser",
                "geometry": {"type": "Point", "coordinates": [40.0, 40.0]},
                "priority": 2,
                "candidate_policy": {"radial_count": 0},
            },
        ]
        return LabelPlan.compile(labels=labels, camera={}, viewport=(100, 100))

    def test_rationale_cites_displaced_label_ids(self):
        plan = self._colliding_plan()
        placed = next(
            record
            for record in plan.rationale
            if record["kind"] == "placed" and record["label_id"] == "winner"
        )
        assert [entry["label_id"] for entry in placed["displaced"]] == ["loser"]
        dropped = next(
            record
            for record in plan.rationale
            if record["kind"] == "dropped" and record["label_id"] == "loser"
        )
        assert dropped["priority_lost"] is True
        assert [entry["label_id"] for entry in dropped["blocking"]] == ["winner"]
        solver = next(
            record for record in plan.rationale if record["kind"] == "solver"
        )
        assert solver["algorithm"] == "optimal"
        assert solver["certified"] is True
        assert solver["gap"] <= GAP_TOLERANCE

    def test_render_rationale_derives_from_records_alone(self):
        plan = self._colliding_plan()
        from forge3d.label_plan import LabelPlan

        rehydrated = LabelPlan.from_dict(
            json.loads(json.dumps(plan.to_dict()))
        )
        assert rehydrated.render_rationale() == plan.render_rationale()
        rendered = "\n".join(plan.render_rationale())
        assert "placed 'winner'" in rendered
        assert "dropped 'loser' (priority_lost)" in rendered


# ---------------------------------------------------------------------------
# THE gated DoD test: all three measurable-win conditions in one test.
# ---------------------------------------------------------------------------

class TestCartographerPrimeDoDGate:
    def test_dod_gate(self):
        # (1) Optimality gap <= 2% versus the brute-force optimum on every
        # small instance solvable to optimality.
        for candidates in SOLVER_INSTANCES:
            placements, gap, _rationale = _native.declutter_optimal(candidates)
            optimum = brute_force_optimum(candidates)
            achieved = solver_objective(candidates, placements)
            assert achieved >= (1.0 - GAP_TOLERANCE) * optimum
            assert gap <= GAP_TOLERANCE

        # (2) Zero labels placed on occluded anchors.
        plan = _compile_ridgeline_plan()
        occluded_placements = [
            label
            for label in plan.accepted
            if label.candidate.terrain_sample.get("visible") is False
            or (label.candidate.details or {}).get("visible") is False
        ]
        assert occluded_placements == [], (
            f"labels placed on occluded anchors: {occluded_placements}"
        )

        # (3) Byte-identical plan hashes across the 3-runner-config matrix:
        # the committed golden triplet pins one hash per CI OS runner and the
        # recomputed hash must equal all three.
        from forge3d._map_scene_common import _stable_hash

        plan_hash = _stable_hash(plan.to_dict())
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        triplet = golden["plan_hash"]
        assert set(triplet) == {"windows-latest", "ubuntu-latest", "macos-latest"}
        assert len(set(triplet.values())) == 1, "golden triplet must agree"
        assert plan_hash == triplet["windows-latest"], (
            f"compiled plan hash {plan_hash} diverges from the committed "
            f"cross-device golden {triplet['windows-latest']}"
        )
