# Cigar Smoke Refinement Pass 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce late-frame blanket effect in smoke plume while preserving atmospheric haze presence.

**Architecture:** Four targeted parameter adjustments in `_update_residual_haze()` and `_inject_hybrid_sources()`. All changes are numeric constants - no logic changes.

**Tech Stack:** Python, NumPy, PIL

**Spec:** `docs/superpowers/specs/2026-06-11-cigar-smoke-refinement-pass2-design.md`

---

## File Structure

All changes are in a single file:

| File | Purpose |
|------|---------|
| `examples/california_cigar_smoke_demo.py` | All parameter tuning (modify) |
| `tests/test_california_cigar_smoke_hybrid.py` | Regression tests (run only, no changes) |

---

## Task 1: Haze Feed Reduction

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2152`

- [ ] **Step 1: Run baseline tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer -v
```

Expected: PASS

- [ ] **Step 2: Update haze_feed coefficients**

Find line 2152 in `_update_residual_haze()`:

```python
        haze_feed = np.clip(old_smoke * 0.015 + high_slab * 0.0058 + self.density * 0.0032, 0.0, 1.0)
```

Replace with:

```python
        haze_feed = np.clip(old_smoke * 0.0125 + high_slab * 0.0050 + self.density * 0.0028, 0.0, 1.0)
```

- [ ] **Step 3: Run regression test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
refine: reduce haze accumulation coefficients

- old_smoke: 0.015 -> 0.0125 (-17%)
- high_slab: 0.0058 -> 0.0050 (-14%)
- density: 0.0032 -> 0.0028 (-12%)

Pass 2 refinement to reduce late-frame blanket effect.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Texture Modulation Widening

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2159`

- [ ] **Step 1: Update texture modulation parameters**

Find line 2159 in `_update_residual_haze()`:

```python
        injected = (broad_feed + regional_feed) * np.clip(0.66 + 0.34 * texture, 0.42, 1.06)
```

Replace with:

```python
        injected = (broad_feed + regional_feed) * np.clip(0.62 + 0.46 * texture, 0.34, 1.12)
```

- [ ] **Step 2: Run regression test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps -v
```

Expected: PASS (2/2)

- [ ] **Step 3: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
refine: widen texture modulation for internal contrast

- Base: 0.66 -> 0.62
- Multiplier: 0.34 -> 0.46
- Bounds: [0.42, 1.06] -> [0.34, 1.12]

Pass 2 refinement for more tonal variation in haze.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Haze Decay and Blur Reduction

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2160,2162`

- [ ] **Step 1: Update decay multiplier**

Find line 2160 in `_update_residual_haze()`:

```python
        residual = advected * 0.993 + injected
```

Replace with:

```python
        residual = advected * 0.990 + injected
```

- [ ] **Step 2: Update blur radius**

Find line 2162 in `_update_residual_haze()`:

```python
        residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.65)
```

Replace with:

```python
        residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.40)
```

- [ ] **Step 3: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_rgba_is_filamentary_and_translucent -v
```

Expected: PASS (2/2)

- [ ] **Step 4: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
refine: faster haze decay and sharper structure

- Decay: 0.993 -> 0.990 (faster thinning)
- Blur: 1.65 -> 1.40 (sharper edges)

Pass 2 refinement to preserve ribbon structure in haze.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Near-Source Curl Amplitude

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:1912`

- [ ] **Step 1: Update curl_offset first term**

Find line 1912 in `_inject_hybrid_sources()`:

```python
        curl_offset = radius * (
            2.8 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
```

Replace `2.8` with `3.1`:

```python
        curl_offset = radius * (
            3.1 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
```

- [ ] **Step 2: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_density_advects_from_single_event_and_tracks_age tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps -v
```

Expected: PASS (2/2)

- [ ] **Step 3: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
refine: increase near-source curl amplitude

- First curl term: 2.8 -> 3.1 (+11%)

Pass 2 refinement for lateral streamer variation near fire.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final Validation

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py -v
```

Expected: All 26 tests PASS

- [ ] **Step 2: Verify commit history**

```bash
git log --oneline -5
```

Expected: 4 refine commits from this pass

---

## Rollback Procedures

If any task causes test failures:

1. Revert that specific parameter to pass 1 value
2. Re-run tests to confirm
3. Document for pass 3 consideration

```bash
# To revert a specific commit
git revert <commit-hash> --no-commit
git commit -m "revert: <task> caused regression"
```
