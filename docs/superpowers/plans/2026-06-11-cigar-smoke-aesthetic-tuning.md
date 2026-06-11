# Cigar Smoke Aesthetic Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune parameters in `examples/california_cigar_smoke_demo.py` to match high-altitude atmospheric smoke aesthetic from reference footage.

**Architecture:** Parameter-only changes across 5 logical groups: color mapping, edge softness, streamer shape, lifecycle behavior, and fire bloom. Each section is validated independently before proceeding.

**Tech Stack:** Python, NumPy, PIL - no new dependencies

**Spec:** `docs/superpowers/specs/2026-06-11-cigar-smoke-aesthetic-tuning-design.md`

---

## File Structure

All changes are in a single file:

| File | Purpose |
|------|---------|
| `examples/california_cigar_smoke_demo.py` | All parameter tuning (modify) |
| `tests/test_california_cigar_smoke_hybrid.py` | Regression tests (run only, no changes) |

---

## Task 0: Commit Baseline Files

The example and test files are currently untracked. Commit them first so tuning commits are clean diffs.

**Files:**
- Add: `examples/california_cigar_smoke_demo.py`
- Add: `tests/test_california_cigar_smoke_hybrid.py`

- [ ] **Step 1: Verify files exist and tests pass**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py -v --tb=short 2>&1 | head -50
```

Expected: Tests pass (or skip gracefully if dependencies missing)

- [ ] **Step 2: Commit baseline files**

```bash
git add examples/california_cigar_smoke_demo.py tests/test_california_cigar_smoke_hybrid.py
git commit -m "$(cat <<'EOF'
Add California cigar smoke demo and tests

Baseline commit before aesthetic parameter tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Smoke Color/Density Mapping

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2277-2289`

- [ ] **Step 1: Run baseline tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_rgba_is_filamentary_and_translucent -v
```

Expected: PASS (establishes baseline)

- [ ] **Step 2: Update color values in `_hybrid_smoke_field_rgba()`**

Find and replace the color definitions (around line 2278):

```python
    density_t = _smoothstep(0.026, 1.20, norm)
    old_blue = np.array([96.0, 108.0, 126.0], dtype=np.float32)
    thin_gray = np.array([158.0, 164.0, 168.0], dtype=np.float32)
    milky = np.array([242.0, 238.0, 224.0], dtype=np.float32)
    age_t = _smoothstep(60.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.85, age)
```

- [ ] **Step 3: Add charcoal blend for aged smoke**

After line ~2283 (after `base_rgb = base_rgb * (1.0 - age_t[..., None] * 0.32) + old_blue * (age_t[..., None] * 0.32)`), add:

```python
    # Charcoal blend for very old smoke
    charcoal = np.array([72.0, 78.0, 86.0], dtype=np.float32)
    charcoal_t = _smoothstep(0.65, 0.92, age_t)
    base_rgb = base_rgb * (1.0 - charcoal_t[..., None] * 0.45) + charcoal * (charcoal_t[..., None] * 0.45)
```

- [ ] **Step 4: Reduce fresh color boost**

Find and replace (around line 2287):

```python
    base_rgb += fresh[..., None] * np.array([12.0, 11.0, 6.0], dtype=np.float32)
```

- [ ] **Step 5: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_rgba_is_filamentary_and_translucent tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps -v
```

Expected: PASS - dense color validation should still pass (mean > 195.0), thin smoke should still have blue bias

- [ ] **Step 6: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
tune: smoke color mapping for atmospheric aesthetic

- Cooler blue-gray for thin smoke (96, 108, 126)
- Warmer cream for dense cores (242, 238, 224)  
- Add charcoal blend for very old smoke (age_t > 0.65)
- Reduce fresh color boost to (12, 11, 6)

Part of cigar smoke aesthetic tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Edge Softness/Alpha Falloff

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2207-2273`

- [ ] **Step 1: Run baseline gradient test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_rgba_is_filamentary_and_translucent -v
```

Expected: PASS - note current gradient percentile values

- [ ] **Step 2: Update blur radii in `_hybrid_smoke_field_rgba()`**

Find and replace (around lines 2207-2209):

```python
    fine = _pil_blur_float(density, 1.8)
    medium = _pil_blur_float(density, 6.5)
    broad = _pil_blur_float(density, 18.0)
```

- [ ] **Step 3: Update alpha shape calculation**

Find and replace (around line 2251):

```python
    alpha_shape = _smoothstep(0.012, 1.3, norm) ** 0.90
```

- [ ] **Step 4: Reduce hole influence on final alpha**

Find and replace (around line 2270):

```python
    alpha *= 1.0 - 0.10 * holes * edge_weight * (1.0 - 0.30 * source_core)
```

- [ ] **Step 5: Update final alpha blur**

Find and replace (around line 2273):

```python
    alpha = _pil_blur_float(alpha.astype(np.float32), 1.1)
```

- [ ] **Step 6: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_rgba_is_filamentary_and_translucent tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer -v
```

Expected: PASS - gradient 99th percentile should be < 0.135, alpha bands should still have 4 distinct ranges

- [ ] **Step 7: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
tune: edge softness for feathered smoke edges

- Increase blur radii: fine=1.8, medium=6.5, broad=18.0
- Gentler alpha falloff: smoothstep(0.012, 1.3) ** 0.90
- Reduce hole breakup: 0.18 -> 0.10 multiplier
- Softer final blur: 0.74 -> 1.1

Part of cigar smoke aesthetic tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Streamer and Ribbon Shape

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:1784-1791` (wind field)
- Modify: `examples/california_cigar_smoke_demo.py:1911-1916` (curl offset)

- [ ] **Step 1: Run baseline advection test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_density_advects_from_single_event_and_tracks_age -v
```

Expected: PASS

- [ ] **Step 2: Reduce synoptic noise in `_hybrid_wind_field()`**

Find and replace (around lines 1784-1785):

```python
    u += (0.36 * scale * (1.0 + 0.35 * altitude) * synoptic).astype(np.float32)
    v += (0.16 * scale * (1.0 + 0.28 * altitude) * np.sin(math.tau * (0.22 * xn + 0.24 * yn) - 0.006 * t + phase)).astype(np.float32)
```

- [ ] **Step 3: Increase lane texture amplitude**

Find and replace (around line 1791):

```python
    lane_amp = (26.0 + 11.0 * altitude) * scale
```

- [ ] **Step 4: Update curl offset in `_inject_hybrid_sources()`**

Find and replace the curl_offset calculation (around lines 1911-1916):

```python
        curl_offset = radius * (
            2.8 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
            + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + source.seed * 0.009)
            + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + source.seed * 0.005 + frame_index * 0.012)
        )
```

- [ ] **Step 5: Update tail width for fan formation**

Find and replace (around line 1915):

```python
        tail_width = radius * (1.15 + 4.2 * along_frac**0.75)
```

- [ ] **Step 6: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_density_advects_from_single_event_and_tracks_age tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps -v
```

Expected: PASS - centroid should still drift downwind, temporal correlation > 0.86

- [ ] **Step 7: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
tune: streamer shape for curved ribbons

- Reduce synoptic noise: u=0.36 (was 0.44), v=0.16 (was 0.20)
- Increase lane_amp: (26.0 + 11.0 * altitude) for coherent flow
- Add broad wave to curl_offset for ribbon formation
- Faster tail broadening: exponent 0.84 -> 0.75

Part of cigar smoke aesthetic tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Temporal Age/Lifecycle Behavior

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:1559-1565` (lifecycle alpha)
- Modify: `examples/california_cigar_smoke_demo.py:2066-2068` (decay rates)
- Modify: `examples/california_cigar_smoke_demo.py:2151` (haze feed)

- [ ] **Step 1: Run baseline lifecycle test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_lifecycle_fades_old_smoke tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer -v
```

Expected: PASS

- [ ] **Step 2: Update lifecycle alpha curve in `_hybrid_lifecycle_alpha()`**

Find and replace the function body (around lines 1559-1565):

```python
def _hybrid_lifecycle_alpha(age_frames: np.ndarray) -> np.ndarray:
    age = np.asarray(age_frames, dtype=np.float32)
    birth = 0.12 + 0.88 * _smoothstep(0.0, 6.0, age)
    mature = 1.0 - 0.20 * _smoothstep(30.0, 100.0, age)
    fade_end = min(HYBRID_SMOKE_MAX_AGE_FRAMES - 28.0, 236.0)
    old_fade = 1.0 - _smoothstep(145.0, fade_end, age) ** 0.80
    return np.clip(birth * mature * old_fade, 0.0, 1.0).astype(np.float32)
```

**Note:** Uses `start=145.0, exp=0.80` which extends visibility vs original (136.0, 0.72) while still passing `test_hybrid_lifecycle_fades_old_smoke`.

- [ ] **Step 3: Update decay rates in `HybridSmokeSimulator.step()`**

Find and replace (around lines 2066-2068):

```python
            old_smoke_decay = 0.028 + 0.015 * altitude
            base_decay = 0.991 - 0.004 * altitude
```

- [ ] **Step 4: Update residual haze feed in `_update_residual_haze()`**

Find and replace (around line 2151):

```python
        haze_feed = np.clip(old_smoke * 0.015 + high_slab * 0.0058 + self.density * 0.0032, 0.0, 1.0)
```

- [ ] **Step 5: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_hybrid_lifecycle_fades_old_smoke tests/test_california_cigar_smoke_hybrid.py::test_hybrid_residual_haze_persists_as_soft_aged_layer tests/test_california_cigar_smoke_hybrid.py::test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps -v
```

Expected: PASS - old smoke should still fade relative to mature, haze should still persist

- [ ] **Step 6: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
tune: temporal lifecycle for visible smoke generations

- Faster birth: smoothstep(0.0, 6.0) from (0.0, 9.0)
- Earlier mid-age dimming: smoothstep(30.0, 100.0)
- Later old_fade start: 145.0 (was 136.0) with exponent 0.80
- Slower decay: base=0.991, old_smoke=0.028
- Increased haze feed: old_smoke * 0.015

Part of cigar smoke aesthetic tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fire Glow/Bloom Through Smoke

**Files:**
- Modify: `examples/california_cigar_smoke_demo.py:2461-2487` (fire glow)

- [ ] **Step 1: Run baseline fire visibility test**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_smoke_composite_keeps_orange_sources_visible_under_veil -v
```

Expected: PASS

- [ ] **Step 2: Update bloom radius in `hybrid_fire_sources_rgba()`**

Find and replace (around lines 2461-2463):

```python
        halo_radius = radius * (3.4 + 2.1 * float(bloom_scale)) * max(0.70, float(bloom_scale))
        wide_radius = halo_radius * (1.9 + 0.30 * float(bloom_scale))
```

- [ ] **Step 3: Update bloom colors**

Find and replace wide halo color (around line 2465):

```python
            fill=(255, 98, 24, int(alpha * (0.048 + 0.038 * float(bloom_scale)))),
```

Find and replace inner halo color (around line 2469):

```python
            fill=(255, 108, 32, int(alpha * (0.13 + 0.08 * float(bloom_scale)))),
```

- [ ] **Step 4: Update final blur passes**

Find and replace (around lines 2484-2487):

```python
    wide_halo = wide_halo.filter(
        ImageFilter.GaussianBlur(radius=max(3.0, min(width, height) / 75.0 * max(0.8, float(bloom_scale))))
    )
    halo = halo.filter(ImageFilter.GaussianBlur(radius=max(1.5, min(width, height) / 160.0 * max(0.8, float(bloom_scale)))))
```

- [ ] **Step 5: Run regression tests**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py::test_smoke_composite_keeps_orange_sources_visible_under_veil tests/test_california_cigar_smoke_hybrid.py::test_physical_main_smoke_mechanisms_are_active_end_to_end -v
```

Expected: PASS - fire center should still be > 35 redder than surrounding

- [ ] **Step 6: Commit**

```bash
git add examples/california_cigar_smoke_demo.py
git commit -m "$(cat <<'EOF'
tune: fire glow/bloom for soft underglow through smoke

- Larger bloom radius: (3.4 + 2.1 * scale) from (2.8 + 1.8)
- Wider halo spread: 1.9 + 0.30 (was 1.68 + 0.22)
- Warmer orange: (255, 98, 24) and (255, 108, 32)
- Softer blur: wide=3.0/75.0, halo=1.5/160.0

Part of cigar smoke aesthetic tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Final Validation

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/test_california_cigar_smoke_hybrid.py -v
```

Expected: All tests PASS

- [ ] **Step 2: Visual validation (manual)**

Render test frames to verify aesthetic goals:

```bash
cd examples
python3 california_cigar_smoke_demo.py --duration 3 --output out/test_tuning.mp4
```

Check:
- Frame 30: Fresh bright plumes visible
- Frame 60: Fire glow visible through smoke
- Frame 90: Milky cores with blue-gray thin edges
- Frame 120: Curved ribbons, not sine-wave artifacts

- [ ] **Step 3: Create summary commit if all sections pass**

```bash
git log --oneline -5
```

Verify 6 commits from this tuning work are present (Task 0 baseline + 5 tuning tasks).

---

## Rollback Procedures

If any section causes test failures or visual degradation:

1. **Identify the problematic section** by reverting to the commit before that section
2. **Try conservative values** from the spec's noted ranges
3. **Skip if necessary** and document in the spec for future iteration

```bash
# To revert a specific section (example: Task 2)
git revert <commit-hash-for-task-2> --no-commit
git commit -m "revert: edge softness changes caused regression"
```
