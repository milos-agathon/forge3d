# Cigar Smoke Aesthetic Tuning Design

**Date:** 2026-06-11  
**Scope:** Parameter tuning for `examples/california_cigar_smoke_demo.py`  
**Goal:** Match high-altitude atmospheric smoke aesthetic from frame-by-frame reference analysis

## Target Aesthetic

Based on frame-by-frame analysis of reference footage (30 fps, first 30 seconds):

- **Color:** Milky white dense cores fading to blue-gray thin smoke and charcoal-gray aged haze
- **Edges:** Heavily feathered, no hard particle outlines
- **Texture:** Low-frequency, smooth with subtle internal streaks (advected density field, not sprites)
- **Movement:** Wind-field advection creating curved streamers, ribbons, hooks, loops, fan-shaped sheets
- **Generations:** Multiple smoke ages visible simultaneously (fresh bright plumes, mid-age bands, faint residual haze)
- **Source glow:** Orange fire points visible through smoke with soft bloom underneath

## Tuning Order

Each section is tuned and validated before moving to the next:

1. Smoke color/density mapping
2. Edge softness/alpha falloff
3. Streamer and ribbon shape
4. Temporal age/lifecycle behavior
5. Fire glow/bloom through smoke

## Guardrails

**Risk:** Multiple softness/persistence increases can turn the plume into a flat blanket.

**Mitigation:**
- All numeric values below are **first-pass ranges**, not final constants
- Tune one parameter group at a time; validate visually before proceeding
- If detail disappears, back off the most recent change before trying alternatives
- Render test frames at key points: frame 30 (early), frame 120 (mid), frame 200 (late)

---

## Section 1: Smoke Color/Density Mapping

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_hybrid_smoke_field_rgba()` (~lines 2277-2290)

### Current Values

```python
old_blue = np.array([108.0, 121.0, 136.0])   # thin/old smoke
thin_gray = np.array([164.0, 172.0, 174.0])  # transitional
milky = np.array([236.0, 235.0, 225.0])      # dense cores
```

### Proposed Changes (First-Pass)

```python
# Cooler blue-gray for thin smoke
old_blue = np.array([96.0, 108.0, 126.0])

# Slightly cooler transitional
thin_gray = np.array([158.0, 164.0, 168.0])

# Warmer cream for dense cores
milky = np.array([242.0, 238.0, 224.0])

# NEW: charcoal for very old smoke (age > 0.7 * MAX_AGE)
charcoal = np.array([72.0, 78.0, 86.0])
```

### Additional Adjustments

- Adjust `age_t` threshold: `_smoothstep(60.0, MAX_AGE * 0.85, age)` (was 40.0, MAX_AGE)
- Reduce `fresh` color boost: `[12, 11, 6]` (was `[16, 15, 10]`)

### Charcoal Blend for Aged Smoke

Add after the existing `base_rgb` calculations (~line 2283):

```python
# NEW: blend toward charcoal for very old smoke
charcoal = np.array([72.0, 78.0, 86.0], dtype=np.float32)
charcoal_t = _smoothstep(0.65, 0.92, age_t)  # activates for age_t > 0.65
base_rgb = base_rgb * (1.0 - charcoal_t[..., None] * 0.45) + charcoal * (charcoal_t[..., None] * 0.45)
```

---

## Section 2: Edge Softness/Alpha Falloff

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_hybrid_smoke_field_rgba()` (~lines 2207-2276)

### Blur Radii (First-Pass Ranges)

| Parameter | Current | Proposed Range | Notes |
|-----------|---------|----------------|-------|
| fine blur | 1.45 | 1.8 - 2.0 | Preserve filament detail |
| medium blur | 5.2 | 6.5 - 7.5 | |
| broad blur | 15.0 | 18.0 - 22.0 | |
| final alpha blur | 0.74 | 1.1 - 1.35 | Start conservative |

**Recommendation:** Start at lower end of ranges. Going straight to upper values may erase filament detail.

### Alpha Shape

```python
# Current
alpha_shape = _smoothstep(0.020, 1.0, norm) ** 1.02

# Proposed: gentler falloff
alpha_shape = _smoothstep(0.012, 1.3, norm) ** 0.90
```

### Holes/Breakup

**Strategy:** Reduce multiplier first, preserve internal texture. Only shift thresholds if multiplier reduction alone is insufficient.

`holes` affects three places in the code:
1. `texture_gain *= 1.0 - 0.28 * holes * edge_weight` (line ~2245)
2. `filament_mask *= 1.0 - 0.20 * holes * edge_weight` (line ~2248)
3. `alpha *= 1.0 - 0.18 * holes * edge_weight * (1.0 - 0.30 * source_core)` (line ~2270)

**Pass 1:** Only change the final alpha multiplier. Leave `texture_gain` and `filament_mask` unchanged to preserve internal texture.

```python
# Current (line ~2270)
alpha *= 1.0 - 0.18 * holes * edge_weight * (1.0 - 0.30 * source_core)

# First pass: reduce multiplier only
alpha *= 1.0 - 0.10 * holes * edge_weight * (1.0 - 0.30 * source_core)  # range: 0.10 - 0.12
```

**Pass 2 (only if needed):** If hard breakup persists after pass 1, also reduce `texture_gain` and `filament_mask` multipliers conservatively:

```python
# texture_gain (line ~2245): 0.28 -> 0.18-0.22
texture_gain *= 1.0 - 0.20 * holes * edge_weight

# filament_mask (line ~2248): 0.20 -> 0.12-0.16
filament_mask *= 1.0 - 0.14 * holes * edge_weight
```

**Pass 3 (last resort):** Only shift hole calculation thresholds if multiplier reductions are insufficient:

```python
holes = _smoothstep(0.18, 0.78, 1.0 - broad_texture) * ...  # was 0.12, 0.70
```

---

## Section 3: Streamer and Ribbon Shape

**Files:** `_inject_hybrid_sources()` (~lines 1872-1979), `_hybrid_wind_field()` (~lines 1750-1796)

### Curl/Wave Amplitude

```python
# Current
curl_offset = radius * (
    2.4 * np.sin(along / max(radius * 8.8, 1.0) + ...)
    + 3.8 * along_frac * np.sin(along / max(radius * 17.0, 1.0) + ...)
)

# Proposed: add broad wave for ribbon formation
curl_offset = radius * (
    2.8 * np.sin(along / max(radius * 8.0, 1.0) + ...)      # slight increase
    + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + ...)
    + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + ...)  # NEW broad wave
)
```

**Note:** The short curl wave increase is conservative. Watch for regular sine-wave ribbon artifacts. The new broad wave is the most valuable addition.

### Tail Width (Fan/Sheet Formation)

```python
# Current
tail_width = radius * (1.08 + 3.65 * along_frac**0.84)

# Proposed: faster broadening
tail_width = radius * (1.15 + 4.2 * along_frac**0.75)
```

### Wind Field Adjustments

**Location:** `_hybrid_wind_field()` (lines ~1784-1791)

The synoptic noise is applied inline to `u` and `v` separately:

```python
# Current (lines 1784-1785)
u += (0.44 * scale * (1.0 + 0.35 * altitude) * synoptic).astype(np.float32)
v += (0.20 * scale * (1.0 + 0.28 * altitude) * np.sin(...)).astype(np.float32)

# Proposed: reduce synoptic noise that breaks up flow lanes
u += (0.36 * scale * (1.0 + 0.35 * altitude) * synoptic).astype(np.float32)  # was 0.44
v += (0.16 * scale * (1.0 + 0.28 * altitude) * np.sin(...)).astype(np.float32)  # was 0.20
```

Lane texture amplitude (line ~1791):

```python
# Current
lane_amp = (22.0 + 9.0 * altitude) * scale

# Proposed: increase for coherent curved paths
lane_amp = (26.0 + 11.0 * altitude) * scale
```

**Rationale:** Increasing `lane_amp` and reducing synoptic noise creates coherent flow lanes, which is the core visual characteristic of the reference.

---

## Section 4: Temporal Age/Lifecycle Behavior

**Files:** `_hybrid_lifecycle_alpha()` (~lines 1559-1565), `HybridSmokeSimulator.step()`

### Lifecycle Alpha Curve

**CRITICAL:** The `old_fade` formula must use a **later start** and **higher exponent** for a longer visible tail.

```python
# Current
birth = 0.18 + 0.82 * _smoothstep(0.0, 9.0, age)
mature = 1.0 - 0.16 * _smoothstep(36.0, 118.0, age)
fade_end = min(HYBRID_SMOKE_MAX_AGE_FRAMES - 28.0, 236.0)
old_fade = 1.0 - _smoothstep(136.0, fade_end, age) ** 0.72

# Proposed: clearer generation separation with longer tail
birth = 0.12 + 0.88 * _smoothstep(0.0, 6.0, age)           # faster birth
mature = 1.0 - 0.20 * _smoothstep(30.0, 100.0, age)        # earlier mid-age dimming
old_fade = 1.0 - _smoothstep(155.0, fade_end, age) ** 1.0  # later start, linear fade
```

**Tuning range for `old_fade`:**
- Start: 150 - 170 (later = longer visibility)
- Exponent: 0.9 - 1.15 (higher = gentler initial fade)

### Decay Rates in `step()`

```python
# Current
base_decay = 0.988 - 0.005 * altitude
old_smoke_decay = 0.034 + 0.018 * altitude

# Proposed: slower decay for visible haze
base_decay = 0.991 - 0.004 * altitude         # range: 0.990 - 0.993
old_smoke_decay = 0.028 + 0.015 * altitude    # range: 0.026 - 0.030
```

### Residual Haze Feed

```python
# Current
haze_feed = np.clip(old_smoke * 0.012 + high_slab * 0.0048 + self.density * 0.0026, ...)

# Proposed: conservative increase
haze_feed = np.clip(old_smoke * 0.015 + high_slab * 0.0058 + self.density * 0.0032, ...)
```

**Range:** `old_smoke * 0.014 - 0.016` (0.018 is too aggressive as a first pass)

---

## Section 5: Fire Glow/Bloom Through Smoke

**File:** `hybrid_fire_sources_rgba()` (~lines 2440-2492)

**Note:** Bloom tuning interacts with smoke composite. Test with smoke overlay, not in isolation.

### Bloom Radius

```python
# Current
halo_radius = radius * (2.8 + 1.8 * bloom_scale) * max(0.65, bloom_scale)
wide_radius = halo_radius * (1.68 + 0.22 * bloom_scale)

# Proposed: larger, softer
halo_radius = radius * (3.4 + 2.1 * bloom_scale) * max(0.70, bloom_scale)
wide_radius = halo_radius * (1.9 + 0.30 * bloom_scale)
```

### Bloom Color (Warmer, More Visible)

```python
# Wide halo: warmer orange, higher alpha
fill=(255, 98, 24, int(alpha * (0.048 + 0.038 * bloom_scale)))  # was (255, 86, 18, ...)

# Inner halo: softer orange
fill=(255, 108, 32, int(alpha * (0.13 + 0.08 * bloom_scale)))   # was (255, 98, 24, ...)
```

### Final Blur

**Location:** lines ~2484-2487

```python
# Current (lines 2484-2487)
wide_halo = wide_halo.filter(
    ImageFilter.GaussianBlur(radius=max(2.0, min(width, height) / 92.0 * max(0.8, float(bloom_scale))))
)
halo = halo.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) / 210.0 * max(0.8, float(bloom_scale)))))
wide_halo.alpha_composite(halo)
halo = wide_halo

# Proposed: more diffuse bloom
wide_halo = wide_halo.filter(
    ImageFilter.GaussianBlur(radius=max(3.0, min(width, height) / 75.0 * max(0.8, float(bloom_scale))))
)
halo = halo.filter(ImageFilter.GaussianBlur(radius=max(1.5, min(width, height) / 160.0 * max(0.8, float(bloom_scale)))))
wide_halo.alpha_composite(halo)
halo = wide_halo
```

---

## Validation Checkpoints

After each section, render test frames and verify:

| Checkpoint | Frame | What to Check |
|------------|-------|---------------|
| Color mapping | 90 | Dense cores cream/white, thin smoke blue-gray |
| Edge softness | 90 | No hard edges, but filament detail preserved |
| Streamers | 120 | Curved ribbons visible, not sine-wave artifacts |
| Lifecycle | 180 | Three generations visible (fresh, mid, haze) |
| Fire bloom | 60 | Orange glow visible through smoke overlay |

### Regression Tests

The existing test suite in `tests/test_california_cigar_smoke_hybrid.py` provides automated guardrails for this tuning:

- **Alpha coverage assertions:** Ensure smoke doesn't disappear or become a flat blanket
- **Gradient checks:** Verify edge softness produces smooth falloff
- **Dense color validation:** Confirm milky cores remain bright
- **Temporal coherence:** Check age-based generation separation
- **Fire visibility:** Assert source glow transmits through smoke layer

Run the test suite after each section to catch regressions early:

```bash
python -m pytest tests/test_california_cigar_smoke_hybrid.py -v
```

## Rollback Strategy

If a section degrades the visual:
1. Revert that section's changes
2. Try values at the conservative end of the range
3. If still problematic, skip that section and note for future iteration
