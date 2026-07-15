# Why the HDR barely changes the Italy-forest PT poster's colour mood

**Date:** 2026-07-13
**Scope:** Root-cause diagnosis + ranked remediation plan (read-only investigation; no code changed)
**Subject:** `examples/forest_cover_copernicus/italy_forest_pt_3d.py` (PT poster) and its imported
composer `examples/forest_cover_copernicus/italy_forest_cover_3d.py`
**Evidence:** cached per-HDR light fields in `examples/out/italy_forest_pt/pt_light_field_*.npz`
(snow_field_1k, venice_sunset_4k, brown_photostudio_02_4k); tracer source under
`src/path_tracing/`. Two measurement harnesses (per-stage attenuation budget + cosine-weighted
irradiance/mood extractor) reproduce every number below.

> **Independent audit (2026-07-13):** verified against the working tree by an independent audit and a
> `float64` re-derivation. The root cause and the Option 1 recommendation stand; two headline
> magnitudes were corrected — a `float32`-accumulator artifact and a domain mislabel ("plate-wide"
> conflated the full light field with the map subject). Over the **Italy map subject** the colour
> shift is **~2.3%** (~0.6% over the full light field) and the anchor counterfactual is
> **−0.023 → −0.071**; both corrections strengthen the same causal direction. See *Provenance &
> precision notes* in the appendix.

---

## TL;DR

The HDR **does** change the poster's *luminance* mood (brightness, relief contrast, shadow depth
— that is why venice reads `medianL` 181 vs snow 175) — but it **cannot** change the *colour*
mood, and the small colour signal it would deliver points the **wrong way**: venice's physical
light is **blue**, not warm. Three stacked mechanisms neutralise colour, only one of which is the
tint-anchor originally suspected — and the anchor turns out to be *helping*, not hurting. To make
the "sunset read warm" you cannot let more of the physical cast through (that makes venice bluer);
you need an **artistic mood-grade derived from the HDR's warm horizon band**, added as one
compose-time knob.

---

## (a) Quantified root-cause diagnosis

### The phenomenon, stated precisely

The overlay is a **fixed, light-free palette**
([`_forest_rgba`](../examples/forest_cover_copernicus/italy_forest_cover_3d.py) — colour is a
function of forest cover only, plus a deterministic light-free moss grain, so it takes **no lighting
input**). The HDR reaches the finished plate through exactly **two** channels out of
[`_shade_on_overlay_grid`](../examples/forest_cover_copernicus/italy_forest_pt_3d.py): the
achromatic `shade` (→ brightness/contrast) and the chromatic `tint`. Everything else in the
compose path (`_apply_final_map_brightness`, page vignette, contact-shadow glows) is
**HDR-independent**.

- **Luminance mood transfers fine.** Replaying `shade` → `scale` over the **Italy map subject**
  (overlay alpha): mean-brightness snow **0.857** → venice **0.890** (+3.9%) → brown **0.912**
  (+6.5%); mid-dark area fraction (`shade < 0.5`) snow **30.0%** vs brown **13.2%** (>2×). This
  matches the observed `medianL`/relief-contrast spread. Nothing is broken here.
- **Colour mood does not.** Over the **Italy map subject** the HDR shifts colour by only **~2.3%**
  (~0.6% over the full light field); the only larger movement (~9%) is confined to dark shadows —
  and for venice it is toward **blue**.

### The three stacked neutralisers (colour), with the attenuation budget

Measured venice-vs-snow warm ratio (R/B; `<1` = bluer than the neutral snow control) at each
pipeline stage, over the **full light field** unless noted:

| Stage | venice R/B | snow R/B | venice ÷ snow | what happens |
|---|---|---|---|---|
| **Env up-irradiance** (physical diffuse fill) | 0.528 | 1.016 | **0.52×** | ← **root: geometry** |
| Raw light field (all hits) | 0.882 | 1.022 | 0.86× | sun dilutes the blue |
| Raw light field (shadow) | 0.463 | 1.001 | 0.46× | shadows = pure blue fill |
| Raw shadow *tint* | 0.444 | 0.991 | 0.45× | |
| After anchor + **clamp** (shadow) | 0.788 | 0.973 | 0.81× | clamp caps the residual |
| Final `tint_mix` (shadow) | 0.899 | 0.988 | 0.91× | ×0.45 strength halves it |
| **Final `tint_mix` (full light field)** | 0.995 | 1.001 | **0.994×** | **−0.61%** |
| **Final `tint_mix` (Italy map subject)** | 0.981 | 1.004 | **0.977×** | **−2.27% — the map figure** |

**Layer A — geometry + physics (the root cause).** The render surface is an *up-facing* canopy
shot from *nadir*, and the IBL fill is `albedo · env(cosine-sampled upper hemisphere)`
([`hybrid_terrain_traversal.wgsl` line 481](../src/shaders/hybrid_terrain_traversal.wgsl),
orientation `vv = acos(d.y)/π` so row 0 = zenith). So a Lambertian canopy integrates the **sky**.
For `venice_sunset`, the cosine-weighted up-facing irradiance is **R/B 0.528 (strongly blue)** —
the vivid sunset orange lives in a narrow horizon band (central 10 of the 64 env rows; R/B 1.905) /
a few high-radiance texels (saturation × peak-radiance weighted, R/B 14.6) that an up-facing surface
barely samples. **The light field is blue
at birth** (lf_all 0.882, shadow 0.463), *before any overlay logic runs.* This was adversarially
upheld three ways, including against the tracer's own cached output (shadow R/B 0.463 matches
row0=up, not the flipped 0.963).

**Layer B — the fixed warm-white sun (is the sun dominating? YES).** The only warm term is the
directional sun `light_color = [I, 0.97·I, 0.92·I]`
([`render_terrain.rs` line 395](../src/path_tracing/hybrid_compute/render_terrain.rs)), with
`ambient_color = [0,0,0]` (line 457) and **no `sun_color` parameter on the Python seam**
([`path_tracing.py` line 877](../python/forge3d/path_tracing.py)). Solving venice's lit-pixel
chromaticity as a sun/env mix, the fixed sun supplies **~70–80% of the lit-surface chromaticity**
(an independent least-squares fit lands at **75.9%**; this is a chromaticity estimate, not a raw
irradiance fraction — the cached RGB is Reinhard-tonemapped,
[`hybrid_terrain_traversal.wgsl` line 512](../src/shaders/hybrid_terrain_traversal.wgsl)). It
is *identical for all three HDRs*, so it can never carry sunset-specific colour — it just drags the
lit majority toward neutral (env 0.528 → lit 0.892).

**Layer C — the colour-fidelity guardrails (which steps neutralise, quantified).** This is where
the "it's the anchor" hypothesis was close but the mechanism is subtler than "the anchor removes
the cast":

- The **anchor** (`tint /= median_lit_tint`,
  [`italy_forest_pt_3d.py` line 303](../examples/forest_cover_copernicus/italy_forest_pt_3d.py))
  divides out the global cast, pinning the lit majority to `tint≈1` — this is what holds the
  map-subject colour shift down to **~2.3%** (~0.6% over the full light field). Adversarial
  counterfactual (anchor disabled): the venice−snow transfer moves **−0.023 → −0.071** over the map
  subject (**−0.0061 → −0.0639** over the full field; shadow **−0.0894 → −0.0984**), i.e. **removing
  the anchor makes venice *much bluer*** because its global cast is blue (anchor R/B 0.889). **The
  anchor is materially *warming* venice, not suppressing warmth.**
- The **CLAMP `(0.88, 1.12)`**, not the anchor, is the operative governor of what little survives:
  raw shadow tint **0.444 → 0.787** by the clamp; with vs without the anchor the clamped shadow
  differs by only **0.0008** R/B (0.7870 vs 0.7862), dropping below 0.0005 only after the
  0.45-strength mix. Then `LIGHT_TINT_STRENGTH=0.45` halves the residual again (→ 0.899).

**Net:** colour is held to ~2.3% over the map subject (~0.6% over the full light field) *by design*
(anchor pins lit areas → clamp caps shadows → 0.45 strength halves them), and the ~9% that leaks
into shadows is venice's true
**blue**. There is no warmth for any downstream step to reveal — the signal was blue before the
overlay ever touched it. (*Ordering:* Layer A is the wrong-sign **source**, Layer B the fixed-sun
**dilution**, Layer C the compose-time **bottleneck** — for "why venice isn't warm" A is primary;
for "why *any* HDR chroma barely reaches the plate" C is the binding constraint.)

### Two verified corrections to the original framing

1. **The anchor is not the villain.** Deleting/relaxing it (the literal "let the full cast
   through") makes venice **cooler**, the opposite of the goal.
2. **`PALETTE_CHROMA_RESTORE = 1.38` is inert in this PT path.** `COMPOSE_POSTER=False` →
   `_compose_snapshot` takes the branch at
   [`italy_forest_cover_3d.py` lines 1919–1935](../examples/forest_cover_copernicus/italy_forest_cover_3d.py),
   which never calls `_restore_palette_chroma_from_source`. That restore lives on the *separate*
   composer `_render` flow (`_combine_render_passes` at line 2045 → `_compose_clean_palette_with_relief`
   at 1596–1644), which the PT poster never enters — it is not in `_compose_snapshot`'s poster branch.
   So it is neither a lever nor a
   regression risk here; the live "not-too-pale" guards in the PT path are the warm palette itself,
   `FINAL_MAP_BRIGHTNESS=0.96`, and `HIGHLIGHT_SCALE_CAP=1.06`.

---

## (b) Ranked remediation plan

The key realisation: "mood" splits into two incompatible notions the original brief conflates.
**"Full physical cast"** (as described) = honest but **blue** for venice. **"Warm sunset look"**
(what is actually wanted) = an artistic grade toward the env's *dominant warm* chromaticity. The
plan delivers one `MOOD_STRENGTH ∈ [0,1]` knob (default 0 = byte-identical) with a
`MOOD_MODE ∈ {grade, cast}` selector so both notions are available; **default `grade`**.

| # | Option | Function / knob | Delivers warm venice? | Dark risk | Pale risk | Effort | Compose-time? |
|---|---|---|---|---|---|---|---|
| **1 ✅** | **Env mood-grade (artistic warm)** | `_modulate_overlay` + new `_env_mood_tint(hdr)`; `MOOD_STRENGTH` | **Yes** | low | med* | low | **Yes** |
| 2 | Un-anchor blend (honest cast) | `_shade_on_overlay_grid` anchor `**(1−k)` + clamp co-scale | No (→ true blue) | med | low | low | Yes |
| 3 | Upstream trace (sun_color / SUN:ENV ratio) | `render_terrain.rs` + Python seam | No (wrong sign) | med | med | high | **No** |
| 4 | Naive: widen clamp + raise strength | `LIGHT_TINT_*` constants | No (amplifies blue) | low | low | low | Yes |

`*` pale risk is `med` only for the naive spec; the recommended luminance-preserving + capped
variant drops it to low.

### ★ RECOMMENDED DEFAULT — Option 1: env mood-grade

In [`_modulate_overlay`](../examples/forest_cover_copernicus/italy_forest_pt_3d.py), multiply the
plate by `mood_mult = 1 + MOOD_STRENGTH·(mood_tint − 1)`, where `mood_tint` is a new
`_env_mood_tint(hdr)` reusing `_load_hdr_env`. It is the **only** option that can warm venice,
because it decouples from the physically-blue fill. Non-negotiable design conditions:

- **Horizon-band extractor, NOT saturation-weighted.** The horizon band (central 10 of the 64 env
  rows) auto-adapts correctly (venice R/B **1.905**, snow 1.031, brown 1.258); saturation × peak-
  radiance weighting is pathological (venice 14.6 extreme, **brown 0.847 points cool — wrong way**).
- **Luminance-preserving** multiply (re-add the luminance delta, as
  `_restore_palette_chroma_from_source` does at
  [`italy_forest_cover_3d.py` lines 1244–1248](../examples/forest_cover_copernicus/italy_forest_cover_3d.py))
  → protects both "not too dark" and "not too pale."
- **Cap** the warm boost (analogue of `HIGHLIGHT_SCALE_CAP`) so bright cream-sand highlights don't
  clip toward white (the pale-flake failure).
- **Grade the legend swatches too** (`_draw_moss_legend`), else the legend↔map read drifts.
- Default is `0` (off, byte-identical to current); `~0.25–0.35` is the **recommended preset** when
  mood is wanted. Sweeps under `--reuse-shade` in seconds (a full HDR decode then downsample to 64
  rows — cheap vs a re-trace; no GPU).

### Option 2 — un-anchor blend (ship as the honest companion mode)

`tint /= anchor**(1 − MOOD_STRENGTH)` with clamp width co-scaled. This is the literal "full cast,"
and it is worth having, but be clear it delivers venice's **true blue** mood (cooler/darker),
addresses only Layer C, and is really 2–3 coupled params (the un-anchored blue blows past the
`1.12` clamp asymmetrically). Guard `k==0` with an explicit branch for byte-identity.

### Option 3 — upstream trace (reject as default)

Wrong sign (lowering sun→env pushes venice toward its blue fill; a warm `sun_color` is then
stripped by the very anchor that neutralises lit pixels), can't iterate at compose-time (re-trace
at the 512 MiB ceiling), needs a Rust+PyO3 change patched at **two** sites (`render_terrain.rs:395`
*and* the ReSTIR dir-light at `:497`), and has a stale-cache footgun (the `.npz` key is HDR-stem
only, [`italy_forest_pt_3d.py` line 419](../examples/forest_cover_copernicus/italy_forest_pt_3d.py)
— a sun_color change would silently reuse a stale field). Its only honest niche is subtle warmth
for already-neutral HDRs.

### Option 4 — naive clamp/strength (reject)

Amplifies the *true* residual, which is blue → venice gets **cooler**; only bites shadows anyway.
Explicitly the wrong lever.

---

## (c) Verification recipe

**Iterate at compose-time** on the cached fields:

```bash
# venice vs snow, grade mode, same palette, cached light fields (no GPU)
.venv/Scripts/python examples/forest_cover_copernicus/italy_forest_pt_3d.py --reuse-shade \
  --hdr D:/ghsl-population/venice_sunset_4k.hdr --palette swatch \
  --snapshot examples/out/italy_forest_pt/ab_venice_moodX.png
.venv/Scripts/python examples/forest_cover_copernicus/italy_forest_pt_3d.py --reuse-shade \
  --hdr C:/Users/milos/forge3d/assets/hdri/snow_field_1k.hdr --palette swatch \
  --snapshot examples/out/italy_forest_pt/ab_snow_moodX.png
```

**Metric** (measure on the masked map subject of the finished PNGs), three gauges:

1. **Mood lands (the target):** shadow-region **R − B** (0–255). Baseline today: venice **56**,
   snow **57** (venice is −1, i.e. *cooler*). Success: at `MOOD_STRENGTH≈0.3`,
   **venice_shadow(R−B) − snow_shadow(R−B) ≥ +8**, and monotonically increasing with the knob.
   (Also check overall R−B, not just shadow, since grade is global.)
2. **"Not too dark" holds:** mean plate luminance within **±3%** of the `MOOD_STRENGTH=0` render.
3. **"Not too pale" holds:** fraction of subject pixels with any channel ≥ 250 **does not
   increase** vs baseline (guards the cream-highlight clip).

*Mask definitions.* The map **subject** = the overlay alpha mask (`alpha > 0`) propagated to poster
coordinates — NOT a luminance threshold, which would sweep in the title, legend and contact shadows.
The **shadow** region = subject pixels whose carried `shade ≤ 0.25`; carry the actual `shade` field
through the same resize as the plate — do **not** approximate it with a darkest-luminance quartile,
which confounds dark-green forest cover with lighting shadow. Measure R−B under the identical mask
across variants.

A cheap **predictive** check without rendering (drives Option 1 tuning): on the cached field, the
finished warm shift ≈ `MOOD_STRENGTH · (mood_tint_R − mood_tint_B)` with `mood_tint` = the
luminance-normalised horizon-band env colour — venice horizon R/B 1.905 vs snow 1.031 means venice
gains warmth while snow stays ~neutral by construction. Confirm the *sign* is positive for venice
before committing a full render.

---

## Appendix — how the numbers were produced

Three pure-numpy harnesses (no forge3d import, no GPU) replicate `_load_hdr_env`,
`_shade_on_overlay_grid` and `_modulate_overlay` verbatim and read the cached `.npz` light fields:

- **per-stage attenuation budget** — env chromaticity → raw light field (all / lit / shadow) →
  tint anchor → clamp → `tint_mix`, with venice-vs-snow ratios at each stage (full light field).
- **cosine-weighted irradiance + mood extractor** — proper up-facing diffuse irradiance colour
  (confirms venice fill is physically blue) and the saturation-weighted / horizon-band "mood
  colour" (confirms a warm grade source exists for Option 1).
- **subject-domain harness** — applies the production bicubic resize to overlay resolution and masks
  by the overlay alpha, so the Italy map-subject figures are reported separately from the full
  rectangular light field. All summary means use a `float64` accumulator.

The diagnosis was adversarially red-teamed (7 independent verification/stress agents): the
"anchor is the sole cause" claim was **refuted** (removing the anchor makes venice bluer); the
"venice fill is physically blue" claim was **upheld** on three independent lines; and the
"colour barely changes" money-metric was judged **valid for chroma but incomplete for mood**
(it omits the achromatic `shade` channel, through which brightness/relief mood *does* transfer).

**Provenance & precision notes.** The cached `.npz` files store only `rgb` and `hit` (no HDR path
or hash); the HDR each field was traced against is inferred from the filename stem
(`venice_sunset_4k` → `D:/ghsl-population/venice_sunset_4k.hdr`, distinct from the unused
`assets/hdri/venice_sunrise_4k.hdr`, whose up-irradiance is R/B **0.598**, not 0.528).

Two audit passes corrected the quantitative claims; the mechanism, causal ordering and the Option 1
recommendation were unchanged in both. **(1) Accumulator:** an earlier draft used numpy's default
`float32` accumulator over millions of pixels; all summary means are now `float64` (this moved the
full-field colour shift 0.17% → 0.61% and the full-field anchor counterfactual −0.0017 → −0.0082 up
to −0.0061 → −0.0639). **(2) Domain:** "plate-wide" originally meant the **full rectangular light
field**, 78% of which is transparent no-data; the honest map figure is the **Italy map subject**
(overlay alpha, after the production bicubic resize). Three domains are now kept distinct — full
light field, Italy subject, and finished poster. Headline subject values: colour shift **−2.27%**,
anchor counterfactual **−0.023 → −0.071**, luminance scale **0.857 / 0.890 / 0.912**
(snow / venice / brown). All figures were independently re-derived and confirmed against the working
tree.
