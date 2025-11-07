## M1 — Dielectric Cook-Torrance GGX (direct light) ✅ *one scene, end-to-end*

**Goal**
Produce physically plausible, gamma-corrected PNGs for a gray dielectric sphere lit by a single directional light, with roughness sweep and component debugs (NDF, G, spec radiance). This fixes core issues: exposure/gamma, roughness binding, denominator stability, and correct GGX math.

**Scope (in / out)**

* **In:** GGX microfacet BRDF, Schlick Fresnel, Smith GGX G, stable denominator, linear→sRGB encode, per-tile roughness bind, debug views.
* **Out (later):** Principled energy comp, metals, IBL/LUTs, tone mapping alt, perf.

**Scene parameters (hardcoded for the gallery)**

* Geometry: sphere radius = 1.
* Camera: `V = normalize(0,0,1)`.
* Normal: from sphere.
* Light: directional `L = normalize(0.5, 0.5, 1)`.
* Dielectric: `F0 = 0.04` (scalar), `baseColor = 0.5` gray, diffuse = `baseColor/π`.
* Roughness sweep tiles: `r ∈ {0.10, 0.30, 0.50, 0.70, 0.90}`, `α = r²`.
* Radiance: `Li = 3.0`.
* Output: **sRGB PNG, RGB8** (no alpha).

**Reference math (must match)**

* `H = normalize(L + V)`
* `NL = saturate(dot(N,L))`, `NV = saturate(dot(N,V))`, `NH = saturate(dot(N,H))`, `VH = saturate(dot(V,H))`
* GGX NDF: `a = α; a2=a*a; den=(NH*NH)*(a2-1)+1; D=a2/(π*den*den)`
* Smith G1 (Schlick-GGX form):
  `k = (a+1)^2/8`, `G1(x)= x/(x*(1-k)+k)` with `x ∈ {NL,NV}`; `G=G1(NL)*G1(NV)`
* Schlick Fresnel: `F = F0 + (1-F0)*(1-VH)^5`
* Specular: `den_spec = max(4*NL*NV, 1e-4)`, `f_spec=(D*F*G)/den_spec`
* Diffuse: `f_diff = baseColor/π`
* Shading: `Lo = (f_spec + f_diff) * Li * NL`
* Encode: per-channel sRGB OETF.

**Deliverables**

1. `m1_brdf_gallery_ggx.png` — 1×5 tiles, caption `GGX  r=0.30  α=0.0900`.
2. `m1_debug_ndf.png` — `D` remapped (see generator).
3. `m1_debug_g.png` — `G(N,L,V)` grayscale.
4. `m1_debug_spec_radiance.png` — `(D*F*G/den_spec)*Li*NL`.
5. `m1_meta.json` — exact numeric inputs per tile + **hash** of GPU constants.

**Acceptance (must pass)**

* **A1 Visibility:** mean sRGB luminance ∈ **[75, 140]** for each tile.
* **A2 Roughness monotonic:** in `m1_debug_spec_radiance.png`, **area( radiance > 0.5 linear )** increases with `r`.
* **A3 NDF shape:** `m1_debug_ndf.png` is a peaked lobe that widens with `r` (not a disc).
* **A4 No blow-ups:** no NaN/Inf and no linear > **32** before encode (log if violated).
* **A5 Binding sanity:** `m1_meta.json` shows unique `α` per tile and captions match.

**Tasks (order)**
T1 encode, T2 saturate/eps, T3 GGX D/G/F, T4 stable denom, T5 dielectric path, T6 push constants, T7 debug switch, T8 gallery/labels/gutters, T9 meta dump, T10 CPU luminance stats.

---

## M2 — **Disney Principled (Burley) + Energy Compensation**  *(expanded, strict)*

**Goal**
Add **Burley diffuse** and **specular energy compensation** so that for the same scene as M1:

* At **r=0.50**, **mean luminance** of Principled vs GGX differs by **≤ ±5%**.
* At high roughness (r≥0.7) Principled **does not darken** vs GGX after compensation (±3% brighter allowed).

**Scope**

* **In:** Burley diffuse; microfacet same as M1 (GGX), **compensated** specular (Karis-style); dielectric only.
* **Out:** Metals, IBL.

**Reference math**

* Burley diffuse (Disney 2012):

  ```
  fd90 = 0.5 + 2 * r * VH^2
  FL = (1 - NL)^5
  FV = (1 - NV)^5
  f_diff_burley = (baseColor/π) * (1 + (fd90 - 1)*FL) * (1 + (fd90 - 1)*FV)
  ```

  with `r = roughness`. Clamp all dot terms to [0,1].
* Specular energy compensation (screened diffuse):

  * Compute **white-Fresnel average** for GGX: `E = integrate_specular_energy_approx(F0, roughness)` using Karis approximation:
    `f_add = 0.04 * (1 - roughness) + roughness*roughness*0.5` (fixed scalar)
    Compensation factor: `k_comp = 1 - E` (clamped [0,1])
  * Final: `Lo = (f_spec * k_comp + f_diff_burley) * Li * NL`

**Numerics**

* Epsilon identical to M1; all math in **linear**; encode **sRGB** at the end.

**Deliverables**

1. `m2_brdf_gallery_principled.png` — 1×5 tiles as M1.
2. `m2_debug_energy.png` — 1×5 tiles of **energy compensation factor** visualized as grayscale (0..1).
3. `m2_meta.json` — all M1 fields + `model:"principled"`, `comp_factor`, and **side-by-side** GGX/Principled **mean luminance** per tile.

**Acceptance**

* **B1 r=0.50 match:** `|meanLum_principled - meanLum_ggx| / meanLum_ggx ≤ 0.05`.
* **B2 high-r brighter:** for r ∈ {0.7,0.9}, `meanLum_principled ≥ meanLum_ggx`.
* **B3 factor bounds:** `comp_factor ∈ [0,1]` everywhere; min/max reported in meta.
* **B4 energy monotonic:** `comp_factor` **increases** with roughness (non-decreasing).

**Tasks**

* P1 implement Burley diffuse; P2 implement `comp_factor` Karis approx; P3 plumb debug view; P4 generator compares to M1 GGX meta and writes comparative stats; P5 hard fail if any acceptance fails.

---

## M3 — **Metal Workflow** (RGB F0, zero diffuse)  *(expanded, strict)*

**Goal**
Add **metal** mode where **diffuse = 0** and **F0 = baseColor RGB**. Validate that hue tracks baseColor and no diffuse leakage exists.

**Scope**

* **In:** dielectric vs metal **switch**, RGB F0, same GGX microfacet.
* **Out:** IBL.

**Scene**

* Two galleries, same roughness set as M1/M2.

  * **Dielectric ref:** `F0=0.04`, `baseColor=0.5`.
  * **Metal:** `F0=baseColorRGB={1.0, 0.71, 0.29}` (gold reference), `diffuse=0`.

**Math**

* Dielectric branch = M1/M2.
* Metal branch: `f_diff=0`, `F0 := baseColorRGB`, **Schlick** Fresnel.

**Deliverables**

1. `m3_gallery_dielectric.png` — same as M1 output for comparison.
2. `m3_gallery_metal_gold.png` — 1×5 tiles.
3. `m3_meta.json` — adds `material:"dielectric|metal"`, `F0_rgb`, and **per-channel** mean luminance per tile.

**Acceptance**

* **C1 zero diffuse:** in metal gallery, **mean of diffuse debug** must be **0** (create `m3_debug_diffuse.png` and assert all pixels == 0 ± 1e-6).
* **C2 color fidelity:** for r=0.3 tile, **specular tint** channel ratios match input F0 within **±5%** (R/G and R/B ratios).
* **C3 no white clipping:** max linear value ≤ **32**; per-channel sRGB ≤ **255**.

**Tasks**

* M1 metal branch; M2 RGB F0 input path; M3 diffuse kill switch; M4 color-ratio audit.

---

## M4 — **Preintegrated IBL (split-sum)**  *(expanded, strict)*

**Goal**
Add environment lighting using split-sum approximation: **Lambert irradiance** + **prefiltered specular** + **DFG LUT**. Validate energy behavior across roughness and **no seams**.

**Scope**

* **In:** Cube-map IBL; GGX specular prefilter (importance sample GGX), **GGX/Smith visibility** in DFG LUT; cosine-convolved diffuse irradiance.
* **Out:** MIS, area lights.

**Assets / Sizes (fixed for M4)**

* Input HDR equirect: `assets/studio_small.hdr`.
* Convert to cubemap: size **512** (per face), 32-bit float RGBA.
* Prefiltered specular mip chain: base **512**, down to **1**, **128 samples** at top mip decreasing linearly to **16** at lower mips.
* Irradiance cubemap: **64** per face, **cosine-weighted** 1024 samples.
* DFG LUT: **256×256**, UV = (NoV, roughness), float RG16F or RG32F.
* **Clamp** roughness to [0.02, 1].

**Reference**

* Specular prefilter integrates `D*G*F/den_spec` over directions with **GGX VNDF importance sampling** (Heitz).
* DFG LUT stores **(integrated F, integrated visibility)** per (NoV, r).

**Deliverables**

1. `m4_gallery_env_ggx.png` — 1×5 tiles under IBL only (no direct light).
2. `m4_dfg_lut.png` — 256×256 visualization (R=F_avg, G=vis term).
3. `m4_env_prefilter_levels.png` — contact sheet of mip levels (square grid).
4. `m4_meta.json` — sampler counts, cube sizes, RNG seed, min/max of LUT channels, per-tile mean luminance.

**Acceptance**

* **D1 seams:** per-face edge difference RMS < **1e-3** (CPU check on adjacent faces across all mips).
* **D2 roughness energy:** mean luminance **non-increasing** with roughness for identical F0 in IBL-only scene.
* **D3 LUT bounds:** DFG LUT values within **[0,1.05]**; log any overflow.
* **D4 determinism:** identical output hashes across **2 runs** with the same RNG seed.

**Tasks**

* IBL import + cube build; VNDF sampler; prefilter pass + mips; irradiance pass; LUT compute; CPU seam audit; gallery.

---

## M5 — **Color Management & Tone Mapping**  *(expanded, strict)*

**Goal**
Add **ACES Filmic** and **Reinhard** tone mapping toggles. Ensure **linear pipeline** remains verifiable (can bypass tonemap). Validate output ranges and avoid chroma shifts.

**Scope**

* **In:** three paths: **Linear (no tonemap)**, **Reinhard**, **ACES Filmic**.
* **Out:** Hable, HDR10.

**Reference curves**

* **Reinhard:** `C_out = C / (1 + C)` per channel in linear, then sRGB encode.
* **ACES** (Narkowicz optimized): apply matrix to ACES space → RRT+ODT approx → back to linear display; constants frozen in code with a unit test verifying the curve matches reference within **±1e-3** at 33 sample points.

**Deliverables**

1. `m5_tonemap_compare.png` — 3 columns (Linear/Reinhard/ACES) × 5 roughness rows under **direct+IBL** mixed.
2. `m5_meta.json` — white point, exposure=1.0, per-mode mean/median, and `max_linear` before tone map.

**Acceptance**

* **E1 bypass works:** Linear mode **bit-identical** to prior milestones (hash check vs regeneration when tonemap disabled).
* **E2 bounded output:** After tonemap, sRGB **[0,255]**; no channel clips (count ≤ **0.01%** pixels).
* **E3 monotone:** For each pixel channel, tone-mapped value is **non-decreasing** with the linear input (sampled LUT test).

**Tasks**

* Implement both curves; add mode flag; LUT test; comparison generator.

---

## M6 — **Validation Harness (CPU reference vs GPU)**  *(expanded, strict)*

**Goal**
Create a **CPU GGX reference** renderer at **32×32** sample points on the sphere (stratified over view hemisphere). Compare GPU shader outputs **within 1e-3 RMS** per channel in linear space for direct light.

**Scope**

* **In:** CPU reference for GGX dielectric (M1 math), direct light only, same inputs; difference maps and CSV.
* **Out:** Burley/IBL on CPU.

**Deliverables**

1. `m6_diff_heatmap.png` — false-color |GPU-CPU| per pixel for r=0.5 tile.
2. `m6_diff.csv` — rows: `(u,v, NL,NV,VH,NH, gpu_r,gpu_g,gpu_b, cpu_r,cpu_g,cpu_b, abs_err_r,...)`.
3. `m6_meta.json` — RMS error per channel, max error, seed.

**Acceptance**

* **F1 RMS:** ≤ **1e-3** per channel.
* **F2 Max abs error:** ≤ **5e-3** for **99.9%** pixels; list coordinates of worst offenders.
* **F3 Determinism:** two runs produce identical CSV hashes.

**Tasks**

* CPU implementation; GPU capture at matching points; diff metrics; heatmap; CI job that fails on threshold.

---

## M7 — **Extensions & Performance**  *(expanded, strict)*

**Goal**
Add **Clearcoat** and **Sheen (Charlie NDF)** to Principled; perform a basic perf pass. Maintain image correctness (hash or tolerance-check) and reduce GPU time by **≥15%** on gallery renders.

**Scope**

* **In:**

  * **Clearcoat:** secondary GGX lobe with fixed IOR=1.5, narrow roughness `r_c ∈ [0.03..0.2]`, energy-conserving mix (coat over base).
  * **Sheen (Charlie):** NDF Charlie for grazing retro-reflection; parameter `sheen_tint ∈ [0..1]`.
  * **Perf:** UBO → push-const where appropriate, state sorting, fewer barriers, renderpass merge.
* **Out:** Anisotropy, subsurface.

**Reference**

* **Charlie D:** `D_charlie(α, θh) = ((2 + 1/α) * pow(cos(θh), 1/α)) / (2π)` with `α = max(1e-3, roughness)`; use Disney’s form.
* **Coat Fresnel:** Schlick with `F0_coat = ((1.5-1)/(1.5+1))^2 ≈ 0.04`; blend `Lo = coat * mix + base * (1-mix)`; ensure **energy preservation** (coat absorbs base).

**Deliverables**

1. `m7_gallery_principled_extended.png` — grid: rows=roughness, cols={base, +coat, +sheen, +coat+sheen}.
2. `m7_perf_report.json` — timings (GPU/CPU), draw/dispatch counts, VRAM usage, and **baseline vs optimized** deltas.
3. `m7_meta.json` — clearcoat params, sheen params, energy audit (average luminance change vs M2 with features off).

**Acceptance**

* **G1 energy safe:** With coat+sheen off, **bit-identical** to M2 images (hash check).
* **G2 perf gain:** gallery total GPU time **≤ 0.85×** of baseline (median of 5 runs).
* **G3 feature bounds:** no pixel NaN/Inf; linear max ≤ **32**.

**Tasks**

* Implement features; unit tests for off==pass-through; perf instrumentation; optimization steps (constant folding, bindless where legal, merge passes).

---

## Conventions (applies to all milestones)

* **Tile/gutter:** tile **512×512**, **16 px** pure-black gutters; captions in top-left (white, drop shadow).
* **PNG:** **RGB8 only**; no alpha.
* **Meta JSON common keys:** `description`, `tile_size`, `roughness_values`, `base_color`, `f0` (or `f0_rgb`), `light`, `camera`, `tiles[ i ].{roughness,alpha,caption,mean_srgb_luminance}`, `hashes`.
* **Determinism:** Fixed RNG seed `"forge3d-seed-42"` wherever sampling exists; meta must include `rng_seed`.
* **Failure policy:** Generators **raise** if any acceptance check fails; write an `*_FAIL.txt` with the reason alongside partial outputs.

---

## Hand-off Checklist per Milestone

1. All **PNGs present**, correct size, gutters, captions.
2. **Meta JSON** present with all required fields.
3. **Console log** includes acceptance verdicts (True/False) and the exact measured values.
4. Re-running the generator with identical inputs produces **identical hashes** (except where sampling is enabled; then use seed).
