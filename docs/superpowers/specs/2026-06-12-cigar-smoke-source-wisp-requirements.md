# Cigar Smoke Source-Wisp Requirements

**Date:** 2026-06-12  
**Scope:** Requirements for improving `examples/california_cigar_smoke_demo.py` so `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4` better matches the source-attached smoke behavior in `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4`.

**2026-06-13 scope addendum:** the source-wisp implementation now meets the local cigar-smoke target, but the reference video is also a polished atmospheric dataviz film. Matching its perceived quality requires a second layer of requirements: full-frame map composition, time-series storytelling, multi-scale regional smoke, fire-point density, typography, grading, and delivery quality.

## Target Behavior

The target effect is not just denser or prettier atmospheric smoke. The reference reads as many thin, source-attached smoke wisps:

- Thin smoke emerges from every visible fire point or active fire-front segment.
- Fresh smoke initially rises from heat, curls near the source, then bends downwind.
- Wind direction is visible in screen space through coherent drift.
- Smoke expands, cools/desaturates, breaks into filaments, and disappears into thin air over time.
- Mid-age smoke should transform from narrow source stems into wider, thinner, semi-transparent plume ribbons with soft, disappearing ends.
- The final read should not be a compact set of opaque white brush strokes, even when those strokes are source-attached.
- Fire points flare, die, smolder briefly, and stop producing strong white smoke after burnout.
- Late frames should show smoke thinning over ash/burn scars rather than a persistent blanket.

At full film quality, the reference also reads as a continent-scale wildfire smoke data sequence:

- The camera is full-bleed and map-like, not a tilted terrain slab with visible model edges.
- Smoke has three simultaneous scales: source-attached wisps, mid-scale clustered plumes, and broad synoptic transport ribbons.
- Many small, sharp fire points appear across the map with localized halos and changing intensity.
- Date, burned-area value, active fire distribution, and smoke field all evolve together over time.
- Smoke color and opacity are mostly cool gray, translucent, and terrain-integrated, with warm lift only close to flame cores.
- The final encode preserves soft smoke, fire points, terrain relief, and type at 1080p-class delivery quality.

## Confidence

Parameter tuning alone is not sufficient. Raising `--physical-max-sources` or adjusting density alpha may add detail, but the current broad hybrid density and residual haze layers can still dominate the perception.

The likely sufficient path is a dedicated per-source wisp/particle layer, with broad atmospheric smoke kept as a secondary background layer.

## Current Implementation Status

Status as of the current `examples/california_cigar_smoke_demo.py` implementation and `examples/out/california_cigar_smoke/source_wisp_audit/` artifacts:

- Done: a dedicated `SourceWispSimulator` exists and tracks source-attached puffs with age, position, velocity, radius, alpha, color, intensity, and breakup state.
- Done: source wisps are behind CLI controls (`--source-wisps`, `--no-source-wisps`, `--source-wisp-max-particles`, `--source-wisp-max-emitters`, `--source-wisp-warmup-mode`).
- Done: wisps spawn from active/smoldering `HybridSmokeSource` lifecycle state, include a small source-to-smoke delay, and fade out by lifetime.
- Done: wisp advection uses the same procedural wind field family as the smoke system with age/altitude mixing, buoyancy, shear, curl, and tumbling.
- Done: the renderer has ablation modes for broad smoke, physical smoke, source wisps, no-broad, and combined output.
- Done: audit artifacts include component frame sheets, generated/reference encoded frame extraction, and JSON metrics at fixed timestamps.
- Done: `--render-preset source-wisp-reference` makes source wisps primary, uses fire-core emitters, keeps broad smoke at very low alpha, and disables physical smoke by default.
- Done: source emission can come from `active_fire_core_intensity_field()` and `fire_core_emitter_sources()`, which derive pre-bloom fire-core/front emitters instead of sampling bloom halos or final RGB.
- Done: the audit gate rejects broad/physical carpet smoke, low-frequency haze dominance, weak fire-core visibility, weak source attachment, and insufficient encoded strand detail.
- Done: `source_wisps_rgba()` now has a target plume-ribbon mode that keeps fresh stems narrow while adding a source-derived, low-alpha old-tail envelope with endpoint fade, edge erosion, and internal filaments.
- Done: `--source-wisp-plume-ribbons` / `--no-source-wisp-plume-ribbons` controls the plume stage, and `--render-preset source-wisp-brush-baseline` preserves the compact source-attached stroke output as a known-bad baseline.
- Done: the audit now measures fresh-stem, transition-plume, and old-tail morphology after final perspective warp, including width growth, old-tail coverage growth, endpoint alpha, edge softness, diffuse/core area, and brush-bundle score.
- Done: `source_wisp_audit.json` schema version `source-wisp-audit-v3` records morphology fields, age-band definitions, thresholds, regeneration commands, negative baselines, threshold calibration notes, and render-size/codec assumptions.
- Done: encoded-frame metrics now include `soft_tail_like_fraction` in addition to strand-like detail, so H.264 output has a gate for old-tail preservation rather than only PNG alpha masks.

Reference-film-quality status after inspecting the first 30 seconds of `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4` at 2-second spacing, plus full-resolution reference stills around 6s, 14s, and 24s:

- Done: the current render has a source-attached local smoke effect with visible old-tail broadening and encoded strand preservation.
- Done: the current render has fire-core emitters and separates glow/bloom from smoke emission.
- Done: the current audit proves the accepted 8-second local render passes its cigar-wisp gates.
- Done: `--render-preset reference-film` now selects a full-bleed `map-film` composition, 1920x1080 delivery profile, BT.709 metadata, regional smoke, dynamic date/area labels, map grade, and fine fire-point renderer.
- Done: film-mode audits now record `reference_film_frame_reports`, `reference_film_encoded_reports`, `reference_film_gate_report`, probed video stream metadata, and a first-30s generated/reference contact sheet.
- Done: a full 30-second 1920x1080 reference-film candidate now exists at `examples/out/california_cigar_smoke/reference_film_30s.mp4`, with audit artifacts under `examples/out/california_cigar_smoke/reference_film_30s_audit/`.
- Stale evidence: the previous full 30-second reference-film candidate passed its then-current automated film gates (`reference_film_gate_report.passed == true`, `failed_gate_count == 0`) and probed at about 2.50 Mbps H.264, but that artifact predates the latest map-crop, terrain-grade, and regional-smoke changes.
- Partial: the procedural regional smoke layer now uses soft stamped transport patches plus a faint flow underlay over a reframed regional California map extent, with smoke-coverage, soft-dense, texture, and axis-band gates. It is still synthetic, and the latest quick visual review still rejects it as not reference-quality because some regional patches read as isolated synthetic blobs with visible edges rather than natural broad smoke ribbons.
- Done: fire points are finer and source-derived in local mode; reference-film mode now adds disclosed synthetic distributed regional fire-context points plus gates for distributed clusters, occupied map cells, far-fire fraction, and primary-fire dominance.
- Partial: time-series storytelling is deterministic and synthetic for date and burned area; the audit records this real/synthetic policy, date-span and median-date-step gates, plus a real-data replacement schema, while real daily replacement inputs remain future work.
- Done: map-film mode removes slab edges and now uses a full-width regional California crop for reference-film mode instead of the local August Complex crop.
- Done: typography and map grading are implemented for film mode, and label contrast/overlap gates pass on encoded 1080p audit frames.
- Partial: designer-level reference-film sign-off is now specified through `reference_film_visual_signoff_contract` in audit JSON; actual human approval of a regenerated candidate remains pending.

## First-30s Every-Frame Reference Audit

**2026-06-15 audit scope:** inspected every frame in the first 30 seconds of `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4`, not only 2-second contact-sheet samples.

Evidence generated during the audit:

- Reference video metadata: 1920x1080 H.264, 30 fps, first 30 seconds = frames 0-899.
- Temporary all-frame extraction: `/private/tmp/cigar_ref_first30_every_frame/ref_0001.jpg` through `ref_0900.jpg`, scaled to 480 px wide for analysis.
- Temporary all-frame contact sheets and metrics: `/private/tmp/cigar_ref_first30_analysis/`.
- Current generated candidate comparison extraction: `/private/tmp/cigar_gen_first30_every_frame/`.
- Workspace summary artifacts retained under `examples/out/california_cigar_smoke/reference_first30_every_frame_audit/`: all-900-frame micro contact sheet, 0.5s reference contact sheet, 0.5s generated-candidate contact sheet, and `first30_frame_metrics.json`.

Measured first-30s reference behavior:

- The first 30 seconds are not a continuous local plume animation. They read as a rapidly changing 2023 North America wildfire/smoke time series.
- The map extent is continent-scale, including the western coast, western/northern interior, large lakes, and multiple active-fire regions. A regional California extent cannot match the reference first read.
- The top-right date advances through 2023 dates, and the lower-left burned-area statistic grows from about `5.9 k ha` to about `12.3 M ha` by 29.5s. The current 2020 August Complex synthetic timeline is not the reference timeline.
- Smoke appears in daily/event-like bursts. Large structures form, curl, advect, detach, and are replaced by different structures over short intervals rather than persisting as one smooth procedural field.
- Reference frame-difference spikes cluster around approximately 5-7s, 13-15s, 24-26s, and 27s. These are visible plume-state changes, not just camera or label changes.
- Reference smoke coverage is highly variable: measured smoke-mask median about `0.062`, p90 about `0.131`, and max about `0.214` over the 900-frame analysis extraction.
- The current generated reference-film candidate is too continuously smoky by comparison: measured median smoke-mask coverage about `0.116`, roughly `1.9x` the reference median in the same analysis.
- The current generated candidate is too temporally smooth: reference p90 frame-difference was about `1.94`, while the generated candidate was about `0.90` in the same analysis.
- The strongest reference smoke events are not all source-attached cigar wisps. Many are broad satellite-like smoke masks with embedded sharper plumes, curls, arcs, and dissipating sheets.
- Fire points and smoke are correlated but not one-to-one. Many fire points do not emit visible source wisps every frame, and some broad smoke masses reflect transported smoke away from visible fire points.
- The reference smoke has bright, high-opacity white/gray pulses during peak events, but those pulses are localized and temporary. The plan should not require all smoke to remain uniformly low-alpha if the target is full reference-film match.
- The generated candidate's regional layer still reads as contour/isoline bands and persistent rings in the all-frame sheet. Existing axis-band/texture gates are not enough to reject this failure.

Missing implementation-plan requirements found by the all-frame audit:

1. Add a separate **reference-2023 continent mode** rather than treating `reference-film` as a California/August Complex variant. It needs the reference geography, date range, active-fire distribution, area scale, and label semantics.
2. Add a **daily/keyframed smoke-state driver**. The reference behaves like a sequence of observed daily smoke fields or keyframed satellite-smoke masks with interpolation, not like only a continuous advective simulation.
3. Add an **observed-smoke replacement path as a first-class requirement**, not just a future optional replacement. Acceptable inputs should include NOAA HMS smoke polygons, HRRR-Smoke rasters, satellite aerosol/smoke products, or a curated reference-derived mask sequence.
4. Add **per-event plume primitives** for broad smoke: independent plumes with their own birth/death times, wind direction, curl/roll-up, opacity envelope, and breakup. A single procedural regional texture field is insufficient.
5. Add **temporal reset/change gates** over all first-30s frames: frame-difference spike count, smoke-coverage peak/trough cadence, centroid jump range, and maximum persistence of any single regional smoke component.
6. Add **coverage-envelope gates** that match the reference's intermittent smoke: median coverage, p90 coverage, max coverage, and low-smoke intervals. The current film gates allow too much always-on haze.
7. Add **anti-contour/ring gates** for synthetic regional smoke: reject alpha isolines, nested contour bands, visible ring boundaries, and smooth topographic-looking level sets.
8. Add **multi-geography active-fire gates**: active fire clusters must appear and disappear across a continent-scale field, not only as a primary local cluster plus a few synthetic regional points.
9. Add **fire/smoke decoupling rules** for film mode: source-attached wisps remain valuable near active fire clusters, but regional smoke should be allowed to originate from transported/observed smoke masks and should not require visible attachment to every fire point.
10. Add an **every-frame visual sign-off artifact** for reference-film acceptance: all 900 first-30s frames as a micro contact sheet, plus larger 0.5s or 0.2s sheets and frame-difference spike frames. The current fixed 2s/still review is too sparse to catch temporal smoothness and contour persistence failures.

## Completion Matrix

| Area | Status | Notes |
| --- | --- | --- |
| Source-wisp particle system | Done | `SourceWispSimulator` tracks puff state, lifecycle, source attachment, wind advection, curl, shear, radius, alpha, color, and breakup. |
| CLI and presets | Done | `source-wisp-reference` is the target preset and records layer ownership through explicit broad, physical, and wisp settings. |
| Fire-core emitter extraction | Done | Target preset can emit from pre-bloom active fire-core/front intensity rather than glow, bloom, or final RGB. |
| Broad smoke suppression | Done | Target preset keeps broad smoke as faint context and disables physical smoke by default. |
| Ablation artifacts | Done | Audit artifacts include broad-only, physical-only, source-wisps-only, no-broad, combined, generated/reference frame sheets, and encoded frames. |
| Existing hard gates | Done | Current gates cover source attachment, screen attachment, fire visibility, broad carpet rejection, low-frequency haze, combined retention, and encoded strand presence. |
| Wisp visual vocabulary | Done | Target mode renders source-attached stems plus wider, softer old-tail plume ribbons; the brush-like source-attached output is retained only as a negative baseline. |
| Plume transformation | Done | Fresh, transition, and old-tail age bands are rendered and measured after perspective warp; gates enforce width growth, coverage growth, endpoint fade, softness, and brush rejection. |
| Morphology audit schema | Done | `source_wisp_audit.json` schema `source-wisp-audit-v3` records morphology fields, thresholds, age bands, encoded soft-tail metrics, and regeneration commands. |
| Brush-bundle baseline | Done | `--render-preset source-wisp-brush-baseline` disables plume ribbons and should fail the morphology gates while preserving source attachment. |
| Diffuse plume/ribbon implementation policy | Done | The old-tail plume stage is implemented inside `source_wisps_rgba()` as a source-derived, age-limited, low-alpha envelope, not as broad regional haze or physical smoke. |
| Human morphology review | Done | The audit records review order and requires human review of `source-wisps-only`, `no-broad`, and `combined`; automated gates are necessary but not sufficient for final visual sign-off. |
| Local cigar-smoke effect | Done | The accepted render is a strong local source-wisp implementation. |
| Full-bleed map-film mode | Done | Implemented as `--composition-mode map-film` and selected by `--render-preset reference-film`; uses a regional California crop with extent metadata recorded in audit JSON. |
| Time-series fire/smoke story | Partial | Film mode animates date and burned area through a deterministic synthetic driver and records the real/synthetic policy; date-span and median-date-step gates exist. Real daily fire/smoke inputs remain future replacement work. |
| Regional smoke hierarchy | Partial | `regional_transport_smoke_rgba()` adds a separate procedural regional layer, now based on soft transported stamps plus a faint underlay; automated gates pass in focused tests, but designer review still rejects the current quick preview for isolated blob/edge artifacts. Remaining work is a more natural regional-smoke primitive or observed-smoke replacement plus human approval. |
| Fire-point visual density | Done for synthetic film target | `reference_fire_points_rgba()` renders small source-derived local points and, in `reference-film`, disclosed synthetic distributed regional context points. Audit covers active cores, hot fraction, visibility, radius, halo/core ratio, distributed clusters, grid spread, far-fire fraction, and primary dominance. |
| Film-grade typography and layout | Done for gate-level delivery | Film mode has a restrained data/source label, date, dynamic area statistic, dark map grade, and encoded label contrast/overlap gates. Final art-direction sign-off remains part of reference-film review. |
| 1080p delivery quality | Partial | `reference-1080p` profile defines the 30s 1920x1080 H.264/BT.709 delivery target. A previous stale candidate probed at about 2.50 Mbps; a current full render is still required after the latest visual changes. |
| Regional visual audit gates | Partial | `reference_film_gate_report` checks full-bleed coverage, smoke coverage, regional density, mid-scale smoke, smoke centroid motion, temporal span/date-step cadence, active-fire temporal change, fire density/visibility/radius/halo ratio, distributed-fire geography, smoke texture/axis-band artifacts, label contrast/overlap, encoded smoke, and delivery metadata. Current human review shows these gates still do not fully catch isolated blob/ring-edge artifacts. |
| Reference-2023 continent target | Missing | The first-30s reference is a 2023 North America smoke/fire sequence with area growing to about 12.3M ha, not a California/August Complex 2020 sequence. A matching mode needs the correct geography, timeline, active-fire distribution, and label semantics. |
| Daily/keyframed smoke-state driver | Missing | Every-frame review shows event-like daily plume-state changes and resets. A continuous procedural field alone does not match the reference cadence. |
| Observed-smoke primitive | Missing | Regional smoke should be driven by observed or observed-like smoke masks/polygons/rasters, with procedural simulation used for interpolation and texture rather than as the sole source of truth. |
| Every-frame temporal audit gates | Missing | Current gates sample sparse fixed times. Reference-film acceptance needs all-frame checks for frame-difference spikes, smoke-coverage peaks/troughs, component persistence, and cadence. |
| Anti-contour/ring regional-smoke gates | Missing | Current generated smoke can pass automated gates while reading as contour/isoline bands. Add explicit rejection for nested bands, ring edges, and persistent smooth level-set artifacts. |

## Spec Completeness Checklist

This checklist separates completed implementation work from items that are only specified, and from items that still need to be specified before the next implementation pass.

| Item | Spec status | Implementation status | Next doc action |
| --- | --- | --- | --- |
| Source-attached particle layer | Complete | Done | Keep current description unless the implementation path changes. |
| CLI controls and target preset | Complete | Done | Keep `source-wisp-reference` as the accepted target preset. |
| Pre-bloom fire-core emitter source | Complete | Done | Keep explicit prohibition on bloom/glow/final-RGB emission. |
| Broad/physical layer ownership | Complete | Done for target preset | Keep physical smoke optional until it passes morphology and haze gates. |
| Existing audit artifacts | Complete | Done | Keep artifact contract and exact-command recording. |
| Existing source/haze gates | Complete | Done | Keep thresholds for source attachment, haze, carpet rejection, fire visibility, retention, and encoded strands. |
| Plume age stages | Complete | Done | Age bands: fresh stem 0.00-0.26, transition plume 0.24-0.62, old tail 0.56-1.01 of puff lifetime. |
| Width growth and tail fade | Complete | Done | Hard gates cover transition width growth, old-tail width growth, endpoint alpha fraction, old-tail coverage growth, and encoded soft-tail presence. |
| Edge softness and erosion | Complete | Done | Audit records old-tail edge softness, diffuse/core area ratio, and hard endpoint fraction; hard gates enforce minimum softness and diffuse envelope. |
| Brush-bundle rejection | Complete | Done | `brush_bundle_score` combines old-tail coverage growth, width growth, and endpoint fade; the brush-baseline preset is expected to fail it. |
| Diffuse plume/ribbon pass | Complete | Done | Implemented as an old-puff envelope inside `source_wisps_rgba()` behind `--source-wisp-plume-ribbons`. |
| Smoke-region alpha stats | Complete | Done | The audit records smoke-region alpha p50/p90/p95 inside detected smoke pixels, plus age-band alpha percentiles. |
| Reference measurement values | Complete for local source-wisp audit | Done | Thresholds are calibrated behaviorally against extracted reference/audit frames and separated from known-bad carpet and brush baselines; full-frame film composition is now covered by the 2026-06-13 addendum. |
| Morphology audit JSON schema | Complete | Done | Schema v3 names units, age bands, hard thresholds, morphology field meanings, and examples can be derived from passing and brush-baseline audit payloads. |
| Regeneration commands | Complete | Done | Audit JSON records commands for accepted render, source-wisps-only, no-broad, carpet baseline, brush baseline, and reference frame extraction. |
| Visual sign-off record | Complete | Done | The accepted audit artifacts plus review order are the sign-off record; humans review the three ablations after automated gates pass. |
| Reference first-30s inspection | Complete | Done | A 2026-06-13 review inspected first-30s contact sheets and stills; it found that source-wisp quality is not the only reference-quality gap. |
| Full-frame map composition spec | Complete | Done | Full-bleed/no-slab map-film mode uses the regional California extent and records bounds/fire UV in audit JSON. |
| Temporal dataviz narrative spec | Complete for synthetic target | Partial | Synthetic date/burned-area driver, date-span gate, median-date-step gate, and real-data replacement schema exist. Real daily data ingestion remains future work. |
| Multi-scale regional smoke spec | Complete for current gates, incomplete for visual rejection | Partial | Regional layer exists, is separate from source wisps, and has coverage/density/texture/axis-band gates. The doc now explicitly records the remaining failed criteria: isolated synthetic blobs, visible ring/edge artifacts, and insufficient natural broad-ribbon continuity. Observed-smoke replacement remains future work. |
| Fire mark design spec | Complete for current gates | Done for synthetic target | Fine source-derived fire points and synthetic regional context points exist; visibility, median radius, halo-core, distributed-cluster, grid-spread, far-fire, and primary-dominance gates are implemented. |
| Film-grade output spec | Complete for technical output | Done for technical output | 1080p bitrate/color profile, 30s candidate render, encoded audit, and first-30s contact sheet exist; human visual approval remains a separate acceptance step. |
| Typography and annotation spec | Complete for legibility gates | Done for legibility gates | Film labels, dynamic area/date, and label contrast/overlap gates are implemented and pass; final hierarchy can still be tuned with the reference-film visual pass. |
| Regional audit metrics | Partial | Partial | `reference_film_gate_report` covers coverage, regional smoke, mid-scale smoke, smoke centroid motion, temporal change, active-fire temporal change, fire-core density, post-smoke fire visibility, mark radius, halo/core ratio, label contrast/overlap, encoded smoke, and delivery. The latest code passes focused unit tests, but no current 30-second 1080p gated render exists after the latest tuning, and the current quick preview still fails designer sign-off. |
| Reference-2023 continent mode | Incomplete | Missing | Specify North America map extent, 2023 date/area semantics, active-fire source data, and target labels before implementation. |
| Daily/keyframed smoke-state driver | Incomplete | Missing | Specify whether the first implementation uses observed smoke products, curated masks, or reference-derived masks; define interpolation and reset cadence. |
| Every-frame temporal acceptance | Incomplete | Missing | Promote first-30s all-frame extraction, micro contact sheet, frame-difference spikes, coverage-envelope gates, and component-persistence gates into the audit contract. |
| Regional anti-contour acceptance | Incomplete | Missing | Define metrics for contour/isoline bands, ring edges, and smooth level-set artifacts that current texture/axis-band gates do not catch. |

## Doc Authoring Status

Already covered in this document:

- Done: target behavior now distinguishes source-attached smoke, plume widening, thin old-tail disappearance, and brush-bundle failure.
- Done: current implementation status identifies completed source-wisp plumbing, fire-core emitters, preset ownership, ablations, and existing audit gates.
- Done: layer policy makes source wisps primary, broad smoke faint/contextual, and physical smoke optional until it can pass the same gates.
- Done: plume morphology requirements describe fresh stems, transition plumes, old tails, edge erosion, endpoint fade, and diffuse-envelope constraints.
- Done: acceptance requirements include source attachment, carpet rejection, low-frequency haze, fire visibility, encoded strand preservation, width growth, tail fade, edge softness, and brush-bundle rejection.
- Done: morphology gates now cover the previous wide/thin disappearing-tail failure mode, while the ordered ablation review remains required for final human approval.

Implemented in this pass:

- Done: `source_wisps_rgba()` now includes a plume-ribbon mode that expands old puffs into a source-derived, low-alpha envelope with endpoint fade, edge erosion, internal filaments, and reduced centerline dominance.
- Done: morphology audit metrics are computed after `warp_map_layer_to_plate()` from three age-band renders: fresh stem, transition plume, and old tail.
- Done: hard morphology gates cover stage coverage, transition width growth, old-tail width growth, old-tail coverage growth, old-tail endpoint alpha, old-tail edge softness, old-tail diffuse/core ratio, brush-bundle score, and encoded soft-tail preservation.
- Done: `source_wisp_audit.json` schema version `source-wisp-audit-v3` records field meanings, units, age bands, thresholds, regeneration commands, negative baselines, threshold calibration, and render-size/codec assumptions.
- Done: local source-wisp reference alignment is behavior-based, not full-frame composition matching. Fixed audit times remain 1.0s, 3.5s, 5.5s, and 7.0s, with reference/generated frame extraction retained for visual review. The 2026-06-13 film-quality addendum separately covers full-frame composition matching.
- Done: known-bad baselines are regenerated on demand under `examples/out`: `legacy-combined` for carpet smoke and `source-wisp-brush-baseline` for source-attached brush bundles.
- Done: visual sign-off is recorded by retaining the accepted audit JSON and ordered ablation artifacts; the automated gates are a required precondition, not a replacement for human review.

Morphology threshold table:

The 1.0s audit frame validates source attachment, haze rejection, and age-band coverage while the clip is still fresh-stem dominated. Transition/old-tail plume-transformation gates apply from 2.0s onward.

| Metric | Hard threshold | Unit / meaning |
| --- | --- | --- |
| `morphology_stage_coverage_fraction` | >= 0.00012 | Minimum full-frame coverage across fresh, transition, and old-tail age-band masks. |
| `transition_width_growth_ratio` | >= 1.12 | Transition plume screen width divided by fresh-stem width. |
| `old_tail_width_growth_ratio` | >= 1.42 | Old-tail screen width divided by fresh-stem width. |
| `old_tail_alpha_p90_fraction` | <= 1.55 | Old-tail p90 alpha divided by fresh-stem p90 alpha; high overlap is allowed but tracked. |
| `old_tail_endpoint_alpha_fraction` | <= 0.44 | Downwind endpoint alpha divided by fresh-stem p90 alpha. |
| `old_tail_coverage_growth_ratio` | >= 2.50 | Old-tail smoke-mask coverage divided by fresh-stem coverage. |
| `old_tail_edge_softness_px` | >= 1.05 | Screen-pixel diffuse edge width proxy. |
| `old_tail_diffuse_to_core_area_ratio` | >= 1.10 | Low-alpha old-tail diffuse pixels divided by high-alpha core pixels. |
| `brush_bundle_score` | <= 0.44 | Combined narrowness, weak old-tail area, and endpoint opacity score; lower is better. |
| `encoded_soft_tail_like_fraction` | >= 0.0025 | H.264 frame fraction with low-saturation, low-gradient soft-tail smoke signal. |

## Source Requirements

- Emit smoke from the actual active fire signal, not only the current random `HybridSmokeSource` list.
- Prefer true hot cores or active fire-front pixels as emitters.
- Do not sample from bloomed fire or glow halos, because that would emit smoke from visual bloom instead of fire.
- Every visible fire point, or a controlled sampling of active hot pixels, should be eligible to emit smoke.
- Source emission should be proportional to active flame intensity, not just source existence.
- The fire source field should evolve over time: new hot points appear, active points move or flicker, old points burn out.
- Static emitters with staggered lifetimes are not enough if the reference fire front visibly changes.
- Fire and smoke must share lifecycle state:
  - active flame emits fresh smoke
  - recently expired flame emits reduced smolder smoke
  - old burn scars emit little or no white smoke
- Smoke should trail flame flicker with a small source-to-smoke delay instead of appearing as a static halo.
- Warmup behavior must be controlled. The final video should not begin with only old smoke already present unless that is intentional.

## Wisp Simulation Requirements

- Add a separate source-attached wisp or particle layer.
- Spawn small puffs from active fire points every few frames.
- Advect puffs using the same wind field as the smoke system, with additional local curl/noise.
- Include near-source buoyancy: smoke should rise/curl first, then bend downwind.
- Include vertical or age shear: fresh low smoke and older higher smoke may drift differently.
- Add source-local vorticity/tumbling so the first few frames read like cigar smoke rather than straight painted streaks.
- Maintain temporal continuity or motion blur so wisps do not pop between frames.
- Each puff/wisp should track age, position, radius, alpha, color, and breakup state.
- Radius should grow with age while alpha and contrast decrease.
- Color should transition from warm/milky near active flame to gray or blue-gray, then transparent.
- Dissipation must include fragmentation/noise erosion, not only uniform alpha fade.
- Old smoke should actually dissipate or be removed from simulation, not merely hidden by low alpha.
- Wisp lifetime should be short enough to disappear within the visible 8 second clip unless intentionally retained as background haze.

## Plume Morphology Requirements

The reference contains a good kind of broadness: source-derived smoke widens and thins into soft plume ribbons before disappearing. This is different from the rejected regional carpet/haze layer.

Current status:

- Done: individual puffs track age, radius, alpha, color, velocity, source identity, and breakup state.
- Done: the renderer computes anisotropic strand length and width from puff age and motion.
- Done: current rendering includes procedural striation, erosion noise, color cooling, and alpha fade.
- Done: target mode creates a staged transformation from fresh source stems to wider, softer, lower-contrast old-tail plume ribbons.
- Done: the audit measures age-stage morphology after the final perspective warp and rejects source-attached brush-bundle morphology separately from carpet-smoke failure.

Required age stages:

- Fresh source stem: near the flame, smoke starts narrow, bright/milky, source-attached, and visibly lifted by heat before bending downwind.
- Transition plume: after the source stem, smoke must widen laterally, lose contrast, and develop soft internal gaps and filament breakup.
- Old tail: farthest downwind smoke must be wider, thinner, cooler, lower contrast, and visibly disappearing at both side edges and the downwind end.
- Expired/smolder stage: old burn scars may retain faint gray smoke but should not continue producing strong white stems.

Concrete morphology requirements:

- Width must grow with age in screen space, not only in map space. A mid/old plume segment should be measurably wider than the fresh source stem.
- Alpha and contrast must fall as width grows. Wider old smoke should be thinner, not a fatter opaque stroke.
- Downwind endpoints must fade out smoothly. Hard clipped white tips, triangular ends, or uniformly bright stroke caps should fail review.
- Edges must erode with noise and softness. The plume edge should break into translucent holes and wisps rather than a sharp painted boundary.
- Internal structure should contain several semi-transparent filaments embedded in a faint diffuse envelope, not only a few high-contrast centerlines.
- The layer may include a controlled diffuse plume/ribbon envelope if it is source-derived, age-limited, low-alpha, and passes carpet/haze rejection.
- Brush-bundle failure must be treated separately from carpet-smoke failure: a render can pass blanket rejection and still fail if all smoke remains narrow, opaque, and stroke-like.
- The final encoded MP4 must preserve both thin source stems and soft old-tail disappearance; validating only PNG alpha masks is not enough.

## Broad Smoke And Haze Requirements

- Split the smoke into two roles:
  - broad atmospheric/background plume
  - foreground source-attached wisps
- Keep the broad HRRR/guidance/residual haze layer secondary.
- Reduce broad haze alpha/feed enough that it does not mask the source wisps.
- Avoid making one raster density field responsible for both regional haze and source-attached cigar smoke.
- The broad background plume may remain for August Complex context, but it should not dominate the visual read.
- Do not ban all wide smoke. Wide smoke is acceptable only when it is source-derived, age-related, semi-transparent, and visibly dissipating at its edges and downwind ends.

## Full-Frame Map Film Requirements

The first 30 seconds of the reference video read as a designed map film, not as a local terrain-object render. A reference-quality mode should be separate from the current oblique terrain demo if preserving the 3D slab view remains useful.

Current status:

- Done: the current render has a coherent 3D terrain plate, terrain relief, fire marks, smoke layers, labels, and MP4 output.
- Done: `map_film_plate()` and `--composition-mode map-film` provide a full-bleed, no-slab composition for `--render-preset reference-film`.
- Done: the current map-film extent uses a full-width regional California crop from the cached `max1700` terrain assets, with bounds recorded in audit JSON.

Requirements:

- Add a full-bleed map-film render mode with no visible terrain-slab side walls, black slab edges, or large empty 3D stage area.
- Favor a stable orthographic or near-orthographic camera for the reference-match mode. Perspective depth is acceptable only if it does not call attention to the terrain as a physical block.
- Compose the region so fire points, smoke transport, coastline/terrain relief, and labels all share the first read. The smoke should not be confined to a small local patch in a large low-information frame.
- Preserve terrain relief and coastline/region context, but keep them subordinate to fire and smoke.
- The map extent should support regional smoke travel. A local August Complex crop can remain as a separate showcase, but the reference-quality target needs enough geographic room for broad transport ribbons.
- Avoid UI-like overlay clutter. Labels should feel like part of the film grade, not explanatory scaffolding.

## Temporal Dataviz Story Requirements

The reference's perceived quality comes from seeing an evolving event, not only a better smoke shader.

Current status:

- Done: fire sources have lifecycle state, flicker, burnout, smolder, and source-to-smoke delay.
- Done: `reference_film_frame_info()` animates displayed date and burned-area values across the film-mode sequence.
- Done: the reference-film audit records the synthetic/observed input policy, including a visible disclosure label and replacement policy.
- Done: reference-film gates now enforce temporal date span and median date-step cadence so contact-sheet rows read as event/day steps.
- Partial: active fire and smoke fields evolve procedurally, but the time series is not yet backed by real daily fire/smoke inputs.

Requirements:

- Animate the displayed date and burned-area value from a time series rather than keeping `418 k ha` fixed for the whole clip.
- Active fire points should appear, intensify, spread, decay, and disappear over the sequence.
- Smoke should reflect accumulated transport from prior frames/days, not only instant local plume motion.
- The story cadence should match the reference: each visible moment should imply a new day or event step, not merely a camera-stationary seconds-long plume simulation.
- Add a data contract for time-series inputs. If real daily sources are unavailable, define a deterministic synthetic daily series with explicit labels saying what is synthetic versus observed.

## Multi-Scale Regional Smoke Requirements

The current source-wisp layer solves the local cigar-smoke read. The reference also needs broad atmospheric ribbons that travel across large map distances while remaining visually designed and not becoming a rejected haze blanket.

Current status:

- Done: source wisps now provide local source-attached stems and soft old tails.
- Done: `regional_transport_smoke_rgba()` adds a separate procedural regional transport layer with broad ribbons, holes, edge fade, and directional flow.
- Done: regional smoke now has separate film-mode gates, first-30s generated/reference contact sheets, and a real-versus-synthetic data policy in audit JSON.
- Done: regional smoke now has texture and axis-band artifact gates to reject overly smooth synthetic ribbons and rectangular/axis-aligned banding.
- Partial: the latest procedural approach reduces the earlier full-frame contour-sheet artifact but still fails visual review when regional smoke appears as isolated stamped blobs or ring-edged patches rather than naturally connected transport smoke.
- Partial: observed-smoke replacement remains future work.

Requirements:

- Represent at least three simultaneous smoke scales:
  - source-attached wisps from visible hot cores
  - mid-scale cluster plumes around fire complexes
  - broad synoptic ribbons crossing the map
- Broad ribbons must have coherent direction, curl, thinning, holes, and soft edges. They should read as transported smoke, not a semitransparent blanket.
- Broad smoke should be mostly gray/blue-gray and lower contrast than source-adjacent smoke.
- Regional ribbons may be large and continuous, but they need internal density variation, edge breakup, and visible flow lanes.
- Add layer ownership for regional smoke separately from the source-wisp target preset. The previous carpet/haze rejection remains valid for local source-wisp acceptance, but a new regional mode needs gates that allow wide transport while rejecting flat low-frequency cover.
- Compare source-wisp-only, regional-smoke-only, and combined-regional outputs. The regional layer should add scale and atmosphere without erasing source hot points.

## Fire Mark Design Requirements

The reference fire points are numerous, small, sharp, and bright, with localized bloom. The current render's fire points are source-derived but visually too clustered and patch-like for reference-level dataviz.

Current status:

- Done: fire-core emitter extraction avoids bloom/final-RGB smoke emission.
- Done: active sources flicker and feed smoke.
- Done: `reference_fire_points_rgba()` renders smaller source-derived fire points with tight halos for map-film mode.
- Done: audit coverage measures active fire-core pixels, active-fire temporal change, hot-fire fraction, post-smoke fire visibility, median mark radius, and halo/core area ratio on encoded 1080p frames.
- Done: reference-film mode now adds disclosed synthetic distributed regional context fire points and gates for distributed clusters, occupied map grid cells, far-fire fraction, and primary-fire dominance.

Requirements:

- Render many small active fire points with high-contrast white/yellow cores, orange halos, and occasional tiny red ember points.
- Keep bloom local and tight. Glow should support visibility, not become a large orange patch.
- Fire-point density should vary over time and across regions.
- Fire marks should remain visible through smoke in the final composite.
- Add audit metrics for active fire-point count, median point radius, halo/core radius ratio, and fire visibility after smoke compositing.

## Color, Lighting, And Grading Requirements

Current status:

- Done: source wisps cool from warm/milky smoke toward gray/blue-gray and the compositor adds nearby fire lift.
- Done: `_reference_map_grade()` adds a dark full-scene map grade with cool terrain, west-side glow, and vignette for map-film mode.
- Done: the generated reference-film candidate now has a full 30-second 1080p contact sheet against the reference.
- Done: regional smoke texture and axis-band gates now reject the most obvious field-boundary artifacts.
- Partial: final human visual review still rejects the current quick preview for synthetic regional-smoke blob/edge artifacts; density, continuity, terrain integration, and gray/blue-gray tuning remain open before a regenerated candidate can be approved.

Requirements:

- Reduce the dominant white plume look in the final film-quality mode. Smoke should mostly sit in translucent gray, with bright white only in small dense/source-adjacent regions.
- Add subtle blue-gray shadowing and terrain integration so smoke appears embedded in the map atmosphere.
- Use warm fire illumination only near hot cores; do not warm large smoke regions.
- Apply a consistent map-grade pass: dark terrain, readable relief, restrained highlight rolloff, and light atmospheric vignette.
- Validate that smoke remains legible over both dark land and brighter coastal/ocean/background regions.

## Typography And Annotation Requirements

Current status:

- Done: the current render has readable source labels and burned-area text.
- Done: map-film mode uses smaller source labels, a top-right date, and a dynamic burned-area statistic.
- Done: label contrast, label text fraction, and smoke/fire overlap checks are implemented and pass on full-resolution encoded frames.
- Partial: final hierarchy and placement can still be tuned once the regional event composition is changed.

Requirements:

- Use a restrained annotation system: small data/source label, date label, and one clear burned-area statistic.
- Keep type outside the main smoke/action path where possible.
- Scale type for 1920x1080 delivery first, then downsample if needed.
- The burned-area number should change over time if the date changes.
- Add a label-legibility check on representative encoded frames.

## Delivery Quality Requirements

Current status:

- Done: the current accepted render encodes to H.264 and audits encoded strand/soft-tail preservation.
- Done: color metadata and encoded detail are handled for the 960x540 8-second local render.
- Done: `reference-1080p` delivery profile sets 1920x1080, H.264 bitrate controls around `2600k`, and BT.709 metadata for `--render-preset reference-film`.
- Stale evidence: a previous full 30-second 1920x1080 candidate was rendered at `examples/out/california_cigar_smoke/reference_film_30s.mp4`; that artifact predates the latest regional smoke and map-crop changes.
- Missing: regenerate the full 30-second 1920x1080 candidate from the current code, then obtain human visual approval against the first 30 seconds of the reference.

Requirements:

- Add a render/export profile for 1920x1080 at 30 fps.
- Target a bitrate/CRF class comparable to the reference. The inspected reference is 1920x1080 at about 2.55 Mbps; the previous stale reference-film candidate was 1920x1080 at about 2.50 Mbps.
- Preserve BT.709 color metadata through output.
- Validate final encoded frames, not only source PNGs.
- Include a contact sheet for the first 30 seconds of the reference and the matched generated sequence when evaluating film-quality changes.

## Reference-Film Candidate Review

The previous reference-film candidate is a valid technical milestone, not final art approval. A current candidate has not yet been regenerated after the latest regional-smoke and map-crop changes.

- Stale evidence: candidate MP4 exists at `examples/out/california_cigar_smoke/reference_film_30s.mp4`, but it was generated before the latest regional smoke, terrain-grade, and crop changes.
- Stale evidence: audit JSON exists at `examples/out/california_cigar_smoke/reference_film_30s_audit/source_wisp_audit.json`, but it does not prove the current code.
- Stale evidence: first-30s reference/generated contact sheet exists at `examples/out/california_cigar_smoke/reference_film_30s_audit/reference_film_first_30s_contact_sheet.png`, but it shows the old contour-band failure mode and is not a current candidate.
- Done: the previous automated film gates passed with `reference_film_gate_report.passed == true`; the current code adds/revises stricter gates and requires a regenerated audit before final approval.
- Stale evidence: the previous probed stream was 1920x1080 H.264 at about 2.50 Mbps; the current code still needs a fresh probed full render.
- Done: the current code adds distributed regional fire-context points and stricter gates for geographic spread, smoke naturalism, and date-step cadence.
- Done: the human visual sign-off rubric is serialized as `reference_film_visual_signoff_contract`.
- Missing: regenerate the full 30-second candidate under the current gates, inspect the updated contact sheet, and record human visual approval or the next failed criteria.
- Current quick-preview failure: `/tmp/reference_film_quick_1fps.preview.png` from 2026-06-14 shows strong local source wisps and improved crop/terrain framing, but regional smoke still reads as isolated synthetic blob patches with visible edges. Do not treat the current reference-film mode as visually approved until this is fixed.

## Layer Ownership Requirements

The target render needs explicit ownership rules so broad smoke cannot quietly re-enter as the main effect:

- The source-wisp layer must be the primary readable smoke layer in the target render.
- The source-wisps-only ablation must pass visual review before broad or physical smoke is added back.
- The combined render must not look less source-attached than the source-wisps-only ablation.
- Broad HRRR/guidance/residual haze should be disabled, near-transparent, or clipped to background context until the wisp layer carries the effect.
- The physical 3D layer should be disabled or capped unless its ablation remains source-attached and does not form a continuous sheet.
- The final target preset should make the intended layer weights explicit rather than relying on current defaults.
- Any layer that produces a large connected low-frequency smoke mass should be treated as failing the cigar-wisp target, even if it contains useful detail.
- Fire cores should remain visible through fresh wisps; smoke may pass in front of flame, but broad opacity must not bury active fire points.

## Physical Smoke Path Requirements

- The existing physical 3D smoke path is closer to source-attached smoke than the 2.5D blanket, but it currently selects only a capped subset of sources.
- Increase or adapt physical source selection so all important active fire points can contribute.
- Consider a higher cap than the current default when rendering the final target video.
- Narrow physical emitter radius where needed so smoke starts as thin wisps.
- Use enough physical resolution and substeps to preserve thin structures after projection.
- The physical layer should be promoted visually only if it remains source-attached and does not become another broad veil.

## Projection And Screen-Space Requirements

- Wisps must remain attached to fire points after `warp_map_layer_to_plate()`.
- Wind direction must be checked after perspective warp; correct map-space wind can look wrong in final screen space.
- Wisp width must be calibrated after perspective warp, not just in map space.
- Wisps need a screen-space minimum thickness/contrast so they survive resizing and video encoding.
- Near/far perspective may require different source widths or alpha compensation.
- Smoke should feel attached to terrain and fire, not pasted on top as a flat overlay.
- Depth/occlusion cues should be preserved where possible.

## Compositing Requirements

- Fresh wisps should be partly in front of fire but also lit by active fire.
- Layer ordering must avoid burying flames under opaque smoke.
- Fire glow should illuminate nearby fresh smoke without turning glow halos into smoke emitters.
- Use correct premultiplied alpha handling for thin smoke edges.
- Validate color-space behavior so thin smoke does not turn muddy or vanish.
- Keep enough local contrast for thin strands while avoiding harsh artificial outlines.

## Encoding Requirements

- Validate the final encoded MP4, not only intermediate PNG/RGBA frames.
- H.264 `yuv420p` can erase thin alpha-derived details; tune against the encoded video.
- If wisps vanish after encoding, consider higher source resolution, slightly larger minimum width, stronger local alpha/contrast, less blur, or encoding changes.
- Compression softness, light footage noise, subtle motion blur, and contrast rolloff may be needed for the synthetic render to read closer to the reference.

## Reference Audit Requirements

Before locking implementation details, extract comparable frames from both videos and compare them side by side.

Suggested frame times:

- early: about 1s
- mid: about 3s to 4s
- late-mid: about 5s to 6s
- late: about 7s

Measure or visually score:

- number of smoke wisps per visible fire point
- average wisp width after final perspective warp
- wind direction in screen space
- drift speed in screen space
- wisp lifetime before disappearance
- expansion rate with age
- screen-space width growth from fresh source stem to old tail
- tail alpha falloff and endpoint disappearance
- edge softness and erosion rate
- brush-bundle versus plume-ribbon classification
- smoke-region alpha percentiles measured inside the smoke mask, not only across the full frame
- breakup/fragmentation rate
- broad haze coverage versus source-attached smoke coverage
- source burnout and smolder behavior
- whether final MP4 encoding preserves the thin strands

## Acceptance Gates

The current "looks better" criterion is too loose. Add objective gates before considering the target render done:

- Source attachment gate: at each audit timestamp, most visible active fire cores should have a visible downwind wisp after perspective warp.
- Source-wisp-only gate: the `source-wisps-only` ablation must show multiple distinct wisps with clear source attachment, downwind drift, and thinning before any broad/physical layer is reintroduced.
- Combined-quality gate: the combined render must preserve the same source-attached read as `source-wisps-only`; adding broad or physical smoke is not allowed to reduce source readability.
- Blanket rejection gate: reject frames where one connected smoke mass covers a large fraction of the fire region or reads as a continuous carpet rather than separated plumes.
- Low-frequency haze gate: measure blurred alpha/brightness coverage separately from strand coverage; broad low-frequency coverage must stay below a chosen threshold.
- Fire visibility gate: active fire cores must remain visible in the final composite, especially under fresh smoke near the sources.
- Late-frame dissipation gate: old smoke must fade or fragment over burn scars instead of accumulating into persistent white/gray cover.
- Encoded-detail gate: the final H.264 MP4 must retain enough thin-strand contrast at the audit timestamps, not just in PNG intermediates.
- Width-growth gate: old/downwind plume segments must be wider than fresh source stems by a chosen screen-space ratio.
- Tail-fade gate: downwind plume endpoints must fall below a chosen alpha/contrast fraction relative to the source stem.
- Edge-softness gate: plume edges must show a minimum soft/eroded transition width rather than hard clipped stroke boundaries.
- Brush-bundle rejection gate: reject frames where source-wisp-only smoke consists mainly of a few narrow high-alpha strokes with little diffuse, thinning old-tail area.
- Diffuse-envelope gate: if a diffuse plume/ribbon envelope is added, it must remain source-derived and low-frequency coverage must stay below the carpet/haze thresholds.

Additional film-quality gates for the reference-match mode:

- Full-bleed composition gate: reject frames with visible terrain-slab side walls, black slab edges, or a small local smoke event stranded inside excessive empty stage space.
- Multi-scale smoke gate: require detectable source wisps, mid-scale plumes, and broad regional ribbons in representative frames.
- Temporal-change gate: date, burned area, active fire-point count/distribution, and smoke coverage must change across the sequence.
- Fire-point density gate: require many small hot-core points rather than a few broad orange patches.
- Regional-ribbon gate: broad smoke may occupy large regions, but must retain flow lanes, holes, soft edges, and directionality.
- Grade gate: smoke should remain mostly gray/blue-gray with restrained white highlights and localized warm lift near fire.
- Label legibility gate: date, data/source label, and burned-area value must remain readable on encoded 1080p frames without competing with the main smoke read.
- Delivery gate: final reference-quality output must use the 1080p delivery profile and pass encoded-frame review at representative first-30s timestamps.

## Test And Validation Requirements

- Add visual regression artifacts, not only unit tests.
- Build side-by-side frame sheets comparing the target reference and generated output.
- Validate the final MP4 at fixed timestamps.
- Add tests or scripts that can verify source-attached smoke exists and fades, instead of only asserting broad smoke coverage.
- Existing tests that require large broad alpha coverage may need revision once wisps become primary.
- Include ablation renders:
  - broad smoke only
  - physical smoke only
  - source wisps only
  - combined final composite
- Keep the wisp system behind a CLI flag during development so tuning does not permanently break the current demo.
- Define a stop condition before tuning. Example:
  - most active fire points have visible downwind wisps
  - old wisps fade within a chosen frame window
  - broad haze coverage stays below a chosen threshold
  - final encoded MP4 still shows thin strands
- Add pass/fail audit metrics for source attachment, connected smoke carpet area, low-frequency haze coverage, fire-core visibility, and encoded strand preservation.
- Compare the combined render against the source-wisps-only ablation; the combined render should fail if it hides or washes out source-attached wisps.
- Track the ratio of strand-like smoke to broad haze in screen space after final perspective warp.
- Track the largest connected smoke component in the active fire region as a carpet-smoke rejection metric.
- Track source-to-smoke distance and downwind direction in screen space, not only in map coordinates.
- Track fresh-stem width, mid-plume width, old-tail width, endpoint alpha, edge softness, and brush-bundle score after final perspective warp.
- Track smoke alpha percentiles inside detected smoke regions. Full-frame alpha percentiles can be zero even when visible smoke exists, so they are not sufficient for morphology.
- Store the exact CLI command and layer weights used for any accepted render next to the audit artifacts.
- Done for film mode: first-30s contact sheets are sampled from the reference and generated render at the same cadence into `reference_film_first_30s_contact_sheet.png`.
- Done for film mode: `reference_film_gate_report` tracks active fire-core pixel count, active-fire temporal change, hot-fire fraction, post-smoke fire visibility, median fire mark radius, and halo/core area ratio.
- Done for film mode: full-frame smoke coverage, broad regional ribbon coverage, mid-scale smoke coverage, and smoke-centroid motion are tracked.
- Done for film mode: date span, burned-area growth, temporal luma delta, active-fire temporal change, and smoke-centroid movement are tracked.
- Done for film mode: label contrast and overlap with smoke/fire regions are tracked on representative frames.
- Done for film mode: delivery metadata, encoded size, codec, and configured bitrate are recorded in the audit JSON when `ffprobe` is available.

## Former Missing Requirements Now Resolved

The original spec did not yet spell out these implementation constraints. Current status:

- Done: target render preset `source-wisp-reference` encodes broad, physical, and wisp layer weights.
- Done: active-fire emitter source of truth is pre-bloom fire-core/front intensity via `active_fire_core_intensity_field()` and `fire_core_emitter_sources()`.
- Done: emitter count, spatial distribution, source attachment, and morphology gates now reject compact brush-bundle failures.
- Done: acceptance includes map-space and final screen-space source-attachment checks.
- Done: the local cigar-smoke target is explicit smoke-behavior matching. The 2026-06-13 addendum now splits full-frame reference-film quality into separate composition, time-series, regional smoke, typography, and delivery requirements.
- Done: final layer policy keeps the large regional August Complex plume as faint context only in the target preset.
- Done: review order is source-wisps-only, then no-broad, then combined.
- Done: regression tests and audit gates reject broad haze/carpet regressions and morphology regressions.

## Remaining Missing Or Underspecified Requirements

The original source-wisp spec is complete for local smoke morphology, but it was missing several requirements needed to reach the reference video's overall design quality:

- Done: a named `reference-film` render preset now targets full-frame dataviz quality separately from the local August Complex terrain-slab showcase.
- Done: concrete no-slab camera/composition settings now exist, and reference-film mode uses a regional California map extent with bounds/fire UV recorded in audit JSON.
- Partial: a synthetic time-series model exists for dates and burned-area values, with a disclosure/replacement policy and real-data replacement schema in audit JSON; actual daily active fire and observed regional smoke ingestion remain future work.
- Done: audit JSON records a policy for using real versus synthetic regional smoke data in reference-match mode, including the visible disclosure label and replacement policy.
- Done: visual design tokens exist in code for fire marks, typography, terrain grade, smoke color, vignette, and annotation placement; distributed-fire, smoke-texture, axis-band, and date-step gates now cover the previous contact-sheet gaps.
- Done: broad regional smoke gates now distinguish film-mode regional coverage/density from local source-wisp carpet rejection, and include smoke-centroid, mid-scale plume, and label-overlap gates.
- Missing implementation: add or replace the regional smoke primitive so human review rejects neither isolated blob patches nor ring/outline artifacts. Candidate approaches include observed smoke rasters, an advected noise/particle field with many overlapping soft masses, or a stronger screen-space visual metric for blob/ring artifacts.
- Done: a 1080p render/export profile now includes expected bitrate/CRF and color metadata.
- Done: film-level first-30s contact-sheet artifacts now exist for audit runs.
- Done: a rendered 30-second 1920x1080 candidate exists and passes automated film gates.
- Done: explicit acceptance criteria now cover evolving event cadence, distributed fire geography, regional transport naturalism, terrain/context, and restrained dataviz composition through gates plus `reference_film_visual_signoff_contract`.
- Missing: regenerate the 30-second 1920x1080 candidate after the current regional-smoke fixes, then inspect/sign off the new first-30s contact sheet.
- Done: artifact-specific rejection criteria now include regional smoke texture and axis-band gates for visible procedural contouring, vertical/rectangular smoke bands, and overly smooth synthetic ribbons.
- Done: the current target remains a California/August Complex regional film, augmented with disclosed synthetic distributed regional fire-context points; switching to a western-North-America domain is out of scope for this implementation pass.
- Done: a daily-data ingestion plan is recorded in audit JSON through `real_data_replacement_schema`, naming active fire detections, daily perimeters/burned area, observed smoke layers, accepted sources, required fields, fallback behavior, and attribution requirements.

## What Is Still Missing From This Doc

The document now covers the implemented source-wisp system, first reference-film technical milestone, stricter reference-film gates, and sign-off contract. Remaining work is no longer a doc-spec gap; it is candidate regeneration and evidence collection.

- Done: a designer-facing scorecard for the reference-film contact sheet is recorded in `reference_film_visual_signoff_contract`, covering geographic spread, smoke naturalism, narrative cadence, typography hierarchy, and terrain context.
- Done: quantified distributed-fire targets are implemented through `minimum_median_distributed_fire_cluster_count`, `minimum_median_fire_spread_grid_cell_count`, `minimum_median_far_fire_core_fraction`, and `maximum_median_primary_fire_dominance_fraction`.
- Partial: regional-smoke naturalism targets are implemented through `minimum_median_regional_smoke_texture_score` and `maximum_median_regional_smoke_axis_band_score`, but current visual review proves they are insufficient to catch isolated stamped blobs and subtle ring/edge artifacts.
- Done: narrative cadence is implemented through `minimum_temporal_date_span_days`, `minimum_median_date_step_days`, and `maximum_median_date_step_days`.
- Done: real-data replacement schema exists in audit JSON for active fire detections, perimeter/burned-area, and observed smoke layers, including attribution requirements.
- Done: sign-off ownership is specified as a designer/dataviz reviewer who inspects final MP4, preview PNG, first-30s contact sheet, and audit JSON after gates pass.
- Missing: produce a fresh 30-second candidate with the current gate set and record whether the reviewer approves or which scorecard criteria still fail.
- Missing implementation evidence: the current code passes `tests/test_california_cigar_smoke_hybrid.py` (`57 passed` on 2026-06-14), but no current full 30-second gated render exists, and the latest quick preview fails the `regional_smoke_naturalism` human scorecard criterion.

## Resolved Morphology Decisions

- Numeric plume-morphology thresholds are defined in the morphology threshold table above and serialized into `source_wisp_audit.json`.
- Reference alignment uses fixed generated/reference frame times at 1.0s, 3.5s, 5.5s, and 7.0s. Comparisons are behavior-only and preserve generated/reference frame sheets for human review.
- The plume stage is implemented inside `source_wisps_rgba()` with `--source-wisp-plume-ribbons`; it is source-derived, age-limited, low-alpha, and still subject to carpet/haze gates.
- The brush-bundle negative baseline is `--render-preset source-wisp-brush-baseline --smoke-ablation source-wisps-only`; it is stored/regenerated under `examples/out` and is expected to fail morphology gates.
- Temporal continuity is inherited from `SourceWispSimulator` particle history; old tails fade by puff lifetime, endpoint alpha, and removal from simulation rather than by hiding a persistent layer.
- Screen-space geometry is measured after `warp_map_layer_to_plate()` using the screen wind vector, component widths, lengths, endpoint alpha, and source-to-smoke attachment metrics.
- Smoke-region alpha stats are measured only inside detected smoke pixels and age-band masks; full-frame p95 remains advisory only.
- Edge-quality metrics include old-tail edge softness, diffuse/core area ratio, hard endpoint fraction, endpoint alpha fraction, and brush-bundle score.
- Encoded old-tail preservation is tracked with `encoded_soft_tail_like_fraction` alongside `encoded_strand_like_fraction`.
- Performance budget remains the accepted preset: 1150 source-wisp particles, 132 source-wisp emitters, 96 physical emitters when enabled, 520x408 smoke map, and morphology metrics only at fixed audit timestamps.
- If true hot-core/fire-front extraction is unavailable, the target preset falls back only to reduced smolder emitters; old burn scars emit no fresh white smoke and morphology gates still apply to source-wisps-only output.
- Fast tests cover metric math, plume-vs-brush separation, preset ownership, source attachment, and gate evaluation; full MP4/audit runs remain explicit visual-regression artifacts.
- Baseline retention policy: accepted and negative baseline artifacts are regenerated on demand under `examples/out`; only code, tests, and audit-command contracts are checked in.

## Resolved Implementation Decisions

Implemented in `examples/california_cigar_smoke_demo.py` and covered by `tests/test_california_cigar_smoke_hybrid.py`:

- Target preset: `--render-preset source-wisp-reference` is the default accepted preset. It uses pre-bloom fire-core emitters for source wisps and the optional physical smoke path, `--source-wisp-warmup-mode visible-only`, `--source-wisp-max-particles 1150`, `--source-wisp-max-emitters 132`, `--source-wisp-plume-ribbons`, `--physical-max-sources 96`, `--broad-smoke-alpha 0.025`, `--physical-alpha 0.0`, and `--no-physical-smoke` by default. The physical layer remains available with `--physical-smoke` but is not part of the accepted default until its ablation stays below the haze/carpet gates.
- Legacy baseline: `--render-preset legacy-combined` preserves the previous broad/physical composite settings for negative-baseline comparison, not for acceptance.
- Brush-bundle baseline: `--render-preset source-wisp-brush-baseline` uses fire-core emitters but disables `--source-wisp-plume-ribbons`; it preserves source-attached compact stroke morphology as a known-bad baseline.
- Reference-film preset: `--render-preset reference-film` selects full-bleed `map-film` composition, synthetic date/area progression, procedural regional smoke, fine fire points, film-grade map styling, and the `reference-1080p` delivery profile.
- Active-fire source of truth: `active_fire_core_intensity_field()` derives a pre-bloom flame core/front raster from lifecycle-active fire sources and connected fire-front segments. `fire_core_emitter_sources()` samples spatially separated emitters from that raster. Glow-only bloom, wide bloom, composited fire halos, and final RGB imagery are excluded from smoke emission.
- Smolder fallback: if no active pre-bloom hot-core pixels exist, recently expired lifecycle sources may emit reduced smolder smoke; old burn scars emit no fresh white smoke.
- Audit JSON schema: `source_wisp_audit.json` uses schema version `source-wisp-audit-v3` and records component reports, encoded reports, morphology field meanings, age bands, `gate_report`, `target_layer_policy`, `source_data_contract`, `accepted_artifact_contract`, review order, negative baselines, threshold calibration, regeneration commands, and exact CLI command. Reference-film audit runs additionally record `reference_film_frame_reports`, `reference_film_encoded_reports`, `reference_film_gate_report`, `reference_film_audit_schema`, probed stream metadata, and `reference_film_first_30s_contact_sheet.png`.
- Accepted artifacts: source-wisp mode requires final MP4, preview PNG, reference/generated frame sheet, ablation sheets, audit JSON, and exact CLI command. Reference-film mode requires final 1080p MP4, preview PNG, audit JSON, first-30s contact sheet, reference-film frame reports, reference-film gate report, and exact CLI command.
- Review order: approve `source-wisps-only`, then `no-broad`, then `combined`.
- Hard thresholds: attached source fraction >= 0.58, screen attached source fraction >= 0.52, active fire emitters >= 16 on early/mid audit frames, emitter bbox fraction >= 0.012 on early/mid audit frames, source-wisp components >= 4, largest broad/physical smoke-carpet component fraction <= 0.34, broad/physical low-frequency haze fraction <= 0.30, strand-to-haze ratio >= 0.18, fire-core visibility fraction >= 0.62 on frames with enough active fire, combined strand retention >= 0.74, encoded strand-like fraction >= 0.0035, late low-frequency haze fraction <= 0.20, morphology stage coverage >= 0.00012, transition width growth >= 1.12, old-tail width growth >= 1.42, old-tail endpoint alpha fraction <= 0.44, old-tail coverage growth >= 2.50, old-tail edge softness >= 1.05 px, old-tail diffuse/core ratio >= 1.10, brush-bundle score <= 0.44, and encoded soft-tail fraction >= 0.0025. Transition/old-tail transformation gates apply from 2.0s onward.
- Enforcement: `--enforce-audit-gates` rejects source-wisp runs if any hard gate in `gate_report` fails; for `reference-film`, enforcement uses `reference_film_gate_report` so local morphology gates do not block the separate full-frame film target.
- Implementation ownership: fire-core emitter extraction, `SourceWispSimulator`, physical emitter selection, source wisp rendering, compositing weights, audit metrics, and cigar smoke regression tests are the owned surfaces for this target.
- Performance budget: the accepted preset caps source-wisp particles at 1150, source-wisp emitters at 132, physical emitters at 96, keeps default map size at 520x408, physical dimensions at 84x22x66 unless overridden, and computes audit metrics only at fixed audit timestamps.
- Visual-review ownership: acceptance is recorded by retaining the generated audit JSON and artifacts for the accepted run; the automated gate is necessary but not a substitute for human review of the ordered ablations.
- Plume morphology ownership: target plume ribbons, morphology gates, encoded soft-tail checks, and brush-baseline separation are implemented and covered by focused tests.

## Performance Requirements

- Emitting from every fire point may create hundreds of wisps.
- Use a cap or weighted sampling strategy if needed.
- Sampling should prioritize visually important hot cores/front pixels.
- The implementation should avoid per-frame cost that makes the demo impractical to render.

## Implementation Direction

Recommended sequence, with current status:

1. Done: extract matched reference/generated frames and create a comparison sheet.
2. Done: add a flagged source-wisp prototype.
3. Done: render ablations and the combined MP4.
4. Done: tune against the final encoded MP4 and comparison frames through encoded strand and soft-tail metrics; full visual review remains an explicit artifact review step.
5. Done: reduce or disable broad haze until the viewer reads source-attached smoke first and regional haze second.
6. Done: review order is source-wisps-only, then no-broad, then combined; brush-bundle output is a separate negative baseline.
7. Done: add automated pass/fail metrics for carpet smoke, low-frequency haze, fire-core visibility, source attachment, and encoded strand preservation.
8. Done: migrate source emission from synthetic `HybridSmokeSource` points to actual active hot-core/fire-front pixels for the target preset.
9. Done: add plume-morphology metrics and gates for width growth, tail fade, edge softness, endpoint disappearance, and brush-bundle rejection.
10. Done: implement a source-derived old-tail plume/ribbon stage that creates wide, thin, disappearing smoke without restoring broad carpet haze.
11. Done: keep both old negative baselines: broad carpet smoke and source-attached brush-bundle smoke.

Recommended next sequence for reference-film quality:

1. Done: add a full-bleed `reference-film` render preset separate from `source-wisp-reference`.
2. Done: define and implement the map extent/camera for a reference-like composition with no terrain-slab edges; reference-film mode now uses a full-bleed regional California extent.
3. Partial: replace static display values with a deterministic time-series driver; date and burned area now evolve and cadence gates exist, but real daily input ingestion remains future replacement work.
4. Done: redesign fire marks as many small hot-core points with tight local halos; visibility/radius/halo metrics and distributed-fire geography gates exist, with disclosed synthetic regional context points in reference-film mode.
5. Partial: add a controlled regional smoke layer with source-local, mid-scale, and broad synoptic scales; procedural regional ribbons, mid-scale/centroid metrics, smoke-texture/axis-band gates, and real/synthetic policy exist, but observed-smoke replacement remains future work.
6. Partial: add color-grade and typography settings for the reference-film mode; label contrast/overlap gates pass, but final hierarchy may need retuning after the regional composition changes.
7. Done: add first-30s generated/reference contact sheets and 1080p encoded-frame review artifacts for audit runs.
8. Done: add regional film-quality gates for composition, temporal change, active-fire temporal change, fire-point density/visibility, smoke scale distribution, label legibility, and output delivery.
9. Done: render a 1920x1080 reference-film candidate and compare it against the first 30 seconds of the reference.
10. Done: define a human visual sign-off scorecard for the 30-second candidate in audit JSON.
11. Done: current target remains a California/August Complex regional film with disclosed synthetic distributed context, not a domain switch.
12. Missing: regenerate and inspect the stricter 30-second candidate before claiming final visual approval.

## Non-Goals

- Do not treat this as a simple alpha/color tuning pass.
- Do not rely solely on broad HRRR-style smoke guidance for the reference effect.
- Do not emit smoke from bloom halos.
- Do not judge success from a single still frame or unencoded PNG.
- Do not conflate the completed local cigar-smoke target or gate-passing 1080p candidate with final full-frame reference-film visual approval.
