# NEPHELE review corrections — 2026-10-22

**Steps 2 and 3 remain incomplete. No transport blocker or impossibility has
been demonstrated.** The earlier claim that the failed approximate forecasts
justified ending the CPU investigation is withdrawn. Shader implementation,
new GPU CI and release preparation still require the offline proof specified
by the release prompt. No such proof, renderer fix or release is claimed.

The subsequent review's missing full-scattering measurement is now complete
for the G4b ROI: **0.02187742 ± 0.00009397 standard error**, below the frozen
allowed minimum **0.02985339**. See the
[CPU transport report and owner decision](nephele-open-surface-transport-2026-10-22.md).
The policy remains unchanged; no full-frame candidate passes all five checks.

The six reviewed files are preserved under
`.release-evidence/2026-10-22/pre-review-audit/`. The new
[JSON](nephele-residual-study-2026-10-22.json) records the corrected experiments,
their actual inputs, all five metrics and margins, prediction-error
sensitivity, the complete combination matrix and the remaining uncertainty.
It is now the program's unmodified output for `reference-open` first-order
sources: 16 factorial, 24 actual-pixel hybrids and 24 archived cases. The old
height-rejected run remains preserved in raw evidence. Hybrid higher-order
closures retain their historical domain rules and are labeled accordingly.
The [B1 capture analysis](nephele-b1-regression-2026-10-22.md) remains valid;
the independent review reproduced its capture scores and hashes.

## Corrections to F1–F7

| Finding | Correction and evidence | Remaining limit |
|---|---|---|
| F1: unqualified approximate models | Predict the already captured `ac739096` → `68cc04e9` sampler change and measure its per-pixel error. New direct-source terms evaluate all 192 directions at the actual queried positions, with analytic bilinear terrain intersections and exact piecewise-trilinear optical-depth integration. | The empirical sampler error is not a transferable bound for different physics. A complete angular higher-order solution is still unproven. |
| F2: confounded pixel comparison | Use the same sun and environment estimator in both lighting ablations, the midpoint of each actually executed segment, and the same camera T/integration weights. Remove the historical cached-sun/direct-sun asymmetry and nearest-froxel zero fallback from the new term. Save source components for inspection. | This isolates a first-order source-resolution term; it does not certify every higher scattering order. |
| F3: omitted combinations/wrong best | Evaluate every distinct archived continuation/boundary/tent profile with both continuation choices, both primary-surface choices and both source-resolution choices. Also evaluate the complete three-term direct-source factorial with the tent independently controlled. Deduplicate byte-identical profile frames. | The highest-scoring model is a forecast, not a native result. |
| F4: bundled visibility/estimator/filter changes | Compare visibility and source-domain rejection with the 192-direction estimator and tent fixed; evaluate both operation orders and report their interaction. Calculate all five requested G4a terms, close them to the observed residual with an explicit unresolved remainder, and report the source-only lateral term separately. | Algebraic closure is not an independent measurement of unrecorded reference path families. The estimated primary sun/cloudlight split is labeled. |
| F5: unsupported numerical choices | Replace new ray marches with cell-bounded quadratic roots and optical-depth integration exact for each cubic density restriction. Use the existing 192-direction and production world-cell spacing contracts. Add a measured nested camera-integration comparison. Cite the existing one-phase-direction and 64-step shader contracts. | Archived 0.25/0.02/400/96/48 diagnostics are not promoted to exact/converged runtime predictions. |
| F6: HEAD/provenance/input mutation | Label `68cc04e9` as the captured revision and `ca4f5bd3` as uncaptured. Use repository-relative POSIX hash keys or explicit absolute paths for external inputs. Bind the historical checkout lookup in memory. Load the cloudlight Git blob in memory. Expose historical intermediates through an in-memory AST, without rewriting its program. Require fresh output directories and verify input hashes after each run. | Merge source inspection cannot prove physical equivalence. |
| F7: hidden source-domain change | Explicitly separate geometric source-domain rejection from lighting visibility. Test fixed-domain ablations, retain the untruncated reference medium on both sides of its open terrain surface in the reference first-order study, and quantify the distinction. | No new production domain/ablation policy is selected or implemented. Every new visibility term must remain controlled by the existing ablation flag. |

## Measured prediction error and corrected best forecast

For the held-out sampler change, the historical model predicts G4a **0.655748**;
the actual capture gives **0.710983**, an error of **-0.055234**. Its G4b
relative-error prediction differs by **+0.037350**. Mean absolute linear RGB
prediction error is (0.002908, 0.002582, 0.002173) on sky and
(0.002332, 0.002136, 0.001849) on shadowed terrain. The JSON also records maximum
per-pixel errors, saved error maps and candidate sensitivity to the observed
signed discrepancy. These are empirical sensitivities, not confidence bounds.

The original reported best G4a of **0.842004 was wrong**. Including the omitted
surface/continuation combinations gives the following highest-G4a archived
hypothesis, using the same unfitted composition as the reviewed study:

| Criterion | Latest actual capture | Highest-G4a model | Requirement | Model margin |
|---|---:|---:|---:|---:|
| B1 fraction | 0.712766 | 0.998818 | >= 0.95 | +0.048818 |
| Shaft SSIM | 0.991600 | 0.990499 | > 0.95 | +0.040499 |
| G4a fraction | 0.710983 | 0.945408 | >= 0.95 | -0.004592 |
| Medium changed fraction | 0.321566 | 0.345315 | >= 0.10 | +0.245315 |
| G4b relative error | 0.473969 | 0.441516 | <= 0.20 | -0.241516 |

The model is existing-source scalar continuation plus coupled surface return.
Its small G4a gap is below the measured sampler-prediction error and therefore
cannot establish failure of a physical implementation. Its G4b gap remains
substantial. Neither that gap nor the other forecasts proves impossibility.
OD-10 is **not established as the only remaining path**.

## Reference source domain and numerical evidence

`ReferenceScene` uses the production bilinear terrain surface, not a closed
terrain solid. The canonical medium is not clipped by terrain height. The
reference-faithful first-order calculation therefore retains the same medium
source domain in both lighting ablations. Surface queries use the reference
`tmin = 1e-3`; they do not also add a normal offset.

The fresh first-order study uses the existing 192-direction quadrature,
analytic terrain roots with no ray-length/march cap, and exact optical-depth
integration through the trilinear density cells. The two Gauss nodes follow
from the cubic ray polynomial, not a sampling budget. Camera integration uses
the production world-cell spacing, 0.625, and its existing half-cell rule,
0.3125. Fine integration executes 536,681 source points, at most 210 segments
per camera ray, with every ray ending at the medium boundary or primary hit.
Those counts are observed geometry-derived results.

The fine first-order sunlight ablation response is **0.0153853**, close to the
archived estimate 0.0154541. The environment response is **0.0074143**, rather
than the archived 0.0218626. The enabled contribution originating below the
open surface is 0.0144591 for environment and 0.0000847 for sunlight. Treating
all such points as blocked reproduces approximately **0.0373434** total,
close to the accepted archived total **0.0373167**. This identifies a concrete
source-domain/occlusion-semantic difference in the old diagnostic; it does
not authorize changing the frozen gate policy or establish a new full-order
ablation reference.

Coarse-to-fine mean absolute RGB changes are about
(0.000021, 0.000019, 0.000017) on sky and
(0.000055, 0.000053, 0.000048) on shadowed terrain. This is a measured nested
difference, not a rigorous infinite-resolution bound. The separate recursive
CPU study now measures higher-order transport in the G4b ROI. Full-frame path
attribution remains unresolved. The JSON preserves the numerical comparison
and the explicit first-order scope of this integral.

The G4a decomposition reports camera-path in-scatter, camera T, the estimated
surface sun term, surface environment and lateral source reconstruction,
plus a named unresolved remainder. Their sum closes to the measured shadow
RGB residual (-0.015434, -0.008488, -0.000383) at floating-point roundoff.
The visibility/source-domain factorial reports both orderings and their
interaction, with the estimator and tent held fixed; it no longer calls an
order-dependent difference a unique environment share.

## Checks, provenance and work still open

Six analytic CPU checks passed: exact constant/cubic density integration,
primary/secondary flat-surface intersections, the later admissible root of a
bilinear cell, fixed geometric-domain ablations, and retention of the medium
on both sides of the reference's open surface. The analytic primary hit mask
matches the reference exactly. Input-hash guards verify that each study
leaves all its inputs unchanged. Raw captures, previous reports and complete
logs remain unstaged in `.release-evidence/2026-10-22/`.
That directory is now git-ignored, including its analysis environment and
wheels. Three additional CPU path checks pass; the recursive ROI result also
agrees with the earlier direct first-order integral and closely matches the
enabled frozen image's mean RGB. These are CPU checks, not a native replay.

The isolated analysis environment uses Python 3.13.14, NumPy 2.3.5 and Numba
0.63.1. A task-local venv supplies the NumPy version supported by the already
installed Numba; no production dependency changed. The original B1 capture
measurements used NumPy 2.5.1 and were reused rather than unnecessarily rerun.

No renderer, viewer, bindings, frame graph, memory accounting, existing test,
fixture, mask, gate, pin, conversion ledger or golden changed. The 14 canonical
NPY files and protected acceptance paths remain unchanged. No local GPU work,
new GPU CI, wheel build/install, native validation, commit/push, merge, release
dry run, tag or publication occurred. D3D12, Metal and WASM evidence is
**ABSENT**. No protected-value, merge or tag approval was requested or received.
The unrelated main checkout and pre-existing worktrees remain untouched.

The outstanding requirement is a sufficiently qualified complete transport
candidate and independent physical-path attribution. The revised studies
provide better measurements and corrected tools, not that missing proof.
The release remains incomplete. The full CPU measurement supports the
review's terrain-semantics mismatch, so Milos's open-surface versus
solid-interior decision is now requested before further model development.
Do not infer an accepted shader implementation or an all-model impossibility.

Reproduce with the task-local interpreter, preserved inputs and **new** output
directories:

```powershell
.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/nephele-first-order-study-2026-10-22.py --artifacts .release-evidence/2026-10-22 --output .release-evidence/2026-10-22/independent-first-order-recheck
.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/nephele-correction-study-2026-10-22.py --evidence .release-evidence/2026-10-22 --artifacts .release-evidence/2026-10-22 --review-models .release-evidence/2026-10-22/review-models --first-order-study .release-evidence/2026-10-22/independent-first-order-recheck --output .release-evidence/2026-10-22/independent-correction-recheck
.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/test_nephele_transport_cpu.py
```

Proposed documentation improvements, **not applied**: state the reference
in-scatter path classification, distinguish an open surface from a closed
terrain solid when defining source visibility, and identify the production
64³ world field when specifying the half-cell rule. No AGENTS.md or skill
was changed, and no subagents were used.
