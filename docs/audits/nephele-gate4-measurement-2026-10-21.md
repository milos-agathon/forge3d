# NEPHELE Gate 4 measurement — 2026-10-21

**Physical acceptance remains NOT_PROVEN; release is blocked.** This note
records the requested Gate 4 diagnosis and a new CPU prediction of the chosen
terrain-lighting term. No renderer, test, gate, policy, pin, golden, frozen
reference source, fixture, 640-spp reference, or 320-spp control was changed.

## Inputs and reproduction

The source capture is [full CI run 39517423981](https://github.com/milos-agathon/forge3d/actions/runs/39517423981),
head `d38758e183c522dfb660805987d751f5a6263b24`, artifact
`nephele-physical-gpu-evidence-39517423981-1` (artifact ID `12311573537`).
Its adapter record reports an NVIDIA GeForce RTX 3070, Vulkan, NVIDIA driver
610.60, discrete GPU, no software fallback. These are the **existing CI
capture's** conditions; no new physical-GPU render was run for this note.

The local PR checkout started at `07841ba1d44f7bfe9c23f15b732fc1748a35718a`
on `codex/pr172-sync-main`, matching remote PR branch `codex/20-nephele`.
The existing local `codex/20-nephele` branch points at an older commit and was
left untouched. The worktree's pre-existing deleted fixture files are unrelated
and were not staged or restored.

The 34 same-named files shared by `tests/nephele/fixture` and the downloaded
artifact match byte for byte, including the scene inputs, masks, both reference
sample counts, and reference provenance. Input and tool SHA-256 values are in
the output JSON. CPU calculations used NumPy 2.5.1 and the repository's existing
`tests/_deltae.py` and `tests/_ssim.py`; they do not import or execute forge3d's
native extension.

The additive patch is preserved in
[`measure_gate4.py`](nephele-gate4-measurement-2026-10-21/measure_gate4.py).
It executes the historical round-2 `nephele_first_order.py` with its original
96-direction environment quadrature and 48 camera steps, adding parallel
`ss_sun_unoccluded` and `ss_env_unoccluded` accumulators with `vis` / `visw`
omitted. It writes the exact patched program into the scratch directory and
records that program's hash. The historical tool is unchanged.

Run from the repository root (keep the scratch directory unstaged; some file
extensions inside it are ignored, but the directory itself is not):

```powershell
gh run download 39517423981 --repo milos-agathon/forge3d `
  --name nephele-physical-gpu-evidence-39517423981-1 `
  --dir .release-evidence/2026-10-21/ci-39517423981
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:FORGE3D_TEST_INSTALLED_WHEEL = '1'
python docs/audits/nephele-gate4-measurement-2026-10-21/measure_gate4.py `
  --artifact-dir .release-evidence/2026-10-21/ci-39517423981 `
  --scratch .release-evidence/2026-10-21/first-order `
  --output docs/audits/nephele-gate4-measurement-2026-10-21/measurement.json
python docs/audits/nephele-gate4-measurement-2026-10-21/predict_surface_cloudlight.py `
  --artifact-dir .release-evidence/2026-10-21/ci-39517423981 `
  --output docs/audits/nephele-gate4-measurement-2026-10-21/surface-cloudlight-prediction.json
```

Full first-order stdout is retained locally at
`.release-evidence/2026-10-21/measure-gate4.log`. Machine-readable outputs:
[`measurement.json`](nephele-gate4-measurement-2026-10-21/measurement.json),
[`surface-cloudlight-prediction.json`](nephele-gate4-measurement-2026-10-21/surface-cloudlight-prediction.json).

## Recomputed measurements

| Measurement | Result | Existing requirement |
|---|---:|---:|
| Gate 3 sky/cloud pixels with ΔE2000 < 2.5 | 95.5871% | ≥95% |
| Gate 3 godray SSIM | 0.982692 | >0.95 |
| G4a shadowed terrain pixels with ΔE2000 < 2 | 59.6660% of 1,557 pixels | ≥95% |
| G4a mean Lab error, realtime minus reference | (−1.028839, −0.109543, −1.207016) | Diagnostic |
| G4a failing pixels that are too dark | 84.5541% | Diagnostic |
| G4a ΔE76 < 2 after subtracting each 8×8 tile's masked mean Lab error | 95.5684% | Diagnostic only; not the ΔE2000 gate |
| G4b realtime ablation SSIM in rows 48–63 | 0.989950 | <0.80 |
| G4b estimated reference ablation SSIM | 0.970583 | Diagnostic first-order estimate |
| Reference first-order mean linear luminance change in ROI | 0.0373167 | Diagnostic |
| Realtime mean linear luminance change in ROI | 0.0171081 | Diagnostic |
| Realtime / reference first-order change | 45.8457% | No approved tolerance |

The error is predominantly a smooth offset. Subtracting measured tile means
is a diagnosis, not a permitted lighting correction or proof that the chosen
physical term will satisfy G4a.

For the reference-ablation estimate, add the difference between the unoccluded
and occluded camera single-scatter accumulators to the linearized 640-spp
reference. Decode and encode with round 2's sRGB + ACES conventions. This
reproduces the reported approximately 0.971 SSIM and 46% response ratio.
It strongly challenges a required SSIM below 0.80 on this fixture, but is **not**
a converged path-traced reference ablation or a proof of impossibility for all
correct renderers. It omits changes to higher-order paths.

## New G4a prediction

[`predict_surface_cloudlight.py`](nephele-gate4-measurement-2026-10-21/predict_surface_cloudlight.py)
adds the saved physical 192-direction result from round 2's
`surface_cloudlight.py` to the current CI frame as
`linear_realtime + realtime_camera_transmittance * surface_cloudlight`, then
applies the same presentation transform and unchanged ΔE2000 mask scoring.
The presentation inverse first proves a byte-identical round trip of the
captured frame. No gain or fitted constant is used.

| Prediction | G4a fraction ΔE2000 < 2 |
|---|---:|
| Captured realtime | 59.6660% |
| Captured realtime plus physical terrain term | 62.1708% |

The mean added linear RGB on shadowed terrain is
`(0.01894085, 0.01778171, 0.01583128)`. Predicted sky pixels are byte-identical
to the capture. This is an offline prediction using a saved quadrature result,
not proof of the runtime implementation or B1. It indicates that the selected
term alone is insufficient to demonstrate the requested 95% G4a acceptance.

## Owner decisions still required

Proposed G4b replacement: compare the real-time ablation's mean linear
luminance change in the existing godray ROI with the reference physical change,
using relative error `abs(delta_realtime - delta_reference) / abs(delta_reference)`.
Milos must set the tolerance and approve the reference measurement source.
The first-order diagnostic must not silently become a path-traced reference.
A zero reference response cannot support that relative comparison and must
remain unresolved unless the owner approves a separate rule. No tolerance,
gate implementation, or policy approval is invented here. A new strong-shaft
fixture and regenerated reference remain the separately approved OD-10 option.

The new G4a prediction exposes a remaining physical error beyond the chosen
term. Expanding the investigation beyond that term requires Milos's approval
under the local investigation-expansion rule. No fitted correction, new
subsystem investigation, or runtime change was substituted for that decision.

Pins, shader contracts, f32 ledgers, tests, goldens, merges, and tag pushes still
require their specified owner approvals. None was changed. No release step
was executed. The requested head-wheel CPU acceptance, NVIDIA rerender, full
acceptance, merge-SHA acceptance, publication, and PyPI install check remain
unrun because no runtime fix or owner-approved gate replacement exists yet.

Reusable steering: a mean radiance deficit matching a physical term does not
establish a per-pixel ΔE pass fraction. For future shader work, propose adding
an offline score of the exact physical term against the current capture before
treating it as a sufficient G4a fix. No agent contract or skill was modified.
