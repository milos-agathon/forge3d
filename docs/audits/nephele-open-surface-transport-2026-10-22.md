# NEPHELE: full CPU transport and G4b owner decision

The reference-domain CPU response is **0.02187742**, with paired Monte Carlo
standard error **0.00009397**. The unchanged G4b policy accepts
**0.02985339–0.04478009**, centered on **0.03731674**. Full scattering does not
close the first-order gap: its measured contribution beyond the direct
first-order terms is **−0.00087302**, standard error **0.00007997**.

This resolves the review's missing full-scattering measurement. It does not
select terrain semantics, change the policy, or prove that every possible
real-time approximation fails the five image checks. The release remains
incomplete. No native/GPU result or release readiness is claimed.

## Calculation and checks

The [program](nephele_multiple_scattering_cpu.py) evaluates all **1,024** frozen
ROI pixels at **640 samples per pixel**, the existing frozen-reference count,
in both visibility arms: **1,310,720 paths**. Its RGB spectral delta tracking,
HG phase, Lambert surface transport, environment MIS and Russian roulette
follow the Rust reference. Roulette starts at bounce 3 with minimum probability
0.05, from `src/media_py.rs`; there is no bounce cap. Maximum observed bounce
was 14. Exact piecewise-trilinear shadow optical depth and uniform sampling of
the fixture's constant environment preserve the transport expectation, while
reducing variance. This is an independent CPU estimator, not a bit-identical
f32 replay or execution of the GPU reference adapter.

Both arms retain medium on both sides of the open bilinear terrain surface.
The disabled arm ignores terrain visibility for sun/environment queries at
medium collisions and replaces terrain reached by medium continuation rays
with environment transport. Primary and surface-origin queries retain their
terrain. This mirrors the runtime medium visibility/terminal-light ablation;
the frozen reference has no disabled image to replay directly.

Two independent cross-checks support the calculation:

| Check | Result |
|---|---|
| Sampled direct first-order response | 0.02275044, standard error 0.00008040 |
| Earlier integrated first-order response | 0.02279963 |
| Enabled mean RGB minus frozen reference mean | (−0.00007240, −0.00011557, −0.00006243) |
| CPU enabled mean RGB standard error | (0.00010128, 0.00008527, 0.00007251) |
| Mean absolute per-pixel RGB difference from frozen reference | (0.00311396, 0.00262030, 0.00221486) |

The frozen image also has 640-spp noise; its unavailable sampling variance is
not inferred. Standard errors are empirical statistical estimates, not
deterministic error bounds or a new acceptance threshold. The full response
misses the frozen target by **41.37%**. The numerical gap to the lower allowed
response is about **85 paired standard errors**, so another sample-count
increase is not needed to distinguish these measured values.

Three new deterministic checks pass: Rust integer sample identity, phase and
local-direction basis, and vacuum/open-surface transport from below. The six
previous analytic geometry/optical-depth checks remain applicable; their
inputs have not changed. No native `forge3d` module was imported.

## Review issues N1–N6

| Finding | Action and remaining limit |
|---|---|
| N1: target versus reference terrain semantics | Full recursive CPU measurement completed above. The old helper's solid-interior estimate is not a measurement of this open-surface transport. The policy choice remains Milos's. |
| N2: missing reference-domain source cases | Reran all 16 factorial and 24 actual-pixel combinations with `--source-domain reference-open`, now the default. The 24 unchanged archived cases are retained for comparison. The higher-order residual in these hybrids is explicitly historical, so they are not labeled complete reference-domain transport. The separate recursive CPU check supplies that missing ROI measurement. |
| N3: conflicting image criteria | No candidate among the 64 passes all five checks. G4b-passing archived/hybrid cases still retain historical closure behavior and have B1 at most 0.8503. No shader candidate is qualified, and no GPU dispatch follows. |
| N4: external input paths | Provenance uses repository-relative paths where possible and explicit absolute paths otherwise. Historical checkout lookup is bound in memory, independent of evidence-directory depth. The complete study ran using the reviewer's out-of-tree first-order inputs; its out-of-tree historical evidence also loads successfully. |
| N5: unpublished JSON postprocessing | The correction program now emits `g4a_fraction_change_after_removing_term` itself. Published JSON is copied byte-for-byte from generated output, with no added fields. |
| N6: evidence could be staged accidentally | The worktree `.gitignore` now ignores `/.release-evidence/`, including the local environment and wheels. No evidence files are tracked or staged. |

The [CPU JSON](nephele-open-surface-transport-2026-10-22.json) includes program,
fixture, policy and relevant reference-source hashes, uncertainty, controls,
environment and input-integrity checks. The updated
[correction JSON](nephele-residual-study-2026-10-22.json) is the generated
reference-source-domain run. Original reports and raw arrays remain under
`.release-evidence/2026-10-22/`; no frozen input was overwritten.

## Owner decision needed

Milos must specify whether G4b should measure the reference's **open surface**
or retain the **solid-interior behavior** behind the accepted first-order
target. Following the open-surface expectation would require a separately
authorized policy revision, which the current release prompt forbids.
Retaining the target leaves the need to justify and qualify a different
real-time transport model against every unchanged image criterion.

Neither choice has been applied. No fixture, policy, renderer, existing test,
gate, pin, golden, release metadata or hosted state changed. The next owner
choice determines the permitted continuation; the existing evidence does not
justify claiming the release complete or an all-model impossibility.

Reproduce from this worktree with a fresh output directory:

```powershell
.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/nephele_multiple_scattering_cpu.py --first-order-study .release-evidence/2026-10-22/first-order-reference-verified --output .release-evidence/2026-10-22/independent-full-open-surface
.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe -m unittest discover -s docs/audits -p test_nephele_multiple_scattering_cpu.py -v
```

Proposed reusable guidance, not applied to agent rules: define open-surface
versus solid-interior visibility when approving an ablation reference. On
Windows, use the repository's existing line-ending settings for `git diff
--check`; overriding `core.autocrlf=false` falsely treated this checkout's
CRLF lines as whole-file whitespace changes. The normal check passes.
