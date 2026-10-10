# D04 ordered composition review — updated 2026-10-10

The requested Python composition behavior and scoped review corrections are
implemented and proven locally. The owner approved version 4 composition bundles
with binary snapshots on 2026-10-09; ordinary no-pass bundles remain version 3.
The scoped ticket is complete. The owner explicitly chose to retain the tracked
generated PDB produced by the binding rebuild. The broader D04 assessment remains partial: this
bounded slice does not establish all recipe-family campaigns or other backends.

## Scope and implementation

Work is in `D:/forge3d/.worktrees/d04-composition`, on `codex/d04-composition`
from verified current main `08b5ad1423d5c897938be8f96ada91a8a0765490`. `git ls-remote` confirmed that
remote main and the local remote-tracking ref matched. The original checkout is
`chronos-frame-perfect` at `7c81644a`; its unrelated untracked data and agent docs
were preserved. One parent Codex agent performed implementation and review;
no subagents or model escalation were used.

- `python/forge3d/render_pass.py` and `.pyi`: immutable RGBA inputs; named
  color/relief/overlay specifications; explicit ordered references, source
  opacity, linear/sRGB and straight/premultiplied semantics; multiply, screen
  and full Porter-Duff alpha-over. Invalid inputs and graphs raise before drawing.
  Snapshot equality and hashing include pixels; specification hashing includes
  immutable parameters. Disconnected graphs and explicit empty replacements fail.
- `python/forge3d/_render_pass_bundle.py`: content-addressed NumPy 1.0 snapshots,
  stored once per unique pixel payload, with little-endian float64 C-order data.
  Paths, checksums, encoding, shape, dtype and canonical contents are verified;
  references and symlinks/junctions cannot escape the declared bundle root.
- `python/forge3d/map_scene.py` and `.pyi`: public `render_passes`, optional recipe
  fields, immutable compiled JSON, direct replay, and recipe/plan mismatch refusal.
  Preset and alignment reconstruction preserve passes. `render` retains the
  native-or-diagnostic contract and directs configured image passes to the explicit
  Python compositor through a structured `MapSceneNativeUnavailable` diagnostic.
- `python/forge3d/recipe_manifest.py` and `.pyi`: optional compiled specification.
  Compiled specifications and bundle recipes contain compact snapshot references.
  Standalone recipe dictionaries retain inline data. No-pass fields remain absent;
  v3 no-pass bundles and existing defaults are unchanged. `bundle.py` also accepts
  the owner-approved v4 composition format only through explicit metadata-reader
  opt-in, used by `MapScene.load_bundle`. That loader requires nonempty passes for
  v4. Generic/viewer readers and the ordinary writer retain their v3 defaults.
  Public `recipe_manifest()` summaries list compact pass references and share the
  compiled plan's recipe hash; scene, recipe and mapping forms agree.
- `python/forge3d/__init__.py` and `.pyi`: matching public exports. No new native
  symbol, Rust/WGSL source change or production dependency.
- `tests/test_mapscene_render_passes.py`, `test_mapscene_render_passes_gpu.py`,
  `test_api_contracts.py`, `test_mapscene_typing.py`, `test_mapscene_docs.py`:
  hand-computed opaque/transparent fixtures, exact PNG channels, order/opacity,
  color/alpha conversion, invalid graphs, canonical snapshots, stale-plan handling,
  native diagnostics, public typing, and byte-identical bundle replay. Float
  comparisons use NumPy's float64 decimal precision with zero relative tolerance;
  integer pixel fixtures use exact equality.
  Review corrections add exact default-linear multiply/screen/alpha-over,
  premultiplied uint8, 16-bit PNG channels, failed replacement state preservation,
  equality/hash, disconnected graph, binary deduplication and tamper tests. The
  physical test independently checks every composed pixel and proves that relief
  and overlay affect the result; native vector byte channels verify linear output.
- `docs/examples/mapscene_render_passes.py`, `docs/guides/ordered_map_composition.md`,
  `docs/guides/feature_map.md`, `docs/guides/offline_3d_map_rendering.md`,
  `docs/api/api_reference.rst`: usable native-render example and API contracts.
- The local `D:/forge3d/docs/moonshot-roadmap.html` D04 node and
  `MAP-D04-COMPOSITION` ticket record this review. Other nodes/items are preserved.
  That local atlas is ignored/unshared; its update is not a published change.

## Observed checks

Logs and JUnit files are in the worktree's `artifacts/d04/`.

| Check | Observed result |
| --- | --- |
| `cargo fmt --check` | Passed, `fmt.log` |
| `CARGO_NET_OFFLINE=true cargo forge3d-clippy` | Passed, `clippy.log` |
| `maturin develop --release` with the repository feature list | Rebuilt and installed into the isolated worktree environment; release compilation 9m 54s, `bindings.log` |
| Combined composition/bundle/recipe/policy/manifest/cache/docs/API/typing/SUTURA/physical example | 280 passed, 1 example failure, zero skips, 53.32 s, `gates.xml` / `gates.log` |
| Corrected composition and physical example | 39 passed, zero skips, 28.93 s, `physical.xml` / `physical.log`; includes the failed example and all 38 composition checks |
| `scripts/assert_junit_zero_skips.py artifacts/d04/physical.xml` | Passed: 39 tests, no failures/errors/skips |
| Public stub typing | Passed with installed mypy, including both new public types and `render_passes` |
| Review-remediated composition/bundle/recipe/policy/manifest/cache/docs/API/typing/SUTURA/physical example | 315 passed, zero skips, 44.24 s, `review-final.xml` / `review-final.log` |
| Required-lane zero-skip assertion | Passed: 315 tests, no failures/errors/skips |
| Final output-depth/asset-path composition and bundle recheck | 76 passed, zero skips, 6.92 s, `review-final-path.xml` / `review-final-path.log`; zero-skip assertion passed |
| Exact audited-main reader against a v4 composition bundle | Rejected with `Bundle version 4 > supported version 3`, `legacy-reader.json` |
| Follow-up version isolation, summary, bundle/API/typing/SUTURA and physical replay gates | 313 passed, zero skips, 73.36 s, `followup-final.xml` / `followup-final.log`; zero-skip assertion passed |
| PR integration correction: composition/bundle/recipe/manifest/docs/API/typing/SUTURA, certificate/cache surface, inventory and fresh physical replay | 334 passed, zero skips, 29.80 s, `pr-final.xml` / `pr-final.log`; zero-skip assertion passed |

The combined run's sole failure was an empty example overlay caused by supplying
pixel coordinates to the native vector API, which uses identity view/projection
and normalized device coordinates. The corrected example passed. Unchanged API,
bundle and SUTURA results are reused; no failed test is treated as a pass.

Required hardware environment: `FORGE3D_NO_BOOTSTRAP=1`, `WGPU_BACKEND=vulkan`,
`FORGE3D_TESSELLA_REQUIRED_GPU=1`, `FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN=1`.
Windows, Python 3.13.14, NVIDIA GeForce RTX 3070, discrete physical GPU, Vulkan,
NVIDIA driver 610.60; `software_fallback=false`. The corrected hardware run was
outside the Windows sandbox after a sandboxed adapter probe crashed with
`0xc0000008`; no gate was weakened or skipped.

The current-main native binary SHA-256 was
`a84297b420116b383592f2dbb5bcc76d7e53485c15e91529008a338beb7a5368`.
An earlier existing binary lacked main's D02/splat symbols. Those API failures
disappeared after rebuilding current-main bindings; old-binary results are not
credited as current-main GPU evidence.

The first review CPU batch had 88 passes and seven sandbox temporary-directory
cleanup failures (`WinError 5`), all in unchanged bundle tests. Workspace-local
`TEMP`/`TMP` fixed that environment issue; the subsequent 315-test run passed.
No test was weakened or skipped. Native/Rust inputs are unchanged by the review
corrections, so the successful binding build, clippy and formatting evidence above
is reused.
The final recheck covers the added writer escape refusal and invalid output
depth refusal before pass replacement. The physical composition math and native
inputs were unchanged; their successful observed reference-pixel evidence is reused.

The follow-up's first run had 312 passes and one failure: the offline manifest
gate rejected a newly introduced `map_scene` import. This import was removed;
compact serialization is detected through the existing `to_dict` signature using
the standard library. The corrected 313-test run passed, including that gate and
fresh physical replay. No forbidden dependency was allowed or test weakened.

## Initial measured example (before review corrections)

256 × 192 output, native terrain color and material-relief renders, native
transparent vector OIT render, then explicit Python image composition.

| Executed work | Wall time |
| --- | ---: |
| Color render, including setup and PNG output | 2684.181 ms |
| Relief render | 158.221 ms |
| Vector overlay render | 21.864 ms |
| Public composition, including validation/compilation and PNG output | 5314.141 ms |

The terrain render calls themselves took 56.647 ms and 58.296 ms according to
existing `python_perf_counter` metadata. These are wall measurements around
executed rendering, not GPU timestamp or comparative performance claims.
Composition stores full input snapshots in JSON; these timings are a small
correctness fixture and establish no performance target.

Composed PNG and replay PNG SHA-256 both:
`f8ad1cae9bc48cf89949c811bcfb8849742836446b5aa153c6c271b74e057882`.
The validation reports match exactly. Source PNG hashes and full measurements
are in `artifacts/d04/pytest-physical/test_actual_render_outputs_com0/measurements.json`.
The independent CPU fixtures also prove canonical save/load/re-save of recipe,
compiled specification, report and manifest bytes, with destination metadata
held equal, without recompiling the loaded plan.

## Review corrections and measurements

All six pasted findings were addressed: old readers reject v4 compositions;
the active render/compile/bundle path no longer materializes pixel lists; output
eligibility is checked before replacing saved passes; snapshot/spec equality and
hashing work; disconnected graphs fail; explicit empty pass replacements fail.
Helper stubs are present. The owner approved the format change explicitly.

The same seed-0 random-input workload ran before and after, on this Windows/Python
environment, without a performance threshold or extrapolated acceptance claim.
`review-before.json` / `review-after.json` record executed wall times and file sizes:

| One input | Composition before → after | Bundle save before → after | Recipe + compiled JSON before → after | Binary snapshot |
| --- | ---: | ---: | ---: | ---: |
| 256 × 192 | 1.798 s → 0.037 s | 2.572 s → 0.053 s | 16,832,246 → 4,162 bytes | 1,572,992 bytes |
| 512 × 384 | 7.633 s → 0.118 s | 14.939 s → 0.042 s | 67,295,350 → 4,162 bytes | 6,291,584 bytes |

The remediated physical example composed its three actual GPU outputs in
53.118 ms wall time at 256 × 192. The composed/replayed PNG hash remains
`f8ad1cae9bc48cf89949c811bcfb8849742836446b5aa153c6c271b74e057882`;
pixels and reports are unchanged. These are wall measurements around executed
work. Independent reference pixels and contributions are asserted by
`test_mapscene_render_passes_gpu.py`; measurements are in
`artifacts/d04/pytest-review-final/test_actual_render_outputs_com0/measurements.json`.
The save/load/save fixture verifies exact snapshot, recipe, compiled specification
and manifest bytes. No poster-sized or throughput campaign has been run.

## Follow-up review corrections

Both new Low findings are corrected. `BundleManifest.load` defaults to the v3 cap;
only the MapScene reader opts into composition metadata v4 and verifies nonempty
passes. Empty/absent pass payloads with a v4 manifest are rejected. Generic bundle
loading and the viewer loader's rejection path are exercised under the repository's
existing deterministic Pro test fixture. No live viewer, IPC or production license
activation was exercised by that test. The exact audited-main reader again refused
v4 (`legacy-reader-followup.json`).

The MapScene-specific SUTURA future-version expectation remains 4 because that
reader explicitly supports composition v4. The generic reader's default-cap-3
expectation has its own regression test; reverting the MapScene expectation to 3
would misstate its supported format. Both still reject version 999.

`recipe_manifest()` uses compact serialization without importing a runtime scene
backend. It lists the pass specification, matches the compiled recipe hash, and
changes that hash when pixels change. Scene, recipe, compact mapping and inline
mapping inputs agree; the caller's mapping remains unchanged. Inline mappings
necessarily normalize their supplied pixels once; object summaries avoid that work.
The no-pass summary and hash remain unchanged. A signature capability check and
the existing offline import gate document and enforce the backend-free boundary.

On the same seed-0 512 × 384 single-input fixture, the public summary call took
2.899 s before and 0.679 ms after (`summary-before.json` /
`summary-after-final.json`). This is an executed wall measurement, with no latency
threshold or large-image claim. The final physical replay passed on the same
RTX 3070/Vulkan adapter; composed/replayed bytes and reports remain equal, with the
same PNG SHA-256 recorded above. This fresh 256 × 192 composition took 115.602 ms
wall time; no performance threshold is claimed. Current evidence is in
`artifacts/d04/pytest-followup-final/test_actual_render_outputs_com0/measurements.json`.

## Clean final review supplied by the owner — 2026-10-10

The supplied follow-up review reports no remaining findings and 142 focused tests
passed with no skips, covering composition, bundle round trips with the repository's
test Pro license, SUTURA, recipe manifests, typing, docs and public APIs. It confirms
that loader gates, compact summaries, hash agreement and documentation match the
implementation. This is reviewer-reported evidence, separate from the agent's
313-test run and observed physical GPU measurements above. The reviewer did not
rerun the GPU test.

At the clean review, local worktree HEAD was `08b5ad14`; its scoped changes and
owner-retained PDB were unchanged. No source corrections were needed after
this review. The ticket remains done, and broader D04 remains partial for
the unproven requirements listed below.

## Pull request preparation — 2026-10-10

The owner authorized committing, pushing and opening a mergeable pull request
into main, with green tests. Remote main was rechecked and still matched
`08b5ad1423d5c897938be8f96ada91a8a0765490`; no base integration changes were needed.
The reviewed source, tests and documentation are prepared on
`codex/d04-composition`. The generated PDB is retained locally, unchanged by this
preparation, and excluded from the commit. Hosted PR checks must be observed
before claiming integration readiness. No merge is authorized or performed.

The additional local fast profile ran 580 tests: 576 passed and four failed,
with no skips (`pr-fast.xml` / `pr-fast.log`, 322.27 s). Three inventory checks
ran before the new tests were staged; staging the intended files fixes their
tracked-inventory prerequisite. The public render-surface gate also identified
a missing `cache` keyword on `MapScene.render_passes`. The API and stub now
forward `certificate` and `cache` to ordinary native rendering on the no-pass
path and explicitly reject them for composition before replacing saved passes.
Regression tests cover forwarding and refusal without changing the gate or
adding an exclusion. All four failed gates and the composition, bundle, API,
typing, docs, SUTURA and physical replay gates then passed: 334 tests, zero skips,
29.80 s (`pr-final.xml` / `pr-final.log`). The unchanged 576 fast-profile checks
retain their observed passes; the hosted PR will run the complete fast profile.

The fresh physical replay used the same RTX 3070/Vulkan adapter and required
environment above. Composition took 32.895 ms wall time at 256 × 192; replay
pixels and reports match, with the same composed PNG hash recorded above.
Measurements are in
`artifacts/d04/pytest-pr-final/test_actual_render_outputs_com0/measurements.json`.
No Rust/WGSL/build/lint inputs changed, so the successful binding rebuild, clippy
and formatting evidence is reused.

## Boundaries, remaining checks and cleanup

No style generation, furniture implementation, render-graph rewrite, native
composition pipeline, new dependency, committed golden/certificate/hash update,
deployment or release was performed. Existing no-pass behavior
and canonical no-pass formats were retained. Composition format v4 was explicitly
approved by the owner. HDR/AOV output and native certificate,
provenance and cache controls are outside this explicit PNG compositor.

Hosted CI is pending during PR preparation. Linux/Metal physical execution, all six recipe-family campaigns,
competitor benchmarks and new golden/certificate acceptance were not run. They
are not claimed by this slice. D04 stays partial for those unproven broader claims.
Full licensed viewer operation and a broader native vector color-space design
review were not run; the native overlay fixture proves its declared raw linear
channel convention only.

The binding build changed tracked generated `python/forge3d/forge3d.pdb` in this
otherwise clean worktree. Automatic approval review rejected restoration from
HEAD because the supplied AGENTS.md prohibits editing generated products.
No workaround was attempted. On 2026-10-09 the owner explicitly instructed the
agent to leave that generated PDB unchanged. It remains modified as an accepted
build artifact; it is not an authored source change and the diff is not claimed
to contain only source files. No further cleanup or approval is pending.

Reusable steering gaps (proposals only): document the native vector API's NDC
coordinates beside its public signature; document that Cargo aliases ending in
`--` forward appended flags to rustc, so offline mode belongs in
`CARGO_NET_OFFLINE`; clarify whether an explicitly authorized binding rebuild
also authorizes restoring its tracked generated PDB. No agent contract or skill
was edited. No promoted project skills or nested contracts were present in this
current-main worktree; the supplied root contract governed the task.
Integration also showed that new test files must be staged before running the
tracked-inventory gate, and public render API changes need the certificate/cache
surface gate in their focused validation. These are documentation proposals;
no agent contract was changed.
