<role>
You are Claude Fable 5 acting as a senior independent implementation auditor for forge3d.
</role>

<objective>
Rigorously audit whether every requirement in:

`docs/prompts/fable5-moonshots/14-censor.md`

is implemented in the current local repository.

For every numbered build section, public API/CI requirement, measurable win, and required validation item, assign exactly one status:

- `full`
- `partial`
- `none`

Then write an evidence-backed audit report to:

`docs/audits/fable5-moonshots/14-censor-implementation-audit.md`

The audit must expose the exact remaining engineering work. It must not repair the implementation.
</objective>

<why_this_matters>
CENSOR exists to make execution-honesty claims independently checkable. An audit that trusts comments, certificates without verification, permissive CI exclusions, constructor-time shader inventories, or tests that never run would reproduce the failure CENSOR is intended to prevent. The result must let a stranger determine whether forge3d can prove how a render executed, what degraded, which GPU resources and shaders it used, and whether CI can catch a lie.
</why_this_matters>

<context>
- Work in the current local `forge3d` checkout. Do not assume the implementation is on `main` or require a GitHub diff.
- Begin with `git status --short`. Treat relevant local changes as implementation evidence and do not revert them.
- Read `AGENTS.md`, but treat it only as historical context. Reflections and completion claims are not proof.
- Audit the repository as it exists locally. Follow wrappers, native registrations, callsites, render entrypoints, tests, generated certificates, and CI routing end to end.
- Use `C:\Users\milos\Downloads\fable-5-prompting-report.md` as prompting policy when available: hard objective, strict boundaries, externally verifiable evidence, no hidden chain-of-thought, and verification before finalizing. If unavailable, record that fact and continue.
</context>

<materials>
Primary requirements:
- `docs/prompts/fable5-moonshots/14-censor.md`

Audit-report boilerplate source:
- `docs/fable-5-p0-p1-blender-plan-implementation-audit-prompt.md`

Likely implementation evidence areas:
- `src/core/capabilities.rs`
- `src/core/gpu.rs`
- `src/core/gpu_timing.rs`
- `src/core/degradation.rs`
- `src/core/certificate.rs`
- `src/core/resource_tracker.rs`
- `src/core/memory_tracker/`
- `src/core/error.rs`
- `src/core/pipeline_scope.rs`
- `src/core/shader_registry.rs`
- `src/core/session.rs`
- `src/py_module/`
- `src/py_functions/`
- `src/terrain/renderer/`
- `src/viewer/render/main_loop/`
- `src/vector/`
- `python/forge3d/certificate.py`
- `python/forge3d/diagnostics.py`
- `python/forge3d/_degradation.py`
- `python/forge3d/mem.py`
- `python/forge3d/map_scene.py`
- `python/forge3d/offline.py`
- `python/forge3d/path_tracing.py`
- `python/forge3d/sdf.py`
- `python/forge3d/buildings.py`
- `python/forge3d/lighting.py`
- `python/forge3d/vector.py`
- `python/forge3d/smoke.py`
- `python/forge3d/geometry.py`
- `python/forge3d/__init__.py`
- `python/forge3d/*.pyi`
- `Cargo.toml`
- `pyproject.toml`
- `.cargo/config.toml`
- `.github/workflows/ci.yml`
- `scripts/ci_pytest_lane.py`
- `tests/test_render_certificate.py`
- `tests/test_render_certificate_contract.py`
- `tests/test_certificate_verifier.py`
- `tests/test_capability_negotiation.py`
- `tests/test_budget_enforce.py`
- `tests/test_device_init_failure.py`
- `tests/test_allocation_gate.py`
- `tests/test_pipeline_scope_gate.py`
- `tests/test_no_silent_degradation.py`
- `tests/test_dead_render_structure_gate.py`
- `tests/test_recipe_goldens.py`
- `tests/UNRUN.toml`
- `tests/degradation_allowlist.toml`
- `tests/allocation_allowlist.toml`
- `tests/golden/certificates/`
- `AGENTS.md`

Use this list only as a starting point. Follow imports, native/Python call boundaries, shader construction, render routing, and CI commands wherever the evidence leads.
</materials>

<boundaries>
- Modify only `docs/audits/fable5-moonshots/14-censor-implementation-audit.md`.
- Do not change source, tests, fixtures, certificates, goldens, CI, allowlists, prompts, or other docs.
- Do not regenerate certificates or goldens.
- Do not install dependencies, push branches, create PRs, or trigger external workflows.
- Do not run destructive commands.
- Do not reveal hidden chain-of-thought. Report only inspected evidence, concise rationale, assumptions, command outcomes, and conclusions.
- Do not mark a requirement `full` from intent, comments, type signatures, stubs, isolated helper tests, or the existence of a certificate field alone.
- Do not accept an allowlist or UNRUN entry merely because it is documented. Verify its scope, owner, expiry, and necessity.
- Do not treat a skipped golden job as a passing golden job. Distinguish `passed`, `absent`, `skipped(paths)`, and `not run`.
- Do not treat zero-valued synthetic pass timing as evidence of live GPU timestamps.
</boundaries>

<status_definitions>
Use these exact meanings:

`full`
- The required behavior is implemented on every public/intended path named by CENSOR.
- Native/Python/API/stub/registration/CI surfaces are connected where required.
- The evidence proves the requirement's definition of done, including negative and tamper cases where specified.
- Relevant focused tests pass in the environment actually being claimed.
- Any remaining work is optional polish, not needed for the stated requirement.

`partial`
- Real implementation exists, but a closure condition, public route, platform, test, CI proof, or measurable win is missing.
- Examples: a certificate is emitted by some renderers but not all; shader hashes describe ownership rather than actual use; timings are always zero; allocation evidence is process-lifetime instead of render-local; a fallback records outside the active capture; a Cargo feature is merely explained away rather than compiled in any CI lane; test files disappear from clean checkout; a golden mismatch cannot fail the aggregate; or required scratch-branch red-CI evidence is absent.

`none`
- No meaningful implementation was found for the requirement's core behavior.
- Stubs, dead code, docs-only claims, source-text tests, TODOs, and tests expecting future behavior do not count as implementation.
</status_definitions>

<required_requirement_inventory>
Before scoring, extract a complete inventory from `14-censor.md`. At minimum, enumerate and score separately:

1. Capability negotiation and removal of `Features::empty()`.
2. Optional-capability degradation recording.
3. Live nonblocking GPU timestamp behavior and pass population.
4. Adapter/device initialization error propagation, validation scopes, and device-loss poisoning.
5. `Session(backend=...)` honesty.
6. Total tracked buffer/texture allocation surface and source gate.
7. Enforce-by-default 512 MiB host-visible budget, error detail, and policy API.
8. Allocation-ledger invariant and render-local peak/by-label evidence.
9. Dead/duplicate memory-budget removal.
10. Complete `RenderCertificate` schema and render-entrypoint coverage.
11. Signed-field determinism, canonical JSON reuse, and Ed25519 tamper evidence.
12. Exact preprocessed WGSL hashes for shaders actually used by the render.
13. Single degradation sink and every minimum named fallback path.
14. Standalone verifier without the native module.
15. Honesty gates a–f, including expiring allowlists and exhaustive test accounting.
16. Cargo feature truth, wheel feature truth, and actual CI compile coverage.
17. Golden-lane pass/absent/fail semantics and aggregate enforcement.
18. Dead-structure decision and per-frame bind-group caching.
19. All seven measurable wins.
20. Adapter-init Python exception, clippy, formatting, build, focused tests, full CI lane, full-suite accounting, and red scratch-branch golden evidence.

If the source prompt contains additional independently testable requirements, add them. Do not merge distinct requirements merely to improve the status count.
</required_requirement_inventory>

<audit_method>
1. Read `14-censor.md` completely and build a traceability matrix from each requirement to expected implementation and verification evidence.
2. Inspect the actual code path end to end:
   - public Python method or function
   - wrapper/fallback behavior
   - PyO3 registration and native signature
   - Rust implementation and lifecycle
   - WGSL source preprocessing/use where relevant
   - certificate assembly/signing/verifier
   - tests and CI collection/routing
3. Grep all public pixel-producing render entrypoints and prove each either accepts the certificate contract or is explicitly outside CENSOR's render definition with evidence.
4. Grep every raw `device.create_buffer(` and `device.create_texture(` site. Independently confirm the gate's count and allowlist behavior.
5. Parse Cargo features, default-feature closure, maturin features, clippy features, and every CI `--features` command. Confirm every live feature is compiled by at least one real lane; prose-only exclusions do not count.
6. Compare `glob("tests/test_*.py")`, `git ls-files`, the default-lane selection, explicit lanes, and `UNRUN.toml`. Check for collection hooks or ignored example dependencies that make tests disappear.
7. Inspect committed certificates for schema, signatures, exact shader-use sets, render-local allocation evidence, pass timing semantics, and degradations.
8. Verify dead structures by caller search. Check surviving viewer/postfx code for per-frame resource creation.
9. Run the required verification commands where the environment permits. Use the freshly built/installed wheel for native-surface tests; do not mix updated Python with a stale extension.
10. Separate confirmed implementation facts, failed checks, skipped checks, unavailable external evidence, and inference.
</audit_method>

<required_dynamic_checks>
Run or faithfully adapt these checks, recording exact commands and outcomes:

```powershell
git status --short
cargo fmt --check
cargo forge3d-clippy
cargo test --workspace --features default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
maturin build --release
python -m pytest -q tests/test_render_certificate.py tests/test_render_certificate_contract.py tests/test_certificate_verifier.py tests/test_capability_negotiation.py tests/test_budget_enforce.py tests/test_device_init_failure.py tests/test_allocation_gate.py tests/test_pipeline_scope_gate.py tests/test_no_silent_degradation.py tests/test_dead_render_structure_gate.py tests/test_recipe_goldens.py
python scripts/ci_pytest_lane.py -q --tb=short
```

Also verify:
- two consecutive fixed-scene signed payload SHA-256 values are equal after excluding only permitted timing nondeterminism;
- one-byte certificate tampering fails verification;
- a shader-source mutation test changes the expected hash/golden result without committing the mutation;
- a 600 MiB host-visible request raises the named budget exception with the offending label and top consumers;
- raw allocation-site count is zero outside the tracker;
- all committed certificate degradations are empty or backed by a valid, necessary, non-expired allowlist entry;
- the standalone verifier works in an environment that cannot import `forge3d._forge3d`;
- the golden negative control really rejects a corrupted image even when baseline-update mode is available;
- the red scratch-branch CI link required by the source prompt exists and demonstrates a probe-positive golden mismatch failing the aggregate. If no link is available, mark that requirement no better than `partial`; do not substitute a local assertion for external CI evidence.

If a command cannot run, state exactly why and lower the affected status when the missing evidence is required for `full`.
</required_dynamic_checks>

<required_report>
Create `docs/audits/fable5-moonshots/14-censor-implementation-audit.md` with this structure:

```markdown
# CENSOR Implementation Audit

**Audit date:** YYYY-MM-DD
**Commit/branch:** ...
**Working tree:** clean|dirty, with relevant paths listed
**Overall status:** `full|partial|none`

## Executive Verdict

Concise, unsparing conclusion.

## Requirement Traceability

| ID | Requirement | Status | Implementation evidence | Verification evidence | Remaining coding/evidence |
| --- | --- | --- | --- | --- | --- |
| CENSOR-01 | ... | `full|partial|none` | paths and symbols | tests/commands | exact gap or `None` |

## Measurable Wins

One subsection for each of the seven wins, including observed values and command evidence.

## Public Render Surface Audit

List every discovered pixel-producing public entrypoint and its certificate behavior.

## CI and Test-Accounting Audit

Include test-file counts, default/explicit/UNRUN counts, expiries, feature compile coverage, and golden pass/absent/fail routing.

## Verification Log

| Command | Outcome | Evidence/notes |
| --- | --- | --- |

## Findings

Ordered by severity. Every finding must include file/symbol evidence, impact, and exact remediation.

## Final Counts

| Status | Count |
| --- | ---: |
| full | N |
| partial | N |
| none | N |
```

Every inventory item must appear exactly once in the traceability table. `Overall status` cannot be stronger than the weakest load-bearing measurable win.
</required_report>

<evidence_standard>
- Cite concrete paths and symbols; include line numbers where practical.
- A passing test is evidence only for the behavior it actually executes.
- Source-text assertions are supporting evidence, not substitutes for runtime behavior.
- A signed certificate is evidence only after independent signature verification and mutation rejection.
- A shader hash is evidence only if captured from preprocessed source actually bound/used during that render.
- A zero `gpu_ms` is not a live timing.
- A clean isolated test does not prove the whole CI lane is free of global-state leaks.
- A local untracked file cannot satisfy clean-checkout CI coverage.
- Clearly label inferences and missing external evidence.
</evidence_standard>

<verification_before_finalizing>
1. Read back the completed report.
2. Confirm every requirement inventory item appears exactly once.
3. Confirm every `full` status has implementation and dynamic verification evidence.
4. Confirm status counts sum to the traceability-row count.
5. Confirm no file except `docs/audits/fable5-moonshots/14-censor-implementation-audit.md` changed during the audit.
6. Run:

```powershell
git diff -- docs/audits/fable5-moonshots/14-censor-implementation-audit.md
git status --short
```
</verification_before_finalizing>

<output_format>
Final response must include only:
- audit report path
- overall status
- count of `full`, `partial`, and `none`
- highest-severity findings
- commands run and outcomes
- skipped checks, missing external evidence, or blockers
</output_format>
