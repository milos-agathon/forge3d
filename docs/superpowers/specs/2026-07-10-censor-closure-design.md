# CENSOR Closure Design

**Date:** 2026-07-10
**Source:** `docs/prompts/fable5-moonshots/14-censor.md`
**Goal:** Close every requirement and every defect found by the 2026-07-10 implementation audit, then produce an independent Fable 5 audit prompt.

## Decisions

Work proceeds as bounded, test-first slices. Each slice must leave a runnable, reviewable repository state and must prove the relevant CENSOR requirements directly. Existing helpers are extended or deleted before any new abstraction is considered. No dependencies are added.

### 1. Render-local certificate state

The certificate must describe one render, not process lifetime. `begin_render_capture` will establish render-local allocation and degradation state; `finish_render_capture` will freeze only values observed in that capture. Consecutive identical renders on the same adapter must therefore have identical signed payloads even when earlier tests or renders allocated resources.

Shader hashes must be tied to the render capture rather than returned from an ever-growing process-global registry. The existing shader creation helper remains the source of hashes over preprocessed WGSL. Render paths will expose only the hashes belonging to the renderer/pass set used by that capture.

Canonical JSON will have one shared pure-Python implementation extracted from the existing recipe-manifest encoder. It will normalize negative zero and reject non-finite floats. Signing will cross the existing native boundary and use the repository's `ed25519-dalek`; the standalone verifier remains pure Python and importable without the native extension.

### 2. GPU truthfulness

Every production device request will route through the existing capability negotiation policy; production `Features::empty()` requests are removed. Pipeline creation will use the existing validation-scope helper, enforced by a source test covering every render and compute pipeline constructor. Device errors remain surfaced as structured render errors/degradations.

Interactive timing will poll without blocking and publish completed data from an earlier frame. Offline/reference rendering may retain an explicit blocking resolve because it is not on an interactive frame loop.

### 3. Allocation truthfulness

The existing tracked allocation wrappers remain the only allocation boundary. Texture byte accounting will use the whole descriptor: block dimensions, mip extents, array/depth layers, and sample count. The ledger will expose render-local peaks and maintain its existing sum invariant. No parallel budget or allocation system will be introduced.

### 4. One degradation and render contract

All actual fallback, placeholder, synthetic, and unsupported-success paths will write to the existing degradation sink. Shared fallback helpers will be fixed once and caller paths audited. Source and behavior tests will cover the named paths in CENSOR and prevent unrecorded siblings.

Every public render entry point will accept the documented `certificate=False|True|path` contract or will explicitly raise `DegradedCapability` when a documented native capability is unavailable. PyO3 registration, package exports, stubs, and API contract locks change together.

### 5. Delete dead architecture

The zero-caller legacy `PostFxChain` and `FrameGraph` surfaces will be deleted rather than routed into live rendering. Remaining viewer post-processing will cache bind groups with the resources that determine them, rebuilding only when those resources change. Every surviving pass remains represented in certificate pass records.

### 6. Literal repository gates

CI feature coverage will equal the set of live referenced Cargo features across its combined Rust and wheel jobs; it will not hide discrepancies behind an exclusion constant. Every filesystem `tests/test_*.py` file will be tracked and either collected by CI or represented by one current `UNRUN.toml` entry. The five ignored example tests and all failures from the complete suite are repaired rather than excluded.

Golden red proof must be probe-positive: a deliberately corrupted committed golden must reach the visual comparison, fail for the pixel mismatch, and produce a red scratch-branch CI run. The corruption is then reverted.

### 7. Final evidence and audit prompt

Completion requires the complete Python suite, curated Rust matrix, format/lint/native build, focused CENSOR tests, GPU certificate probes, terrain and recipe goldens, offline verification, and requirement-by-requirement evidence. Image quality claims use numeric comparisons and committed golden gates.

The final deliverable `docs/prompts/fable5-moonshots/14-censor-audit.md` will be based on `docs/fable-5-p0-p1-blender-plan-implementation-audit-prompt.md`, adapted to audit CENSOR's seven measurable wins, public surfaces, CI proof, and current repository state without trusting implementation intent.

## Slice order

1. Certificate determinism, canonicalization, native signing, and exact capture state.
2. Device negotiation, pipeline validation scopes, and non-blocking interactive timing.
3. Descriptor-accurate allocation accounting and render-local ledger evidence.
4. Degradation sink coverage and certificate kwargs on every render entry point.
5. Dead-structure deletion and bind-group lifetime correction.
6. CI feature/test accounting, complete-suite repairs, and golden red proof.
7. Full verification and the Fable 5 audit prompt.

Each slice starts with a failing test, implements the smallest shared/root-cause fix, runs focused verification, and receives a source/diff review before the next slice.
