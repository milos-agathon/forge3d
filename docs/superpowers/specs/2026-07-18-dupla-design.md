# DUPLA Design

## Goal

Carry f64 world coordinates through WebGPU as normalized `(hi, lo)` f32 pairs, prove the implementation on the active backend, and use it in one opt-in absolute-coordinate demo that resolves millimetre camera motion at ECEF scale.

## Architecture

`scripts/generate_dd.py` is the single operation-order authority and generates the marked arithmetic sections of `src/core/dd.rs` and `src/shaders/includes/dd.wgsl`; `--check` fails on drift. The Rust file also owns f64 encoding/decoding, operation bounds, and deterministic operand generation. The GPU implementation lives under `src/core/dd/`: it assembles the shared include with small compute and render entry-point shaders, creates all buffers through the tracked resource API, and registers every assembled shader through the certificate shader registry.

The global GPU context runs the exactness canary when deterministic mode initializes. Other callers initialize the same process-wide DD capability lazily. Initialization compiles an FMA-selected candidate (`two_prod` directly calls WGSL `fma`) and accepts it only if the exactness canary passes; otherwise it records the rejected candidate and compiles the scaled split-selected candidate. Selection is fixed in the assembled source before shader compilation, not a runtime branch. A failed selected canary records `precision_selftest_failed/double_float` in the degradation sink. Harness and demo entry points check that capability and raise; there is no f32 fallback.

Python exposes `forge3d.precision.dd_selftest`, `dd_harness`, and `dd_jitter_demo`. The first two are also package-level native exports as required by the prompt. The demo report contains the two 1000-frame error curves and hashes from two identical DD renders.

## Arithmetic contract

- `DD { hi, lo }` is normalized with `quick_two_sum`; `DDVec3` stores three `DD` values.
- f64 encoding is `hi = value as f32`, `lo = (value - hi as f64) as f32`, then renormalization. `Anchor::to_dd` is the named world-coordinate crossing.
- `two_sum` is Knuth's six-operation error-free transform under round-to-nearest.
- `two_prod_split` is Dekker/Veltkamp with split constant 4097 and exact power-of-two scaling outside the safe splitter range so finite huge/tiny products cannot overflow the split intermediate. The source also defines a WGSL `fma` implementation. Rust assembles a direct call to the candidate selected before shader compilation; the canary chooses FMA only where it is demonstrably error-free, otherwise scaled split. The selected label is `dupla.dd.two_prod.<variant>` in the shader registry/certificate.
- `dd_add`, `dd_mul`, and `dd_div` cite Joldeș–Muller–Popescu 2017 bounds of 3u², 7u², and 15u². `dd_sqrt` uses the prompt's 15u² gate.
- Reciprocal uses an integer seed; square root uses the prompt-required WGSL `inverseSqrt` seed. Rust starts with its native f32 inverse-square-root seed. Both then run the generated fixed-order add/multiply Newton sequence and deterministically canonicalize the result by comparing residuals for the refined value and its adjacent f32 bit patterns, with a fixed tie rule. That final step makes the refined seed independent of backend seed ULPs; full-corpus bit lockstep is still a hard runtime gate. One DD correction follows. Their WGSL transformation/refinement bodies contain no `/` or `sqrt(`; the one seed call is isolated and source-gated.

## GPU proof flow

The canary shader consumes explicit pairs covering signed zero, normal/subnormal values, near-limit finite values, huge/tiny straddles, cancellation, and finite products. Readback checks `hi + lo` against the exact f64 sum/product and checks every returned bit against the Rust mirror, including the scaled-split extremes.

The bulk shader generates operands from the invocation index with integer-only xorshift state. A separate first phase runs 1,000,000 hand-built adversarial-family indices; the requested `n` phase then runs at least 100,000,000 mixed-sign, exponent `[-60, 60]`, mantissa, and cancellation-biased pairs. Each dispatch writes a bounded chunk. Rust maps the chunk, regenerates the identical operands, checks `(hi, lo)` lockstep, and reduces the maximum f64 relative error divided by `u²`. `n` must be positive and counts only the generated phase. Each report writes optional precision evidence into the existing execution/RenderCertificate payload and returns the same certificate object: adapter/backend, selected product variant, registered shader label/hash, random/adversarial counts, mismatch count, and per-operation measured maximum versus cited bound. The GPU matrix uploads these JSON certificates per backend.

## Absolute-coordinate demo

The demo shader has two vertex entries over identical ECEF DD position buffers. The DD entry performs `dd_sub_vec3(position, camera)` and narrows only the local residual before multiplying by the f32 view-projection. The ablation entry subtracts raw absolute f32 coordinates. A companion compute entry emits the screen coordinate for all 1000 one-millimetre camera steps so Rust can compare both curves with the f64 CPU projection.

The demo renders a deterministic Everest-scale point/triangle trace twice through the DD entry and returns both SHA-256 hashes. It uses `OneShotTiming`, records valid live GPU milliseconds, writes DD precision evidence into the existing certificate payload, and owns all tracked resources through RAII wrappers so early returns release ledger entries. Tests require equality, DD error below 0.01 px for every frame, raw-f32 error above 1 px for at least 100 frames, and no allocation-ledger growth after success or forced early failure. The pipeline is opt-in and does not alter the existing anchor default.

## Validation and refusal

Focused tests cover generator drift, source restrictions, Rust textbook/round-trip cases, GPU canaries, ≥10⁸-vector bounds, bit lockstep, API registration, structured forced failure in a fresh process, jitter curves, and determinism. The existing backend matrix runs the canary, all four bound operations, lockstep, and jitter evidence under its pinned backend and uploads the per-backend JSON certificate. Final gates are the exact commands from `30-dupla.md`, including release `maturin develop`, `cargo forge3d-clippy`, formatting, the curated Rust matrix, and focused pytest.

The public API page documents report keys, refusal behavior, copy-pasteable examples, costs, and the default `n`. No dependency, feature flag, fallback, general shader rewrite, or committed render artifact is added.
