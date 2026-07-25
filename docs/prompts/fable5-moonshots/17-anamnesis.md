# 17 — ANAMNESIS: The Content-Addressed Render — a build system for maps

**Track:** Singularity  ·  **Depends on:** TERRA-DETERMINATA (#04) for bit-exactness (without it, caching is unsound); CENSOR (#14) for the capability set and WGSL module hashes that must enter every cache key; CHRONOS (#06) consumes it.

You are working in the local `forge3d` repository.

## Objective

A 600-frame terrain flythrough takes hours. Change one label's text and forge3d re-renders all 600 frames from scratch, because a render is an opaque side effect rather than a pure function of its inputs. Build the machine that makes it a pure function: a content-addressed, Merkle-keyed frame graph where every pass's output is stored under `H(pass identity ‖ shader hash ‖ uniform bytes ‖ input keys ‖ negotiated capabilities)`, and a re-render recomputes **exactly** the passes whose key changed — nothing more, nothing less — while producing output **byte-identical** to a cold full render.

This is Bazel for cartography, and it is only possible in forge3d. Blender's Cycles cannot do it (no bit-exact reproducibility across a resumed session). Cesium and deck.gl cannot (no offline pass graph). Unreal cannot (nondeterministic frame state). forge3d can, because TERRA-DETERMINATA makes pixels a deterministic function of inputs and CENSOR makes the input set enumerable and hashable. The hard, load-bearing deliverable is **hermeticity**: proving that the cache key captures every input, so a stale hit is impossible — enforced by a mutation fuzzer that flips bytes in the input space and demands the key change 100% of the time. This also resurrects `src/core/framegraph_impl`, which is currently instantiated exactly once, in a diagnostics function.

## Operating rules

- Inspect the actual repository state; do not rely on memory. `git status --short` first; leave unrelated dirty files alone.
- Build after any Rust/WGSL change with `maturin develop` (or `--release`). Lint with `cargo forge3d-clippy` and format with `cargo fmt`.
- **No new dependencies.** Hashing is `sha2`; serialization is `serde_json` through the existing canonical encoder; the store is the filesystem.
- **Soundness beats speed.** A false cache hit is a catastrophic bug; a false miss is a performance bug. Every design tradeoff resolves toward recompute if unsure. A cache entry must be self-describing.
- The cache is **inert by default**: `cache=None` reproduces existing behaviour exactly, byte-for-byte.
- Respect the 512 MiB host-visible budget. The on-disk store is bounded by explicit `max_bytes` with LRU eviction.
- Register native symbols through `src/py_module/*`, `__all__`, `.pyi`, and `tests/test_api_contracts.py`.

## What to build

### 1. Resurrect the frame graph as the execution substrate

Route the offline render path (`src/offscreen/forward.rs` and the terrain draw/AOV encoders) through `framegraph_impl::FrameGraph`. Each pass declares a stable label, reads, writes, pipeline/WGSL module identity, and uniform blob. Compilation must produce the same execution plan for the same declaration set without depending on `HashMap` iteration order.

Do not convert the interactive viewer. Delete any unexecuted parallel post-processing chain rather than retaining a second fictional execution model.

### 2. Hard core: cache keys and hermeticity

Add `src/core/anamnesis/key.rs`. A pass key is:

```text
key(pass) = SHA256(
    "forge3d.anamnesis/1"
  ‖ pass.label
  ‖ pipeline_descriptor_hash
  ‖ uniform_bytes
  ‖ sorted(key(input) for input in pass.reads)
  ‖ capability_fingerprint
  ‖ engine_fingerprint
)
```

A resource key is the key of the pass that produced it. Leaf resources use SHA-256 over their content.

Enumerate every input capable of changing a pixel: sampler, blend/depth, viewport, scissor, clear, texture formats and mips, uniform padding, RNG seed, accumulation frame, negotiated features and limits, backend, DX12 compiler, and Naga capability policy. Inputs must retain semantic binding roles so swapping same-shaped resources changes the key. Anything excluded must be proven output-irrelevant in a committed ledger and rejecting test.

Uniform bytes are hashed after padding is initialized deterministically. Audit all `#[repr(C)]` uniform structs for bytemuck `Pod` and `Zeroable`.

### 3. Hard core: store and incremental scheduler

Add `src/core/anamnesis/store.rs`: a content-addressed filesystem store under `.forge3d/cache/<aa>/<hash>/{blob,meta.json}`. Metadata records the pass label, role-separated input keys, byte length, creation engine fingerprint, self-hash, LRU timestamps, and enough canonical derivation material to reconstruct the key.

The complete on-disk tree — blobs, metadata, directories, control files, and quarantine — is bounded by `max_bytes`. Corruption is quarantined and never served.

The scheduler walks the compiled graph bottom-up. On a hit it binds/restores the cached texture or buffer as that pass's output and skips the encoder. On a miss it executes and stores. Cached resources retain the same dependent transition/barrier plan as freshly encoded resources.

Report `CacheReport { hits, misses, bytes_read, bytes_written, wall_ms_saved }`.

### 4. Frame-sequence driver

`forge3d.anamnesis.render_sequence(recipe, frames=range(600), cache=".forge3d/cache")` renders a flythrough. Changing one label's text must recompute exactly the label-compile pass, label-composite passes on visible frames, and transitive dependents. Terrain, atmosphere, shadow, and accumulation must hit.

Compile and freeze an a-priori dry-run plan from the prior manifest and recipe diff before reading the store or executing an encoder. At the end, predicted and observed recompute sets must match exactly.

### 5. Cross-machine portability

Given TERRA-DETERMINATA's byte exactness, a store populated on one physical backend may serve another physical backend only through a declared deterministic compatibility profile whose fresh renders independently match the committed golden. A mismatch is a cache miss, never a wrong pixel. CI transfers a real production pass output, restores it as a GPU resource on the consumer, requires at least 99% hits, and checks the committed golden.

### 6. Invalidation diagnostics

- `forge3d.anamnesis.explain(key)` prints a reconstructible recursive derivation tree.
- `forge3d.anamnesis.gc(max_bytes)` performs bounded LRU collection.
- `forge3d.anamnesis.verify()` re-hashes entries and quarantines mismatches.

## Public API

- Rust: `src/core/anamnesis/{mod,key,store,scheduler,report}.rs`; deterministic framegraph pass declarations and key hooks; offline forward and terrain encoders routed through the graph.
- Python: `forge3d.anamnesis.render_sequence`, `explain`, `gc`, `verify`, and `CacheReport`.
- Every public render entry point accepts `cache: str | None = None`; unsupported paths conservatively recompute.
- WGSL changes are not required, but every pipeline has a stable label and preprocessed source identity.
- No new feature flags.

## Definition of done

All of the following are hard CI gates:

1. **Byte-identical incrementality.** Cold original 600-frame render, one-label edit, incremental render, and cold modified render. Every incremental modified frame hash equals its cold modified hash: 600/600, zero-byte tolerance.
2. **Exact recompute set.** Predicted and observed `(frame, pass_label)` sets are identical, with no strict superset or subset.
3. **Speedup.** Incremental wall clock is at most 1/20 of the cold modified render on the same machine.
4. **Hermeticity fuzz.** 10,000 seeded single-byte mutations across the entire declared production input space all change the top-level key. Rendered output must change unless the input appears in the committed irrelevant-input ledger with a checkable proof.
5. **No false hits.** Uniform-padding, sampler-filter, blend-mode, semantic-role-swap, and Naga-capability adversarial pairs produce distinct keys. Corrupting one blob causes quarantine and recomputation.
6. **Cross-machine portability.** A physical producer/consumer pair achieves at least 99% hits and the committed TERRA golden. A compatibility mismatch achieves 0% hits.

Also required: `cache=None` remains byte-identical on all committed goldens; `cargo forge3d-clippy` is clean; the frame graph has exactly one construction path and it belongs to production renderers, not diagnostics.

## Tests and validation

- `tests/test_anamnesis_incremental.py`: wins 1–3 and real 600-frame GPU gate.
- `tests/test_anamnesis_hermeticity.py`: seeded 10,000-mutation slow lane and 500-mutation fast lane.
- `tests/test_anamnesis_adversarial_keys.py`: key pairs, corruption, store bound, and Rust/Python interoperability.
- `tests/test_anamnesis_inertness.py`: `cache=None` and stale-hit rejecting controls.
- Rust: deterministic 100-shuffle compilation, graph scheduler restoration under barriers, real cached GPU texture restoration, uniform padding, and store quarantine.
- CI: required physical 600-frame and physical cross-machine portability lanes; backend-specific lanes may report `ABSENT` only for a proven lack of a physical adapter.

Validation commands:

```text
maturin develop --release
cargo forge3d-clippy
cargo fmt --check
cargo test --workspace --features <curated list> -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
python -m pytest tests/test_anamnesis_incremental.py tests/test_anamnesis_adversarial_keys.py tests/test_anamnesis_inertness.py tests/test_api_contracts.py -v --tb=short
python -m pytest -m slow tests/test_anamnesis_hermeticity.py -v
python -m forge3d.anamnesis explain <key>
```

## Non-goals

- No distributed cache, daemon, or server.
- No interactive-viewer conversion.
- No GPU-side memoization, pipeline cache, shader compilation cache, or provenance-signing duplication.
- No new storage or hashing dependency.
- Do not excuse a failing mutation by adding it to the irrelevant ledger without a written, checkable pixel-irrelevance argument.
- Do not touch unrelated in-flight dirty files.
