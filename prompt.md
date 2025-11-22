I’ll first look up the exact text of the P5.5 – CLI & Viewer integration item in `todo-5.md`, then turn that into a very concrete, milestone-based requirement spec tailored for a high‑reasoning ChatGPT-5.1. I’ll also set up a small internal TODO list for this planning task so you can see the steps.

# P5.5 – CLI & Viewer Integration (End-to-End)  
**Requirements spec for a high‑reasoning ChatGPT‑5.1 implementor**

I’m basing this strictly on the P5.5 section in [todo-5.md](cci:7://file:///Users/mpopovic3/forge3d/todo-5.md:0:0-0:0) and the broader P5 context (P5.0–P5.4, Definition of Done, quick‑start commands), plus your preference for a single‑terminal viewer workflow (viewer + colon commands + snapshot).

---

## 0. Overall Goal

Provide a **fully wired, round‑trippable interface** between:

- **CLI flags** (`cargo run ... --gi ..., --ssao-*, --ssgi-*, --ssr-*`) and  
- **viewer runtime commands** (`:gi`, `:viz gi`, `:snapshot`, depth tools),

such that:

- Every GI‑related CLI flag has a **1:1 mapping** to a viewer command.
- Viewer state can be **queried and echoed** back as flags/commands.
- Running the **golden scripts** yields all P5 artifacts (PNG, opaque) deterministically.

---

## 1. Global Constraints & Invariants

- **No regression in P5.0–P5.4 behavior**:
  - With all GI effects disabled, the frame must remain **bit‑identical** to the pre‑P5 baseline.
  - Existing quick‑start commands in [todo-5.md](cci:7://file:///Users/mpopovic3/forge3d/todo-5.md:0:0-0:0) must still run and respect new flags.
- **Single‑terminal workflow preserved**:
  - Launch viewer from CLI, then accept text commands (`:gi`, `:snapshot`, etc.) in the **same terminal**.
  - Do not introduce blocking input patterns that break this model.
- **Deterministic output for golden runs**:
  - Same CLI flags + same seed ⇒ identical PNG bytes and hashes.
  - All snapshots used for P5 artifacts must be **opaque (no alpha)**.
- **Cross‑platform**:
  - CLI, scripts, and viewer commands work on Windows/macOS/Linux.
  - `.bat` and `.sh` variants behave equivalently.

---

## 2. Milestones

### Milestone 1 – CLI Flag Schema & Parsing (`src/cli/args.rs`)

**Objective**

Define and implement the **authoritative CLI schema** for all GI‑related options, including `--gi`, `--ssao-*`, `--ssgi-*`, `--ssr-*`, and any viz/snapshot flags needed to drive P5 artifacts.

**Requirements**

- **Flag inventory (minimum)**:
  - `--gi`:
    - Accept values consistent with [todo-5.md](cci:7://file:///Users/mpopovic3/forge3d/todo-5.md:0:0-0:0): e.g. `ssao`, `ssgi`, `ssr`, plus an explicit `off`.
    - Align with quick‑start examples, e.g. `--gi ssao:on` style, by:
      - Either: supporting a `mode:state` syntax (`ssao:on`, `ssgi:on`, `ssr:on`, `off`).
      - Or: defining a **single canonical syntax** and updating quick‑start docs accordingly.
    - Clearly define default when `--gi` is omitted.
  - `--ssao-*`:
    - At least: `--ssao-radius`, `--ssao-intensity`, `--ssao-technique`, `--ssao-composite`, `--ssao-mul`.
    - Respect ranges used in P5.1 acceptance (radius/intensity grids, etc.).
  - `--ssgi-*`:
    - At least: `--ssgi-steps`, `--ssgi-radius`, `--ssgi-half`, `--ssgi-temporal-alpha`, and any upsample/quality knobs required by P5.2.
  - `--ssr-*`:
    - At least: `--ssr-max-steps`, `--ssr-thickness`, and any additional knobs used by P5.3.
- **Args model**:
  - Extend the existing `Args`/config structs with explicit GI sections, e.g.:
    - `gi_mode: Option<GiMode>` (`Off | Ssao | Ssgi | Ssr | Composite`… as needed).
    - Nested structs for `SsaoParams`, `SsgiParams`, `SsrParams`, mirroring P5.1–P5.3 param sets.
  - Enforce **validation**:
    - Reject unknown values with clear error messages (printed to stderr, non‑zero exit code).
    - Clamp to valid ranges where appropriate; log warnings if clamped.
- **Round‑tripability**:
  - Provide a helper that can serialize an in‑memory GI state back into a **canonical** CLI string (for testing round‑trip in M5).

**Deliverables**

- Updated `src/cli/args.rs` with:
  - Full GI flag definitions and parsing.
  - Helper(s) for serializing GI config back to text form.

- Short inline documentation or enum doc comments (no new top‑level docs) describing value ranges and defaults.

**Acceptance**

- `cargo run --release --example interactive_viewer -- --gi ssao:on --ssao-radius 0.5 --ssao-intensity 1.0` parses successfully and sets expected in‑memory state.
- Invalid flags (e.g. `--gi xyz`) produce clear, actionable errors and non‑zero exit status.
- Round‑trip test sketch (to be formalized in M5): construct a GI config, serialize to flags, re‑parse, compare equality (except benign float rounding).

---

### Milestone 2 – Viewer GI State Wiring (`examples/interactive_viewer.rs`)

**Objective**

Wire the **CLI args → viewer runtime** and **viewer command → GI state** paths so they are consistent, queryable, and update the same underlying state.

**Requirements**

- **Initialization from CLI**:
  - On viewer startup:
    - Read `Args` GI config.
    - Initialize the GI manager / `Scene` / `GiParams` with these values **before** the first frame.
  - Guarantee CLI defaults ≡ viewer defaults when no flags/commands are given.
- **`:gi` command**:
  - Syntax:
    - `:gi off`
    - `:gi ssao`
    - `:gi ssgi`
    - `:gi ssr`
    - Optional: `:gi status` to print current mode and key parameters.
  - Behavior:
    - Switching mode updates the same GI state that CLI initialization uses.
    - Mode changes take effect on the next frame without restarting the viewer.
- **Parameter commands**:
  - Define colon commands that map 1:1 to the `--ssao-*`, `--ssgi-*`, `--ssr-*` flags:
    - Examples:
      - `:ssao-radius 0.5`
      - `:ssgi-steps 12`
      - `:ssr-thickness 0.2`
    - Exact command names must be isomorphic to CLI flag names (minus leading `--` and with `:` prefix).
  - Implement **state echo**:
    - A bare command (no argument) prints current value, e.g.:
      - `:ssao-radius` → logs `ssao-radius = 0.5`.
- **Synchronization invariant**:
  - All GI state is held in a **single source of truth** (e.g. a GI config struct):
    - CLI init populates it once.
    - Viewer commands mutate it.
    - Query commands and CLI round‑trip read from it.

**Deliverables**

- Updated `examples/interactive_viewer.rs`:
  - Parser/dispatcher for `:gi` and `:ssao-*`, `:ssgi-*`, `:ssr-*` commands.
  - Logging / stdout helpers for echoing state.

**Acceptance**

- Start viewer with a GI config via CLI, then:
  - Query via colon command (e.g. `:ssgi-steps`) and see an exact match.
  - Mutate a value via colon command; a subsequent query and a programmatic read via internal state both match.
- Toggling modes with `:gi` does not require restarting the viewer and does not break ongoing rendering.

---

### Milestone 3 – GI Visualization Commands (`:viz gi`)

**Objective**

Expose a **GI debug visualization** mode that can be controlled via both CLI and viewer, showing either:

- A **3‑channel composite view**: (AO, SSGI, SSR) mapped into RGB, or
- A **single selected channel** (AO only, SSGI only, SSR only).

**Requirements**

- **CLI**:
  - Add a `--viz-gi` flag (or equivalent) that selects:
    - `none` (default) – standard lit image.
    - `composite` – AO/SSGI/SSR packed into channels.
    - `ao`, `ssgi`, `ssr` – single‑channel debug views.
- **Viewer**:
  - Add `:viz gi` commands:
    - `:viz gi` – prints current GI viz mode.
    - `:viz gi composite`
    - `:viz gi none`
    - `:viz gi ao|ssgi|ssr`
  - Ensure `:viz` still supports existing non‑GI modes (normal|depth|material|gi) per P5.0, and `gi` now dispatches into the above sub‑modes.
- **Rendering path**:
  - Hook GI debug outputs that P5.1–P5.4 already compute:
    - AO buffer → AO channel.
    - SSGI result → SSGI channel.
    - SSR result → SSR channel.
  - Make this a **post‑composition visualization**:
    - Switching viz mode **must not change** the underlying GI buffers or their interaction with lighting; only the display path.

**Deliverables**

- Extended CLI args and viewer command handling for GI visualization modes.
- Wiring in the framegraph / viewer to route GI buffers into the display surface according to the selected viz mode.

**Acceptance**

- From CLI:
  - `--viz-gi composite` launches viewer and shows 3‑channel debug.
- From viewer:
  - `:viz gi ao` swaps to AO‑only display; switching back to `:viz gi none` returns to normal render.
- Switching between viz modes does **not** change final lit buffer when viz is `none` (verify via hash match).

---

### Milestone 4 – Snapshot & Deterministic Artifacts

**Objective**

Ensure `:snapshot` and CLI‑driven capture can produce **all P5 PNG artifacts** as **opaque, deterministic** images under scripted control.

**Requirements**

- **Snapshot semantics**:
  - `:snapshot`:
    - Captures the current frame to a PNG in `reports/p5/` when running under P5 golden scripts.
    - Guarantees:
      - 8‑bit RGB or RGBA with **alpha forced to 255** (fully opaque).
  - If a CLI `--output` or similar flag already exists, it may be reused, but P5 artifacts should still land in `reports/p5/` with the exact filenames listed in [todo-5.md](cci:7://file:///Users/mpopovic3/forge3d/todo-5.md:0:0-0:0) (P5.0–P5.4 deliverables).
- **Determinism**:
  - For GI modes that use randomness (blue noise, temporal accumulation):
    - Expose a seed or frame‑indexing scheme such that:
      - Given the same CLI arguments and seed, `:snapshot` of frame N produces identical PNG bytes across runs.
    - Golden scripts should **fix** any such seeds and frame counts.
- **Integration with GI state**:
  - Snapshots must reflect:
    - Current `--gi` mode.
    - All GI parameters (`--ssao-*`, `--ssgi-*`, `--ssr-*`).
    - Current `:viz` mode, if the artifact is a viz output (e.g., GI stack ablations).

**Deliverables**

- Snapshot plumbing in `interactive_viewer.rs` (or existing capture utilities) updated to:
  - Write correct PNG format.
  - Land in correct `reports/p5/` paths when invoked by golden scripts.
- Any minimal extension in CLI or viewer commands needed to select P5‑specific output filenames (e.g. `:snapshot p5_ssao_cornell`).

**Acceptance**

- For a fixed P5 golden command (see M5), rerunning the viewer twice yields:
  - Byte‑identical PNGs in `reports/p5/`.
  - Hashes match across runs.
- Inspecting PNG metadata (via a script) shows no transparency channel or alpha = 255 everywhere.

---

### Milestone 5 – Golden Scripts & CLI Tests

**Objective**

Encode the P5 pipeline’s end‑to‑end behavior into **golden scripts** and **CLI tests**.

**Requirements**

- **Golden scripts**:
  - `scripts/p5_golden.sh` and `scripts/p5_golden.bat` must:
    - Build the interactive viewer example as needed.
    - Invoke `cargo run --release --example interactive_viewer -- ...` with:
      - GI flags and params needed to produce:
        - All P5 PNGs listed in P5.0–P5.4 deliverables:
          - `p5_gbuffer_normals.png`, `p5_gbuffer_depth_mips.png`, `p5_gbuffer_material.png`.
          - `p5_ssao_cornell.png`, `p5_ssao_params_grid.png`.
          - `p5_ssgi_cornell.png`, `p5_ssgi_temporal_compare.png`.
          - `p5_ssr_glossy_spheres.png`, `p5_ssr_thickness_ablation.png`.
          - `p5_gi_stack_ablation.png`.
      - Use colon commands (via stdin or preconfigured script input) to:
        - Switch GI modes (`:gi ...`).
        - Set parameters (`:ssao-radius`, etc.).
        - Set viz mode (`:viz gi`).
        - Trigger `:snapshot` with the correct filenames.
    - Produce/update `reports/p5/p5_meta.json` as defined in P5.0–P5.4 (formats, timings, metrics).
- **CLI tests (`tests/test_p5_cli.rs`)**:
  - Test cases must at least cover:
    - Parsing each GI‑related CLI flag and mapping to the correct internal state.
    - Round‑trip tests:
      - Construct GI state in Rust, serialize to CLI string, parse back, ensure equivalence.
    - Interaction with viewer commands:
      - Simulate (or mock) viewer receiving `:gi`, `:ssao-*`, `:viz gi`, ensuring that the same config structure is updated as by CLI.
  - Tests should **not** depend on GPU presence (pure config + parsing where possible).

**Deliverables**

- `scripts/p5_golden.sh` and `.bat` committed and executable.
- `tests/test_p5_cli.rs` with:
  - Unit tests for arguments and round‑trips.
  - At least one end‑to‑end test wiring CLI args into a viewer config and asserting consistency.

**Acceptance**

- Running `scripts/p5_golden.sh` on a supported platform:
  - Produces all listed artifacts in `reports/p5/` with deterministic hashes.
  - Leaves `p5_meta.json` populated with GI‑relevant entries.
- `cargo test tests::test_p5_cli` (or equivalent) passes on CI on all platforms.

---

### Milestone 6 – End‑to‑End Verification & 1:1 Mapping

**Objective**

Verify that the full system satisfies the **P5.5 acceptance criteria** and the P5 Definition of Done.

**Requirements**

- **1:1 mapping**:
  - For each GI flag (`--gi`, `--ssao-*`, `--ssgi-*`, `--ssr-*`, `--viz-gi`):
    - There exists a viewer command (`:gi`, `:ssao-*`, `:ssgi-*`, `:ssr-*`, `:viz gi`) that:
      - Can set that value.
      - Can query that value.
  - For each viewer GI command:
    - There is a corresponding CLI representation in the arg schema.
- **State echo**:
  - For any supported GI parameter:
    - Sequence:
      1. Launch viewer with some CLI flags.
      2. Use a colon command with no argument (e.g. `:ssao-radius`) to print the value.
      3. Optionally change value via colon command; print again.
    - The printed value always matches the current config used in rendering and what would be emitted by the CLI serialization helper.
- **P5‑wide invariants**:
  - With `--gi off` (or equivalent):
    - Output frame is bit‑identical to pre‑P5 baseline for the same camera, scene, and seed.
  - Toggling GI modes or viz modes does not alter non‑GI aspects:
    - No unintended changes to PBR, IBL, or terrain beyond specified GI behavior.

**Deliverables**

- A short internal checklist (could be comments in `test_p5_cli.rs` or a small Markdown note) enumerating each flag/command pair and its tested behavior.
- Any additional micro‑tests or assertions needed to enforce bit‑identical behavior in GI‑off mode.

**Acceptance**

- Manual/automated audit confirms:
  - Every GI CLI flag ↔ viewer command mapping exists and works both ways.
- Running P5 golden scripts with GI off and with prior baseline scenes confirms:
  - Bit‑identical output in GI‑off mode.
- All P5‑related tests and golden checks pass on CI.


you must read fully @AGENTS.md to get familiar with my codebase. Next, you must carefully read prompt.md and then fully implement Step 1. Test after every change. Do not stop until you meet all the requirements for Step 1