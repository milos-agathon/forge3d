# vulkan-forge — Project memory (Claude & humans)

This repository delivers a **Rust-first (wgpu) renderer** with a **thin Python API** for headless/interactive rendering of **terrain, vectors, and graphs**. Keep this file concise; put deep details in the nested memories listed below.

See @README for overview and @.claude/hooks.md for safety hooks. Sub-agents live in `.claude/agents/`; project settings/hooks are in `.claude/settings.json`.

---

## Ground rules for Claude in this repo
- Propose a short plan before edits. Prefer **small, reversible diffs**.
- Default to **headless**. Never introduce windowing/toolkit dependencies.
- Use the **specialist sub-agents** for focused work (terrain, tessellation, lines, points, graph, QA, wheels, CI, docs).
- Respect the **permissions** in `.claude/settings.json`. Dangerous commands are blocked by **PreToolUse** hooks.
- In CI or headless contexts, export `VULKAN_FORGE_PREFER_SOFTWARE=1` to prefer a software adapter.
- **Follow the Agent Editing Rules below** and always show diffs for every change.

---

## Agent Editing Rules (MUST FOLLOW)

**Always show diffs. Never guess. Modify files in place.**

### 1) Diffs required (no truncation)
- You **must** show unified diffs for every file you modify, create, or delete.
- Prefer these commands (Bash):
  ```bash
  git status --porcelain
  for f in $(git status --porcelain | awk '{print $2}'); do
    echo "### DIFF: $f"
    git --no-pager diff --unified=3 -- "$f"
    echo
  done
````

* PowerShell:

  ```powershell
  git status --porcelain
  $files = (git status --porcelain) -replace '^\S+\s+',''
  foreach ($f in $files) {
    "### DIFF: $f"
    git --no-pager diff --unified=3 -- "$f"
    "`n"
  }
  ```
* **If `git` is unavailable**, print a **synthetic unified diff** with +/- lines and \~5 lines of context around each hunk. Mark it clearly as synthetic.
* **Do not truncate diffs.** If output is long, split by file with headings.

### 2) Anchor-based edits (no guessing)

* Use precise, context-anchored edits. If an anchor is missing:

  1. **Stop** (don’t guess).
  2. Print what you searched, the nearest matching lines, and the intended edit.
  3. Ask for confirmation before proceeding.

### 3) Modify in place

* Do **not** create duplicate files.
* If you must create/delete a file, clearly show a diff header (`--- /dev/null` or `+++ /dev/null`) and explain why.

### 4) No silent success

* If no changes are required, print **“No changes required”** and why.

### 5) Determinism constraints to preserve (A1.2/A1.3)

* Render target: **`Rgba8UnormSrgb`** (use a `TEXTURE_FORMAT` constant).
* **No MSAA** (`count=1`), **no blending** (`blend: None`).
* **Fixed clear color** constant in render passes; **fixed viewport/scissor** covering the full target.
* Shader/geometry determinism:

  * Canonical triangle baked (no uniforms).
  * WGSL uses **`@interpolate(linear)`** for color; no branching in FS for edge conditions.
  * Primitive state: **CCW** front face + **back-face cull**.

### 6) Repository-specific build & test (for agents)

> Use these when your task needs to build/run locally. (Humans can use the “Build & test (fast path)” section below.)

**Bash (macOS/Linux/Git Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin
# abi3 forward-compat allows Python 3.13 with PyO3 0.21.x
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

maturin develop --release
cd python
python -m examples.triangle
python ../scripts/dev/edge_consistency.py
```

**PowerShell (Windows):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip maturin
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"

maturin develop --release
cd python
python -m examples.triangle
python ..\scripts\dev\edge_consistency.py
```

### 7) Required evidence per change

Every edit session must **print**:

1. `git status --porcelain` **before** and **after** edits (to prove scope).
2. Unified diffs for each changed file (`git --no-pager diff --unified=3 -- <path>`).
3. If anchors were missing, the diagnostics (nearest context + intended edit).
4. Final build/test outputs:

   * `maturin develop --release`
   * `python -m examples.triangle`
   * `python scripts/dev/edge_consistency.py` (should print identical coverage counts and “OK: coverage stable”).

### 8) Prompt snippets (reuse)

**Edit Task Skeleton**

```
<Task name="...">
  <Policy>
    - Follow /CLAUDE.md.
    - Modify files IN PLACE.
    - Show diffs for EVERY modified/created/deleted file (see /CLAUDE.md §Agent Editing Rules → Diffs).
    - If an anchor is missing, STOP and print diagnostics; do not guess.
  </Policy>
  <Edits>
    <!-- your anchored edits -->
  </Edits>
  <Commands shell="bash">
    <Run>git status --porcelain</Run>
    <Run>maturin develop --release</Run>
    <Run>python -m examples.triangle</Run>
    <Run>python scripts/dev/edge_consistency.py</Run>
    <Run>git status --porcelain</Run>
    <Run>for f in $(git status --porcelain | awk '{print $2}'); do echo "### DIFF: $f"; git --no-pager diff --unified=3 -- "$f"; echo; done</Run>
  </Commands>
</Task>
```

**Read-Only Verification Task**

```
<Task name="Verify determinism (READ-ONLY)">
  <Policy>
    - READ-ONLY. Do not modify files.
    - Print exact code blocks with file paths and line numbers.
  </Policy>
  <Requests>
    <!-- print ColorTargetState, RenderPass ops, Primitive state, buffer binds + draw call, constants + Renderer format, WGSL linear/no-branching -->
  </Requests>
</Task>
```

### 9) Common anchors (for this repo)

* ColorTargetState: `targets: &[Some(wgpu::ColorTargetState {`
* Primitive state: `primitive: wgpu::PrimitiveState {`
* Render pass ops: `ops: wgpu::Operations {`
* Buffer binds/draw: `rpass.set_pipeline(&pipeline);`
* Determinism constants: `const TEXTURE_FORMAT` / `const CLEAR_COLOR`
* WGSL: `@location(0) @interpolate(linear) color`

### 10) Commit etiquette (for humans)

* Squash agent changes into a single commit per task.
* Commit message should start with the roadmap item (e.g., `A1.3:`) and briefly list changed files and why.

---

## Build & test (fast path)

* Dev install:
  `maturin develop --release`
* Quick smoke:

  ```bash
  python - <<'PY'
  import vulkan_forge as vf
  r = vf.Renderer(512,512)
  print(r.info())
  a = r.render_triangle_rgba()
  print(a.shape, a.dtype)
  PY
  ```
* Unit & golden tests (determinism/SSIM):
  `pytest -q`
* **Note:** If using Python 3.13 locally with PyO3 0.21.x, set:

  * Bash: `export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
  * PowerShell: `$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1"`

---

## Wheels (manual local build)

* Linux (manylinux2014):
  `maturin build --release --strip --compatibility manylinux2014 -o wheels`
* macOS / Windows:
  `maturin build --release --strip -o wheels`
* Smoke install:
  `pip install --no-index --find-links wheels vulkan-forge`

---

## Determinism & color policy (must follow)

* Render target: **Rgba8UnormSrgb**. No MSAA in MVP.
* Shading in **linear**; rely on sRGB target for gamma on write. Avoid double-gamma.
* **Texture readback/copies**: when copying via buffer, ensure `bytes_per_row` is **multiple of 256**; strip padding on CPU.
* Avoid non-deterministic shader features (time, derivatives) unless using `textureSampleLevel(..., 0.0)`.

---

## Data contracts (MVP)

* **Terrain**: NumPy DEM `float32` `(H,W)`, C-contiguous. Upload as `R32Float`. Provide `(dx,dy)`, `h_min/h_max` (auto or user). Colormap LUT 256×1 `RGBA8 sRGB`.
* **Polygons**: packed `(coords f32 [x,y...], ring_offsets u32, feature_offsets u32)`. Exterior **CCW**, holes **CW**. Reject self-intersections/invalid nesting.
* **Lines**: packed `(coords f32, path_offsets u32)`. Screen-space width (px) via instanced quads; AA in fragment.
* **Points**: packed `(coords f32)`. Instanced SDF sprites; size/color can be arrays.
* **Graph**: nodes via points, edges via lines. Provide planar positions.

---

## Sub-agents to use

* **Core**: `rust-core`, `shader-smith`
* **Terrain**: `terrain-pipeline`
* **Tessellation**: `geo-tessellator`
* **Vectors/Graph**: `vector-lines`, `vector-points`, `graph-snapshot`
* **QA/Packaging/CI**: `determinism-qa`, `wheel-maker`, `ci-wright`
* **Docs/Tooling**: `docs-scribe`, `video-turntable`, `perf-profiler`, `device-diagnostics`, `security-licenser`, `release-captain`

---

## CI policy

* Matrix: Linux (manylinux2014 x86\_64), macOS (arm64 & x86\_64 or universal2), Windows (x86\_64).
* Gates: **SSIM ≥ 0.99** for unit goldens; **SSIM ≥ 0.98** cross-platform examples.
* Artifacts: wheels, example PNGs, and metrics JSON. Smoke test logs adapter/backend.
* **Agent compliance check:** CI jobs should fail if:

  * `blend != None` in any `ColorTargetState`,
  * render-pass clear is not `CLEAR_COLOR`,
  * edge consistency script reports mismatch.

---

## Security & publishing

* Never write secrets. Hooks block sensitive paths and dangerous shell patterns.
* Releases only via `release-captain` using **Trusted Publishing**. TestPyPI first; PyPI on tag.

---

## Nested memories (loaded contextually)

* @python/CLAUDE.md
* @src/CLAUDE.md
* @shaders/CLAUDE.md
* @ci/CLAUDE.md
* @docs/CLAUDE.md
* @.claude/hooks.md
