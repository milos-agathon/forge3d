# LITTERA Native Text Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace forge3d's dependency-backed/bitmap text paths with one in-tree OpenType shaping and outline pipeline shared by GPU MSDF, CPU composition, curved labels, and SVG export.

**Architecture:** Rust owns immutable font collections, logical `ShapedText`, OpenType/Unicode processing, positioned outlines, MSDF baking, and analytic rasterization. Concrete line ranges are chosen before UAX #9 L1-L4 ordering; every rendering surface consumes the resulting positioned outlines and verifies exact font fingerprints.

**Tech Stack:** Rust, PyO3 0.21, NumPy 0.21, `ttf-parser` 0.20, `lyon_path`/`lyon_geom`/`lyon_tessellation` 1.0, existing SHA-256/resource tracking, Python stdlib plus NumPy.

## Global Constraints

- No new dependencies or feature flags; remove `rustybuzz` and `unicode-bidi`.
- No shaping/rasterization/Unicode crate and no Pillow/SciPy text path.
- `pyproject.toml` runtime dependencies remain exactly `["numpy>=1.21"]`.
- Store generated Unicode tables and their generator/version under `src/labels/unicode/`; never fetch at build time.
- Store shaping metrics as signed 1/64-em integers using `round_half_away_from_zero(font_units * 64 / units_per_em)`.
- `size` is finite and positive and affects only device scaling: `pixels = q26_6 * size / 64`.
- Keep `ShapedText` in logical order; apply UAX #9 L1-L4 only after concrete line ranges exist.
- Missing glyphs and unsupported scripts/lookups raise structured diagnostics; never emit `.notdef`, reversed codepoints, or tofu silently.
- Reuse tracked texture allocation; all three surfaces consume the same shaped glyphs and outline source.
- Preserve unrelated dirty worktree files and stage only explicit LITTERA paths.

---

### Task 1: Font collection, fingerprints, metrics, variation instances, and outlines

**Files:**
- Create: `src/labels/font/mod.rs`
- Create: `src/labels/font/face.rs`
- Create: `src/labels/font/fvar.rs`
- Create: `src/labels/font/outline.rs`
- Modify: `src/labels/mod.rs`
- Modify: `src/core/text_mesh/builder.rs`
- Test: Rust unit tests inside the new modules

**Interfaces:**
- Produces: `FontCollection::load(&[FontRequest]) -> Result<FontCollection, TextError>`; `FaceDescriptor { sha256: [u8; 32], face_index: u32, variations: Vec<(Tag, i32)> }`; `FontCollection::glyph_for(char) -> Result<FontGlyph, TextError>`; `FontCollection::outline(font_index, GlyphId) -> Result<Path, TextError>`.

- [ ] **Step 1: Write failing unit tests** for ordered fallback, missing-codepoint diagnostics, stable SHA-256 descriptors, half-away normalization, raw named-instance parsing, and reuse of the shared outline adapter.

```rust
#[test]
fn q26_6_rounds_half_away_from_zero() {
    assert_eq!(to_q26_6(1, 128), 1);
    assert_eq!(to_q26_6(-1, 128), -1);
}

#[test]
fn missing_glyph_names_codepoint_and_chain() {
    let err = collection.glyph_for('\u{10FFFF}').unwrap_err();
    assert!(err.to_string().contains("U+10FFFF"));
    assert!(err.to_string().contains("NotoSans-subset.ttf"));
}
```

- [ ] **Step 2: Verify RED.** Run `cargo test labels::font --lib --features extension-module`; expect unresolved `labels::font`/missing API failures.

- [ ] **Step 3: Implement the bounded font core.** Move `PathSink` from `text_mesh/builder.rs` into `labels/font/outline.rs`; use it from both callers. Parse TTC face indices, `head`/`hhea`/`OS_2`, horizontal/vertical metrics, GDEF glyph/mark classes, and raw `fvar` instance records. Hash exact bytes with `crate::core::provenance::sha256`; sort fixed 16.16 variation coordinates by tag.

```rust
pub fn to_q26_6(value: i32, units_per_em: u16) -> i32 {
    let n = i64::from(value) * 64;
    let d = i64::from(units_per_em);
    let q = n / d;
    let r = n % d;
    (q + if r.abs() * 2 >= d { n.signum() } else { 0 }) as i32
}
```

- [ ] **Step 4: Verify GREEN.** Run the Task 1 Rust test command, `cargo fmt --check`, and `maturin develop --release` because Rust changed.

- [ ] **Step 5: Commit.** `git add src/labels/font src/labels/mod.rs src/core/text_mesh/builder.rs && git commit -m "feat(text): add deterministic font collection"`.

### Task 2: Generated Unicode data and lookup API

**Files:**
- Create: `src/labels/unicode/mod.rs`
- Create: `src/labels/unicode/generated.rs`
- Create: `src/labels/unicode/generate.py`
- Create: `src/labels/unicode/UCD_VERSION`
- Create: `src/labels/unicode/PROVENANCE.md`
- Modify: `src/labels/mod.rs`
- Test: Rust unit tests in `src/labels/unicode/mod.rs`

**Interfaces:**
- Produces: `script(char) -> Script`; `joining_type(char) -> JoiningType`; `bidi_class(char) -> BidiClass`; `mirrored(char) -> Option<char>`; `line_break_class(char) -> LineBreakClass`.

- [ ] **Step 1: Write failing tests** for Arabic joining data, bracket mirroring, isolate classes, Devanagari/CJK scripts, and representative UAX #14 classes.

```rust
#[test]
fn generated_tables_cover_required_scripts() {
    assert_eq!(script('क'), Script::Devanagari);
    assert_eq!(joining_type('ب'), JoiningType::DualJoining);
    assert_eq!(mirrored('('), Some(')'));
    assert_eq!(bidi_class('\u{2067}'), BidiClass::Rli);
}
```

- [ ] **Step 2: Verify RED.** Run `cargo test labels::unicode --lib --features extension-module`; expect missing module failures.

- [ ] **Step 3: Add the generator and checked-in tables.** The generator accepts local UCD text-file paths and a required `--version`, coalesces adjacent equal-value ranges, and writes deterministic Rust arrays. `PROVENANCE.md` records exact Unicode version, file SHA-256 values, and command. Commit generated output; runtime/build scripts read no network.

- [ ] **Step 4: Verify GREEN and reproducibility.** Run the Rust tests, then rerun `generate.py` against the recorded inputs and assert `git diff --exit-code -- src/labels/unicode/generated.rs`.

- [ ] **Step 5: Commit.** `git add src/labels/unicode src/labels/mod.rs && git commit -m "feat(text): add generated Unicode tables"`.

### Task 3: OpenType table reader and feature selection

**Files:**
- Create: `src/labels/shape/mod.rs`
- Create: `src/labels/shape/types.rs`
- Create: `src/labels/shape/ot.rs`
- Create: `src/labels/shape/layout.rs`
- Modify: `src/labels/mod.rs`
- Test: unit tests in `src/labels/shape/ot.rs` and `layout.rs`

**Interfaces:**
- Produces: `ShapedGlyph`, logical `ShapedRun`, `ShapedText`, `TextError`; checked `Reader`; `LayoutTable::parse`; `selected_lookup_indices(script, language, features)`.

- [ ] **Step 1: Write failing tests** for truncated offsets/counts, coverage formats 1/2, class definitions 1/2, `DFLT`/`dflt` fallback, required feature inclusion, and explicit feature disable/enable.

```rust
#[test]
fn reader_rejects_out_of_bounds_offsets() {
    assert!(Reader::new(&[0, 1]).slice_at(4, 2).is_err());
}

#[test]
fn required_feature_precedes_optional_features() {
    assert_eq!(table.selected_lookup_indices(tag(*b"arab"), None, &[]), vec![3, 7, 9]);
}
```

- [ ] **Step 2: Verify RED.** Run `cargo test labels::shape::ot labels::shape::layout --lib --features extension-module`.

- [ ] **Step 3: Implement only shared checked parsing and selection.** Use big-endian `u16/i16/u32` readers, no unsafe code, and exact OpenType offset bases. Define `ShapedText` with logical runs, per-character levels, legal breaks, `Arc<FontCollection>`, descriptors, and size; do not add rendering behavior.

- [ ] **Step 4: Verify GREEN**, format, and run `maturin develop --release`.

- [ ] **Step 5: Commit.** `git add src/labels/shape src/labels/mod.rs && git commit -m "feat(text): add OpenType layout core"`.

### Task 4: GSUB execution and Arabic/Devanagari preprocessing

**Files:**
- Create: `src/labels/shape/gsub.rs`
- Create: `src/labels/shape/arabic.rs`
- Create: `src/labels/shape/devanagari.rs`
- Modify: `src/labels/shape/mod.rs`
- Test: unit tests in all three modules

**Interfaces:**
- Produces: `apply_gsub(face, buffer, selection)` supporting lookup types 1/2/3/4/6/7; `arabic_features(chars)`; `reorder_devanagari(chars)`.

- [ ] **Step 1: Write failing tests** for single/multiple/alternate/ligature substitution, extension lookup dispatch, a hand-built type-6 chained-context table, Arabic joining states including transparent marks, lam-alef through `rlig`, pre-base i-matra, and three Devanagari conjunct patterns.

```rust
#[test]
fn chained_context_applies_nested_lookup() {
    let table = fixture_type6_backtrack_input_lookahead();
    assert_eq!(apply_fixture(&table, &[10, 20, 30]), vec![10, 99, 30]);
}

#[test]
fn arabic_joining_skips_transparent_marks() {
    assert_eq!(arabic_forms("ب\u{064E}ب"), vec![Form::Initial, Form::None, Form::Final]);
}
```

- [ ] **Step 2: Verify RED.** Run `cargo test labels::shape::gsub labels::shape::arabic labels::shape::devanagari --lib --features extension-module`.

- [ ] **Step 3: Implement minimal required lookup executors.** Preserve UTF-8 cluster offsets through substitutions; ligatures keep the minimum contributing cluster; multiple substitutions inherit their source cluster. Unknown required lookup types return `TextError::UnsupportedLookup { table: "GSUB", lookup_type, script }`.

- [ ] **Step 4: Verify GREEN**, format, build the extension.

- [ ] **Step 5: Commit.** `git add src/labels/shape && git commit -m "feat(text): implement required GSUB shaping"`.

### Task 5: GPOS execution and legacy kerning

**Files:**
- Create: `src/labels/shape/gpos.rs`
- Modify: `src/labels/shape/mod.rs`
- Test: unit tests in `src/labels/shape/gpos.rs`

**Interfaces:**
- Produces: `apply_gpos` for lookup types 1/2/4/5/6/9 and `apply_legacy_kern` only when GPOS is absent.

- [ ] **Step 1: Write failing tests** for value-record fields, glyph/class pair positioning, mark-to-base, mark-to-ligature component selection, mark-to-mark, extension dispatch, and `kern` fallback suppression when GPOS exists.

```rust
#[test]
fn mark_to_base_uses_anchor_delta() {
    let out = position_fixture(mark_to_base_fixture(), &[base(10), mark(20)]);
    assert_eq!((out[1].x_offset, out[1].y_offset), (120, 340));
}
```

- [ ] **Step 2: Verify RED.** Run `cargo test labels::shape::gpos --lib --features extension-module`.

- [ ] **Step 3: Implement checked anchor/value parsing and positioning.** Accumulate in font units, normalize once with Task 1's integer helper, attach marks without advancing, and retain base/ligature association for curved transforms.

- [ ] **Step 4: Verify GREEN**, format, build.

- [ ] **Step 5: Commit.** `git add src/labels/shape && git commit -m "feat(text): implement required GPOS shaping"`.

### Task 6: Complete bidi and line breaking with deferred visual ordering

**Files:**
- Create: `src/labels/shape/bidi.rs`
- Create: `src/labels/shape/linebreak.rs`
- Modify: `src/labels/shape/mod.rs`
- Create: `tests/data/unicode/BidiCharacterTest.txt`
- Create: `tests/data/unicode/BidiTest.txt`
- Create: `tests/data/unicode/PROVENANCE.md`
- Create: `tests/test_bidi_conformance.py`

**Interfaces:**
- Produces: `resolve_bidi(text, paragraph_level) -> BidiParagraph`; `visual_order(levels, line_ranges) -> Vec<usize>`; `line_breaks(text) -> Vec<usize>`.

- [ ] **Step 1: Write failing Rust rule tests** for P2-P3, X1-X8 including isolates, W1-W7 individually, N0-N2 individually, I1-I2, mirroring, and L1-L4 with two different line partitions of the same logical text.

- [ ] **Step 2: Add the Python conformance harness** that parses committed Unicode test syntax and asserts at least 2,000 lines from each file with zero mismatches.

```python
def test_bidi_conformance(native):
    results = run_bidi_files(native, DATA_FILES)
    assert results.lines_by_file["BidiCharacterTest.txt"] >= 2000
    assert results.lines_by_file["BidiTest.txt"] >= 2000
    assert results.failures == []
```

- [ ] **Step 3: Verify RED.** Run the focused Rust bidi tests and `python -m pytest tests/test_bidi_conformance.py -v`.

- [ ] **Step 4: Implement UAX #9/#14.** Store logical resolved levels in `ShapedText`; apply L1-L4 only in `visual_order` for caller-supplied line ranges. Curved/unwrapped callers pass `[0..text.len()]`. Implement the UAX #14 pair/state rules needed by the committed full required class data, including mandatory breaks and combining marks.

- [ ] **Step 5: Verify GREEN**, reporting exact line totals and zero failures.

- [ ] **Step 6: Commit.** `git add src/labels/shape tests/data/unicode tests/test_bidi_conformance.py && git commit -m "feat(text): implement Unicode bidi and line breaking"`.

### Task 7: End-to-end shaper, public PyO3 surface, and dependency removal

**Files:**
- Modify: `src/labels/shape/mod.rs`
- Create: `src/labels/py_text.rs`
- Modify: `src/labels/mod.rs`
- Modify: `src/py_functions/labels.rs`
- Modify: `src/py_module/functions/labels.rs`
- Create: `python/forge3d/text.py`
- Create: `python/forge3d/text.pyi`
- Modify: `python/forge3d/__init__.py`
- Modify: `python/forge3d/__init__.pyi`
- Modify: `python/forge3d/label_plan.py`
- Modify: `tests/test_api_contracts.py`
- Modify: `tests/test_p2_complex_shaping_decision.py`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Create: `tests/test_shaping_conformance.py`
- Create: `tests/data/shaping/PROVENANCE.md`
- Create: `tests/data/shaping/*.json`
- Create: `assets/fonts/*-subset.ttf`
- Create: `assets/fonts/*-OFL.txt`
- Create: `assets/fonts/PROVENANCE.md`

**Interfaces:**
- Produces: `forge3d.text.shape(...) -> ShapedText`; native `text_shape`; structured Python exceptions with `diagnostics`.

- [ ] **Step 1: Add failing API and shaping conformance tests.** Require at least 200 exact HarfBuzz golden cases across Latin, Arabic, Hebrew, Devanagari, CJK, and mixed-direction text; compare glyph IDs, UTF-8 clusters, x advances, and x offsets as integers.

```python
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_shape_matches_harfbuzz(case):
    shaped = forge3d.text.shape(case["text"], case["fonts"], case["size"], **case["options"])
    assert shaped.to_dict()["runs"] == case["runs"]
```

- [ ] **Step 2: Verify RED.** Run shaping/API/old complex-shaping tests and confirm missing new API plus old dependency assertions.

- [ ] **Step 3: Wire the shaper and migrate callers.** Replace `shape_text_py` with the three LITTERA native functions' registration seam, expose immutable PyO3 `ShapedText`, route `label_plan.py` through `forge3d.text.shape`, delete `_ARABIC_FORMS`/`_shape_arabic_run`/unsupported short-circuit, and preserve structured unsupported-script errors.

- [ ] **Step 4: Remove `rustybuzz` and `unicode-bidi`.** Regenerate `Cargo.lock`; update `EXPECTED_FUNCTIONS` from `shape_text` to `text_shape`, `rasterize_shaped_run`, `bake_msdf_atlas`; replace old tests rather than skip them.

- [ ] **Step 5: Verify GREEN.** Run `maturin develop --release`, shaping/API tests, `cargo tree -i rustybuzz` and `cargo tree -i unicode-bidi` (both must report no package), format, and clippy.

- [ ] **Step 6: Commit.** Stage only the listed Task 7 paths and commit `feat(text): replace dependency-backed shaper`.

### Task 8: Positioned outlines, analytic rasterization, and SVG outline serialization

**Files:**
- Create: `src/labels/positioned.rs`
- Create: `src/labels/raster.rs`
- Modify: `src/labels/py_text.rs`
- Modify: `src/export/svg_labels.rs`
- Modify: `src/export/mod.rs`
- Modify: `python/forge3d/_map_scene_render.py`
- Modify: `python/forge3d/export.py`
- Create: `tests/test_text_three_surfaces.py`
- Modify: `tests/test_export_svg.py`

**Interfaces:**
- Produces: `positioned_outlines(shaped, line_ranges) -> Iterator<PositionedOutline>`; `rasterize_shaped_run(...) -> PyArray2<f32>`; SVG `<path>` data generated from the same iterator.

- [ ] **Step 1: Write failing tests** asserting deterministic coverage, line-dependent bidi order, SVG contains `<path` and no `<text`, halo uses the same path data, and CPU composition contains neither `_draw_text_fallback` nor Pillow text drawing.

```python
def test_svg_labels_are_outlines(scene):
    svg = scene.export_svg()
    assert "<path" in svg
    assert "<text" not in svg
```

- [ ] **Step 2: Verify RED** with focused Rust/Python export and three-surface tests.

- [ ] **Step 3: Implement positioned outlines and deterministic analytic coverage.** Apply chosen line ranges, UAX L1-L4, glyph advance/offset, font selection, and size transform once. Flatten/tessellate with fixed tolerance; compute 8x8 deterministic subpixel analytic coverage accumulated into `f32` output. Python colors and alpha-composites the returned mask.

- [ ] **Step 4: Rewrite SVG labels.** Serialize lyon events with locale-independent decimal formatting, filled paths, and a stroked identical path for halos; retain CairoSVG PDF routing.

- [ ] **Step 5: Verify GREEN**, format/build, and commit `feat(text): share outlines across CPU and SVG`.

### Task 9: True RGB MSDF atlas and tracked GPU upload

**Files:**
- Create: `src/labels/msdf/mod.rs`
- Create: `src/labels/msdf/edge.rs`
- Create: `src/labels/msdf/distance.rs`
- Create: `src/labels/msdf/atlas.rs`
- Modify: `src/labels/atlas.rs`
- Modify: `src/labels/mod.rs`
- Modify: `src/labels/py_text.rs`
- Modify: `src/shaders/text_overlay.wgsl`
- Create: `tests/test_msdf_fidelity.py`

**Interfaces:**
- Produces: `bake_msdf_atlas(collection, glyphs, size, px_range, padding) -> BakedMsdfAtlas`; RGB image, metrics, bake milliseconds, and byte count.

- [ ] **Step 1: Write failing Rust geometry tests** for contour orientation, sharp-corner detection, three-color edge assignment, signed pseudo-distance, and collision correction.

- [ ] **Step 2: Write failing fidelity tests** requiring 12px IoU >= 0.995, Hausdorff <= 0.5px, 96px SSIM >= 0.999, and a single-channel ablation Hausdorff > 0.5px.

- [ ] **Step 3: Verify RED.** Run focused MSDF tests and capture failing metrics.

- [ ] **Step 4: Implement Chlumsky-style MSDF.** Flatten shared contours, split at angle-threshold corners, color adjacent segments across RGB, evaluate signed pseudo-distance per channel, correct sign/collisions, and pack deterministic atlas cells. Keep `px_range` configurable and record timing/memory.

- [ ] **Step 5: Make the shader branch live.** Upload RGB through `tracked_create_texture`, change `set_channels(1)` to `set_channels(3)`, preserve median/fwidth smoothing, and add a Rust assertion that atlas metadata and renderer channel count agree.

- [ ] **Step 6: Verify GREEN** with `maturin develop --release`, fidelity metrics, shader tests, allocation gate, format, clippy.

- [ ] **Step 7: Commit.** `git add src/labels/msdf src/labels/atlas.rs src/labels/mod.rs src/labels/py_text.rs src/shaders/text_overlay.wgsl tests/test_msdf_fidelity.py && git commit -m "feat(text): bake true tracked RGB MSDF"`.

### Task 10: Dependency-free `text_atlas.py` compatibility

**Files:**
- Modify: `python/forge3d/text_atlas.py`
- Modify: `python/forge3d/text_atlas.pyi`
- Modify: `tests/test_text_atlas.py`
- Modify: `python/forge3d/data/fonts/atlas_latin_default.png`
- Modify: `python/forge3d/data/fonts/atlas_latin_default.json`

**Interfaces:**
- Preserves: `BakedAtlas`, `DEFAULT_LATIN_CHARSET`, `bake_atlas`, `save_atlas`, `validate_atlas_metrics`, `load_atlas_metrics`, `default_latin_atlas_paths`.

- [ ] **Step 1: Change tests first** to require RGB channels=3, native bake metadata, `_png.save_png` round-trip, and zero `PIL`/`scipy` tokens in `text_atlas.py`.

- [ ] **Step 2: Verify RED.** Run `python -m pytest tests/test_text_atlas.py -v`.

- [ ] **Step 3: Reduce the module to adapters.** `bake_atlas` resolves the default bundled OFL font when `font_path is None` and delegates to `forge3d.text.bake_msdf_atlas`; `save_atlas` calls `_png.save_png` plus `Path.write_text(json.dumps(...))`; retain current validation helpers unchanged except channels=3.

- [ ] **Step 4: Regenerate the packaged default atlas deliberately** from the committed font subset and record its command/hash in font provenance.

- [ ] **Step 5: Verify GREEN** and commit `feat(text): retire Python bitmap atlas baker`.

### Task 11: Curved shaped text and MapScene reachability

**Files:**
- Modify: `src/labels/curved.rs`
- Modify: `src/labels/line_label.rs`
- Modify: `src/labels/mod.rs`
- Modify: `src/labels/optimal.rs`
- Modify: `python/forge3d/label_plan.py`
- Modify: `python/forge3d/_map_scene_labels.py`
- Create: `tests/test_curved_labels.py`
- Modify: `tests/test_p2_complex_shaping_decision.py`

**Interfaces:**
- Produces: curved placement from `ShapedText` plus one full-range line; removes `compute_glyph_advances` heuristic export.

- [ ] **Step 1: Write failing tests** for center-sampled tangent rotations, <=0.25px normal deviation, upright reversed paths, RTL opposite traversal, mark/base shared rotation, and `MapScene` acceptance/render of a curved Arabic river label.

- [ ] **Step 2: Verify RED.** Run Rust curved tests and `python -m pytest tests/test_curved_labels.py tests/test_p2_complex_shaping_decision.py -v`.

- [ ] **Step 3: Feed shaped advances to curved layout.** Reorder the full text as one line, accumulate centers using real advances, sample tangent at each center, normalize angles upright, reverse RTL arc progression, and reuse base transforms for attached marks. Delete `compute_glyph_advances` and the curved `unsupported_geometry_type` rejection.

- [ ] **Step 4: Render and commit the required curved-Arabic sample** under `tests/golden/labels/littera_curved_arabic.png` using the real MapScene path.

- [ ] **Step 5: Verify GREEN**, report max deviation, build/format/clippy, and commit `feat(text): render shaped curved labels`.

### Task 12: Three-surface conformance, hidden-dependency gates, and determinism

**Files:**
- Complete: `tests/test_text_three_surfaces.py`
- Create: `tests/test_text_no_hidden_deps.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md` only if this implementation produces a reusable hard-won lesson

**Interfaces:**
- Produces: six CI-gated measurable wins and cross-platform `ShapedText` hash output.

- [ ] **Step 1: Complete the three-surface metric test** for committed Latin/Arabic/Devanagari straight and curved labels. Compare GPU↔CPU, GPU↔SVG, CPU↔SVG with Delta E 2000 pass fraction >=0.99 and SSIM >0.99.

```python
for left, right in combinations((gpu, cpu, svg), 2):
    covered = np.maximum(left[..., 3], right[..., 3]) > 0
    assert np.mean(delta_e2000(left, right)[covered] < 2.0) >= 0.99
    assert ssim(left_bbox, right_bbox) > 0.99
```

- [ ] **Step 2: Add source/dependency gates.** Assert `_draw_text_fallback` count=0; `ImageFont`, `PIL`, and `scipy` count=0 across text paths; runtime dependencies exactly NumPy; forbidden Cargo packages absent; SVG label output contains no `<text>`; deterministic shaped fixture SHA-256 equals the committed hash.

- [ ] **Step 3: Verify RED**, then make only integration corrections exposed by these tests; do not weaken thresholds or add skips.

- [ ] **Step 4: Add CI commands** for focused LITTERA tests and cross-platform shaped-hash comparison using the existing determinism matrix.

- [ ] **Step 5: Verify GREEN** for all six wins and commit `test(text): gate LITTERA conformance`.

### Task 13: Full verification, review bundle, code review, and final commit

**Files:**
- Evidence only outside repository: `%TEMP%/forge3d-littera-review-*`
- Modify only files required by review findings

**Interfaces:**
- Consumes all prior tasks; produces final verification evidence and a scoped change set.

- [ ] **Step 1: Run fresh formatting/build/lint.** `cargo fmt --check`; `maturin develop --release`; `cargo forge3d-clippy`.

- [ ] **Step 2: Run curated Rust tests.** `cargo test --workspace --features default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile`.

- [ ] **Step 3: Run focused Python tests.** `python -m pytest tests/test_shaping_conformance.py tests/test_bidi_conformance.py tests/test_msdf_fidelity.py tests/test_text_three_surfaces.py tests/test_curved_labels.py tests/test_text_no_hidden_deps.py tests/test_api_contracts.py -v --tb=short`.

- [ ] **Step 4: Run full Python suite.** `python -m pytest tests/ -v --tb=short`; regenerate only intended label goldens and document every changed baseline.

- [ ] **Step 5: Capture required numbers.** Record shaping exact/total and HarfBuzz provenance, bidi lines/failures, MSDF IoU/Hausdorff/SSIM and ablation, three pairwise Delta E fractions/SSIMs, curved deviation/sample path, grep counts, atlas bake time/memory, and cross-platform hash.

- [ ] **Step 6: Build the slice review bundle outside the repo** with scoped status/diffs/metadata and real validation logs, then use the code-review skill. Fix findings test-first and rerun affected commands.

- [ ] **Step 7: Confirm final scope.** `git status --short` must distinguish pre-existing unrelated files from LITTERA files; `git diff --check` must be clean.

- [ ] **Step 8: Commit any review fixes** with explicit LITTERA paths and `git commit -m "fix(text): close LITTERA review findings"`.

## Plan self-review

- Spec coverage: Tasks 1-12 map to every font, shaping, Unicode, MSDF, curved, three-surface, public API, asset, dependency, tracker, and six-win requirement; Task 13 maps every required verification command/evidence item.
- Contract consistency: `ShapedText` remains logical throughout; visual ordering requires chosen line ranges; `ShapedGlyph.font_index` resolves immutable fingerprinted face bytes; all device scaling uses the same q26.6 formula.
- Scope: label placement, vertical CJK, Mongolian, ruby, hinting, subpixel AA, and native PDF remain excluded.
