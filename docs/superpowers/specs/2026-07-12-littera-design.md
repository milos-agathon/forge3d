# LITTERA Native Text Pipeline Design

**Source:** `docs/prompts/fable5-moonshots/18-littera.md`

## Goal

Make font outlines and deterministic `ShapedText`/`ShapedRun` records the single source for GPU text, CPU label composition, curved labels, and SVG/PDF export. The implementation must meet all six conformance wins in the source specification without adding shaping, rasterization, Unicode, or Python runtime dependencies.

## Architecture

The labels module owns one native pipeline. The existing interim pipeline is replaced, not wrapped: remove `rustybuzz` and `unicode-bidi` from `Cargo.toml`/`Cargo.lock`, delete `src/py_functions/labels.rs::shape_text_py`, remove its native registration and `EXPECTED_FUNCTIONS` entry, migrate `label_plan.py` to `forge3d.text.shape`, and replace `tests/test_p2_complex_shaping_decision.py` assertions that require those dependencies with LITTERA conformance and structured-diagnostic assertions.

1. `font` loads ordered font faces, selects named variable-font instances, resolves fallback through `cmap`, exposes metrics/GDEF data, and converts glyph outlines to `lyon_path::Path`.
2. `unicode` contains generated, versioned lookup tables for scripts, joining, bidi, mirroring, and line-break classes. Its checked-in generator is the only hand-maintained source transformation; builds perform no network access.
3. `shape` resolves bidi runs and line breaks, chooses OpenType script/language/features, executes the required GSUB/GPOS and legacy `kern` lookups, and returns deterministic 1/64-em `ShapedText` records containing multiple visual `ShapedRun` segments. Unsupported scripts or lookup types return structured errors; `.notdef` is never accepted as fallback.
4. A shared positioned-outline iterator combines each shaped glyph with the selected face and transform. GPU MSDF baking, analytic CPU coverage, curved placement, and SVG path serialization all consume this iterator.

Existing `ttf-parser`, `lyon_path`, `lyon_geom`, `lyon_tessellation`, NumPy/PyO3, and tracked GPU allocation helpers are reused. The existing outline adapter in `src/core/text_mesh/builder.rs` will be moved or shared instead of copied.

## Font and shaping contracts

`ShapedGlyph` stores `glyph_id`, source `font_index`, UTF-8 cluster byte offset, advances, and offsets as signed 1/64-em integers. `ShapedRun` is one same-direction, same-script visual segment and stores glyphs, bidi level, direction, script, and language. `ShapedText` stores runs in final visual order plus legal line-break offsets. Mixed-direction curved text traverses each run according to its own direction while preserving visual run order; it is never collapsed to one paragraph direction.

Every shaped result owns an immutable native `FontCollection` (`Arc<[u8]>` per face), so later rasterization, atlas baking, and export cannot substitute font bytes. Its serializable descriptor records each face's SHA-256 over the exact bytes, TTC face index, and sorted variation coordinates as `(four-byte tag, signed 16.16 value)` pairs. Each `ShapedGlyph.font_index` resolves through that descriptor. APIs accepting a serialized result plus external fonts verify all fingerprints, face indices, and coordinates before consuming it.

Shaping integers use one exact normalization: `q26_6 = round_half_away_from_zero(font_units * 64 / units_per_em)`, evaluated with signed integer numerator/denominator so platforms agree. The public `size` must be finite and positive but does not alter glyph selection or stored shaping metrics; it is retained as the device transform and consumers compute pixels as `q26_6 * size / 64`. Thus changing `size` scales the same shaped geometry without changing its deterministic font-unit payload.

Font fallback happens before shaping per character cluster, producing maximal same-face spans inside each visual run. Missing glyph errors include the Unicode code point and every font path tried. `ttf-parser` 0.20 supplies typed axes and variation application but does not expose named instances; the font loader therefore parses bounded raw `fvar` instance records, resolves the requested instance name through `name`, applies every instance coordinate through `Face::set_variation` (which performs `avar` mapping), and records the resolved coordinates in the face descriptor.

OpenType parsing uses bounded big-endian readers over `RawFace` table bytes. Every offset/count access is checked. Lookup selection honors requested script/language, `DFLT`/`dflt`, required features, and caller feature overrides. GSUB types 1/2/3/4/6/7 and GPOS types 1/2/4/5/6/9 are implemented once in shared coverage/class/anchor/value-record primitives. Arabic joining enables contextual features from generated joining data; Devanagari preprocessing supplies the required matra/conjunct ordering. GPOS absence alone enables legacy `kern`.

UAX #9 is a standalone deterministic pass covering paragraph selection, explicit embeddings/overrides/isolates, weak and neutral resolution including brackets, implicit levels, reset/reordering, and mirroring. UAX #14 returns legal break offsets for multi-line labels and callouts. Conformance fixtures lock both algorithms.

## Rendering surfaces

The MSDF baker flattens shared outlines deterministically, detects corners, colors edges across RGB, computes signed pseudo-distance, and applies sign/collision correction. It records `px_range`, bake time, dimensions, and byte count; GPU upload uses the resource tracker and sets three channels.

CPU composition tessellates the same outlines and computes deterministic analytic pixel coverage in Rust. PyO3 returns a NumPy `float32` coverage array. Python only colors/composites this mask; Pillow, SciPy, bitmap fonts, and bit-noise fallbacks are removed from all text paths.

SVG export serializes positioned outlines as filled `<path>` geometry. Halo output is a stroked copy of the same path with paint order preserved. PDF may continue through CairoSVG because the SVG no longer depends on installed fonts.

Curved layout consumes shaped advances and offsets. Glyph centers sample arc length, tangents determine rotation, reversed paths are flipped upright, RTL traverses in the opposite sense, and marks inherit their base glyph transform.

## Python surface and integration

`forge3d.text` exposes exactly:

- `shape(text, font_chain, size, script=None, language=None, features=None) -> ShapedText`
- `rasterize_shaped_run(...) -> numpy.ndarray[float32]`
- `bake_msdf_atlas(...)`

The native symbols travel together through registration, `__all__`, stubs, and API contract tests. `label_plan.py` calls the native shaper, accepts supported curved geometry, and propagates structured shaping diagnostics. Existing callers require `text_atlas.py`, so it remains as a dependency-free compatibility module: `bake_atlas` delegates to `forge3d.text.bake_msdf_atlas`; `save_atlas` uses the in-tree PNG writer and stdlib JSON; `validate_atlas_metrics`, `load_atlas_metrics`, `default_latin_atlas_paths`, `BakedAtlas`, and `DEFAULT_LATIN_CHARSET` retain their current contracts. All Pillow/SciPy helpers and imports are deleted, and tests migrate from single-channel expectations to RGB MSDF metrics.

## Validation strategy

Implementation proceeds in test-first vertical slices: font/outline primitives; OpenType shaping; Unicode bidi/line breaking; MSDF and analytic rasterization; curved layout; Python/MapScene/SVG integration; conformance assets and gates. Each behavior is observed failing before its minimal implementation.

The final evidence is the exact suite required by the source specification: at least 200 pinned HarfBuzz shaping goldens, at least 2,000 lines from each bidi conformance corpus, MSDF fidelity and single-channel ablation metrics, three-surface Delta E/SSIM comparisons, curved-label geometry/render evidence, hidden-dependency grep gates, deterministic hashes, tracked atlas metrics, `maturin develop --release`, formatting, curated Rust tests, `cargo forge3d-clippy`, focused Python tests, and the full Python suite.

## Scope and failure policy

LITTERA supplies letterforms, metrics, extents, outlines, and path-following transforms; it does not change label placement policy. Required scripts are Latin, Arabic, Hebrew, Devanagari, CJK, and mixed-direction text. Other Indic/SE Asian scripts may fail only through a structured diagnostic naming the script and missing lookup type. Vertical CJK, Mongolian, ruby, hinting, subpixel AA, and a native PDF writer remain out of scope exactly as specified.

Unrelated dirty worktree changes are preserved. Shared dirty registration files receive only additive LITTERA edits, and review/commit scopes name explicit paths.
