# LITTERA Native Text Pipeline Design

**Source:** `docs/prompts/fable5-moonshots/18-littera.md`

## Goal

Make font outlines and a deterministic `ShapedRun` the single source for GPU text, CPU label composition, curved labels, and SVG/PDF export. The implementation must meet all six conformance wins in the source specification without adding shaping, rasterization, Unicode, or Python runtime dependencies.

## Architecture

The labels module owns one native pipeline:

1. `font` loads ordered font faces, selects named variable-font instances, resolves fallback through `cmap`, exposes metrics/GDEF data, and converts glyph outlines to `lyon_path::Path`.
2. `unicode` contains generated, versioned lookup tables for scripts, joining, bidi, mirroring, and line-break classes. Its checked-in generator is the only hand-maintained source transformation; builds perform no network access.
3. `shape` resolves bidi runs and line breaks, chooses OpenType script/language/features, executes the required GSUB/GPOS and legacy `kern` lookups, and returns deterministic 1/64-em `ShapedRun` records. Unsupported scripts or lookup types return structured errors; `.notdef` is never accepted as fallback.
4. A shared positioned-outline iterator combines each shaped glyph with the selected face and transform. GPU MSDF baking, analytic CPU coverage, curved placement, and SVG path serialization all consume this iterator.

Existing `ttf-parser`, `lyon_path`, `lyon_geom`, `lyon_tessellation`, NumPy/PyO3, and tracked GPU allocation helpers are reused. The existing outline adapter in `src/core/text_mesh/builder.rs` will be moved or shared instead of copied.

## Font and shaping contracts

`ShapedGlyph` stores `glyph_id`, source `font_index`, UTF-8 cluster byte offset, advances, and offsets as signed 1/64-em integers. `ShapedRun` stores glyphs, direction, script, language, and units-per-em metadata. Integer output makes serialization and cross-platform hashing stable.

Font fallback happens before shaping per character cluster, producing maximal same-face runs. Missing glyph errors include the Unicode code point and every font path tried. Named variable instances are applied through `ttf-parser` variation coordinates, including `fvar` axes and `avar` normalization exposed by the parser/raw tables.

OpenType parsing uses bounded big-endian readers over `RawFace` table bytes. Every offset/count access is checked. Lookup selection honors requested script/language, `DFLT`/`dflt`, required features, and caller feature overrides. GSUB types 1/2/3/4/6/7 and GPOS types 1/2/4/5/6/9 are implemented once in shared coverage/class/anchor/value-record primitives. Arabic joining enables contextual features from generated joining data; Devanagari preprocessing supplies the required matra/conjunct ordering. GPOS absence alone enables legacy `kern`.

UAX #9 is a standalone deterministic pass covering paragraph selection, explicit embeddings/overrides/isolates, weak and neutral resolution including brackets, implicit levels, reset/reordering, and mirroring. UAX #14 returns legal break offsets for multi-line labels and callouts. Conformance fixtures lock both algorithms.

## Rendering surfaces

The MSDF baker flattens shared outlines deterministically, detects corners, colors edges across RGB, computes signed pseudo-distance, and applies sign/collision correction. It records `px_range`, bake time, dimensions, and byte count; GPU upload uses the resource tracker and sets three channels.

CPU composition tessellates the same outlines and computes deterministic analytic pixel coverage in Rust. PyO3 returns a NumPy `float32` coverage array. Python only colors/composites this mask; Pillow, SciPy, bitmap fonts, and bit-noise fallbacks are removed from all text paths.

SVG export serializes positioned outlines as filled `<path>` geometry. Halo output is a stroked copy of the same path with paint order preserved. PDF may continue through CairoSVG because the SVG no longer depends on installed fonts.

Curved layout consumes shaped advances and offsets. Glyph centers sample arc length, tangents determine rotation, reversed paths are flipped upright, RTL traverses in the opposite sense, and marks inherit their base glyph transform.

## Python surface and integration

`forge3d.text` exposes exactly:

- `shape(text, font_chain, size, script=None, language=None, features=None) -> ShapedRun`
- `rasterize_shaped_run(...) -> numpy.ndarray[float32]`
- `bake_msdf_atlas(...)`

The native symbols travel together through registration, `__all__`, stubs, and API contract tests. `label_plan.py` calls the native shaper, accepts supported curved geometry, and propagates structured shaping diagnostics. `text_atlas.py` becomes a thin compatibility caller only if existing callers require the module; otherwise it is deleted.

## Validation strategy

Implementation proceeds in test-first vertical slices: font/outline primitives; OpenType shaping; Unicode bidi/line breaking; MSDF and analytic rasterization; curved layout; Python/MapScene/SVG integration; conformance assets and gates. Each behavior is observed failing before its minimal implementation.

The final evidence is the exact suite required by the source specification: at least 200 pinned HarfBuzz shaping goldens, at least 2,000 lines from each bidi conformance corpus, MSDF fidelity and single-channel ablation metrics, three-surface Delta E/SSIM comparisons, curved-label geometry/render evidence, hidden-dependency grep gates, deterministic hashes, tracked atlas metrics, `maturin develop --release`, formatting, curated Rust tests, `cargo forge3d-clippy`, focused Python tests, and the full Python suite.

## Scope and failure policy

LITTERA supplies letterforms, metrics, extents, outlines, and path-following transforms; it does not change label placement policy. Required scripts are Latin, Arabic, Hebrew, Devanagari, CJK, and mixed-direction text. Other Indic/SE Asian scripts may fail only through a structured diagnostic naming the script and missing lookup type. Vertical CJK, Mongolian, ruby, hinting, subpixel AA, and a native PDF writer remain out of scope exactly as specified.

Unrelated dirty worktree changes are preserved. Shared dirty registration files receive only additive LITTERA edits, and review/commit scopes name explicit paths.
