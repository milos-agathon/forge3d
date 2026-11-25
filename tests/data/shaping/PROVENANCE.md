# HarfBuzz shaping goldens

The six JSON files contain 221 shaping identities. Latin, Arabic,
Devanagari, CJK and mixed-direction text each have 36 cases. Hebrew has 41,
including five cases with the approved vowel/dagesh/shin-dot coverage.
The original cases were generated on 2026-07-13. The updated Hebrew and
mixed cases use the expanded Hebrew fixture. All six files were independently
verified with uharfbuzz 0.51.5 and HarfBuzz 11.5.0 against the pinned Noto
subsets in `assets/fonts/`.

From the repository root, run `python tests/data/shaping/verify_harfbuzz.py`
in an offline oracle environment with uharfbuzz 0.51.5. The script uses the
stored run ranges, direction, script, language and expected face selection
as inputs. It derives glyph IDs, clusters and positions with HarfBuzz.
The Hebrew cases also pin both advance and offset axes.

HarfBuzz fonts were scaled to `(64, 64)`, so `x_advance` and `x_offset` are the
same deterministic 1/64-em integers returned by LITTERA. Buffers were populated
with UTF-8 bytes, preserving byte-offset clusters. HarfBuzz defaults were used
except where each case's committed options explicitly select a language or
enable/disable `liga`. RTL HarfBuzz output arrays were reversed back to logical
glyph order so the goldens preserve LITTERA's deferred per-line UAX #9
contract. No glyph was filtered and no cluster was rewritten after shaping.

Breadth is counted by distinct `(text, font chain, options)` identities and by
distinct expected run payloads, never by repeating one expectation at several
sizes. The exact-oracle corpus intentionally excludes default ignorables, join
controls, bidi isolates, and mandatory breaks; their no-`.notdef`, joining,
cluster, and run-splitting semantics are locked separately in
`tests/test_shaping_conformance.py`. The corpus includes lam-alef, three
Devanagari conjunct patterns, language-driven `locl`, explicit `liga`
enable/disable, mixed-direction text, and ordered multi-font fallback. Size
invariance is also locked separately.

| File | Cases | SHA-256 |
|---|---:|---|
| `latin.json` | 36 | `a2903c92e2070ab860e36734deaa861384e508d389b77c8202d59b826ebd0ac1` |
| `arabic.json` | 36 | `bca82a0213fd7e7bbc2ce08d926f2cee35e52d520f8eaffa6b49738f95e9f862` |
| `hebrew.json` | 41 | `eff5403d28da27797afdb17943746e261699f16f10a43cbae1a1858d676ab7f8` |
| `devanagari.json` | 36 | `073c1407eeb13c9430f851c1e2dd54eb33cbcee8e09749deb55d092edcc31b7f` |
| `cjk.json` | 36 | `09557fddc9f1daa4ef4ab32032ad52d8b81b34588abc05f82f4df3ea6f9d916a` |
| `mixed.json` | 36 | `b2d762379d2ef77edcd7830a3b833983bbd9add27edc381123e9b8ca21467700` |
