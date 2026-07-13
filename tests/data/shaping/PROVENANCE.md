# HarfBuzz shaping goldens

The six JSON files contain 216 genuinely distinct exact shaping expectations
(36 each for Latin, Arabic, Hebrew, Devanagari, CJK, and mixed-direction text).
They were generated
on 2026-07-13 with uharfbuzz 0.51.5, backed by HarfBuzz, against the committed
Noto subsets in `assets/fonts/`.

HarfBuzz fonts were scaled to `(64, 64)`, so `x_advance` and `x_offset` are the
same deterministic 1/64-em integers returned by LITTERA. Buffers were populated
with UTF-8 bytes, preserving byte-offset clusters. Default HarfBuzz features
were used; no application-specific feature overrides were supplied. RTL
HarfBuzz output arrays were reversed back to logical glyph order so the
goldens preserve LITTERA's deferred per-line UAX #9 contract.

Breadth is counted by distinct `(text, font chain, options)` identities and by
distinct expected run payloads, never by repeating one expectation at several
sizes. The corpus includes lam-alef and join controls, three Devanagari
conjunct patterns, language-driven `locl`, explicit `liga` enable/disable,
mixed bidi isolates and mandatory breaks, and ordered multi-font fallback.
Size invariance is locked separately in `tests/test_shaping_conformance.py`.

| File | Cases | SHA-256 |
|---|---:|---|
| `latin.json` | 36 | `4cad73c5c68ce7d4c8be5178a24effd9bb29974b79841a662c2946c3d5894013` |
| `arabic.json` | 36 | `b047a57631dca2c95821a69668b69613497d00d46ea1beb1cfe686e2a4a89371` |
| `hebrew.json` | 36 | `85cc93bdf8a22d8905a08c4924dacea9d636f5c73839a5894ee0ee3bcab8a8b8` |
| `devanagari.json` | 36 | `073c1407eeb13c9430f851c1e2dd54eb33cbcee8e09749deb55d092edcc31b7f` |
| `cjk.json` | 36 | `09557fddc9f1daa4ef4ab32032ad52d8b81b34588abc05f82f4df3ea6f9d916a` |
| `mixed.json` | 36 | `57d9099be1d1f0d1ee8c0c06f47a14bac073cdd933904febd7bfe4bbe25593d4` |
