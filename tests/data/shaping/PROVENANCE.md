# HarfBuzz shaping goldens

The six JSON files contain 216 exact shaping cases (36 each for Latin,
Arabic, Hebrew, Devanagari, CJK, and mixed-direction text). They were generated
on 2026-07-13 with uharfbuzz 0.51.5, backed by HarfBuzz, against the committed
Noto subsets in `assets/fonts/`.

HarfBuzz fonts were scaled to `(64, 64)`, so `x_advance` and `x_offset` are the
same deterministic 1/64-em integers returned by LITTERA. Buffers were populated
with UTF-8 bytes, preserving byte-offset clusters. Default HarfBuzz features
were used; no application-specific feature overrides were supplied. RTL
HarfBuzz output arrays were reversed back to logical glyph order so the
goldens preserve LITTERA's deferred per-line UAX #9 contract.

| File | Cases | SHA-256 |
|---|---:|---|
| `latin.json` | 36 | `b94458b89279380d3d3c7094fa557267d6de7912d142bd1346746d9094ab4899` |
| `arabic.json` | 36 | `bcedf37c47b7d5e5b06a19cdddafaec79a262fd51646c0bb5f4f5bfa13f2ea51` |
| `hebrew.json` | 36 | `2d3c82099e29df1b1add5926a9b61fdb5fff6c2194aa7ec9576b5adb3688cd29` |
| `devanagari.json` | 36 | `e62e47911efd39298bbf7b9e065952b792b4585034fb90530e3737deab8eacd8` |
| `cjk.json` | 36 | `b23a6cd8f5ffb57ddce4e01929b70ecc7b2b7cde7a56f24bd862e615cf6d26c1` |
| `mixed.json` | 36 | `b53f74ca4e62e76a9e42738cd143bed78815a4664804dd5c855b7985842830a5` |
