# LITTERA conformance font subsets

These files are deterministic test subsets of the Noto fonts published in the
[Google Fonts repository](https://github.com/google/fonts/tree/main/ofl):

| Committed file | Upstream font | SHA-256 |
|---|---|---|
| `NotoSans-subset.ttf` | `ofl/notosans/NotoSans[wdth,wght].ttf` | `4d2f97d1ac7507089c27e3bd63b0b3bc07918122c8f5e72526d8aa593efd694a` |
| `NotoSansLatin-subset.ttf` | `ofl/notosans/NotoSans[wdth,wght].ttf` | `3cdd82f05768b6101094b7daa24287053e07d6db5a21411d5076f5617f190e68` |
| `NotoSansArabic-subset.ttf` | `ofl/notosansarabic/NotoSansArabic[wdth,wght].ttf` | `65384a54d1272bb35b3780c480e353773f18965adaa8cf52dd8d6e358a521949` |
| `NotoSansHebrew-subset.ttf` | `ofl/notosanshebrew/NotoSansHebrew[wdth,wght].ttf` | `da63572372a6c876e38d9c17575527cab7752dd93dfa414fbbbb338d6ccca7e9` |
| `NotoSansDevanagari-subset.ttf` | `ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf` | `d6509fc1f37caf01184e8f9a4f150b606f2164a90df6354258adac42e125fb7c` |
| `NotoSansSC-subset.ttf` | `ofl/notosanssc/NotoSansSC[wght].ttf` | `a366764f4a666e85ba791d0eef9bbda8eca4a96e70287037d6c81c6b90ff8190` |

Downloaded from the `main` branch on 2026-07-13 and subset with fontTools
4.58.5 using the complete character repertoire exercised by
`tests/data/shaping/*.json`, retaining all
layout features, recommended glyphs, `.notdef`, glyph names, and canonical
Unicode cmaps. Lookup references outside LITTERA's required GSUB 1/2/3/4/6/7 surface
and GPOS tables covered independently by Tasks 5/6 were removed. Each adjacent
`*-OFL.txt` is the upstream SIL Open Font License 1.1.

## Task 10 Latin atlas subset and packaged MSDF

The shaping conformance subset intentionally contains only its pinned corpus,
so the atlas uses a separate ASCII subset without changing conformance glyph
IDs. The upstream variable font SHA-256 was
`bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d`.
It was generated with fontTools 4.58.5:

```powershell
$source = Join-Path $env:TEMP 'NotoSans-wdth-wght.ttf'
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf' -OutFile $source
pyftsubset $source --output-file=assets/fonts/NotoSansLatin-subset.ttf --unicodes=U+0020-007E --layout-features='' --drop-tables+=GSUB,GPOS,GDEF --glyph-names --symbol-cmap --legacy-cmap --notdef-glyph --notdef-outline --recommended-glyphs
```

The packaged RGB MSDF is regenerated with bake timing normalized to zero so
the committed JSON is byte-deterministic; measured timing is reported by the
same native result before normalization:

The five runtime subset TTFs are copied byte-for-byte into
`python/forge3d/data/fonts/` so wheels and sdists do not depend on a repository
checkout or system fonts; their hashes are identical to the source files above.

```powershell
$env:PYTHONPATH='python'
python -c "from forge3d.text_atlas import bake_atlas, save_atlas, default_latin_atlas_paths; p=default_latin_atlas_paths(); a=bake_atlas(font_size=24, px_range=6, padding=3); print(a.metrics['bake_ms'], a.metrics['byte_count']); a.metrics['bake_ms']=0.0; save_atlas(a, *p)"
```

- `atlas_latin_default.png` SHA-256: `2360e58d97aa24d436693d3481e87b3728b76b4c1adeb453dbd87e2a4e7f0290`
- `atlas_latin_default.json` SHA-256: `d28d280511fc89b5dae2cadb00f64ec48150561d59ceaeb5bc021d9c26b5dc67`
- RGB byte count: `516000`
- Runtime bake timing is diagnostic-only; `save_atlas` normalizes persisted
  `bake_ms` to `0.0` so canonical JSON is byte-stable.

## RGBA atlas for compiled labels

The opt-in atlas uses the same Latin subset as the default RGB atlas. Its RGB
planes hold the multichannel distance field and its alpha plane holds the
signed scalar distance field. The bake uses 96 px, range 32, and padding 2.
Both signed fields encode half the range on either side of the outline.

```powershell
python -c "from forge3d.text_atlas import bake_atlas, save_atlas, default_latin_atlas_paths; a=bake_atlas(font_size=96, px_range=32, padding=2, channels=4); print(a.metrics['bake_ms'], a.image.nbytes); save_atlas(a, *default_latin_atlas_paths(rgba=True))"
```

- `atlas_latin_rgba.png` SHA-256 `bec2e50df6787552ea42afb31ff2ddba4716638931eb617449598fd0ef47f2da`
- `atlas_latin_rgba.json` SHA-256 `e38d9f3dd879232e636a174a837c99a628025b2d7bdc5ae641a3ff937d432c6e`
- RGBA byte count 10,230,000 (1550 × 1650 × 4).

## Hebrew mark extension

`NotoSansHebrew-marks-subset.ttf` extends the cmap of
`NotoSansHebrew-subset.ttf` with U+05B0, U+05B8, U+05B9, U+05BC and U+05C1.
It preserves the font layout features, glyph names, canonical and legacy cmaps,
recommended glyphs and .notdef outline. The original subset remains available
for plans pinned to its content.

The upstream `ofl/notosanshebrew/NotoSansHebrew[wdth,wght].ttf`
file at Google Fonts revision `4c1913251a6dd1ba34a6ef4b7a630178d01b88ff`
has SHA-256 `7ef36a2c3593758cdb622e1bdef4f84523e92fbc3ccc667438dd80ff54c2de88`.
The subset SHA-256 is `06dffb6483277cb91ad4ecfefa5b2c8527399d2914e225b64ee6a0816f80244d`
in both the archive and the Python package. Use fontTools 4.58.5 with
`--no-recalc-timestamp --layout-features='*' --glyph-names --symbol-cmap --legacy-cmap --notdef-glyph --notdef-outline --recommended-glyphs`
and the union of the original cmap and the five code points above.
