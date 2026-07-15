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

- `atlas_latin_default.png` SHA-256: `c7c1de53801fd72cf7a0fd26d61377e6993412a652c488200f71796c53e885fa`
- `atlas_latin_default.json` SHA-256: `fc07dea94ade3492d83df01a8dc7c9efee8faca1b7e469edc1967529e8bd2b52`
- RGB byte count: `516000`
- Regeneration bake time observed on 2026-07-14: `43.4691 ms`
