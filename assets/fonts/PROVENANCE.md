# LITTERA conformance font subsets

These files are deterministic test subsets of the Noto fonts published in the
[Google Fonts repository](https://github.com/google/fonts/tree/main/ofl):

| Committed file | Upstream font | SHA-256 |
|---|---|---|
| `NotoSans-subset.ttf` | `ofl/notosans/NotoSans[wdth,wght].ttf` | `4d2f97d1ac7507089c27e3bd63b0b3bc07918122c8f5e72526d8aa593efd694a` |
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
