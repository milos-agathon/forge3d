# Unicode table provenance

Version: Unicode 17.0.0, released 2025-09-09.

Official source base: `https://www.unicode.org/Public/17.0.0/ucd/`

| Input | SHA-256 |
| --- | --- |
| `Scripts.txt` | `9f5e50d3abaee7d6ce09480f325c706f485ae3240912527e651954d2d6b035bf` |
| `ArabicShaping.txt` | `39afa01e680e27d0fd10b67a9b27be13fbaa3d0efecfb5be45991de9a0d267d0` |
| `extracted/DerivedJoiningType.txt` | `f39ebe974825d6736aee15582250307aa532b2cfab3caf3f86bd23fddc9c5c4d` |
| `extracted/DerivedBidiClass.txt` | `4867b4b7f0731ed1bfcd34cc6251211ff1542541fce0734b6fbda139ee80b3a4` |
| `BidiMirroring.txt` | `a2f16fb873ab4fcdf3221cb1a8a85a134ddd6ed03603181823ff5206af3741ce` |
| `LineBreak.txt` | `e6a18fa91f8f6a6f8e534b1d3f128c21ada45bfe152eb6b1bcc5e15fd8ac92e6` |

`DerivedJoiningType.txt` supplies transparent marks omitted from
`ArabicShaping.txt`; the generator checks every explicit ArabicShaping joining
value against the derived property before emitting Rust.

Generation command, run with the six files above in `<ucd>`:

```text
python src/labels/unicode/generate.py --version 17.0.0 --scripts <ucd>/Scripts.txt --arabic-shaping <ucd>/ArabicShaping.txt --joining-type <ucd>/extracted/DerivedJoiningType.txt --bidi-class <ucd>/extracted/DerivedBidiClass.txt --bidi-mirroring <ucd>/BidiMirroring.txt --line-break <ucd>/LineBreak.txt --output src/labels/unicode/generated.rs
```

Generated UTF-8 bytes SHA-256:
`320d8cdfefe3d0e292ea2c00d831224c117a3c409212550090669b1c655592c6`.

Unicode data use is governed by the terms linked in each source file:
`https://www.unicode.org/terms_of_use.html`.
