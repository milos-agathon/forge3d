"""Verify the fixture glyphs offline with uharfbuzz 0.51.5 / HarfBuzz 11.5.0."""

import json
from pathlib import Path

import uharfbuzz as hb


DATA = Path(__file__).resolve().parent
ROOT = DATA.parents[2]
PROJECTION = ("glyph_id", "font_index", "cluster", "x_advance", "x_offset")


def shape_run(case, run):
    font_index, = {glyph["font_index"] for glyph in run["glyphs"]}
    font_path = ROOT / case["fonts"][font_index]
    font = hb.Font(hb.Face(font_path.read_bytes()))
    font.scale = (64, 64)
    buffer = hb.Buffer()
    first, last = run["text_range"]
    buffer.add_utf8(case["text"].encode("utf-8"), first, last - first)
    buffer.direction = run["direction"]
    buffer.script = run["script"]
    if run["language"]:
        buffer.language = run["language"]
    hb.shape(font, buffer, case["options"].get("features", {}))
    glyphs = list(zip(buffer.glyph_infos, buffer.glyph_positions, strict=True))
    if buffer.direction == "rtl":
        glyphs.reverse()
    return [
        {
            "glyph_id": info.codepoint,
            "font_index": font_index,
            "cluster": info.cluster,
            "x_advance": position.x_advance,
            "x_offset": position.x_offset,
            "y_advance": position.y_advance,
            "y_offset": position.y_offset,
        }
        for info, position in glyphs
    ]


def main():
    assert (hb.__version__, hb.version_string()) == ("0.51.5", "11.5.0")
    count = 0
    for path in sorted(DATA.glob("*.json")):
        cases = json.loads(path.read_text(encoding="utf-8"))
        for case in cases:
            for i, run in enumerate(case["runs"]):
                glyphs = shape_run(case, run)
                assert [{k: glyph[k] for k in PROJECTION} for glyph in glyphs] == run["glyphs"], case["id"]
                if "mark_positions" in case:
                    assert glyphs == case["mark_positions"][i], case["id"]
        count += len(cases)
        print(path.name, len(cases))
    print(f"uharfbuzz {hb.__version__}, HarfBuzz {hb.version_string()}, {count} identities match")


if __name__ == "__main__":
    main()
