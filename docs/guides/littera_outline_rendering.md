# Compiled label outlines

Call `MapScene.label_svg()` to export the labels visible in the current
compiled plan as SVG paths. The SVG viewBox matches the render viewport, so
the paths can be placed directly over the corresponding map image.

```python
from pathlib import Path

scene.compile_plan()
scene.render("map.png")
Path("map-labels.svg").write_text(scene.label_svg(), encoding="utf-8")
```

Labels keep their compiled glyph IDs, positions, orientations, colors and
round halos. Forge3D verifies the font SHA-256 values from the plan before
extracting the glyph outlines. Supply custom font paths in the label layer's
`glyph_atlas` binding. Bundled fonts are resolved by their recorded names.
An absent or changed font produces a `MapSceneTextLayoutError` diagnostic.

## Binding the packaged RGBA atlas

```python
from forge3d import LabelLayer
from forge3d.text_atlas import default_latin_atlas_paths

image, metrics = default_latin_atlas_paths(rgba=True)
labels = LabelLayer(
    "place-names",
    labels=[
        {"id": "town", "text": "Grindelwald",
         "geometry": {"type": "Point", "coordinates": [230, 120]}},
    ],
    typography={"font_size": 12, "halo_width_px": 1},
    glyph_atlas={
        "font_chain": [str(image.parent / "NotoSansLatin-subset.ttf")],
        "image_path": str(image),
        "metrics_path": str(metrics),
    },
)
```

The optional RGBA atlas contains a 96 px Latin bake with a range of 32 atlas
pixels. If a label's scale and halo exceed the encoded distance, native
rendering reports `insufficient_atlas_range` with the required bake range.
Bake an atlas from the matching font chain and shaped glyph IDs for other
scripts or custom fonts. The public atlas baker and default Latin asset
selection continue to default to RGB.

SVG-rendered outlines and native GPU glyphs use the same positions. The
fidelity suite re-rasterizes the exported SVG paths using the same analytic
coverage routine as its CPU reference. This measures the export round trip and
compares it with an independent GPU reconstruction.
