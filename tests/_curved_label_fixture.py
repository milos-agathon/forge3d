"""A screen-label fixture shared with the physical main comparison."""
import math
import numpy as np

import forge3d as f3d
from forge3d import map_scene


def curved_label_scene(angle_deg):
    angle = math.radians(angle_deg)
    path = [[80.0, 96.0], [80.0 + 240.0 * math.cos(angle),
                          96.0 + 240.0 * math.sin(angle)]]
    layer = f3d.LabelLayer(
        layer_id="river", occlusion="none",
        labels=[{"id": "silver", "text": "Silver River", "curved_text": True,
                 "geometry": {"type": "LineString", "coordinates": path}}],
        typography={"font_size": 24.0, "halo_width_px": 1.0},
        glyph_atlas={"glyphs": list("Silver River")},
        metadata={"source_id": "curved-screen-fixture"},
    )
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=np.zeros((32, 32), np.float32), crs="EPSG:3857",
                                 metadata={"source_id": "flat"}),
        layers=[layer], lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=384, height=256),
    )
    return scene


def render_curved_label(angle_deg):
    scene = curved_label_scene(angle_deg)
    plan = scene.compile_plan()
    assert not plan.validation_report.render_blocked(), plan.validation_report.to_dict()
    accepted = plan.label_plans["river"].accepted
    assert len(accepted) == 1
    base = np.zeros((256, 384, 4), np.uint8)
    base[..., 3] = 255
    pixels, rendered = map_scene._composite_native_label_layers(base, scene.recipe, plan.label_plans)
    assert rendered
    return pixels, base, accepted[0].candidate
