# Color Management

forge3d render paths treat shader RGB as linear Rec.709 scene-referred color until the output stage. The shared tone-map operator table lives in `src/shaders/includes/tonemap_common.wgsl` and uses stable IDs: Reinhard `0`, Reinhard Extended `1`, ACES `2`, Uncharted2 `3`, Exposure `4`, and terrain filmic `5`.

The display path is:

1. Apply white balance in linear scene space when enabled.
2. Apply exposure.
3. Apply the selected tone-map operator.
4. Apply LUT grading after tone mapping, before display encoding.
5. Encode to sRGB exactly once.

Pipelines that render into `Rgba8UnormSrgb` write linear post-tonemap values and let the target format perform sRGB encoding. Pipelines that write storage textures such as offline `Rgba8Unorm` call the shared `linear_to_srgb()` helper before storing bytes.

EXR outputs remain linear HDR. PNG outputs are display-referred. `OutputSpec(bit_depth=8)` writes RGBA8 PNG. `OutputSpec(bit_depth=16)` writes RGBA16 PNG with the same display-referred values scaled to 16-bit samples for print/export headroom.
