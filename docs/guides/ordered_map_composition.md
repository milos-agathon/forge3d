# Ordered map composition

`MapScene.render_passes` composes existing RGBA render outputs using a Python
image compositor. It never substitutes for unavailable native scene rendering.
Use `MapScene.render` and the public vector renderer to produce source images;
the [executable example](../examples/mapscene_render_passes.py) renders color,
relief and a transparent vector overlay on physical hardware, then composes and
replays them through a bundle.

```python
inputs = {
    "color_image": f3d.RenderPassInput(color_rgba),
    "relief_image": f3d.RenderPassInput(relief_rgba),
    "vector_image": f3d.RenderPassInput(overlay_rgba),
}
passes = [
    f3d.RenderPassSpec("color", "color", ("color_image",)),
    f3d.RenderPassSpec("relief", "relief", ("color", "relief_image"),
                       "multiply", {"opacity": 0.5}),
    f3d.RenderPassSpec("overlay", "overlay", ("relief", "vector_image"),
                       "alpha_over"),
]
report = scene.render_passes(passes, inputs, "composed.png")
scene.save_bundle("composition")
replay = f3d.MapScene.load_bundle("composition.forge3d")
replay.render_passes(path="replay.png")
```

Passes execute in list order. Each input reference names either an image snapshot
or a previously executed pass. Two references mean `(backdrop, source)`; one is
allowed only for `alpha_over` over transparent black. The final pass is the output.
Every supplied input and earlier pass must contribute to the final pass;
disconnected branches and unused snapshots are rejected. An explicit empty
pass list is rejected; omit both arguments to replay or retain no-pass rendering.
Names must be nonempty and unique across inputs and passes. Missing, forward and
self references, dimensions differing from `OutputSpec`, unsupported operations
or parameters, NaN/infinite channels, and channels outside their domain raise
before composition. `kind` is `color`, `relief` or `overlay`: it records the role
and does not generate a style or alter pixels.

Inputs accept uint8 RGBA in `[0,255]` or floating RGBA in `[0,1]`, and copy them
into immutable snapshots. Each input declares `color_space="srgb"` (default) or
`"linear"`, and `alpha_mode="straight"` (default) or `"premultiplied"`.
Premultiplication applies in the input's declared color space and is undone
before color conversion. RGB at zero alpha is ignored. Opacity defaults to one
and scales source alpha only; it is the sole supported pass parameter.

Each pass blends in its declared `color_space`, defaulting to linear light.
Conversions use the standard sRGB transfer function. For straight RGB `Cb`, `Cs`
and backdrop/source alpha `ab`, `as` (including opacity):

- `multiply`: `B = Cb * Cs`
- `screen`: `B = Cb + Cs - Cb * Cs`
- `alpha_over`: `B = Cs`

All operations use source-over alpha `ao = as + ab*(1-as)` and premultiplied
color `Co = as*(1-ab)*Cs + as*ab*B + (1-as)*ab*Cb`. Output RGB is `Co/ao`, or zero
at zero alpha. Intermediate passes retain float precision. PNG output is
straight-alpha sRGB, rounded once by the existing PNG writer (8 or 16 bits).
HDR, AOV export, native certificates, provenance and render cache are unsupported
for this compositor and are rejected explicitly.

Pass order, references, parameters, color/alpha semantics and snapshot references
are stored in the recipe and frozen compiled manifest. Composition bundles use
version 4, which older readers reject rather than silently dropping the passes.
Only `MapScene.load_bundle` opts into that format, and it requires nonempty
passes. The generic and viewer bundle loaders keep their version-3 cap.
Each unique pixel snapshot is stored once in `scene/render_pass_inputs/<sha256>.npy`:
NumPy format 1.0, C order, little-endian float64. The descriptor records shape,
dtype, color/alpha semantics and SHA-256; loading verifies the file, descriptor
and checksum before use. Replay reads the compiled
specification directly and rejects a recipe/plan mismatch. Optional fields are
absent for scenes without passes, preserving their existing serialization and
native rendering; those bundles remain version 3. No new dependency is required.
Standalone `to_dict()` recipes retain inline RGBA data for self-contained JSON;
the render, compile and bundle paths use compact references to immutable snapshots.
`forge3d.recipe_manifest(scene)` also uses these references, lists the passes,
and reports the same recipe hash as the compiled plan. Scene, recipe and mapping
inputs produce the same summary; inline mappings are normalized without changing
the caller's data.
`MapScene(..., pass_specs=passes, pass_inputs=inputs)` also configures composition;
call `render_passes()` to execute it. `render()` rejects configured image passes
and retains its native-frame contract. `render_passes()` on a scene without passes
delegates to ordinary `render()`.
Its `certificate` and `cache` keywords are forwarded on that no-pass path;
composition rejects nondefault native controls before replacing any saved passes.
