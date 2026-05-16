# Contract: LabelPlan Compiler

Final module path is TBD. Public behavior:

```python
plan = LabelPlan.compile(
    labels: LabelSource,
    camera: CameraLike,
    viewport: OutputSpecLike,
    terrain: TerrainSamplerLike | None = None,
    keepouts: Sequence[KeepoutRegion] = (),
    priority_rules: PriorityRules | None = None,
    typography: TypographySettings | None = None,
    glyph_atlas: GlyphAtlasLike | None = None,
    seed: int = 0,
)
```

Required properties:

- `plan.accepted`
- `plan.rejected`
- `plan.diagnostics`
- `plan.bounds`
- `plan.seed`
- `plan.to_dict()`
- `LabelPlan.from_dict(...)`
- `plan.to_render_payload()`
- `plan.to_export_payload()`

Contract rules:

- Fixed inputs and seed produce identical accepted/rejected/diagnostic/bounds/payload ordering.
- Every rejected label or candidate has a required reason code.
- Unsupported render/export backends return structured diagnostics, not empty success.
- Missing glyphs produce `missing_glyphs` diagnostics and `missing_glyph` rejection where applicable.
- Keepout regions are active in collision/placement solving.
