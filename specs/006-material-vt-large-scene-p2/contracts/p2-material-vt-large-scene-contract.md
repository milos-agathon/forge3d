# Contract: P2 Material, VT, And Large-Scene Workflows

Final module paths extend the P0/P1 product APIs.

```python
scene = MapScene(...)
report = scene.validate()

VirtualTextureFamilyRequest(family="normal", source=...)
TexturedBuildingMaterial(albedo_texture=..., uv_required=True)
AdvancedLabelRuleSet(repeat_distance=..., curved_text=...)
LargeSceneResourceSummary(...)
```

Contract rules:

- Requested VT `normal` and `mask` families either render in native runtime or emit `vt_unsupported_family` before render.
- Building texture intent either renders albedo texture through `MapScene` or reports missing texture, missing UV, unsupported format, Pro-gated, or fallback diagnostics.
- Advanced labels remain deterministic and reason-coded.
- Large-scene summaries use available metadata and mark unavailable cache/LOD/instancing stats explicitly.
- Deferred P2 paths do not block P0 MVP unless a documented product decision changes scope.
