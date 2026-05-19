# Contract: Public Label API Truth

Public behavior is available through high-level `ViewerHandle` methods in `python/forge3d/viewer.py` with stubs in `python/forge3d/viewer.pyi`, not raw IPC, for MVP workflows.

```python
label_id = viewer.add_label(text: str, world_pos: tuple[float, float, float], **style) -> int | LabelOperationResult
result = viewer.add_labels(labels: Sequence[Mapping[str, object]]) -> LabelBatchResult
line_id = viewer.add_line_label(text: str, polyline: Sequence[Point3], **style) -> int | LabelOperationResult
curved_result = viewer.add_curved_label(text: str, path: Sequence[Point3], **style) -> LabelOperationResult
callout_id = viewer.add_callout(text: str, anchor: Point3, **style) -> int | LabelOperationResult
overlay_id = viewer.add_vector_overlay(...) -> int
viewer.clear_labels() -> LabelOperationResult
viewer.remove_label(label_id: int) -> LabelOperationResult
viewer.set_labels_enabled(enabled: bool) -> LabelOperationResult
viewer.load_label_atlas(atlas_png_path: str | Path, metrics_json_path: str | Path) -> LabelOperationResult
viewer.set_label_typography(*, tracking: float | None = None, kerning: bool | None = None, line_height: float | None = None, word_spacing: float | None = None) -> LabelOperationResult
viewer.set_declutter_algorithm(algorithm: str, *, seed: int | None = None, max_iterations: int | None = None) -> LabelOperationResult
viewer.label_configuration_state() -> dict[str, object]
```

Contract rules:

- Successful create calls return stable IDs where later reference is needed.
- Batch creation preserves input order and reports per-label diagnostics.
- Successful setters mutate real state, affect future behavior, render/layout output, or serializable state.
- Unsupported setters return typed diagnostics; they do not return success.
- Line/curved labels either emit renderable glyph instances or return `experimental_feature`/unsupported diagnostics.
- The public basic workflow must not require raw IPC.
