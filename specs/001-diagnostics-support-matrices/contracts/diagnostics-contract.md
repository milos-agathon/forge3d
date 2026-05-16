# Contract: Diagnostics And Validation Report

Public API names are contract-level names; final file paths are TBD until task inspection chooses the package location.

```python
Diagnostic(
    code: str,
    severity: Literal["info", "warning", "error", "fatal"],
    message: str,
    remediation: str,
    support_level: str | None = None,
    layer_id: str | None = None,
    object_id: str | None = None,
    details: Mapping[str, JSONValue] | None = None,
)

ValidationReport(
    status: Literal["ok", "warning", "error", "fatal"],
    diagnostics: Sequence[Diagnostic],
    layer_summaries: Sequence[LayerSummary] = (),
    estimated_gpu_memory_bytes: int | None = None,
    supported_features: Mapping[str, str] | None = None,
    unsupported_features: Mapping[str, str] | None = None,
)
```

Required methods:

- `Diagnostic.to_dict()` and `Diagnostic.from_dict()`.
- `ValidationReport.to_dict()` and `ValidationReport.from_dict()`.
- `ValidationReport.has_errors`.
- `ValidationReport.render_blocked(policy="continue_on_warning" | "fail_on_warning")`.

Required diagnostic codes:

```text
crs_mismatch
missing_glyphs
unsupported_style_field
unsupported_style_layer_type
pro_gated_path
placeholder_fallback
experimental_feature
vt_unsupported_family
python_public_3dtiles_incomplete
estimated_gpu_memory
label_rejection_summary
```

Contract rules:

- Unknown support-level terms are invalid.
- Unknown severities are invalid.
- Serialization must preserve all public fields.
- Repeated serialization of fixed inputs must be byte-stable when encoded with sorted keys.
- A warning may block only under fail-on-warning; an error or fatal always blocks.
