# Quickstart: Diagnostics And Support Matrices

These scenarios are validation targets for task implementation, not current working code.

## Validate Style Support

1. Build a local style fixture with supported `fill`, `line`, and `circle` layers plus one unsupported layer.
2. Run the public validation helper or `MapScene.validate()` once feature `004` exists.
3. Expect structured diagnostics for unsupported layer types and fields.
4. Confirm docs state local/provided feature styling only, not streamed MVT.

## Validate Render Warning Policy

1. Create a report with one `missing_glyphs` warning.
2. Confirm `continue_on_warning` does not block by itself.
3. Confirm `fail_on_warning` blocks successful render completion.
4. Add an `error` diagnostic and confirm both policies block.

## Validate Bundle-Ready Serialization

1. Serialize a report containing every required diagnostic code.
2. Deserialize it and compare code, severity, support level, affected IDs, remediation, and details.
3. Repeat serialization and compare exact output ordering.

## Negative Path Checks

- Pro-gated building path emits `pro_gated_path`, not success.
- Zero-geometry building fallback emits `placeholder_fallback`, not success.
- Non-albedo VT family emits `vt_unsupported_family`, not a warning-only log.
- Incomplete public 3D Tiles path emits `python_public_3dtiles_incomplete`, not supported status.
