# Offline 3D Map Rendering

forge3d's product direction is an offline 3D map-production workflow:

```text
MapScene + LabelPlan + ValidationReport + Bundle
```

Feature `001` establishes the diagnostics contract and support matrices used by
that workflow. The complete typed `MapScene` and deterministic `LabelPlan`
APIs are owned by later P0 features.

| Area | Support level in this feature | Notes |
| --- | --- | --- |
| Structured diagnostics | `supported` | `Diagnostic` and `ValidationReport` are public Python objects. |
| Full MapScene rendering | `missing` | Owned by feature `004`. |
| Deterministic LabelPlan | `missing` | Owned by feature `003`. |
| Unsupported-path validation | `underdeveloped` | Diagnostic factories exist; full render-path wiring is incremental. |
| Web-first hosted tile delivery | `non-goal` | Offline map production remains the scope. |

Unsupported, `Pro-gated`, `placeholder/fallback`, `experimental`, or
`underdeveloped` paths must be reported before successful render completion.
