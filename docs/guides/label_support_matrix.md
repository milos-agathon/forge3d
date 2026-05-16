# Label Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Point label substrate | `underdeveloped` | Existing viewer and native label paths exist, but high-level API truth is owned by feature `002`. | `missing_glyphs`, `experimental_feature`. |
| Line labels | `experimental` | Must render glyph instances and path rotation before production support can be claimed. | `experimental_feature` until proven or replaced by typed unsupported behavior. |
| Curved labels | `experimental` | Curved placement is not production-stable in the P0 diagnostics feature. | `experimental_feature`. |
| Deterministic `LabelPlan` | `missing` | Owned by feature `003`. | `label_rejection_summary` only when reason-coded plan data exists. |
| Missing glyph detection | `underdeveloped` | Diagnostics contract exists; full atlas integration belongs to label features. | `missing_glyphs`. |

No label command should report success while doing nothing. Unsupported or
unverified label behavior must produce typed diagnostics or explicit failure.
