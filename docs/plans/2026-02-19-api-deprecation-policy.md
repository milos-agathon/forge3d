# API Deprecation Policy

Date: 2026-02-19
Status: approved
Phase: P2.4 of API Consolidation Plan
References: `docs/plans/2026-02-20-deprecation-policy-decision.md`

## Sign-off

This policy is approved as part of the API consolidation execution checklist.
Sign-off granted (sign-off): no `#[deprecated]` annotations or
`DeprecationWarning` markers shall be merged without following the protocol
below.

## Policy

### Decision

Deprecation is **deferred** for this consolidation round. No code-level
deprecation annotations will be merged.

Rationale: all candidate modules (`bundle`, `export`, `tiles3d`, `style`,
`pointcloud`, `sdf`, `labels`) have active tests, public exports, and non-zero
cross-module references. Deprecating actively-tested, publicly-exported modules
without a migration path would break downstream users. See the decision record
at `docs/plans/2026-02-20-deprecation-policy-decision.md` for the full
evidence analysis.

### Migration Window

When deprecation becomes warranted in a future consolidation round:

1. **Minimum window**: deprecated APIs must remain functional for at least
   **2 minor versions** after the deprecation warning is introduced.
2. **Removal**: deprecated APIs may be removed only in the next **major
   version** release after the migration window expires.
3. **Timeline**: each deprecation notice must include the earliest version
   in which removal may occur.

### Deprecation warning strategy

#### Rust

```rust
#[deprecated(since = "X.Y.Z", note = "Use <replacement> instead. Will be removed in vN.0.0.")]
```

- Apply to the public item (`fn`, `struct`, `trait`).
- The `note` field must name the concrete replacement.

#### Python

```python
import warnings
warnings.warn(
    "<old_api> is deprecated since vX.Y.Z. Use <replacement> instead. "
    "Will be removed in vN.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

- Emit on first call or import of the deprecated API.
- `stacklevel=2` so the warning points to the caller, not the wrapper.

#### Tests

- Update tests to use `pytest.deprecated_call()` or
  `@pytest.mark.filterwarnings("ignore::DeprecationWarning")` for
  tests that intentionally exercise deprecated paths.
- CI must **not** treat deprecation warnings as errors during the
  migration window.

### Pre-deprecation Requirements

Before any `#[deprecated]` or `DeprecationWarning` is merged:

1. **Evidence**: document which callers use the deprecated API (grep tests,
   examples, and downstream packages).
2. **Replacement**: provide a concrete alternative API that covers all
   existing use cases.
3. **Migration guide**: write a short guide showing before/after usage.
4. **Approval**: the deprecation must be reviewed and approved in a PR
   that includes the above three items.

## Acceptance Criteria

1. This signed-off policy document exists at the required path.
2. No code-level deprecation annotations are merged without following
   the protocol above.
3. The decision record at `docs/plans/2026-02-20-deprecation-policy-decision.md`
   remains the canonical evidence base.
