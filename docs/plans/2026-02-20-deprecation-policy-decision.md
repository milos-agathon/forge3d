# Deprecation Policy Decision Record

Date: 2026-02-20
Status: deferred
Phase: P2.4 of API Consolidation Plan

## Context

The API consolidation plan (P2.4) requires a signed-off deprecation policy
before any `#[deprecated]` annotations or `DeprecationWarning` markers are
added to the codebase.

## Evidence Gathered

### Current State

- **Zero** existing `#[deprecated]` annotations in Rust (`src/`)
- **Zero** existing `DeprecationWarning` calls in Python (`python/forge3d/`)
- No deprecation infrastructure exists in either layer

### Module Exposure Analysis

The audit identified these modules with minimal native Python exposure (L0/L1):

| Module | LOC | Test Files | Python Exports | Rust Cross-refs |
|--------|-----|------------|----------------|-----------------|
| `bundle` | 182 | 3 | Yes (`__all__`) | Minimal |
| `export` | 994 | 2 | Yes (`__all__`) | Minimal |
| `tiles3d` | 1940 | 2 | Yes (`__all__`) | 2 internal |
| `style` | 1918 | 5+ | Yes (`__all__`) | 4 internal |
| `pointcloud` | 1067 | 2 | Yes (`__all__`) | Active dev |
| `sdf` | 1965 | -- | Unregistered | Foundational |
| `labels` | 3395 | -- | Partial (P2.3) | Core rendering |

### Key Findings

1. **All L0/L1 modules have active tests** — none are dead code.
2. **All are exported in `python/forge3d/__init__.py`** — actively public API.
3. **Rust cross-module references are light** (12 total) but non-zero.
4. **Plan non-goal**: "Broad deprecation of Rust modules (`export/style/tiles3d/bundle`) without usage evidence."

## Decision

**Deprecation is deferred.** No code-level deprecation annotations will be
merged in this consolidation round.

### Rationale

1. No module is truly dead — all have tests and public exports.
2. Deprecating actively-tested, publicly-exported modules without a migration
   path would break downstream users.
3. The consolidation plan explicitly lists broad deprecation as a non-goal.
4. The correct sequence is: (a) identify concrete replacement APIs, (b) build
   migration guides, (c) add deprecation warnings with timeline, (d) remove
   after migration window. We are not at step (a) for any module.

### Migration Strategy (for future use)

When deprecation is warranted, follow this protocol:

1. **Evidence**: Document which callers use the deprecated API (grep tests,
   examples, and downstream packages).
2. **Replacement**: Provide a concrete alternative API that covers all use cases.
3. **Warning**: Add `#[deprecated(since = "X.Y.Z", note = "Use ... instead")]`
   in Rust and `warnings.warn("...", DeprecationWarning, stacklevel=2)` in
   Python wrappers.
4. **Window**: Maintain deprecated API for at least 2 minor versions.
5. **Removal**: Remove in the next major version after the window expires.

### Warning Strategy

- Rust: Use `#[deprecated]` attribute with `since` and `note` fields.
- Python: Use `warnings.warn()` with `DeprecationWarning` category.
- Tests: Update tests to suppress expected deprecation warnings using
  `pytest.deprecated_call()` or `@pytest.mark.filterwarnings`.
- CI: Do not treat deprecation warnings as errors during the migration window.

## Acceptance Criteria Verification

1. **Signed-off decision record exists**: This document.
2. **No code-level deprecation annotations merged without approved policy**: Confirmed — zero `#[deprecated]` or `DeprecationWarning` in codebase.
