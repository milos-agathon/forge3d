# GPT-5.6 Sol `AGENTS.md` Design

## Goal

Replace the chronological reflection log with a compact, self-contained root
instruction file that helps GPT-5.6 Sol make correct changes in `forge3d`.

## Structure

1. Exact build, test, lint, and focused-test commands.
2. Safety boundaries grouped as always, ask first, and never.
3. Specific stack, repository structure, and authoritative files.
4. Durable domain invariants for PyO3, GPU/WGSL, MapScene, resource tracking,
   render honesty, deterministic caching, and CI.
5. Change-specific verification gates and a short definition of done.

## Content Policy

- Use imperative, current-tense rules.
- Reserve `NEVER` and `MUST` for safety or correctness boundaries; state style
  preferences plainly so strong instruction-following models do not over-apply
  them.
- Include exact copy-pasteable commands and the focused-test pattern near the
  top. Use `.github/workflows/ci.yml` as the feature-matrix authority.
- Name the core stack and compatibility versions from `Cargo.toml` and
  `pyproject.toml`; do not copy long dependency inventories.
- Include one compact code example only when it expresses a verified,
  repository-wide pattern better than a checklist.
- Keep rationale only where it prevents a known class of failure.
- Prefer source files and executable gates over duplicated prose as authorities.
- Retain current behavioral contracts and verified technical constraints.
- Remove dates, campaign names, superseded implementations, commit narratives,
  stale pass counts, machine-specific evidence, and speculative refactors.
- Do not duplicate detailed architecture or feature documentation already owned
  by tracked project documents; point to the authoritative location instead.

## Repository-Specific Boundaries

- Never read, print, write, or commit credentials, private keys, tokens,
  `.env` files, or other secrets.
- Never edit generated binaries, build products, vendored code, or committed
  golden/certificate evidence except through the repository's owning workflow.
- Never claim a check passed unless its successful result was observed.
- Ask before adding a production dependency, changing public compatibility,
  weakening a gate, updating goldens, or performing destructive/external
  actions outside the requested scope.
- Preserve unrelated working-tree changes and make the smallest complete
  root-cause fix.

Package-manager lock-in, frontend conventions, and generic PR policy will not
be added because this repository does not define those constraints. `AGENTS.md`
will be authoritative for agent behavior, while code, tests, manifests, and CI
remain authoritative for product behavior and executable configuration.

## Compatibility

The rewrite will preserve instructions required by current contract tests,
including native symbol registration, wheel feature synchronization, MapScene's
no-placeholder behavior, tracked GPU allocation, render-certificate honesty,
shader layout compatibility, deterministic serialization, and physical
cross-backend portability proof.

## Verification

- Review the diff to ensure no active invariant was lost.
- Check every referenced path and command against the current repository.
- Scan the result for historical headings, superseded symbols, duplication, and
  ambiguous or non-actionable language.
- Confirm only the intended documentation files changed.
