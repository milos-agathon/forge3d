# GPT-5.6 Sol `AGENTS.md` Design

## Goal

Replace the chronological reflection log with a compact, self-contained root
instruction file that helps GPT-5.6 Sol make correct changes in `forge3d`.

## Structure

1. Repository purpose, architecture, and authoritative files.
2. Working rules that apply to every change.
3. Durable domain invariants for PyO3, GPU/WGSL, MapScene, resource tracking,
   render honesty, deterministic caching, and CI.
4. Change-specific verification commands and a short definition of done.

## Content Policy

- Use imperative, current-tense rules.
- Keep rationale only where it prevents a known class of failure.
- Prefer source files and executable gates over duplicated prose as authorities.
- Retain current behavioral contracts and verified technical constraints.
- Remove dates, campaign names, superseded implementations, commit narratives,
  stale pass counts, machine-specific evidence, and speculative refactors.
- Do not duplicate detailed architecture or feature documentation already owned
  by tracked project documents; point to the authoritative location instead.

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
