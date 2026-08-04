# CI validation tiers

forge3d separates merge validation from high-cost acceptance. A green hosted
check is evidence only for the claim named by that check; it is not evidence of
physical NVIDIA/Vulkan execution.

## Pull-request core

`PR Core Success` is the stable branch-protection context for every pull
request to `main` or `develop`. It requires:

- the cheap workflow-contract preflight before fan-out;
- the three hosted Rust jobs and four independently built platform wheels;
- one full Python suite on Linux/Python 3.11;
- compatibility smoke on Linux/Python 3.10 and 3.13, Windows/Python 3.11, and
  macOS/Python 3.11;
- the representative slow lane, TERMINUS, Linux aarch64 install smoke, docs,
  and path-selected hosted visual goldens.

For pull requests, the preflight checks out the live base branch, fetches and
verifies the immutable PR head, and materializes their merge locally before it
runs the workflow contracts from a separate base-owned checkout. This avoids
stale activity-event SHAs, makes policy tests added on the base branch available
to older PR heads, and prevents a candidate from weakening its own validator. A
merge conflict fails preflight. Wheels, source tests, and acceptance evidence
remain built from the exact PR head SHA.

The exhaustive 3 operating systems by 4 Python versions suite runs only in the
nightly schedule or a manual `full` dispatch. It replaces, rather than
duplicates, the ordinary PR Python lanes in those runs.

Configure branch protection to require `PR Core Success`. Do not require `Full
Acceptance Summary` globally: it is intentionally skipped when no expensive
acceptance family is selected.

After this policy first lands on the base branch, each already-open pull
request needs one new `pull_request` event so it receives the new required
context. Add or remove the harmless `ci` label, use **Update branch**,
rebase/merge the current base branch into the PR, or push a new commit. Future
PRs receive the check automatically.

## Hosted determinism

The render, F3DZ, and ANAMNESIS determinism families are path-selected
independently. They install the exact Linux, Windows, or macOS wheel artifacts
already built in the caller run; they do not rebuild the extension. A nightly
run and manual `full` or `determinism` scope select the relevant complete set.

Hosted render legs may record an explicit `ABSENT` result when no qualifying
adapter exists. That result documents hosted-runner capability only and never
satisfies a physical acceptance requirement.

## Physical acceptance

M-06, F3DZ, and ANAMNESIS keep their existing exact NVIDIA/Vulkan or DX12
acceptance commands, zero-skip contracts, thresholds, and 90-day evidence. The
frequency is narrower:

- the nightly schedule runs every physical family;
- manual `full`, `m06`, `f3dz`, or `anamnesis` selects the named family;
- an internal pull request runs a physical family only when both its paths
  match and the PR has the `run-physical` label;
- ordinary pushes and unlabeled PRs do not allocate a self-hosted GPU runner.

Adding or removing `run-physical` emits a PR event and starts a replacement CI
run immediately; PR concurrency cancels the obsolete run.

`Full Acceptance Summary` validates every selected hosted and physical family.
It is a reporting/acceptance context, not the global merge gate.

There is no TESSELLA job on `main`. Its path classifier is reserved as a
fail-closed contract: any future TESSELLA physical job must consume the
`tessella` path output or an explicit manual scope and must not run on every PR.

## Certificate refresh

Certificate mutation is not part of CI. `Refresh recipe certificates` is a
manual-only workflow that runs only from the protected `main` ref, builds one
Windows wheel for that exact SHA, and proves the checkout on the physical
NVIDIA/Vulkan runner. Its signing job uses the protected `production-signing`
Environment, requires the production signing secret, renders/signs the catalog,
and uploads the signed certificates plus provenance for 90 days. Configure the
secret and required reviewer on that Environment before enabling the workflow.
