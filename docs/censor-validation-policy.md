# CENSOR Validation Policy

**Status:** authoritative repository policy
**Effective:** 2026-08-05

CENSOR is forge3d's execution-truth contract. It ensures that a render reports
which backend, capabilities, shaders, passes, and allocations actually
participated, and that a fallback or unsupported path cannot silently report
full success. CENSOR is not a requirement to reproduce release certification
on every pull request.

## Permanent architectural invariants

These invariants remain part of normal implementation work:

- GPU devices use the negotiated capability path; private default-device paths
  are forbidden.
- Optional-capability absence is explicit through a degradation or a typed
  error.
- GPU buffers and textures use the tracked allocation boundary, and the
  host-visible budget defaults to enforcement.
- Public pixel-producing entry points participate in the execution-report
  contract or document why they are outside its scope.
- Reports identify the executed preprocessed WGSL, adapter, capabilities,
  passes, allocations, and degradations.
- Canonical payload and offline tamper verification remain deterministic.
- Golden, probe, and acceptance workflows distinguish `passed`, `absent`, and
  `failed`; a crash may never masquerade as hardware absence.

## Validation lanes

| Lane | Trigger | Required content | What it may block |
| --- | --- | --- | --- |
| PR core | Every pull request and protected-branch push; reported by `PR Core Success` | Formatting, curated lint/build, focused capability/degradation/allocation/certificate tests, affected package smoke, and workflow-policy contracts | The affected change |
| Affected integration | Explicitly selected when a change touches a renderer, shader, resource/certificate core, packaging boundary, or validation workflow | Focused subsystem tests and the smallest representative render or golden | The affected change |
| Acceptance | Manual `scope=full` dispatch for a named candidate SHA, plus explicitly selected or scheduled evidence lanes | Complete Python lane, curated cross-platform Rust/feature matrix, full wheel matrix, the golden and physical-GPU probes named by that candidate's acceptance plan, and acceptance artifacts summarized by `Full Acceptance Summary`; optional Metal diagnostics record `ABSENT` and are excluded from this verdict | Declaring the named moonshot or candidate accepted |
| Release | Explicit release promotion; `v*` tag pushes and production publish dispatches require exact-SHA full acceptance first | A completed `ci.yml` run on the release SHA, from `schedule` or `workflow_dispatch` with recorded `scope=full`, whose `Full Acceptance Summary` succeeded; then production signing, complete certificate verification and release artifacts | Every release job; explicit `dry_run=true` dispatches bypass evidence lookup and only build/test artifacts |

`PR Core Success` must not require a production signing secret, a scarce physical
runner, an all-golden render, a repository-wide test matrix, or deliberate
scratch-branch corruption.

The routine Python command is
`python scripts/ci_pytest_lane.py --profile fast`. The exhaustive selector is
`--profile full` and belongs only to the scheduled or explicitly dispatched
acceptance lanes that need it.

## Dependency rule

When another prompt depends on CENSOR, it inherits the architectural
interfaces and invariants above. It does **not** inherit CENSOR's complete
acceptance suite. A downstream change runs the PR-core fast contract plus focused
affected-integration checks for any CENSOR interface it changes. Full CENSOR
acceptance runs only for an explicitly named acceptance or release candidate.
Changing capability negotiation, resource tracking, the degradation sink, the
certificate/signing format, the shader registry, or validation routing makes
fresh acceptance evidence relevant, but does not promote a routine pull request
into the closure suite.

## Evidence and signing

Routine and fork pull requests are explicitly untrusted with respect to
production signing. They still verify canonicalization, schema, signature
rejection, and untrusted provenance without access to
`FORGE3D_CERT_SIGNING_KEY`. Production signatures are created and required only
in protected acceptance or release workflows.

Full-suite, exhaustive-golden, physical-GPU, and red-proof results are
attributable to the candidate SHA on which they ran. They are not transitive
proof for later commits and their absence does not block unrelated feature
development. The repository does not automatically run acceptance after every
merge. Release promotion requires green full acceptance on the exact release
SHA; promotion owners request it before attempting publication. A missing run
refuses all release work and prints `gh workflow run ci.yml -f scope=full --ref
<tag>`. A dispatch summary records its selected scope in an executed job step,
so narrower acceptance dispatches cannot authorize a release. An explicit dry
run bypasses this check and publishes neither PyPI packages nor release assets.

Optional Metal SUBSTRATIA diagnostics do not block acceptance. Probe exit 2
with matching `absent` evidence records an explicit summary/annotation and
omits the hardware controls; it is not GPU success. Probe crashes and
inconsistent or missing evidence remain failures. The required physical
NVIDIA lane and its zero-skip evidence requirements are unchanged.
