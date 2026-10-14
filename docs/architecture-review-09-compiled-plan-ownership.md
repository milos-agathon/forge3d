# W18: compiled-plan ownership

This implements item 09, “Typed map scenes — Make compiled-plan ownership
explicit”, from the local `docs/architecture-review-forge3d-50.html` review.
The review document is not tracked on `main`; this note retains the item's
requirements and the context needed to review the scoped change.

The requested boundary remains `SceneRecipe → compile_plan → CompiledScenePlan`.
One scene compiler must own diagnostics, label decisions, recipe identity,
frozen-plan creation/restoration and mutation invalidation. Fresh compilation
and saved-plan restoration remain separate paths. Consumers receive frozen
decisions, and rendering must not mutate labels. Preserve public names,
diagnostics, eligible native rendering, SUTURA recipe-mutation and frozen
occlusion coverage, and byte-identical manifest/bundle round trips. Add
validation call-order cases for both freshly compiled and restored plans.

`_SceneCompiler` publishes complete state only after an operation returns
successfully. `MapScene` delegates its existing interfaces and exposes the
compiler's latest report. Its mutable inspection dictionary is detached from
the frozen plan. Frozen label reads return detached copies, including nested
data. CHRONOS reads those frozen decisions. Duplicate label-layer IDs, missing
saved layers, mismatched saved recipe identities and non-finite saved manifests
are rejected without publishing partial state.

In-memory label snapshots preserve unknown DEM elevations. CHRONOS encodes
those optional sampled elevations as JSON `null`; positions and other required
frame values remain subject to finite checks. Manifest persistence still
rejects non-finite data. Fresh snapshots normalize signed zero; the guide
documents older-writer compatibility. Current-main render-pass inputs, manifest
fields, label instance visibility, depth authority and projection-reservation
cleanup are preserved.

Review against the PR's base and head, including `_map_scene_compiler.py`,
`map_scene.py`, `chronos.py`, the public stubs, the guide and affected tests.
For each requirement above, identify the owning code and an executed test or
an explicit proof gap. Exercise validate/compile/render/save/load ordering,
recipe mutation, failed operations, restored decisions without replanning,
nested inspection edits, native-frame attachment, duplicate IDs, saved identity
and finite-data rejection, nodata flythrough replay, and every-file bundle byte
round trips. Verify that the newer render-pass and world-projection contracts
still hold. Report only concrete regressions or missing requirements, with
location, reproducer, impact and correction; distinguish unrun backend checks
from successful evidence.
