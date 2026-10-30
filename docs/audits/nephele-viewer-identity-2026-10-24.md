# NEPHELE viewer identity and panic removal — 2026-10-24

This change follows `3f349648bc14882d74a1205bb1f606bc77a13e1a`. That parent repaired the viewer's terrain identity when no production terrain trace exists and retained medium-independent legacy geometry across canonical medium uploads. The Apple Metal viewer regression passed in all-Rust CI at that parent. This follow-up removes the remaining production unwrap/expect sites introduced by PR #172.

Milos's exact approval was “Approved—apply the exact batch”. The applied patch SHA-256 was `4e7ac39b6d8e676acf4330eee547c8b97f4b0c81b32f5ed9aa40239e8df8e26f`. Density and medium identities now return serialization errors, which native and Python callers propagate. Successful serde serialization, digest inputs, Python identity strings and repr values are unchanged. The only existing assertion changes are the six approved successful-identity adaptations; one approved test setup statement also unwraps its successful identity. Exact old/new text is preserved in the accompanying JSON. No fallback identity is introduced.

The CDF, phase frame, AOV output-path, viewer transmittance and viewer post-processing paths propagate errors instead of the remaining added panics. The JSON lists each affected file and its reason.

## CPU evidence

Fmt and acceptance Clippy passed. With `extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion`, 56 media tests, five frame-graph tests, two draw leaf-key tests and three viewer clipping tests passed: 66 actual tests, zero failures or ignored tests. One initial leaf-key filter selected zero tests; it was rejected as evidence and corrected to `terrain::renderer::draw::leaf_keys::tests::`. The receipt preserves both attempts. Target and temporary paths belong to this worktree; exact commands, durations and environment are in the JSON.

The repository ratchet's production counts and source paths equal main's. A wider scan over every changed Rust production file finds no newly added unwrap/expect line. The f32 inventory remains 1,936 sites / `8f9f378ce8e27a98e9b99961073ae70128584485778adf1dc2c597c464c0cee7`. Shader sources, pins, shader contracts, ledgers, committed certificates/goldens, fixture, policy, gates, thresholds and CI deselections are unchanged. Main's existing ratchet failures are not waived.

## Timing evidence and release limits

The parent scoped run [40121981661](https://github.com/milos-agathon/forge3d/actions/runs/40121981661) measured two states on the original 1500² DEM at 1920×1080, RTX 3070/Vulkan: 69.2682638 s for a first completed frame and 306.0231000 ms for an unchanged frame. Both timed certificates had no degradation. The untimed diagnostic follow-up reported 28 orders, convergence, zero source/surface change and cache reuse. Those are parent observations, not new-head validation or a controlled speedup comparison. The accepted CHANGELOG observations of 76.151 s / 328.481 ms remain unchanged.

Camera movement was not measured. The directional solve key includes the camera-dependent uniforms, so moving the camera invalidates the static transport cache and requires a full solve. Moved-camera latency and iteration count are unknown; neither the unchanged-frame latency nor the 28-order diagnostic proves them. CHANGELOG and future benchmark records now state this gap explicitly. A live rendered sky also prevents reuse. No order cap or new stopping rule is added.

Main's scheduled full CI [40075490450](https://github.com/milos-agathon/forge3d/actions/runs/40075490450) is red. Main must be repaired in a separate PR before NEPHELE integrates it and reruns full acceptance. This change does not waive that prerequisite. Exact-head wheel and scoped/full CI results follow this committed CPU snapshot; no green result is claimed here for an unexecuted head.

No local GPU work occurred. D3D12, Metal and WASM full NEPHELE release qualification remains ABSENT; the specific Apple Metal viewer test at the parent did execute successfully in CI. The reference sampler RNG-dimension overlap remains untouched. No merge, tag or publication is authorized.

Proposed steering, not applied: record the exact library test filters beside validation receipts, and distinguish the ratchet's module scope from the owner's requirement covering every added production panic site. Neither AGENTS.md nor any skill was changed.
