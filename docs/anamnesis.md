# ANAMNESIS render cache

`forge3d.anamnesis` provides a local content-addressed store for deterministic
offline frame sequences. `render_sequence(recipe, cache=...)` keys each declared
pass from its pipeline state, exact uniform bytes, input keys, negotiated
capabilities, and engine fingerprint. `cache=None` performs no cache I/O.

The store defaults to `.forge3d/cache`, verifies blob hashes before reuse,
quarantines corrupt entries, and evicts least-recently-used payloads when its
`max_bytes` budget is reached. Use:

```text
python -m forge3d.anamnesis explain <key>
python -m forge3d.anamnesis verify
python -m forge3d.anamnesis gc <max_bytes>
python -m forge3d.anamnesis render-sequence recipe.json --frames 600
```

All public render entry points accept `cache=None`. Renderers whose complete
mutable state is not yet serializable conservatively recompute when a cache path
is supplied; they never reuse an incomplete key. The interactive viewer is out
of scope. Native one-shot terrain rendering and deterministic `MapScene` PNG
output currently provide verified cache hits.

Cross-machine reuse is gated by the determinism workflow. A physical
Linux/Vulkan producer and physical Windows/DX12 consumer must independently
match the committed TERRA golden before sharing the portable compatibility
profile; unavailable physical adapters produce an explicit `ABSENT` result.
