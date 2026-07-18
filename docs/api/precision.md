# Verified double-float precision

`forge3d.precision` exposes the GPU proofs behind forge3d's opt-in
double-float (`DD`) world-coordinate path. These calls execute real GPU work;
they do not emulate unsupported hardware or silently fall back to raw `f32`.

```python
from forge3d import precision

canary = precision.dd_selftest()
addition = precision.dd_harness("add")
jitter = precision.dd_jitter_demo()
```

## `dd_selftest()`

Runs exactness canaries for Knuth `two_sum` and the selected Dekker/Veltkamp
or FMA `two_prod` implementation. The report includes `passed`, `backend`,
`adapter`, `two_prod_variant`, shader identity, canary and mismatch counts,
rejected variants, and failure details. A backend that cannot preserve the
required round-to-nearest behavior records a `precision_selftest_failed`
degradation and raises; it never returns quietly degraded low components.

## `dd_harness(operation, n=100_000_000)`

`operation` is one of `add`, `mul`, `div`, or `sqrt`. The minimum `n` is
100,000,000 generated pairs; every run also evaluates 1,000,000 hand-built
adversarial cases and checks their `(hi, lo)` words against the Rust mirror.
The result reports `max_err_u2` and `cited_bound_u2`, where
`u = 2^-24`, plus backend, adapter, selected product variant, counts, shader
identity, mismatches, and a signed render-certificate JSON payload.

The enforced Joldeș–Muller–Popescu bounds are 3 u² for addition, 7 u² for
multiplication, and 15 u² for division and square root. Expect a full default
run to be expensive: it is an acceptance proof, not a per-frame diagnostic.

## `dd_jitter_demo(frames=1000)`

Runs the committed Everest ECEF scene while the camera dollies by one
millimetre per frame. It returns both screen-space error curves, their maxima,
the number of raw-`f32` frames above one pixel, two DD render hashes, backend
and shader identity, and a signed certificate. Acceptance requires every DD
error below 0.01 px, at least 10% of raw-`f32` frames above 1 px, and identical
DD hashes across both render executions.

All three calls raise `RuntimeError` for missing hardware, unsupported or
unverified arithmetic, invalid arguments, proof failures, and resource or
pipeline errors. Pin `WGPU_BACKENDS` before process startup when proof output
must be attributed to a specific backend.
