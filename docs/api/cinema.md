# Terrain shot planning

`forge3d.cinema` declares static subjects and `reveal`, `orbit`, `flyover`, and
`dolly` shots. `plan(shots, TerrainScatterSource)` produces a `ShotPlan` whose
`bake()` returns the existing native `CameraAnimation`. Existing rigs and
animation defaults are unchanged. Motion budgets, framing, duration and
clearance are explicit inputs; refinement uses the existing rig's default
`TerrainClearance.max_refine_passes` unless supplied.

Subjects are positive-volume world AABBs. `Subject.point(position, diameter=...)`
declares a point's framing envelope. `Subject.from_lon_lat_height(...,
to_world=..., diameter=...)` requires an explicit geographic-to-local transform;
degrees are never treated as world distances. Coordinates follow the rig's
local terrain contract: X/Z in `[0, terrain_width]`, Y measured above the DEM
minimum after `z_scale`.
`Subject.from_lon_lat_bounds(bounds, to_world=...)` encloses the eight
transformed geographic corners in a world AABB.

For terrain-attached subjects, visibility uses the declared center and corners
that lie strictly above the bilinear DEM. Buried or touching reference points
describe the terrain's interior and are excluded; the report records exposed
and buried sample counts. A subject with no exposed reference point raises
`subject_embedded`. Any exposed reference point hidden by intervening terrain
still counts as an occluded frame. Framing uses the full declared AABB.

`FramingSpec.region` uses normalized screen coordinates with Y increasing
downward. `size_frac_band` bounds the larger projected AABB dimension, and
`headroom` requires the projected top to be at least that fraction below the
screen top. FOV and aspect must match the eventual render. Optional fixed eyes
constrain the shot endpoints; bearings/elevations determine generated eyes.
`reveal` endpoints must rise or pull back from the subject; `dolly` must end
closer to it along a straight line. A dolly never detours or lifts around an
obstruction: its clearance, framing or occlusion diagnostic names the blocking
constraint. The straight dolly clearance test splits the actual line at DEM cell
boundaries and evaluates all quadratic clearance extrema of the bilinear surface.
It also checks fixed endpoints against their actual terrain height. Off-path
walls in an enclosing box cannot reject the line. Framing distance, Y-up
orientation, and screen-region attainability retain distinct diagnostics.
Consumer resampling, projected framing, visibility and physical GPU trace gates
remain mandatory. A dolly validation failure keeps its binding diagnostic;
requesting a geometric repair does not turn it into `dolly_intent`.
Contradictory endpoints raise `reveal_intent` or `dolly_intent`.

The Rust planner searches a deterministic visibility graph over the DEM lattice
and framing distance shell. Search cell counts are limited by the declared
clearance distance, and candidate visibility is evaluated when the search
reaches it. Each routed reveal/flyover edge must be above a rectangle of the dilated
PROMETHEUS min-max pyramid. Sight lines solve the bilinear surface's quadratic
height along each crossed cell, including interior extrema. Quintic segments stop at waypoints; their position
and look-target are C2 and minimize integrated squared jerk for those endpoints
and endpoint derivatives. This conservative formulation may reject a path that
a less conservative planner could find. `CinemaInfeasible.diagnostic` names the
binding constraint; an unsafe best-effort path is never returned.
Straight corridor segments stop at interior waypoints. An orbit instead uses
one analytic circular arc with a quintic minimum-jerk angle profile. Angular
motion stops only at the arc's two ends. Its horizontal radius is fixed by the
start's framing distance; optional vertical endpoint lifts preserve fixed eyes
and join the arc with zero velocity and acceleration. Cardinal-angle intervals
bound exact X/Z extrema, so dilated PROMETHEUS rectangles enclose the entire
arc continuously. The interval count does not add motion knots. Independent
validation can lift the whole arc for clearance or occlusion, within the declared
framing shell. The emitted cubic's radius error is recorded as a diagnostic;
consumer position error retains the specification's 0.1% subject-distance gate.
No separate orbital shape tolerance is inferred from that gate.
Fixed and automatically joined orbit starts retain the winding of the declared
starting bearing: 270° to 360° sweeps 90°. A fixed eye on a contradictory starting
bearing raises `orbit_start_bearing`; only floating-point roundoff is excused.
Canonical saved straight segments retain their original fields and bytes.
Analytic segments add optional `arc` data (center and sweep), omitted on legacy
segments. Legacy plans remain readable; existing animation keyframes and their
serialization are unchanged. Replay rejects inconsistent analytic endpoints.
Saved-path tests check position joins and the consistency of the declared
rest-to-rest basis. Their six positions are generated from that basis, so they
do not independently establish C2 continuity. The native validator compares
the solved spline's position, velocity and acceleration on both sides of each
knot. These checks do not establish C2 playback interpolation.
Independent clearance or occlusion failures insert a lifted waypoint. Framing
failures add an aiming waypoint on the existing path. Straight paths repeat corridor
search and fitting within `max_refine_passes`; orbit repair refits the whole arc. Edges crossing a vertical
spherical-camera pole are routed around so their azimuth can be resampled. Cubic
approximation failures refine the keyframe sampling within that same budget.
Initial keyframe density follows the quintic degree and segment count, then refines as required by
consumer approximation and jerk checks. Validation remains dense regardless
of the emitted keyframe cadence.
Search altitudes reserve the allowed 0.1% resampling displacement at the outer
framing radius plus native rounding, protecting clearance during cubic baking.

The separate validator evaluates the actual native consumer cubic, using the
renderer projection matrices, bilinear terrain clearance, and physical GPU
queries to unchanged production `terrain_trace`. Deterministic down/forward
ray fans and center/eight-corner subject rays measure sampled clearance and
occlusion. `max_jerk` bounds each consumer interval from its stored native key
values using exact polynomial extrema and conservative spherical-camera chain
rules. Both this interval bound and `spline_max_jerk` must meet the declared budget.
`analytic_max_jerk` records sampled exact derivatives; `consumer_fd_max_jerk`
records dense double-precision finite differences as a diagnostic only.
Differences crossing knots depend on the sampling step because consumer
acceleration can jump. `consumer_max_acceleration_jump` reports the largest
left/right acceleration difference at an interior keyframe in world units/s².
The unchanged playback cubic has no finite global jerk bound at such jumps;
passing the interval gate does not establish global bounded jerk or C2 playback.
When a reveal/flyover's shortest corridor fails consumer jerk, planning may retry
an analytic arc for centered framing or a lateral corridor whose offset comes
from the declared framing shell. Centered arc phases share duration equally,
using the reference rule's timing. All motion, clearance, visibility, framing
and resampling gates still apply within the declared refinement passes. Dolly
and requested orbit paths cannot use these alternatives. Budgets never change.
`sampled_max_jerk` retains the native float32 evaluation's finite differences,
including arithmetic roundoff, as a diagnostic. It is not treated as physical
motion or used to relax the declared motion budget. Existing animation
interpolation is unchanged. Reports also include dense native speed, spline
jerk (a conservative analytic bound for circular segments), knot continuity, relative resampling error, linear-path ablation, adapter,
backend and ray counts. Ray fans are sampled cones, not a proof over all ray
directions. The continuous corridor proof applies to the solved spline; the
consumer is checked at the declared dense temporal resolution.
The trace dispatch is a diagnostic compute query that produces no rendered
frame; it is excluded from the render-certificate entry-point contract.

`to_json()` / `from_json()` preserve canonical JSON, including the terrain
digest, spline, keyframes, solver statistics and measured report. NaN is
rejected and negative zero is canonicalized. `validate(source)` renews physical
validation of a replay and rejects a different DEM. Captured reports are data,
not signatures; untrusted artifacts need revalidation. Two adjacent shots must
share their eye and look target. The joined consumer cubic is validated again.

Orbit inputs allow a fixed endpoint discrepancy of at most 0.1% of its distance
to the subject center. The starting eye is preserved exactly; its actual bearing
is unwrapped nearest the requested starting bearing. A contradictory fixed
start raises `orbit_start_bearing`. A joined start uses the exact preceding eye
and derives its bearing from that eye, superseding the nominal starting bearing.
The requested signed sweep (end bearing minus start bearing), including clockwise
motion and multiple turns, is always retained. A supplied end is projected onto
that arc, preserving its supplied altitude, only within the same tolerance;
otherwise `orbit_endpoint` is raised. `ShotPlan.orbit_adjustments` records
requested/effective inputs and displacement. Projection must still pass all
clearance, framing, visibility, motion and consumer resampling gates.

Fixture jerk budgets are frozen in `tests/data/autochiron_jerk_budgets.json`.
`tests/cinema_reference.py` derives them solely from declared geometry and
duration, without importing the solver. The reference uses three equal-duration
rest-to-rest minimum-jerk phases (rise, sweep, settle), with constant look target
at the declared subject center. The top is the greater of terrain maximum plus
clearance and subject upper Y. Interpolate spherical coordinates about that
center with q(u)=10u³−15u⁴+6u⁵; use the shortest azimuth change except that an
orbit retains its declared signed sweep. For each phase of duration h, write
D=|Δphi|+|Δtheta| and R=|Δradius|. Their derivative bounds are
v=15D/(8h), a=10D/(sqrt(3)h²), j=60D/h³ and the corresponding vr, ar, jr
using R. Its conservative Cartesian jerk bound is
jr+3ar·v+3vr·(a+v²)+max(radius0,radius1)·(j+3v·a+v³).
The budget is the largest of the three bounds, without a calibration factor.
This independently declared yardstick need not itself satisfy the scene gates.

| Fixture | Frozen jerk (world units/s³) |
| --- | ---: |
| Knife edge | 188.63378265834356 |
| Valley col | 188.63378265834356 |
| Summit cliff | 28.61644347291019 |
| Canyon straight dolly | 8.032319345455331 |
| Terrain attached mountain | 470.8511574047889 |
| Infeasible scene | 408.32632039418183 |

The canyon's declared cameras enter at x=12.5, below 64-unit side walls at
z=43.75 and z=56.25, and
cross a 51-unit floor ridge at y=55. The true straight-line clearance is 4 units
against the declared 3. An interior ridge raised to 54 is a rejection control;
both endpoints remain clear. Its reference top stays 68, so the rule and frozen
8.032319345455331 budget are unchanged by this terrain correction.

Valley col's measured solved-spline jerk is 188.10547788854922 against
188.63378265834356: 0.52830476979434 world units/s³, or about 0.2801%, of
headroom. It is a near-budget acceptance case, not evidence of a robustness
margin. Reports include signed spline and consumer-interval headroom. The budget
is not increased and no extra acceptance margin is invented.

The insufficient-budget control declares 30·15/16³=0.10986328125 before
solving a 15-unit, 16-second straight radial dolly. Its analytic minimum-jerk
requirement is 60·15/16³=0.2197265625; it must reject with `duration_speed_jerk`.
Both the solved spline and actual consumer interval bounds must meet the frozen
budget; consumer acceleration jumps remain reported and prevent a global C2 claim.

`link_certificate(path, pubkey=..., frame_time=...)` associates an existing
verified render certificate's payload digest with the keyframe digest and
frame time. It neither signs the plan nor rewrites the certificate, and this
association does not attest that the certificate depicts that frame. The
viewer animation path remains outside offscreen render-certificate scope.

The authored evaluation inputs are in `tests/data/cinema_scenes/`. Run the
physical acceptance tests with freshly rebuilt bindings:

```powershell
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:WGPU_BACKEND = 'vulkan'
python -m pytest tests/test_cinema_solver.py -v -s --tb=short
```

Acceptance tests fail when native planning or a physical GPU is unavailable.
They do not skip. No renderer or trace result is substituted with CPU output.
GPU absence and negotiated trace-capacity failures raise
`CinemaNativeUnavailable`, rather than an infeasible geometry diagnostic.

Build the existing viewer and render the two-shot example from the same checkout:

```powershell
$env:CARGO_TARGET_DIR = 'D:\forge3d\.worktrees\autochiron\target'
cargo build --release --bin interactive_viewer --features async_readback,enable-gpu-instancing
python examples/autochiron_reveal_orbit.py --output artifacts/autochiron-example
```

If the executable lives in a custom `CARGO_TARGET_DIR`, set
`FORGE3D_VIEWER_BINARY` to its absolute path. The example writes the canonical
plan, measured reports, PNG frames and viewer adapter statistics. Its declared
framing aspect is square; `--size` changes both output dimensions.
It explicitly enables the existing PBR terrain path, whose world-height mapping
matches the planner. The legacy simple preview normalizes relief by raster
width and would flatten this scene. Viewer defaults remain unchanged.

`examples/autochiron_render_golden.py` renders the baked two-shot camera at
31 seconds twice on physical hardware. Its frame revisions and actual consumer
pose are checked by `tests/test_determinism_hash.py` against the approved PNG
hash and included PNG in `tests/goldens/determinism/autochiron_viewer_v1.*`. The approved
candidate used RTX 3070 / Vulkan, driver 610.60. Viewer frames remain outside
offscreen certificate scope; no signing or frame-certificate attestation is
claimed. Other GPU and WASM portability are not v1 acceptance requirements.
General test runs qualify this reference only on physical NVIDIA/Vulkan with a
built viewer; other backends or unavailable prerequisites report a skip. The
NVIDIA determinism CI lane builds the viewer, uses the caller's exact installed
wheel, pins Vulkan, sets `FORGE3D_REQUIRE_AUTOCHIRON_GOLDEN=1`, and requires zero
skips. With that flag a missing viewer, wrong backend or unqualified adapter fails.
