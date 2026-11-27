# Resume a terrain render or poster

The native hybrid terrain renderer can save its accumulation after a completed
GPU frame. A later Python process can continue from that frame using the same
scene, Forge3D native binary, shaders, physical adapter, backend and driver.
Checkpoints require a physical Vulkan adapter that reports device and driver
UUIDs and a driver version. The loaded native extension must expose a
linker build identifier, as in the Windows MSVC and Linux builds. Rebuilding
the extension can invalidate existing checkpoints, even at the same Git
revision. Set WGPU_BACKEND=vulkan before importing Forge3D if your system
defaults to another backend.

```python
from forge3d import TerrainRenderCancelled, hybrid_render_terrain_reference

def report(progress):
    print(progress.completed_frames, progress.variance_at_frame,
          progress.variance, progress.checkpoint_path)
    return not stop_requested()  # application-owned cancellation signal

# Supply the same DEM, camera, dimensions and render settings on restart.
try:
    result = hybrid_render_terrain_reference(
        dem, width, height, camera,
        checkpoint_path=checkpoint, checkpoint_every=frames_per_checkpoint,
        on_progress=report,
    )
except TerrainRenderCancelled as cancelled:
    print(cancelled.progress.completed_frames, cancelled.checkpoint_path)

# In a later process, after a checkpoint exists:
result = hybrid_render_terrain_reference(
    dem, width, height, camera, checkpoint_path=checkpoint, resume=True,
)
```

`on_progress` runs synchronously on the caller's thread. A render started
inside the callback has its own certificate capture. Return `False` to
cancel or return `None` or `True` to continue. Callback exceptions propagate
to the caller. Cancellation raises `TerrainRenderCancelled`, a `RuntimeError`
whose `code` is `terrain_render_cancelled`. Its `progress` and
`checkpoint_path` report the last completed boundary and durable record.

A progress event includes elapsed time for the invocation, completed frames
for the current tile or terrain render, and the most recent variance value,
measurement frame and age in frames. All three variance fields are `None`
until the first convergence measurement. A checkpoint between measurements
keeps the original measurement schedule on resume.

For `render_terrain_poster`, pass a path to a dedicated job directory. Each
completed tile is saved with its global pixel window and diagnostics. Set
`checkpoint_every` to also save frames within each tile. The callback's `tile`
is a zero-based `(column, row)` pair, and `tile_complete` marks a completed
tile boundary. Resuming skips completed tiles and continues a partial tile
from its last saved frame. A tile with no saved frame starts at frame zero.

Checkpoint cadence is supplied by the caller. During terrain resume, omitting
the cadence keeps the existing checkpoint unchanged. Use a fresh file or job
directory for a new render. Terrain checkpoint parent directories must already
exist. A checkpointed poster owns a snapshot of its input arrays for the duration
of the job. Resume compares the supplied scene to that snapshot.

When recovery or progress is enabled, terrain results include `resumed`,
`resume_frame` and `frames_dispatched`. Poster results include `resumed`,
`recovered_tiles` and `frames_dispatched`. A resumed certificate records
recovery and dispatch counts for that invocation. A checkpointed poster
certificate also binds the manifest, assembled RGBA and original GPU identity.

Files are validated against the current request before the renderer creates
image-sized GPU resources. A failed or cancelled job can leave recoverable
checkpoints and tile records. Its final output and certificate require a
successful render. Device recreation and cross-adapter recovery are outside
this workflow.

## Estimate a poster before rendering

```python
from forge3d import mem, render_terrain_poster

job = dict(heightmap=dem, width=width, height=height, tile=tile)
preview = mem.estimate({"entry_point": "render_terrain_poster", **job})
try:
    poster = render_terrain_poster(**job, process_limit_bytes=available_process_bytes)
except mem.MemoryAdmissionRefused as error:
    print(error.code, error.limit_bytes, error.estimate["contributors"])
else:
    print(poster["memory"])
```

Supply an **absolute process limit in bytes**, including memory the process already
owns. Omit the limit for unlimited rendering. The renderer repeats preflight at
the public entry point before converting the inputs, opening a checkpoint,
allocating the RGBA plate or calling the GPU. It refuses when the prediction
exceeds the limit. Preflight needs integer dimensions for the array inputs. An
unlimited call can render a NumPy-compatible provider with missing dimensions,
including `shape=None`. Its certificate records that the estimate was
unavailable and includes the measured OS peak. Limited calls fail before
converting such a provider.

The report gives the current baseline, prior OS high-water mark, predicted peak,
measured peak through render completion, and their ratio (predicted / measured).
Windows uses private committed bytes and peak private commit. Linux uses
`VmRSS` and `VmHWM`. macOS reads current resident size and the process high-water
mark. When a prior job masks the kernel peak, the native sampler supplies the
largest current footprint it observed during this job and labels it
`sampled_current`. The historical mark remains an upper bound. Other activity
in the same process contributes to both observations.
Limited calls return the report in `memory`. Unlimited calls preserve the
original result keys. After a successful poster, `diagnostics.render_certificate(sign=False)`
reads `process_memory` from the latest completed native capture.
`certificate.emit_render_certificate(path)` exports that same capture, including
when the poster was called without `certificate=`. A certificate requested
directly on the render uses the calling thread's completed capture. The
signed payload includes the memory measurements for that invocation.

Use `certificate.payload_sha256(cert)` to compare render identities. Its
input excludes the top-level `process_memory` and `flythrough_process_memory`
reports. The digest therefore matches the same render made before MODUS.
Use `certificate.verify(cert, trusted_public_key)` to authenticate a run,
including its reported peak and ratio. Editing either report invalidates
the signature. Existing certificates use the same signing and verification
format.

### MapScene and flythrough admission

Use the same byte limit for a terrain scene or a CHRONOS sequence.

```python
scene_job = dict(entry_point="MapScene.render", scene=scene, path="map.png")
preview = mem.estimate(scene_job)
scene.render("map.png", process_limit_bytes=available_process_bytes)
report = scene.last_process_memory

options = dict(camera_path=scenes, base_seed=7, samples=1, out_dir="frames")
preview = mem.estimate(dict(entry_point="render_flythrough", **options))
manifest = render_flythrough(**options, process_limit_bytes=available_process_bytes)
report = manifest.process_memory
```

Preflight runs before scene compilation, native renderer construction and
frame output. A flythrough charges one shared renderer, the largest live
frame, and the recipes, compiled frames and cached image payloads retained
across the sequence. A fresh frame's certificate includes `process_memory`.
The last fresh capture also carries `flythrough_process_memory` for the whole
sequence. An all-cache-hit run exposes its observations on the scene or
manifest. The saved CHRONOS manifest keeps its deterministic wire format.

For an unstyled terrain scene, the estimate charges each DEM cell for two
source-width values, one validity byte, three float32 values and two
eight-byte native or upload values. Frame buffers scale with output pixels,
guides. Offline accumulation textures have a separate device-local estimate.
The mapped readback uses a row
pitch rounded up to 256 bytes plus two 512-byte timing slots. The flythrough
estimate retains five descriptor-sized copies per compiled frame for the
prepared recipe, frozen inputs, native compiled data, provenance and manifest.
The process sum includes those retained bytes once even where CHRONOS also
reserves host budget for the same payload.

On the six unmodified-main controls, cold growth after the initial modeled
buffers ranged from 461,195,747 to 471,258,376 bytes. Admission uses the
largest residual plus the measured spread, 481,321,005 bytes, for cold runtime
cost. The one-sample device-local term starts at the observed 222,573,122-byte
peak for the default shadow atlas. The offline and shadow adjustments are
described below. These calibrations come from Windows
Vulkan on the RTX 3070.

With a 64-square output and a 2048-square input, the first unmodified-main
MapScene control peaked at 2,262,175,744 bytes. The identical stage-instrumented
repeat peaked at 1,419,735,040 bytes. A same-shape empty process held
883,490,816 bytes of private commit at entry. After deducting that baseline,
155,277,853 modeled bytes and the cold allowance, 742,086,070 bytes remained
on the high run. The terrain runtime allowance uses the ceiling of this
residual per normalized input byte, 45 times the largest float32 DEM payload.
The lower repeat makes this deliberately conservative. The experiment does
not establish which allocator produced the difference.

Estimation needs input dimensions before decoding. GeoTIFF inputs use the
native raster metadata reader for dimensions, band count and sample size.
The estimate includes configured virtual texture sources with known dimensions.
Opaque arrays and external geometry without decoded sizes can still lack
a finite estimate. A limited call rejects these inputs before rendering.
An unlimited call records the reason and proceeds.

### Reading the observed ledgers

The native sampler reads current process memory and allocation counters in
the same sampling pass. It runs while Python holds the GIL, at the Python
interpreter's switch interval and at instrumented render boundaries.
`ledger_samples.at_peak` records the sample with the largest observed current
footprint. Rust heap bytes are live allocation requests in the extension.
Python array bytes count owned buffers at instrumented copy sites, with views
and caller-owned input arrays excluded. The GPU ledgers retain the existing
host-visible budget and record device-local resources separately.

The measured residual subtracts growth in Rust heap bytes, tracked NumPy
bytes and mapped GPU buffers from process growth. Host-budget reservations
for Rust-owned payloads are already in the heap ledger. Device-local bytes
are reported separately. `kernel_to_sample_gap_bytes` gives the difference
between the reported peak and the largest synchronized sample. Sub-interval
spikes can leave a gap. `residual_by_phase_at_peak` assigns net residual
changes to the render operations executing when they were observed. This
locates the operation where memory rose. Allocation ownership inside the
GPU driver and foreign libraries requires separate evidence.
The observed residual starts at `ledger_samples.baseline.process`, immediately
after admission. The prediction starts at the earlier `process_baseline_bytes`.
Certificate serialization follows the report snapshot. Qualification also
checks the independent OS peak immediately after the public call returns.

The process prediction is the sum of
the baseline, the largest overlapping allocation phase, and a calibrated
allowance for driver, pipeline and allocator memory. The full plate costs
`4 × width × height` bytes. A contiguous input of the required dtype can be
borrowed from the caller. Checkpoint snapshots and dtype or layout conversions
allocate their own storage. Each native tile also owns its input vectors.
The phase calculation compares validation, upload, mesh building, mapped
readback and publication. It uses the actual largest tile dimensions and
256-byte texture row alignment. Rust output vectors become NumPy arrays by
ownership transfer. The renderer copies RGBA into a view of the plate and
releases the tile result before the next dispatch.

The separate host-visible and device-local estimates describe GPU allocations.
On the main-based poster, preflight assumes the DEM may contain no-data
samples. The native masked upload holds a temporary pair of float32 values per
sample and writes an RG32Float height texture. A finite-only DEM uses R32Float.
This upper bound lets preflight run before inspecting or copying the input.
For a NumPy masked array, the conversion phase includes the cast values,
the mask and the filled float32 array. A checkpoint's owned copy overlaps
the normalized heightmap in a later phase.
In the **predicted** peak phase, modeled CPU and mapped-host terms sum to
`known_growth_bytes`. The CPU min/max pyramid stays live through native
readback. The OS high-water counter and the synchronized sampler are independent.
The report records the gap between their observations.
The device-local ledger includes the DEM pyramid, scene textures, reservoirs,
G-buffers, output targets and timing resources. The process estimate charges the
mapped bytes once and excludes device-local bytes. The existing 512 MiB
host-visible gate still runs. Field `modeled_unexplained_bytes` subtracts the
baseline and predicted live bytes from the measured peak. The historical
poster-only tables below used this modeled residual. Current reports also
provide the observed residual in `unexplained_bytes`.

### Windows poster qualification, 2026-11-26

The source checkout was `7c81644a`, 91 commits ahead of and 870 behind verified
`origin/main` at `cf678e48`. The first candidate used local Python changes on
`cf678e48`. Both sides used the same Forge3D 1.52.0 extension, SHA-256
`63b6d8d21508e58298759de886d5197eb86f83b9b686dd47c3eb792bccb74a44`,
on Windows 11 build 26300, Python 3.13.14, NVIDIA RTX 3070, Vulkan driver
610.60. The installed control's poster, checkpoint and certificate Python
sources matched the base revision after line-ending normalization.

The first qualification held the DEM at 192². Independent review then measured
a 1399455744-byte peak for an admitted 1146987452-byte limit using a 2048² DEM
and a 4² poster with tile 2. The following measurements are for the correction.
Each row ran in a fresh process with seed 7, one sample per pixel and 32 frames
per rendered tile. The material case converts a Fortran-order float64 DEM and
float64 RGBA material. All memory columns are Windows private commit in bytes.

| DEM | Poster, tile | Baseline | Predicted | OS peak | Ratio | Unexplained |
|---|---|---:|---:|---:|---:|---:|
| 192² | 4², 2 | 868626432 | 1016635188 | 1010540544 | 1.0060 | 140212944 |
| 1024² | 4², 2 | 872738816 | 1151541044 | 1077346304 | 1.0689 | 173841104 |
| 960 × 1536 | 4², 2 | 874192896 | 1268993892 | 1142894592 | 1.1103 | 212157808 |
| 2048² | 4², 2 | 885280768 | 1579319092 | 1373773824 | 1.1496 | 365451984 |
| 2048² | 4², 2, limit equal to preview | 885440512 | 1579478836 | 1400893440 | 1.1275 | 392411856 |
| 2048² | 8², 2 | 885350400 | 1579413204 | 1509130240 | 1.0466 | 500714288 |
| 192² | 513 × 257, 256 | 868401152 | 1256402852 | 1140961280 | 1.1012 | 250988004 |
| 192² | 256² material, 128 | 869789696 | 1082737504 | 999948288 | 1.0828 | 122667448 |
| 192² | 1024², 512 | 868499456 | 1981125584 | 1865932800 | 1.0617 | 911643064 |
| 192² | 513 × 257, 256, resumed | 868249600 | 1256401468 | 1078812672 | 1.1646 | 188840780 |

Every fresh unlimited plate matched the control byte for byte. The resumed run
read a pre-change tile receipt and continued the next tile at frame 8. It
dispatched 152 of 192 frames and matched the uninterrupted plate. The original
1146987452-byte limit now refused the 2048² DEM before dispatch with a
1579065140-byte estimate. The 16384² poster also refused at its 1073741824-byte
limit with a 2914953944-byte estimate. Both refusals and their corresponding
empty controls grew peak private commit by zero. An earlier cold working-set
probe grew by 53248 bytes on refusal against zero empty-call growth.

The allowance was calibrated **before qualification** using control builds and
the same private-commit metric. Control peak growth was 142069760 bytes for a
4 × 4 calibration plate with tile 2, then 132911104, 275857408 and 1009905664
bytes for tiles 128, 256 and 512. Subtracting modeled live bytes, including the
previous tile's 48 bytes/pixel of still-owned results in the control, left
140368400, 125332576, 251838604 and 912231520 bytes. The fixed term is the tiny
control's residual. The tile term is the largest observed increase in residual
per additional tile pixel, rounded up to 3359 bytes. Per-tile Python diagnostic
values measured 2024 bytes, or 2476 with recovery fields.

Holding each tile at 2² also exposed growth across repeated DEM uploads.
The original candidate, rendering sixteen tiles from a 2048² DEM in a fresh
process, grew by 649334784 bytes. Its largest modeled phase used 123065552
bytes. After accounting for this phase and the original fixed and tile
allowances, 385887396 bytes remained. The DEM, padded mip chain and placeholder
textures total 61516488 bytes. Rounding the observed ratio of 6.27 upward gives
the additional runtime allowance of 7 bytes per byte of scene texture payload.
This term estimates untracked process overhead empirically. Device-local
allocations still appear separately. Other allocation sizes, drivers, and
concurrent jobs can yield different residuals.

A fresh adapter probe alone grew peak private commit by 106438656 bytes.
Driver initialization is the largest separately measured untracked source.
The remaining tile-dependent driver and allocator costs are an aggregate
inference. The model has evidence for this matrix and platform. Other drivers,
Unix metrics, MapScene and flythrough require independent qualification.

To reproduce, run `python tools/measure_modus_poster.py JOB OUTPUT` separately
for each job with `WGPU_BACKEND=vulkan`. Windows validation uses psutil as an
independent observer. Pass `--control` only when running the pre-MODUS
implementation. Run each candidate job with `--reference CONTROL/JOB.png`.
Every candidate render requires a memory report from its requested certificate
or, with `--exact-limit`, from its limited result. An underestimate fails the
job. Run `empty` then `refuse`, and `empty-dem` then `refuse-dem`, sharing an
output directory for each pair. Run `cancel --control` on the control and
`resume` on the candidate with the same fresh `--checkpoint DIRECTORY`, giving
`resume` the plain control PNG as its reference. The tool writes per-process
JSON, compares pixels and enforces the peak and refusal gates.

### Rebuilt poster certificate capture

The release extension built from the uncommitted worktree at `cf678e48` had
SHA-256 `5a62ac11a8f596c794a4fcd44b508581c035a9f1a6d2be07cf0bdc17a8cc4516`.
During qualification, remote main advanced to `6d2a6f55`. At the final
check the tested base was 10 commits behind main, and the starting checkout
at `7c81644a` was 91 ahead and 880 behind.
On the same Windows RTX 3070 and Vulkan driver, all ten fresh-process jobs
again matched the control images. Both public certificate readers returned the
same memory report as the limited call. An unlimited 32² poster from a
`shape=None` array provider also matched its pre-change RGBA bytes.

| Job | Baseline bytes | Predicted bytes | OS peak bytes | Ratio | Unexplained bytes |
|---|---:|---:|---:|---:|---:|
| Calibration | 868114432 | 1016123188 | 1008480256 | 1.0076 | 138664656 |
| 1024² DEM | 872751104 | 1151553332 | 1077735424 | 1.0685 | 174217936 |
| 960 × 1536 DEM | 874164224 | 1268965220 | 1142222848 | 1.1110 | 211514736 |
| 2048² DEM | 885223424 | 1579261748 | 1374326784 | 1.1491 | 366062288 |
| 2048² DEM, exact limit | 885252096 | 1579290420 | 1373421568 | 1.1499 | 365128400 |
| 2048² DEM, sixteen tiles | 885145600 | 1579208404 | 1510215680 | 1.0457 | 502004528 |
| 513 × 257 poster | 868450304 | 1256452004 | 1141710848 | 1.1005 | 251688420 |
| 256² material poster | 869113856 | 1082061664 | 1000443904 | 1.0816 | 123838904 |
| 1024² poster | 868438016 | 1981064144 | 1865994240 | 1.0617 | 911765944 |
| 513 × 257 resumed | 867885056 | 1256036924 | 1078677504 | 1.1644 | 189070156 |

The 2048² DEM refused at 1146987452 bytes with a prediction of 1579253556.
The 16384² poster refused at 1073741824 bytes with a prediction of 2914892504.
The OS high-water mark grew by zero bytes for each refusal and its empty control.
The resume run restored one tile and dispatched 152 of 192 frames.

### Integration with main's no-data terrain renderer

On `6d2a6f55`, the unmodified release wheel's finite 4² calibration poster
grew peak private commit by 223064064 bytes. Subtracting its 1701360 modeled
live bytes gives the updated fixed allowance of 221362704 bytes. Calibration
preceded candidate measurements. A separate fresh call to `device_probe("vulkan")`
grew peak private commit by 17698816 bytes. This measures adapter discovery
alone. The rest of the untracked growth has no measured attribution on this build.

Candidate and control were release wheels of Forge3D 1.52.0 on Windows 11 build 26300, Python 3.13.14, NumPy 2.5.1 and an RTX 3070 using Vulkan driver 610.60. Both used base `6d2a6f55`. The uncommitted candidate extension had SHA-256 `c360ef9c8119a1ffe112de3752dfa376b2c3290e3d8c5d4743b963cf1f383342` and the unmodified control had SHA-256 `e32e84fdd1392fb5808b5dc161fbe127b687e1917f18696478f5ab13574c95d7`.

Twelve fresh-process candidate images matched unmodified main, including a 2048-square heightmap with one NaN, a NumPy masked array, an exact-limit job and checkpoint resume. Every independent OS peak matched the report and was within its prediction.

| Job | Baseline bytes | Predicted bytes | OS peak bytes | Ratio | Unexplained bytes |
|---|---:|---:|---:|---:|---:|
| calibration | 868491264 | 1098968884 | 1090637824 | 1.0076 | 220003024 |
| dem-1024 | 872263680 | 1274003252 | 1173868544 | 1.0853 | 258255568 |
| dem-rect | 874651648 | 1409429348 | 1210015744 | 1.1648 | 261125488 |
| dem-2048 | 885153792 | 1827958580 | 1450844160 | 1.2599 | 392317648 |
| dem-2048-many | 884801536 | 1827630804 | 1586728960 | 1.1518 | 528530224 |
| dem-2048-nodata | 885403648 | 1828208436 | 1574686720 | 1.1610 | 515910352 |
| masked | 868507648 | 1338683300 | 1209696256 | 1.1066 | 319469028 |
| plain | 867999744 | 1338027940 | 1209090048 | 1.1066 | 319518180 |
| material | 869703680 | 1164677984 | 1072795648 | 1.0856 | 195600824 |
| large | 868691968 | 2063344592 | 1950523392 | 1.0578 | 996041144 |
| resume | 868540416 | 1338718780 | 1149575168 | 1.1645 | 259312460 |
| dem-2048 exact limit | 885334016 | 1828138804 | 1406369792 | 1.2999 | 347663056 |

refuse predicted 3163867864 bytes at limit 1073741824. Refusal grew 0 bytes against 0 bytes for its empty control. refuse-dem predicted 1828011828 bytes at limit 1146987452. Refusal grew 0 bytes against 0 bytes for its empty control.

### Three-route Windows qualification, 2026-11-27

The candidate and the unmodified control use Forge3D 1.52.0 at
`6d2a6f5557e356b9628e167670e3aa6342f6d003` on Windows 11 build 26300,
Python 3.13.14, NumPy 2.5.1 and NVIDIA RTX 3070 Vulkan driver 610.60.
Candidate HEAD and the verified remote main coincide (0 ahead, 0 behind).
The starting checkout at `7c81644a` is 91 ahead and 880 behind.
The rebuilt extension SHA-256 is `5456442244ed7e787b683d68024e970a16d7794eafe7cb0d0bff36420cf3f9aa`.
The worktree used its own `target` directory. All changes are local and uncommitted.

Each row used a fresh process. The independent Windows peak private commit
taken after return matched the reported OS peak in all 30 renders.
The first twelve rows qualify the poster separately. Each poster matched
unmodified main pixel for pixel, including masked input, mixed NaN input,
the exact-limit render and checkpoint resume. The checkpoint recovered one
tile and resumed the next at frame 8. For MapScene and CHRONOS, each setting
ran at its public preview limit with a certificate and again without a limit.
The 64, 128 and 256 flythroughs used one, three and four frames respectively.
The 2048-input flythrough used three frames. PNG pixels, CHRONOS manifests
and provenance records matched unmodified main. The public certificate reader
returned the job report, and the requested certificate files matched it.

| Job | Process baseline B | Prediction B | OS peak B | Prediction / OS peak | Observed residual B | OS peak after return minus report B |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 867991552 | 1098469172 | 1059045376 | 1.0372 | 189037229 | 0 |
| dem-1024 | 872767488 | 1274507060 | 1172942848 | 1.0866 | 282936315 | 0 |
| dem-rect | 874442752 | 1409220452 | 1237823488 | 1.1385 | 333241538 | 0 |
| dem-2048 | 885092352 | 1827897140 | 1470681088 | 1.2429 | 522200060 | 0 |
| dem-2048-many | 885571584 | 1828400852 | 1587150848 | 1.1520 | 636521975 | 0 |
| dem-2048-nodata | 884994048 | 1827798836 | 1652920320 | 1.1058 | 669433519 | 0 |
| plain | 868864000 | 1338892196 | 1238999040 | 1.0806 | 349100657 | 0 |
| masked | 868274176 | 1338449828 | 1211654144 | 1.1046 | 320100604 | 0 |
| material | 869847040 | 1164821344 | 1072181248 | 1.0864 | 195927290 | 0 |
| large | 868769792 | 2063422416 | 1962389504 | 1.0515 | 1017490150 | 0 |
| resume | 868954112 | 1339132476 | 1148682240 | 1.1658 | 259099813 | 0 |
| exact dem-2048 | 885215232 | 1828020020 | 1470910464 | 1.2428 | 522306556 | 0 |
| map-64 admit | 866967552 | 1349265994 | 1338388480 | 1.0081 | 454160994 | 0 |
| map-256 admit | 867733504 | 1364593226 | 1337679872 | 1.0201 | 452793686 | 0 |
| map-512 admit | 868233216 | 1411689034 | 1345748992 | 1.0490 | 460363094 | 0 |
| fly-64 admit | 867078144 | 1349386021 | 1338093568 | 1.0084 | 453650779 | 0 |
| fly-128 admit | 867176448 | 1352415451 | 1338089472 | 1.0107 | 453340070 | 0 |
| fly-256 admit | 867500032 | 1364397494 | 1340121088 | 1.0181 | 455176815 | 0 |
| map-64-dem1024 admit | 871350272 | 1580300874 | 1361657856 | 1.1606 | 468703498 | 0 |
| map-64-dem2048 admit | 883957760 | 2275531338 | 1420308480 | 1.6021 | 502169241 | 0 |
| fly-64-dem2048 admit | 883924992 | 2275526875 | 1438625792 | 1.5817 | 520533418 | 0 |
| map-64 unlimited | 867151872 | 1349450314 | 1338105856 | 1.0085 | 453592818 | 0 |
| map-256 unlimited | 867364864 | 1364224586 | 1338605568 | 1.0191 | 454104727 | 0 |
| map-512 unlimited | 867885056 | 1411340874 | 1342500864 | 1.0513 | 456414038 | 0 |
| fly-64 unlimited | 867098624 | 1349406501 | 1337417728 | 1.0090 | 452885893 | 0 |
| fly-128 unlimited | 867246080 | 1352485083 | 1340358656 | 1.0090 | 455870985 | 0 |
| fly-256 unlimited | 867815424 | 1364712886 | 1339998208 | 1.0184 | 454737438 | 0 |
| map-64-dem1024 unlimited | 870850560 | 1579801162 | 1361162240 | 1.1606 | 473175703 | 0 |
| map-64-dem2048 unlimited | 883798016 | 2275371594 | 1420382208 | 1.6019 | 502417675 | 0 |
| fly-64-dem2048 unlimited | 883568640 | 2275170523 | 1440657408 | 1.5793 | 522921017 | 0 |

#### Refusal against an empty call

| Input | Limit B | Prediction B | Refusal growth B | Empty growth B |
|---|---:|---:|---:|---:|
| refuse | 1073741824 | 3163949784 | 0 | 0 |
| refuse-dem | 1146987452 | 1828065076 | 0 | 0 |
| map-64 | 1 | 1349278282 | 0 | 0 |
| fly-64 | 1 | 1349239808 | 0 | 0 |
| map-2048 | 1514657309 | 2274929226 | 0 | 0 |

The first two rows are the 16384-square plate and the original 2048-square
heightmap overrun. The last row uses the earlier MapScene preview as its
limit for a 64-square output backed by a resident 2048-square float32 DEM.
For the three scene refusal fixtures the independent current private commit
also matched the pre-call high-water mark and stayed unchanged after refusal.

#### Largest residual and repeated-process observation

For the 1024-square poster, the prediction was 868,769,792 baseline bytes
plus 85,790,280 modeled live bytes and runtime allowances of 221,362,704
fixed, 880,541,696 for the tile and 6,957,944 for scene data. The modeled
live phase included a 4,194,304-byte plate, 47,185,920 readback-vector bytes
and 20,971,520 bytes of reservoir initialization. Its synchronized sample
recorded 71,934,234 bytes of Rust heap growth, 4,194,304 of NumPy growth,
1,024 of mapped buffers and 103,608,576 device-local bytes. Process private
commit peaked at 1,962,389,504. The residual was 1,017,490,150 bytes.
Tile buffer and texture creation accounted for the largest net residual
increase at the sampled peak, 708,650,987 bytes. Tracer and pipeline creation
accounted for 117,429,137 bytes. The sampled process peak was 11,628,544
bytes below the kernel peak. The underlying driver allocations are unresolved.

In a separate fresh process a 513 x 257 poster established a kernel mark of
1,232,867,328 bytes. A subsequent 4 x 4 poster observed 1,217,257,472 bytes,
with a prediction / sample ratio of 1.1890 and a residual of 103,504 bytes.
The earlier kernel mark bounded the second job from above. Its sample is a
lower bound on the true job peak.

The runners are `tools/measure_modus_poster.py` and
`tools/measure_modus_scene.py`. Raw receipts, main controls and wheel
identification are under `artifacts/modus-routes/`. The matrix definitions
and CI gates are present in the worktree. On the matching release wheel,
236 focused Python checks passed, including API, concurrent certificate,
cache, CHRONOS manifest and repeated-process tests. The eight Rust certificate
tests and 22 resource tracker tests passed, as did `cargo forge3d-clippy`
and `git diff --check`. The Python tests used `FORGE3D_NO_BOOTSTRAP=1`,
`FORGE3D_TEST_INSTALLED_WHEEL=1`, `FORGE3D_RUN_TERRAIN_GOLDENS=1`,
`FORGE3D_RUN_OBLIQUA_GPU=1`, `FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN=1`
and `WGPU_BACKEND=vulkan`. MODUS remains open for integration,
certificate digest parity and input configurations lacking a preflight
estimate. Hosted CI and other operating systems and backends are unverified.

### Offline MapScene, flythrough and PCF qualification, 2026-11-27

The previous 64 × 64 four-sample MapScene render exceeded its
1,349,347,938-byte limit by 1,268,516,254 bytes. An independent run of
unmodified main reached 2,596,114,432 bytes from a baseline of 866,471,936.
Both high peaks inform the offline allowance. Other fresh processes on the
same machine peaked much lower for the same scene.

The process estimate adds 1,290,261,894 bytes for an offline session.
After deducting the ordinary live set, the two high-water runs needed an
additional 1,268,712,862 and 1,247,163,830 bytes. The allowance uses the
larger residual plus their observed spread. For larger outputs, two
unmodified-main four-sample controls at 256² and 1024² grew by 512,888,832
and 734,224,384 bytes. Their difference is 225.15 bytes per added pixel.
Frame arrays and mapped readback already charge 64 bytes per pixel, leaving
162 additional private bytes per pixel above 64² after rounding up. Sample
batches reuse the same textures. A flythrough charges the largest live frame
and retains its existing per-frame descriptor and cache accounting.

At 64² the reusable offline frame targets total 705,536 device-local bytes.
The estimate includes HDR and depth, three guides, six RGBA32F accumulation
textures, reference depth, quarter-resolution luminance, four resolved RGBA16F
targets and RGBA8 output. The half-resolution reflection pair adds 8,192
bytes. In a cold capture, shared renderer, material and environment resources
were live at atlas replacement. Summed per-label high-water marks were
254,080,414 bytes. Removing mapped buffers (17,408), heightmap (16,384), frame
targets and resized reflections leaves a 253,332,894-byte default shared
device envelope. Its default replacement atlas is counted once.

The render and preflight use the same terrain geometry resolver and offline
selection. The shadow ledger uses the terrain renderer's validated settings.
Depth costs four bytes per texel per cascade. Moment shadows add two
eight-byte surfaces for the moment atlas and blur intermediate. A 4096² PCF
depth atlas therefore grows from 67,108,864 bytes to 268,435,456 bytes between
one and four cascades. The two main controls differed by 202,551,296 bytes of
peak private growth across these settings. After deducting the ordinary
process allowance, both controls needed less than one additional private
byte per extra atlas byte above default. Admission rounds that factor up to one.
Device-local bytes remain a separate ledger.

A separate instrumented 64² four-sample run measured private commit of
1,223,131,136 bytes immediately before native offline initialization and
1,372,585,984 immediately after. The first accumulation batch brought it to
1,380,990,976. Resolve and tonemap did not raise that peak. Initialization
was the largest measured native offline step on this run. This trace does
not identify the allocator responsible for the higher cold peaks.

#### Corrected job matrix

Windows 11, GeForce RTX 3070, Vulkan driver 610.60, Forge3D 1.52.0,
Python 3.13.14 and NumPy 2.5.1. Candidate HEAD and verified origin/main
were both `6d2a6f55` (0 ahead, 0 behind). The starting checkout at
`7c81644a` was 91 ahead and 880 behind. The rebuilt wheel used the
worktree’s own Cargo target. Its native SHA-256 was
`5456442244ed7e787b683d68024e970a16d7794eafe7cb0d0bff36420cf3f9aa`.

Each row ran in a fresh process on the same 64² float32 DEM.
Bytes are Windows private commit. Ratio is prediction divided by the
independent post-call OS peak. Residual is private growth less observed
Rust heap, owned Python arrays and mapped buffers at the sampled peak.

| Job | Admission | Process baseline | Prediction | OS peak | Ratio | Residual |
|---|---|---:|---:|---:|---:|---:|
| Map 64², 2 samples | exact limit | 867,192,832 | 2,639,933,392 | 1,379,811,328 | 1.9133 | 490,443,249 |
| Map 64², 2 samples | unlimited | 867,172,352 | 2,639,912,912 | 1,378,840,576 | 1.9146 | 489,511,072 |
| Map 64², 4 samples | exact limit | 867,299,328 | 2,640,039,888 | 1,378,660,352 | 1.9149 | 489,147,898 |
| Map 64², 4 samples | unlimited | 866,988,032 | 2,639,728,592 | 1,380,151,296 | 1.9126 | 493,748,805 |
| Map 64², 8 samples | exact limit | 867,364,864 | 2,640,105,424 | 1,382,989,824 | 1.9090 | 491,225,768 |
| Map 64², 8 samples | unlimited | 867,061,760 | 2,639,802,320 | 1,382,498,304 | 1.9094 | 493,734,894 |
| Map 256², 4 samples | exact limit | 866,918,400 | 2,653,544,400 | 2,638,798,848 | 1.0056 | 1,748,836,234 |
| Map 256², 4 samples | unlimited | 867,151,872 | 2,653,777,872 | 1,379,229,696 | 1.9241 | 489,600,910 |
| Map 1024², 4 samples | exact limit | 867,336,192 | 2,876,129,232 | 1,600,520,192 | 1.7970 | 711,747,549 |
| Map 1024², 4 samples | unlimited | 866,869,248 | 2,875,662,288 | 1,602,347,008 | 1.7947 | 714,008,541 |
| Flythrough 64², two frames, 4 samples | exact limit | 866,951,168 | 2,639,710,598 | 1,382,379,520 | 1.9095 | 492,727,122 |
| Flythrough 64², two frames, 4 samples | unlimited | 866,979,840 | 2,639,739,270 | 1,383,415,808 | 1.9081 | 492,001,019 |
| Map 64², 4096 PCF × 1 | exact limit | 867,233,792 | 1,399,866,774 | 1,387,900,928 | 1.0086 | 503,273,620 |
| Map 64², 4096 PCF × 1 | unlimited | 867,229,696 | 1,399,862,678 | 1,388,478,464 | 1.0082 | 503,890,246 |
| Map 64², 4096 PCF × 4 | exact limit | 867,213,312 | 1,601,172,886 | 1,588,568,064 | 1.0079 | 704,199,724 |
| Map 64², 4096 PCF × 4 | unlimited | 867,450,880 | 1,601,410,454 | 1,590,247,424 | 1.0070 | 705,641,004 |
| Map 64², disabled grid | exact limit | 866,631,680 | 1,348,931,343 | 1,337,110,528 | 1.0088 | 453,298,327 |
| Map 64², disabled grid | unlimited | 867,520,512 | 1,349,820,175 | 1,271,730,176 | 1.0614 | 381,666,081 |

18 of 18 in-call reports matched the independent post-call peak exactly.
All exact-limit and unlimited RGBA outputs matched nine unmodified-main
controls. Both flythrough manifests and their frame provenance matched.
The runner also checked public certificate retrieval for every candidate
and exported certificates on the exact-limit runs. The host-visible and
device-local captures each stayed within their respective estimates.

The largest sampled residual was 1,748,836,234 bytes on Map 256², 4 samples
(admit). The largest observed increase was 1,115,066,952 bytes during
“render setup and publication”. The kernel peak was 0 bytes above the sample.
The allocation owner within that phase remains unidentified.

At the old 1,349,347,938-byte limit the 64² four-sample call predicted
2,639,564,752 bytes and raised `memory_admission_refused` before output.
Pre-call and post-call peak were both 866,824,192 bytes. The matching
empty control stayed at 866,721,792 bytes. Peak growth was zero for both.

Raw controls, phase measurements and final admission and parity receipts
are under `artifacts/modus-offline/`. The earlier poster qualification is
unchanged. MODUS remains open for certificate digest parity, complete
input coverage and integration of the uncommitted matrix. Hosted CI and
other platforms and backends were unrun.

### Active clipmaps, virtual textures and four-sample allocation tracing

On 2026-11-27, candidate HEAD and remote main were both `6d2a6f55`
(0 ahead, 0 behind). The starting checkout at `7c81644a` was 91 ahead and
880 behind. The measurements used Windows 11, RTX 3070, Vulkan 610.60,
Forge3D 1.52.0, Python 3.13.14 and a release wheel from the worktree's
own Cargo target.

Clipmap admission bounds the native planner's grid and skirt vertices,
the indexed renderer's retained LOD variants, draw buffers and upload
scratch. Each vertex has 36 bytes and each 32-bit triangle index has four.
The CPU planner and indexed renderer run sequentially. A three-ring
frustum draw also has two 240-byte selection readbacks and 24 bytes of
visibility statistics. On unmodified main, this 32² DEM with 64² output
reached 2,458,243,072 bytes of private commit. The device ledger excludes
the mapped selection buffers.

VT preflight resolves the requested families and tile geometry without
generating source pixels. It charges registered RGBA8 mip chains once,
along with conversion scratch, the tile working set, packed RGBA32F page
tables and bounded feedback. Each enabled feedback readback is 262,152
bytes. In the qualified build, the three copy-only staging buffers are
device-local. Each has the larger of 8 MiB and one RGBA tile with borders.
The atlas estimate covers the raw RGBA8 and three-family BC layouts without
requiring adapter discovery at admission.

#### Source of the offline residual

A diagnostic Vulkan layer sampled Windows private commit immediately
before and after driver calls in the rendering PID. The application inputs
were the same 256² output, 64² DEM and four samples as the largest previous
residual. A new per-process shader-cache directory exercised cold pipeline
creation. The layer forwarded the Vulkan arguments unchanged. The rendered
pixels matched the control.

The process started at 866,717,696 bytes and reached 2,616,442,880.
The sampled and kernel peaks agreed. One `vkCreateGraphicsPipelines` call
advanced the kernel peak from 1,341,345,792 to 2,616,442,880 bytes.
Private commit fell to 1,747,988,480 by the time the call returned.
Graphics-pipeline creation accounted for 1,320,755,200 bytes of increases
in the kernel high-water mark over the traced calls. `vkAllocateMemory`
accounted for 281,829,376 bytes. The synchronized residual at the peak
was 1,726,701,543 bytes. The largest observed source was temporary host
memory inside the NVIDIA Vulkan driver's graphics-pipeline creation.
The layer does not resolve the driver's internal allocator symbols.

The warm run of the same frame reached 1,379,536,896 within the traced
calls. Its driver memory-allocation calls accounted for 290,381,824 bytes
of increases in the kernel peak. PID-separated call receipts, profiler
source and cold/warm frame measurements are in
`artifacts/modus-stream/driver-probe/`, `profile-scoped-cache/` and
`profile-pid-256/` under the same artifact root.

#### Active-feature job matrix

All jobs used a 32² float32 DEM and 64² output. Clipmaps used three rings,
16² ring resolution and 16² center resolution. VT registered four materials
for each of albedo, normal and mask at 256 × 512. The atlas side was 1024
unless listed as 4096. The two-frame flythrough used both features and a
4096 atlas to accommodate the frozen per-frame residency schedule. Both
main and candidate rejected that schedule with a 1024 atlas.

Each row used a fresh process. Counts are bytes. Ratio divides the prediction
by the independent post-call Windows peak. Residual uses the synchronized
CPU and mapped-host observations at the sampled peak.

| Job | Admission | Baseline | Estimate | OS peak | Ratio | Residual |
|---|---|---:|---:|---:|---:|---:|
| clipmap | exact limit | 867,291,136 | 2,645,848,301 | 1,364,307,968 | 1.9393 | 463,941,770 |
| clipmap | unlimited | 866,856,960 | 2,645,414,125 | 1,363,668,992 | 1.9399 | 464,052,586 |
| vt | exact limit | 867,463,168 | 1,376,628,812 | 1,282,654,208 | 1.0733 | 389,120,925 |
| vt | unlimited | 867,119,104 | 1,376,284,748 | 1,349,300,224 | 1.0200 | 456,113,045 |
| both | exact limit | 867,069,952 | 2,673,160,943 | 1,372,299,264 | 1.9479 | 463,771,947 |
| both | unlimited | 867,328,000 | 2,673,418,991 | 1,373,765,632 | 1.9461 | 464,996,327 |
| 4096 atlas | exact limit | 866,975,744 | 1,424,896,340 | 1,418,145,792 | 1.0048 | 525,072,470 |
| 4096 atlas | unlimited | 866,779,136 | 1,424,699,732 | 1,416,282,112 | 1.0059 | 523,296,894 |
| two-frame flythrough | exact limit | 867,733,504 | 2,725,946,495 | 1,314,893,824 | 2.0731 | 411,827,531 |
| two-frame flythrough | unlimited | 866,910,208 | 2,725,123,199 | 1,513,385,984 | 1.8007 | 604,941,367 |

All ten predictions covered the OS peaks and the respective host-visible
and device-local captures. All ten sets of frame pixels matched their
five unmodified-main controls. The flythrough manifest and frame provenance
also matched. The runner checked public memory reports and exact-limit
certificate exports. Nine reports equaled the post-call OS peak. The
exact-limit clipmap report captured 1,363,304,448 bytes, and the independent
counter subsequently reached 1,364,307,968. The table uses the latter.
A separate VT-only run using an isolated shader cache
predicted 1,375,957,068 bytes and reached 1,371,762,688.

| Refusal at 1 byte | Estimate | Pre-call OS peak | Post-call OS peak | Empty-call growth |
|---|---:|---:|---:|---:|
| clipmap | 2,645,856,493 | 867,299,328 | 867,299,328 | 0 |
| vt | 1,376,002,124 | 866,836,480 | 866,836,480 | 0 |

Both refusals raised `memory_admission_refused` with the limit, estimate
and contributors before output. Eight focused estimator and refusal tests
passed on the final installation. Entry points and fixtures are in
`tools/measure_modus_scene.py` and `tests/test_modus_scene_memory.py`.
Receipts are under `artifacts/modus-stream/{control,verified,extra,final-tests}`.
Local runs used `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_TEST_INSTALLED_WHEEL=1`,
`WGPU_BACKEND=vulkan` and `FORGE3D_DETERMINISM_TEST_BACKEND=vulkan`.
The separate poster matrix was reused. Hosted CI, other backends and the
previously blocked SSAO shader path were unrun.

### Qualification of tracked CPU growth

The CPU estimate covers growth from the pre-call baseline. At each native
observation the sampler reads the Rust heap and Forge3D-owned NumPy array
bytes. It retains the sample with the largest **combined** CPU count in
`ledger_samples.at_cpu_peak`. Subtracting the same two baseline counters
gives `sampled_cpu_peak_growth_bytes`. Array views share the owner's counter.
This sampled maximum is a lower bound between observations. The independent
process-peak sample still supplies the synchronized ledgers for the residual.
Both qualification runners require the CPU sample and reject any underestimate,
even when CPU usage at the process peak would pass.

Preflight adds a measured fixed CPU cost to the shape-derived buffers alive
in the same native phase. For a flythrough it takes the largest such frame
and includes the retained recipes, compiled frames and cache payloads.
Each fixed cost is a share of the existing process runtime allowance. Adding
it again to the process estimate would count those bytes twice.
`cpu_peak_contributors` gives the terms for the selected CPU phase.

| Renderer | Calibration jobs | Tracked growth above shape model, bytes | Fixed CPU cost, bytes |
|---|---|---:|---:|
| Grid terrain | 64² grid and disabled-grid configuration | 19,688,022–22,253,053 | 24,818,084 |
| Offline terrain | 256² and 64² output, four samples | 22,287,198–25,288,765 | 28,290,332 |
| Indexed clipmap | Ring resolutions 16 and 4, 32² input, 64² output | 28,863,580–34,477,432 | 40,091,284 |
| Poster tracer | 32² compact poster and calibration poster | 1,624,403–2,166,094 | 2,707,785 |

Each fixed cost is the largest listed shortfall plus the difference between
the two shortfalls. A staged clipmap probe measured 15,659,097 bytes of Rust
heap growth during terrain-renderer construction alone. Before calibration,
the 6,772,034-byte prediction missed both the user's 33,234,919-byte observation
and a fresh 35,635,614-byte combined CPU sample. In the latter job the CPU
growth at the process-peak sample was only 32,072,358 bytes.

An opaque array provider can expose neither dimensions nor a buffer until
conversion. Preview and limited rendering then raise a TypeError with code
`memory_estimate_unavailable` before conversion. A limited call carries
`limit_bytes`. Unlimited rendering still accepts the provider and records
the reason a finite estimate was unavailable.

#### Rebuilt-wheel results for the CPU ledger

Windows 11, Python 3.13.14, Forge3D 1.52.0, NVIDIA RTX 3070, Vulkan 610.60. Candidate and verified origin/main were `6d2a6f55` (0 ahead, 0 behind). The starting checkout was `7c81644a` (91 ahead, 880 behind). All qualification changes are local. Each row used a fresh interpreter with the candidate release wheel. CPU columns give the predicted growth and the two simultaneously observed components of the sampled maximum, in bytes. The process ratio uses the independent post-call Windows peak private commit. Residual uses the synchronized ledgers at the sampled process peak. The final column measures the independent peak minus the peak in the certificate.

| Job | Mode | CPU estimate | Rust growth | NumPy growth | CPU estimate / observed | Process baseline | Process estimate | OS peak | Process estimate / OS | Residual | OS minus report |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact/dem-2048 | exact | 176,080,505 | 64,980,907 | 64 | 2.7097 | 885,493,760 | 1,828,298,548 | 1,426,817,024 | 1.2814 | 477,934,540 | 0 |
| poster/calibration | unlimited | 4,851,321 | 4,309,566 | 64 | 1.1257 | 868,835,328 | 1,099,312,948 | 1,073,553,408 | 1.0240 | 202,701,437 | 0 |
| poster/compact | unlimited | 3,044,313 | 1,873,575 | 4,096 | 1.6213 | 868,515,840 | 1,093,789,720 | 1,051,205,632 | 1.0405 | 180,843,354 | 0 |
| poster/dem-1024 | unlimited | 46,057,081 | 18,843,604 | 64 | 2.4442 | 872,919,040 | 1,274,658,612 | 1,145,856,000 | 1.1124 | 255,695,838 | 0 |
| poster/dem-2048 | unlimited | 176,080,505 | 64,980,907 | 64 | 2.7097 | 885,575,680 | 1,828,380,468 | 1,471,279,104 | 1.2427 | 522,314,700 | 0 |
| poster/dem-2048-many | unlimited | 176,104,985 | 65,274,311 | 256 | 2.6979 | 885,092,352 | 1,827,921,620 | 1,587,474,432 | 1.1515 | 637,330,694 | 0 |
| poster/dem-2048-nodata | unlimited | 176,080,505 | 98,491,496 | 64 | 1.7878 | 885,547,008 | 1,828,351,796 | 1,572,577,280 | 1.1626 | 591,005,388 | 0 |
| poster/dem-rect | unlimited | 76,946,393 | 31,732,563 | 64 | 2.4248 | 874,516,480 | 1,409,294,180 | 1,210,482,688 | 1.1642 | 305,836,172 | 0 |
| poster/large | unlimited | 88,497,041 | 79,274,498 | 4,194,304 | 1.0602 | 868,384,768 | 2,063,037,392 | 1,961,824,256 | 1.0516 | 1,005,746,686 | 0 |
| poster/masked | unlimited | 24,426,341 | 19,686,803 | 674,820 | 1.1996 | 868,986,880 | 1,339,162,532 | 1,239,502,848 | 1.0804 | 343,632,758 | 0 |
| poster/material | unlimited | 10,197,905 | 7,642,010 | 999,424 | 1.1801 | 870,445,056 | 1,165,419,360 | 1,098,911,744 | 1.0605 | 219,422,822 | 0 |
| poster/plain | unlimited | 24,278,885 | 20,771,285 | 527,364 | 1.1399 | 868,810,752 | 1,338,838,948 | 1,237,864,448 | 1.0816 | 345,910,823 | 0 |
| poster/resume | unlimited | 24,429,053 | 19,952,632 | 674,820 | 1.1843 | 868,655,104 | 1,338,833,468 | 1,148,514,304 | 1.1657 | 256,719,876 | 0 |
| atlas/fly | admit | 126,256,488 | 44,638,345 | 0 | 2.8284 | 867,221,504 | 2,725,434,495 | 1,512,632,320 | 1.8018 | 603,876,259 | 0 |
| atlas/fly | unlimited | 126,256,488 | 43,888,202 | 0 | 2.8768 | 867,069,952 | 2,725,282,943 | 1,511,079,936 | 1.8035 | 602,475,427 | 0 |
| atlas/map | admit | 100,953,795 | 29,826,628 | 0 | 3.3847 | 866,443,264 | 1,424,363,860 | 1,415,696,384 | 1.0061 | 522,195,794 | 0 |
| atlas/map | unlimited | 100,953,795 | 30,044,875 | 0 | 3.3601 | 867,401,728 | 1,425,322,324 | 1,414,877,184 | 1.0074 | 520,417,614 | 0 |
| offline/fly-64-s4 | admit | 28,663,023 | 25,677,992 | 0 | 1.1162 | 867,467,264 | 2,640,226,694 | 1,384,648,704 | 1.9068 | 491,942,755 | 0 |
| offline/fly-64-s4 | unlimited | 28,663,023 | 25,699,349 | 0 | 1.1153 | 867,237,888 | 2,639,997,318 | 1,385,218,048 | 1.9058 | 495,661,313 | 0 |
| offline/map-1024-s4 | admit | 78,779,193 | 25,630,274 | 0 | 3.0737 | 867,020,800 | 2,875,813,840 | 1,603,031,040 | 1.7940 | 714,561,453 | 0 |
| offline/map-1024-s4 | unlimited | 78,779,193 | 25,639,531 | 0 | 3.0726 | 867,037,184 | 2,875,830,224 | 1,601,945,600 | 1.7952 | 713,451,437 | 0 |
| offline/map-256-s4 | admit | 31,593,273 | 25,653,725 | 0 | 1.2315 | 866,979,840 | 2,653,605,840 | 1,382,285,312 | 1.9197 | 496,917,589 | 0 |
| offline/map-256-s4 | unlimited | 31,593,273 | 24,438,579 | 0 | 1.2928 | 867,069,952 | 2,653,695,952 | 1,380,880,384 | 1.9217 | 491,350,186 | 0 |
| offline/map-64-s2 | admit | 28,644,153 | 24,216,119 | 0 | 1.1829 | 866,914,304 | 2,639,654,864 | 1,380,212,736 | 1.9125 | 491,119,139 | 0 |
| offline/map-64-s2 | unlimited | 28,644,153 | 24,385,301 | 0 | 1.1746 | 867,123,200 | 2,639,863,760 | 1,379,274,752 | 1.9140 | 492,512,362 | 0 |
| offline/map-64-s4 | admit | 28,644,153 | 25,687,737 | 0 | 1.1151 | 867,459,072 | 2,640,199,632 | 1,379,430,400 | 1.9140 | 489,714,241 | 0 |
| offline/map-64-s4 | unlimited | 28,644,153 | 25,292,662 | 0 | 1.1325 | 867,602,432 | 2,640,342,992 | 1,380,069,376 | 1.9132 | 492,807,814 | 0 |
| offline/map-64-s8 | admit | 28,644,153 | 25,049,079 | 0 | 1.1435 | 867,000,320 | 2,639,740,880 | 1,380,622,336 | 1.9120 | 489,292,037 | 0 |
| offline/map-64-s8 | unlimited | 28,644,153 | 25,595,345 | 0 | 1.1191 | 866,697,216 | 2,639,437,776 | 1,381,601,280 | 1.9104 | 490,556,004 | 0 |
| offline/map-grid | admit | 25,042,054 | 21,434,211 | 0 | 1.1683 | 867,008,512 | 1,349,308,175 | 1,336,401,920 | 1.0097 | 450,982,809 | 1,003,520 |
| offline/map-grid | unlimited | 25,042,054 | 22,455,735 | 0 | 1.1152 | 866,869,248 | 1,349,168,911 | 1,336,442,880 | 1.0095 | 452,166,553 | 0 |
| offline/map-pcf1 | admit | 25,043,725 | 22,465,635 | 0 | 1.1148 | 866,848,768 | 1,399,481,750 | 1,389,707,264 | 1.0070 | 504,447,769 | 1,003,520 |
| offline/map-pcf1 | unlimited | 25,043,725 | 22,444,299 | 0 | 1.1158 | 867,287,040 | 1,399,920,022 | 1,387,675,648 | 1.0088 | 502,981,401 | 0 |
| offline/map-pcf4 | admit | 25,043,725 | 22,466,922 | 0 | 1.1147 | 867,020,800 | 1,600,980,374 | 1,591,726,080 | 1.0058 | 706,517,917 | 1,003,520 |
| offline/map-pcf4 | unlimited | 25,043,725 | 22,445,586 | 0 | 1.1158 | 867,090,432 | 1,601,050,006 | 1,587,994,624 | 1.0082 | 703,720,765 | 0 |
| ordinary/fly-128 | admit | 25,720,402 | 19,748,065 | 0 | 1.3024 | 867,233,792 | 1,352,472,795 | 1,339,301,888 | 1.0098 | 454,590,123 | 0 |
| ordinary/fly-128 | unlimited | 25,720,402 | 22,576,509 | 0 | 1.1393 | 867,168,256 | 1,352,407,259 | 1,340,588,032 | 1.0088 | 456,132,232 | 0 |
| ordinary/fly-256 | admit | 28,334,893 | 21,752,567 | 0 | 1.3026 | 867,758,080 | 1,364,655,542 | 1,338,580,992 | 1.0195 | 453,367,794 | 0 |
| ordinary/fly-256 | unlimited | 28,334,893 | 21,355,216 | 0 | 1.3268 | 867,844,096 | 1,364,741,558 | 1,340,854,272 | 1.0178 | 455,550,899 | 0 |
| ordinary/fly-64 | admit | 25,050,268 | 21,705,847 | 0 | 1.1541 | 866,930,688 | 1,349,238,565 | 1,337,753,600 | 1.0086 | 453,610,347 | 0 |
| ordinary/fly-64 | unlimited | 25,050,268 | 21,472,711 | 0 | 1.1666 | 867,024,896 | 1,349,332,773 | 1,338,589,184 | 1.0080 | 453,344,111 | 962,560 |
| ordinary/fly-64-dem2048 | admit | 180,106,834 | 38,491,113 | 0 | 4.6792 | 883,830,784 | 2,275,432,667 | 1,440,792,576 | 1.5793 | 522,756,265 | 0 |
| ordinary/fly-64-dem2048 | unlimited | 180,106,834 | 38,257,794 | 0 | 4.7077 | 884,019,200 | 2,275,621,083 | 1,438,932,992 | 1.5815 | 520,736,936 | 0 |
| ordinary/map-256 | admit | 28,297,153 | 22,690,827 | 0 | 1.2471 | 867,803,136 | 1,364,662,858 | 1,338,130,432 | 1.0198 | 452,142,695 | 1,003,520 |
| ordinary/map-256 | unlimited | 28,297,153 | 21,292,727 | 0 | 1.3290 | 868,003,840 | 1,364,863,562 | 1,272,545,280 | 1.0725 | 386,807,399 | 0 |
| ordinary/map-512 | admit | 38,717,377 | 23,477,259 | 0 | 1.6491 | 867,991,552 | 1,411,447,370 | 1,344,065,536 | 1.0501 | 458,875,686 | 0 |
| ordinary/map-512 | unlimited | 38,717,377 | 23,509,263 | 0 | 1.6469 | 868,274,176 | 1,411,729,994 | 1,343,123,456 | 1.0511 | 455,525,158 | 0 |
| ordinary/map-64 | admit | 25,040,833 | 21,446,628 | 0 | 1.1676 | 867,409,920 | 1,349,708,362 | 1,337,839,616 | 1.0089 | 452,016,457 | 1,138,688 |
| ordinary/map-64 | unlimited | 25,040,833 | 22,477,071 | 0 | 1.1141 | 866,856,960 | 1,349,155,402 | 1,337,016,320 | 1.0091 | 452,977,991 | 0 |
| ordinary/map-64-dem1024 | admit | 63,686,593 | 25,671,951 | 0 | 2.4808 | 871,907,328 | 1,580,857,930 | 1,360,535,552 | 1.1619 | 465,416,774 | 0 |
| ordinary/map-64-dem1024 | unlimited | 63,686,593 | 26,622,987 | 0 | 2.3922 | 871,096,320 | 1,580,046,922 | 1,361,350,656 | 1.1606 | 467,743,134 | 0 |
| ordinary/map-64-dem2048 | admit | 180,078,529 | 34,173,716 | 0 | 5.2695 | 883,736,576 | 2,275,310,154 | 1,420,476,416 | 1.6018 | 501,649,774 | 0 |
| ordinary/map-64-dem2048 | unlimited | 180,078,529 | 39,205,923 | 0 | 4.5931 | 883,806,208 | 2,275,379,786 | 1,421,557,760 | 1.6006 | 502,661,486 | 0 |
| stream/both | admit | 74,134,992 | 44,777,993 | 0 | 1.6556 | 867,610,624 | 2,673,701,615 | 1,371,947,008 | 1.9488 | 462,895,371 | 0 |
| stream/both | unlimited | 74,134,992 | 44,821,129 | 0 | 1.6540 | 867,106,816 | 2,673,197,807 | 1,372,659,712 | 1.9475 | 464,042,042 | 0 |
| stream/clip-min | admit | 41,352,918 | 35,719,399 | 0 | 1.1577 | 866,852,864 | 2,639,899,629 | 1,366,761,472 | 1.9315 | 466,800,795 | 1,007,616 |
| stream/clip-min | unlimited | 41,352,918 | 35,110,373 | 0 | 1.1778 | 867,229,696 | 2,640,276,461 | 1,363,992,576 | 1.9357 | 463,682,906 | 1,007,616 |
| stream/clipmap | admit | 46,863,318 | 36,292,745 | 0 | 1.2913 | 867,168,256 | 2,645,725,421 | 1,365,352,448 | 1.9378 | 465,109,082 | 1,003,520 |
| stream/clipmap | unlimited | 46,863,318 | 36,137,772 | 0 | 1.2968 | 867,000,320 | 2,645,557,485 | 1,364,344,832 | 1.9391 | 465,272,154 | 0 |
| stream/vt | admit | 52,198,843 | 29,830,120 | 0 | 1.7499 | 867,479,552 | 1,376,645,196 | 1,349,419,008 | 1.0202 | 455,881,246 | 0 |
| stream/vt | unlimited | 52,198,843 | 30,852,643 | 0 | 1.6919 | 867,180,544 | 1,376,346,188 | 1,280,937,984 | 1.0745 | 387,556,890 | 0 |

All 61 rows passed the CPU, host-visible GPU, device-local GPU and whole-process gates. All 121 output files compared with main matched byte for byte, including resumed poster pixels, flight manifests and frame provenance. Each receipt under `artifacts/modus-cpu/final/` contains the contributor breakdown and sampled counters. The estimates are calibrated for these workloads on this adapter.

| Refused job | Limit | Process estimate | OS peak before | OS peak after | Empty-call growth |
|---|---:|---:|---:|---:|---:|
| map terrain, 1 sample(s) | 1 | 1,349,610,058 | 867,065,856 | 867,065,856 | 0 |
| fly terrain, 1 sample(s) | 1 | 1,349,604,352 | 867,041,280 | 867,041,280 | 0 |
| map terrain, 4 sample(s) | 1,349,347,938 | 2,639,646,672 | 866,906,112 | 866,906,112 | 0 |
| map clipmap, 1 sample(s) | 1 | 2,645,291,245 | 866,734,080 | 866,734,080 | 0 |
| map vt, 1 sample(s) | 1 | 1,376,067,660 | 866,902,016 | 866,902,016 | 0 |
| poster refuse | 1,073,741,824 | 3,164,076,760 | 902,348,800 | 902,348,800 | 0 |
| poster refuse-dem | 1,146,987,452 | 1,828,667,188 | 885,862,400 | 885,862,400 | 0 |

The largest synchronized residual was 1,005,746,686 bytes on the 1024-square
poster. Of that, 708,708,331 bytes accumulated during tile buffer and texture
creation. The earlier cold four-sample MapScene trace identified a different
source on that job, a 1,275,097,088-byte increase in one NVIDIA graphics-pipeline
creation call.

Nine reports closed below the independent post-call OS peak, by up to
1,138,688 bytes. The qualification gate uses the independent peak. An additional
instrumented exact-limit clipmap run stayed at 1,273,266,176 bytes through
report attachment, certificate assembly, file export and return. This probe
does not locate the nine increases. Its boundaries are recorded under
`artifacts/modus-cpu/boundary/`.

Fourteen focused tests passed on the installed wheel. The native binary matched
the calibration wheel (SHA-256
`88520732f946d6a261e48377ec80026bd429e75015191f8c5e8aca8aa35b80e0`).
Render certificates include volatile process measurements in their signed payload.
Full MODUS remains open. All changes are uncommitted. Hosted CI, other platforms
and backends, and the previously blocked SSAO path were unrun.

### Whole-call peak at the public return boundary

On the previous wheel an additional exact-limit 64-square map reported 1,336,426,496 bytes. The kernel had reached 1,336,791,040 bytes before the independent reader ran. The reader left the high-water mark unchanged. A second, minimum-clipmap probe on that wheel showed no difference. The return discrepancy can occur before observation. These probes do not identify its allocation site.

The wrapper now releases the render implementation's local resources before taking the completion sample. It attaches the report to the completed capture, publishes the requested certificate, then reads the kernel high-water mark again. If report attachment or export advanced the peak, it updates the observation, ratio, residual, and published digest. The scene and poster runners require exact equality with the independent OS counter at public return in a fresh process. The separate estimate gate also covers the subsequent runner checks. A previous high-water mark in a long-lived process still requires the documented job-local bounds.

The table covers 61 independent interpreters and 121 byte-identical control files. All three ledger gates and process admission passed. The OS column uses Windows peak private commit. Certificate minus return peak was zero in each row. Raw calculation terms, CPU and GPU counters, synchronized residuals and frame hashes are in `artifacts/modus-return/qualified/`. The starting checkout was `7c81644a` (91 ahead, 880 behind). Candidate HEAD and origin/main were `6d2a6f55` (0 ahead, 0 behind). These are uncommitted candidate results from Forge3D 1.52.0, Python 3.13.14 and Windows 11 on the RTX 3070, Vulkan 610.60.

| Job | Mode | Baseline B | Estimate B | OS peak at return B | Estimate / OS | Sampled residual B |
|---|---|---:|---:|---:|---:|---:|
| exact/dem-2048 | exact | 885,448,704 | 1,828,253,492 | 1,406,631,936 | 1.2997 | 457,795,642 |
| poster/calibration | unlimited | 868,433,920 | 1,098,911,540 | 1,060,630,528 | 1.0361 | 190,179,965 |
| poster/compact | unlimited | 869,019,648 | 1,094,293,528 | 1,051,312,128 | 1.0409 | 180,446,042 |
| poster/dem-1024 | unlimited | 872,660,992 | 1,274,400,564 | 1,146,916,864 | 1.1112 | 257,004,268 |
| poster/dem-2048 | unlimited | 885,383,168 | 1,828,187,956 | 1,451,532,288 | 1.2595 | 502,747,731 |
| poster/dem-2048-many | unlimited | 885,526,528 | 1,828,355,796 | 1,543,467,008 | 1.1846 | 594,516,060 |
| poster/dem-2048-nodata | unlimited | 885,334,016 | 1,828,138,804 | 1,574,027,264 | 1.1614 | 590,199,826 |
| poster/dem-rect | unlimited | 874,500,096 | 1,409,277,796 | 1,211,293,696 | 1.1634 | 306,653,220 |
| poster/large | unlimited | 868,376,576 | 2,063,029,200 | 1,950,797,824 | 1.0575 | 994,716,158 |
| poster/masked | unlimited | 868,798,464 | 1,338,974,116 | 1,210,413,056 | 1.1062 | 318,326,001 |
| poster/material | unlimited | 870,047,744 | 1,165,022,048 | 1,073,033,216 | 1.0857 | 193,613,683 |
| poster/plain | unlimited | 868,814,848 | 1,338,843,044 | 1,210,736,640 | 1.1058 | 318,523,596 |
| poster/resume | unlimited | 868,708,352 | 1,338,886,716 | 1,147,940,864 | 1.1663 | 255,715,367 |
| atlas/fly | admit | 866,959,360 | 2,725,172,351 | 1,511,665,664 | 1.8028 | 603,172,386 |
| atlas/fly | unlimited | 867,336,192 | 2,725,549,183 | 1,511,878,656 | 1.8028 | 602,953,567 |
| atlas/map | admit | 867,053,568 | 1,424,974,164 | 1,415,462,912 | 1.0067 | 522,328,188 |
| atlas/map | unlimited | 867,090,432 | 1,425,011,028 | 1,415,458,816 | 1.0067 | 522,303,794 |
| offline/fly-64-s4 | admit | 866,627,584 | 2,639,387,014 | 1,383,571,456 | 1.9077 | 494,209,211 |
| offline/fly-64-s4 | unlimited | 866,975,744 | 2,639,735,174 | 1,385,689,088 | 1.9050 | 496,404,746 |
| offline/map-1024-s4 | admit | 867,201,024 | 2,875,994,064 | 1,601,957,888 | 1.7953 | 713,287,597 |
| offline/map-1024-s4 | unlimited | 867,291,136 | 2,876,084,176 | 1,601,900,544 | 1.7954 | 713,156,525 |
| offline/map-256-s4 | admit | 867,328,000 | 2,653,954,000 | 1,381,421,056 | 1.9212 | 494,504,969 |
| offline/map-256-s4 | unlimited | 866,861,056 | 2,653,487,056 | 1,380,241,408 | 1.9225 | 491,171,918 |
| offline/map-64-s2 | admit | 867,282,944 | 2,640,023,504 | 1,381,003,264 | 1.9117 | 494,325,553 |
| offline/map-64-s2 | unlimited | 867,221,504 | 2,639,962,064 | 1,381,548,032 | 1.9109 | 492,178,765 |
| offline/map-64-s4 | admit | 867,008,512 | 2,639,749,072 | 1,380,859,904 | 1.9117 | 492,625,869 |
| offline/map-64-s4 | unlimited | 867,319,808 | 2,640,060,368 | 1,381,527,552 | 1.9110 | 492,027,934 |
| offline/map-64-s8 | admit | 867,442,688 | 2,640,183,248 | 1,380,519,936 | 1.9125 | 492,488,450 |
| offline/map-64-s8 | unlimited | 867,192,832 | 2,639,933,392 | 1,381,670,912 | 1.9107 | 494,423,042 |
| offline/map-grid | admit | 867,135,488 | 1,349,435,151 | 1,339,273,216 | 1.0076 | 454,999,055 |
| offline/map-grid | unlimited | 867,123,200 | 1,349,422,863 | 1,337,925,632 | 1.0086 | 453,401,056 |
| offline/map-pcf1 | admit | 866,816,000 | 1,399,448,982 | 1,388,208,128 | 1.0081 | 504,254,185 |
| offline/map-pcf1 | unlimited | 867,147,776 | 1,399,780,758 | 1,386,086,400 | 1.0099 | 501,785,318 |
| offline/map-pcf4 | admit | 867,074,048 | 1,601,033,622 | 1,590,489,088 | 1.0066 | 706,274,035 |
| offline/map-pcf4 | unlimited | 867,098,624 | 1,601,058,198 | 1,590,108,160 | 1.0069 | 705,854,460 |
| ordinary/fly-128 | admit | 867,012,608 | 1,352,251,611 | 1,338,863,616 | 1.0100 | 454,563,832 |
| ordinary/fly-128 | unlimited | 867,135,488 | 1,352,374,491 | 1,340,825,600 | 1.0086 | 456,191,179 |
| ordinary/fly-256 | admit | 867,577,856 | 1,364,475,318 | 1,341,886,464 | 1.0168 | 456,587,186 |
| ordinary/fly-256 | unlimited | 867,868,672 | 1,364,766,134 | 1,339,973,632 | 1.0185 | 454,603,120 |
| ordinary/fly-64 | admit | 867,344,384 | 1,349,652,261 | 1,339,404,288 | 1.0077 | 454,908,272 |
| ordinary/fly-64 | unlimited | 867,078,144 | 1,349,386,021 | 1,338,183,680 | 1.0084 | 453,909,643 |
| ordinary/fly-64-dem2048 | admit | 883,830,784 | 2,275,432,667 | 1,438,658,560 | 1.5816 | 520,614,425 |
| ordinary/fly-64-dem2048 | unlimited | 884,072,448 | 2,275,674,331 | 1,439,408,128 | 1.5810 | 521,134,248 |
| ordinary/map-256 | admit | 867,549,184 | 1,364,408,906 | 1,338,970,112 | 1.0190 | 454,282,269 |
| ordinary/map-256 | unlimited | 868,253,696 | 1,365,113,418 | 1,338,712,064 | 1.0197 | 453,277,287 |
| ordinary/map-512 | admit | 868,290,560 | 1,411,746,378 | 1,344,802,816 | 1.0498 | 457,217,318 |
| ordinary/map-512 | unlimited | 868,134,912 | 1,411,590,730 | 1,346,256,896 | 1.0485 | 460,927,782 |
| ordinary/map-64 | admit | 867,364,864 | 1,349,663,306 | 1,338,937,344 | 1.0080 | 454,433,827 |
| ordinary/map-64 | unlimited | 867,700,736 | 1,349,999,178 | 1,336,729,600 | 1.0099 | 451,640,333 |
| ordinary/map-64-dem1024 | admit | 871,407,616 | 1,580,358,218 | 1,361,104,896 | 1.1611 | 466,551,366 |
| ordinary/map-64-dem1024 | unlimited | 871,477,248 | 1,580,427,850 | 1,364,893,696 | 1.1579 | 470,919,062 |
| ordinary/map-64-dem2048 | admit | 883,916,800 | 2,275,490,378 | 1,420,771,328 | 1.6016 | 501,764,974 |
| ordinary/map-64-dem2048 | unlimited | 883,990,528 | 2,275,564,106 | 1,422,532,608 | 1.5997 | 504,366,008 |
| stream/both | admit | 868,089,856 | 2,674,180,847 | 1,374,568,448 | 1.9455 | 464,970,775 |
| stream/both | unlimited | 867,127,296 | 2,673,218,287 | 1,374,978,048 | 1.9442 | 466,341,764 |
| stream/clip-min | admit | 867,110,912 | 2,640,157,677 | 1,364,713,472 | 1.9346 | 465,544,647 |
| stream/clip-min | unlimited | 867,180,544 | 2,640,227,309 | 1,364,357,120 | 1.9351 | 465,119,431 |
| stream/clipmap | admit | 867,459,072 | 2,646,016,237 | 1,365,225,472 | 1.9382 | 465,708,485 |
| stream/clipmap | unlimited | 867,196,928 | 2,645,754,093 | 1,364,094,976 | 1.9396 | 464,797,851 |
| stream/vt | admit | 867,065,856 | 1,376,231,500 | 1,348,628,480 | 1.0205 | 455,625,495 |
| stream/vt | unlimited | 867,233,792 | 1,376,399,436 | 1,349,017,600 | 1.0203 | 455,584,381 |

The largest sampled residual was 994,716,158 bytes on poster/large (unlimited). Its largest measured increase was 706,549,739 bytes during tile buffer and texture creation. The driver-call attribution in the earlier cold four-sample trace applies to that MapScene workload.

| Refused job | Limit B | Estimate B | OS before B | OS after B | Matched empty growth B |
|---|---:|---:|---:|---:|---:|
| map terrain 1 sample(s) | 1 | 1,349,724,746 | 867,180,544 | 867,180,544 | 0 |
| fly terrain 1 sample(s) | 1 | 1,349,358,592 | 866,795,520 | 866,795,520 | 0 |
| map terrain 4 sample(s) | 1,349,347,938 | 2,639,531,984 | 866,791,424 | 866,791,424 | 0 |
| map clipmap 1 sample(s) | 1 | 2,645,663,981 | 867,106,816 | 867,106,816 | 0 |
| map vt 1 sample(s) | 1 | 1,376,108,620 | 866,942,976 | 866,942,976 | 0 |
| poster refuse | 1,073,741,824 | 3,163,962,072 | 902,234,112 | 902,234,112 | 0 |
| poster refuse-dem | 1,146,987,452 | 1,828,560,692 | 885,755,904 | 885,755,904 | 0 |

Receipt reproduction uses `tools/measure_modus_scene.py` and `tools/measure_modus_poster.py` on the rebuilt wheel. The refusal controls and focused test logs are under `artifacts/modus-return/`. The run used `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_TEST_INSTALLED_WHEEL=1`, `WGPU_BACKEND=vulkan`, `WGPU_BACKENDS=vulkan` and `FORGE3D_DETERMINISM_TEST_BACKEND=vulkan`. Hosted CI, other platforms and backends, and SSAO remain unverified.

In the minimum-clipmap pair, all PNG bytes and all canonical certificate fields except `process_memory` matched. The actual digests began `d82a0b72` and `4de085bf`. Omitting that field for comparison gave `4ff2b81e` for both. The signed view is an owner decision. Input providers without discoverable dimensions and compressed external geometry without decoded sizes still lack finite predictions. The matrix remains local and uncommitted.

### Decoded vector features with a GeoJSON source path

The scene estimator follows the compiler's test for path-only vectors. When
the layer carries decoded features, the recipe estimate includes their
coordinates, properties, and source descriptor. Preflight does not reopen
the GeoJSON. A path-only layer with no decoded features still raises
`memory_estimate_unavailable`.

The previous wheel rejected the validated public label/vector fixture as
soon as the two decoded lines carried a source path. The corrected wheel
rendered its 96 × 64 frame on the native terrain backend. Both candidate
PNGs matched the unmodified-main PNG byte for byte (SHA-256
`e1f4a2d298e296dffbabbb8f7dd6b051a98c5fbcad18d48a4e366f8eb05f7c98`).
The control and both candidate jobs each ran in a fresh process on Windows 11,
Python 3.13.14, Forge3D 1.52.0 and an RTX 3070 with Vulkan driver 610.60.
Candidate HEAD and verified origin/main were `6d2a6f55` (0 ahead, 0 behind).
The starting checkout was `7c81644a` (91 ahead, 880 behind).

| MapScene job | Process baseline B | Prediction B | Independent OS peak at return B | Prediction / OS | Sampled residual B |
|---|---:|---:|---:|---:|---:|
| Exact-limit | 872,333,312 | 1,353,826,641 | 1,350,291,456 | 1.0026 | 464,655,433 |
| Unlimited | 872,431,616 | 1,353,924,957 | 1,350,811,648 | 1.0023 | 462,138,964 |

The process allowance for cold runtime was 481,321,005 bytes in each job.
The exact-limit job included 26,340 bytes of recipe copies. Its CPU estimate
was 24,945,096 bytes against 22,460,943 bytes of sampled growth. Host-visible
and device-local GPU estimates were 33,792 and 222,647,362 bytes, with
observations of 33,792 and 222,573,122 bytes. Both runs passed the three
ledger checks. Both the public certificate and the exported certificate
contained the same report, including the exact OS peak. Terrain renderer
construction accounted for 268,940,534 bytes of the exact-limit job's residual.

At a one-byte limit the fixture predicted 1,354,141,994 bytes and raised
`memory_admission_refused` before output. Peak private commit stayed at
872,648,704 bytes. The empty control stayed at 872,398,848 bytes.
The scoped regression is `test_decoded_vector_features_with_source_path`
in `tests/test_modus_scene_memory.py`. Fresh receipts, controls and logs are
under `artifacts/modus-vector/`. The earlier 61-job matrix is separate.

### Reproducing the MODUS matrix

In the first native run of the integrated decoded-vector fixture, the
unlimited job grew by 511,381,504 bytes. Its prediction missed the return
peak by 29,888,130 bytes even though each tracked ledger was within its
estimate. Subtracting modeled buffers from fresh renders of the fixture
gave fixed costs from 477,785,820 to 511,209,135 bytes. For scenes with
decoded vector overlays, admission now uses the upper observation plus
that measured spread, or 544,632,450 bytes. The
`runtime_vector_overlay` term accounts for the 63,311,445 bytes above
the ordinary cold-runtime allowance.

On a qualifying native GPU, the following two pytest entries run the 52
MapScene and flythrough cases and 13 poster cases in fresh processes:

```console
python -m pytest tests/test_modus_scene_memory.py::test_scene_peak_in_fresh_process tests/test_obliqua_poster.py::test_modus_poster_peak_in_fresh_process
```

The refusal gates use the same fixture generators:

```console
python -m pytest tests/test_modus_scene_memory.py::test_refused_scene_growth_within_empty_call_noise tests/test_obliqua_poster.py::test_modus_poster_refusal_matches_empty_call
```

To compare decoded-vector renders against a baseline install, run the first
command below with the baseline wheel and the next two with the candidate.
Use the same adapter and backend for both installations.

```console
python tools/measure_modus_scene.py map control artifacts/modus/control --decoded-vector --certificate
python tools/measure_modus_scene.py map admit artifacts/modus/admit --decoded-vector --certificate --reference artifacts/modus/control
python tools/measure_modus_scene.py map unlimited artifacts/modus/unlimited --decoded-vector --certificate --reference artifacts/modus/control
```

### Current qualification on Windows Vulkan

The candidate and the unmodified main controls used base `6d2a6f55`.
On Windows 11, Python 3.13.14 and the NVIDIA RTX 3070 (Vulkan 610.60), 63
renders and eight refusal controls exercised the integrated vector change.
A second release wheel added GeoTIFF header inspection. Two renders and
a matched refusal exercised that branch. The first 63 jobs use other inputs.
Both builds contained the same native extension, SHA-256
`88520732f946d6a261e48377ec80026bd429e75015191f8c5e8aca8aa35b80e0`.
Each measured job ran in a fresh process. All table sizes are bytes.
The OS column is the independent Windows peak private commit at return.

The file-backed regression used a Float64 GeoTIFF. The renderer accepted
the file before the correction, while the Pillow-only preflight could not
estimate it. The regression test compares its terrain allocation term to
the same samples held in a Float64 array and checks refusal before reading
pixels. The decoded-vector and GeoTIFF fixtures both run through the normal
MapScene entry point in the matrix.

| Job | Mode | Process baseline | Prediction | OS peak at return | Prediction / OS | Sampled residual |
|---|---|---:|---:|---:|---:|---:|
| fly 4096 atlas | admit | 867,389,440 | 2,725,602,431 | 1,512,570,880 | 1.80197 | 603,635,202 |
| fly 4096 atlas | unlimited | 867,856,384 | 2,726,069,375 | 1,511,124,992 | 1.80400 | 601,733,041 |
| fly 128 | admit | 867,438,592 | 1,352,677,595 | 1,338,744,832 | 1.01041 | 453,805,284 |
| fly 128 | unlimited | 867,110,912 | 1,352,349,915 | 1,340,432,384 | 1.00889 | 455,884,933 |
| fly 256 | admit | 868,220,928 | 1,365,118,390 | 1,346,162,688 | 1.01408 | 460,479,980 |
| fly 256 | unlimited | 867,799,040 | 1,364,696,502 | 1,338,544,128 | 1.01954 | 453,295,502 |
| fly 64 | admit | 866,779,136 | 1,349,087,013 | 1,341,845,504 | 1.00540 | 457,869,260 |
| fly 64 | unlimited | 867,512,320 | 1,349,820,197 | 1,339,375,616 | 1.00780 | 454,478,644 |
| fly 64-dem2048 | admit | 884,154,368 | 2,275,756,251 | 1,439,051,776 | 1.58143 | 520,671,400 |
| fly 64-dem2048 | unlimited | 884,027,392 | 2,275,629,275 | 1,439,064,064 | 1.58133 | 520,827,048 |
| fly 64-s4 | admit | 867,233,792 | 2,639,993,222 | 1,387,565,056 | 1.90261 | 502,885,525 |
| fly 64-s4 | unlimited | 867,356,672 | 2,640,116,102 | 1,383,174,144 | 1.90874 | 490,006,374 |
| map 4096 atlas | admit | 866,979,840 | 1,424,900,436 | 1,418,874,880 | 1.00425 | 525,799,765 |
| map 4096 atlas | unlimited | 867,037,184 | 1,424,957,780 | 1,417,498,624 | 1.00526 | 524,505,440 |
| map Float64 GeoTIFF | admit | 892,129,280 | 1,374,460,232 | 1,364,094,976 | 1.00760 | 454,812,762 |
| map Float64 GeoTIFF | unlimited | 892,133,376 | 1,374,464,340 | 1,363,701,760 | 1.00789 | 454,417,242 |
| map both | admit | 867,094,528 | 2,673,185,519 | 1,377,267,712 | 1.94093 | 468,701,727 |
| map both | unlimited | 867,016,704 | 2,673,107,695 | 1,376,657,408 | 1.94174 | 468,161,456 |
| map clipmap | admit | 867,532,800 | 2,646,089,965 | 1,300,226,048 | 2.03510 | 400,617,402 |
| map clipmap | unlimited | 867,201,024 | 2,645,758,189 | 1,364,455,424 | 1.93906 | 465,182,810 |
| map decoded vector | admit | 873,230,336 | 1,418,035,599 | 1,352,622,080 | 1.04836 | 464,022,692 |
| map decoded vector | unlimited | 872,734,720 | 1,417,539,989 | 1,357,303,808 | 1.04438 | 462,470,648 |
| map 1024-s4 | admit | 867,438,592 | 2,876,231,632 | 2,723,725,312 | 1.05599 | 1,833,242,538 |
| map 1024-s4 | unlimited | 867,569,664 | 2,876,362,704 | 1,605,087,232 | 1.79203 | 716,064,941 |
| map 256 | admit | 867,745,792 | 1,364,605,514 | 1,339,744,256 | 1.01856 | 454,845,734 |
| map 256 | unlimited | 868,122,624 | 1,364,982,346 | 1,345,474,560 | 1.01450 | 460,197,638 |
| map 256-s4 | admit | 867,414,016 | 2,654,040,016 | 1,383,563,264 | 1.91826 | 497,035,125 |
| map 256-s4 | unlimited | 867,196,928 | 2,653,822,928 | 1,381,834,752 | 1.92051 | 495,670,533 |
| map 512 | admit | 868,294,656 | 1,411,750,474 | 1,345,851,392 | 1.04896 | 458,199,558 |
| map 512 | unlimited | 868,442,112 | 1,411,897,930 | 1,345,679,360 | 1.04921 | 460,038,950 |
| map 64 | admit | 867,315,712 | 1,349,614,154 | 1,339,957,248 | 1.00721 | 455,488,806 |
| map 64 | unlimited | 867,270,656 | 1,349,569,098 | 1,342,894,080 | 1.00497 | 458,469,126 |
| map 64-dem1024 | admit | 871,784,448 | 1,580,735,050 | 1,364,140,032 | 1.15878 | 469,839,341 |
| map 64-dem1024 | unlimited | 871,882,752 | 1,580,833,354 | 1,296,052,224 | 1.21973 | 398,540,880 |
| map 64-dem2048 | admit | 883,826,688 | 2,275,400,266 | 1,421,672,448 | 1.60051 | 503,666,236 |
| map 64-dem2048 | unlimited | 884,084,736 | 2,275,658,314 | 1,422,008,320 | 1.60031 | 503,725,566 |
| map 64-s2 | admit | 867,528,704 | 2,640,269,264 | 1,382,346,752 | 1.90999 | 495,937,287 |
| map 64-s2 | unlimited | 867,639,296 | 2,640,379,856 | 1,382,940,672 | 1.90925 | 493,125,569 |
| map 64-s4 | admit | 867,307,520 | 2,640,048,080 | 1,381,933,056 | 1.91040 | 492,387,066 |
| map 64-s4 | unlimited | 867,082,240 | 2,639,822,800 | 1,380,016,128 | 1.91289 | 492,840,762 |
| map 64-s8 | admit | 867,422,208 | 2,640,162,768 | 1,381,625,856 | 1.91091 | 492,509,225 |
| map 64-s8 | unlimited | 867,405,824 | 2,640,146,384 | 1,382,215,680 | 1.91008 | 492,557,492 |
| map grid | admit | 867,201,024 | 1,349,500,687 | 1,344,229,376 | 1.00392 | 459,874,054 |
| map grid | unlimited | 867,258,368 | 1,349,558,031 | 1,337,303,040 | 1.00916 | 452,864,103 |
| map pcf1 | admit | 867,684,352 | 1,400,317,334 | 1,395,367,936 | 1.00355 | 510,529,510 |
| map pcf1 | unlimited | 867,463,168 | 1,400,096,150 | 1,388,883,968 | 1.00807 | 504,268,294 |
| map pcf4 | admit | 867,213,312 | 1,601,172,886 | 1,596,784,640 | 1.00275 | 712,414,556 |
| map pcf4 | unlimited | 867,422,208 | 1,601,381,782 | 1,591,099,392 | 1.00646 | 706,521,596 |
| map minimum clipmap | admit | 867,188,736 | 2,640,235,501 | 1,365,741,568 | 1.93319 | 466,460,602 |
| map minimum clipmap | unlimited | 867,762,176 | 2,640,808,941 | 1,364,054,016 | 1.93600 | 464,135,505 |
| map vt | admit | 867,483,648 | 1,376,649,292 | 1,354,129,408 | 1.01663 | 463,597,538 |
| map vt | unlimited | 867,147,776 | 1,376,313,420 | 1,349,627,904 | 1.01977 | 456,446,859 |
| poster calibration | unlimited | 869,052,416 | 1,099,530,036 | 1,059,536,896 | 1.03775 | 188,467,837 |
| poster compact | unlimited | 868,835,328 | 1,094,109,208 | 1,051,652,096 | 1.04037 | 180,970,330 |
| poster dem-1024 | unlimited | 872,787,968 | 1,274,527,540 | 1,146,679,296 | 1.11149 | 256,652,235 |
| poster dem-2048 | admit | 885,587,968 | 1,828,392,756 | 1,451,507,712 | 1.25965 | 502,518,137 |
| poster dem-2048 | unlimited | 885,133,312 | 1,827,938,100 | 1,451,388,928 | 1.25944 | 502,866,892 |
| poster dem-2048-many | unlimited | 885,268,480 | 1,828,097,748 | 1,588,932,608 | 1.15052 | 640,066,033 |
| poster dem-2048-nodata | unlimited | 885,592,064 | 1,828,396,852 | 1,654,431,744 | 1.10515 | 670,344,602 |
| poster dem-rect | unlimited | 874,676,224 | 1,409,453,924 | 1,210,429,440 | 1.16442 | 305,625,347 |
| poster large | unlimited | 868,675,584 | 2,063,328,208 | 1,951,014,912 | 1.05757 | 994,658,814 |
| poster masked | unlimited | 869,253,120 | 1,339,428,772 | 1,211,981,824 | 1.10516 | 318,912,497 |
| poster material | unlimited | 869,777,408 | 1,164,751,712 | 1,071,489,024 | 1.08704 | 192,413,774 |
| poster plain | unlimited | 868,962,304 | 1,338,990,500 | 1,211,809,792 | 1.10495 | 319,429,231 |
| poster resume | unlimited | 868,691,968 | 1,338,870,332 | 1,146,798,080 | 1.16749 | 254,588,930 |

#### Refusal versus empty call

| Job | Limit | Prediction | Before | After | Empty-call growth |
|---|---:|---:|---:|---:|---:|
| fly base | 1 | 1,350,087,680 | 867,524,608 | 867,524,608 | 0 |
| map 4 samples | 1,349,347,938 | 2,640,424,912 | 867,684,352 | 867,684,352 | 0 |
| map Float64 GeoTIFF | 1 | 1,374,288,446 | 891,711,488 | 891,711,488 | 0 |
| map base | 1 | 1,349,872,202 | 867,328,000 | 867,328,000 | 0 |
| map clipmap | 1 | 2,645,553,389 | 866,996,224 | 866,996,224 | 0 |
| map decoded vector | 1 | 1,418,383,807 | 873,578,496 | 873,578,496 | 0 |
| map vt | 1 | 1,376,505,932 | 867,340,288 | 867,340,288 | 0 |
| poster refuse | 1,073,741,824 | 3,163,925,208 | 902,197,248 | 902,197,248 | 0 |
| poster refuse-dem | 1,146,987,452 | 1,828,011,828 | 885,207,040 | 885,207,040 | 0 |

#### Residual and default output comparison

The largest sampled residual was 1,833,242,538 bytes for map 1024-s4 (admit). Its largest measured phase increase was 1,566,249,576 bytes during render setup and publication. The phase locates the growth. The earlier cold four-sample MapScene trace separately attributed host memory to graphics-pipeline creation in the driver.

The comparison matched 125 files byte for byte (83 PNGs, 30 provenance sidecars, 12 manifests). Both decoded-vector modes matched the baseline certificate digest and verified with the baseline public key. 40 requested certificate reports matched their per-job telemetry.

All 65 job reports matched their independent return peaks. The three ledger estimates and the whole-process predictions covered every observed peak. All 9 refusals stayed within their matched empty calls.
