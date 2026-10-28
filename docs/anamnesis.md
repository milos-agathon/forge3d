# ANAMNESIS content-addressed renderer

`forge3d.anamnesis` is the deterministic offline build system for frame
sequences. `render_sequence(recipe, cache=...)` declares a Merkle pass graph and
keys every output from:

- the stable pass label;
- the complete pipeline/copy descriptor bytes;
- the exact uploaded uniform bytes, including initialized padding;
- role-separated input keys such as `heightmap@0` and `water_mask@1`;
- negotiated features, codegen limits, backend, DX12 compiler policy, and Naga
  capability policy;
- crate version, full Git SHA, locked Naga version, and the preprocessed WGSL
  tree identity.

Frame output requires a `render_frame` callback or `pass_executors`. Without either,
execution raises an explicit unavailable error. `reference_only=True` (CLI
`--reference-only`) explicitly selects digest-only reference output; it does not
render images. `dry_run=True` only plans execution.

Unknown opaque-renderer state is never guessed. A `render_frame` callback must
provide both `render_frame_fingerprint` and `render_frame_context`, and its
output key includes the complete pixel-affecting recipe projection. Structured
`pass_executors` must provide a fingerprint and captured-input context for
every executor; this is the supported boundary for exact pass-level
invalidation.

Explicit external capabilities are retained without probing the local GPU,
including their backend identity. If no capabilities are supplied and the
native probe fails or returns an incomplete report, the sequence recomputes
without opening the store. An unknown device cannot reuse another unknown
device's outputs. `bytes_written` counts successful store writes.

The prediction is independent of execution. ANAMNESIS first compiles every key,
then compares the resulting manifest with the prior manifest before reading the
store or invoking an encoder. Changed keys are forced to execute; unchanged
keys must restore. A missing unchanged entry is reported as a prediction
mismatch instead of quietly redefining the prediction from the cache lookup.

`cache=None` performs no cache I/O and follows the original render path. A
cache path never permits a hit from incomplete material: unsupported native
terrain state conservatively recomputes.

## Store and diagnostics

The Rust and Python implementations use the same
`forge3d.anamnesis.store/2` metadata and LRU clock. Each store has one
`data.v2.bin` payload file and one `index.v2.json` index, whose schema is
`forge3d.anamnesis.packed-index/2`. Version-1 directories and manifests are
ignored as misses, with no migration or serving of their bytes. The index
retains each payload's exact offset, length, self-hash and complete derivation.
A fresh process reads the pair once, then independently verifies every entry
it restores. A changed pair signature reloads the snapshot; verified protected
reads remain scoped to their run. GC invalidates the manifest/index before
relocating bytes and publishes the replacement index after the payload. A torn
index is a miss. Reservations cover the payload and index growth before writes.
 Entries retain the canonical
key material needed to reconstruct the key. Payload, index, directories,
control files, and quarantine all count toward the hard
`max_bytes` on-disk bound. Corrupt data is quarantined and its build manifest
is invalidated before the next render.

An oversized store is admitted by eviction before any write. Its starting
excess is reported separately as `admission_cleanup`; admission raises if the
remaining control/directory footprint cannot fit. Rust also exposes the
cumulative starting excess through `ContentStore.admission_cleanup_bytes()`.
After admission, payloads, metadata, directory/index growth, manifests and
access logs reserve space before creation or growth. The bound covers staging
and publication, rather than only the final tree after eviction.

Warm sequence builds keep a bounded, signature-checked in-process L1 for stores
they just populated. A changed data or index signature reloads the packed
snapshot and routes entries through normal integrity/quarantine checks. Unchanged
ancestors committed by a restored descendant key are elided; only terminal
outputs and the direct inputs of actual misses are materialized. LRU touches
are appended once per sequence to the shared `access.log` journal, which both
the Python and Rust garbage collectors consult. This avoids thousands of
per-hit metadata rewrites without weakening cross-language eviction order.

The store defaults to `.forge3d/cache`. Use:

```text
python -m forge3d.anamnesis explain <key>
python -m forge3d.anamnesis verify
python -m forge3d.anamnesis gc <max_bytes>
python -m forge3d.anamnesis render-sequence recipe.json --frames 600 --reference-only
```

`explain` reconstructs and checks each pass key and recursively follows its
named inputs. `verify` uses the native store reader when the extension is
present; bidirectional interoperability is contract-tested.

Eviction and quarantine invalidate `last_manifest.json` in both store
implementations. Python sequences publish only the portion of the plan
whose entries remain resident after reservation and eviction. The next render
predicts missing work instead of repeatedly promising unavailable restoration.

## Native framegraph restoration

Offline renderers construct graphs only through `RendererGraphBuilder`.
Diagnostics reads the last compiled production report and does not create a
parallel synthetic graph. Each compiled pass owns its actual resource
descriptions, declaration bytes, deterministic order, and transition plan.

`GraphScheduler` walks that compiled graph bottom-up. A hit invokes the
resource-restoration callback under the graph's transition contract; a miss
executes the encoder and stores its output. The shared offscreen forward path
contains the production proof: cached tightly packed HDR bytes are uploaded
into the actual tracked color texture, and the declared readback consumer then
runs with the same compiled color transition as a cold producer. Borrowed wgpu
pipelines are cacheable only when their owner supplies a
`ForwardCacheDeclaration` containing the reconstructible pipeline, uniform,
external-resource, capability, and engine bytes.

The terrain draw and AOV phases execute through the compiled graph. One-shot
beauty caching records the actual pass keys in
`TerrainRenderer.last_anamnesis_cache_report["pass_keys"]`. Color snapshots
preserve both RGBA8 and RGBA16F texels, including their different row strides;
legacy RGBA8 blobs remain readable. The updated declaration revision prevents
reuse of the former RGBA8-only encoding for an HDR texture.

Material virtual textures (active layers or registered sources), moment
shadows, multi-attachment AOV, mutable scatter, viewer heightmaps and streaming
conservatively render fresh without opening the cache. The last report adds
`bypass_reason: str | None`: `material_virtual_textures`, `moment_shadows`,
`render_with_aov`, `scatter`, `viewer_heightmap`, `height_streaming`, or
`certificate`. A poisoned VT-state lock reports `material_vt_state_unavailable`.
`None` means the request was eligible or no cache was requested. Reasons reset
on every one-shot call. The interactive viewer is explicitly out of scope.

The four material-map files and detail-normal file are identified by content,
including edits at the same path. The height-curve LUT is keyed by its
normalized float32 samples.

The native shadow map is keyed but never stored or served. Locked wgpu has no
`Queue::write_texture` path into Depth32Float, so `terrain.shadow` is a
cache-disabled pass materialized on demand: when `terrain.forward` executes,
the shadow pass is encoded freshly immediately before it and reported as a
miss; when the forward output is restored, the shadow pass is elided and
reported as a hit, because the restored forward key commits to the shadow key.
An all-hit render therefore submits no GPU work. Without a cache the pass runs
in graph order exactly as before. Declaration revision v3 retires every entry
written while shadow depth was uploaded through the rejected path.

## Portability and acceptance

The required physical acceptance lane renders 600 real TerrainRenderer frames,
changes one visible label, and requires:

- 600/600 byte-identical outputs against a cold modified render;
- an exact three-pass prediction/observation set (`label.compile`,
  `label.composite`, and `frame.output`);
- zero incremental native-terrain encodes;
- at least 20× cold-to-incremental wall-clock speedup (12× for the
  fresh-process gate below).

The original and cold modified sequences each invoke the native renderer from
their structured sequence executor, using separate empty native stores. The
speedup compares the complete incremental and cold modified sequence calls.
Native resource restoration is checked separately and is excluded from those
timings. The cold modified comparison cannot reuse the original frame array.

The opt-in native portable profile currently retains the complete backend,
compiler, negotiated-limit and capability fingerprint. It does not authorize
cross-backend serving: physical Vulkan/DX12 output still differs. Existing
profile entries that omitted that identity become misses. The cross-backend
lane therefore remains unqualified; its hit threshold and fresh/golden checks
are unchanged. No qualification-record mechanism is implemented yet.

The compatibility lane seeds the production TERRA graph on physical Vulkan
and restores it in a separate consumer process with the same full compatibility
profile on this machine. It requires at least 99% hits, exact equality with an
independent fresh render, and the committed TERRA golden. Machine IDs remain
diagnostic; a second machine is outside node 17's acceptance contract.
The driver uses the actual device profile without an environment override.
Both compatibility processes use `FORGE3D_DETERMINISTIC=1` before GPU
initialization so their native frames match the protected canonical TERRA
golden. This selects the real canonical renderer policy; it does not replace
the compatibility fingerprint. The ordinary 600-frame timing gates retain
`FORGE3D_DETERMINISTIC=0`.
Mismatch controls reconstruct the seeded production keys and require zero native
store hits when backend, features, limits, compiler or Naga policy changes.
The backend remains in the key. Cross-backend reuse is outside this PR.

The reference-executor 500/10,000 mutation tests establish key propagation in
their synthetic corpus. The separate physical native mutation gate walks the
terrain configuration's floating-point fields, finite enum/bool choices and
integer boundaries/seeded byte changes, plus source height, water-mask, HDR,
material and time bytes. It also changes one decoded pixel byte in each of the
four material-map files and the detail-normal file, then writes a valid PNG at
the same path. Encoding checksums change along with the texel; malformed PNG
bytes are not claimed as valid production mutations. It checks the entire
canonical parameter declaration and the negotiated capability declaration,
then compares every restored frame byte-for-byte with an independent fresh
render. Each mutation starts with a copy of the original native store, so
an incorrectly unchanged intermediate key cannot hide behind an empty store.
The float corpus changes the low three IEEE-754 bytes; values rejected or
clamped back to the original by the public constructor are not mutations.
The seed remains `0xA11A_17` and the full eligible corpus remains 10,000
cases. Seeded candidate orders cycle to cover every accepted finite choice
and recorded integer alternative; constructor rejections are recorded with
reasons. Texture dimensions obey the physical device limit. Moment-shadow
alternatives run as additional fresh-only controls, never as fake all-hit
cases. Every eligible case still requires a changed final key, exact
cached/restored/fresh equality, no GPU errors and zero warm submissions. A
sun-direction edit after an all-hit frame must visibly change pixels and
freshly materialize the shadow before forward shading. Temporary stores keep
memory and disk use independent of case count.

On 2026-10-21 Milos approved finite enum/bool coverage plus integer boundaries
and seeded byte mutations: unbounded integer domains cannot be exhaustively
enumerated in 10,000 cases. Coverage and rejected values are written in the
native result. `tests/anamnesis_irrelevant_inputs.toml` enumerates exactly the
six label-only frozen-frame fields removed from the terrain projection. A
ledger equality test and per-field physical terrain comparisons enforce the
exclusions; complete validated frames still reach native rendering unchanged.

Milos approved this native rule on 2026-10-20: mutations must change the
relevant keys, and cached pixels must equal fresh pixels; separate controls
must visibly change the image. A small float change or an occluded texel can
legitimately leave quantized pixels unchanged. The reference corpus alone
does not establish production hermeticity, and a local GPU run does not
establish cross-machine portability.

Material identity records the exact uploaded mip bytes, resolved dimensions,
texture/view descriptions and sampler. Editing a source file leaves a resident
MaterialSet's identity unchanged; a new MaterialSet keys its new upload.
Memory-driven resizing therefore keys the resource the renderer actually uses.
The physical stale-hit regression also compares equivalent resized uploads.

Without explicit executors, `render_sequence` recognizes MapScene, SceneRecipe,
serialized scene recipes and mappings of frame indices to MapScene objects.
Its production adapter freezes native resources and compiled label decisions,
then executes terrain rendering, label composition and PNG encoding as separate
passes. A label-only edit reuses terrain and recomputes composition only where
the label is visible. Empty accepted-label sets schedule no label work. Cached
label composition reuses native resources, clears glyphs before each run, and
recreates the compositor after a size change or failure. CHRONOS uses this adapter while retaining its frozen frame
seeds, morphs and residency. Existing compiled-frame and bundle formats do not
change. The private native resource probe has no content-store side effects.

Cacheable scenes currently use unstyled terrain, label layers and single-sample
8-bit PNG output without furniture, custom material maps, virtual textures,
external water masks, custom render passes, AOVs or denoising. Recognized scenes
outside that surface render freshly through MapScene. Arbitrary reference DAG
mappings remain available for explicit propagation tests; they are not physical
render evidence. Native validation errors prevent storing invalid scene pixels.

Path-backed GeoTIFF terrain whose GDAL source list includes companion files
(such as `.tfw` or `.aux.xml`) renders freshly with
`scene_terrain_external_metadata`. A single-file pinned snapshot cannot retain
the complete grid, nodata or mask supplied by those files. GeoTIFFs with embedded
metadata remain cacheable.

Deterministic RGBA16F rendering source-isolates the optional ORBIS group-7 probe
from ordinary seven-group terrain modules. Reflection fallback uploads use the
actual color texel size. The HDR regression requires visible finite output,
a visible camera control, exact floating-point cold/cache equality and no GPU
validation errors, in addition to eight-byte cache transport.

The required physical CI jobs retain the original native 600-frame gate and
separately run the public adapter's 600-frame/20× gate, native 10,000-mutation
gate, material regression and unsupported-output controls. Each job checks its
exact candidate and uploads its evidence; their presence in CI does not prove
a passing local run. Local validation does not establish cross-machine credit.


The public MapScene 64×64/600-frame acceptance times its independent cold
modified sequence with `cache=None`. It retains the exact three-pass label
recompute set, 600/600 byte equality, zero incremental native terrain calls and
the unchanged 20× wall-clock gate. Public sequence reuse does not independently
prove native intermediate-resource restoration; both physical gates run.

The same-process public gate is retained. A separate gate runs the original,
label edit and `cache=None` cold phases in three new subprocesses, using
`SequenceResult.elapsed_seconds` and a 12× requirement. Each fresh
incremental process commits 600 independent snapshot copies and loads the
packed JSON index, which held measured runs at 13.1–14.3× on an NVIDIA
RTX 3070; the 20× target returns with the binary index in 1.47. Its receipt
also records each subprocess's wall time (including the original phase) and
the store age at the incremental phase's first read. One packed snapshot
replaces the former per-entry file opens. Required entries receive full
self-hash/metadata checks only after prediction freezes; quarantine and LRU
changes remain on the calling thread. Refresh/filter costs stay inside the
unchanged measured sequence execution.


Native `TerrainRenderParams` keeps its public live `python_object` reference.
Cache identity also captures an immutable digest at construction, so changing
that Python config cannot make older decoded native settings alias a newly
constructed parameter object. Large numeric LUT/water arrays are hashed directly.
Unsupported custom input values conservatively render fresh and report
`parameter_identity_unavailable` in `last_anamnesis_cache_report.bypass_reason`.


The mutation corpus uses the public constructors and existing native render
contracts to determine valid inputs. `lod.level` is signed in native decoding;
its uniform contract permits 0–65536. The POM refinement uniform permits 0–128
(`src/terrain/renderer/runtime_contract.rs`). Boundary/byte candidates outside
those production ranges are recorded as rejected and do not count toward the
10,000 successful comparisons. No production guard is changed. Procedural
water is the baseline so screen-camera alternatives are valid; physical water
is separately mutated while the camera remains mesh. The seed stays
`0xA11A_17`, with exact field/value coverage recorded in the result and trace.


The native corpus inventories the active dataclass tree, including optional
objects, numeric tuples and arrays. Optional values are activated and their
return to `None` is tested. Membership choices are read from the corresponding
dataclass validators; documented camera modes retain an explicit domain because
that Python field has no membership validator. Unknown leaves fail coverage.
Opaque atmosphere handles and packed VT stores receive explicit physical
fresh/bypass controls instead of eligible-cache credit. The trace records the
selected alternative or exact byte offset and XOR, and the report lists every
field classification and constructor rejection. `changed_output_cases` counts
cases whose output differs; `sun_changed_pixels` counts visible changed pixels.

The committed TERRA hash test also rejects all-black output, GPU validation or
uncaptured errors, software fallback and a different backend. Equal hashes of
invalid frames cannot establish deterministic rendering. A source change or a
single successful timing attempt does not establish a passing final native
validation; local evidence must identify the exact build and every attempt.


Before packed schema 2, the directory store's Windows dispatch measurement
covered every worker count from 1 through the standard library default of 28:
five workers took 19.8 ms on first use and 20.7 ms when reused, versus
26.0/23.1 ms for 28. That is historical directory-format evidence, not a packed
store acceptance result. Current acceptance retains the same workload, cold
path, complete verification and threshold (20× same-process, 12×
fresh-process); every attempt must identify its
native binary, source and host load.



The corpus's material centres match the four uploaded `terrain_default()`
layers, and its optional water grids match `canonical_heightmap().shape`.
Standalone parameter construction cannot validate these cross-resource
conditions; the physical preflight checks the actual render boundary.

The cached scene adapter uses the private native-only
`_forge3d._anamnesis_snapshot_recipes` helper for owned recipe snapshots.
Each frame has an independent deepcopy memo; aliases and cycles within a
recipe are preserved. Exact built-in containers and plain recipe classes
are copied natively after a callback-free graph preflight. Built-in non-object
arrays retain NumPy's deepcopy behavior. If a subclass, object array, custom
copy/reduction hook or modified copy dispatcher may run Python, the entire
batch uses Python's per-recipe `copy.deepcopy` from the beginning. A callback
can change later fields, classes or reducers, so partially copying a batch
before delegating is unsafe. Aliases, cycles, errors and callback order retain
the Python behavior. This helper is not a public package export and does not
change serialized recipes.

Resident sampling records provenance requests in five invocation-private
vectors: the existing three albedo projections plus normal and mask, each
with the existing four material components. This stores the actual resolved
keys, including POM-adjusted coordinates, without atomics during shading.
Forward, AOV and visibility entry points publish them after normal derivatives;
uncovered emulated visibility lanes discard without publishing. The bitset's
unique count, overflow checks, ordinary feedback stream and specular AA remain
active. No extra GPU resources, bindings or serialized formats are introduced.

The scene adapter keeps no self-reference or owned bound-callback cycle.
The active scheduler owns its callbacks; releasing them releases otherwise
acyclic frame snapshots without waiting for GC during a later sequence.
User-provided cyclic data follows Python's normal collection rules. This
ownership rule applies equally to cached and independent fresh runs.

Sequence results expose `bypass_reason: str | None`. A requested cache that is
conservatively disabled reports its unsupported scene feature, missing native
resource identity, or unavailable capability fingerprint. An intentional
`cache=None` call has no bypass reason. During sequence execution, GC preserves
verified bytes of entries still needed by that run before evicting their disk
files; this keeps the complete disk budget enforced without invalidating the
already-frozen recompute prediction. Scoped preservation ends on failure too.
After eviction, the saved manifest lists only resident entries, so later calls
can reuse the remaining work. A store reloads its packed index/data snapshot
on open and whenever pooled reuse observes a changed pair signature, before
reading the previous manifest. This includes external eviction and public
native `gc`/`verify` even when the root inode is unchanged. There are two file
signature queries, without per-entry filesystem calls. In-run publication,
eviction, quarantine and missing reads maintain the inventory; manifest
filtering makes no filesystem queries. Protected in-memory reads never restore
disk residency. Fresh-process acceptance records refresh/filter cost inside
the unchanged elapsed timer.
Eviction removes keys outside the current plan first. Payload and control writes reserve disk space before staging their bytes.

`MapScene.render(cache=...)` uses the same production adapter as a scene
sequence, including owned height values, pinned sky/IBL bytes and uploaded
resource identity. PNG and NPY raster sources are snapshotted and retained in
the terrain input key; raster layers render in the terrain pass while labels
remain separate. Frozen raster frames keep their complete frame state keyed.
Raster formats with untracked external metadata/sidecars conservatively render
fresh with `scene_raster_external_metadata`; their ordinary rendering support
is unchanged. Cached publication uses the ordinary native metadata path.
The DEM alignment grid is resolved once into the owned terrain snapshot and
the cache key, including file-backed transforms used to position PNG overlays.

Direct MapScene requests for certificates, provenance/signing, non-PNG,
16-bit, AOV or HDR output render fresh and expose `bypass_reason` in
`last_render_metadata`. Other unsupported scene features use the sequence's
reason in the same metadata. CHRONOS surfaces bypass reasons as
`RuntimeWarning`, including certificate requests. Public render helpers whose
return type has no cache metadata issue the same warning when their `cache`
argument is unsupported; `cache=None` stays quiet.

Native terrain snapshots each active material/detail file once for both its
key and upload. Inactive detail maps (disabled or zero strength) are not opened.
Packed-store signature checks cover data/index size and modification time;
missing or changed files retain normal integrity handling. An unreadable or
missing index is quarantined before new writes, so stale offsets cannot be
published against appended orphan bytes.

Callback-free blank scene batches reuse invariant validation and serialization
within that batch. A private type-sensitive native digest compares every
non-camera input; camera state, height bytes and the live render policy remain
checked for each frame. Custom copy/serialization hooks and changed inputs use
the ordinary path. Terrain render recipes are created only when needed by a
miss. The built-in default environment retains its owned file and resource
fingerprint per thread. Reuse checks the environment file bytes, the material
directory setting, working directory and all four default albedo files' contents;
changed sources rebuild the resource fingerprint. An active-batch lease keeps
the owned inputs alive. Custom environment files are still snapshotted per batch.
Eligible PNG output uses the ordinary PNG encoder directly in memory for both
cached and independent fresh rendering.
Plain blank batches also defer ordinary scene construction until a miss needs
it and freeze height arrays already owned by the snapshot copier. Custom scene
constructors and other callback-bearing paths retain ordinary construction and
the separate native-height copy.

The existing in-memory store pool retains key calculation results for its last
successful terrain/output scene plan. Before reusing uniform bytes or a pass key, a native
type-sensitive digest compares the complete current owned uniform tree, and the
lookup also includes the exact pipeline, input bindings/keys, capabilities,
engine and encoder/key functions. Unknown values use the ordinary path. This
retains no scene adapter, mutable render state or GPU resources, and replaces
the previous plan rather than accumulating a history. Input snapshots,
validation, recompute prediction and blob/metadata verification still run.

VectorScene rendering, fused splat rendering (including sequence and reference), and the native Scene branch of `helpers.offscreen.render_offscreen_rgba` render fresh and emit `RuntimeWarning` with `render_cache_unsupported` when `cache` is non-None. `cache=None` is quiet; signatures and return types are unchanged.

Packed-store incremental writes append only the new index record; compaction
still rewrites a complete index. The full payload and index growth are
reserved before either write. Verified prefetched bytes become protected
reads for the current run, and still update the shared LRU journal.

Native sequence preparation overlaps ordinary device initialization with
scene snapshotting and a read-only packed-file preload. Both submission and
joining remain inside `SequenceResult.elapsed_seconds`. The preload cannot
perform admission, eviction, quarantine, or entry verification. Store opening
checks the two packed-file signatures again; each required entry is verified
only after the independent recompute prediction is frozen. Deferred resource
identities resolve before pass declarations and never enter a persisted key
as a placeholder. `cache=None` does not preload any store.

Callback-free native sequences also prepare canonical uniform fragments and
semantic pass descriptions during device initialization. The resource field
is inserted structurally after its actual identity resolves; nested user
fields with the same spelling cannot be substituted. Merkle keys still use
the resolved device profile, and the independent prediction freezes before
entry verification or pass execution. Packed payload appends grow an owned
buffer; restored values remain immutable owned bytes across later writes
and eviction. No garbage-collection policy or timer boundary is changed.

Completed resource futures use the ordinary warm declaration path; they do
not repeat preparation intended to overlap an initializing device. The
in-process key memo compares complete canonical uniform bytes, pipeline
bytes, inputs and profile context. It reuses only trusted key calculations,
never pixel verification or a store admission decision. Existing packed
payload segments are immutable and are not copied when new entries append.

The optional native packed snapshot parses metadata alongside ordinary
preparation and borrows the immutable payload bytes. Required entries are
then verified in one batch after prediction freezes, using the same Rust
validator as the native store. It returns only self-hash- and key-verified
bytes. Bad reads retain the quarantine path; parsing never serves an entry.
The pooled store releases the additional parsed tree after required reads
are protected. A required label compositor can initialize alongside that
verification. This happens only after a label miss is predicted, and its
work joins before the timed sequence returns.


For an already formatted packed store, the built-in scene driver groups its
actual incremental pass writes. Every entry is self-hash/metadata verified,
and the complete data/index growth must fit the existing reservation before
either file opens for writing. It opens the file pair once, closes data before
publishing index records, then records residency and the final manifest. The
work remains inside `SequenceResult.elapsed_seconds`. Capacity pressure,
existing keys and non-appendable indexes retain the ordinary per-pass path,
including LRU and protected reads. New stores, reference execution and external
callbacks retain their existing write order. No GPU profile or timer changes.
