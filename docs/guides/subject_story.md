# Subject-led terrain reveals

`forge3d.compile_subject_shot` resolves a source feature ID and supplies its
terrain-local target to an existing `TerrainOrbitRig`. Set the reveal
duration on the rig and the hold, frame rate, output size, framing, and
reserved text rectangle on the shot. The [Grindelwald example](../../examples/c07_grindelwald_eiger.py)
shows both variants. Its captions use `{subject}` to resolve the display name.

Use a finite 512 × 512 north-up NPY height field with metre elevations.
Supply a GDAL corner geotransform in `TerrainSource.metadata["geotransform"]`
and pass the bounds of the outer pixel centres. The terrain and feature CRS
must be the same projected metre CRS. MapScene uses its native north-up
camera mode. The compiler checks the decoded native camera against the
reflected geographic mesh with its northwest–southeast triangle diagonal.
If `TerrainSource.style` is supplied, it must produce a complete finite height
field with `height_units="m"`. The compiler checks the same styled heights
MapScene renders. Register obstacle elevations in the resulting metre datum.

Use a point for a named summit, a `line_xy` polyline for a trail, or
`area_triangles_xy` for a triangulated area. Each vertex is an easting and
northing in the registered CRS. The feature's easting and northing remain
its explicit editorial orbit anchor. An area is the union of its triangles,
so supply a triangulation that preserves the boundary and any holes.

```python
from forge3d import StoryObstacle, SubjectFeature

trail = SubjectFeature(
    "trail-17", "Ridge trail", 2644000, 1164000,
    line_xy=((2643980, 1163960), (2644000, 1164000), (2644020, 1164040)),
)
building = StoryObstacle(
    "building-83", "building",
    (2644030, 1164000, 1284, 2644050, 1164012, 1298),
)
# Include trail in features and select it with subject_id or must_see.
# Pass obstacles=(building,) to compile_subject_shot.
```

The compiler drapes the entire extent over the native mesh triangles. It
checks framing, near/far clipping, the reserved text rectangle, and
occlusion for each resulting segment or surface piece at every required
output frame. A hidden line interior or area interior fails even when all
source vertices are visible. These checks describe geometric visibility;
subpixel features and label rendering have separate visual acceptance.

Supply buildings and vegetation as registered opaque, axis-aligned solid
envelopes. The six coordinates are minimum easting, northing and elevation
followed by their maxima, in the terrain's projected metre CRS and vertical
datum. Exaggeration applies to obstacle elevation just as it does to the
height field. Use multiple boxes for finer envelopes. A box can represent
a roof or an opaque canopy. Transparent foliage, openings, and detailed
asset meshes are outside this domain. The shot records the envelopes in
its scene metadata. Scene assets are authored separately from these
validation inputs.

The compiler checks the decoded consumer eye against terrain and obstacle
solids at frame times and across the eased, baked AUTOCHIRON orbit between
frames. Terrain clearance is vertical; over a box footprint the same
minimum applies above its top, and entering a solid always fails.
The compiler bounds each native cubic segment and subdivides ambiguous
intervals until it proves clearance or identifies an infeasible or
numerically unresolved interval. The frame receipt records the terrain
lower bound under `nonlinear_terrain_clearance_lower_bound_from_previous_m`.
The straight eye chord remains a diagnostic. CHRONOS continues to own
integer frame selection and replay. There are no new motion comfort or
performance thresholds.

```powershell
python examples/c07_swiss_sources.py artifacts/c07/input
python examples/c07_grindelwald_eiger.py artifacts/c07/input artifacts/c07/eiger --size 960 540
python examples/c07_grindelwald_eiger.py artifacts/c07/input artifacts/c07/ostegg --subject-id '{B056353F-275D-47FA-AB13-DE1E0E7826B5}' --size 720 720
python examples/c07_grindelwald_eiger.py artifacts/c07/input artifacts/c07/ostegg-short --subject-id '{B056353F-275D-47FA-AB13-DE1E0E7826B5}' --size 720 720 --hold-seconds 1
```

Use a fresh output directory for each attempt. Add `--plan-only` for
camera and caption checks. Source preparation requires rasterio, pyproj,
GeoPandas and pyogrio. The compiler uses pyproj from the existing raster
extra. Encoding requires FFmpeg with libx264. The shared encoder checks the
expected numbered PNG set and publishes an MP4 only after a successful encode.
It uses H.264, 8-bit 4:2:0, limited range and BT.709 matrix and primaries.
The Swiss example declares sRGB transfer. The CHRONOS manifest, PNGs and
provenance sidecars remain separate from the MP4.
Source preparation records the pinned TLM edition and each required archive
member's SHA-256. Subsequent runs verify those pins and the derived input hashes
before publishing files. For an older receipt without those pins, prepare in a
fresh directory to acquire and pin the sources.
Run `python -m pytest examples/test_c07_swiss_sources.py` with the source
preparation dependencies to check the cache and receipt rules locally.

`CompiledSubjectShot.render` forwards `cache=` and `certificate=` to
CHRONOS. Use `key_frames` with the CHRONOS manifest to identify the stills.
The example encodes the same PNGs and replays the reveal frame against its
stored pixel hash. Infeasible checks identify the constraint, frame, and
feature before encoding.

The [registered Unreal example](../../examples/c07_unreal/README.md) takes
the evaluated Swiss shot to Sequencer and Movie Render Queue. It keeps the
external frame receipt separate and finishes the returned sequence with
the frozen CHRONOS captions and a leader to the projected subject. The
subject movie uses sharp frames. Motion blur and depth of field did not
affect the assessed CHRONOS and offline final render paths.
