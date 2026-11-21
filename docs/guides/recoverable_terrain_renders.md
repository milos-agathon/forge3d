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
