# ORBIS acceptance evidence

ORBIS physical acceptance runs only in the protected NVIDIA/Vulkan visual job.
The ordinary Python lane skips the physical tests unless
`FORGE3D_RUN_ORBIS_GPU=1` is set. In the selected job, the adapter probe must
identify a non-software NVIDIA discrete GPU on Vulkan and the ORBIS JUnit file
must contain zero skipped tests.

The job restores `assets/tif/dem_rainier.tif` from the verified shared LFS
artifact, renders the complete default descent, and compares the ground frame
with
`tests/golden/terrain/orbis_rainier_ground.nvidia-vulkan.png`. Acceptance
requires measured camera-relative jitter below 0.5 px from at least 32 samples,
an absolute-f32 control more than 10 times worse, a GPU-visible high-water mark
below 512 MiB, and zero ring-boundary crack pixels across at least 64 boundary
samples. The 25-frame descent must execute exactly one bounded poll per frame,
make streaming progress on at least one of the 25 frames, and observe both pending work
and ancestor fallback. The final golden and actual image must each contain
credible ground content (at least 32 distinct RGB colors and at least 5% pixels
outside the modal color), with SSIM at least 0.995 and mean absolute pixel
difference at most 2.

`scripts/write_orbis_evidence.py` creates the retained JSON record from the
runtime files. It reads `git rev-parse HEAD` while the job is running, checks it
against the selected candidate SHA, requires a clean tracked worktree, validates
the exact adapter/metric correspondence and thresholds, and records the exact,
replayable commands supplied by the workflow, including their probe and JUnit
paths. This document intentionally contains no
claimed final commit SHA or copied metric values; the generated
`orbis-evidence.json`, JUnit XML, adapter probe, actual PNG, and raw metrics JSON
are the authoritative run-specific evidence.
