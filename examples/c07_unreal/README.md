# Registered Swiss subject shot in Unreal

This optional example uses the C07 Swiss comparison kit, the native
Forge3D build with the bundled caption atlas, and an isolated copy of the
`C07Acceptance.uproject` fixture. The local handoff and fixture are under
`artifacts/c07-subject-story/` in the primary checkout. The comparison kit
includes `comparison.json`, the north-first LV95/LN02 terrain, and the
indexed camera and landmark tables. Keep the original kit and project
separate from the attempt.

Use Unreal 5.8.3, the project’s Python/Sequencer/Movie Render Pipeline and
Geometry Scripting plugins, and Vulkan SM5. Use FFmpeg with libx264 and
FFprobe for the MP4. The supported fixture has centred perspective,
zero authored roll, static terrain, and one frame-centred spatial and
temporal sample at 24 fps. Its X axis points north and Y points east.
Its origin is the kit's LV95/LN02 origin and its unit is the centimetre.
The adapter requires this registration. Unsupported requests fail.

Run this setup from the C07 worktree in **both PowerShell terminals**.
Use the Python environment and native Forge3D 1.47.1 runtime retained for
this C07 run. Set the same kit and attempt paths in both terminals and
choose a fresh attempt directory for preparation. The import override
keeps the example shim from putting the worktree's Python package ahead
of the selected native runtime. The preflight runs that shim and checks
where both modules load from.

```powershell
$python = 'D:/forge3d/.venv/Scripts/python.exe'
$runtime = (Resolve-Path 'D:/forge3d/artifacts/c07-subject-story/fix-20261109/runtime').Path
$env:PYTHONPATH = $runtime
$env:FORGE3D_TEST_INSTALLED_WHEEL = '1'
$tools = (Resolve-Path examples/c07_unreal).Path
$kit = 'D:/forge3d/artifacts/c07-subject-story/review-828df486/kit'
$attempt = 'D:/path/to/new/attempt'
@'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("examples").resolve()))
from _import_shim import ensure_repo_import

ensure_repo_import()
import forge3d
from forge3d import _forge3d

runtime = Path(sys.argv[1]).resolve()
for module in (forge3d, _forge3d):
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(runtime):
        raise SystemExit(f"Expected {module.__name__} under {runtime}, imported {origin}")
    print(origin)
'@ | & $python - $runtime
if ($LASTEXITCODE -ne 0) { throw 'C07 native runtime import failed' }
```

In the first terminal, set the path to the isolated project copy. If the
project uses a filesystem shader cache, keep its path within Unreal's
Windows path limit. An isolated copy of the matching installation's
existing cache avoids a cold compile.

```powershell
$project = 'D:/path/to/isolated/C07Acceptance.uproject'
$ue = 'D:/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
& $python "$tools/prepare_c07_mrq.py" $attempt $kit --asset-folder C07_New --neutral
if ($LASTEXITCODE -ne 0) { throw 'C07 job preparation failed' }
$env:FORGE3D_C07_MRQ_ATTEMPT = $attempt
& $ue $project "-ExecutePythonScript=$tools/editor_entry.py" "-abslog=$attempt/editor.log" -vulkan -ddc=C07Local -unattended -NoSplash -NoSound -NoAnalytics -NoEpicPortal -stdout
```

The planner verifies the input hashes, prepares Eiger and Ostegg, then
compiles a one-second Ostegg hold in Python. It confirms that the new
96-frame shot has exactly the earlier poses and geographic checks through
its last frame. The editor creates the level sequences and render jobs
without hand-retiming. `--scene-from` reopens the exact saved sequences and
presets using a scene receipt for the copied project directory. Before
rebinding an archived receipt to a copy, verify the saved level, material,
sequences and presets against the archived assets. The full camera/sample
verification applies to both creation and reuse. Use `--lifecycle` to also
request a repeat, cancellation and a missing-sequence failure.

`--neutral` keeps the registered level's unlit height tint, fixes exposure
at EV100 0 with zero compensation, turns off bloom and vignette, disables
the filmic curve and writes sRGB PNGs. Antialiasing, motion blur and focus
blur are disabled for this fixture. The earlier default-tone-curve
output uses a separate preset treatment. [Unreal exposure settings](https://dev.epicgames.com/documentation/en-us/unreal-engine/auto-exposure-in-unreal-engine) · [Unreal MRQ colour output](https://dev.epicgames.com/documentation/en-us/unreal-engine/cinematic-render-settings-and-formats-in-unreal-engine)

While the editor is open, wait for `queue-finished.json`. Require the
terminal status, reported MRQ work and camera samples, plus the exact PNG
set. A process exit alone is insufficient. Run verification and finishing
from the second terminal after running the same setup and import preflight.

```powershell
& $python "$tools/verify_c07_mrq.py" $attempt --editor-log "$attempt/editor.log" --engine-root 'D:/Epic Games/UE_5.8/Engine'
$chronos = 'D:/path/to/the/matching/CHRONOS/frames'
& $python "$tools/finish_c07_mrq.py" "$attempt/eiger-job.json" "$attempt/eiger" $kit $chronos "$attempt/publication/eiger"
New-Item -ItemType File "$attempt/quit-editor"
```

Use the Ostegg CHRONOS source for both Ostegg runs. The shorter job refers
to its matching 96-frame prefix. The finishing step draws the existing
CHRONOS caption and a screen-space leader to the verified terrain point,
then writes new PNGs, still references and an MP4 alongside its publication
receipt. The source CHRONOS records and raw MRQ frames stay separate.
Keep the job, scene and project hashes, camera samples, raw result and
`external-cache-context.json` with the movie. The context binds the engine,
plugins, renderer, saved assets, settings and evaluated times. The receipt
identifies Unreal as an external renderer.
