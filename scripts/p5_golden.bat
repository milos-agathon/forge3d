@echo off
REM scripts\p5_golden.bat
REM P5.5: Golden artifact generator for screen-space GI (P5.0–P5.4)
REM
REM Generates all P5 PNG artifacts and updates reports\p5\p5_meta.json by
REM driving the interactive viewer example (plus the headless P5.0 exporter)
REM using the shared GI CLI schema and P5 capture helpers.
REM
REM Usage:
REM   scripts\p5_golden.bat

setlocal enabledelayedexpansion

if not exist Cargo.toml (
  echo Error: Must run from repository root (Cargo.toml not found)
  exit /b 1
)

echo === P5 Golden Artifact Generator (P5.0–P5.4) ===

REM P5.0 – G-Buffer & HZB export (headless)
echo.
echo [P5.0] Generating G-buffer + HZB artifacts (headless)...
cargo run --release --example p5_dump_gbuffer -- --size 1280 720 --out reports/p5
if errorlevel 1 goto :error

call :run_p5_1
if errorlevel 1 goto :error

call :run_p5_2
if errorlevel 1 goto :error

call :run_p5_3_glossy
if errorlevel 1 goto :error

call :run_p5_3_thickness
if errorlevel 1 goto :error

call :run_p5_4
if errorlevel 1 goto :error

echo.
echo === P5 golden artifacts written under reports\p5\ ===
echo Verify p5_meta.json and PNGs before committing.
exit /b 0

:run_p5_1
echo.
echo [P5.1] Generating SSAO/GTAO artifacts via interactive_viewer...
set FORGE3D_AUTO_SNAPSHOT_PATH=reports\p5\p5_p51_auto.png
(
  echo :p5 cornell
  echo :p5 grid
  echo :p5 sweep
  echo :quit
) | cargo run --release --example interactive_viewer -- --gi gtao:on --ssao-radius 0.5 --ssao-intensity 1.5 --ssao-mul 1.0 --ao-blur on --ao-temporal-alpha 0.0 --gi-seed 42
exit /b %errorlevel%

:run_p5_2
echo.
echo [P5.2] Generating SSGI artifacts via interactive_viewer...
set FORGE3D_AUTO_SNAPSHOT_PATH=reports\p5\p5_p52_auto.png
(
  echo :p5 ssgi-cornell
  echo :p5 ssgi-temporal
  echo :quit
) | cargo run --release --example interactive_viewer -- --size 1920x1080 --viz material --gi ssgi:on --ssgi-steps 24 --ssgi-radius 1.0 --ssgi-temporal-alpha 0.15 --ssgi-temporal-enable on --ssgi-half off --ssgi-edges on --ssgi-upsample-sigma-depth 0.02 --ssgi-upsample-sigma-normal 0.25 --gi-seed 42
exit /b %errorlevel%

:run_p5_3_glossy
echo.
echo [P5.3] Generating SSR glossy spheres artifact via interactive_viewer...
set FORGE3D_AUTO_SNAPSHOT_PATH=reports\p5\p5_p53_glossy_auto.png
(
  echo :load-ssr-preset
  echo :p5 ssr-glossy
  echo :quit
) | cargo run --release --example interactive_viewer -- --size 1920x1080 --viz lit --gi ssr:on --ssr-max-steps 96 --ssr-thickness 0.08 --gi-seed 42
exit /b %errorlevel%

:run_p5_3_thickness
echo.
echo [P5.3] Generating SSR thickness ablation artifact via interactive_viewer...
set FORGE3D_AUTO_SNAPSHOT_PATH=reports\p5\p5_p53_thickness_auto.png
(
  echo :load-ssr-preset
  echo :p5 ssr-thickness
  echo :quit
) | cargo run --release --example interactive_viewer -- --size 1920x1080 --viz lit --gi ssr:on --ssr-max-steps 64 --ssr-thickness 0.08 --gi-seed 42
exit /b %errorlevel%

:run_p5_4
echo.
echo [P5.4] Generating GI stack ablation artifact via interactive_viewer...
set FORGE3D_AUTO_SNAPSHOT_PATH=reports\p5\p5_p54_auto.png
(
  echo :load-ssr-preset
  echo :p5 gi-stack
  echo :quit
) | cargo run --release --example interactive_viewer -- --size 1920x1080 --viz lit --gi-seed 42
exit /b %errorlevel%

:error
echo.
echo P5 golden artifact generation FAILED.
exit /b 1
