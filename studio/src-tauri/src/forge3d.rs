//! Forge3D engine discovery and DEM access.
//!
//! Studio no longer opens a separate viewer window: rendering happens in the
//! viewport's Rendered mode (`render.rs`). What remains here is what that path
//! and the terrain preview need — finding the interpreter that can import
//! `forge3d`, resolving the default DEM, and measuring it.
//!
//! Process control is deliberately honest: if the engine cannot be imported,
//! the probe says exactly what is missing.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::Serialize;

/// One layer of a starter scene: the file it draws and its own style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecipeLayer {
    /// A raster draped on the terrain (the land cover, an orthophoto).
    Imagery {
        relative: &'static str,
        opacity: f32,
        preserve_colors: bool,
        despeckle: bool,
    },
    /// A vector layer draped on the terrain (the rail network).
    Vector {
        relative: &'static str,
        colour: [f32; 4],
        width_px: f32,
    },
    /// The labels of the vector layer above it (the place names).
    Labels {
        relative: &'static str,
        field: &'static str,
        colour: [f32; 4],
        size_px: f32,
    },
    /// The map plate around the map, with its furniture.
    Plate {
        title: &'static str,
        caption: &'static str,
    },
}

/// A starter scene: a DEM and the cartography that goes with it.
///
/// The gallery's recipes are the source: the Swiss land-cover, the Luxembourg
/// rail network and the Mount Fuji labels. The framing is the studio's own map
/// camera (the terrain's footprint, from the given direction) because the
/// studio's still and its viewport have to agree on what they show.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneRecipe {
    /// What the UI sends back.
    pub id: &'static str,
    pub name: &'static str,
    /// The DEM, relative to the repo root.
    pub dem: &'static str,
    pub description: &'static str,
    /// The map's layers, back to front.
    pub layers: &'static [RecipeLayer],
    /// The map's camera: the azimuth (from east towards north) and the polar
    /// angle from straight down.
    pub camera: Option<(f64, f64)>,
    /// The sun: the azimuth (from north, clockwise) and the elevation.
    pub sun: Option<(f64, f64)>,
}

/// The starter scenes offered in the New Project dialog.
pub const SCENE_PRESETS: [SceneRecipe; 5] = [
    SceneRecipe {
        id: "bryce",
        name: "Bryce Canyon",
        dem: "assets/tif/Bryce_Canyon_1024.tif",
        description: "4 x 3.8 km · 1024² DEM · the NAIP orthophoto draped on the hoodoos",
        layers: &[RecipeLayer::Imagery {
            relative: "assets/tif/Bryce_Canyon_naip.tif",
            opacity: 0.82,
            preserve_colors: true,
            despeckle: false,
        }],
        camera: None,
        sun: None,
    },
    SceneRecipe {
        id: "swiss",
        name: "Swiss Alps",
        dem: "assets/tif/switzerland_dem.tif",
        description: "353 x 226 km · national DEM · Alps to Jura",
        layers: &[],
        camera: None,
        sun: None,
    },
    SceneRecipe {
        id: "swiss-landcover",
        name: "Swiss land-cover",
        dem: "assets/tif/switzerland_dem.tif",
        description: "the gallery recipe: land cover draped on the Alps, with the plate",
        layers: &[
            RecipeLayer::Imagery {
                relative: "assets/tif/switzerland_land_cover.tif",
                opacity: 0.82,
                preserve_colors: true,
                despeckle: true,
            },
            RecipeLayer::Plate {
                title: "Swiss land cover",
                // The plate's caption is drawn with the packaged Latin atlas,
                // which has no middot: a plain dash keeps it readable.
                caption: "Swiss Alps - land cover draped on the national DEM",
            },
        ],
        camera: Some((270.0, 10.0)),
        sun: Some((315.0, 17.0)),
    },
    SceneRecipe {
        id: "luxembourg-rail",
        name: "Luxembourg rail network",
        dem: "assets/tif/luxembourg_dem.tif",
        description: "the gallery recipe: 2,035 rail features draped on the country",
        layers: &[RecipeLayer::Vector {
            relative: "assets/gpkg/luxembourg_rail.gpkg",
            colour: [0.87, 0.1, 0.1, 1.0],
            width_px: 3.0,
        }],
        camera: Some((270.0, 31.0)),
        sun: None,
    },
    SceneRecipe {
        id: "fuji-labels",
        name: "Mount Fuji labels",
        dem: "assets/tif/Mount_Fuji_30m.tif",
        description: "the gallery recipe: the climbing stations named on the cone",
        layers: &[
            RecipeLayer::Vector {
                relative: "assets/gpkg/Mount_Fuji_places.gpkg",
                colour: [0.87, 0.1, 0.1, 1.0],
                width_px: 2.0,
            },
            RecipeLayer::Labels {
                relative: "assets/gpkg/Mount_Fuji_places.gpkg",
                field: "name.en",
                colour: [0.08, 0.08, 0.1, 1.0],
                size_px: 16.0,
            },
        ],
        camera: Some((342.0, 54.0)),
        sun: None,
    },
];

/// Bundled DEM used as the default scene. Overridable with `FORGE3D_STUDIO_DEM`.
const DEFAULT_DEM_RELATIVE: &str = "assets/tif/Bryce_Canyon_1024.tif";

const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Probe program: reports the package version. The viewer binary is optional
/// diagnostics only — rendering runs in-process (`studio_render.py`), so a
/// missing `interactive_viewer` must not gate Rendered mode.
const PROBE_SCRIPT: &str = r#"
import json

def main():
    try:
        import forge3d
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": "import forge3d failed: %s" % (exc,)}))
        return
    version = getattr(forge3d, "__version__", "unknown")
    backend = adapter = None
    software = False
    try:
        # `device_probe()` reports the adapter as "name"; accepting the other
        # spellings keeps an older engine usable, and the explicit software
        # flag means a CPU fallback can never be mistaken for a GPU render.
        probe = forge3d.device_probe()
        backend = probe.get("backend")
        adapter = (
            probe.get("name")
            or probe.get("adapter_name")
            or probe.get("adapter")
            or probe.get("device_type")
        )
        software = bool(probe.get("software_fallback"))
    except Exception:
        pass
    try:
        from forge3d.denoise_oidn import oidn_available
        denoise = bool(oidn_available())
    except Exception:
        denoise = False
    features = None
    try:
        # What the device actually granted — the fact every later "needs X"
        # decision reads, instead of guessing from the adapter name.
        granted = forge3d.capabilities().get("granted")
        if isinstance(granted, list):
            features = sorted(str(name) for name in granted)
    except Exception:
        pass
    try:
        from forge3d._viewer_binary import find_viewer_binary
        import forge3d.viewer as viewer
        binary = find_viewer_binary(viewer.__file__)
        detail = "engine ready (viewer binary present)"
    except Exception as exc:
        binary = None
        detail = "engine ready (viewer binary not found — optional: %s)" % (exc,)
    print(json.dumps({
        "ok": True,
        "version": version,
        "viewer": binary,
        "detail": detail,
        "backend": backend,
        "adapter": adapter,
        "software": software,
        "denoise": denoise,
        "features": features,
    }))

main()
"#;

/// Sampling script for the Studio heightfield preview.
///
/// Reading, nodata masking and the metric footprint are the engine's
/// (`forge3d.io.load_dem`, `forge3d.gis`), so a preview and a render can never
/// disagree about a DEM. Only the down-sample and the JSON shape are the
/// shell's.
const PREVIEW_SCRIPT: &str = r#"
import json, sys

try:
    import numpy as np
    import forge3d.io as fio
    from forge3d import gis
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine is unavailable: %s" % (exc,)}))
    raise SystemExit(0)

path = sys.argv[1]
resolution = max(8, min(256, int(sys.argv[2])))
try:
    dem = fio.load_dem(path, fill_nodata_values=False)
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the DEM could not be read: %s" % (exc,)}))
    raise SystemExit(0)

raw = np.asarray(dem.data, dtype="float64")
if dem.nodata_value is not None:
    # A sentinel such as -3.4e38 would otherwise define the range and turn the
    # mesh into spikes.
    raw = np.where(np.isclose(raw, float(dem.nodata_value)), np.nan, raw)
valid = np.isfinite(raw)
if not valid.any():
    print(json.dumps({"ok": False, "detail": "the DEM contains no valid elevation samples"}))
    raise SystemExit(0)

# A robust range: single corrupted samples (one 11 km pixel in the Swiss DEM)
# would otherwise define the display range and flatten everything.
values = raw[valid]
low, high = np.percentile(values, [0.01, 99.99])
min_elevation = float(low)
max_elevation = float(high)
outliers = int(np.count_nonzero((values < low) | (values > high)))

height, width = raw.shape
ys = np.linspace(0, height, resolution + 1).astype(int)
xs = np.linspace(0, width, resolution + 1).astype(int)
out = np.empty((resolution, resolution), dtype="float64")
for j in range(resolution):
    for i in range(resolution):
        block = raw[ys[j]:max(ys[j] + 1, ys[j + 1]), xs[i]:max(xs[i] + 1, xs[i + 1])]
        good = block[np.isfinite(block)]
        if good.size:
            out[j, i] = float(np.clip(good.mean(), min_elevation, max_elevation))
        else:
            out[j, i] = min_elevation

# World footprint in metres. Geographic rasters (degrees) are reprojected to
# their local UTM zone; treating degrees as metres reported a 353 km DEM as
# 0.00 km.
crs = str(dem.crs or "")
geographic = "4326" in crs or "wgs84" in crs.lower()
origin_xz = [0.0, 0.0]
if not crs:
    span_xz = [float(width), float(height)]
elif geographic and dem.bounds is not None:
    local = gis.estimate_local_utm(dem.bounds)["crs"]
    left, bottom, right, top = gis.transform_bounds(dem.crs, local, dem.bounds, densify=21)
    span_xz = [abs(right - left), abs(top - bottom)]
else:
    res = dem.resolution or (1.0, 1.0)
    span_xz = [abs(float(res[0])) * width, abs(float(res[1])) * height]
    transform = dem.transform
    if transform is not None:
        origin_xz = [
            float(getattr(transform, "c", 0.0)),
            float(-getattr(transform, "f", 0.0)),
        ]

payload = {
    "ok": True,
    "width": int(width),
    "height": int(height),
    "minElevation": min_elevation,
    "maxElevation": max_elevation,
    "extentMeters": span_xz,
    "originXz": origin_xz,
    "spanXz": span_xz,
    "geographic": bool(geographic),
    "outliers": outliers,
    "resolution": resolution,
    "heights": [round(float(value), 2) for value in out.ravel()],
}
print(json.dumps(payload))
"#;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    pub available: bool,
    /// Interpreter that was probed (the one Rendered mode will use).
    pub python: String,
    pub package_version: Option<String>,
    pub viewer_binary: Option<String>,
    /// wgpu backend the engine selected, when the probe could reach the GPU.
    pub backend: Option<String>,
    /// Adapter name — the GPU actually rendering.
    pub adapter: Option<String>,
    /// True when the interpreter can denoise with OIDN (the `denoise` extra).
    pub denoise: Option<bool>,
    /// True when the engine is running on a software (CPU) adapter — the
    /// status bar must not present that as GPU rendering.
    pub software: Option<bool>,
    /// Device features the adapter actually granted (`forge3d.capabilities()`
    /// `granted`), sorted. `None` when the probe could not ask.
    pub features: Option<Vec<String>>,
    /// Always present: what is missing, or what is ready.
    pub detail: String,
}

pub struct Forge3dState {
    engine: std::sync::Mutex<Option<EngineStatus>>,
    /// Resolved default DEM plus its measured metadata (sampling is not free).
    terrain_cache: std::sync::Mutex<Option<(String, studio_core::TerrainSeed)>>,
}

impl Default for Forge3dState {
    fn default() -> Self {
        Self::new()
    }
}

impl Forge3dState {
    pub fn new() -> Self {
        Self {
            engine: std::sync::Mutex::new(None),
            terrain_cache: std::sync::Mutex::new(None),
        }
    }

    /// Probe the engine once and cache the result.
    pub fn probe(&self, force: bool) -> EngineStatus {
        if !force {
            if let Some(cached) = self.engine.lock().ok().and_then(|guard| guard.clone()) {
                return cached;
            }
        }

        // Keep the *most informative* failure, not the last one: an interpreter
        // that imports forge3d but lacks the viewer binary is far more useful
        // than "python3: no module named forge3d".
        let mut best: Option<EngineStatus> = None;

        for python in interpreter_candidates() {
            match probe_interpreter(&python) {
                Ok(status) => {
                    if status.available {
                        if let Ok(mut guard) = self.engine.lock() {
                            *guard = Some(status.clone());
                        }
                        return status;
                    }
                    let better = status.package_version.is_some();
                    let current = best
                        .as_ref()
                        .map(|entry| entry.package_version.is_some())
                        .unwrap_or(false);
                    if best.is_none() || (better && !current) {
                        best = Some(status);
                    }
                }
                Err(detail) => {
                    if best.is_none() {
                        best = Some(EngineStatus {
                            available: false,
                            python: python.clone(),
                            package_version: None,
                            viewer_binary: None,
                            backend: None,
                            adapter: None,
                            denoise: None,
                            software: None,
                            features: None,
                            detail,
                        });
                    }
                }
            }
        }

        let last = best.unwrap_or(EngineStatus {
            available: false,
            python: "python3".to_owned(),
            package_version: None,
            viewer_binary: None,
            backend: None,
            adapter: None,
            denoise: None,
            software: None,
            features: None,
            detail: "no Python interpreter was found".to_owned(),
        });
        if let Ok(mut guard) = self.engine.lock() {
            *guard = Some(last.clone());
        }
        last
    }

    /// A starter DEM by preset id ("bryce" | "swiss" | "dataset:<name>" | "url:<url>"),
    /// else the default. Dataset names resolve (and download, if need be)
    /// through the engine's own registry; the registry is the allow-list.
    /// Errors are returned, not swallowed: a failed remote fetch must reach
    /// the user as an error, not as an empty scene.
    pub fn terrain_seed(
        &self,
        preset: Option<&str>,
    ) -> Result<Option<studio_core::TerrainSeed>, String> {
        if let Some(url) = preset.and_then(|id| id.strip_prefix("url:")) {
            // Phase 18.3: open without a whole-file download. The engine's
            // range reader supplies the metadata (one HEAD + one coarse
            // overview); the still streams the tiles it needs later.
            let python = self.probe(false).python;
            let info = remote_cog_info(&python, url)?;
            let overview = remote_cog_preview(&python, url, 64)?;
            let trimmed = url.trim_end_matches('/');
            let title = trimmed
                .rsplit('/')
                .next()
                .map(str::to_owned)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "remote-dem".to_owned());
            let dimensions = info
                .levels
                .first()
                .map(|level| [level.width, level.height])
                .filter(|pair| pair[0] > 0 && pair[1] > 0);
            let elevation_range = overview
                .get("minElevation")
                .and_then(|low| low.as_f64())
                .zip(overview.get("maxElevation").and_then(|high| high.as_f64()))
                .map(|(low, high)| [low as f32, high as f32]);
            return Ok(Some(studio_core::TerrainSeed {
                title: Some(title.clone()),
                name: title,
                path: url.to_owned(),
                byte_size: 0,
                overlay: None,
                dimensions,
                elevation_range,
                world_origin_xz: None,
                world_span_xz: None,
            }));
        }
        if let Some(name) = preset.and_then(|id| id.strip_prefix("dataset:")) {
            let python = self.probe(false).python;
            let title = dataset_list(&python)
                .ok()
                .and_then(|rows| rows.into_iter().find(|row| row.name == name))
                .and_then(|row| row.filename);
            let path = dataset_fetch(&python, name)?;
            return Ok(self.seed_for_absolute(PathBuf::from(path), title));
        }
        let relative = preset
            .and_then(|id| SCENE_PRESETS.iter().find(|recipe| recipe.id == id))
            .map(|recipe| recipe.dem.to_owned())
            .unwrap_or_else(|| DEFAULT_DEM_RELATIVE.to_owned());
        Ok(self.seed_for_relative(&relative))
    }

    /// The starter scene's recipe, when the id names one.
    pub fn scene_recipe(&self, preset: Option<&str>) -> Option<&'static SceneRecipe> {
        preset.and_then(|id| SCENE_PRESETS.iter().find(|recipe| recipe.id == id))
    }

    fn seed_for_relative(&self, relative: &str) -> Option<studio_core::TerrainSeed> {
        let path = resolve_dem(relative)?;
        self.seed_for_absolute(path, None)
    }

    fn seed_for_absolute(
        &self,
        path: PathBuf,
        title: Option<String>,
    ) -> Option<studio_core::TerrainSeed> {
        let path_text = path.to_string_lossy().into_owned();

        if let Ok(cache) = self.terrain_cache.lock() {
            if let Some((cached_path, seed)) = cache.as_ref() {
                if cached_path == &path_text {
                    return Some(seed.clone());
                }
            }
        }

        let metadata = std::fs::metadata(&path).ok()?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "terrain.tif".to_owned());
        let title = title.or_else(|| {
            SCENE_PRESETS
                .iter()
                .find(|recipe| recipe.dem == path_text)
                .map(|recipe| recipe.name.to_owned())
        });
        let mut seed = studio_core::TerrainSeed {
            title,
            name,
            path: path_text.clone(),
            byte_size: metadata.len(),
            dimensions: None,
            elevation_range: None,
            world_origin_xz: None,
            world_span_xz: None,
            overlay: None,
        };

        let python = self.probe(false).python;
        match sample_terrain(&python, &path_text, 48) {
            Ok(value) => {
                let width = value.get("width").and_then(|v| v.as_u64());
                let height = value.get("height").and_then(|v| v.as_u64());
                if let (Some(width), Some(height)) = (width, height) {
                    seed.dimensions = Some([width as u32, height as u32]);
                }
                let low = value.get("minElevation").and_then(|v| v.as_f64());
                let high = value.get("maxElevation").and_then(|v| v.as_f64());
                if let (Some(low), Some(high)) = (low, high) {
                    seed.elevation_range = Some([low as f32, high as f32]);
                }
                let origin = value.get("originXz").and_then(|v| v.as_array());
                let span = value.get("spanXz").and_then(|v| v.as_array());
                if let (Some(origin), Some(span)) = (origin, span) {
                    let pair = |values: &Vec<serde_json::Value>| -> Option<[f64; 2]> {
                        Some([values.first()?.as_f64()?, values.get(1)?.as_f64()?])
                    };
                    seed.world_origin_xz = pair(origin);
                    seed.world_span_xz = pair(span);
                }
            }
            Err(error) => log::warn!(
                "DEM metadata unavailable ({error}); the renderer will use its own framing"
            ),
        }

        if let Ok(mut cache) = self.terrain_cache.lock() {
            *cache = Some((path_text, seed.clone()));
        }
        Some(seed)
    }
}

/// The Bryce starter ships its NAIP orthophoto as the terrain's albedo. Only
/// the untouched default Bryce scene gets it: an explicit `FORGE3D_STUDIO_DEM`
/// override or another preset means the user brought their own DEM, and
/// Resolve a bundled file: `FORGE3D_STUDIO_DEM` overrides the default sample,
/// otherwise the named file is searched from the executable and the working
/// directory.
fn resolve_dem(relative: &str) -> Option<PathBuf> {
    if relative == DEFAULT_DEM_RELATIVE {
        if let Some(path) = std::env::var_os("FORGE3D_STUDIO_DEM") {
            let candidate = PathBuf::from(path);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    resolve_bundled(relative)
}

/// Search the executable's and the working directory's ancestors for a file
/// relative to the repo root (`assets/…`).
pub(crate) fn resolve_bundled(relative: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().take(6).map(Path::to_path_buf));
    }
    if let Ok(current) = std::env::current_dir() {
        roots.extend(current.ancestors().take(6).map(Path::to_path_buf));
    }
    for root in roots {
        candidates.push(root.join(relative));
    }
    candidates.into_iter().find(|candidate| candidate.is_file())
}

const SCRIPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Run a one-shot Python script with a deadline.
///
/// `Command::output()` has no timeout, so a hung interpreter leaves the caller
/// pending forever. The pipes are drained on threads, otherwise a large payload
/// (a DEM preview, a base64 PNG) can fill the pipe buffer and stall the child
/// before it exits.
fn run_script(
    python: &str,
    script: &str,
    args: &[String],
    timeout: Duration,
) -> Result<(String, String), String> {
    let mut command = Command::new(python);
    command
        .arg("-c")
        .arg(script)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(extra) = python_path_entries() {
        command.env("PYTHONPATH", extra);
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not run {python}: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "the script has no stdout".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "the script has no stderr".to_owned())?;
    let out_reader = std::thread::spawn(move || read_all(stdout));
    let err_reader = std::thread::spawn(move || read_all(stderr));

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = out_reader.join().unwrap_or_default();
                let stderr = err_reader.join().unwrap_or_default();
                if !status.success() && stdout.trim().is_empty() {
                    return Err(format!("{python} exited with {status}: {}", stderr.trim()));
                }
                return Ok((stdout, stderr));
            }
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = out_reader.join();
                let _ = err_reader.join();
                return Err(format!(
                    "{python} did not answer within {}s",
                    timeout.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(error) => return Err(format!("could not wait for {python}: {error}")),
        }
    }
}

fn read_all(mut reader: impl std::io::Read) -> String {
    let mut text = String::new();
    let _ = reader.read_to_string(&mut text);
    text
}

/// Sample a DEM into a heightfield for the Studio viewport.
pub fn sample_terrain(
    python: &str,
    path: &str,
    resolution: u32,
) -> Result<serde_json::Value, String> {
    let (stdout, stderr) = run_script(
        python,
        PREVIEW_SCRIPT,
        &[path.to_owned(), resolution.to_string()],
        SCRIPT_TIMEOUT,
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the DEM sampler produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the DEM sampler returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the DEM could not be sampled")
            .to_owned());
    }
    Ok(value)
}

/// The engine's colormap sampled into the stops the Material preview paints.
///
/// `Colormap.rgba` is **linear** RGBA (the LUT's own working space), so the
/// stops go straight into the preview's vertex colours without an sRGB decode;
/// the panel converts them for CSS. Sampling 32 of the 256 texels keeps the
/// preview within ~1/255 of the rendered LUT.
const COLORMAP_SCRIPT: &str = r#"
import json, sys

try:
    import numpy as np
    import forge3d as f3d
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine is unavailable: %s" % (exc,)}))
    raise SystemExit(0)

name = sys.argv[1]
token = name if ":" in name else "forge3d:" + name
try:
    cmap = f3d.get_colormap(token)
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "unknown colormap %r: %s" % (name, exc)}))
    raise SystemExit(0)

rgba = np.asarray(cmap.rgba, dtype=float)
if rgba.ndim != 2 or rgba.shape[0] < 2:
    print(json.dumps({"ok": False, "detail": "the colormap has no stops"}))
    raise SystemExit(0)
count = 32
indices = np.linspace(0, rgba.shape[0] - 1, count).round().astype(int)
stops = [[round(float(rgba[i][0]), 6), round(float(rgba[i][1]), 6), round(float(rgba[i][2]), 6)] for i in indices]
print(json.dumps({"ok": True, "name": token, "stops": stops}))
"#;

/// One-shot colormap sample. No GPU session: the preview only needs the ramp.
pub fn colormap_stops(python: &str, name: &str) -> Result<Vec<[f64; 3]>, String> {
    let (stdout, stderr) = run_script(python, COLORMAP_SCRIPT, &[name.to_owned()], SCRIPT_TIMEOUT)?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the colormap sampler produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the colormap sampler returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the colormap could not be sampled")
            .to_owned());
    }
    let stops = value
        .get("stops")
        .and_then(|stops| stops.as_array())
        .ok_or_else(|| "the colormap sampler returned no stops".to_owned())?;
    stops
        .iter()
        .map(|stop| {
            let channels = stop
                .as_array()
                .ok_or_else(|| "a colormap stop is not an array".to_owned())?;
            if channels.len() != 3 {
                return Err("a colormap stop does not have three channels".to_owned());
            }
            let mut rgb = [0.0_f64; 3];
            for (index, value) in channels.iter().enumerate() {
                rgb[index] = value
                    .as_f64()
                    .ok_or_else(|| "a colormap channel is not a number".to_owned())?;
            }
            Ok(rgb)
        })
        .collect()
}

/// Solar position from the engine's SIDERA ephemeris.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Azimuth from north toward east, degrees (the engine's convention).
    pub azimuth_deg: f64,
    /// Degrees above the horizon.
    pub elevation_deg: f64,
    /// Direction toward the sun in MapScene axes: +X east, +Y up, +Z north.
    pub direction: [f32; 3],
    pub daytime: bool,
}

/// One-shot ephemeris call. No GPU session: the shell's solar button must not
/// wait for the renderer service.
const SOLAR_SCRIPT: &str = r#"
import json, sys

try:
    import forge3d
    position = forge3d.sun_position(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
    direction = position.to_scene_direction()
    print(json.dumps({
        "ok": True,
        "azimuth": position.azimuth,
        "elevation": position.elevation,
        "direction": [direction[0], direction[1], direction[2]],
        "daytime": bool(position.is_daytime()),
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "detail": str(exc)}))
"#;

/// Ask the engine where the sun is for a UTC timestamp and a place.
///
/// The engine owns the astronomy (SIDERA); Studio only turns the returned
/// direction into the light's rotation (`workspace::sun_rotation`). The
/// direction is in MapScene axes, so no azimuth conversion happens here.
pub fn solar_position(
    python: &str,
    latitude: f64,
    longitude: f64,
    datetime_utc: &str,
) -> Result<SolarPosition, String> {
    let (stdout, stderr) = run_script(
        python,
        SOLAR_SCRIPT,
        &[
            latitude.to_string(),
            longitude.to_string(),
            datetime_utc.to_owned(),
        ],
        SCRIPT_TIMEOUT,
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the solar probe produced no output: {}", stderr.trim()))?;
    parse_solar_reply(line)
}

fn parse_solar_reply(line: &str) -> Result<SolarPosition, String> {
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the solar probe returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the engine could not compute the sun position")
            .to_owned());
    }
    let azimuth = value
        .get("azimuth")
        .and_then(|value| value.as_f64())
        .ok_or_else(|| "the solar probe returned no azimuth".to_owned())?;
    let elevation = value
        .get("elevation")
        .and_then(|value| value.as_f64())
        .ok_or_else(|| "the solar probe returned no elevation".to_owned())?;
    let direction = value
        .get("direction")
        .and_then(|value| value.as_array())
        .filter(|entries| entries.len() == 3)
        .ok_or_else(|| "the solar probe returned no direction".to_owned())?;
    let mut parsed = [0.0f32; 3];
    for (index, entry) in direction.iter().enumerate() {
        parsed[index] = entry
            .as_f64()
            .ok_or_else(|| "the solar direction is not numeric".to_owned())?
            as f32;
    }
    Ok(SolarPosition {
        azimuth_deg: azimuth,
        elevation_deg: elevation,
        direction: parsed,
        daytime: value
            .get("daytime")
            .and_then(|value| value.as_bool())
            .unwrap_or(true),
    })
}

/// The Cartography viewport layers the still draws, over IPC.
///
/// Reads the vector layer's geometry (or the graticule's grid) in the DEM's
/// own frame and normalises it to the same 0..1 lattice the terrain preview
/// uses, so the lines land exactly where the still would put them. The
/// vector reader is the still's own (`studio_render.read_vector_features`),
/// imported from the service file — never a second implementation.
const VECTOR_PREVIEW_SCRIPT: &str = r#"
import json, os, sys

kind = sys.argv[1]
dem_path = sys.argv[2]
service_dir = sys.argv[3]

sys.path.append(service_dir)

import studio_render as sr

# Bounds and CRS only — the features are what the preview is for, so the DEM
# header is read, never the raster (a full read of a 90 MB Swiss grid to
# learn two corners would be silly).
try:
    import rasterio
    with rasterio.open(dem_path) as src:
        dem_crs_value = src.crs
        dem_bounds_value = src.bounds
    crs = str(dem_crs_value or "")
    bounds = (dem_bounds_value.left, dem_bounds_value.bottom, dem_bounds_value.right, dem_bounds_value.top)
except Exception:
    from forge3d import io as fio
    try:
        dem = fio.load_dem(dem_path, fill_nodata_values=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": "the DEM could not be read: %s" % (exc,)}))
        raise SystemExit(0)
    crs = str(dem.crs or "")
    bounds = dem.bounds
if bounds is None:
    print(json.dumps({"ok": False, "detail": "the DEM has no bounds"}))
    raise SystemExit(0)
west, south, east, north = (float(value) for value in bounds)
span_x = max(east - west, 1e-9)
span_y = max(north - south, 1e-9)

def normalize(line):
    return [[(float(x) - west) / span_x, (north - float(y)) / span_y] for x, y in line]

if kind == "vector":
    vector_path = sys.argv[4]
    label_field = sys.argv[5]
    legend_field = sys.argv[6] if len(sys.argv) > 6 else ""
    lines, points, vector_crs, categories = sr.read_vector_features(
        vector_path, label_field, legend_field
    )
    file_code = sr.crs_code(vector_crs)
    dem_code = sr.crs_code(crs)
    if dem_code and file_code and dem_code != file_code:
        print(json.dumps({
            "ok": False,
            "detail": "the vector file is in EPSG:%s and the DEM in EPSG:%s; reproject the file onto the DEM's CRS first" % (file_code, dem_code),
        }))
        raise SystemExit(0)
    kept = [(line, category) for line, category in zip(lines, categories) if len(line) >= 2]
    out_lines = [normalize(line) for line, _ in kept]
    out_categories = [category for _, category in kept]
    assignments, ordered = sr.category_assignments(out_categories)
    palette = {
        value: [round(float(c), 6) for c in rgb]
        for value, rgb, _count in ordered
    }
    out_points = []
    for _identifier, x, y, text in points:
        u = (float(x) - west) / span_x
        v = (north - float(y)) / span_y
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            out_points.append({"u": u, "v": v, "text": str(text)})
    print(json.dumps({"ok": True, "kind": "vector", "lines": out_lines, "points": out_points,
                      "categories": out_categories, "palette": palette}))
    raise SystemExit(0)

if kind == "polygon":
    polygon_path = sys.argv[4]
    import studio_render as _sr  # the still's reader, never a second GPKG parser
    polygons, file_crs = _sr.read_polygon_features(polygon_path)
    dem_code = _sr.crs_code(crs)
    file_code = _sr.crs_code(file_crs)
    if dem_code and file_code and dem_code != file_code:
        print(json.dumps({
            "ok": False,
            "detail": "the polygon file is in EPSG:%s and the DEM in EPSG:%s; reproject the file onto the DEM's CRS first" % (file_code, dem_code),
        }))
        raise SystemExit(0)
    exteriors = []
    holes = []
    for exterior, ring_holes in polygons:
        exteriors.append([[(float(x) - west) / span_x, (north - float(y)) / span_y] for x, y in exterior])
        holes.append([
            [[(float(x) - west) / span_x, (north - float(y)) / span_y] for x, y in ring]
            for ring in ring_holes
        ])
    print(json.dumps({
        "ok": True,
        "kind": "polygon",
        "exteriors": exteriors,
        "holes": holes,
    }))
    raise SystemExit(0)

if kind == "graticule":
    interval = float(sys.argv[4])
    with_labels = sys.argv[5] == "1"
    lines, annotations = sr.graticule_geometry(
        (west, south, east, north), crs, interval, with_labels
    )
    out_lines = [normalize(line) for line in lines if len(line) >= 2]
    labels = []
    for index, (line, annotation) in enumerate(zip(lines, annotations)):
        text = str(annotation.get("text", "") or "")
        if not text:
            continue
        kind_of = str(annotation.get("kind", "") or "")
        # Same rule the still uses to place a grid annotation: the point that
        # leaves the frame — lowest for a meridian, leftmost for a parallel.
        if kind_of == "meridian":
            ax, ay = min(line, key=lambda point: point[1])
        else:
            ax, ay = min(line, key=lambda point: point[0])
        labels.append({
            "kind": kind_of,
            "text": text,
            "u": (float(ax) - west) / span_x,
            "v": (north - float(ay)) / span_y,
        })
    print(json.dumps({"ok": True, "kind": "graticule", "lines": out_lines, "labels": labels}))
    raise SystemExit(0)

print(json.dumps({"ok": False, "detail": "unknown preview kind: %s" % kind}))
"#;

/// Decoded raster preview for the Material preview's terrain drape: a
/// downsampled RGBA PNG, colour handling mirroring
/// `studio_render.build_overlay`'s preserve-colours path so Material and
/// Rendered drape the same pixels. The webview cannot decode GeoTIFF, hence
/// the engine-python round trip.
const RASTER_PREVIEW_SCRIPT: &str = r#"
import base64, io, json, sys

path = sys.argv[1]
width = max(64, min(4096, int(sys.argv[2])))
try:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "raster preview needs the engine python: %s" % (exc,)}))
    raise SystemExit(0)

try:
    with rasterio.open(path) as src:
        bands = min(3, src.count)
        height = max(1, int(round(width * src.height / max(1, src.width))))
        planes = [
            src.read(b + 1, out_shape=(height, width), resampling=Resampling.average, masked=True)
            for b in range(bands)
        ]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for index in range(bands):
        rgb[:, :, index] = np.clip(np.asarray(planes[index].filled(0.0), dtype=np.float32), 0.0, None)
    if bands == 1:
        rgb[:, :, 1] = rgb[:, :, 0]
        rgb[:, :, 2] = rgb[:, :, 0]
    # Preserve-colours contract (matches the renderer): 8-bit passes through,
    # wider integers are bit-shifted down to display range.
    peak = max((float(np.ma.max(plane)) if plane.count() else 0.0) for plane in planes)
    if peak > 255.0:
        rgb *= 255.0 / 65535.0
    valid = np.ones((height, width), dtype=bool)
    for index in range(bands):
        valid &= ~np.ma.getmaskarray(planes[index])
    rgba = np.concatenate(
        [np.clip(rgb, 0.0, 255.0).astype(np.uint8), np.where(valid, 255, 0).astype(np.uint8)[..., None]],
        axis=2,
    )
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    print(json.dumps({"ok": True, "png": base64.b64encode(buffer.getvalue()).decode("ascii")}))
except Exception as exc:
    print(json.dumps({"ok": False, "detail": str(exc)}))
    raise SystemExit(0)
"#;

/// Decode a raster into a downsampled PNG for the Solid viewport's drape.
pub fn raster_preview_png(python: &str, path: &str, width: u32) -> Result<Vec<u8>, String> {
    let (stdout, stderr) = run_script(
        python,
        RASTER_PREVIEW_SCRIPT,
        &[path.to_owned(), width.to_string()],
        SCRIPT_TIMEOUT,
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the raster preview produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the raster preview returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the raster could not be decoded")
            .to_owned());
    }
    let png = value
        .get("png")
        .and_then(|png| png.as_str())
        .ok_or_else(|| "the raster preview returned no image".to_owned())?;
    base64::engine::general_purpose::STANDARD
        .decode(png)
        .map_err(|error| format!("invalid preview image: {error}"))
}

/// Read a vector layer's geometry (feature lines and labelled points) in
/// the DEM's own frame, normalised to the terrain preview's 0..1 lattice.
///
/// The one job the still and the viewport cannot share over stdin: the
/// viewport needs the geometry itself, not pixels. `label_field` empty reads
/// only the lines; the field is the labels layer's attribute.
pub fn vector_geometry(
    python: &str,
    dem_path: &str,
    vector_path: &str,
    label_field: &str,
    legend_field: &str,
) -> Result<serde_json::Value, String> {
    run_vector_preview(
        python,
        "vector",
        dem_path,
        &[
            vector_path.to_owned(),
            label_field.to_owned(),
            legend_field.to_owned(),
        ],
    )
}

/// Read a polygon layer's rings (exteriors and holes) in the DEM's own
/// frame, normalised to the preview's 0..1 lattice — the still's reader,
/// never a second GeoPackage parser.
pub fn polygon_geometry(
    python: &str,
    dem_path: &str,
    vector_path: &str,
) -> Result<serde_json::Value, String> {
    run_vector_preview(python, "polygon", dem_path, &[vector_path.to_owned()])
}

/// Read the graticule's grid (and, when asked, its coordinate annotations)
/// for the DEM's CRS and bounds — the engine's own `generate_graticule`.
pub fn graticule_geometry(
    python: &str,
    dem_path: &str,
    interval_deg: f64,
    with_labels: bool,
) -> Result<serde_json::Value, String> {
    run_vector_preview(
        python,
        "graticule",
        dem_path,
        &[
            interval_deg.to_string(),
            if with_labels { "1" } else { "0" }.to_owned(),
        ],
    )
}

fn run_vector_preview(
    python: &str,
    kind: &str,
    dem_path: &str,
    tail: &[String],
) -> Result<serde_json::Value, String> {
    let service = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("studio_render.py");
    let mut args = vec![
        kind.to_owned(),
        dem_path.to_owned(),
        service.to_string_lossy().into_owned(),
    ];
    args.extend_from_slice(tail);
    let (stdout, stderr) = run_script(python, VECTOR_PREVIEW_SCRIPT, &args, SCRIPT_TIMEOUT)?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the vector preview produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the vector preview returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the vector preview failed")
            .to_owned());
    }
    Ok(value)
}

/// The engine's dataset registry, read once per probe — offline-safe: with
/// the repo checkout present, every sample dataset resolves locally (nothing
/// is downloaded until `dataset_fetch` is called and no local copy exists).
const DATASET_LIST_SCRIPT: &str = r#"
import json
try:
    import forge3d.datasets as datasets
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine's dataset registry is unavailable: %s" % (exc,)}))
    raise SystemExit(0)
try:
    info = datasets.dataset_info()
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the dataset registry failed: %s" % (exc,)}))
    raise SystemExit(0)
rows = []
for name, meta in sorted(info.items()):
    rows.append({
        "name": name,
        "kind": meta.get("kind"),
        "filename": meta.get("filename"),
        "description": meta.get("description"),
        "path": meta.get("path"),
    })
print(json.dumps({"ok": True, "datasets": rows}))
"#;

/// Resolve (and fetch, if need be) a dataset by registry name. The engine's
/// own fetch path does the hash verification; a missing `pooch` fails here
/// with the engine's install instructions.
const DATASET_FETCH_SCRIPT: &str = r#"
import json, sys
name = sys.argv[1]
try:
    import forge3d.datasets as datasets
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine's dataset registry is unavailable: %s" % (exc,)}))
    raise SystemExit(0)
try:
    path = str(datasets.fetch(name))
except Exception as exc:
    print(json.dumps({"ok": False, "detail": str(exc)}))
    raise SystemExit(0)
print(json.dumps({"ok": True, "path": path}))
"#;

/// One entry of the engine's dataset registry.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetInfo {
    pub name: String,
    pub kind: Option<String>,
    pub filename: Option<String>,
    pub description: Option<String>,
    /// The local path when a file resolves — otherwise nothing would be
    /// downloaded until the fetch is invoked.
    pub path: Option<String>,
}

/// The registry, as the shell sees it. Errors here are non-fatal: the
/// gallery just does not appear.
pub fn dataset_list(python: &str) -> Result<Vec<DatasetInfo>, String> {
    let (stdout, stderr) = run_script(python, DATASET_LIST_SCRIPT, &[], SCRIPT_TIMEOUT)?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the dataset registry produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the dataset registry returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the dataset registry failed")
            .to_owned());
    }
    let rows = value
        .get("datasets")
        .and_then(|rows| rows.as_array())
        .ok_or_else(|| "the dataset registry returned no list".to_owned())?;
    Ok(rows
        .iter()
        .filter_map(|row| {
            Some(DatasetInfo {
                name: row.get("name")?.as_str()?.to_owned(),
                kind: row.get("kind").and_then(|v| v.as_str()).map(String::from),
                filename: row
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                description: row
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                path: row.get("path").and_then(|v| v.as_str()).map(String::from),
            })
        })
        .collect())
}

/// Resolve a dataset to a local file, fetching it if need be. The name must
/// come from the registry — no arbitrary engine calls.
pub fn dataset_fetch(python: &str, name: &str) -> Result<String, String> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return Err(format!("invalid dataset name {name:?}"));
    }
    let known = dataset_list(python)?;
    if !known.iter().any(|entry| entry.name == name) {
        return Err(format!("unknown dataset {name:?}"));
    }
    let (stdout, stderr) = run_script(
        python,
        DATASET_FETCH_SCRIPT,
        &[name.to_owned()],
        Duration::from_secs(300),
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the dataset fetch produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the dataset fetch returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the dataset fetch failed")
            .to_owned());
    }
    value
        .get("path")
        .and_then(|path| path.as_str())
        .map(str::to_owned)
        .ok_or_else(|| "the dataset fetch returned no path".to_owned())
}

/// Probe a remote DEM (a COG, or any file the range-reader can open).
/// One HEAD + header reads — nothing downloads beyond what the engine needs
/// to answer the question "what is this file?".
const REMOTE_COG_INFO_SCRIPT: &str = r#"
import json, sys
url = sys.argv[1]
try:
    import forge3d.cog as cog
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine's COG reader is unavailable: %s" % (exc,)}))
    raise SystemExit(0)
try:
    ds = cog.open_cog(url)
    levels = []
    total_tiles = 0
    for level in range(ds.overview_count):
        info = ds.ifd_info(level)
        total_tiles += info.tile_count
        levels.append({
            "level": level,
            "width": info.width,
            "height": info.height,
            "tileWidth": info.tile_width,
            "tileHeight": info.tile_height,
            "tilesAcross": info.tiles_across,
            "tilesDown": info.tiles_down,
            "tiles": info.tile_count,
            "compression": info.compression,
            "bitsPerSample": info.bits_per_sample,
        })
    b = ds.bounds
    print(json.dumps({
        "ok": True,
        "url": url,
        "overviewCount": ds.overview_count,
        "bounds": list(b) if b else None,
        "streamable": True,
        "levels": levels,
        "totalTiles": total_tiles,
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "detail": str(exc)}))
"#;

/// A remote DEM's preview, read through the engine's own range reader.
///
/// The ladder's first rung: the coarsest overview level that still carries
/// the requested resolution, assembled from the engine's tiles — no whole
/// file is downloaded, and the bytes that arrive are the overview's. The
/// payload matches the local `terrainPreview` shape, so the studio's own
/// elevation ramp draws both.
const REMOTE_COG_PREVIEW_SCRIPT: &str = r#"
import json, sys
url = sys.argv[1]
resolution = max(8, min(256, int(sys.argv[2])))
try:
    import numpy as np
    import forge3d.cog as cog
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "the engine is unavailable: %s" % (exc,)}))
    raise SystemExit(0)
try:
    ds = cog.open_cog(url)
    # `overview_count` counts the levels, full resolution included: the
    # engine's own probe reads `ifd_info(level)` for level in that range.
    levels = max(1, int(ds.overview_count))
    # The coarsest level that still has `resolution` pixels on both sides
    # (the levels run fine to coarse, so the last acceptable one wins); if
    # none does, the coarsest available.
    level = levels - 1
    for candidate in range(levels):
        info = ds.ifd_info(candidate)
        if info.width >= resolution and info.height >= resolution:
            level = candidate
    info = ds.ifd_info(level)
    height, width = int(info.height), int(info.width)
    grid = np.zeros((height, width), dtype="float64")
    for ty in range(int(info.tiles_down)):
        for tx in range(int(info.tiles_across)):
            tile = np.asarray(ds.read_tile(tx, ty, level), dtype="float64")
            if tile.ndim != 2 or tile.size == 0:
                continue
            th, tw = tile.shape[:2]
            y0 = ty * int(info.tile_height)
            x0 = tx * int(info.tile_width)
            y1 = min(height, y0 + th)
            x1 = min(width, x0 + tw)
            if y1 <= y0 or x1 <= x0:
                continue
            grid[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
    valid = np.isfinite(grid)
    if not valid.any():
        print(json.dumps({"ok": False, "detail": "the remote DEM has no valid samples"}))
        raise SystemExit(0)
    values = grid[valid]
    low, high = np.percentile(values, [0.01, 99.99])
    ys = np.linspace(0, height, resolution + 1).astype(int)
    xs = np.linspace(0, width, resolution + 1).astype(int)
    out = np.empty((resolution, resolution), dtype="float64")
    for j in range(resolution):
        for i in range(resolution):
            block = grid[ys[j]:max(ys[j] + 1, ys[j + 1]), xs[i]:max(xs[i] + 1, xs[i + 1])]
            good = block[np.isfinite(block)]
            out[j, i] = float(np.clip(good.mean(), low, high)) if good.size else float(low)
    print(json.dumps({
        "ok": True,
        "width": width,
        "height": height,
        "resolution": resolution,
        "minElevation": float(low),
        "maxElevation": float(high),
        "heights": [round(float(value), 2) for value in out.ravel()],
        "level": int(level),
        "levelWidth": width,
        "levelHeight": height,
        "overviewCount": int(ds.overview_count),
        "bounds": list(ds.bounds) if ds.bounds else None,
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "detail": str(exc)}))
"#;

/// The remote DEM's preview, read through the engine's range reader.
pub fn remote_cog_preview(
    python: &str,
    url: &str,
    resolution: u32,
) -> Result<serde_json::Value, String> {
    validate_remote_url(url)?;
    let (stdout, stderr) = run_script(
        python,
        REMOTE_COG_PREVIEW_SCRIPT,
        &[url.to_owned(), resolution.to_string()],
        SCRIPT_TIMEOUT,
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the remote preview produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the remote preview returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the remote preview failed")
            .to_owned());
    }
    Ok(value)
}

/// Download a remote DEM to the studio cache. `url` must be http(s) or
/// `file://`; the engine already validated it via `remote_cog_info`.
/// Remote-DEM probe: what `remote_cog_info` returns to the gallery dialog.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteCogInfo {
    pub url: String,
    pub overview_count: u32,
    pub bounds: Option<[f64; 4]>,
    pub streamable: bool,
    pub levels: Vec<RemoteCogLevel>,
    pub total_tiles: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteCogLevel {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
    pub tiles: u32,
    pub compression: u32,
    pub bits_per_sample: u32,
}

fn validate_remote_url(url: &str) -> Result<(), String> {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("file://"))
    {
        return Err("a remote DEM URL must be http://, https:// or file://".to_owned());
    }
    if url.len() > 2048 || url.chars().any(|c| c.is_control()) {
        return Err("the remote DEM URL is malformed".to_owned());
    }
    Ok(())
}

/// Probe a remote DEM with the engine's COG reader (native range reads; one
/// HEAD request, no pixel data fetched).
pub fn remote_cog_info(python: &str, url: &str) -> Result<RemoteCogInfo, String> {
    validate_remote_url(url)?;
    let (stdout, stderr) = run_script(
        python,
        REMOTE_COG_INFO_SCRIPT,
        &[url.to_owned()],
        SCRIPT_TIMEOUT,
    )?;
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("the remote probe produced no output: {}", stderr.trim()))?;
    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the remote probe returned invalid JSON: {error}"))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
        return Err(value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the remote probe failed")
            .to_owned());
    }
    let levels = value
        .get("levels")
        .and_then(|rows| rows.as_array())
        .cloned()
        .unwrap_or_default();
    let parsed = levels
        .iter()
        .map(|row| RemoteCogLevel {
            level: row.get("level").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            width: row.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            height: row.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tile_width: row.get("tileWidth").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tile_height: row.get("tileHeight").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tiles_across: row.get("tilesAcross").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tiles_down: row.get("tilesDown").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tiles: row.get("tiles").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            compression: row.get("compression").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            bits_per_sample: row
                .get("bitsPerSample")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        })
        .collect();
    let bounds = value
        .get("bounds")
        .and_then(|b| b.as_array())
        .and_then(|rows| {
            let nums: Vec<f64> = rows.iter().filter_map(|v| v.as_f64()).collect();
            (nums.len() == 4).then(|| [nums[0], nums[1], nums[2], nums[3]])
        });
    Ok(RemoteCogInfo {
        url: url.to_owned(),
        overview_count: value
            .get("overviewCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        bounds,
        streamable: value
            .get("streamable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        levels: parsed,
        total_tiles: value
            .get("totalTiles")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    })
}

/// Labelled interpreter candidates, most specific first.
///
/// The label is what `--engine-check` prints, so both the stranger and the CI
/// read the same deterministic shape.
fn labeled_interpreter_candidates() -> Vec<(String, String)> {
    let mut candidates: Vec<(String, String)> = Vec::new();

    if let Some(python) = std::env::var_os("FORGE3D_STUDIO_PYTHON") {
        candidates.push((
            "FORGE3D_STUDIO_PYTHON".to_owned(),
            python.to_string_lossy().into_owned(),
        ));
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().take(6).map(Path::to_path_buf));
    }
    if let Ok(current) = std::env::current_dir() {
        roots.extend(current.ancestors().take(6).map(Path::to_path_buf));
    }
    for root in roots {
        let venv = root.join(".venv").join("bin").join("python");
        if venv.is_file() {
            let candidate = venv.to_string_lossy().into_owned();
            candidates.push(("repo .venv".to_owned(), candidate));
            break;
        }
    }

    candidates.push(("PATH python3".to_owned(), "python3".to_owned()));
    candidates.push(("PATH python".to_owned(), "python".to_owned()));
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|(_, path)| seen.insert(path.clone()));
    candidates
}

/// Interpreter candidates, most specific first.
fn interpreter_candidates() -> Vec<String> {
    labeled_interpreter_candidates()
        .into_iter()
        .map(|(_, path)| path)
        .collect()
}

/// Directories to add to `PYTHONPATH` so a bare interpreter can import the
/// checkout's package without an editable install.
fn python_path_entries() -> Option<String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().take(6).map(Path::to_path_buf));
    }
    if let Ok(current) = std::env::current_dir() {
        roots.extend(current.ancestors().take(6).map(Path::to_path_buf));
    }
    for root in roots {
        let package = root.join("python").join("forge3d").join("__init__.py");
        if package.is_file() {
            return Some(root.join("python").to_string_lossy().into_owned());
        }
    }
    None
}

fn probe_interpreter(python: &str) -> Result<EngineStatus, String> {
    let mut command = Command::new(python);
    command.arg("-c").arg(PROBE_SCRIPT);
    if let Some(path) = python_path_entries() {
        // Inside a repo checkout the probe tests the checkout's engine; the
        // caller's own PYTHONPATH entries keep the tail so a plain interpreter
        // with forge3d only on the user's path still probes correctly.
        let merged = merge_python_path(path, std::env::var("PYTHONPATH").ok());
        command.env("PYTHONPATH", merged);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not run {python}: {error}"))?;

    let deadline = Instant::now() + PROBE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    return Err(format!(
                        "{python} did not answer the engine probe within {}s",
                        PROBE_TIMEOUT.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(format!("could not wait for {python}: {error}")),
        }
    }

    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut stdout)
            .map_err(|error| format!("could not read the probe output: {error}"))?;
    }
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }

    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| {
            let detail = stderr.trim();
            if detail.is_empty() {
                format!("{python} produced no probe output")
            } else {
                format!("{python}: {}", first_line(detail))
            }
        })?;

    let value: serde_json::Value = serde_json::from_str(line.trim())
        .map_err(|error| format!("the engine probe returned invalid JSON: {error}"))?;

    Ok(engine_status_from_probe(python, &value))
}

/// A stranger-readable probe result for `--engine-check`.
///
/// One line per candidate, then a one-line verdict. Exit codes: 0 found,
/// 1 not found. Deliberately no auto-install, no sudo, and no bare-python
/// fallback that cannot see forge3d.
pub fn engine_check() -> i32 {
    // The candidates list is never empty: `PATH python3` and `PATH python`
    // are always the last entries. (Duplicates, e.g. an env var pointing at
    // `python3` itself, are dropped by the retain in
    // `labeled_interpreter_candidates`.)
    let candidates = labeled_interpreter_candidates();
    println!("{:<24}interpreter  result", "candidate");
    let mut found: Option<(String, String)> = None;
    for (label, candidate) in &candidates {
        match probe_interpreter(candidate) {
            Ok(status) if status.available => {
                let version = status.package_version.as_deref().unwrap_or("unversioned");
                println!("{:<24}{}  ok (forge3d {version})", label, candidate);
                if found.is_none() {
                    found = Some((label.clone(), version.to_owned()));
                }
            }
            Ok(status) => {
                let detail = status.detail.lines().next().unwrap_or("unavailable");
                println!("{:<24}{}  fail: {}", label, candidate, detail);
            }
            Err(error) => {
                let detail = error.lines().next().unwrap_or("probe failed");
                println!("{:<24}{}  fail: {}", label, candidate, detail);
            }
        }
    }
    match found {
        Some((label, version)) => {
            println!("\nengine: found via {label} (forge3d {version})");
            0
        }
        None => {
            println!("\nengine: not found");
            print_engine_check_remedies();
            1
        }
    }
}

fn print_engine_check_remedies() {
    println!("remedies:");
    println!("  pip install --user forge3d");
    println!("  or point the shell at a working interpreter:");
    println!("  FORGE3D_STUDIO_PYTHON=/path/to/python forge3d-studio-shell --engine-check");
}

/// The PYTHONPATH value the probe hands to the interpreter: the checkout's
/// `python/` goes first (dev convenience), the caller's entries keep the tail.
/// An empty user value is treated as unset.
fn merge_python_path(path: String, existing: Option<String>) -> String {
    match existing {
        Some(existing) if !existing.trim().is_empty() => format!("{path}:{existing}"),
        _ => path,
    }
}

/// Map one probe JSON object onto [`EngineStatus`].
///
/// Split out from the process plumbing so the wire shape (including a missing
/// `features` key from an older engine) is testable without Python.
fn engine_status_from_probe(python: &str, value: &serde_json::Value) -> EngineStatus {
    EngineStatus {
        available: value.get("ok").and_then(|ok| ok.as_bool()).unwrap_or(false),
        python: python.to_owned(),
        package_version: value
            .get("version")
            .and_then(|version| version.as_str())
            .map(str::to_owned),
        viewer_binary: value
            .get("viewer")
            .and_then(|viewer| viewer.as_str())
            .map(str::to_owned),
        backend: value
            .get("backend")
            .and_then(|backend| backend.as_str())
            .map(str::to_owned),
        adapter: value
            .get("adapter")
            .and_then(|adapter| adapter.as_str())
            .map(str::to_owned),
        denoise: value.get("denoise").and_then(|denoise| denoise.as_bool()),
        software: value
            .get("software")
            .and_then(|software| software.as_bool()),
        features: value.get("features").and_then(|features| {
            features.as_array().map(|list| {
                list.iter()
                    .filter_map(|name| name.as_str().map(str::to_owned))
                    .collect()
            })
        }),
        detail: value
            .get("detail")
            .and_then(|detail| detail.as_str())
            .unwrap_or("the engine probe returned no detail")
            .to_owned(),
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or_default().to_owned()
}

#[cfg(test)]
mod tests {
    use super::{merge_python_path, parse_solar_reply};

    #[test]
    fn merge_python_path_keeps_the_repo_first_and_the_user_tail() {
        assert_eq!(
            merge_python_path("/repo/python".to_owned(), Some("/user/libs".to_owned())),
            "/repo/python:/user/libs"
        );
    }

    #[test]
    fn merge_python_path_treats_empty_user_value_as_unset() {
        assert_eq!(
            merge_python_path("/repo/python".to_owned(), Some("   ".to_owned())),
            "/repo/python"
        );
        assert_eq!(
            merge_python_path("/repo/python".to_owned(), None),
            "/repo/python"
        );
    }

    #[test]
    fn solar_reply_parses_and_reports_engine_errors() {
        let ok = r#"{"ok":true,"azimuth":197.04,"elevation":66.08,"direction":[-0.118,0.914,-0.387],"daytime":true}"#;
        let position = parse_solar_reply(ok).expect("valid reply");
        assert!((position.azimuth_deg - 197.04).abs() < 1e-9);
        assert!((position.elevation_deg - 66.08).abs() < 1e-9);
        assert!((position.direction[2] + 0.387).abs() < 1e-5);
        assert!(position.daytime);

        let error =
            parse_solar_reply(r#"{"ok":false,"detail":"bad datetime"}"#).expect_err("engine error");
        assert!(error.contains("bad datetime"), "{error}");

        let missing = parse_solar_reply(r#"{"ok":true}"#).expect_err("incomplete reply");
        assert!(missing.contains("azimuth"), "{missing}");
    }
}

#[test]
fn probe_json_maps_the_engine_facts() {
    let value: serde_json::Value = serde_json::from_str(
        r#"{
                "ok": true,
                "version": "1.47.1",
                "viewer": "/bin/viewer",
                "detail": "engine ready",
                "backend": "Vulkan",
                "adapter": "NVIDIA GeForce RTX 5060 Laptop GPU",
                "software": false,
                "denoise": true,
                "features": ["texture_binding_array", "timestamp_query"]
            }"#,
    )
    .expect("probe JSON parses");

    let status = engine_status_from_probe("/venv/bin/python", &value);
    assert!(status.available);
    assert_eq!(status.python, "/venv/bin/python");
    assert_eq!(status.package_version.as_deref(), Some("1.47.1"));
    assert_eq!(status.viewer_binary.as_deref(), Some("/bin/viewer"));
    assert_eq!(status.backend.as_deref(), Some("Vulkan"));
    assert_eq!(
        status.adapter.as_deref(),
        Some("NVIDIA GeForce RTX 5060 Laptop GPU")
    );
    assert_eq!(status.denoise, Some(true));
    assert_eq!(status.software, Some(false));
    assert_eq!(
        status.features,
        Some(vec![
            "texture_binding_array".to_owned(),
            "timestamp_query".to_owned()
        ])
    );
}

#[test]
fn an_older_probe_without_features_is_not_an_error() {
    let value: serde_json::Value =
        serde_json::from_str(r#"{"ok": true, "version": "1.40.0", "detail": "engine ready"}"#)
            .expect("probe JSON parses");
    let status = engine_status_from_probe("python3", &value);
    assert_eq!(status.features, None);
    assert_eq!(status.software, None);
    assert_eq!(status.denoise, None);
    assert!(status.available);
}

#[test]
fn a_failed_probe_stays_unavailable() {
    let value: serde_json::Value =
        serde_json::from_str(r#"{"ok": false, "detail": "import forge3d failed: no module"}"#)
            .expect("probe JSON parses");
    let status = engine_status_from_probe("python3", &value);
    assert!(!status.available);
    assert_eq!(status.detail, "import forge3d failed: no module");
}
