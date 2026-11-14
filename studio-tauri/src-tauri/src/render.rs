//! Embedded offscreen renderer for the Studio viewport's Rendered mode.
//!
//! This is the plan's "Rendered" viewport, not the separate viewer window: a
//! persistent Python process drives the certified `render_terrain_pbr_pom`
//! pipeline and returns PNG frames to the shell, which the viewport displays.
//!
//! Same engine, same shaders, same settings as the golden-tested renders; what
//! differs is the transport (frames over stdio) and the fact that there is no
//! window to manage.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::json;

const SERVICE_SCRIPT: &str = include_str!("studio_render.py");
const READY_TIMEOUT: Duration = Duration::from_secs(240);
pub const FRAME_TIMEOUT: Duration = Duration::from_secs(180);
/// A viewport frame is a preview: a wedged one must not block the queue behind
/// it for the still timeout.
pub const PREVIEW_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_LOG: usize = 200;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderFrame {
    pub width: u32,
    pub height: u32,
    pub ms: f64,
    /// Which engine path made the frame: `concordia` (the quality surface,
    /// with the screen-space effects) or `plain` (the AOV renderer).
    pub render_path: String,
    /// Whether the frame is temporally filtered (the engine's TAA): the
    /// quality surface's frame is the converged one, not byte-stable.
    pub temporal: bool,
    /// PNG bytes, base64: the viewport shows it directly.
    pub png: String,
    /// Where a still was written, when it was written to disk.
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderStatus {
    pub running: bool,
    pub dem: Option<String>,
    pub diagnostics: Vec<String>,
}

struct Session {
    child: Child,
    stdin: ChildStdin,
    dem: String,
}

/// State the reader thread and the waiting command share.
///
/// The condition variable is the point: a render waits for its own reply
/// instead of waking every 40 ms, so a stopped render returns at once and a
/// finished one comes back without a poll interval of latency.
struct Shared {
    replies: Mutex<HashMap<u64, serde_json::Value>>,
    /// Readiness for the current session, tagged with its generation so a
    /// reader from a previous session cannot satisfy (or fail) a new one.
    ready: Mutex<Option<(u64, Result<(), String>)>>,
    log: Mutex<VecDeque<String>>,
    changed: Condvar,
}

impl Shared {
    fn new() -> Self {
        Self {
            replies: Mutex::new(HashMap::new()),
            ready: Mutex::new(None),
            log: Mutex::new(VecDeque::new()),
            changed: Condvar::new(),
        }
    }
}

pub struct RenderState {
    session: Mutex<Option<Session>>,
    shared: Arc<Shared>,
    /// One request at a time. The service is a single-threaded stdio loop, and
    /// a viewport frame must not race a still (or a session restart).
    request: Mutex<()>,
    /// Bumped by [`RenderState::stop`]; a waiter compares it and returns
    /// instead of waiting out its own timeout.
    generation: AtomicU64,
    next_id: AtomicU64,
}

impl Default for RenderState {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderState {
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
            shared: Arc::new(Shared::new()),
            request: Mutex::new(()),
            generation: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn status(&self) -> RenderStatus {
        let (running, dem) = match self.session.lock() {
            Ok(guard) => (
                guard.is_some(),
                guard.as_ref().map(|session| session.dem.clone()),
            ),
            Err(_) => (false, None),
        };
        RenderStatus {
            running,
            dem,
            diagnostics: self
                .shared
                .log
                .lock()
                .map(|log| log.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    /// Render one frame for `dem`, starting the service on first use.
    ///
    /// Blocks. Every caller runs on a worker thread (the commands are `async`
    /// and hand the body to the runtime), never on the UI thread.
    pub fn render(
        &self,
        python: &str,
        dem: &str,
        request: serde_json::Value,
        timeout: Duration,
    ) -> Result<RenderFrame, String> {
        // Serialise first, then start the clock: a viewport frame queued behind
        // a still must not spend the still's time against its own deadline.
        let _request = self.request.lock().map_err(lock_error)?;
        self.ensure_session(python, dem)?;
        let generation = self.generation.load(Ordering::SeqCst);

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut payload = request;
        if let Some(object) = payload.as_object_mut() {
            object.insert("id".into(), json!(id));
        }

        {
            let mut guard = self.session.lock().map_err(lock_error)?;
            let session = guard
                .as_mut()
                .ok_or_else(|| "the renderer is not running".to_owned())?;
            writeln!(session.stdin, "{payload}")
                .and_then(|_| session.stdin.flush())
                .map_err(|error| format!("lost the renderer: {error}"))?;
        }

        let reply = match wait_for_reply(&self.shared, &self.generation, generation, id, timeout) {
            WaitOutcome::Reply(reply) => reply,
            WaitOutcome::Stopped => return Err("the render was stopped".to_owned()),
            WaitOutcome::TimedOut => {
                // A hung render must not wedge the session.
                self.stop();
                return Err("the renderer did not answer in time".to_owned());
            }
        };

        if reply.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
            return Err(reply
                .get("error")
                .and_then(|error| error.as_str())
                .unwrap_or("the renderer reported a failure")
                .to_owned());
        }
        Ok(RenderFrame {
            width: reply.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            height: reply.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            ms: reply.get("ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            render_path: reply
                .get("renderPath")
                .and_then(|path| path.as_str())
                .unwrap_or("plain")
                .to_owned(),
            temporal: reply
                .get("temporal")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            png: reply
                .get("png")
                .and_then(|png| png.as_str())
                .unwrap_or_default()
                .to_owned(),
            path: None,
        })
    }

    /// Stop the renderer. A command waiting on a reply wakes immediately
    /// (`"the render was stopped"`) instead of waiting out its frame timeout.
    pub fn stop(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.shared.changed.notify_all();
        let Ok(mut guard) = self.session.lock() else {
            return;
        };
        if let Some(mut session) = guard.take() {
            let _ = writeln!(session.stdin, "{}", json!({"id": 0, "cmd": "close"}));
            let _ = session.stdin.flush();
            // The service exits on `close` immediately; do not spend a second
            // of a worker's time waiting for a graceful exit that will not come.
            for _ in 0..10 {
                match session.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                    Err(_) => break,
                }
            }
            let _ = session.child.kill();
            let _ = session.child.wait();
        }
        push_log(&self.shared, "renderer stopped".to_owned());
    }

    fn ensure_session(&self, python: &str, dem: &str) -> Result<(), String> {
        {
            let guard = self.session.lock().map_err(lock_error)?;
            if let Some(session) = guard.as_ref() {
                if session.dem == dem {
                    return Ok(());
                }
            }
        }
        self.stop();
        if let Ok(mut ready) = self.shared.ready.lock() {
            *ready = None;
        }

        let directory = std::env::temp_dir().join("forge3d-studio");
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
        let script = directory.join("studio_render.py");
        std::fs::write(&script, SERVICE_SCRIPT)
            .map_err(|error| format!("could not write {}: {error}", script.display()))?;

        push_log(&self.shared, format!("starting renderer for {dem}"));
        let mut child = Command::new(python)
            .arg("-u")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("could not start {python}: {error}"))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "the renderer has no stdin".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "the renderer has no stdout".to_owned())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "the renderer has no stderr".to_owned())?;

        writeln!(stdin, "{}", json!({ "dem": dem, "zScale": 1.0 }))
            .and_then(|_| stdin.flush())
            .map_err(|error| format!("could not configure the renderer: {error}"))?;

        {
            let mut guard = self.session.lock().map_err(lock_error)?;
            *guard = Some(Session {
                child,
                stdin,
                dem: dem.to_owned(),
            });
        }

        // `stop()` above bumped the generation; readiness belongs to this one.
        let generation = self.generation.load(Ordering::SeqCst);
        spawn_reader(stdout, Arc::clone(&self.shared), generation);
        spawn_stderr(stderr, Arc::clone(&self.shared));

        let mut ready = self.shared.ready.lock().map_err(lock_error)?;
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Some((tag, result)) = ready.take() {
                if tag == generation {
                    return result;
                }
                // A stale reader finished after its session was replaced.
            }
            if self.generation.load(Ordering::SeqCst) != generation {
                return Err("the renderer was stopped".to_owned());
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                drop(ready);
                self.stop();
                return Err("the renderer did not become ready in time".to_owned());
            };
            let (guard, _) = self
                .shared
                .changed
                .wait_timeout(ready, remaining)
                .map_err(lock_error)?;
            ready = guard;
        }
    }
}

/// What a waiting command saw.
#[derive(Debug)]
enum WaitOutcome {
    Reply(serde_json::Value),
    Stopped,
    TimedOut,
}

/// Block until `id`'s reply arrives, the renderer is stopped, or `timeout`
/// elapses. Split out so the wake/stop/timeout behaviour is testable without a
/// Python process.
fn wait_for_reply(
    shared: &Shared,
    generation: &AtomicU64,
    expected: u64,
    id: u64,
    timeout: Duration,
) -> WaitOutcome {
    let deadline = Instant::now() + timeout;
    let mut replies = match shared.replies.lock() {
        Ok(replies) => replies,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        if let Some(reply) = replies.remove(&id) {
            return WaitOutcome::Reply(reply);
        }
        if generation.load(Ordering::SeqCst) != expected {
            return WaitOutcome::Stopped;
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return WaitOutcome::TimedOut;
        };
        replies = match shared.changed.wait_timeout(replies, remaining) {
            Ok((guard, _)) => guard,
            Err(poisoned) => poisoned.into_inner().0,
        };
    }
}

fn lock_error<T>(_: T) -> String {
    "the renderer is unavailable after an internal error".to_owned()
}

fn push_log(shared: &Shared, line: String) {
    if let Ok(mut log) = shared.log.lock() {
        log.push_back(line);
        while log.len() > MAX_LOG {
            log.pop_front();
        }
    }
}

fn spawn_reader(stdout: impl std::io::Read + Send + 'static, shared: Arc<Shared>, generation: u64) {
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let trimmed = line.trim();
            let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
                push_log(&shared, trimmed.to_owned());
                continue;
            };
            match value.get("event").and_then(|event| event.as_str()) {
                Some("ready") => {
                    push_log(&shared, "renderer ready".to_owned());
                    if let Ok(mut state) = shared.ready.lock() {
                        *state = Some((generation, Ok(())));
                    }
                    shared.changed.notify_all();
                }
                Some("error") => {
                    let message = value
                        .get("message")
                        .and_then(|message| message.as_str())
                        .unwrap_or("the renderer failed to start")
                        .to_owned();
                    push_log(&shared, message.clone());
                    if let Ok(mut state) = shared.ready.lock() {
                        *state = Some((generation, Err(message)));
                    }
                    shared.changed.notify_all();
                }
                Some("warning") => {
                    // A warning is part of the protocol (a skipped overlay, a
                    // label layer the engine rejected): it belongs in the log
                    // the studio shows, not in the void.
                    let message = value
                        .get("message")
                        .and_then(|message| message.as_str())
                        .unwrap_or("the renderer reported a warning")
                        .to_owned();
                    push_log(&shared, message);
                }
                _ => {
                    if let Some(id) = value.get("id").and_then(|id| id.as_u64()) {
                        if let Ok(mut replies) = shared.replies.lock() {
                            replies.insert(id, value);
                        }
                        shared.changed.notify_all();
                    }
                }
            }
        }
        if let Ok(mut state) = shared.ready.lock() {
            if state.is_none() {
                *state = Some((
                    generation,
                    Err("the renderer exited unexpectedly".to_owned()),
                ));
            }
        }
        shared.changed.notify_all();
        push_log(&shared, "renderer closed".to_owned());
    });
}

fn spawn_stderr(stderr: impl std::io::Read + Send + 'static, shared: Arc<Shared>) {
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            push_log(&shared, format!("stderr: {line}"));
        }
    });
}

/// Camera override from the Studio viewport, so Rendered frames look like what
/// the solid view shows instead of a different framing. The type lives in the
/// core so the animation export and the viewport agree on it by construction.
pub use studio_core::workspace::RenderView;

/// Everything the shell needs for one frame, derived from the project.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderRequest {
    pub width: u32,
    pub height: u32,
    pub dem: String,
    pub phi_deg: f64,
    pub theta_deg: f64,
    pub radius: f64,
    pub target: [f64; 3],
    pub fov_deg: f64,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub sun_intensity: f64,
    /// Linear RGB colour of the scene's directional light.
    pub sun_color: [f64; 3],
    pub z_scale: f64,
    /// The map's layer stack, in draw order (the first is drawn first).
    pub cartography: Vec<studio_core::workspace::CartographyLayerHandoff>,
    pub exposure: f64,
    pub shadows: bool,
    /// Render the sky background; the project can turn it off for Rendered.
    pub sky: bool,
    /// Environment/IBL intensity multiplier (0 = environment light off).
    pub ibl_intensity: f64,
    /// Radiance `.hdr` environment for image-based lighting, when the project
    /// sets one. `None` makes the service use its built-in studio sky.
    pub environment: Option<String>,
    /// Backdrop brightness of the environment map in Rendered frames, 0..=1.
    /// Independent of `ibl_intensity`, which scales the lighting.
    pub environment_background_intensity: f64,
    /// Final-quality AOV export (see `studio_core::RenderSettings::still_aov`).
    pub aov_albedo: bool,
    pub aov_normal: bool,
    pub aov_depth: bool,
    /// Final-quality OIDN denoise of the beauty pass.
    pub still_denoise: bool,
    /// Shadow filter (`PCF` | `PCSS`) and shadow-map edge length in texels.
    pub shadow_technique: String,
    pub shadow_resolution: u32,
    /// Terrain self-shadowing from the heightfield (the engine's sun
    /// visibility pass): how far each ray searches, and how many steps.
    pub sun_visibility: bool,
    pub sun_visibility_distance: f64,
    pub sun_visibility_steps: u32,
    /// Analytic sky model token (`hosek` | `preetham` | `aether`) and its
    /// backdrop parameters. Used only when the project has no HDRI.
    pub sky_model: String,
    pub sky_turbidity: f32,
    pub sky_ground_albedo: f32,
    pub sky_exposure: f32,
    /// Water on the terrain: the level and the slope threshold the render
    /// service turns into the engine's water mask.
    pub water_enabled: bool,
    pub water_level: f32,
    pub water_slope_threshold: f32,
    pub water_roughness: f32,
    /// The cloud shadows on the terrain (the engine's own cloud field).
    pub clouds_enabled: bool,
    pub clouds_shadows_enabled: bool,
    pub clouds_coverage: f32,
    pub clouds_density: f32,
    pub clouds_shadow_strength: f32,
    pub clouds_quality: String,
    /// Atmospheric fog on the terrain.
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_reach: f32,
    pub fog_inscatter: [f32; 3],
    /// Bloom and vignette on the finished frame.
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_softness: f32,
    pub bloom_intensity: f32,
    pub vignette: f32,
}

impl RenderRequest {
    pub fn with_settings(mut self, settings: &studio_core::RenderSettings) -> Self {
        self.exposure = f64::from(settings.exposure);
        self.shadows = settings.shadows;
        self.sky = settings.sky;
        self.ibl_intensity = f64::from(settings.ibl_intensity);
        self.environment_background_intensity =
            f64::from(settings.environment_background_intensity);
        self.aov_albedo = settings.still_aov.albedo;
        self.aov_normal = settings.still_aov.normal;
        self.aov_depth = settings.still_aov.depth;
        self.still_denoise = settings.still_denoise;
        self.shadow_technique = settings.shadow_technique.engine_name().to_owned();
        self.shadow_resolution = settings.shadow_resolution;
        self.sun_visibility = settings.sun_visibility;
        self.sun_visibility_distance = settings.sun_visibility_distance;
        self.sun_visibility_steps = settings.sun_visibility_steps;
        self.sky_model = settings.sky_model.token().to_owned();
        self.sky_turbidity = settings.sky_turbidity;
        self.sky_ground_albedo = settings.sky_ground_albedo;
        self.sky_exposure = settings.sky_exposure;
        self.water_enabled = settings.water.enabled;
        self.water_level = settings.water.level;
        self.water_slope_threshold = settings.water.slope_threshold;
        self.water_roughness = settings.water.roughness;
        self.clouds_enabled = settings.clouds.enabled;
        self.clouds_shadows_enabled = settings.clouds.shadows_enabled;
        self.clouds_coverage = settings.clouds.coverage;
        self.clouds_density = settings.clouds.density;
        self.clouds_shadow_strength = settings.clouds.shadow_strength;
        self.clouds_quality = settings.clouds.quality.clone();
        self.fog_enabled = settings.fog.enabled;
        self.fog_density = settings.fog.density;
        self.fog_reach = settings.fog.reach;
        self.fog_inscatter = settings.fog.inscatter;
        self.bloom_enabled = settings.bloom.enabled;
        self.bloom_threshold = settings.bloom.threshold;
        self.bloom_softness = settings.bloom.softness;
        self.bloom_intensity = settings.bloom.intensity;
        self.vignette = settings.vignette;
        self
    }

    /// Whether the map's stack has a plate: the plate is a final artifact, so
    /// its cut-out asks for the depth AOV it borrows.
    pub fn has_plate(&self) -> bool {
        self.cartography.iter().any(|layer| {
            layer.visible
                && matches!(
                    layer.kind,
                    studio_core::workspace::CartographyLayerHandoffKind::Plate { .. }
                )
        })
    }

    pub fn with_view(mut self, view: Option<&RenderView>) -> Self {
        if let Some(view) = view {
            if view.radius.is_finite() && view.radius > 0.0 {
                self.phi_deg = view.phi_deg;
                self.theta_deg = view.theta_deg;
                self.radius = view.radius;
            }
            if let Some(target) = view.target {
                if target.iter().all(|component| component.is_finite()) {
                    self.target = target;
                }
            }
            if let Some(fov_deg) = view.fov_deg {
                if fov_deg.is_finite() && (1.0..=179.0).contains(&fov_deg) {
                    self.fov_deg = fov_deg;
                }
            }
        }
        self
    }

    pub fn to_payload(&self, quality: &str, output: Option<&str>) -> serde_json::Value {
        let final_quality = quality == "final";
        json!({
            "width": self.width,
            "height": self.height,
            "zScale": self.z_scale,
            "camera": {
                "phiDeg": self.phi_deg,
                "thetaDeg": self.theta_deg,
                "radius": self.radius,
                "target": self.target,
                "fovDeg": self.fov_deg,
            },
            "sun": {
                "azimuthDeg": self.sun_azimuth_deg,
                "elevationDeg": self.sun_elevation_deg,
                "intensity": self.sun_intensity,
                "color": self.sun_color,
            },
            "quality": {

                // Quality presets: interactive previews skip POM and any AO;
                // finals honour the project's settings. Shadows and sky follow
                // the project in both.
                "shadows": self.shadows,
                // Supersampled by the engine: the terrain's per-cell normals
                // alias into sparkles at close camera ranges, and the engine's
                // own 2x render + blit removes them — measured 31 -> 2 sparkle
                // pixels on a preview and 44 -> 0 on a 1920x1080 still, for
                // +0.06 s, with the output (and the AOVs) still at the
                // requested size. Previews and finals both do it, so the still
                // matches the preview.
                // ... but only up to a pixel budget: a "4x viewport" request
                // is already 4x the pixels, and 2x on top of that is 8x
                // internally, which the GPU will refuse or crawl through.
                // The threshold is the *internal* budget: 8 Mpx requested is
                // a 16 Mpx target at 2x, which the GPU takes; above that the
                // frame keeps its pixels and loses the antialiasing.
                "sky": self.sky,
                // Final stills (and the frames of a sequence) may ask for AOVs
                // and an OIDN pass; previews never pay for either.
                "aov": {
                    "albedo": final_quality && self.aov_albedo,
                    "normal": final_quality && self.aov_normal,
                    "depth": final_quality && (self.aov_depth || self.has_plate()),
                },
                "denoise": final_quality && self.still_denoise,
                "shadowTechnique": self.shadow_technique,
                "shadowResolution": self.shadow_resolution,
                // Terrain self-shadowing. Cheap enough for previews (one ray
                // march per pixel), so it is not final-gated like SSAO.
                "sunVisibility": {
                    "enabled": self.sun_visibility,
                    "distance": self.sun_visibility_distance,
                    "steps": self.sun_visibility_steps,
                },
                "skyModel": self.sky_model,
                "skyTurbidity": self.sky_turbidity,
                "skyGroundAlbedo": self.sky_ground_albedo,
                "skyExposure": self.sky_exposure,
                "water": {
                    "enabled": self.water_enabled,
                    "level": self.water_level,
                    "slopeThreshold": self.water_slope_threshold,
                    "roughness": self.water_roughness,
                },
                "clouds": {
                    "enabled": self.clouds_enabled,
                    "shadowsEnabled": self.clouds_shadows_enabled,
                    "coverage": self.clouds_coverage,
                    "density": self.clouds_density,
                    "shadowStrength": self.clouds_shadow_strength,
                    "quality": self.clouds_quality,
                },
                "fog": {
                    "enabled": self.fog_enabled,
                    "density": self.fog_density,
                    "reach": self.fog_reach,
                    "inscatter": self.fog_inscatter,
                },
                "bloom": {
                    "enabled": self.bloom_enabled,
                    "threshold": self.bloom_threshold,
                    "softness": self.bloom_softness,
                    "intensity": self.bloom_intensity,
                },
                "vignette": self.vignette,
            },
            // The map's layer stack, in draw order. The service draws it
            // back to front: the relief, the ramp or the imagery, the vector
            // layers and their labels, the plate.
            "cartography": self.cartography,
            "exposure": self.exposure,
            "shadows": self.shadows,
            "iblIntensity": self.ibl_intensity,
            "environment": self.environment,
            "environmentBackgroundIntensity": self.environment_background_intensity,
            "output": output,
        })
    }

    /// Resolve the renderer's camera from a handoff: the renderer measures the
    /// DEM in its own metres with Z up, so only the direction transfers.
    pub fn from_handoff(
        handoff: &studio_core::Handoff,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let terrain = handoff
            .terrain
            .as_ref()
            .ok_or_else(|| "the scene has no terrain to render".to_owned())?;
        let camera = handoff
            .terrain_camera
            .ok_or_else(|| "the scene has no camera".to_owned())?;

        let span = terrain
            .world_span_xz
            .ok_or_else(|| "the DEM footprint is unknown (open the project again)".to_owned())?;

        Ok(Self {
            width,
            height,
            dem: terrain.path.clone(),
            phi_deg: f64::from(camera.phi_deg),
            theta_deg: f64::from(camera.theta_deg),
            // Use the radius the camera's distance implies, so the frame matches
            // the viewport; fall back to the viewer's framing only when the
            // project cannot say (no camera distance available).
            radius: camera.radius.unwrap_or_else(|| span[0].max(span[1]) * 1.5),
            // The terrain mesh is vertically centred on Z=0 in the render world
            // (the shader centres it "so the camera at origin can see it"), so
            // the fallback aim is the mesh centre — aiming at absolute
            // mid-elevation framed empty sky for high-altitude DEMs.
            target: [0.0, 0.0, 0.0],
            fov_deg: f64::from(camera.fov_deg),
            sun_azimuth_deg: handoff
                .sun
                .map(|sun| f64::from(sun.azimuth_deg))
                .unwrap_or(135.0),
            sun_elevation_deg: handoff
                .sun
                .map(|sun| f64::from(sun.elevation_deg))
                .unwrap_or(35.0),
            sun_intensity: handoff
                .sun
                .map(|sun| f64::from(sun.intensity))
                .unwrap_or(3.0),
            sun_color: handoff
                .sun
                .map(|sun| {
                    [
                        f64::from(sun.color[0]),
                        f64::from(sun.color[1]),
                        f64::from(sun.color[2]),
                    ]
                })
                .unwrap_or([1.0, 0.97, 0.92]),
            z_scale: f64::from(terrain.z_scale),
            // The map's layer stack, in draw order.
            cartography: handoff.cartography.clone(),
            exposure: 1.35,
            shadows: true,
            sky: true,
            ibl_intensity: 1.0,
            environment: handoff
                .environment
                .as_ref()
                .map(|environment| environment.path.clone()),
            environment_background_intensity: 1.0,
            // Off until `with_settings` copies the project's choice in.
            aov_albedo: false,
            aov_normal: false,
            aov_depth: false,
            still_denoise: false,
            shadow_technique: "PCF".to_owned(),
            shadow_resolution: 4096,
            sun_visibility: false,
            sun_visibility_distance: 2000.0,
            sun_visibility_steps: 32,
            sky_model: "hosek".to_owned(),
            sky_turbidity: 2.0,
            sky_ground_albedo: 0.3,
            sky_exposure: 0.08,
            water_enabled: false,
            water_level: 0.0,
            water_slope_threshold: 0.18,
            water_roughness: 0.35,
            clouds_enabled: false,
            clouds_shadows_enabled: true,
            clouds_coverage: 0.5,
            clouds_density: 0.5,
            clouds_shadow_strength: 0.35,
            clouds_quality: "medium".to_owned(),
            fog_enabled: false,
            fog_density: 0.05,
            fog_reach: 0.5,
            fog_inscatter: [1.0, 1.0, 1.0],
            bloom_enabled: false,
            bloom_threshold: 0.8,
            bloom_softness: 0.2,
            bloom_intensity: 0.3,
            vignette: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RenderRequest {
        RenderRequest {
            width: 960,
            height: 600,
            dem: "/data/dem.tif".to_owned(),
            phi_deg: 310.0,
            theta_deg: 68.0,
            radius: 6100.0,
            target: [0.0, 0.0, 39.5],
            fov_deg: 45.0,
            sun_azimuth_deg: 60.0,
            sun_elevation_deg: 50.0,
            sun_intensity: 3.0,
            sun_color: [1.0, 0.97, 0.92],
            z_scale: 1.0,
            cartography: Vec::new(),
            exposure: 1.35,
            shadows: true,
            sky: true,
            ibl_intensity: 1.0,
            environment: None,
            environment_background_intensity: 1.0,
            aov_albedo: false,
            aov_normal: false,
            aov_depth: false,
            still_denoise: false,
            shadow_technique: "PCF".to_owned(),
            shadow_resolution: 4096,
            sun_visibility: false,
            sun_visibility_distance: 2000.0,
            sun_visibility_steps: 32,
            sky_model: "hosek".to_owned(),
            sky_turbidity: 2.0,
            sky_ground_albedo: 0.3,
            sky_exposure: 0.08,
            water_enabled: false,
            water_level: 0.0,
            water_slope_threshold: 0.18,
            water_roughness: 0.35,
            clouds_enabled: false,
            clouds_shadows_enabled: true,
            clouds_coverage: 0.5,
            clouds_density: 0.5,
            clouds_shadow_strength: 0.35,
            clouds_quality: "medium".to_owned(),
            fog_enabled: false,
            fog_density: 0.5,
            fog_reach: 0.5,
            fog_inscatter: [1.0, 1.0, 1.0],
            bloom_enabled: false,
            bloom_threshold: 0.8,
            bloom_softness: 0.2,
            bloom_intensity: 0.3,
            vignette: 0.0,
        }
    }

    #[test]
    fn payload_carries_the_project_hdri() {
        let mut frame = request();
        frame.environment = Some("/hdri/studio_sunrise.hdr".to_owned());
        let payload = frame.to_payload("final", Some("/out/frame.png"));
        assert_eq!(payload["environment"], "/hdri/studio_sunrise.hdr");
        assert_eq!(payload["iblIntensity"], 1.0);
        assert_eq!(payload["environmentBackgroundIntensity"], 1.0);
        assert_eq!(payload["output"], "/out/frame.png");

        // The built-in sky is requested by *not* sending a path.
        let preview = request().to_payload("preview", None);
        assert!(preview["environment"].is_null());
        assert!(preview["quality"].get("ambientOcclusion").is_none());
        assert_eq!(preview["quality"]["aov"]["albedo"], false);
        assert_eq!(preview["quality"]["denoise"], false);
    }

    #[test]
    fn payload_emits_aov_and_denoise_only_for_finals() {
        let settings = studio_core::RenderSettings {
            still_aov: studio_core::StillAov {
                albedo: true,
                normal: false,
                depth: true,
            },
            still_denoise: true,
            ..studio_core::RenderSettings::default()
        };
        let enabled = request().with_settings(&settings);
        let final_payload = enabled.to_payload("final", Some("/out/still.png"));
        assert_eq!(final_payload["quality"]["aov"]["albedo"], true);
        assert_eq!(final_payload["quality"]["aov"]["normal"], false);
        assert_eq!(final_payload["quality"]["aov"]["depth"], true);
        assert_eq!(final_payload["quality"]["denoise"], true);

        // A preview drops both — the interactive frame pays neither.
        let preview = enabled.to_payload("preview", None);
        assert_eq!(preview["quality"]["aov"]["albedo"], false);
        assert_eq!(preview["quality"]["aov"]["depth"], false);
        assert_eq!(preview["quality"]["denoise"], false);
    }

    #[test]
    fn payload_carries_the_shadow_filter_and_map_size() {
        let settings = studio_core::RenderSettings {
            shadow_technique: studio_core::ShadowTechnique::Pcss,
            shadow_resolution: 2048,
            ..studio_core::RenderSettings::default()
        };
        let payload = request()
            .with_settings(&settings)
            .to_payload("preview", None);
        assert_eq!(payload["quality"]["shadowTechnique"], "PCSS");
        assert_eq!(payload["quality"]["shadowResolution"], 2048);
    }

    #[test]
    fn payload_carries_the_water_settings() {
        let settings = studio_core::RenderSettings {
            water: studio_core::WaterSettings {
                enabled: true,
                level: 1650.0,
                slope_threshold: 0.05,
                roughness: 0.2,
            },
            ..studio_core::RenderSettings::default()
        };
        let payload = request()
            .with_settings(&settings)
            .to_payload("preview", None);
        assert_eq!(payload["quality"]["water"]["enabled"], true);
        // The payload carries f32 values widened to f64, so the expected side
        // is the f32 widened the same way.
        assert_eq!(payload["quality"]["water"]["level"], f64::from(1650.0_f32));
        assert_eq!(
            payload["quality"]["water"]["slopeThreshold"],
            f64::from(0.05_f32)
        );
        assert_eq!(payload["quality"]["water"]["roughness"], f64::from(0.2_f32));
    }

    #[test]
    fn previews_and_finals_both_go_through_the_recipe() {
        // The still path is the engine's MapScene recipe in both modes; the
        // recipe converges its temporal history instead of supersampling, so
        // there is no scale key on the payload at all — the engine never
        // refuses to supersample because supersampling is gone.
        assert!(request().to_payload("preview", None)["quality"]
            .get("renderScale")
            .is_none());
        assert!(request().to_payload("final", None)["quality"]
            .get("renderScale")
            .is_none());
    }

    #[test]
    fn payload_carries_the_cartography_stack() {
        use studio_core::workspace::{CartographyLayerHandoff, CartographyLayerHandoffKind};

        let mut payload = request();
        payload.cartography = vec![
            CartographyLayerHandoff {
                id: studio_core::ids::new_id(),
                name: "Terrain relief".to_owned(),
                visible: true,
                kind: CartographyLayerHandoffKind::Relief,
            },
            CartographyLayerHandoff {
                id: studio_core::ids::new_id(),
                name: "Rail network".to_owned(),
                visible: true,
                kind: CartographyLayerHandoffKind::Vector {
                    path: "/data/luxembourg_rail.gpkg".to_owned(),
                    colour: [0.9, 0.29, 0.1, 1.0],
                    width_px: 3.0,
                    legend_field: None,
                },
            },
            CartographyLayerHandoff {
                id: studio_core::ids::new_id(),
                name: "Labels".to_owned(),
                visible: false,
                kind: CartographyLayerHandoffKind::Labels {
                    source: studio_core::ids::new_id(),
                    path: "/data/luxembourg_rail.gpkg".to_owned(),
                    field: "name.en".to_owned(),
                    colour: [0.08, 0.08, 0.1, 1.0],
                    size_px: 14.0,
                },
            },
            CartographyLayerHandoff {
                id: studio_core::ids::new_id(),
                name: "Imagery".to_owned(),
                visible: true,
                kind: CartographyLayerHandoffKind::Imagery {
                    path: "/data/land_cover.tif".to_owned(),
                    opacity: 0.82,
                    preserve_colors: true,
                    despeckle: true,
                },
            },
        ];
        let value = payload.to_payload("preview", None);
        let layers = value["cartography"].as_array().expect("the stack");
        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0]["type"], "relief");
        assert_eq!(layers[1]["type"], "vector");
        assert_eq!(layers[1]["path"], "/data/luxembourg_rail.gpkg");
        assert_eq!(layers[1]["widthPx"], 3.0);
        assert!((layers[1]["colour"][0].as_f64().unwrap() - 0.9).abs() < 1e-6);
        // The label layer travels with its attribute, its colour and its size:
        // the service needs all three to draw the engine's labels.
        assert_eq!(layers[2]["type"], "labels");
        assert_eq!(layers[2]["field"], "name.en");
        assert_eq!(layers[2]["sizePx"], 14.0);
        assert_eq!(layers[2]["visible"], false);
        // The imagery layer's own settings travel with it.
        assert_eq!(layers[3]["type"], "imagery");
        assert_eq!(layers[3]["despeckle"], true);
        assert_eq!(layers[3]["preserveColors"], true);
        // No stack, no layers: the service must see an empty list.
        assert_eq!(
            request().to_payload("preview", None)["cartography"]
                .as_array()
                .expect("an empty stack")
                .len(),
            0
        );
    }

    #[test]
    fn payload_carries_the_sun_visibility() {
        let settings = studio_core::RenderSettings {
            sun_visibility: true,
            sun_visibility_distance: 5000.0,
            sun_visibility_steps: 48,
            ..studio_core::RenderSettings::default()
        };
        let payload = request()
            .with_settings(&settings)
            .to_payload("preview", None);
        // Cheap enough for previews, unlike SSAO: not final-gated.
        assert_eq!(payload["quality"]["sunVisibility"]["enabled"], true);
        assert_eq!(payload["quality"]["sunVisibility"]["distance"], 5000.0);
        assert_eq!(payload["quality"]["sunVisibility"]["steps"], 48);
    }

    /// A plate is a layer now: the stack carries it, and the cut-out still
    /// borrows the renderer's depth pass.
    #[test]
    fn a_plate_layer_borrows_the_depth_pass() {
        use studio_core::workspace::{CartographyLayerHandoff, CartographyLayerHandoffKind};

        let mut payload = request();
        payload.cartography = vec![CartographyLayerHandoff {
            id: studio_core::ids::new_id(),
            name: "Map plate".to_owned(),
            visible: true,
            kind: CartographyLayerHandoffKind::Plate {
                title: "Land cover".to_owned(),
                caption: "Swiss Alps".to_owned(),
                north_arrow: true,
                scale_bar: true,
                legend: true,
            },
        }];
        let final_ = payload.to_payload("final", None);
        assert_eq!(final_["cartography"][0]["type"], "plate");
        assert_eq!(final_["cartography"][0]["title"], "Land cover");
        // The cut-out is the renderer's depth buffer: asking for a plate asks
        // for that AOV even when the project's own AOV boxes are off.
        assert_eq!(final_["quality"]["aov"]["depth"], true);

        // A hidden plate is not drawn, so it does not borrow the depth pass.
        payload.cartography[0].visible = false;
        let preview = payload.to_payload("preview", None);
        assert_eq!(preview["quality"]["aov"]["depth"], false);
    }

    #[test]
    fn the_ramp_layer_carries_its_colormap_and_legend() {
        use studio_core::workspace::{CartographyLayerHandoff, CartographyLayerHandoffKind};

        let mut payload = request();
        payload.cartography = vec![CartographyLayerHandoff {
            id: studio_core::ids::new_id(),
            name: "Elevation ramp".to_owned(),
            visible: true,
            kind: CartographyLayerHandoffKind::ElevationRamp {
                colormap: "viridis".to_owned(),
                strength: 0.8,
                legend: true,
            },
        }];
        let value = payload.to_payload("preview", None);
        assert_eq!(value["cartography"][0]["type"], "elevationRamp");
        assert_eq!(value["cartography"][0]["colormap"], "viridis");
        assert!((value["cartography"][0]["strength"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(value["cartography"][0]["legend"], true);
    }

    #[test]
    fn payload_carries_the_fog_settings() {
        let settings = studio_core::RenderSettings {
            fog: studio_core::FogSettings {
                enabled: true,
                density: 1.25,
                reach: 0.6,
                inscatter: [1.0, 0.9, 0.8],
            },
            ..studio_core::RenderSettings::default()
        };
        let payload = request().with_settings(&settings).to_payload("final", None);
        assert_eq!(payload["quality"]["fog"]["enabled"], true);
        assert_eq!(payload["quality"]["fog"]["density"], f64::from(1.25_f32));
        assert_eq!(payload["quality"]["fog"]["reach"], f64::from(0.6_f32));
        // The tint reaches the engine: it is what makes the haze read as air
        // rather than as a white veil.
        assert_eq!(payload["quality"]["fog"]["inscatter"][0], 1.0);
        assert_eq!(
            payload["quality"]["fog"]["inscatter"][2],
            f64::from(0.8_f32)
        );
    }

    #[test]
    fn payload_carries_the_bloom_and_vignette_settings() {
        let settings = studio_core::RenderSettings {
            bloom: studio_core::BloomSettings {
                enabled: true,
                threshold: 0.6,
                softness: 0.1,
                intensity: 0.5,
            },
            vignette: 0.4,
            ..studio_core::RenderSettings::default()
        };
        let payload = request().with_settings(&settings).to_payload("final", None);
        assert_eq!(payload["quality"]["bloom"]["enabled"], true);
        assert_eq!(payload["quality"]["bloom"]["threshold"], f64::from(0.6_f32));
        assert_eq!(payload["quality"]["bloom"]["softness"], f64::from(0.1_f32));
        assert_eq!(payload["quality"]["bloom"]["intensity"], f64::from(0.5_f32));
        assert_eq!(payload["quality"]["vignette"], f64::from(0.4_f32));
    }

    /// The fixture `studio-tauri/tests/fixtures/lighting-rig.json` is read by
    /// this test and by `src/viewport/lighting.test.ts`. This half asserts the
    /// payload the renderer receives; the node half asserts the numbers the
    /// preview hands to three.js. A drift in either fails, instead of showing
    /// up as "the render looks different from the preview".
    /// End-to-end through the real service: spawn the engine Python, render one
    /// frame over stdio, and stop. Ignored by default because it needs the
    /// checkout's `.venv` and a GPU; run with
    /// `cargo test -p forge3d-studio-shell the_session_renders -- --ignored`.
    #[test]
    #[ignore = "spawns the engine Python session and the GPU"]
    fn the_session_renders_a_frame_and_stops_cleanly() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/bin/python");
        let dem = root.join("assets/tif/Bryce_Canyon_1024.tif");
        if !python.exists() || !dem.exists() {
            eprintln!("skipping: this checkout has no .venv/engine DEM pair");
            return;
        }

        let state = RenderState::new();
        let payload = json!({
            "width": 96,
            "height": 64,
            "camera": {"phiDeg": 310.0, "thetaDeg": 40.0, "radius": 6000.0, "target": [0, 0, 0], "fovDeg": 45.0},
            "sun": {"azimuthDeg": 135.0, "elevationDeg": 40.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
            "quality": {"aov": {}, "sky": true, "skyModel": "hosek", "skyTurbidity": 2.0, "skyGroundAlbedo": 0.3, "skyExposure": 0.08},
            "exposure": 1.35,
            "iblIntensity": 1.0,
            "environmentBackgroundIntensity": 1.0
        });
        let frame = state
            .render(
                python.to_str().expect("utf-8 python path"),
                dem.to_str().expect("utf-8 dem path"),
                payload,
                FRAME_TIMEOUT,
            )
            .expect("one frame from the service");
        // The service clamps each side to at least 64 px.
        assert_eq!((frame.width, frame.height), (96, 64));
        assert!(frame.png.len() > 1000, "a PNG came back");
        assert!(state.status().running);

        state.stop();
        assert!(!state.status().running);
    }

    /// End-to-end proof that the settings added in 5.6 and 5.8 reach pixels
    /// through the real payload — not just that the service's own tests pass.
    /// Ignored by default like the frame test above:
    /// `cargo test -p forge3d-studio-shell the_new_settings -- --ignored`.
    #[test]
    #[ignore = "spawns the engine Python session and the GPU"]
    fn the_new_settings_reach_the_pixels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/bin/python");
        let dem = root.join("assets/tif/Bryce_Canyon_1024.tif");
        if !python.exists() || !dem.exists() {
            eprintln!("skipping: this checkout has no .venv/engine DEM pair");
            return;
        }

        let state = RenderState::new();
        let frame = |quality: serde_json::Value, cartography: serde_json::Value| {
            let payload = json!({
                "width": 160,
                "height": 120,
                "camera": {"phiDeg": 310.0, "thetaDeg": 60.0, "radius": 6000.0, "target": [0, 0, 0], "fovDeg": 45.0},
                "sun": {"azimuthDeg": 135.0, "elevationDeg": 12.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
                "quality": quality,
                "cartography": cartography,
                "exposure": 1.35,
                "iblIntensity": 1.0,
                "environmentBackgroundIntensity": 1.0
            });
            state
                .render(
                    python.to_str().expect("utf-8 python path"),
                    dem.to_str().expect("utf-8 dem path"),
                    payload,
                    FRAME_TIMEOUT,
                )
                .expect("one frame from the service")
        };

        let plain = frame(json!({"sky": false}), json!([]));
        // The quality surface is temporally filtered (the engine's TAA), so
        // its frames are the converged one, not byte-stable. The plain path —
        // the one the AOV finals use — is byte-stable, and that is where the
        // determinism claim is checked.
        assert!(
            plain.temporal,
            "the quality surface reports itself as temporally filtered"
        );
        let stable = frame(json!({"sky": false, "aov": {"depth": true}}), json!([]));
        let stable_again = frame(json!({"sky": false, "aov": {"depth": true}}), json!([]));
        assert_eq!(
            stable.png, stable_again.png,
            "the plain path is byte-stable: the same request twice is the same bytes"
        );

        // The elevation ramp is a layer now: the stack carries it.
        let ramp = frame(
            json!({"sky": false}),
            json!([{
                "id": "ramp",
                "name": "Elevation ramp",
                "visible": true,
                "type": "elevationRamp",
                "colormap": "viridis",
                "strength": 1.0,
                "legend": false
            }]),
        );
        assert_ne!(
            plain.png, ramp.png,
            "the elevation ramp must reach the pixels"
        );

        let sun = frame(
            json!({
                "sky": false,
                "sunVisibility": {"enabled": true, "distance": 4000.0, "steps": 32}
            }),
            json!([]),
        );
        assert_ne!(plain.png, sun.png, "sun visibility must reach the pixels");

        state.stop();
    }

    /// End-to-end proof that a vector layer reaches pixels through the real
    /// payload: the same Luxembourg frame, with and without the rail file.
    #[test]
    #[ignore = "spawns the engine Python session and the GPU"]
    fn the_vector_layer_reaches_the_pixels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/bin/python");
        let dem = root.join("assets/tif/luxembourg_dem.tif");
        let rail = root.join("assets/gpkg/luxembourg_rail.gpkg");
        if !python.exists() || !dem.exists() || !rail.exists() {
            eprintln!("skipping: this checkout has no .venv/engine DEM/rail trio");
            return;
        }

        let state = RenderState::new();
        let frame = |vector: serde_json::Value| {
            let payload = json!({
                "width": 320,
                "height": 240,
                "camera": {"phiDeg": 90.0, "thetaDeg": 45.0, "radius": 60000.0, "target": [0, 0, 0], "fovDeg": 40.0},
                "sun": {"azimuthDeg": 315.0, "elevationDeg": 25.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
                "quality": {"sky": false},
                "exposure": 1.35,
                "iblIntensity": 1.0,
                "environmentBackgroundIntensity": 1.0,
                "cartography": vector
            });
            state
                .render(
                    python.to_str().expect("utf-8 python path"),
                    dem.to_str().expect("utf-8 dem path"),
                    payload,
                    FRAME_TIMEOUT,
                )
                .expect("one frame from the service")
        };

        let plain = frame(json!([]));
        let with_rails = frame(json!([{
            "id": "rails",
            "name": "Rail network",
            "visible": true,
            "type": "vector",
            "path": rail.to_str().expect("utf-8 rail path"),
            "colour": [0.9, 0.29, 0.1, 1.0],
            "widthPx": 3.0
        }]));
        assert_ne!(
            plain.png, with_rails.png,
            "the vector layer must reach the pixels"
        );

        state.stop();
    }

    /// End-to-end proof that the vector labels reach pixels through the real
    /// payload: the same Mount Fuji frame, with and without the places layer.
    #[test]
    #[ignore = "spawns the engine Python session and the GPU"]
    fn the_vector_labels_reach_the_pixels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/bin/python");
        let dem = root.join("assets/tif/Mount_Fuji_30m.tif");
        let places = root.join("assets/gpkg/Mount_Fuji_places.gpkg");
        if !python.exists() || !dem.exists() || !places.exists() {
            eprintln!("skipping: this checkout has no .venv/engine DEM/places trio");
            return;
        }

        let state = RenderState::new();
        let frame = |vector: serde_json::Value| {
            let payload = json!({
                "width": 320,
                "height": 240,
                "camera": {"phiDeg": 90.0, "thetaDeg": 45.0, "radius": 30000.0, "target": [0, 0, 0], "fovDeg": 40.0},
                "sun": {"azimuthDeg": 315.0, "elevationDeg": 35.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
                "quality": {"sky": false},
                "exposure": 1.35,
                "iblIntensity": 1.0,
                "environmentBackgroundIntensity": 1.0,
                "cartography": vector
            });
            state
                .render(
                    python.to_str().expect("utf-8 python path"),
                    dem.to_str().expect("utf-8 dem path"),
                    payload,
                    FRAME_TIMEOUT,
                )
                .expect("one frame from the service")
        };

        let plain = frame(json!([{
            "id": "places",
            "name": "Places",
            "visible": true,
            "type": "vector",
            "path": places.to_str().expect("utf-8 places path"),
            "colour": [0.9, 0.29, 0.1, 1.0],
            "widthPx": 2.0
        }]));
        let labelled = frame(json!([
            {
                "id": "places",
                "name": "Places",
                "visible": true,
                "type": "vector",
                "path": places.to_str().expect("utf-8 places path"),
                "colour": [0.9, 0.29, 0.1, 1.0],
                "widthPx": 2.0
            },
            {
                "id": "labels",
                "name": "Labels",
                "visible": true,
                "type": "labels",
                "source": "places",
                "path": places.to_str().expect("utf-8 places path"),
                "field": "name.en",
                "colour": [0.08, 0.08, 0.1, 1.0],
                "sizePx": 14.0
            }
        ]));
        assert_ne!(
            plain.png, labelled.png,
            "the vector labels must reach the pixels"
        );

        state.stop();
    }

    /// End-to-end proof that the clouds reach the frame through the real
    /// payload: the same Mount Fuji frame, with and without them.
    #[test]
    #[ignore = "spawns the engine Python session and the GPU"]
    fn the_clouds_reach_the_pixels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join(".venv/bin/python");
        let dem = root.join("assets/tif/Mount_Fuji_30m.tif");
        if !python.exists() || !dem.exists() {
            eprintln!("skipping: this checkout has no .venv/engine DEM pair");
            return;
        }

        let state = RenderState::new();
        let frame = |clouds: serde_json::Value| {
            let payload = json!({
                "width": 320,
                "height": 240,
                "camera": {"phiDeg": 75.0, "thetaDeg": 45.0, "radius": 14000.0, "target": [0, 0, 0], "fovDeg": 40.0},
                "sun": {"azimuthDeg": 315.0, "elevationDeg": 28.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
                "quality": {"sky": false, "clouds": clouds},
                "exposure": 1.35,
                "iblIntensity": 1.0,
                "environmentBackgroundIntensity": 1.0
            });
            state
                .render(
                    python.to_str().expect("utf-8 python path"),
                    dem.to_str().expect("utf-8 dem path"),
                    payload,
                    FRAME_TIMEOUT,
                )
                .expect("one frame from the service")
        };

        let plain = frame(json!({"enabled": false}));
        let cloudy = frame(json!({
            "enabled": true,
            "shadowsEnabled": true,
            "coverage": 0.9,
            "density": 0.9,
            "shadowStrength": 1.0,
            "quality": "high"
        }));
        assert_ne!(
            plain.png, cloudy.png,
            "the cloud shadows must reach the pixels"
        );
        assert_eq!(plain.render_path, "concordia");

        state.stop();
    }

    #[test]
    fn reply_wait_returns_the_reply_without_a_poll_interval() {
        let state = RenderState::new();
        state
            .shared
            .replies
            .lock()
            .expect("replies lock")
            .insert(7, json!({"ok": true, "width": 4}));
        match wait_for_reply(
            &state.shared,
            &state.generation,
            0,
            7,
            Duration::from_secs(1),
        ) {
            WaitOutcome::Reply(reply) => assert_eq!(reply["width"], 4),
            other => panic!("expected a reply, got {other:?}"),
        }
    }

    #[test]
    fn stopping_wakes_a_waiting_render_instead_of_waiting_for_the_timeout() {
        let state = Arc::new(RenderState::new());
        let started = Instant::now();
        let waiter = {
            let state = Arc::clone(&state);
            std::thread::spawn(move || {
                wait_for_reply(
                    &state.shared,
                    &state.generation,
                    0,
                    1,
                    Duration::from_secs(60),
                )
            })
        };
        std::thread::sleep(Duration::from_millis(80));
        state.stop();
        let outcome = waiter.join().expect("waiter finished");
        assert!(matches!(outcome, WaitOutcome::Stopped), "got {outcome:?}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "stop must wake the waiter, not wait out the frame timeout"
        );
    }

    #[test]
    fn the_reply_wait_honours_its_timeout() {
        let state = RenderState::new();
        let started = Instant::now();
        let outcome = wait_for_reply(
            &state.shared,
            &state.generation,
            0,
            2,
            Duration::from_millis(60),
        );
        assert!(matches!(outcome, WaitOutcome::TimedOut), "got {outcome:?}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn payload_matches_the_shared_lighting_rig_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/lighting-rig.json"))
                .expect("the lighting-rig fixture parses");
        let project = rig_project(&fixture);
        let handoff = studio_core::workspace::handoff(&project, 1, studio_core::Workspace::Forge3d);
        let payload = RenderRequest::from_handoff(&handoff, 640, 400)
            .expect("the fixture scene has terrain")
            .with_settings(&project.render)
            .to_payload("final", None);

        let expected = &fixture["expected"];
        near_json(
            "sun.azimuthDeg",
            &payload["sun"]["azimuthDeg"],
            &expected["sun"]["azimuthDeg"],
        );
        near_json(
            "sun.elevationDeg",
            &payload["sun"]["elevationDeg"],
            &expected["sun"]["elevationDeg"],
        );
        near_json(
            "sun.intensity",
            &payload["sun"]["intensity"],
            &expected["sun"]["intensity"],
        );
        for (index, channel) in ["r", "g", "b"].iter().enumerate() {
            near_json(
                &format!("sun.color.{channel}"),
                &payload["sun"]["color"][index],
                &expected["sun"]["color"][index],
            );
        }
        near_json("exposure", &payload["exposure"], &expected["exposure"]);
        assert_eq!(
            payload["environment"], expected["environment"],
            "environment must be the project's HDRI path or null"
        );
        assert_eq!(
            payload["quality"]["shadows"],
            expected["shadows"]["enabled"]
        );
        assert_eq!(
            payload["quality"]["shadowTechnique"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase(),
            expected["shadows"]["technique"]
                .as_str()
                .unwrap_or_default(),
            "payload carries the engine name; the fixture stores the model token"
        );
        assert_eq!(
            payload["quality"]["shadowResolution"],
            expected["shadows"]["resolution"]
        );

        // Render-only: the analytic sky draws the backdrop and is not part of
        // the preview rig (the environment map lights the scene).
        let sky = &expected["renderOnly"]["sky"];
        assert_eq!(payload["quality"]["sky"], sky["enabled"]);
        assert_eq!(payload["quality"]["skyModel"], sky["model"]);
        near_json(
            "quality.skyTurbidity",
            &payload["quality"]["skyTurbidity"],
            &sky["turbidity"],
        );
        near_json(
            "quality.skyGroundAlbedo",
            &payload["quality"]["skyGroundAlbedo"],
            &sky["groundAlbedo"],
        );
        near_json(
            "quality.skyExposure",
            &payload["quality"]["skyExposure"],
            &sky["exposure"],
        );
    }

    fn near_json(label: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
        let actual = actual
            .as_f64()
            .unwrap_or_else(|| panic!("{label} is not a number"));
        let expected = expected
            .as_f64()
            .unwrap_or_else(|| panic!("{label} fixture is not a number"));
        assert!(
            (actual - expected).abs() <= 1e-4,
            "{label}: payload {actual} != fixture {expected}"
        );
    }

    /// A starter scene with the fixture's sun and render settings.
    fn rig_project(fixture: &serde_json::Value) -> studio_core::Project {
        let seed = studio_core::TerrainSeed {
            title: None,
            name: "dem.tif".to_owned(),
            path: "/data/dem.tif".to_owned(),
            byte_size: 1,
            dimensions: Some([64, 64]),
            elevation_range: Some([0.0, 100.0]),
            world_origin_xz: Some([0.0, 0.0]),
            world_span_xz: Some([640.0, 640.0]),
            overlay: None,
        };
        let mut project = studio_core::Project::starter("Rig", Some(seed));

        let rotation = &fixture["project"]["sunRotationDeg"];
        let color = &fixture["project"]["sunColor"];
        let sun = project
            .scene
            .objects
            .iter_mut()
            .find(|object| {
                matches!(
                    &object.kind,
                    studio_core::ObjectKind::Light(light)
                        if light.kind == studio_core::LightKind::Directional
                )
            })
            .expect("the starter scene has a sun");
        sun.transform.rotation_deg = studio_core::Vec3::new(
            rotation["x"].as_f64().unwrap_or(0.0) as f32,
            rotation["y"].as_f64().unwrap_or(0.0) as f32,
            rotation["z"].as_f64().unwrap_or(0.0) as f32,
        );
        if let studio_core::ObjectKind::Light(light) = &mut sun.kind {
            light.color = [
                color[0].as_f64().unwrap_or(0.0) as f32,
                color[1].as_f64().unwrap_or(0.0) as f32,
                color[2].as_f64().unwrap_or(0.0) as f32,
            ];
            light.intensity = fixture["project"]["sunIntensity"].as_f64().unwrap_or(0.0) as f32;
        }

        let render = &fixture["project"]["render"];
        project.render.exposure = render["exposure"].as_f64().unwrap_or(1.35) as f32;
        project.render.shadows = render["shadows"].as_bool().unwrap_or(true);
        project.render.shadow_technique = match render["shadowTechnique"].as_str().unwrap_or("pcf")
        {
            "pcss" => studio_core::ShadowTechnique::Pcss,
            _ => studio_core::ShadowTechnique::Pcf,
        };
        project.render.shadow_resolution =
            render["shadowResolution"].as_u64().unwrap_or(4096) as u32;
        project.render.environment = None;
        project
    }

    #[test]
    fn view_override_still_wins_over_the_handoff_camera() {
        let mut frame = request();
        frame.phi_deg = 10.0;
        frame.target = [1.0, 2.0, 3.0];
        let view = RenderView {
            phi_deg: 200.0,
            theta_deg: 55.0,
            radius: 900.0,
            target: Some([4.0, 5.0, 6.0]),
            fov_deg: Some(30.0),
        };
        let frame = frame.with_view(Some(&view));
        assert_eq!(frame.phi_deg, 200.0);
        assert_eq!(frame.theta_deg, 55.0);
        assert_eq!(frame.radius, 900.0);
        assert_eq!(frame.target, [4.0, 5.0, 6.0]);
        assert_eq!(frame.fov_deg, 30.0);
    }
}
