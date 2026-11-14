//! Tauri commands: the only way the UI can touch project state.
//!
//! Every command returns the full [`SessionState`] snapshot rather than a delta.
//! Phase 1 scenes are small and this removes any chance of the window holding a
//! divergent copy of the scene.

use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde::Serialize;
use studio_core::autosave::RecoveryEntry;
use studio_core::command::Command;
use studio_core::editor::{Applied, SessionState};
use studio_core::ids::ObjectId;
use studio_core::project::AssetKind;
use tauri::State;

use studio_core::{
    AutosaveReport, LightKind, ObjectKind, Session, Transform, Workspace, AUTOSAVE_INTERVAL_SECONDS,
};

use crate::forge3d::{EngineStatus, Forge3dState};
use crate::render::{RenderFrame, RenderRequest, RenderState, RenderStatus, RenderView};
use crate::session::AppState;

/// Edit result: what the command did, plus the new snapshot.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EditResult {
    pub applied: Applied,
    pub state: SessionState,
}

/// Autosave health, shown in the status bar instead of failing silently.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AutosaveStatus {
    pub interval_seconds: u64,
    pub last_write_unix_ms: Option<u64>,
    pub last_error: Option<String>,
    pub directory: String,
}

fn lock<'a>(state: &'a State<'_, AppState>) -> Result<std::sync::MutexGuard<'a, Session>, String> {
    state
        .session
        .lock()
        .map_err(|_| "the project session is unavailable after an internal error".to_owned())
}

#[tauri::command]
pub fn project_state(state: State<'_, AppState>) -> Result<Option<SessionState>, String> {
    Ok(lock(&state)?.state())
}

/// Starter scenes offered by the New Project dialog.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenePreset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dem_name: Option<String>,
    pub grid: Option<[u32; 2]>,
    pub elevation_range: Option<[f32; 2]>,
    pub world_span: Option<[f64; 2]>,
    /// False when the DEM is missing from this checkout; the UI disables it.
    pub available: bool,
}

#[tauri::command(async)]
pub fn scene_presets(forge: State<'_, Forge3dState>) -> Vec<ScenePreset> {
    let mut presets = Vec::new();
    for recipe in &crate::forge3d::SCENE_PRESETS {
        let seed = forge.terrain_seed(Some(recipe.id));
        presets.push(ScenePreset {
            id: recipe.id.to_owned(),
            name: recipe.name.to_owned(),
            description: recipe.description.to_owned(),
            dem_name: seed.as_ref().map(|seed| seed.name.clone()),
            grid: seed.as_ref().and_then(|seed| seed.dimensions),
            elevation_range: seed.as_ref().and_then(|seed| seed.elevation_range),
            world_span: seed.as_ref().and_then(|seed| seed.world_span_xz),
            available: seed.is_some(),
        });
    }
    presets
}

#[tauri::command(async)]
pub fn new_project(
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    name: Option<String>,
    preset: Option<String>,
) -> Result<SessionState, String> {
    // A new project starts from the chosen bundled DEM (Bryce by default); the
    // core falls back to the primitive starter when none can be found.
    let seed = match preset.as_deref() {
        // The dialog's "Empty scene" really is empty: no DEM.
        Some("empty") => None,
        other => forge.terrain_seed(other),
    };
    let recipe = forge.scene_recipe(preset.as_deref());
    let mut session = lock(&state)?;
    let mut snapshot = session.new_project_seeded(name, seed);
    if let Some(recipe) = recipe {
        let commands = recipe_commands(recipe, &snapshot);
        if !commands.is_empty() {
            let (_, next) = session
                .apply(studio_core::Command::Batch { commands })
                .map_err(|error| error.to_string())?;
            snapshot = next;
        }
    }
    Ok(snapshot)
}

/// The commands that build a starter scene's cartography.
///
/// The recipe's files become assets and layers (one command each, so the whole
/// recipe is one undo step), and the camera and the sun are placed where the
/// gallery's recipe says.
fn recipe_commands(
    recipe: &crate::forge3d::SceneRecipe,
    snapshot: &SessionState,
) -> Vec<studio_core::Command> {
    use studio_core::cartography::{CartographyLayer, CartographyLayerKind};
    use studio_core::command::Command;
    use studio_core::ids::new_id;
    use studio_core::project::AssetKind;

    let project = &snapshot.project;
    let mut commands = Vec::new();

    // The layers, back to front. A file the checkout does not have is skipped
    // rather than faked: the scene is the data it has.
    let mut vector_layer_id: Option<studio_core::cartography::CartographyLayerId> = None;
    for layer in recipe.layers {
        match layer {
            crate::forge3d::RecipeLayer::Imagery {
                relative,
                opacity,
                preserve_colors,
                despeckle,
            } => {
                let Some(file) = resolve_asset(relative) else {
                    continue;
                };
                let asset_id = new_id();
                commands.push(Command::AddAsset {
                    asset: studio_core::Asset {
                        id: asset_id,
                        name: file.name.clone(),
                        path: file.path.clone(),
                        kind: AssetKind::Orthophoto,
                        byte_size: file.byte_size,
                    },
                });
                commands.push(Command::AddCartographyLayer {
                    layer: CartographyLayer::new(CartographyLayerKind::Imagery {
                        asset: asset_id,
                        opacity: *opacity,
                        preserve_colors: *preserve_colors,
                        despeckle: *despeckle,
                    }),
                    index: None,
                });
            }
            crate::forge3d::RecipeLayer::Vector {
                relative,
                colour,
                width_px,
            } => {
                let Some(file) = resolve_asset(relative) else {
                    continue;
                };
                let asset_id = new_id();
                commands.push(Command::AddAsset {
                    asset: studio_core::Asset {
                        id: asset_id,
                        name: file.name.clone(),
                        path: file.path.clone(),
                        kind: AssetKind::Vector,
                        byte_size: file.byte_size,
                    },
                });
                // The layer's id is minted here so the label layer below can
                // name this one: a label layer without its source is dropped.
                let id = new_id();
                let mut layer = CartographyLayer::new(CartographyLayerKind::Vector {
                    asset: asset_id,
                    colour: *colour,
                    width_px: *width_px,
                });
                layer.id = id;
                commands.push(Command::AddCartographyLayer { layer, index: None });
                vector_layer_id = Some(id);
            }
            crate::forge3d::RecipeLayer::Labels {
                relative,
                field,
                colour,
                size_px,
            } => {
                let Some(file) = resolve_asset(relative) else {
                    continue;
                };
                // The labels name the vector layer's features: the file is the
                // same one, so the asset is reused when it is already there.
                let asset_id = project
                    .assets
                    .iter()
                    .find(|asset| asset.path == file.path)
                    .map(|asset| asset.id)
                    .unwrap_or_else(new_id);
                if project.asset(asset_id).is_none() {
                    commands.push(Command::AddAsset {
                        asset: studio_core::Asset {
                            id: asset_id,
                            name: file.name.clone(),
                            path: file.path.clone(),
                            kind: AssetKind::Vector,
                            byte_size: file.byte_size,
                        },
                    });
                }
                let Some(source) = vector_layer_id else {
                    continue;
                };
                commands.push(Command::AddCartographyLayer {
                    layer: CartographyLayer::new(CartographyLayerKind::Labels {
                        source,
                        field: (*field).to_owned(),
                        colour: *colour,
                        size_px: *size_px,
                    }),
                    index: None,
                });
            }
            crate::forge3d::RecipeLayer::Plate { title, caption } => {
                commands.push(Command::AddCartographyLayer {
                    layer: CartographyLayer::new(CartographyLayerKind::Plate {
                        title: (*title).to_owned(),
                        caption: (*caption).to_owned(),
                        north_arrow: true,
                        scale_bar: true,
                        legend: true,
                    }),
                    index: None,
                });
            }
        }
    }

    // The camera and the sun.
    if let (Some((azimuth, polar)), Some(terrain)) = (recipe.camera, project.terrain_object()) {
        let Some(terrain) = terrain.terrain() else {
            return commands;
        };
        let Some(camera) = project.scene.camera() else {
            return commands;
        };
        let studio_core::ObjectKind::Camera(camera_kind) = camera.kind else {
            return commands;
        };
        let Some((transform, fov)) = studio_core::workspace::map_camera_transform(
            terrain,
            azimuth,
            polar,
            f64::from(camera_kind.fov_deg),
        ) else {
            return commands;
        };
        commands.push(Command::SetTransform {
            id: camera.id,
            transform,
        });
        commands.push(Command::SetCamera {
            id: camera.id,
            camera: studio_core::CameraObject {
                fov_deg: fov,
                ..camera_kind
            },
        });
    }
    if let Some((azimuth, elevation)) = recipe.sun {
        if let Some(sun) = project.scene.objects.iter().find(|object| {
            matches!(&object.kind, studio_core::ObjectKind::Light(light) if light.kind == studio_core::LightKind::Directional)
        }) {
            commands.push(Command::SetTransform {
                id: sun.id,
                transform: studio_core::Transform {
                    translation: sun.transform.translation,
                    rotation_deg: studio_core::workspace::sun_rotation(
                        studio_core::workspace::map_sun_direction(azimuth, elevation),
                    ),
                    scale: sun.transform.scale,
                },
            });
        }
    }
    commands
}

/// A bundled file, resolved against the checkout.
struct BundledFile {
    name: String,
    path: String,
    byte_size: u64,
}

fn resolve_asset(relative: &str) -> Option<BundledFile> {
    let path = crate::forge3d::resolve_bundled(relative)?;
    let metadata = std::fs::metadata(&path).ok()?;
    if metadata.len() == 0 {
        return None;
    }
    Some(BundledFile {
        name: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| relative.to_owned()),
        path: path.to_string_lossy().into_owned(),
        byte_size: metadata.len(),
    })
}

#[tauri::command(async)]
pub fn open_project(state: State<'_, AppState>, path: String) -> Result<SessionState, String> {
    lock(&state)?
        .open_project(Path::new(&path))
        .map_err(|error| error.to_string())
}

#[tauri::command(async)]
pub fn save_project(
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<SessionState, String> {
    let mut session = lock(&state)?;
    let saved = session
        .save_project(path.map(PathBuf::from))
        .map_err(|error| error.to_string())?;

    // The file on disk is now current: snapshots for it are stale.
    if let Some(saved_path) = saved.path.clone() {
        if let Err(error) = session.prune_recovery_for(&state.autosave, Path::new(&saved_path)) {
            log::warn!("failed to prune recovery snapshots: {error}");
        }
    }
    Ok(saved)
}

#[tauri::command]
pub fn close_project(state: State<'_, AppState>) -> Result<Option<SessionState>, String> {
    let mut session = lock(&state)?;
    session.close_project();
    Ok(None)
}

#[tauri::command]
pub fn apply_command(state: State<'_, AppState>, command: Command) -> Result<EditResult, String> {
    let (applied, snapshot) = lock(&state)?
        .apply(command)
        .map_err(|error| error.to_string())?;
    Ok(EditResult {
        applied,
        state: snapshot,
    })
}

#[tauri::command]
pub fn undo(state: State<'_, AppState>) -> Result<EditResult, String> {
    let (applied, snapshot) = lock(&state)?.undo().map_err(|error| error.to_string())?;
    Ok(EditResult {
        applied,
        state: snapshot,
    })
}

#[tauri::command]
pub fn redo(state: State<'_, AppState>) -> Result<EditResult, String> {
    let (applied, snapshot) = lock(&state)?.redo().map_err(|error| error.to_string())?;
    Ok(EditResult {
        applied,
        state: snapshot,
    })
}

#[tauri::command]
pub fn set_selection(
    state: State<'_, AppState>,
    ids: Vec<ObjectId>,
) -> Result<SessionState, String> {
    lock(&state)?
        .set_selection(ids)
        .map_err(|error| error.to_string())
}

/// Move the playhead and evaluate the timeline into the project (view state:
/// not undoable, no autosave). Returns the snapshot so the UI re-renders with
/// the evaluated transforms, plus the clamped frame that was applied.
#[tauri::command]
pub fn set_playhead(state: State<'_, AppState>, frame: f32) -> Result<PlayheadResult, String> {
    let (frame, project_state) = lock(&state)?
        .set_playhead(frame)
        .map_err(|error| error.to_string())?;
    Ok(PlayheadResult {
        frame,
        state: project_state,
    })
}

/// What `set_playhead` returns: the new snapshot plus the clamped frame.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayheadResult {
    pub frame: f32,
    pub state: SessionState,
}

/// Check every referenced asset against the filesystem.
///
/// Read-only: the UI shows "missing"/"changed" next to each asset and offers
/// Relink; nothing here modifies the project. The pure half lives in
/// `studio_core::integrity`, so the rules are testable without disk.
#[tauri::command(async)]
pub fn check_assets(
    state: State<'_, AppState>,
) -> Result<Vec<studio_core::AssetIntegrity>, String> {
    // Clone the snapshot, then drop the guard: statting N assets (possibly on a
    // network mount) must not block every other command.
    let snapshot = lock(&state)?
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;
    Ok(studio_core::asset_integrity(
        &snapshot.project,
        |path| match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => Ok(metadata.len()),
            Ok(_) => Err(format!("{path} is not a file")),
            Err(error) => Err(format!("{path}: {error}")),
        },
    ))
}

/// Place the directional light from the engine's ephemeris.
///
/// One undoable `SetTransform`: the date, place and UTC timestamp are an input
/// to the calculation, not project state. The light's rotation is what both
/// the Solid/Material viewport and the Rendered handoff read.
#[tauri::command(async)]
pub fn set_sun_from_solar(
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    id: ObjectId,
    latitude: f64,
    longitude: f64,
    datetime_utc: String,
) -> Result<SolarResult, String> {
    if !latitude.is_finite() || !(-90.0..=90.0).contains(&latitude) {
        return Err("latitude must be between -90 and 90 degrees".to_owned());
    }
    if !longitude.is_finite() || !(-180.0..=180.0).contains(&longitude) {
        return Err("longitude must be between -180 and 180 degrees".to_owned());
    }
    let datetime_utc = datetime_utc.trim();
    if datetime_utc.is_empty() {
        return Err("a UTC date and time is required".to_owned());
    }

    let engine = forge.probe(false);
    if !engine.available {
        return Err(engine.detail);
    }
    let solar = crate::forge3d::solar_position(&engine.python, latitude, longitude, datetime_utc)?;
    let rotation = studio_core::workspace::sun_rotation(solar.direction);

    let mut session = lock(&state)?;
    let snapshot = session
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;
    let object = snapshot
        .project
        .scene
        .object(id)
        .ok_or_else(|| format!("object {id} does not exist"))?;
    match &object.kind {
        ObjectKind::Light(light) if light.kind == LightKind::Directional => {}
        _ => return Err("solar placement needs a directional light".to_owned()),
    }
    let transform = Transform {
        rotation_deg: rotation,
        ..object.transform
    };
    let (_applied, snapshot) = session
        .apply(Command::SetTransform { id, transform })
        .map_err(|error| error.to_string())?;

    Ok(SolarResult {
        azimuth_deg: solar.azimuth_deg,
        elevation_deg: solar.elevation_deg,
        daytime: solar.daytime,
        rotation_deg: rotation,
        state: snapshot,
    })
}

/// What `set_sun_from_solar` returns: where the sun is and the new snapshot.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SolarResult {
    /// Engine azimuth, north toward east, for display.
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub daytime: bool,
    /// The rotation written to the light (degrees, ZYX).
    pub rotation_deg: studio_core::Vec3,
    pub state: SessionState,
}

#[tauri::command(async)]
pub fn autosave_now(state: State<'_, AppState>) -> Result<Option<AutosaveReport>, String> {
    lock(&state)?
        .autosave_if_needed(&state.autosave)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn autosave_status(state: State<'_, AppState>) -> Result<AutosaveStatus, String> {
    let session = lock(&state)?;
    Ok(AutosaveStatus {
        interval_seconds: AUTOSAVE_INTERVAL_SECONDS,
        last_write_unix_ms: session.last_report().map(|report| report.written_unix_ms),
        last_error: session.last_error().map(str::to_owned),
        directory: state.autosave.dir().display().to_string(),
    })
}

#[tauri::command(async)]
pub fn list_recovery(state: State<'_, AppState>) -> Result<Vec<RecoveryEntry>, String> {
    state.autosave.list().map_err(|error| error.to_string())
}

#[tauri::command(async)]
pub fn restore_recovery(state: State<'_, AppState>, id: String) -> Result<SessionState, String> {
    let project = state
        .autosave
        .load(&id)
        .map_err(|error| error.to_string())?;
    let mut session = lock(&state)?;
    let entries = state.autosave.list().map_err(|error| error.to_string())?;
    let entry = entries
        .into_iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| format!("recovery snapshot {id} no longer exists"))?;
    Ok(session.restore_recovery(&entry, project))
}

#[tauri::command(async)]
pub fn discard_recovery(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state
        .autosave
        .discard(&id)
        .map_err(|error| error.to_string())
}

/// Imports are read whole into memory. Phase 5 streams large assets; until
/// then the cap is what keeps a 2 GB GLB from freezing the window.
const MAX_MESH_BYTES: u64 = 32 * 1024 * 1024;
/// Cap on an HDRI pulled across IPC for the Material preview (base64 adds a
/// third, so this is a real memory bound).
const MAX_HDR_BYTES: u64 = 64 * 1024 * 1024;

/// A mesh file the viewport can parse with GLTFLoader.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshFilePayload {
    pub path: String,
    pub name: String,
    pub kind: AssetKind,
    pub byte_size: u64,
    /// Base64 of the file bytes. IPC carries JSON, and glTF must reach the
    /// webview intact; the cap above bounds the cost.
    pub base64: String,
}

/// What produced a still: the renderer payload (the single source of truth for
/// the framing, sun and settings) plus the provenance only the shell knows.
///
/// Extracted from the command so the recorded shape is testable without a Tauri
/// app or a GPU.
fn still_sidecar(
    request: &RenderRequest,
    handoff: &studio_core::Handoff,
    frame_ms: f64,
) -> serde_json::Value {
    let mut description = request.to_payload("final", None);
    if let Some(map) = description.as_object_mut() {
        map.remove("output");
        map.insert(
            "renderer".into(),
            serde_json::json!("forge3d.mapscene.render"),
        );
        map.insert(
            "stillAov".into(),
            serde_json::json!({
                "albedo": request.aov_albedo,
                "normal": request.aov_normal,
                "depth": request.aov_depth,
            }),
        );
        map.insert(
            "stillDenoise".into(),
            serde_json::json!(request.still_denoise),
        );
        map.insert("frameMs".into(), serde_json::json!(frame_ms));
        map.insert("revision".into(), serde_json::json!(handoff.revision));
        map.insert(
            "handoffKey".into(),
            serde_json::json!(handoff.idempotency_key),
        );
    }
    description
}

/// Read a file for the webview, with the one size ceiling every asset shares.
///
/// Kind validation stays in the command — "this is not a glTF" and "this is not
/// a Radiance .hdr" are different sentences — but the metadata checks, the cap
/// and the base64 encode are identical for all of them.
fn read_file_capped(path: &str, limit: u64, note: &str) -> Result<Vec<u8>, String> {
    let file = PathBuf::from(path);
    let metadata =
        std::fs::metadata(&file).map_err(|error| format!("could not read {path}: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("{path} is not a file"));
    }
    if metadata.len() == 0 {
        return Err(format!("{path} is empty"));
    }
    if metadata.len() > limit {
        return Err(format!(
            "{path} is {:.1} MB; {note} {} MB",
            metadata.len() as f64 / (1024.0 * 1024.0),
            limit / (1024 * 1024)
        ));
    }
    std::fs::read(&file).map_err(|error| format!("could not read {path}: {error}"))
}

/// The IPC shape for "here are the bytes and their identity".
fn payload(path: String, kind: AssetKind, fallback_name: &str, bytes: Vec<u8>) -> MeshFilePayload {
    let name = std::path::Path::new(&path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| fallback_name.to_owned());
    MeshFilePayload {
        path,
        name,
        kind,
        byte_size: bytes.len() as u64,
        base64: base64::engine::general_purpose::STANDARD.encode(&bytes),
    }
}

/// Read a `.gltf`/`.glb` file for import.
///
/// The UI parses the bytes before registering anything, so a corrupt file
/// never reaches the project model.
#[tauri::command(async)]
pub fn read_mesh_file(path: String) -> Result<MeshFilePayload, String> {
    let path = PathBuf::from(path);
    let kind = AssetKind::from_path(&path).ok_or_else(|| {
        format!(
            "{} is not a glTF file; import .gltf or .glb",
            path.display()
        )
    })?;
    // `.hdr` is an asset kind too (the environment map); it is not a mesh.
    if !matches!(kind, AssetKind::Gltf | AssetKind::Glb) {
        return Err(format!(
            "{} is not a glTF file; import .gltf or .glb",
            path.display()
        ));
    }

    let bytes = read_file_capped(
        &path.to_string_lossy(),
        MAX_MESH_BYTES,
        "imports are limited to",
    )?;
    Ok(payload(
        path.to_string_lossy().into_owned(),
        kind,
        "imported",
        bytes,
    ))
}

/// Sample a DEM into a heightfield for the Studio viewport.
///
/// The sample is a preview, not the render: Forge3D still receives the original
/// GeoTIFF. Returns the sampler's JSON so the UI reads exactly what was measured
/// (dimensions, elevation range, extent, heights).
#[tauri::command(async)]
pub fn terrain_preview(
    forge: State<'_, Forge3dState>,
    path: String,
    resolution: Option<u32>,
) -> Result<serde_json::Value, String> {
    let python = forge.probe(false).python;
    crate::forge3d::sample_terrain(&python, &path, resolution.unwrap_or(96))
}

/// A polygon layer: filled areas draped on the terrain, holes and all.
#[tauri::command(async)]
pub fn add_polygon_layer(
    state: State<'_, AppState>,
    path: Option<String>,
    colour: Option<Vec<f32>>,
) -> Result<SessionState, String> {
    let mut session = lock(&state)?;
    let snapshot = session
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;
    let path = path.ok_or_else(|| "no file was chosen".to_owned())?;
    let metadata = std::fs::metadata(&path).map_err(|error| format!("{path}: {error}"))?;
    let existing = snapshot
        .project
        .assets
        .iter()
        .find(|asset| asset.path == path)
        .cloned();
    let asset = existing.unwrap_or_else(|| studio_core::Asset {
        id: studio_core::ids::new_id(),
        name: std::path::Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "polygons.gpkg".to_owned()),
        path: path.clone(),
        kind: studio_core::AssetKind::Vector,
        byte_size: metadata.len(),
    });
    let mut commands = Vec::new();
    if !snapshot
        .project
        .assets
        .iter()
        .any(|entry| entry.id == asset.id)
    {
        commands.push(Command::AddAsset {
            asset: asset.clone(),
        });
    }
    let colour = colour
        .filter(|channels| channels.len() == 4)
        .map(|channels| [channels[0], channels[1], channels[2], channels[3]])
        .unwrap_or([0.2, 0.55, 0.75, 0.7]);
    commands.push(Command::AddCartographyLayer {
        layer: studio_core::CartographyLayer::new(studio_core::CartographyLayerKind::Polygon {
            asset: asset.id,
            colour,
        }),
        index: None,
    });

    session
        .apply(Command::Batch { commands })
        .map_err(|error| error.to_string())?;
    session
        .state()
        .ok_or_else(|| "no project is open".to_owned())
}

/// The vector layer's feature lines (and, with a label field, its labelled
/// points), in the DEM's own frame, normalised to the preview's 0..1 lattice.
/// The still and the preview read the same file through the same reader.
#[tauri::command(async)]
pub fn vector_geometry(
    forge: State<'_, Forge3dState>,
    dem_path: String,
    vector_path: String,
    label_field: Option<String>,
) -> Result<serde_json::Value, String> {
    let python = forge.probe(false).python;
    crate::forge3d::vector_geometry(
        &python,
        &dem_path,
        &vector_path,
        label_field.as_deref().unwrap_or(""),
    )
}

/// The polygon layer's rings (exteriors and holes), in the DEM's own frame,
/// normalised to the preview's 0..1 lattice. Shares the vector reader with
/// the still.
#[tauri::command(async)]
pub fn polygon_geometry(
    forge: State<'_, Forge3dState>,
    dem_path: String,
    vector_path: String,
) -> Result<serde_json::Value, String> {
    let python = forge.probe(false).python;
    crate::forge3d::polygon_geometry(&python, &dem_path, &vector_path)
}

/// The graticule for the DEM's CRS and bounds, from the engine's own
/// `generate_graticule`. Coordinate annotations ride the same call.
#[tauri::command(async)]
pub fn graticule_geometry(
    forge: State<'_, Forge3dState>,
    dem_path: String,
    interval_deg: f64,
    labels: Option<bool>,
) -> Result<serde_json::Value, String> {
    if !interval_deg.is_finite() || interval_deg <= 0.0 {
        return Err("the graticule interval must be a positive number of degrees".to_owned());
    }
    let python = forge.probe(false).python;
    crate::forge3d::graticule_geometry(&python, &dem_path, interval_deg, labels.unwrap_or(true))
}

/// The engine's colormap, sampled into the stops the Material preview paints.
#[tauri::command(async)]
pub fn colormap_stops(
    forge: State<'_, Forge3dState>,
    name: String,
) -> Result<Vec<[f64; 3]>, String> {
    let python = forge.probe(false).python;
    crate::forge3d::colormap_stops(&python, &name)
}

/// Per-parameter capability table: what each material/light/camera parameter
/// actually reaches in the current renderers (studio-core is the one source).
#[tauri::command]
pub fn capability_table() -> studio_core::CapabilityTable {
    studio_core::parameter_capabilities()
}

/// Cheap health check used by the UI on boot; also proves the bridge works.
#[tauri::command]
pub fn shell_info() -> ShellInfo {
    ShellInfo {
        name: "Forge3D Studio".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellInfo {
    pub name: String,
    pub version: String,
}

// ---------------------------------------------------------------------- Forge3D

/// Probe the Forge3D engine without opening anything.
#[tauri::command(async)]
pub fn forge3d_probe(forge: State<'_, Forge3dState>, force: Option<bool>) -> EngineStatus {
    // The probe spawns a Python interpreter, so it must not run on the UI
    // thread. The command is `async`; this line makes the context observable
    // in the shell log (`tokio-runtime-worker`, never `main`).
    log::debug!(
        "engine probe running on thread {:?}",
        std::thread::current().name()
    );
    let status = forge.probe(force.unwrap_or(false));
    log::debug!(
        "engine probe: available={} detail={}",
        status.available,
        status.detail
    );
    status
}

// ---------------------------------------------------------------------- render

/// Render the current scene at `width` × `height` through the embedded
/// Forge3D pipeline (the same one the golden tests certify).
/// Replace the scene terrain's DEM with a raster the user picked; on an empty
/// scene, create the terrain (so the Add menu can start from a DEM too).
#[tauri::command(async)]
pub fn import_terrain_dem(
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    path: String,
) -> Result<SessionState, String> {
    let python = forge.probe(false).python;
    let info = crate::forge3d::sample_terrain(&python, &path, 48)?;
    let metadata = std::fs::metadata(&path).map_err(|error| format!("{path}: {error}"))?;

    let mut session = lock(&state)?;
    let snapshot = session
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;

    let pair_f64 = |key: &str| -> Option<[f64; 2]> {
        let values = info.get(key)?.as_array()?;
        Some([values.first()?.as_f64()?, values.get(1)?.as_f64()?])
    };
    let dimensions = info
        .get("width")
        .and_then(|value| value.as_u64())
        .zip(info.get("height").and_then(|value| value.as_u64()))
        .map(|(width, height)| [width as u32, height as u32]);
    let elevation_range = info
        .get("minElevation")
        .and_then(|value| value.as_f64())
        .zip(info.get("maxElevation").and_then(|value| value.as_f64()))
        .map(|(low, high)| [low as f32, high as f32]);
    let asset = studio_core::Asset {
        id: studio_core::ids::new_id(),
        name: std::path::Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "terrain.tif".to_owned()),
        path: path.clone(),
        kind: studio_core::AssetKind::Dem,
        byte_size: metadata.len(),
    };

    let mut commands = vec![Command::AddAsset {
        asset: asset.clone(),
    }];
    match snapshot.project.terrain_object() {
        Some(terrain) => commands.push(Command::SetTerrainDem {
            id: terrain.id,
            asset: asset.id,
            dimensions,
            elevation_range,
            world_origin_xz: pair_f64("originXz"),
            world_span_xz: pair_f64("spanXz"),
            // The new DEM's own relief decides the vertical scale.
            z_scale: None,
        }),
        None => {
            let mut terrain = studio_core::TerrainObject::new(asset.id);
            if let (Some(dimensions), Some(elevation_range)) = (dimensions, elevation_range) {
                terrain = terrain.with_dem_info(
                    dimensions,
                    elevation_range,
                    pair_f64("originXz"),
                    pair_f64("spanXz"),
                );
            }
            let name = std::path::Path::new(&path)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .filter(|stem| !stem.trim().is_empty())
                .unwrap_or_else(|| "Terrain".to_owned());
            commands.push(Command::CreateObject {
                object: studio_core::SceneObject {
                    id: studio_core::ids::new_id(),
                    name,
                    parent: None,
                    transform: studio_core::Transform::default(),
                    visible: true,
                    locked: false,
                    kind: studio_core::ObjectKind::Terrain(terrain),
                },
                index: Some(0),
            });
        }
    };

    session
        .apply(Command::Batch { commands })
        .map_err(|error| error.to_string())?;
    session
        .state()
        .ok_or_else(|| "no project is open".to_owned())
}

/// Size of a file on disk. The webview cannot stat, and asset registration
/// validates that files are non-empty (`byte_size == 0` reads as "empty file").
#[tauri::command]
pub fn file_size(path: String) -> Result<u64, String> {
    let metadata = std::fs::metadata(&path).map_err(|error| format!("{path}: {error}"))?;
    if metadata.is_dir() {
        return Err(format!("{path} is a directory, not a file"));
    }
    Ok(metadata.len())
}

/// Add an imagery layer to the map's stack.
///
/// The file is registered as an asset and the layer goes on top of the stack,
/// where the author can move it and give it a name.
#[tauri::command]
pub fn add_imagery_layer(
    state: State<'_, AppState>,
    path: Option<String>,
    opacity: Option<f32>,
) -> Result<SessionState, String> {
    let mut session = lock(&state)?;
    let snapshot = session
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;
    let opacity = opacity.unwrap_or(0.82).clamp(0.0, 1.0);
    let path = path.ok_or_else(|| "no file was chosen".to_owned())?;

    let metadata = std::fs::metadata(&path).map_err(|error| format!("{path}: {error}"))?;
    let existing = snapshot
        .project
        .assets
        .iter()
        .find(|asset| asset.path == path)
        .cloned();
    let asset = existing.unwrap_or_else(|| studio_core::Asset {
        id: studio_core::ids::new_id(),
        name: std::path::Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "overlay.tif".to_owned()),
        path: path.clone(),
        kind: studio_core::AssetKind::Orthophoto,
        byte_size: metadata.len(),
    });
    let mut commands = Vec::new();
    if !snapshot
        .project
        .assets
        .iter()
        .any(|entry| entry.id == asset.id)
    {
        commands.push(Command::AddAsset {
            asset: asset.clone(),
        });
    }
    commands.push(Command::AddCartographyLayer {
        layer: studio_core::CartographyLayer::new(studio_core::CartographyLayerKind::Imagery {
            asset: asset.id,
            opacity,
            preserve_colors: true,
            // Inert until the raster is a class raster; the Inspector
            // toggles it on the layer afterwards.
            despeckle: false,
        }),
        index: None,
    });

    session
        .apply(Command::Batch { commands })
        .map_err(|error| error.to_string())?;
    session
        .state()
        .ok_or_else(|| "no project is open".to_owned())
}

/// Add a vector layer to the map's stack.
///
/// The file is a GeoPackage or GeoJSON of lines; the layer keeps its colour and
/// the line's symbol width in pixels. The service reads the file and projects
/// the lines through the engine's own camera, so nothing here parses geometry.
#[tauri::command]
pub fn add_vector_layer(
    state: State<'_, AppState>,
    path: Option<String>,
    colour: Option<Vec<f32>>,
    width_px: Option<f32>,
) -> Result<SessionState, String> {
    let mut session = lock(&state)?;
    let snapshot = session
        .state()
        .ok_or_else(|| "no project is open".to_owned())?;
    let path = path.ok_or_else(|| "no file was chosen".to_owned())?;
    let metadata = std::fs::metadata(&path).map_err(|error| format!("{path}: {error}"))?;
    let existing = snapshot
        .project
        .assets
        .iter()
        .find(|asset| asset.path == path)
        .cloned();
    let asset = existing.unwrap_or_else(|| studio_core::Asset {
        id: studio_core::ids::new_id(),
        name: std::path::Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "vector.gpkg".to_owned()),
        path: path.clone(),
        kind: studio_core::AssetKind::Vector,
        byte_size: metadata.len(),
    });
    let mut commands = Vec::new();
    if !snapshot
        .project
        .assets
        .iter()
        .any(|entry| entry.id == asset.id)
    {
        commands.push(Command::AddAsset {
            asset: asset.clone(),
        });
    }
    let colour = colour
        .filter(|channels| channels.len() == 4)
        .map(|channels| [channels[0], channels[1], channels[2], channels[3]])
        .unwrap_or([0.9, 0.29, 0.1, 1.0]);
    // The layer's label settings are edited in the Inspector (one command
    // each); a new layer starts without labels.
    commands.push(Command::AddCartographyLayer {
        layer: studio_core::CartographyLayer::new(studio_core::CartographyLayerKind::Vector {
            asset: asset.id,
            colour,
            width_px: width_px.unwrap_or(2.0).clamp(1.0, 20.0),
        }),
        index: None,
    });

    session
        .apply(Command::Batch { commands })
        .map_err(|error| error.to_string())?;
    session
        .state()
        .ok_or_else(|| "no project is open".to_owned())
}

/// Read an image asset (PNG/JPEG) for the Solid viewport's textures.
#[tauri::command(async)]
pub fn read_image_asset(path: String) -> Result<MeshFilePayload, String> {
    let lower = path.to_ascii_lowercase();
    if !(lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg")) {
        return Err(format!(
            "{path} is not a PNG/JPEG image; GeoTIFF textures are not yet decoded in the viewport"
        ));
    }
    let bytes = read_file_capped(&path, MAX_HDR_BYTES, "images are limited to")?;
    Ok(payload(
        path,
        studio_core::AssetKind::Orthophoto,
        "texture",
        bytes,
    ))
}

/// Raw bytes of a Radiance `.hdr` environment for the Material preview.
///
/// The webview cannot display RGBE with an `<img>`, so the shell reads the file
/// and the viewport parses it with three's `HDRLoader` before pre-filtering it
/// with PMREM. Rendered frames do not go through here: the engine reads the
/// path itself (`IBL.from_hdr`).
#[tauri::command(async)]
pub fn read_hdr(path: String) -> Result<MeshFilePayload, String> {
    if AssetKind::from_path(std::path::Path::new(&path)) != Some(AssetKind::Hdr) {
        return Err(format!(
            "{path} is not a Radiance .hdr environment — export the HDRI as .hdr"
        ));
    }
    let bytes = read_file_capped(&path, MAX_HDR_BYTES, "the environment map limit is")?;
    Ok(payload(path, AssetKind::Hdr, "environment.hdr", bytes))
}

/// Decoded, downsampled PNG of a raster overlay for the Material preview's
/// terrain drape — the webview cannot decode GeoTIFF, so the engine python
/// does it and hands back a PNG the canvas can display.
#[tauri::command(async)]
pub fn raster_preview(
    forge: State<'_, Forge3dState>,
    path: String,
    width: Option<u32>,
) -> Result<MeshFilePayload, String> {
    let python = forge.probe(false).python;
    let width = width.unwrap_or(2048).clamp(64, 4096);
    let bytes = crate::forge3d::raster_preview_png(&python, &path, width)?;
    Ok(payload(
        path,
        studio_core::AssetKind::Orthophoto,
        "overlay",
        bytes,
    ))
}

/// One render request from a handoff — the single place the payload's camera,
/// sun and settings are assembled, so a still, a viewport frame and an exported
/// sequence frame cannot drift apart.
fn build_request(
    handoff: &studio_core::Handoff,
    settings: &studio_core::RenderSettings,
    width: u32,
    height: u32,
    view: Option<&RenderView>,
) -> Result<RenderRequest, String> {
    Ok(RenderRequest::from_handoff(handoff, width, height)?
        .with_view(view)
        .with_settings(settings))
}

#[tauri::command(async)]
pub fn render_frame(
    render: State<'_, RenderState>,
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    width: Option<u32>,
    height: Option<u32>,
    quality: Option<String>,
    view: Option<RenderView>,
) -> Result<RenderFrame, String> {
    let (handoff, settings) = {
        let session = lock(&state)?;
        let snapshot = session
            .state()
            .ok_or_else(|| "no project is open".to_owned())?;
        (
            studio_core::workspace::handoff(
                &snapshot.project,
                snapshot.revision,
                Workspace::Forge3d,
            ),
            snapshot.project.render,
        )
    };
    let request = build_request(
        &handoff,
        &settings,
        width.unwrap_or(960).clamp(64, 4096),
        height.unwrap_or(600).clamp(64, 4096),
        view.as_ref(),
    )?;
    let python = forge.probe(false).python;
    let quality = quality.unwrap_or_else(|| "final".to_owned());
    render.render(
        &python,
        &request.dem,
        request.to_payload(&quality, None),
        crate::render::PREVIEW_TIMEOUT,
    )
}

/// Render a final still at the project's configured resolution and write it to
/// `path` (the shell picks the path with a save dialog).
#[tauri::command(async)]
pub fn render_still(
    render: State<'_, RenderState>,
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    path: Option<String>,
    view: Option<RenderView>,
) -> Result<RenderFrame, String> {
    let (handoff, settings, project_name, project_path) = {
        let session = lock(&state)?;
        let snapshot = session
            .state()
            .ok_or_else(|| "no project is open".to_owned())?;
        (
            studio_core::workspace::handoff(
                &snapshot.project,
                snapshot.revision,
                Workspace::Forge3d,
            ),
            snapshot.project.render,
            snapshot.project.name.clone(),
            snapshot.path.clone(),
        )
    };

    // Default output lives with the project, under renders/, and carries a
    // sidecar describing exactly what produced it.
    let target = match path {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let base = project_path
                .as_deref()
                .map(std::path::Path::new)
                .and_then(|path| path.parent())
                .map(|dir| dir.join("renders"))
                .unwrap_or_else(|| std::env::temp_dir().join("forge3d-studio/renders"));
            let stamp = studio_core::now_unix_ms();
            let safe: String = project_name
                .chars()
                .map(|ch| {
                    if ch.is_alphanumeric() || ch == '-' || ch == '_' {
                        ch
                    } else {
                        '-'
                    }
                })
                .collect();
            base.join(format!("{safe}-{stamp}.png"))
        }
    };
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }

    let request = build_request(
        &handoff,
        &settings,
        settings.width,
        settings.height,
        view.as_ref(),
    )?;
    let python = forge.probe(false).python;
    let mut frame = render.render(
        &python,
        &request.dem,
        request.to_payload("final", Some(&target.to_string_lossy())),
        crate::render::FRAME_TIMEOUT,
    )?;
    frame.path = Some(target.display().to_string());

    let sidecar = target.with_extension("json");
    let description = still_sidecar(&request, &handoff, frame.ms);
    if let Err(error) = std::fs::write(
        &sidecar,
        serde_json::to_vec_pretty(&description).unwrap_or_default(),
    ) {
        log::warn!("could not write {}: {error}", sidecar.display());
    }

    Ok(frame)
}

#[tauri::command]
pub fn render_status(render: State<'_, RenderState>) -> RenderStatus {
    render.status()
}

/// One frame of an offline sequence export: what the frontend's per-frame
/// loop calls. The shell evaluates the timeline on a **clone** at `frame`,
/// hands the evaluated camera to Forge3D, renders, and writes `output`.
///
/// The clone matters: exporting must not move the session's playhead or
/// dirty the project. Evaluation runs through the same
/// `studio_core::timeline::evaluate` the playhead uses, so the sequence is
/// byte-for-byte the state you scrubbed (the Phase 6 exit criterion).
///
/// One frame per IPC call keeps the command short-lived: the UI gets
/// progress and a working Stop for free, instead of one long blocking call.
// Tauri commands take their IPC fields as arguments; the wire schema is the
// point, so the argument count is not a code smell here.
#[allow(clippy::too_many_arguments)]
#[tauri::command(async)]
pub fn render_animation_frame(
    render: State<'_, RenderState>,
    forge: State<'_, Forge3dState>,
    state: State<'_, AppState>,
    frame: f32,
    width: u32,
    height: u32,
    quality: Option<String>,
    output: String,
) -> Result<WrittenFrame, String> {
    // Evaluate a clone: the session's own playhead stays where the user left it.
    let (mut project, revision) = {
        let session = lock(&state)?;
        let snapshot = session
            .state()
            .ok_or_else(|| "no project is open".to_owned())?;
        (snapshot.project, snapshot.revision)
    };
    let frame = studio_core::timeline::evaluate(&mut project, frame);

    let handoff = studio_core::workspace::handoff(&project, revision, Workspace::Forge3d);
    // The evaluated camera's own view, not the viewport's live orbit: a frame
    // must match a camera-view still of the same object (the export used to
    // fall back to `terrain_camera`, which ignores the camera's rotation and
    // distance). `with_view` also carries the FOV and the terrain-centred aim.
    let view = studio_core::workspace::camera_render_view(&project);
    let request = build_request(
        &handoff,
        &project.render,
        width.clamp(64, 4096),
        height.clamp(64, 4096),
        view.as_ref(),
    )?;

    let target = std::path::PathBuf::from(&output);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }

    let python = forge.probe(false).python;
    let quality = quality.unwrap_or_else(|| "final".to_owned());
    let mut written = render.render(
        &python,
        &request.dem,
        request.to_payload(&quality, Some(&target.to_string_lossy())),
        crate::render::FRAME_TIMEOUT,
    )?;
    written.path = Some(target.display().to_string());

    Ok(WrittenFrame {
        path: target.display().to_string(),
        ms: written.ms,
        evaluated_frame: frame,
    })
}

/// One rendered frame of a sequence: where it landed and how long it took.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WrittenFrame {
    pub path: String,
    pub ms: f64,
    /// The clamped timeline frame actually rendered (differs from the
    /// request only when the request was out of range).
    pub evaluated_frame: f32,
}

#[tauri::command(async)]
pub fn render_stop(render: State<'_, RenderState>) -> RenderStatus {
    render.stop();
    render.status()
}

#[cfg(test)]
mod tests {
    use super::*;
    use studio_core::project::{Project, Workspace};

    #[test]
    fn still_sidecar_carries_the_payload_and_the_provenance() {
        let project = Project::starter(
            "Sidecar",
            Some(studio_core::project::TerrainSeed {
                title: None,
                name: "dem.tif".to_owned(),
                path: "/data/dem.tif".to_owned(),
                byte_size: 1,
                dimensions: Some([64, 64]),
                elevation_range: Some([0.0, 100.0]),
                world_origin_xz: Some([0.0, 0.0]),
                world_span_xz: Some([640.0, 640.0]),
                overlay: None,
            }),
        );
        let handoff = studio_core::workspace::handoff(&project, 4, Workspace::Forge3d);
        let request = RenderRequest::from_handoff(&handoff, 640, 400)
            .expect("the starter scene has terrain")
            .with_settings(&project.render);
        let sidecar = still_sidecar(&request, &handoff, 12.5);

        // One payload producer: these must be the renderer's own keys.
        assert_eq!(sidecar["width"], 640);
        assert_eq!(sidecar["quality"]["aov"]["albedo"], false);
        assert_eq!(sidecar["camera"]["phiDeg"], request.phi_deg);
        assert_eq!(sidecar["sun"]["intensity"], request.sun_intensity);
        assert!(
            sidecar.get("output").is_none(),
            "output is not a render setting"
        );

        // Provenance only the shell knows.
        assert_eq!(sidecar["renderer"], "forge3d.mapscene.render");
        assert_eq!(sidecar["stillDenoise"], false);
        assert_eq!(sidecar["frameMs"], 12.5);
        assert_eq!(sidecar["revision"], 4);
        assert_eq!(sidecar["handoffKey"], handoff.idempotency_key);

        assert_eq!(sidecar["renderer"], "forge3d.mapscene.render");
    }
    /// A starter scene builds its cartography: the gallery's recipes become
    /// layers, and the camera and the sun are placed where the recipe says.
    #[test]
    fn a_starter_scene_carries_its_layers_camera_and_sun() {
        let project = studio_core::Project::starter(
            "Swiss",
            Some(studio_core::TerrainSeed {
                title: Some("Swiss Alps".to_owned()),
                name: "switzerland_dem.tif".to_owned(),
                path: "/data/switzerland_dem.tif".to_owned(),
                byte_size: 1,
                dimensions: Some([1024, 1024]),
                elevation_range: Some([400.0, 4000.0]),
                world_origin_xz: Some([0.0, 0.0]),
                world_span_xz: Some([352_000.0, 226_000.0]),
                overlay: None,
            }),
        );
        let snapshot = SessionState {
            project,
            revision: 0,
            dirty: false,
            path: None,
            can_undo: false,
            can_redo: false,
            undo_label: None,
            redo_label: None,
            history_len: 0,
        };

        let recipe = crate::forge3d::SCENE_PRESETS
            .iter()
            .find(|recipe| recipe.id == "swiss-landcover")
            .expect("the land-cover recipe");
        let commands = recipe_commands(recipe, &snapshot);
        let kinds: Vec<String> = commands
            .iter()
            .filter_map(|command| match command {
                studio_core::Command::AddCartographyLayer { layer, .. } => Some(
                    format!("{:?}", layer.kind)
                        .split(['(', ' '])
                        .next()
                        .unwrap_or("")
                        .to_owned(),
                ),
                _ => None,
            })
            .collect();
        // The land cover is an imagery layer and the plate is a plate layer;
        // the camera and the sun are transforms.
        assert!(
            kinds.iter().any(|kind| kind.contains("Imagery")),
            "the land cover must be a layer: {kinds:?}"
        );
        assert!(
            kinds.iter().any(|kind| kind.contains("Plate")),
            "the plate must be a layer: {kinds:?}"
        );
        assert!(
            commands
                .iter()
                .any(|command| matches!(command, studio_core::Command::SetTransform { .. })),
            "the recipe places the camera and the sun"
        );
    }

    /// The Fujiyama recipe names the vector layer's features with a label
    /// layer: the label layer's source is the vector layer's id.
    #[test]
    fn the_label_layer_names_the_vector_layers_features() {
        let project = studio_core::Project::starter(
            "Fuji",
            Some(studio_core::TerrainSeed {
                title: Some("Mount Fuji".to_owned()),
                name: "Mount_Fuji_30m.tif".to_owned(),
                path: "/data/Mount_Fuji_30m.tif".to_owned(),
                byte_size: 1,
                dimensions: Some([2594, 2594]),
                elevation_range: Some([300.0, 3775.0]),
                world_origin_xz: Some([0.0, 0.0]),
                world_span_xz: Some([18_000.0, 18_000.0]),
                overlay: None,
            }),
        );
        let snapshot = SessionState {
            project,
            revision: 0,
            dirty: false,
            path: None,
            can_undo: false,
            can_redo: false,
            undo_label: None,
            redo_label: None,
            history_len: 0,
        };
        let recipe = crate::forge3d::SCENE_PRESETS
            .iter()
            .find(|recipe| recipe.id == "fuji-labels")
            .expect("the labels recipe");
        let commands = recipe_commands(recipe, &snapshot);
        let mut vector_id = None;
        let mut label_source = None;
        for command in &commands {
            if let studio_core::Command::AddCartographyLayer { layer, .. } = command {
                match &layer.kind {
                    studio_core::CartographyLayerKind::Vector { .. } => vector_id = Some(layer.id),
                    studio_core::CartographyLayerKind::Labels { source, field, .. } => {
                        label_source = Some((*source, field.clone()))
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(
            label_source.as_ref().map(|(_, field)| field.clone()),
            Some("name.en".to_owned())
        );
        assert_eq!(
            label_source.as_ref().map(|(source, _)| *source),
            vector_id,
            "the label layer must name the vector layer above it"
        );
    }
}
