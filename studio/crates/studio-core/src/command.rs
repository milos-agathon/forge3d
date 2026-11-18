//! The command layer: every mutation is a value that can be applied and
//! inverted.
//!
//! `Command::apply` mutates the model and returns the command that reverses it.
//! The editor stores that inverse on the undo stack, so undo and redo are the
//! same code path as editing — there is no separate "restore" logic to drift.

use serde::{Deserialize, Serialize};

use std::path::Path;

use crate::ids::{new_id, AssetId, ObjectId};
use crate::math::Transform;
use crate::project::AssetKind;
use crate::project::{Asset, Project, ViewportState, Workspace};
use crate::scene::{validate_name, CameraObject, LightObject, MeshObject, Primitive, SceneObject};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    #[error("object {id} does not exist")]
    ObjectNotFound { id: ObjectId },

    #[error("an object with id {id} already exists")]
    DuplicateObject { id: ObjectId },

    #[error("object {id} has children; delete or re-parent them first")]
    ObjectHasChildren { id: ObjectId },

    #[error("object {id} is not a camera")]
    NotACamera { id: ObjectId },

    #[error("object {id} is not a light")]
    NotALight { id: ObjectId },

    #[error("object {id} is not a mesh")]
    NotAMesh { id: ObjectId },

    #[error("object {id} is an imported mesh, not a primitive")]
    NotAPrimitive { id: ObjectId },

    #[error("asset {id} does not exist")]
    AssetNotFound { id: AssetId },

    #[error("asset {id} is still used by an object")]
    AssetInUse { id: AssetId },

    #[error("an asset with id {id} already exists")]
    DuplicateAsset { id: AssetId },

    #[error("invalid {field}: {reason}")]
    InvalidValue { field: &'static str, reason: String },

    #[error("cannot insert an object at index {index}: the scene has {len} objects")]
    IndexOutOfRange { index: usize, len: usize },
}

impl CommandError {
    fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidValue {
            field,
            reason: reason.into(),
        }
    }
}

/// The map look's camera: from the south, 31 degrees off straight down — the
/// gallery's Swiss land-cover framing. North is up the screen (the engine's
/// `north` camera mode reads the azimuth from east towards north, so the
/// north-up view stands at 270°; at 90° the map showed upside down).
const MAP_AZIMUTH_DEG: f64 = 270.0;
const MAP_POLAR_DEG: f64 = 31.0;
/// The gallery's sun: north-west, 24 degrees above the horizon.
const MAP_SUN_AZIMUTH_DEG: f64 = 315.0;
const MAP_SUN_ELEVATION_DEG: f64 = 24.0;
/// How far the hillshade's rays reach, and how many steps they take.
const MAP_SUN_REACH_M: f64 = 4_000.0;
const MAP_SUN_STEPS: u32 = 48;

/// A reversible edit to the canonical project.
///
/// New values only: the previous values are read from the model during
/// [`Command::apply`], which is what makes an inverse exact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
// `rename_all` renames the *variants*; `rename_all_fields` is what actually
// makes struct-variant fields (`z_scale`, `byte_size`, `frame_rate`) arrive as
// the camelCase the TypeScript bridge sends. Without it the mock and the Rust
// tests (which build Command values directly) stay green while the real Tauri
// IPC rejects the payload — see tests/wire_shapes.rs.
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum Command {
    /// Insert an object (already built, with a stable id) at `index`
    /// (end of the list when omitted).
    CreateObject {
        object: SceneObject,
        #[serde(default)]
        index: Option<usize>,
    },

    /// Remove an object and detach it from the selection.
    DeleteObject {
        id: ObjectId,
    },

    SetTransform {
        id: ObjectId,
        transform: Transform,
    },

    RenameObject {
        id: ObjectId,
        name: String,
    },

    SetVisibility {
        id: ObjectId,
        visible: bool,
    },

    SetLocked {
        id: ObjectId,
        locked: bool,
    },

    /// Replace the camera payload of a camera object.
    SetCamera {
        id: ObjectId,
        camera: CameraObject,
    },

    /// Replace the light payload of a light object.
    SetLight {
        id: ObjectId,
        light: LightObject,
    },

    /// Replace the mesh's primitive (shape/edge length).
    SetPrimitive {
        id: ObjectId,
        primitive: Primitive,
    },

    /// Create a primitive mesh object at the origin, in one step.
    ///
    /// The id is minted here rather than in the UI: object identity belongs to
    /// the model, and a caller-supplied id would let a stale window collide.
    CreatePrimitive {
        primitive: Primitive,
        #[serde(default)]
        name: Option<String>,
    },

    /// Set a terrain's vertical exaggeration (the DEM itself is unchanged).
    SetTerrainZScale {
        id: ObjectId,
        z_scale: f32,
    },

    /// Aim the scene's directional sun at a compass azimuth (0..=360, clockwise
    /// from north) and an elevation in degrees. The Transform rotation is
    /// derived here, not sent by the caller, so the engine keeps one source
    /// of truth for the camera/sun mapping.
    SetSunAngles {
        id: ObjectId,
        azimuth_deg: f32,
        elevation_deg: f32,
    },

    /// Replace a terrain's DEM (import/re-point), carrying its measured metadata.
    SetTerrainDem {
        id: ObjectId,
        asset: crate::ids::AssetId,
        #[serde(default)]
        dimensions: Option<[u32; 2]>,
        #[serde(default)]
        elevation_range: Option<[f32; 2]>,
        #[serde(default)]
        world_origin_xz: Option<[f64; 2]>,
        #[serde(default)]
        world_span_xz: Option<[f64; 2]>,
        /// Vertical exaggeration; omitted means "the new DEM's default".
        #[serde(default)]
        z_scale: Option<f32>,
    },

    /// Re-point an asset at a new file (relinking after a move).
    SetAssetPath {
        id: crate::ids::AssetId,
        path: String,
        byte_size: u64,
    },

    /// Replace render output settings.
    SetRenderSettings {
        render: crate::project::RenderSettings,
    },

    /// Replace a mesh's Studio material.
    SetMeshMaterial {
        id: ObjectId,
        material: crate::scene::Material,
    },

    /// Pick which node inside an imported glTF asset the object draws
    /// (`None` = the whole file).
    SetGltfNode {
        id: ObjectId,
        #[serde(default)]
        node: Option<String>,
    },

    /// Replace an object's animation channels wholesale (empty = clear the
    /// whole track). The inverse carries the previous channel list, so undo
    /// restores it exactly.
    SetChannelKeys {
        id: ObjectId,
        channels: Vec<crate::timeline::ChannelKeys>,
    },

    /// Change the timeline's frame rate and length (view configuration; the
    /// keyframes themselves are untouched).
    SetTimelineConfig {
        frame_rate: u32,
        length: u32,
    },

    /// Apply the map look in one step: the DEM's own relief, a map camera,
    /// the gallery's north-west sun, and the ray-marched hillshade that makes
    /// a country-sized DEM read as terrain instead of a sheet.
    ApplyMapLook {
        id: ObjectId,
    },

    /// Set the terrain's vertical scale to the DEM's own default: the
    /// exaggeration that gives its relief 6% of its footprint. One command so
    /// the rule lives only in `scene::dem_default_z_scale` (a project that
    /// predates the default can adopt it in one undoable step).
    FitTerrainRelief {
        id: ObjectId,
    },

    /// Add a layer to the map's stack. `index` places it in the draw order
    /// (the end of the list is the front of the map).
    AddCartographyLayer {
        layer: crate::cartography::CartographyLayer,
        #[serde(default)]
        index: Option<usize>,
    },

    /// Remove a layer from the map's stack.
    RemoveCartographyLayer {
        id: crate::cartography::CartographyLayerId,
    },

    /// Move a layer within the map's stack, to the draw position `to`.
    MoveCartographyLayer {
        id: crate::cartography::CartographyLayerId,
        to: usize,
    },

    /// Change a layer: its name, its eye or its style.
    UpdateCartographyLayer {
        id: crate::cartography::CartographyLayerId,
        layer: crate::cartography::CartographyLayer,
    },

    /// Duplicate an object in place. Children are not duplicated in Phase 2.
    DuplicateObject {
        id: ObjectId,
        /// Explicit name; defaults to a free Blender-style name (`Cube.001`).
        #[serde(default)]
        name: Option<String>,
    },

    /// Register an imported asset — batched with the object that uses it.
    AddAsset {
        asset: Asset,
    },

    /// Register a glTF/GLB file and create the mesh object that renders it, as
    /// one undo step. Both ids are minted here, so the UI never invents
    /// identities and redo cannot drift.
    ImportGltf {
        /// Display name, normally the file name.
        name: String,
        /// Path as read from disk. Relative paths resolve to the project
        /// directory on load.
        path: String,
        kind: AssetKind,
        byte_size: u64,
        /// glTF node to render; `None` means the whole asset.
        #[serde(default)]
        node: Option<String>,
    },

    /// Unregister an asset. Refused while an object still references it.
    RemoveAsset {
        id: AssetId,
    },

    /// Point the viewport and Forge3D handoff at a camera (or none).
    SetActiveCamera {
        id: Option<ObjectId>,
    },

    /// Viewport view state (grid, shading mode). Not scene content, but part of
    /// the persisted project, so it goes through the same command path.
    SetViewport {
        viewport: ViewportState,
    },

    /// Switch the active workspace. Persisted view state, and undoable so a
    /// mis-click cannot leave the project in a surprising workspace.
    SetWorkspace {
        workspace: Workspace,
    },

    /// Several commands applied in order as one undo step.
    Batch {
        commands: Vec<Command>,
    },
}

/// What an applied command did: its inverse, plus the object IDs a viewport
/// adapter must re-sync. Internal — callers get the inverse from
/// [`Command::apply`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Outcome {
    pub inverse: Command,
    pub affected: Vec<ObjectId>,
}

impl Command {
    /// Apply to `project` and return the command that reverses this exact edit.
    ///
    /// On error the project is left unchanged: every variant reads and
    /// validates before it writes.
    pub fn apply(&self, project: &mut Project) -> Result<Command, CommandError> {
        self.apply_inner(project).map(|outcome| outcome.inverse)
    }

    /// Apply and also return the object IDs that actually changed.
    ///
    /// This is the authoritative answer, unlike [`Command::affected`]: a
    /// duplicate's new ID does not exist until the command runs.
    pub fn apply_with_effect(
        &self,
        project: &mut Project,
    ) -> Result<(Command, Vec<ObjectId>), CommandError> {
        self.apply_inner(project)
            .map(|outcome| (outcome.inverse, outcome.affected))
    }

    fn apply_inner(&self, project: &mut Project) -> Result<Outcome, CommandError> {
        match self {
            Command::CreateObject { object, index } => {
                object
                    .validate()
                    .map_err(|reason| CommandError::invalid("object", reason))?;
                if project.scene.object(object.id).is_some() {
                    return Err(CommandError::DuplicateObject { id: object.id });
                }
                if let Some(parent) = object.parent {
                    if project.scene.object(parent).is_none() {
                        return Err(CommandError::ObjectNotFound { id: parent });
                    }
                }
                if let Some(asset) = object.mesh().and_then(MeshObject::asset_id) {
                    if project.asset(asset).is_none() {
                        return Err(CommandError::AssetNotFound { id: asset });
                    }
                }
                // Terrain references must resolve too — CreateObject already
                // guards mesh assets; a dangling DEM would fail validation
                // later (autosave) instead of at the door.
                if let Some(asset) = object.terrain().map(|terrain| terrain.asset) {
                    if project.asset(asset).is_none() {
                        return Err(CommandError::AssetNotFound { id: asset });
                    }
                }
                let len = project.scene.objects.len();
                let index = index.unwrap_or(len);
                if index > len {
                    return Err(CommandError::IndexOutOfRange { index, len });
                }
                project.scene.objects.insert(index, object.clone());

                Ok(Outcome {
                    inverse: Command::DeleteObject { id: object.id },
                    affected: vec![object.id],
                })
            }

            Command::DeleteObject { id } => {
                let index = project
                    .scene
                    .index_of(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                if !project.scene.children_of(*id).is_empty() {
                    return Err(CommandError::ObjectHasChildren { id: *id });
                }
                let object = project.scene.objects.remove(index);
                let restore_camera = project.scene.active_camera == Some(*id);
                if restore_camera {
                    project.scene.active_camera = None;
                }
                project.selection.retain(|selected| selected != id);

                // Undoing a deletion that also cleared the active camera must
                // restore both, so the inverse is a batch.
                let inverse = if restore_camera {
                    Command::Batch {
                        commands: vec![
                            Command::CreateObject {
                                object: object.clone(),
                                index: Some(index),
                            },
                            Command::SetActiveCamera { id: Some(*id) },
                        ],
                    }
                } else {
                    Command::CreateObject {
                        object: object.clone(),
                        index: Some(index),
                    }
                };

                Ok(Outcome {
                    inverse,
                    affected: vec![*id],
                })
            }

            Command::SetTransform { id, transform } => {
                transform
                    .validate()
                    .map_err(|reason| CommandError::invalid("transform", reason))?;
                // Terrain carries a measured georeferenced footprint; moving or
                // scaling it only desyncs the Solid view from the renderer,
                // which always draws the DEM centred at the origin.
                if project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?
                    .is_terrain()
                {
                    return Err(CommandError::invalid(
                        "transform",
                        "terrain position is fixed (the DEM is georeferenced)",
                    ));
                }
                let object = project
                    .scene
                    .object_mut(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = object.transform;
                object.transform = *transform;
                Ok(Outcome {
                    inverse: Command::SetTransform {
                        id: *id,
                        transform: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::RenameObject { id, name } => {
                validate_name(name, "object name")
                    .map_err(|reason| CommandError::invalid("object name", reason))?;
                let object = project
                    .scene
                    .object_mut(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = std::mem::replace(&mut object.name, name.clone());
                Ok(Outcome {
                    inverse: Command::RenameObject {
                        id: *id,
                        name: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetVisibility { id, visible } => {
                let object = project
                    .scene
                    .object_mut(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = object.visible;
                object.visible = *visible;
                Ok(Outcome {
                    inverse: Command::SetVisibility {
                        id: *id,
                        visible: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetLocked { id, locked } => {
                let object = project
                    .scene
                    .object_mut(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = object.locked;
                object.locked = *locked;
                Ok(Outcome {
                    inverse: Command::SetLocked {
                        id: *id,
                        locked: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetCamera { id, camera } => {
                camera
                    .validate()
                    .map_err(|reason| CommandError::invalid("camera", reason))?;
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = match object.kind {
                    crate::scene::ObjectKind::Camera(previous) => previous,
                    _ => return Err(CommandError::NotACamera { id: *id }),
                };
                project.scene.object_mut(*id).expect("checked above").kind =
                    crate::scene::ObjectKind::Camera(*camera);
                Ok(Outcome {
                    inverse: Command::SetCamera {
                        id: *id,
                        camera: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetLight { id, light } => {
                light
                    .validate()
                    .map_err(|reason| CommandError::invalid("light", reason))?;
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = match object.kind {
                    crate::scene::ObjectKind::Light(previous) => previous,
                    _ => return Err(CommandError::NotALight { id: *id }),
                };
                project.scene.object_mut(*id).expect("checked above").kind =
                    crate::scene::ObjectKind::Light(*light);
                Ok(Outcome {
                    inverse: Command::SetLight {
                        id: *id,
                        light: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetPrimitive { id, primitive } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = match object.mesh() {
                    Some(mesh) => mesh
                        .primitive_source()
                        .cloned()
                        .ok_or(CommandError::NotAPrimitive { id: *id })?,
                    None => return Err(CommandError::NotAMesh { id: *id }),
                };
                // Validate by building the whole object, so primitive rules
                // live in one place (`scene::Primitive::validate`).
                let mut probe = object.clone();
                probe.kind =
                    crate::scene::ObjectKind::Mesh(MeshObject::primitive(primitive.clone()));
                probe
                    .validate()
                    .map_err(|reason| CommandError::invalid("primitive", reason))?;
                if let Some(object) = project.scene.object_mut(*id) {
                    object.kind =
                        crate::scene::ObjectKind::Mesh(MeshObject::primitive(primitive.clone()));
                }
                Ok(Outcome {
                    inverse: Command::SetPrimitive {
                        id: *id,
                        primitive: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::CreatePrimitive { primitive, name } => {
                let default_name = match primitive {
                    Primitive::Cube { .. } => "Cube",
                    Primitive::Sphere { .. } => "Sphere",
                    Primitive::Plane { .. } => "Plane",
                };
                let object = SceneObject {
                    id: new_id(),
                    name: match name {
                        Some(name) => {
                            validate_name(name, "object name")
                                .map_err(|reason| CommandError::invalid("object name", reason))?;
                            name.clone()
                        }
                        None => project.scene.unique_name(default_name),
                    },
                    parent: None,
                    transform: Transform::default(),
                    visible: true,
                    locked: false,
                    kind: crate::scene::ObjectKind::Mesh(MeshObject::primitive(primitive.clone())),
                };
                object
                    .validate()
                    .map_err(|reason| CommandError::invalid("object", reason))?;
                let id = object.id;
                project.scene.objects.push(object);
                Ok(Outcome {
                    inverse: Command::DeleteObject { id },
                    affected: vec![id],
                })
            }

            Command::SetTerrainZScale { id, z_scale } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = match &object.kind {
                    crate::scene::ObjectKind::Terrain(terrain) => terrain.z_scale,
                    _ => return Err(CommandError::invalid("terrain", "object is not a terrain")),
                };
                // `TerrainRenderParams` raises "z_scale must be 0.1-50.0"
                // rather than clamping, so an out-of-range value must be
                // rejected here — not handed to the renderer.
                if !z_scale.is_finite()
                    || !(crate::scene::MIN_Z_SCALE..=crate::scene::MAX_Z_SCALE).contains(z_scale)
                {
                    return Err(CommandError::invalid(
                        "terrain vertical scale",
                        format!(
                            "must be between {} and {}",
                            crate::scene::MIN_Z_SCALE,
                            crate::scene::MAX_Z_SCALE
                        ),
                    ));
                }
                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Terrain(terrain) = &mut object.kind {
                        terrain.z_scale = *z_scale;
                    }
                }
                Ok(Outcome {
                    inverse: Command::SetTerrainZScale {
                        id: *id,
                        z_scale: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetSunAngles {
                id,
                azimuth_deg,
                elevation_deg,
            } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = object.transform;
                match &object.kind {
                    crate::scene::ObjectKind::Light(light)
                        if light.kind == crate::scene::LightKind::Directional => {}
                    _ => {
                        return Err(CommandError::invalid(
                            "light",
                            "object is not the directional sun",
                        ))
                    }
                }
                if !azimuth_deg.is_finite() || !(0.0..=360.0).contains(azimuth_deg) {
                    return Err(CommandError::invalid(
                        "sun azimuth",
                        "must be between 0 and 360 degrees",
                    ));
                }
                if !elevation_deg.is_finite() || !(-90.0..=90.0).contains(elevation_deg) {
                    return Err(CommandError::invalid(
                        "sun elevation",
                        "must be between -90 and 90 degrees",
                    ));
                }
                // The engine's own convention: azimuth east → south (atan2(z, x)),
                // elevation above the horizon. `sun_rotation` round-trips through
                // `sun_angles`, and the two must never disagree on which angle a
                // fixture means — that is the one thing users will check.
                let az = f64::from(*azimuth_deg).to_radians();
                let el = f64::from(*elevation_deg).to_radians();
                let toward = [
                    (el.cos() * az.cos()) as f32,
                    el.sin() as f32,
                    (el.cos() * az.sin()) as f32,
                ];
                // The engine consumes its own scene frame (east, up, north):
                // the viewer's +Z is south, so the north component flips.
                let scene = [toward[0], toward[1], -toward[2]];
                let rotation = crate::workspace::sun_rotation(scene);
                if let Some(object) = project.scene.object_mut(*id) {
                    object.transform.rotation_deg = rotation;
                }
                Ok(Outcome {
                    inverse: Command::SetTransform {
                        id: *id,
                        transform: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::ApplyMapLook { id } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let crate::scene::ObjectKind::Terrain(terrain) = &object.kind else {
                    return Err(CommandError::invalid("terrain", "object is not a terrain"));
                };
                let terrain = terrain.clone();
                let camera = project
                    .scene
                    .camera()
                    .ok_or_else(|| CommandError::invalid("scene", "the scene has no camera"))?;
                let camera_id = camera.id;
                let camera_kind = match &camera.kind {
                    crate::scene::ObjectKind::Camera(kind) => *kind,
                    _ => return Err(CommandError::invalid("camera", "object is not a camera")),
                };
                let camera_transform = camera.transform;
                let sun = project
                    .scene
                    .objects
                    .iter()
                    .find(|object| {
                        matches!(&object.kind, crate::scene::ObjectKind::Light(light) if light.kind == crate::scene::LightKind::Directional)
                    })
                    .ok_or_else(|| CommandError::invalid("scene", "the scene has no sun"))?;
                let sun_id = sun.id;
                let sun_transform = sun.transform;

                let (map_transform, still_fov) = crate::workspace::map_camera_transform(
                    &terrain,
                    MAP_AZIMUTH_DEG,
                    MAP_POLAR_DEG,
                    f64::from(camera_kind.fov_deg),
                )
                .ok_or_else(|| {
                    CommandError::invalid("terrain", "the DEM has no measured footprint")
                })?;
                let sun_rotation = crate::workspace::sun_rotation(
                    crate::workspace::map_sun_direction(MAP_SUN_AZIMUTH_DEG, MAP_SUN_ELEVATION_DEG),
                );
                let previous_z_scale = terrain.z_scale;
                let previous_render = project.render.clone();

                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Terrain(terrain) = &mut object.kind {
                        terrain.z_scale = crate::scene::dem_default_z_scale(
                            terrain.elevation_range,
                            terrain.world_span_xz,
                        );
                    }
                }
                if let Some(object) = project.scene.object_mut(camera_id) {
                    object.transform = map_transform;
                    if let crate::scene::ObjectKind::Camera(kind) = &mut object.kind {
                        kind.fov_deg = still_fov;
                    }
                }
                if let Some(object) = project.scene.object_mut(sun_id) {
                    object.transform.rotation_deg = sun_rotation;
                }
                project.render.sun_visibility = true;
                project.render.sun_visibility_distance = MAP_SUN_REACH_M;
                project.render.sun_visibility_steps = MAP_SUN_STEPS;

                Ok(Outcome {
                    inverse: Command::Batch {
                        commands: vec![
                            Command::SetTerrainZScale {
                                id: *id,
                                z_scale: previous_z_scale,
                            },
                            Command::SetTransform {
                                id: camera_id,
                                transform: camera_transform,
                            },
                            Command::SetCamera {
                                id: camera_id,
                                camera: camera_kind,
                            },
                            Command::SetTransform {
                                id: sun_id,
                                transform: sun_transform,
                            },
                            Command::SetRenderSettings {
                                render: previous_render,
                            },
                        ],
                    },
                    affected: vec![*id, camera_id, sun_id],
                })
            }

            Command::FitTerrainRelief { id } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let (previous, elevation_range, world_span_xz) = match &object.kind {
                    crate::scene::ObjectKind::Terrain(terrain) => (
                        terrain.z_scale,
                        terrain.elevation_range,
                        terrain.world_span_xz,
                    ),
                    _ => return Err(CommandError::invalid("terrain", "object is not a terrain")),
                };
                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Terrain(terrain) = &mut object.kind {
                        terrain.z_scale =
                            crate::scene::dem_default_z_scale(elevation_range, world_span_xz);
                    }
                }
                Ok(Outcome {
                    inverse: Command::SetTerrainZScale {
                        id: *id,
                        z_scale: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetTerrainDem {
                id,
                asset,
                dimensions,
                elevation_range,
                world_origin_xz,
                world_span_xz,
                z_scale,
            } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let Some(dem) = project.asset(*asset) else {
                    return Err(CommandError::AssetNotFound { id: *asset });
                };
                if dem.kind != crate::project::AssetKind::Dem {
                    return Err(CommandError::invalid(
                        "DEM",
                        "asset is not an elevation model",
                    ));
                }
                let previous = match &object.kind {
                    crate::scene::ObjectKind::Terrain(terrain) => (
                        terrain.asset,
                        terrain.dimensions,
                        terrain.elevation_range,
                        terrain.world_origin_xz,
                        terrain.world_span_xz,
                        terrain.z_scale,
                    ),
                    _ => return Err(CommandError::invalid("terrain", "object is not a terrain")),
                };
                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Terrain(terrain) = &mut object.kind {
                        terrain.asset = *asset;
                        terrain.dimensions = *dimensions;
                        terrain.elevation_range = *elevation_range;
                        terrain.world_origin_xz = *world_origin_xz;
                        terrain.world_span_xz = *world_span_xz;
                        // A new DEM's relief has nothing to do with the old
                        // one's, so its vertical scale is re-derived unless the
                        // command names one (the undo carries the old value).
                        terrain.z_scale = z_scale.unwrap_or_else(|| {
                            crate::scene::dem_default_z_scale(*elevation_range, *world_span_xz)
                        });
                    }
                }
                Ok(Outcome {
                    inverse: Command::SetTerrainDem {
                        id: *id,
                        asset: previous.0,
                        dimensions: previous.1,
                        elevation_range: previous.2,
                        world_origin_xz: previous.3,
                        world_span_xz: previous.4,
                        z_scale: Some(previous.5),
                    },
                    affected: vec![*id],
                })
            }

            Command::SetRenderSettings { render } => {
                // `RenderSettings` holds the colormap name, so it is no longer
                // Copy: clone where a snapshot is needed.
                let previous = project.render.clone();
                let mut probe = project.clone();
                probe.render = render.clone();
                probe
                    .validate()
                    .map_err(|reason| CommandError::invalid("render settings", reason))?;
                project.render = render.clone();
                Ok(Outcome {
                    inverse: Command::SetRenderSettings { render: previous },
                    affected: Vec::new(),
                })
            }

            Command::SetAssetPath {
                id,
                path,
                byte_size,
            } => {
                if path.trim().is_empty() {
                    return Err(CommandError::invalid("asset path", "must not be empty"));
                }
                let index = project
                    .assets
                    .iter()
                    .position(|asset| asset.id == *id)
                    .ok_or(CommandError::AssetNotFound { id: *id })?;
                let previous = project.assets[index].path.clone();
                let previous_size = project.assets[index].byte_size;
                let mut candidate = project.assets[index].clone();
                candidate.path = path.clone();
                candidate.byte_size = *byte_size;
                // Relinking re-runs the same invariant as AddAsset: a size of
                // 0 means "empty file" and would fail autosave (which
                // validates) long after this command succeeded.
                candidate
                    .validate()
                    .map_err(|reason| CommandError::invalid("asset", reason))?;
                project.assets[index] = candidate;
                Ok(Outcome {
                    inverse: Command::SetAssetPath {
                        id: *id,
                        path: previous,
                        byte_size: previous_size,
                    },
                    affected: Vec::new(),
                })
            }

            Command::SetMeshMaterial { id, material } => {
                material
                    .validate()
                    .map_err(|reason| CommandError::invalid("material", reason))?;
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let previous = match &object.kind {
                    crate::scene::ObjectKind::Mesh(mesh) => mesh.material,
                    _ => return Err(CommandError::NotAMesh { id: *id }),
                };
                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Mesh(mesh) = &mut object.kind {
                        mesh.material = *material;
                    }
                }
                Ok(Outcome {
                    inverse: Command::SetMeshMaterial {
                        id: *id,
                        material: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetGltfNode { id, node } => {
                if let Some(name) = node.as_deref() {
                    if name.trim().is_empty() {
                        return Err(CommandError::invalid("node", "must not be empty"));
                    }
                }
                let previous = {
                    let object = project
                        .scene
                        .object(*id)
                        .ok_or(CommandError::ObjectNotFound { id: *id })?;
                    let Some(mesh) = object.mesh() else {
                        return Err(CommandError::NotAMesh { id: *id });
                    };
                    let crate::scene::MeshSource::Gltf { node: existing, .. } = &mesh.source else {
                        return Err(CommandError::invalid(
                            "node",
                            "object is not an imported glTF mesh",
                        ));
                    };
                    existing.clone()
                };
                if let Some(object) = project.scene.object_mut(*id) {
                    if let crate::scene::ObjectKind::Mesh(mesh) = &mut object.kind {
                        if let crate::scene::MeshSource::Gltf { node: slot, .. } = &mut mesh.source
                        {
                            *slot = node.clone();
                        }
                    }
                }
                Ok(Outcome {
                    inverse: Command::SetGltfNode {
                        id: *id,
                        node: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetChannelKeys { id, channels } => {
                let object = project
                    .scene
                    .object(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                if object.is_terrain() {
                    return Err(CommandError::invalid(
                        "channel keys",
                        "terrain position is fixed (the DEM is georeferenced)",
                    ));
                }
                // FOV is a camera-only channel; reject it anywhere else so a
                // track can never claim to animate something that ignores it.
                for keys in channels {
                    if keys.channel == crate::timeline::Channel::FovDeg
                        && !matches!(object.kind, crate::scene::ObjectKind::Camera(_))
                    {
                        return Err(CommandError::invalid(
                            "channel keys",
                            "field of view can only be keyed on a camera",
                        ));
                    }
                }
                let previous = project
                    .timeline
                    .track(*id)
                    .map(|track| track.channels.clone())
                    .unwrap_or_default();
                // Validate the whole resulting document, not just the keys:
                // ordering, finiteness, and FOV range live in Timeline::validate.
                let mut probe = project.clone();
                crate::timeline::set_track_channels(&mut probe.timeline, *id, channels.clone());
                probe
                    .validate()
                    .map_err(|reason| CommandError::invalid("channel keys", reason))?;

                crate::timeline::set_track_channels(&mut project.timeline, *id, channels.clone());
                Ok(Outcome {
                    inverse: Command::SetChannelKeys {
                        id: *id,
                        channels: previous,
                    },
                    affected: vec![*id],
                })
            }

            Command::SetTimelineConfig { frame_rate, length } => {
                let previous = (project.timeline.frame_rate, project.timeline.length);
                let mut probe = project.clone();
                probe.timeline.frame_rate = *frame_rate;
                probe.timeline.length = *length;
                // Shortening the timeline parks the playhead inside it. The
                // playhead is view state (like the selection), so the inverse
                // restores only the configuration.
                probe.timeline.current_frame =
                    probe.timeline.clamp_frame(probe.timeline.current_frame);
                probe
                    .validate()
                    .map_err(|reason| CommandError::invalid("timeline", reason))?;

                project.timeline.frame_rate = *frame_rate;
                project.timeline.length = *length;
                project.timeline.current_frame = probe.timeline.current_frame;
                Ok(Outcome {
                    inverse: Command::SetTimelineConfig {
                        frame_rate: previous.0,
                        length: previous.1,
                    },
                    affected: Vec::new(),
                })
            }

            Command::AddCartographyLayer { layer, index } => {
                let mut layer = layer.clone();
                if layer.id.is_nil() {
                    // A layer without an id gets one: the id is what the stack
                    // is keyed by, and a duplicate would make the second layer
                    // of a kind impossible to add.
                    layer.id = new_id();
                }
                layer
                    .validate()
                    .map_err(|reason| CommandError::invalid("cartography layer", reason))?;
                if let Some(asset) = layer.kind.asset() {
                    if project.asset(asset).is_none() {
                        return Err(CommandError::AssetNotFound { id: asset });
                    }
                }
                if project.cartography.iter().any(|entry| entry.id == layer.id) {
                    return Err(CommandError::invalid(
                        "cartography layer",
                        "a layer with this id already exists",
                    ));
                }
                let position = index
                    .unwrap_or(project.cartography.len())
                    .min(project.cartography.len());
                project.cartography.insert(position, layer.clone());
                Ok(Outcome {
                    inverse: Command::RemoveCartographyLayer { id: layer.id },
                    affected: Vec::new(),
                })
            }

            Command::RemoveCartographyLayer { id } => {
                let position = project
                    .cartography
                    .iter()
                    .position(|entry| entry.id == *id)
                    .ok_or_else(|| {
                        CommandError::invalid("cartography layer", "the layer does not exist")
                    })?;
                let removed = project.cartography.remove(position);
                Ok(Outcome {
                    inverse: Command::AddCartographyLayer {
                        layer: removed,
                        index: Some(position),
                    },
                    affected: Vec::new(),
                })
            }

            Command::MoveCartographyLayer { id, to } => {
                let position = project
                    .cartography
                    .iter()
                    .position(|entry| entry.id == *id)
                    .ok_or_else(|| {
                        CommandError::invalid("cartography layer", "the layer does not exist")
                    })?;
                if *to >= project.cartography.len() {
                    return Err(CommandError::IndexOutOfRange {
                        index: *to,
                        len: project.cartography.len(),
                    });
                }
                if position == *to {
                    return Ok(Outcome {
                        inverse: Command::MoveCartographyLayer { id: *id, to: *to },
                        affected: Vec::new(),
                    });
                }
                let layer = project.cartography.remove(position);
                project.cartography.insert(*to, layer);
                Ok(Outcome {
                    inverse: Command::MoveCartographyLayer {
                        id: *id,
                        to: position,
                    },
                    affected: Vec::new(),
                })
            }

            Command::UpdateCartographyLayer { id, layer } => {
                let layer = layer.clone();
                layer
                    .validate()
                    .map_err(|reason| CommandError::invalid("cartography layer", reason))?;
                if let Some(asset) = layer.kind.asset() {
                    if project.asset(asset).is_none() {
                        return Err(CommandError::AssetNotFound { id: asset });
                    }
                }
                let position = project
                    .cartography
                    .iter()
                    .position(|entry| entry.id == *id)
                    .ok_or_else(|| {
                        CommandError::invalid("cartography layer", "the layer does not exist")
                    })?;
                let previous = project.cartography[position].clone();
                project.cartography[position] = layer;
                Ok(Outcome {
                    inverse: Command::UpdateCartographyLayer {
                        id: *id,
                        layer: previous,
                    },
                    affected: Vec::new(),
                })
            }

            Command::DuplicateObject { id, name } => {
                let index = project
                    .scene
                    .index_of(*id)
                    .ok_or(CommandError::ObjectNotFound { id: *id })?;
                let mut copy = project.scene.objects[index].clone();
                copy.id = new_id();
                copy.name = match name {
                    Some(name) => {
                        validate_name(name, "object name")
                            .map_err(|reason| CommandError::invalid("object name", reason))?;
                        name.clone()
                    }
                    None => project.scene.unique_name(&copy.name),
                };
                // A scene holds one terrain (the renderer reads the first and
                // silently ignores the rest); replacing the DEM is Change DEM.
                if copy.is_terrain() {
                    return Err(CommandError::invalid(
                        "terrain",
                        "a scene holds one terrain; use Change DEM to replace it",
                    ));
                }
                // A duplicated camera must not silently take over rendering.
                let copy_id = copy.id;
                project.scene.objects.insert(index + 1, copy);

                Ok(Outcome {
                    inverse: Command::DeleteObject { id: copy_id },
                    affected: vec![copy_id],
                })
            }

            Command::ImportGltf {
                name,
                path,
                kind,
                byte_size,
                node,
            } => {
                let asset = Asset {
                    id: new_id(),
                    name: name.clone(),
                    path: path.clone(),
                    kind: *kind,
                    byte_size: *byte_size,
                };
                asset
                    .validate()
                    .map_err(|reason| CommandError::invalid("asset", reason))?;

                let stem = Path::new(name)
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .filter(|stem| !stem.trim().is_empty())
                    .unwrap_or_else(|| "Imported".to_owned());
                let object = SceneObject {
                    id: new_id(),
                    name: project.scene.unique_name(&stem),
                    parent: None,
                    transform: Transform::default(),
                    visible: true,
                    locked: false,
                    kind: crate::scene::ObjectKind::Mesh(MeshObject::gltf(asset.id, node.clone())),
                };
                object
                    .validate()
                    .map_err(|reason| CommandError::invalid("object", reason))?;

                let asset_id = asset.id;
                let object_id = object.id;
                project.assets.push(asset);
                project.scene.objects.push(object);

                // Undo removes the object first: the asset may not be dropped
                // while something still references it.
                Ok(Outcome {
                    inverse: Command::Batch {
                        commands: vec![
                            Command::DeleteObject { id: object_id },
                            Command::RemoveAsset { id: asset_id },
                        ],
                    },
                    affected: vec![object_id],
                })
            }

            Command::AddAsset { asset } => {
                asset
                    .validate()
                    .map_err(|reason| CommandError::invalid("asset", reason))?;
                if project.asset(asset.id).is_some() {
                    return Err(CommandError::DuplicateAsset { id: asset.id });
                }
                project.assets.push(asset.clone());
                Ok(Outcome {
                    inverse: Command::RemoveAsset { id: asset.id },
                    affected: Vec::new(),
                })
            }

            Command::RemoveAsset { id } => {
                let index = project
                    .assets
                    .iter()
                    .position(|asset| asset.id == *id)
                    .ok_or(CommandError::AssetNotFound { id: *id })?;
                if project.used_assets().contains(id) {
                    return Err(CommandError::AssetInUse { id: *id });
                }
                let asset = project.assets.remove(index);
                Ok(Outcome {
                    inverse: Command::AddAsset { asset },
                    affected: Vec::new(),
                })
            }

            Command::SetActiveCamera { id } => {
                if let Some(id) = id {
                    match project.scene.object(*id) {
                        Some(object) if object.is_camera() => {}
                        Some(_) => return Err(CommandError::NotACamera { id: *id }),
                        None => return Err(CommandError::ObjectNotFound { id: *id }),
                    }
                }
                let previous = project.scene.active_camera;
                project.scene.active_camera = *id;
                Ok(Outcome {
                    inverse: Command::SetActiveCamera { id: previous },
                    affected: id.iter().copied().collect(),
                })
            }

            Command::SetViewport { viewport } => {
                let previous = project.viewport;
                project.viewport = *viewport;
                Ok(Outcome {
                    inverse: Command::SetViewport { viewport: previous },
                    affected: Vec::new(),
                })
            }

            Command::SetWorkspace { workspace } => {
                let previous = project.workspace;
                project.workspace = *workspace;
                Ok(Outcome {
                    inverse: Command::SetWorkspace {
                        workspace: previous,
                    },
                    affected: Vec::new(),
                })
            }

            Command::Batch { commands } => {
                let mut inverses = Vec::with_capacity(commands.len());
                let mut affected = Vec::new();
                for command in commands {
                    let applied = command.apply_inner(project)?;
                    affected.extend(applied.affected);
                    inverses.push(applied.inverse);
                }
                inverses.reverse();
                Ok(Outcome {
                    inverse: Command::Batch { commands: inverses },
                    affected,
                })
            }
        }
    }

    /// Object IDs this command names *before* it runs.
    ///
    /// A preview only: prefer [`Command::apply_with_effect`], which reports the
    /// IDs that actually changed (a duplicate's ID is created during `apply`).
    pub fn affected(&self) -> Vec<ObjectId> {
        match self {
            Command::CreateObject { object, .. } => vec![object.id],
            Command::DeleteObject { id } => vec![*id],
            Command::SetTransform { id, .. }
            | Command::RenameObject { id, .. }
            | Command::SetVisibility { id, .. }
            | Command::SetLocked { id, .. }
            | Command::SetCamera { id, .. }
            | Command::SetLight { id, .. }
            | Command::SetPrimitive { id, .. }
            | Command::SetTerrainZScale { id, .. }
            | Command::SetSunAngles { id, .. }
            | Command::FitTerrainRelief { id }
            | Command::ApplyMapLook { id }
            | Command::SetTerrainDem { id, .. }
            | Command::SetMeshMaterial { id, .. }
            | Command::SetGltfNode { id, .. }
            | Command::SetChannelKeys { id, .. } => vec![*id],
            Command::SetRenderSettings { .. }
            | Command::SetAssetPath { .. }
            | Command::SetTimelineConfig { .. }
            | Command::AddCartographyLayer { .. }
            | Command::RemoveCartographyLayer { .. }
            | Command::MoveCartographyLayer { .. }
            | Command::UpdateCartographyLayer { .. } => Vec::new(),
            Command::SetActiveCamera { id } => id.iter().copied().collect(),
            // The duplicate's id is only known once applied; resyncing the
            // original object is enough for an adapter to refresh the scene.
            Command::DuplicateObject { id, .. } => vec![*id],
            Command::CreatePrimitive { .. } | Command::ImportGltf { .. } => Vec::new(),
            Command::SetViewport { .. } | Command::SetWorkspace { .. } => Vec::new(),
            Command::AddAsset { .. } | Command::RemoveAsset { .. } => Vec::new(),
            Command::Batch { commands } => {
                let mut ids = Vec::new();
                for command in commands {
                    ids.extend(command.affected());
                }
                ids
            }
        }
    }

    /// Human-readable undo label, e.g. `Move Cube`.
    pub fn label(&self, project: &Project) -> String {
        let name_of = |id: &ObjectId| {
            project
                .scene
                .object(*id)
                .map(|object| object.name.clone())
                .unwrap_or_else(|| "object".to_owned())
        };
        match self {
            Command::CreateObject { object, .. } => format!("Create {}", object.name),
            Command::DeleteObject { id } => format!("Delete {}", name_of(id)),
            Command::SetTransform { id, .. } => format!("Move {}", name_of(id)),
            Command::RenameObject { id, .. } => format!("Rename {}", name_of(id)),
            Command::SetVisibility { id, visible } => {
                if *visible {
                    format!("Show {}", name_of(id))
                } else {
                    format!("Hide {}", name_of(id))
                }
            }
            Command::SetLocked { id, locked } => {
                if *locked {
                    format!("Lock {}", name_of(id))
                } else {
                    format!("Unlock {}", name_of(id))
                }
            }
            Command::SetCamera { id, .. } => format!("Edit camera {}", name_of(id)),
            Command::SetLight { id, .. } => format!("Edit light {}", name_of(id)),
            Command::SetPrimitive { id, .. } => format!("Reshape {}", name_of(id)),
            Command::SetTerrainZScale { id, .. } => {
                format!("Terrain scale {}", name_of(id))
            }
            Command::SetSunAngles { .. } => "Aim the sun".to_owned(),
            Command::FitTerrainRelief { id } => format!("Fit relief of {}", name_of(id)),
            Command::ApplyMapLook { id } => format!("Map look for {}", name_of(id)),
            Command::SetTerrainDem { id, .. } => format!("Change DEM of {}", name_of(id)),
            Command::SetRenderSettings { .. } => "Render settings".to_owned(),
            Command::AddCartographyLayer { layer, .. } => {
                format!("Add layer {}", layer.name)
            }
            Command::RemoveCartographyLayer { id } => {
                format!("Remove layer {}", crate::ids::short_id(id))
            }
            Command::MoveCartographyLayer { .. } => "Reorder layer".to_owned(),
            Command::UpdateCartographyLayer { layer, .. } => {
                format!("Edit layer {}", layer.name)
            }
            Command::SetMeshMaterial { id, .. } => format!("Material of {}", name_of(id)),
            Command::SetGltfNode { id, .. } => format!("Select node of {}", name_of(id)),
            Command::SetChannelKeys { id, channels } => {
                if channels.is_empty() {
                    format!("Clear animation of {}", name_of(id))
                } else {
                    format!("Key {}", name_of(id))
                }
            }
            Command::SetTimelineConfig { .. } => "Timeline settings".to_owned(),
            Command::SetAssetPath { id, .. } => {
                format!("Relink asset {}", crate::ids::short_id(id))
            }
            Command::SetActiveCamera { id } => match id {
                Some(id) => format!("Set active camera {}", name_of(id)),
                None => "Clear active camera".to_owned(),
            },
            Command::CreatePrimitive { primitive, .. } => {
                let label = match primitive {
                    Primitive::Cube { .. } => "Cube",
                    Primitive::Sphere { .. } => "Sphere",
                    Primitive::Plane { .. } => "Plane",
                };
                format!("Add {label}")
            }
            Command::ImportGltf { name, .. } => format!("Import {name}"),
            Command::DuplicateObject { id, .. } => format!("Duplicate {}", name_of(id)),
            Command::AddAsset { asset } => format!("Import {}", asset.name),
            Command::RemoveAsset { .. } => "Remove asset".to_owned(),
            Command::SetViewport { .. } => "Viewport settings".to_owned(),
            Command::SetWorkspace { workspace } => {
                format!("Switch to {} workspace", workspace.label())
            }
            Command::Batch { commands } => commands
                .first()
                .map(|command| command.label(project))
                .unwrap_or_else(|| "Edit".to_owned()),
        }
    }
}
