//! Canonical Forge3D Studio project model.
//!
//! This crate is the single source of truth for a Studio project. It has no
//! desktop, GPU, or Forge3D dependency on purpose: the Tauri shell, any future
//! viewport adapter, and the Forge3D workspace bridge all read and write the
//! same [`Project`] through the same [`Command`] layer.
//!
//! Layout:
//!
//! * [`project`] — the versioned project document (`studio3d` JSON, schema v2).
//! * [`scene`] — the scene graph: objects, transforms, cameras, lights.
//! * [`command`] — every mutation as a reversible command.
//! * [`editor`] — an editing session: revision counter, undo/redo, dirty state.
//! * [`persist`] — atomic save and version-gated load.
//! * [`autosave`] — crash recovery snapshots.
//! * [`session`] — one open project: open, save, autosave, recovery.
//! * [`workspace`] — what a project can carry into the Forge3D workspace.
//!
//! ## Why commands instead of direct mutation
//!
//! The renderer is a projection of scene state, never the owner of it. Every
//! user action goes through [`Command`], which returns the inverse command it
//! pushed onto the undo stack, so a workspace adapter can be told which object
//! IDs changed and re-sync just those.
//!
//! ## Wire format
//!
//! JSON is `camelCase` throughout. The TypeScript mirror lives in
//! `studio/src/bridge/types.ts`; `tests/schema_contract.rs` locks the
//! serialized key shape so the two cannot drift silently.

use std::time::{SystemTime, UNIX_EPOCH};

pub mod autosave;
pub mod cartography;
pub mod command;
pub mod editor;
pub mod error;
pub mod ids;
pub mod integrity;
pub mod math;
pub mod persist;
pub mod project;
pub mod scene;
pub mod session;
pub mod timeline;
pub mod workspace;

pub use autosave::{AutosaveStore, RecoveryEntry, DEFAULT_AUTOSAVE_KEEP};
pub use cartography::{
    default_stack, CartographyLayer, CartographyLayerId, CartographyLayerKind, CartographyStack,
};
pub use command::{Command, CommandError};
pub use editor::{Applied, Editor, HistoryEntry, SessionState, DEFAULT_HISTORY_LIMIT};
pub use error::StudioError;
pub use integrity::{asset_integrity, AssetIntegrity};
pub use math::{Transform, Vec3};
pub use persist::{load, save_atomic, to_json_pretty};
pub use project::{
    Asset, AssetKind, BloomSettings, FogSettings, Project, RenderSettings, ShadowTechnique,
    SkyModel, StillAov, TerrainSeed, TerrainSeedOverlay, ViewportMode, ViewportState,
    WaterSettings, Workspace, FORMAT, SCHEMA_VERSION,
};
pub use scene::{
    CameraObject, LightKind, LightObject, Material, MeshObject, MeshSource, ObjectKind, Primitive,
    Scene, SceneObject, TerrainObject,
};
pub use session::{AutosaveReport, Session, AUTOSAVE_INTERVAL_SECONDS};
pub use timeline::{
    evaluate, sample_keys, Channel, ChannelKeys, Interpolation, Keyframe, ObjectTrack, Timeline,
    DEFAULT_FRAME_RATE, DEFAULT_LENGTH,
};
pub use workspace::{
    camera_render_view, compute_render_view, environment_handoff, handoff, parameter_capabilities,
    CapabilityLevel, CapabilityTable, Handoff, ParameterCapability, RenderView, SunAngles,
    TerrainView,
};

/// Wall-clock milliseconds since the Unix epoch.
///
/// Timestamps are project metadata only; nothing in the model compares or
/// orders on them, so a clock that jumps backwards cannot corrupt a project.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}
