import { invoke } from "@tauri-apps/api/core";

import type { StudioBridge } from "./index";
import type {
  AssetIntegrity,
  AutosaveStatus,
  CapabilityTable,
  EditResult,
  EngineStatus,
  MeshFilePayload,
  PlayheadResult,
  RecoveryEntry,
  RenderFrame,
  RenderStatus,
  ScenePreset,
  SessionState,
  SolarResult,
  ShellInfo,
  TerrainPreview,
  WrittenFrame,
  GraticuleGeometry,
  PolygonGeometry,
  VectorGeometry,
} from "./types";

/** The real bridge: Tauri IPC. Argument names match the Rust command params. */
export const tauriBridge: StudioBridge = {
  kind: "tauri",

  info: () => invoke<ShellInfo>("shell_info"),

  capabilityTable: () => invoke<CapabilityTable | null>("capability_table"),

  state: () => invoke<SessionState | null>("project_state"),

  newProject: (name, preset) =>
    invoke<SessionState>("new_project", { name: name ?? null, preset: preset ?? null }),

  scenePresets: () => invoke<ScenePreset[]>("scene_presets"),

  openProject: (path) =>
    invoke<SessionState>("open_project", { path }),

  saveProject: (path) =>
    invoke<SessionState>("save_project", { path: path ?? null }),

  closeProject: () => invoke<null>("close_project"),

  applyCommand: (command) =>
    invoke<EditResult>("apply_command", { command }),

  undo: () => invoke<EditResult>("undo"),

  redo: () => invoke<EditResult>("redo"),

  setSelection: (ids) => invoke<SessionState>("set_selection", { ids }),

  setPlayhead: (frame) => invoke<PlayheadResult>("set_playhead", { frame }),

  setSunFromSolar: (id, latitude, longitude, datetimeUtc) =>
    invoke<SolarResult>("set_sun_from_solar", {
      id,
      latitude,
      longitude,
      datetimeUtc,
    }),

  renderAnimationFrame: (request) =>
    invoke<WrittenFrame>("render_animation_frame", {
      frame: request.frame,
      width: request.width,
      height: request.height,
      quality: request.quality ?? null,
      output: request.output,
    }),

  autosaveNow: () => invoke("autosave_now"),

  autosaveStatus: () => invoke<AutosaveStatus>("autosave_status"),

  listRecovery: () => invoke<RecoveryEntry[]>("list_recovery"),

  restoreRecovery: (id) =>
    invoke<SessionState>("restore_recovery", { id }),

  discardRecovery: (id) => invoke<void>("discard_recovery", { id }),

  readMeshFile: (path) => invoke<MeshFilePayload>("read_mesh_file", { path }),

  readImageAsset: (path) => invoke<MeshFilePayload>("read_image_asset", { path }),

  readHdr: (path) => invoke<MeshFilePayload>("read_hdr", { path }),

  rasterPreview: (path, width) =>
    invoke<MeshFilePayload>("raster_preview", { path, width: width ?? null }),

  fileSize: (path) => invoke<number>("file_size", { path }),

  checkAssets: () => invoke<AssetIntegrity[]>("check_assets"),

  terrainPreview: (path, resolution) =>
    invoke<TerrainPreview>("terrain_preview", { path, resolution: resolution ?? null }),

  colormapStops: (name) => invoke<number[][]>("colormap_stops", { name }),
  vectorGeometry: (demPath, vectorPath, labelField, legendField) =>
    invoke<VectorGeometry & { ok: boolean }>("vector_geometry", {
      demPath,
      vectorPath,
      labelField,
      legendField,
    }),
  polygonGeometry: (demPath, vectorPath) =>
    invoke<PolygonGeometry & { ok: boolean }>("polygon_geometry", {
      demPath,
      vectorPath,
    }),
  graticuleGeometry: (demPath, intervalDeg, labels) =>
    invoke<GraticuleGeometry & { ok: boolean }>("graticule_geometry", {
      demPath,
      intervalDeg,
      labels,
    }),

  importTerrainDem: (path) => invoke<SessionState>("import_terrain_dem", { path }),

  addImageryLayer: (path, opacity) =>
    invoke<SessionState>("add_imagery_layer", { path, opacity: opacity ?? null }),

  addVectorLayer: (path, colour, widthPx) =>
    invoke<SessionState>("add_vector_layer", {
      path,
      colour: colour ?? null,
      widthPx: widthPx ?? null,
    }),

  addPolygonLayer: (path, colour) =>
    invoke<SessionState>("add_polygon_layer", { path, colour: colour ?? null }),

  forge3dProbe: (force) => invoke<EngineStatus>("forge3d_probe", { force: force ?? false }),

  renderFrame: (width, height, quality, view) =>
    invoke<RenderFrame>("render_frame", {
      width,
      height,
      quality: quality ?? "final",
      view: view ?? null,
    }),

  renderStill: (path, view) =>
    invoke<RenderFrame>("render_still", { path: path ?? null, view: view ?? null }),

  renderStatus: () => invoke<RenderStatus>("render_status"),

  renderStop: () => invoke<RenderStatus>("render_stop"),
};
