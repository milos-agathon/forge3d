/**
 * The bridge between the React UI and the Rust shell: `invoke`, one function
 * per command. Outside the desktop shell there is no bridge to speak of — the
 * app shows a "start the shell" notice instead of mounting the editor.
 */

import type {
  AssetIntegrity,
  AutosaveStatus,
  CapabilityTable,
  Command,
  EditResult,
  EngineStatus,
  PlayheadResult,
  Project,
  RecoveryEntry,
  SessionState,
  MeshFilePayload,
  RenderFrame,
  RenderStatus,
  ScenePreset,
  SolarResult,
  RenderView,
  ShellInfo,
  TerrainPreview,
  AnimationFrameRequest,
  WrittenFrame,
  DatasetInfo,
  GraticuleGeometry,
  PolygonGeometry,
  RemoteCogInfo,
  VectorGeometry,
} from "./types";

export interface StudioBridge {
  readonly kind: "tauri" | "browser";
  info(): Promise<ShellInfo>;
  /** Per-parameter capability table, from studio-core. */
  capabilityTable(): Promise<CapabilityTable | null>;
  state(): Promise<SessionState | null>;
  newProject(name?: string | null, preset?: string | null): Promise<SessionState>;
  scenePresets(): Promise<ScenePreset[]>;
  openProject(path: string): Promise<SessionState>;
  saveProject(path?: string | null): Promise<SessionState>;
  closeProject(): Promise<null>;
  applyCommand(command: Command): Promise<EditResult>;
  undo(): Promise<EditResult>;
  redo(): Promise<EditResult>;
  setSelection(ids: string[]): Promise<SessionState>;
  /** Move the playhead and evaluate the timeline into the project (view
   *  state: not undoable, no autosave). Returns the snapshot plus the
   *  clamped frame that was applied. */
  setPlayhead(frame: number): Promise<PlayheadResult>;
  /** Place a directional light from the engine's solar ephemeris. One
   *  undoable transform; the date/place are inputs, not project state. */
  setSunFromSolar(
    id: string,
    latitude: number,
    longitude: number,
    datetimeUtc: string,
  ): Promise<SolarResult>;
  /** Render one frame of an offline sequence to `output` (PNG). The shell
   *  evaluates the timeline on a clone, so the session's playhead does not
   *  move. Desktop shell only. */
  renderAnimationFrame(request: AnimationFrameRequest): Promise<WrittenFrame>;
  autosaveNow(): Promise<unknown>;
  autosaveStatus(): Promise<AutosaveStatus>;
  listRecovery(): Promise<RecoveryEntry[]>;
  restoreRecovery(id: string): Promise<SessionState>;
  discardRecovery(id: string): Promise<void>;
  /** Read a `.gltf`/`.glb` file for import. Desktop shell only. */
  readMeshFile(path: string): Promise<MeshFilePayload>;
  /** Read a PNG/JPEG for a viewport texture. */
  readImageAsset(path: string): Promise<MeshFilePayload>;
  /** Raw Radiance `.hdr` bytes for the Material preview's PMREM. */
  readHdr(path: string): Promise<MeshFilePayload>;
  /** Decoded, downsampled PNG of a raster overlay — the Material preview
   *  drapes this on the terrain (the webview cannot decode GeoTIFF). */
  rasterPreview(path: string, width?: number): Promise<MeshFilePayload>;
  /** Size of a file on disk — asset registration validates non-empty files. */
  fileSize(path: string): Promise<number>;
  /** Existence and size of every referenced asset (read-only). */
  checkAssets(): Promise<AssetIntegrity[]>;
  /** Sample a DEM into a heightfield for the Studio viewport. */
  terrainPreview(path: string, resolution?: number): Promise<TerrainPreview>;
  /** The engine's colormap as 32 linear-RGB stops, for the Material preview's
   *  elevation ramp and the panel's colour strip. */
  colormapStops(name: string): Promise<number[][]>;
  /** The vector layer's feature lines (and, with a label field, its
   *  labelled points), in the DEM's frame, normalised to the preview lattice. */
  vectorGeometry(
    demPath: string,
    vectorPath: string,
    labelField?: string,
    legendField?: string,
  ): Promise<VectorGeometry & { ok: boolean }>;
  /** The polygon layer's rings (exteriors and holes), in the DEM's frame,
   *  normalised to the preview's 0..1 lattice — shares the still's reader. */
  polygonGeometry(
    demPath: string,
    vectorPath: string,
  ): Promise<PolygonGeometry & { ok: boolean; detail?: string }>;
  /** The graticule for the DEM's CRS and bounds (the engine's own grid),
   *  with its coordinate annotations. */
  graticuleGeometry(
    demPath: string,
    intervalDeg: number,
    labels?: boolean,
  ): Promise<GraticuleGeometry & { ok: boolean }>;
  /** Replace the scene terrain's DEM with a raster the user picked — or add
   *  the terrain when the scene has none yet. */
  importTerrainDem(path: string): Promise<SessionState>;
  /** Probe a remote DEM with the engine's native COG reader before any download. */
  remoteCogInfo(url: string): Promise<RemoteCogInfo>;
  /** The engine's dataset registry (bundled + downloadable samples). */
  datasetList(): Promise<DatasetInfo[]>;
  /** Resolve a registry dataset to a local path (the engine fetches it,
   *  hash-verified, and caches it). One IPC for every import site. */
  datasetPath(name: string): Promise<string>;
  /** Add a polygon layer from a GPKG/GeoJSON feature collection. */
  addPolygonLayer(
    path: string,
    colour?: [number, number, number, number],
  ): Promise<SessionState>;
  /** Add an imagery layer to the map's stack. */
  addImageryLayer(path: string, opacity?: number): Promise<SessionState>;
  /** Add a vector layer to the map's stack. The colour is RGBA 0..=1 and the
   *  width is a symbol width in pixels. */
  addVectorLayer(
    path: string,
    colour?: [number, number, number, number],
    widthPx?: number,
  ): Promise<SessionState>;

  // Forge3D workspace adapter
  forge3dProbe(force?: boolean): Promise<EngineStatus>;

  // Embedded renderer (Rendered viewport mode)
  renderFrame(
    width: number,
    height: number,
    quality?: "preview" | "final",
    view?: RenderView,
  ): Promise<RenderFrame>;
  /** Render a full-resolution still; without a path it goes to renders/. */
  renderStill(path?: string | null, view?: RenderView): Promise<RenderFrame>;
  renderStatus(): Promise<RenderStatus>;
  renderStop(): Promise<RenderStatus>;
}

export type { Project };
