import { createContext, useContext } from "react";

import type { RenderJob } from "./renderQueue";
import type { RecipeId } from "./recipes";
import type {
  Applied,
  AssetIntegrity,
  AutosaveStatus,
  CartographyLayer,
  Command,
  EngineStatus,
  PrimitiveShape,
  RecoveryEntry,
  RenderFrame,
  ScenePreset,
  SessionState,
  ShellInfo,
  SolarResult,
  Tool,
  ToolSpace,
  ViewportMode,
} from "../bridge/types";

export interface SessionApi {
  info: ShellInfo | null;
  state: SessionState | null;
  autosave: AutosaveStatus | null;
  recovery: RecoveryEntry[];
  error: string | null;
  notice: string | null;
  busy: boolean;
  /** Active manipulation tool. UI state, not project state. */
  tool: Tool;
  setTool(tool: Tool): void;
  /** The layer the cartography panel is editing. UI state. */
  selectedLayerId: string | null;
  /** Gizmo orientation. UI state, not project state. */
  space: ToolSpace;
  setSpace(space: ToolSpace): void;
  /** Look through the scene's active camera instead of the free view. */
  cameraView: boolean;
  setCameraView(enabled: boolean): void;
  /** Fly camera: WASD moves the viewport camera, drag looks around. */
  flyMode: boolean;
  setFlyMode(enabled: boolean): void;
  /** Forge3D engine discovery (renderer + probe). */
  engine: EngineStatus | null;
  /** Heightfield resolution used by the Solid terrain preview. */
  terrainResolution: number;
  setTerrainResolution(resolution: number): void;
  /** Last embedded-renderer frame (Rendered viewport mode). */
  render: {
    frame: RenderFrame | null;
    busy: boolean;
    error: string | null;
  };
  /** Frame-sequence export progress, or null when idle. */
  exporting: { done: number; total: number } | null;
  /** On-disk state of every referenced asset (advisory). */
  assetIntegrity: AssetIntegrity[];
  /** Offline render jobs: stills and frame sequences, FIFO. */
  renderJobs: RenderJob[];
  /** Starter scenes for the New Project dialog (empty until probed). */
  presets: ScenePreset[];
  newProjectOpen: boolean;
  actions: {
    requestNewProject(): void;
    dismissNewProject(): void;
    createProject(presetId: string | null): Promise<void>;
    newProject(): Promise<void>;
    openProject(): Promise<void>;
    saveProject(saveAs?: boolean): Promise<void>;
    closeProject(): Promise<void>;
    apply(command: Command): Promise<Applied | null>;
    undo(): Promise<void>;
    redo(): Promise<void>;
    select(ids: string[]): Promise<void>;
    /** Move the playhead and evaluate the timeline (view state: not an edit). */
    setPlayhead(frame: number): Promise<void>;
    /** Place the selected directional light from the engine's solar
     *  ephemeris. Returns the computed angles for display. */
    setSunFromSolar(
      id: string,
      latitude: number,
      longitude: number,
      datetimeUtc: string,
    ): Promise<SolarResult | null>;
    addPrimitive(shape: PrimitiveShape): Promise<void>;
    duplicateSelection(): Promise<void>;
    deleteSelection(): Promise<void>;
    importMesh(): Promise<void>;
    refreshEngine(force?: boolean): Promise<void>;
    setViewportMode(mode: ViewportMode): Promise<void>;
    setHillshade(hillshade: boolean): Promise<void>;
    renderViewport(
      width: number,
      height: number,
      quality?: "preview" | "final",
    ): Promise<void>;
    /** Forget the last Rendered frame (mode switch / project change). */
    clearRenderFrame(): void;
    /** Pick and import a DEM, replacing the terrain's source. */
    importTerrainDem(): Promise<void>;
    /** Pick a GeoPackage/GeoJSON of lines and attach it as the terrain's
     *  vector layer. */
    /** Select the layer the cartography panel is editing. */
    selectCartographyLayer(id: string | null): void;
    addVectorLayer(): Promise<void>;
    addImageryLayer(): Promise<void>;
    addPolygonLayer(path?: string, colour?: [number, number, number, number]): Promise<void>;
    addCartographyLayer(layer: CartographyLayer, index?: number): Promise<void>;
    addRampLayer(colormap: string): Promise<void>;
    addPlateLayer(title: string): Promise<void>;
    addGraticuleLayer(): Promise<void>;
    addLabelLayer(source: string, field: string): Promise<void>;
    updateCartographyLayer(layer: CartographyLayer): Promise<void>;
    removeCartographyLayer(id: string): Promise<void>;
    moveCartographyLayer(id: string, to: number): Promise<void>;
    /** The map look in one undoable step: relief, map camera, north-west sun,
     *  ray-marched hillshade — and the viewport switches to the camera. */
    applyMapLook(): Promise<void>;
    /** Apply a named look recipe (one undoable batch). */
    applyRecipe(id: RecipeId): Promise<void>;
    /** Pick an albedo texture (PNG/JPEG) for the selected mesh. */
    importMaterialTexture(): Promise<void>;
    /** Remove the selected mesh's texture. */
    clearMaterialTexture(): Promise<void>;
    /** Pick a Radiance .hdr environment for Material and Rendered lighting. */
    importEnvironment(): Promise<void>;
    /** Drop the project HDRI and fall back to the built-in studio sky. */
    clearEnvironment(): Promise<void>;
    /** Re-point an asset at a new file. */
    relinkAsset(assetId: string): Promise<void>;
    /** Re-check every asset's existence and size on disk. */
    refreshAssetIntegrity(): Promise<void>;
    /** Render a full-resolution still into the project's renders/ directory. */
    renderStill(): Promise<void>;
    /** Render the whole timeline as a PNG sequence into `directory`.
     *  One IPC per frame: progress shows in `exporting`, and `cancelExport`
     *  stops between frames. The shell evaluates a clone per frame, so the
     *  session's playhead does not move. Desktop shell only. */
    exportFrames(directory: string): Promise<void>;
    /** Stop the running frame-sequence export between frames. */
    cancelExport(): void;
    /** Cancel one queued job, or stop a running sequence between frames. */
    cancelRenderJob(id: string): void;
    /** Drop done/failed/cancelled jobs from the list. */
    clearFinishedJobs(): void;
    restoreRecovery(id: string): Promise<void>;
    discardRecovery(id: string): Promise<void>;
    dismissError(): void;
    /** Short status message (used by viewport shortcuts). */
    notify(message: string): void;
  };
}

export const SessionContext = createContext<SessionApi | null>(null);



/** Access the session. Lives in its own module so `session.tsx` only exports a
 * component (Vite's Fast Refresh refuses modules that mix the two). */
export function useSession(): SessionApi {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
