/**
 * TypeScript mirror of the Rust wire format in `crates/studio-core`.
 *
 * Hand-written on purpose. `crates/studio-core/tests/schema_contract.rs` pins
 * the Rust side of these shapes, so a rename there fails a test instead of
 * silently producing `undefined` in the inspector.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Transform {
  translation: Vec3;
  rotationDeg: Vec3;
  scale: Vec3;
}

export type Primitive =
  | { type: "cube"; size: number }
  | { type: "sphere"; radius: number }
  | { type: "plane"; size: number };

export type PrimitiveShape = Primitive["type"];

export interface CameraObject {
  fovDeg: number;
  near: number;
  far: number;
}

export type LightKind = "directional" | "point";

export interface LightObject {
  kind: LightKind;
  /** Linear RGB, 0..1. */
  color: [number, number, number];
  intensity: number;
}

export type AssetKind = "gltf" | "glb" | "dem" | "orthophoto" | "hdr" | "vector";

/** An imported source file. Stored relative to the project where possible. */
export interface Asset {
  id: string;
  name: string;
  path: string;
  kind: AssetKind;
  byteSize: number;
}

/** One asset's on-disk state, checked at open and after relink. */
export interface AssetIntegrity {
  id: string;
  name: string;
  path: string;
  kind: AssetKind;
  /** Size recorded when the asset was imported or relinked. */
  expectedBytes: number;
  /** Size on disk, when the file could be read. */
  actualBytes: number | null;
  /** Why it could not be read (missing, permissions), when it could not. */
  error: string | null;
}

export type MeshSource =
  | { type: "primitive"; primitive: Primitive }
  | { type: "gltf"; asset: string; node: string | null };

/** A DEM-backed terrain object (Forge3D-native). */
export interface TerrainObject {
  asset: string;
  zScale: number;
  albedo: string | null;
  dimensions: [number, number] | null;
  /** Viewer-basis world origin `[mapX, -mapY]`, metres. */
  worldOriginXz: [number, number] | null;
  /** Viewer-basis world span, metres (the viewer's terrain world units). */
  worldSpanXz: [number, number] | null;
  elevationRange: [number, number] | null;
}

// ------------------------------------------------------------- cartography

/**
 * One layer of the map. The map is a stack of these, drawn back to front:
 * the terrain's relief at the bottom, the ramp and the imagery over it, the
 * vector layers and their labels, the plate at the front.
 *
 * The shape is flat because the wire format is flat (the kind's fields sit
 * beside the layer's own); `type` says which fields are present.
 */
export interface CartographyLayer {
  id: string;
  name: string;
  /** The eye: a hidden layer is not drawn. */
  visible: boolean;
  type:
    | "relief"
    | "elevationRamp"
    | "imagery"
    | "vector"
    | "polygon"
    | "labels"
    | "graticule"
    | "plate";
  /** elevationRamp */
  colormap?: string;
  strength?: number;
  legend?: boolean;
  /** imagery and vector */
  asset?: string;
  opacity?: number;
  preserveColors?: boolean;
  despeckle?: boolean;
  /** vector */
  colour?: [number, number, number, number];
  widthPx?: number;
  /** vector: attribute to colour features by, one swatch per class on the plate. */
  legendField?: string;
  /** labels */
  source?: string;
  field?: string;
  sizePx?: number;
  /** graticule */
  intervalDeg?: number;
  labels?: boolean;
  /** plate */
  title?: string;
  caption?: string;
  northArrow?: boolean;
  scaleBar?: boolean;
}


/** Internally tagged enum: the payload is flattened next to `type`. */
export type ObjectKind =
  | { type: "mesh"; source: MeshSource; material: Material }
  | ({ type: "terrain" } & TerrainObject)
  | ({ type: "camera" } & CameraObject)
  | ({ type: "light" } & LightObject);

export interface SceneObject {
  id: string;
  name: string;
  parent: string | null;
  transform: Transform;
  visible: boolean;
  locked: boolean;
  kind: ObjectKind;
}

export interface Scene {
  id: string;
  name: string;
  objects: SceneObject[];
  activeCamera: string | null;
}

export type Workspace = "studio" | "forge3d";
export type ViewportMode = "solid" | "rendered" | "material";

/** Renderer quality preset: preview/material are interactive, final is a still. */
export type RenderQuality = "preview" | "final";

export interface ViewportState {
  mode: ViewportMode;
  gridVisible: boolean;
  /** Slope-angle hillshade painted onto the terrain in the viewport. */
  hillshade: boolean;
}

export interface RenderSettings {
  width: number;
  height: number;
  exposure: number;
  shadows: boolean;
  /** Sky background in Rendered preview and still frames. */
  sky: boolean;
  /** Environment/IBL intensity multiplier; 0 turns the environment light off. */
  iblIntensity: number;
  /** Radiance .hdr environment asset for image-based lighting, or null for
   *  the built-in studio sky. Shared by the Material preview and Rendered. */
  environment: string | null;
  /** HDRI backdrop brightness (0..1): the Material preview's backdrop and the
   *  Rendered sky exposure. 0 hides it. */
  environmentBackgroundIntensity: number;
  /** Final-quality AOV export for the still: `<name>_albedo.png`,
   *  `<name>_normal.png`, `<name>_depth.png` next to it. */
  stillAov: { albedo: boolean; normal: boolean; depth: boolean };
  /** Final-quality OIDN denoise of the beauty pass. Needs an OIDN runtime
   *  beside the engine; the render fails loudly without one. */
  stillDenoise: boolean;
  /** Shadow filter: PCF is the cheap inset filter, PCSS adds a blocker search. */
  shadowTechnique: ShadowTechnique;
  /** Shadow-map edge length in texels (256-8192); bigger is crisper and slower. */
  shadowResolution: number;
  /** Terrain self-shadowing: ray-marches the heightfield toward the sun, so a
   *  ridge shadows the ground behind it. */
  sunVisibility: boolean;
  /** How far each sun ray searches, in metres (100..20000). */
  sunVisibilityDistance: number;
  /** Steps per sun ray (8..64). */
  sunVisibilitySteps: number;
  /** Analytic sky model for the backdrop when no HDRI is set. */
  skyModel: SkyModel;
  /** Hosek/Preetham turbidity (haze), 1..=10. */
  skyTurbidity: number;
  /** Ground reflectance the sky bounces back, 0..=1. */
  skyGroundAlbedo: number;
  /** Exposure for the analytic sky pass (backdrop only), 0..=4. */
  skyExposure: number;
  /** Water on the terrain: the level and the slope threshold the render
   *  service turns into the engine's water mask. */
  water: WaterSettings;
  /** Elevation colormap for the terrain, by engine name; "" means none. */
  colormap: string;
  /** How strongly the colormap covers the terrain material, 0..=1. */
  colormapStrength: number;
  /** Draw the elevation legend in the frame's corner (needs a colormap). */
  legend: boolean;
  /** Export a map plate next to the still: cut out on white, titled, legend. */
  plate: boolean;
  /** The plate's title; "" means the DEM's file name. */
  plateTitle: string;
  /** Atmospheric fog on the terrain. */
  fog: FogSettings;
  /** The cloud shadows on the terrain. */
  clouds: CloudSettings;
  /** Bloom on the finished frame: the bright pixels glow. */
  bloom: BloomSettings;
  /** Vignette: corner darkening, 0..=1. */
  vignette: number;
}

/** Atmospheric fog on the terrain: the engine applies height-based
 *  exponential fog after the PBR pass and before the tonemap. */
/** The cloud shadows on the terrain: the engine's own cloud field, projected
 *  along the sun onto the ground. */
export interface CloudSettings {
  enabled: boolean;
  shadowsEnabled: boolean;
  /** How much of the sky is covered, 0..1. */
  coverage: number;
  /** How dense the covered part is, 0..1. */
  density: number;
  /** How dark the shadows are, 0..1. */
  shadowStrength: number;
  /** The engine's field resolution. */
  quality: "low" | "medium" | "high" | "ultra";
}

export interface FogSettings {
  enabled: boolean;
  /** Fog density; 0 is no fog. */
  density: number;
  /** How far up the terrain the fog climbs, as a fraction of its height. */
  reach: number;
  /** The colour the fog scatters, linear RGB 0..=1. White is neutral. */
  inscatter: [number, number, number];
}

/** Bloom on the finished frame. Display-referred: the threshold is a display
 *  brightness, not an HDR radiance. */
export interface BloomSettings {
  enabled: boolean;
  /** Display brightness above which pixels glow, 0..=1. */
  threshold: number;
  /** Width of the threshold transition, 0..=1. */
  softness: number;
  /** How much glow is added, 0..=2. */
  intensity: number;
}

/** Water on the terrain. The terrain path takes a mask, not a level: the
 *  render service derives the mask from the DEM with the engine's own
 *  `derive_water_mask`. */
export interface WaterSettings {
  /** Flood the low, flat ground. */
  enabled: boolean;
  /** Water surface elevation in the DEM's own metres; 0 = auto (the lowest
   *  15% of the terrain). */
  level: number;
  /** Metres of rise per DEM cell; steeper ground stays dry. */
  slopeThreshold: number;
  /** The water surface's roughness: higher is a softer, wider sun highlight. */
  roughness: number;
}

export type SkyModel = "hosek" | "preetham" | "aether";

export type ShadowTechnique = "pcf" | "pcss";

/** Studio-side material. The terrain renderer does not consume it. */
export interface Material {
  baseColor: [number, number, number];
  roughness: number;
  metallic: number;
  texture: string | null;
}

export interface Project {
  format: string;
  schemaVersion: number;
  id: string;
  name: string;
  createdUnixMs: number;
  updatedUnixMs: number;
  scene: Scene;
  assets: Asset[];
  selection: string[];
  workspace: Workspace;
  viewport: ViewportState;
  render: RenderSettings;
  timeline: Timeline;
  /** The map's layer stack, in draw order. */
  cartography: CartographyLayer[];
}

// ---------------------------------------------------------------- timeline

/** How the value moves between two keyframes. */
export type Interpolation = "linear" | "constant" | "smooth";

/** One animatable scalar: the nine transform components + camera FOV. */
export type Channel =
  | "translationX"
  | "translationY"
  | "translationZ"
  | "rotationX"
  | "rotationY"
  | "rotationZ"
  | "scaleX"
  | "scaleY"
  | "scaleZ"
  | "fovDeg";

export interface Keyframe {
  frame: number;
  value: number;
  interpolation: Interpolation;
}

export interface ChannelKeys {
  channel: Channel;
  /** Sorted by frame, strictly increasing. */
  keys: Keyframe[];
}

export interface ObjectTrack {
  objectId: string;
  channels: ChannelKeys[];
}

/** The project's animation timeline (studio-core `Timeline`). */
export interface Timeline {
  frameRate: number;
  length: number;
  /** Playhead position. View state: scrubbing is not an edit. */
  currentFrame: number;
  tracks: ObjectTrack[];
}

/** What `setSunFromSolar` returns: the ephemeris and the new snapshot. */
export interface SolarResult {
  /** Azimuth from north toward east, degrees (engine display convention). */
  azimuthDeg: number;
  elevationDeg: number;
  daytime: boolean;
  /** Rotation written to the light (degrees, ZYX). */
  rotationDeg: { x: number; y: number; z: number };
  state: SessionState;
}

/** What `setPlayhead` returns: the new snapshot plus the clamped frame. */
export interface PlayheadResult {
  frame: number;
  state: SessionState;
}

/** One entry of the engine's dataset registry (fetched or baked). */
export interface DatasetInfo {
  name: string;
  kind: string | null;
  filename: string | null;
  description: string | null;
  /** Present only when a local file resolves — nothing would be downloaded. */
  path?: string;
}

/** One frame of an offline sequence export. */
export interface AnimationFrameRequest {
  frame: number;
  width: number;
  height: number;
  quality?: RenderQuality;
  /** Absolute output path (the frontend picks the directory). */
  output: string;
}

/** One rendered frame of a sequence. */
export interface WrittenFrame {
  path: string;
  ms: number;
  /** The clamped timeline frame actually rendered. */
  evaluatedFrame: number;
}

export interface SessionState {
  project: Project;
  revision: number;
  dirty: boolean;
  path: string | null;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel: string | null;
  redoLabel: string | null;
  historyLen: number;
}

export interface Applied {
  label: string;
  revision: number;
  affected: string[];
}

export interface EditResult {
  applied: Applied;
  state: SessionState;
}

export interface RecoveryEntry {
  id: string;
  writtenUnixMs: number;
  sourcePath: string | null;
  projectId: string;
  projectName: string;
  objectCount: number;
  path: string;
}

export interface AutosaveStatus {
  intervalSeconds: number;
  lastWriteUnixMs: number | null;
  lastError: string | null;
  directory: string;
}

export interface ShellInfo {
  name: string;
  version: string;
}

/** A `.gltf`/`.glb` file read by the shell, base64 for the webview. */
export interface MeshFilePayload {
  path: string;
  name: string;
  kind: AssetKind;
  byteSize: number;
  base64: string;
}

// ---------------------------------------------------------------- workspaces

export type CapabilityLevel = "full" | "partial" | "unsupported";

/** One material/light/camera parameter and what currently drives it. */
export interface ParameterCapability {
  parameter: string;
  level: CapabilityLevel;
  detail: string;
}

/** Per-parameter capability table (studio-core is the single source). */
export interface CapabilityTable {
  light: ParameterCapability[];
  environment: ParameterCapability[];
  meshMaterial: ParameterCapability[];
  terrainMaterial: ParameterCapability[];
  camera: ParameterCapability[];
}

/** A feature line from the vector preview IPC, in 0..1 terrain UV. */
export type PreviewLine = [number, number][];

/** A labelled point from the vector preview IPC, in 0..1 terrain UV. */
export interface PreviewPoint {
  u: number;
  v: number;
  text: string;
}

/** A coordinate annotation from the graticule preview IPC, in 0..1 terrain UV. */
export interface PreviewAnnotation {
  kind: string;
  text: string;
  u: number;
  v: number;
}

/** A polygon feature's rings, normalised to the preview lattice. */
export interface PolygonGeometry {
  /** One entry per polygon: the exterior ring, then the holes. */
  exteriors: [number, number][][];
  holes: [number, number][][][];
}

/** The geometry a visible vector layer draws, normalised to the preview lattice. */
export interface VectorGeometry {
  lines: PreviewLine[];
  points: PreviewPoint[];
  /** The legend attribute's value per line (aligned with `lines`); empty without one. */
  categories?: string[];
  /** value -> RGB; absent categories fall back to the layer colour. */
  palette?: Record<string, [number, number, number]>;
}

/** The graticule for one grid layer: the grid and its coordinate annotations. */
export interface GraticuleGeometry {
  lines: PreviewLine[];
  labels: PreviewAnnotation[];
}

/** A DEM sampled for the Studio viewport (heights are row-major, bottom-up). */
export interface TerrainPreview {
  ok: boolean;
  width: number;
  height: number;
  minElevation: number;
  maxElevation: number;
  extentMeters: [number, number];
  originXz: [number, number];
  spanXz: [number, number];
  resolution: number;
  heights: number[];
}

export interface EngineStatus {
  available: boolean;
  python: string;
  packageVersion: string | null;
  viewerBinary: string | null;
  /** wgpu backend and GPU adapter, when the probe reached the device. */
  backend: string | null;
  adapter: string | null;
  /** True when the probed interpreter can denoise with OIDN. */
  denoise?: boolean | null;
  /** True when the engine fell back to a software (CPU) adapter. */
  software?: boolean | null;
  /** Device features the adapter granted (`forge3d.capabilities().granted`). */
  features?: string[] | null;
  detail: string;
}

/** A starter scene offered when creating a project. */
export interface ScenePreset {
  id: string;
  name: string;
  description: string;
  demName: string | null;
  grid: [number, number] | null;
  elevationRange: [number, number] | null;
  worldSpan: [number, number] | null;
  available: boolean;
}

/** Camera override from the viewport, so frames match the solid view. */
export interface RenderView {
  phiDeg: number;
  thetaDeg: number;
  /** Orbit distance in the renderer's terrain units (metres). */
  radius: number;
  /**
   * Aim point in the renderer's mesh frame: metres, terrain centred on the
   * origin, Z up. The viewport maps its camera onto this frame before
   * rendering, so the frame matches what the user is looking at.
   */
  target: [number, number, number];
  /** Vertical field of view, degrees. */
  fovDeg: number;
}

export interface RenderFrame {
  width: number;
  height: number;
  /** Render time in milliseconds. */
  ms: number;
  /** Which engine path made the frame: `concordia` (the quality surface, with
   *  the screen-space effects) or `plain` (the AOV renderer). */
  renderPath: string;
  /** Whether the frame is temporally filtered (the engine's TAA): the quality
   *  surface's frame is the converged one, not byte-stable. */
  temporal: boolean;
  /** Base64 PNG. */
  png: string;
  /** Where a still was written, when it went to disk. */
  path: string | null;
}

export interface RenderStatus {
  running: boolean;
  dem: string | null;
  diagnostics: string[];
}

export type Command =
  | { type: "createObject"; object: SceneObject; index?: number | null }
  | { type: "createPrimitive"; primitive: Primitive; name?: string | null }
  | { type: "deleteObject"; id: string }
  | { type: "duplicateObject"; id: string; name?: string | null }
  | { type: "addAsset"; asset: Asset }
  | {
      type: "importGltf";
      name: string;
      path: string;
      kind: AssetKind;
      byteSize: number;
      node?: string | null;
    }
  | { type: "removeAsset"; id: string }
  | { type: "setTransform"; id: string; transform: Transform }
  | { type: "renameObject"; id: string; name: string }
  | { type: "setVisibility"; id: string; visible: boolean }
  | { type: "setLocked"; id: string; locked: boolean }
  | { type: "setCamera"; id: string; camera: CameraObject }
  | { type: "setLight"; id: string; light: LightObject }
  | { type: "setPrimitive"; id: string; primitive: Primitive }
  | { type: "setMeshMaterial"; id: string; material: Material }
  | { type: "setGltfNode"; id: string; node: string | null }
  | { type: "setChannelKeys"; id: string; channels: ChannelKeys[] }
  | { type: "setTimelineConfig"; frameRate: number; length: number }
  | { type: "setRenderSettings"; render: RenderSettings }
  | { type: "setAssetPath"; id: string; path: string; byteSize: number }
  | { type: "setTerrainZScale"; id: string; zScale: number }
  /** Set the vertical scale to the DEM's own 6%-relief default. */
  | { type: "fitTerrainRelief"; id: string }
  /** Apply the map look: relief, map camera, north-west sun, hillshade. */
  | { type: "applyMapLook"; id: string }
  | {
      type: "setTerrainDem";
      id: string;
      asset: string;
      dimensions?: [number, number] | null;
      elevationRange?: [number, number] | null;
      worldOriginXz?: [number, number] | null;
      worldSpanXz?: [number, number] | null;
      /** Omitted means "the new DEM's own default". */
      zScale?: number | null;
    }
  | { type: "addCartographyLayer"; layer: CartographyLayer; index?: number }
  | { type: "removeCartographyLayer"; id: string }
  | { type: "moveCartographyLayer"; id: string; to: number }
  | { type: "updateCartographyLayer"; id: string; layer: CartographyLayer }
  | { type: "setActiveCamera"; id: string | null }
  | { type: "setViewport"; viewport: ViewportState }
  | { type: "setWorkspace"; workspace: Workspace }
  | { type: "batch"; commands: Command[] };

/** Active manipulation tool, Blender-style. */
export type Tool = "select" | "move" | "rotate" | "scale";

/** Gizmo orientation: world axes or the object's own axes. */
export type ToolSpace = "world" | "local";

export const DEFAULT_PRIMITIVES: Record<PrimitiveShape, Primitive> = {
  cube: { type: "cube", size: 2 },
  sphere: { type: "sphere", radius: 1 },
  plane: { type: "plane", size: 2 },
};

export function objectKindLabel(kind: ObjectKind): string {
  switch (kind.type) {
    case "mesh":
      return kind.source.type === "primitive"
        ? kind.source.primitive.type
        : "imported";
    case "terrain":
      return "terrain";
    case "camera":
      return "camera";
    case "light":
      return `${kind.kind} light`;
  }
}

