import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";

import { bridge } from "../bridge/env";
import type {
  CameraObject,
  LightObject,
  Primitive,
  SceneObject,
  Transform,
  ViewportMode,
} from "../bridge/types";
import { ViewportToolbar } from "../components/ViewportToolbar";
import { useSession } from "../state/session-context";
import { AxisGizmo, type Axis } from "./AxisGizmo";
import type { CartographyLayer, Material, TerrainPreview } from "../bridge/types";
import {
  base64ToArrayBuffer,
  clearAssetCaches,
  disposeTree,
  loadAssetScene,
} from "./assets";
import { viewStore } from "./viewStore";
import {
  graticuleLayers,
  labelLayers,
  polygonLayers,
  topImagery,
  topRamp,
  vectorLayers,
} from "../state/cartography";
import { lightingRig, type LightingRig } from "./lighting";
import { rampColor } from "./colormap";
import { TERRAIN_VIEW_SIZE, computeRenderView } from "./viewMath";
import {
  anchorWorld,
  drapeLift,
  drapePolylines,
  expandPairs,
} from "./cartography";
import { buildPolygonMesh } from "./polygonMesh";

const DEG = Math.PI / 180;
/** Fly-camera keys: WASD in the view plane, Q/E down and up, Shift to sprint. */
const FLY_KEYS = new Set(["w", "a", "s", "d", "q", "e", "shift"]);
const ACCENT = 0xf0a94a;
/**
 * Solid is a Blender-style clay view: no albedo, no textures, no material
 * colour. Shape comes from the scene light only.
 */
const CLAY_COLOUR = 0x8b95a4;
const CAMERA_COLOUR = 0x6fa8ff;
const SOLID_BACKGROUND = 0x101318;
/** Material preview backdrop: the studio HDRI provides the light, this the
 *  neutral surround, so nothing hides the albedo. */
const MATERIAL_BACKGROUND = 0x1c2129;
const SOLID_BG = new THREE.Color(SOLID_BACKGROUND);
const MATERIAL_BG = new THREE.Color(MATERIAL_BACKGROUND);
/** Built-in studio sky — the same gradient the render service writes as its
 *  studio HDR, so the preview and Rendered frames share one environment. */
const SKY_ZENITH = [0.25, 0.40, 0.75] as const;
const SKY_HORIZON = [0.85, 0.72, 0.55] as const;
const SKY_GROUND = [0.10, 0.09, 0.08] as const;
const SKY_GAIN = 1.6;

type Quaternion = [number, number, number, number];

/** A project HDRI decoded into an equirect source plus its prefiltered PMREM. */
interface LoadedEnvironment {
  path: string;
  /** Equirectangular half-float texture — the PMREM input. */
  source: THREE.Texture;
  /** Display-referred 8-bit copy used as the backdrop when the GPU cannot
   *  filter half-float textures (some WebKit/ANGLE webviews). */
  background: THREE.Texture;
  /** PMREM output: what `scene.environment` wants. */
  environment: THREE.Texture;
  target: THREE.WebGLRenderTarget;
}

interface Runtime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  /** Cartography lines (vector, graticule) draped on the terrain; cleared and rebuilt per stack change. */
  overlay: THREE.Group;
  /** HTML labels for the graticule annotations and the named features. */
  labelLayer: HTMLDivElement | null;
  /** Line materials, whose resolution follows the canvas size in pixels. */
  lineMaterials: LineMaterial[];
  /** The shadow rig values the renderer is currently configured with. */
  shadowKey: string;
  orbit: OrbitControls;
  gizmo: TransformControls;
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  grid: THREE.GridHelper;
  objects: Map<string, THREE.Group>;
  helpers: THREE.BoxHelper[];
  /** Selection + camera-view identity the current helpers were built for. */
  helperKey: string;
  /** Built-in sky gradient, used until a project HDRI replaces it. */
  studioEnvironment: THREE.Texture | null;
  /** Environment lighting the scene right now (PMREM output). */
  environment: THREE.Texture | null;
  /** HDRI source texture for the Material background, when one is loaded. */
  environmentBackground: THREE.Texture | null;
  pmrem: THREE.PMREMGenerator | null;
  /** GPU capability probe: 0 = full quality, 1 = hard shadows, 2 = no IBL,
   *  3 = no shadows. Raised one step at a time when `gl.getError()` reports a
   *  rejected draw (WebKit/ANGLE rejects three's shadow depth textures or
   *  half-float filtering). */
  glStep: number;
  /** False until the probe has seen a clean frame after the last downgrade. */
  glProbed: boolean;
  /** Set by anything that changes the picture; the loop renders and clears it. */
  needsRender: boolean;
  /** Latest selection, so the canvas click handler is not a stale closure. */
  selection: string[];
}

/** Per-mesh shading state: what the built content looks like, and its clay. */
interface ShadingRecord {
  studioMaterial?: THREE.Material | THREE.Material[];
  clayMaterial?: THREE.Material | THREE.Material[];
  /** Terrain only: decoded albedo drape. */
  drape?: THREE.Texture | null;
  /** Terrain only: the mesh carries per-vertex elevation-ramp colours. */
  ramp?: boolean;
  /** Terrain only: the baked hillshade texture (grey multiplier, below). */
  hillshade?: THREE.Texture | null;
}

/** Clamp half-float RGBA data to 8-bit linear values for a safe backdrop. */
function clampHalfFloatToByte(
  data: Uint16Array,
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    for (let channel = 0; channel < 4; channel += 1) {
      const value = THREE.DataUtils.fromHalfFloat(data[index * 4 + channel]);
      const clamped = Math.min(1, Math.max(0, value));
      out[index * 4 + channel] = Math.round(clamped * 255);
    }
  }
  return out;
}

/**
 * The engine's built-in sky as a linear equirect texture.
 *
 * `write_sky_hdr` (studio_render.py) bakes the same gradient into the studio
 * HDR the renderer uses when no project environment is set. Keeping one source
 * of truth in both places is what makes Material's orbit and the Rendered frame
 * share a light environment instead of a studio room vs. a sky.
 */
function studioSkyTexture(width = 128, height = 64): THREE.DataTexture {
  const data = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const v = y / (height - 1);
    // 0 = zenith, 0.5 = horizon, 1 = nadir.
    const t = Math.min(1, Math.max(0, (0.5 - v) / 0.5));
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const sky = SKY_HORIZON[channel] * (1 - t) + SKY_ZENITH[channel] * t;
        const value = v < 0.5 ? sky : SKY_GROUND[channel];
        data[offset + channel] = value * SKY_GAIN;
      }
      data[offset + 3] = 1;
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  // Radiance, not an sRGB image: keep the values the engine would decode.
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.needsUpdate = true;
  return texture;
}

function clayMaterial(side: THREE.Side = THREE.FrontSide): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: CLAY_COLOUR,
    roughness: 0.62,
    metalness: 0.0,
    side,
  });
}

function buildPrimitive(primitive: Primitive, material?: Material): THREE.Object3D {
  let geometry: THREE.BufferGeometry;
  switch (primitive.type) {
    case "cube":
      geometry = new THREE.BoxGeometry(primitive.size, primitive.size, primitive.size);
      break;
    case "sphere":
      geometry = new THREE.SphereGeometry(primitive.radius, 32, 24);
      break;
    case "plane":
      geometry = new THREE.PlaneGeometry(primitive.size, primitive.size);
      break;
  }
  const side = primitive.type === "plane" ? THREE.DoubleSide : THREE.FrontSide;
  const standard = new THREE.MeshStandardMaterial({
    color: material
      ? new THREE.Color(material.baseColor[0], material.baseColor[1], material.baseColor[2])
      : CLAY_COLOUR,
    roughness: material?.roughness ?? 0.62,
    metalness: material?.metallic ?? 0.04,
    side,
  });
  const mesh = new THREE.Mesh(geometry, standard);
  // Both materials live on the mesh: Solid swaps in the clay, Material swaps
  // the authored material back. Neither is rebuilt when the mode changes.
  mesh.userData.owned = true;
  mesh.userData.studioMaterial = standard;
  mesh.userData.clayMaterial = clayMaterial(side);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (primitive.type === "plane") mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

/**
 * Swap a subtree between its authored materials and Solid's clay.
 *
 * Imported glTF scenes share materials with the asset cache, so the original
 * is only referenced (never disposed) and restored verbatim. Terrain has its
 * own drape-aware path ([`applyTerrainShading`]).
 */
function setSolidShading(root: THREE.Object3D, solid: boolean): void {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!(mesh as { isMesh?: boolean }).isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const record = node.userData as ShadingRecord;
    if (!record.studioMaterial || !record.clayMaterial) {
      record.studioMaterial = mesh.material;
      record.clayMaterial = clayMaterial(THREE.DoubleSide);
    }
    mesh.material = solid ? record.clayMaterial : record.studioMaterial;
  });
}

/** Solid is clay; Material shows the albedo drape (or clay without one).
 *
 * The drape is drawn opaque. It used to inherit the overlay's strength as
 * material opacity, which made the terrain 82% opaque and blended it with the
 * dark backdrop — the preview then read darker than the same frame from the
 * engine, which composites the overlay onto a lit material instead. Neither the
 * sun nor the exposure may change with a slider, so the preview keeps the
 * surface opaque; the still honours the opacity in its overlay pass.
 */
function applyTerrainShading(mesh: THREE.Mesh, solid: boolean, hillshade: boolean): void {
  const material = mesh.material as THREE.MeshStandardMaterial;
  const { drape = null, ramp = false, hillshade: hillshadeMap = null } = mesh.userData as ShadingRecord;
  // Material shows the cartography: a draped raster when one is attached,
  // otherwise the elevation ramp when a colormap is chosen. Solid is clay.
  // Hillshade rides on top unless a raster drape is already there.
  const draped = !solid && drape !== null;
  const ramped = !solid && !draped && ramp;
  const map = draped ? drape : hillshade ? hillshadeMap : null;
  const colour = draped || ramped ? 0xffffff : CLAY_COLOUR;
  // Playback re-applies shading on every snapshot; only touch the material
  // when the mode (or the drape/ramp) actually changed, so `needsUpdate` does
  // not rebuild the program every frame.
  let changed = false;
  if (material.map !== map) {
    material.map = map;
    changed = true;
  }
  if (material.vertexColors !== ramped) {
    material.vertexColors = ramped;
    changed = true;
  }
  if (material.color.getHex() !== colour) {
    material.color.setHex(colour);
    changed = true;
  }
  if (material.transparent) {
    material.transparent = false;
    changed = true;
  }
  if (material.opacity !== 1) {
    material.opacity = 1;
    changed = true;
  }
  if (changed) material.needsUpdate = true;
}

/** Re-apply the mode's shading to one object's built content. */
function applyShading(
  runtime: Runtime,
  object: SceneObject,
  mode: ViewportMode,
  hillshade: boolean,
): void {
  const group = runtime.objects.get(object.id);
  if (!group) return;
  const solid = mode === "solid";
  if (object.kind.type === "terrain") {
    for (const child of group.children) {
      if (child instanceof THREE.Mesh) applyTerrainShading(child, solid, hillshade);
    }
    return;
  }
  if (object.kind.type === "mesh") setSolidShading(group, solid);
}

/**
 * Environment lighting and backdrop for the current mode.
 *
 * Solid stays neutral: no IBL and no HDRI sky (the clay view is for shape).
 * Material uses the project HDRI — including as the background, like Blender's
 * Material preview — or the built-in studio sky. The project's IBL intensity
 * scales the light; the fallback is additionally dimmed so it reads as fill.
 */
function recompileMaterials(runtime: Runtime): void {
  for (const group of runtime.objects.values()) {
    group.traverse((node) => {
      const mesh = node as THREE.Mesh;
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) material.forEach((entry) => (entry.needsUpdate = true));
      else if (material) material.needsUpdate = true;
    });
  }
}

/**
 * The GPU rejected a draw: three's shadow depth texture with a comparison
 * sampler, or half-float filtering (WebKit/ANGLE). Degrade one step at a time
 * and say so in the viewport instead of spamming INVALID_OPERATION forever.
 * Returns the note to show.
 */
function applyGpuStep(runtime: Runtime): string {
  runtime.glStep += 1;
  const step = runtime.glStep;
  if (step >= 1) {
    runtime.renderer.shadowMap.type = THREE.BasicShadowMap;
    recompileMaterials(runtime);
  }
  if (step >= 2) {
    // Keep the 8-bit backdrop; drop only the IBL, which needs the half-float
    // PMREM output the platform cannot sample. Every mode now needs the same
    // fallback fill the engine's environment would have provided.
    runtime.environment = null;
    runtime.scene.environment = null;
    runtime.hemi.intensity = 0.25;
  }
  if (step >= 3) {
    runtime.sun.castShadow = false;
    for (const group of runtime.objects.values()) {
      group.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (mesh.isMesh) mesh.castShadow = false;
      });
    }
  }
  return step === 1
    ? "GPU limit: hard shadows"
    : step === 2
      ? "GPU limit: no image-based lighting"
      : "GPU limit: shadows off";
}

/**
 * Environment lighting and backdrop for the current mode.
 *
 * Lighting is identical in every mode: one sun, one environment. Solid and
 * Material differ only in the surface they show (clay vs. albedo) and in the
 * backdrop (neutral vs. the project HDRI). The project's IBL intensity scales
 * the environment exactly as it scales the engine's.
 */
function applyEnvironment(
  runtime: Runtime,
  mode: "solid" | "material",
  iblIntensity: number,
  backgroundIntensity: number,
): void {
  const solid = mode === "solid";
  const noIbl = runtime.glStep >= 2;
  runtime.scene.environment = noIbl ? null : runtime.environment;
  runtime.scene.environmentIntensity = noIbl ? 0 : iblIntensity;
  // The engine applies no ambient beyond the environment image, so neither does
  // the preview. Only a device that cannot sample the environment (glStep 2)
  // gets the fallback fill, and then every mode gets it.
  runtime.hemi.intensity = noIbl ? 0.25 : 0;
  const backdrop = runtime.environmentBackground;
  if (solid || !backdrop || backgroundIntensity <= 0.001) {
    // Solid stays neutral; 0 on the slider hides the HDRI and leaves the
    // neutral studio backdrop rather than fading to black.
    runtime.scene.background = solid ? SOLID_BG : MATERIAL_BG;
    runtime.scene.backgroundIntensity = 1;
  } else {
    runtime.scene.background = backdrop;
    runtime.scene.backgroundIntensity = backgroundIntensity;
  }
}

/**
 * Apply the project's shadow settings to the directional light.
 *
 * The still renders with the project's shadow filter and map size
 * (`renderer_config.shadows`: `shadowTechnique`, `shadowResolution`,
 * `slope_scale_bias` 1e-04, `normal_bias` 1e-05). The preview honours the
 * filter and the map size: `pcf` keeps three's PCF, `pcss` maps to
 * PCFSoftShadowMap (three has no blocker search — this is the softer PCF
 * variant, not the engine's PCSS). The slope bias maps to a depth bias of
 * the same magnitude, with a world-space normal bias so the 12-unit viewer
 * does not acne at grazing light.
 *
 * The engine's ray-marched sun visibility has no equivalent in three.js:
 * a ridge cannot shadow the ground behind it here. That gap is documented,
 * not worked around.
 */
function applyShadowRig(runtime: Runtime, rig: LightingRig): void {
  const enabled = rig.shadows.enabled && runtime.glStep < 3;
  const key = `${enabled ? 1 : 0}|${rig.shadows.technique}|${rig.shadows.resolution}`;
  runtime.sun.castShadow = enabled;
  if (runtime.shadowKey === key) return;
  runtime.shadowKey = key;
  const size = Math.max(256, Math.min(8192, Math.round(rig.shadows.resolution)));
  if (runtime.sun.shadow.mapSize.x !== size) {
    runtime.sun.shadow.mapSize.set(size, size);
    // A map of a different size never reallocates on its own: free it and
    // the next frame builds one at the new size.
    if (runtime.sun.shadow.map) {
      runtime.sun.shadow.map.dispose();
      runtime.sun.shadow.map = null;
    }
  }
  // The still's slope_scale_bias is a slope-proportional depth offset; at
  // the viewer's scale the same magnitude is applied uniformly.
  runtime.sun.shadow.bias = -1.0e-4;
  runtime.sun.shadow.normalBias = 0.05;
  const type =
    rig.shadows.technique === "pcss" ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
  if (runtime.renderer.shadowMap.type !== type) {
    runtime.renderer.shadowMap.type = type;
    // three bakes the PCF strides into the material programs; a type change
    // without a recompile keeps stale samplers.
    recompileMaterials(runtime);
  }
}

/** A small frustum, drawn where the project's active camera sits. */
function buildCameraGizmo(camera: CameraObject): THREE.LineSegments {
  const forward = 1.15;
  const halfHeight = Math.tan((camera.fovDeg * Math.PI) / 360) * forward;
  const halfWidth = halfHeight * 1.45;
  const apex: [number, number, number] = [0, 0, 0];
  const corners: Array<[number, number, number]> = [
    [-halfWidth, -halfHeight, -forward],
    [halfWidth, -halfHeight, -forward],
    [halfWidth, halfHeight, -forward],
    [-halfWidth, halfHeight, -forward],
  ];

  const points: number[] = [];
  corners.forEach((corner, index) => {
    points.push(...apex, ...corner);
    points.push(...corner, ...corners[(index + 1) % corners.length]);
  });
  points.push(0, 0, 0, 0, 0.24, 0, 0, 0, 0, 0.24, 0, 0);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: CAMERA_COLOUR }),
  );
  lines.userData.owned = true;
  return lines;
}

function buildLightGizmo(light: LightObject): THREE.Group {
  const group = new THREE.Group();
  const colour = new THREE.Color(light.color[0], light.color[1], light.color[2]);

  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 14, 10),
    new THREE.MeshBasicMaterial({ color: colour }),
  );
  bulb.userData.owned = true;
  group.add(bulb);

  if (light.kind === "directional") {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, -1.6], 3),
    );
    const ray = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.75 }),
    );
    ray.userData.owned = true;
    group.add(ray);
  } else {
    for (let index = 0; index < 8; index += 1) {
      const angle = (index / 8) * Math.PI * 2;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(
          [0, 0, 0, Math.cos(angle) * 0.42, Math.sin(angle) * 0.42, 0],
          3,
        ),
      );
      const spike = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.6 }),
      );
      spike.userData.owned = true;
      group.add(spike);
    }
  }
  group.userData.owned = true;
  return group;
}

/**
 * A displaced grid from a sampled DEM.
 *
 * The Studio viewport is a preview: this is real elevation data at reduced
 * resolution, scaled so the scene fits the viewport. Forge3D still receives the
 * original GeoTIFF and renders the full-resolution surface.
 *
 * The surface is neutral clay here. Elevation shading used to double as a fake
 * albedo ramp; Solid must show no colour at all, and Material supplies the real
 * drape ([`applyTerrainShading`]).
 */
/**
 * Hillshade: a greyscale relief map baked from the DEM's slope/aspect against
 * a fixed sun azimuth/altitude (matches QGIS: 315° azimuth, 45° altitude, ½ past). Keyed to
 * the grid's metric resolution, not the preview's world scale. Tracks the
 * terrain data, not the camera, so it composes with any view direction.
 */
function computeHillshadeTexel(
  dzDx: number,
  dzDy: number,
  azimuth: number,
  zenith: number,
): number {
  const slope = Math.atan(Math.sqrt(dzDx * dzDx + dzDy * dzDy));
  // ArcGIS aspect convention: 0 = north, clockwise.
  const aspect = Math.atan2(dzDy, -dzDx);
  const value = Math.cos(zenith) * Math.cos(slope) +
                Math.sin(zenith) * Math.sin(slope) * Math.cos(azimuth - aspect);
  return Math.max(0, Math.min(255, Math.floor(value * 255)));
}

/** Bake a hillshade texture from the heights: central differences, metric
 *  cell size from the footprint, 315° az / 45° alt, no exaggeration. Linear filter. */
function bakeHillshade(preview: TerrainPreview): THREE.DataTexture {
  const resolution = preview.resolution;
  const heights = preview.heights;
  const [spanX, spanZ] = preview.spanXz;
  const cellX = spanX / Math.max(1, resolution - 1);
  const cellY = spanZ / Math.max(1, resolution - 1);
  const azimuth = (315 * Math.PI) / 180;
  // Zenith angle from the vertical, the ESRI formula: matches the standard
  // "QGIS / ArcGIS" relief shades (SE-lit slopes face dark).
  const zenith = (45 * Math.PI) / 180;
  const data = new Uint8Array(resolution * resolution * 4);
  for (let row = 0; row < resolution; row += 1) {
    for (let column = 0; column < resolution; column += 1) {
      const left = column === 0 ? heights[row * resolution + column]
                                : heights[row * resolution + column - 1];
      const right = column === resolution - 1 ? heights[row * resolution + column]
                                : heights[row * resolution + column + 1];
      const up = row === 0 ? heights[row * resolution + column]
                          : heights[(row - 1) * resolution + column];
      const down = row === resolution - 1 ? heights[row * resolution + column]
                          : heights[(row + 1) * resolution + column];
      const dzDx = (right - left) / (2 * cellX);
      const dzDy = (down - up) / (2 * cellY);
      const value = computeHillshadeTexel(dzDx, dzDy, azimuth, zenith);
      // PlaneGeometry UVs follow row order; match the mesh's own indexing.
      // RGBA: three.js .map sampling expects channels the material can read;
      // replicate the grey into R/G/B so it multiplies cleanly in the shader.
      const base = (row * resolution + column) * 4;
      data[base] = value;
      data[base + 1] = value;
      data[base + 2] = value;
      data[base + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(
    data,
    resolution,
    resolution,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  texture.flipY = true;
  texture.colorSpace = THREE.NoColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

function buildTerrainMesh(
  preview: TerrainPreview,
  zScale: number,
  ramp: { stops: [number, number, number][]; strength: number } | null,
  hillshade: boolean,
): THREE.Mesh {
  const resolution = preview.resolution;
  const geometry = new THREE.PlaneGeometry(1, 1, resolution - 1, resolution - 1);
  geometry.rotateX(-Math.PI / 2);

  const [spanX, spanZ] = preview.spanXz;
  const horizontal = TERRAIN_VIEW_SIZE / Math.max(spanX, spanZ);
  const [minElevation] = [preview.minElevation];

  const position = geometry.attributes.position as THREE.BufferAttribute;
  const span = Math.max(1e-6, preview.maxElevation - preview.minElevation);
  const colours = ramp ? new Float32Array(position.count * 3) : null;
  for (let index = 0; index < position.count; index += 1) {
    const row = Math.floor(index / resolution);
    const column = index % resolution;
    const elevation = preview.heights[row * resolution + column] ?? minElevation;
    position.setY(index, (elevation - minElevation) * zScale * horizontal);
    if (colours && ramp) {
      // The same elevation domain the renderer maps through the LUT, and the
      // engine's linear stops — which is the space vertex colours live in.
      const [r, g, b] = rampColor(
        ramp.stops,
        (elevation - minElevation) / span,
        ramp.strength,
      );
      colours[index * 3] = r;
      colours[index * 3 + 1] = g;
      colours[index * 3 + 2] = b;
    }
  }
  if (colours) {
    geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  }

  geometry.computeVertexNormals();
  // The still lights terrain with lambert_physical: a Lambertian BRDF with
  // direct-sun visibility — no specular lobe, not even a dielectric one.
  // MeshPhysicalMaterial with specularIntensity 0 is exactly that, so the
  // Material preview's slopes glint no more than the clay would.
  const material = new THREE.MeshPhysicalMaterial({
    color: CLAY_COLOUR,
    roughness: 0.95,
    metalness: 0.0,
    specularIntensity: 0.0,
    side: THREE.DoubleSide,
  });
  // Hillshade stays in the mesh so the workspace can flip it live; its
  // texture is baked once against the DEM.
  const hillshadeTexel = bakeHillshade(preview);
  if (hillshade) {
    material.map = hillshadeTexel;
    material.vertexColors = colours !== null;
  }
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  // The grid is built in unit space; the footprint is what gives it metres.
  mesh.scale.set(spanX * horizontal, 1, spanZ * horizontal);
  mesh.userData.owned = true;
  mesh.userData.studioMaterial = mesh.material;
  mesh.userData.drape = null;
  mesh.userData.hillshade = hillshadeTexel;
  // Whether this mesh carries an elevation ramp at all (the attribute is fixed
  // once built; the mode decides if it is shown).
  mesh.userData.ramp = colours !== null;
  return mesh;
}

/** One label anchor: where its div goes, and what it says. */
interface CartographyAnchor {
  id: string;
  text: string;
  className: string;
  /** Layer colour, pre-composed to CSS. */
  colourCss: string;
  /** Text-shadow halo: readable on any ground. */
  haloCss: string;
  fontSizePx: number;
  /** World position on the terrain (the div rides the projection). */
  world: THREE.Vector3;
  /** Hidden this frame because the ground stands in front of the anchor. */
  occluded: boolean;
}

/** Compose a layer RGBA into a readable label colour and its halo. */
function labelInk(rgba: [number, number, number, number] | undefined): {
  colourCss: string;
  haloCss: string;
} {
  const colour = rgba ?? [1, 1, 1, 0.95];
  const [r, g, b] = colour;
  const luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
  return {
    colourCss: `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`,
    haloCss:
      luminance < 0.5
        ? "0 0 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(255, 255, 255, 0.9)"
        : "0 0 2px rgba(0, 0, 0, 0.85), 0 0 1px rgba(0, 0, 0, 0.9)",
  };
}

/** Remove every overlay child (draped lines and filled polygon meshes) and its GPU buffers. */
function disposeOverlayObjects(group: THREE.Group): void {
  for (const child of [...group.children]) {
    group.remove(child);
    const line = child as LineSegments2;
    line.geometry?.dispose();
    (line.material as LineMaterial | undefined)?.dispose();
  }
}

/** Rebuild the label divs to match the current anchors. */
function rebuildCartographyLabels(
  layer: HTMLDivElement,
  anchors: CartographyAnchor[],
): Map<string, HTMLDivElement> {
  layer.replaceChildren();
  const divs = new Map<string, HTMLDivElement>();
  for (const anchor of anchors) {
    const div = document.createElement("div");
    div.className = `cartography-label ${anchor.className}`;
    div.textContent = anchor.text;
    div.style.color = anchor.colourCss;
    div.style.textShadow = anchor.haloCss;
    div.style.fontSize = `${anchor.fontSizePx}px`;
    div.style.display = "none";
    layer.appendChild(div);
    divs.set(anchor.id, div);
  }
  return divs;
}

function contentSignature(
  object: SceneObject,
  terrainResolution: number,
  ramp: string,
  imagery: { asset: string; opacity: number } | null,
): string {
  switch (object.kind.type) {
    case "mesh":
      return `mesh:${JSON.stringify(object.kind.source)}:${JSON.stringify(object.kind.material)}`;
    case "terrain":
      // Preview resolution AND the draped albedo are part of the identity:
      // changing either must rebuild (and re-drape) the mesh, which is exactly
      // what used to be ignored.
      // The elevation ramp is baked into the mesh's vertex colours, so a
      // colormap change is a rebuild — not just a shading swap.
      return `terrain:${object.kind.asset}:${object.kind.zScale}:${terrainResolution}:${imagery?.asset ?? "none"}:${imagery?.opacity ?? ""}:${ramp}`;
    case "camera":
      return `camera:${object.kind.fovDeg}`;
    case "light":
      return `light:${object.kind.kind}:${object.kind.color.join(",")}`;
  }
}

function applyTransform(group: THREE.Object3D, transform: Transform): void {
  group.position.set(
    transform.translation.x,
    transform.translation.y,
    transform.translation.z,
  );
  // Studio's model applies X, then Y, then Z (`Rz * Ry * Rx`), which is three's
  // "ZYX" order — see crates/studio-core/src/math.rs.
  group.rotation.set(
    transform.rotationDeg.x * DEG,
    transform.rotationDeg.y * DEG,
    transform.rotationDeg.z * DEG,
    "ZYX",
  );
  group.scale.set(transform.scale.x, transform.scale.y, transform.scale.z);
}

function readTransform(object: THREE.Object3D): Transform {
  const euler = new THREE.Euler().setFromQuaternion(object.quaternion, "ZYX");
  return {
    translation: { x: object.position.x, y: object.position.y, z: object.position.z },
    rotationDeg: {
      x: euler.x / DEG,
      y: euler.y / DEG,
      z: euler.z / DEG,
    },
    scale: { x: object.scale.x, y: object.scale.y, z: object.scale.z },
  };
}

/**
 * A camera as the scene transform the renderer expects.
 *
 * The camera's own orientation: the studio transform's forward is -Z, the same
 * axis a three.js camera looks down, and `applyTransform` writes the Euler back
 * in the same ZYX order — so the round-trip is exact. Aiming at a fixed point
 * instead would silently pitch the captured camera towards it.
 */
function cameraTransform(camera: THREE.PerspectiveCamera): Transform {
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "ZYX");
  return {
    translation: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
    rotationDeg: { x: euler.x / DEG, y: euler.y / DEG, z: euler.z / DEG },
    scale: { x: 1, y: 1, z: 1 },
  };
}

/**
 * Remove a group's contents. Only geometry this crate created is disposed:
 * imported glTF scenes share geometry and materials with the asset cache, and
 * disposing them would break every other object using the same file. The one
 * material we always own is the Solid clay we put on that clone.
 */
function disposeChildren(group: THREE.Group): void {
  for (const child of [...group.children]) {
    group.remove(child);
    const owned = Boolean(child.userData.owned);
    const seen = new Set<THREE.Material>();
    const disposeMaterial = (material: unknown) => {
      if (!material) return;
      const entries = Array.isArray(material) ? material : [material];
      for (const entry of entries as THREE.Material[]) {
        if (seen.has(entry)) continue;
        seen.add(entry);
        entry.dispose();
      }
    };
    child.traverse((node) => {
      const mesh = node as {
        geometry?: { dispose(): void };
        material?: THREE.Material | THREE.Material[];
        userData?: ShadingRecord;
      };
      if (owned) {
        mesh.geometry?.dispose();
        disposeMaterial(mesh.material);
        disposeMaterial(mesh.userData?.studioMaterial);
      }
      disposeMaterial(mesh.userData?.clayMaterial);
    });
  }
}

export function Viewport3D() {
  const {
    state,
    actions,
    tool,
    space,
    cameraView,
    render,
    terrainResolution,
    flyMode,
    setCameraView,
  } = useSession();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const actionsRef = useRef(actions);
  actionsRef.current = actions;

  /** Current viewport mode: Solid is clay, Material is the lit preview, and
   *  only Rendered composes an engine frame. */
  const mode = state?.project.viewport.mode ?? "solid";
  const viewportModeRef = useRef<ViewportMode>(mode);
  viewportModeRef.current = mode;

  const [quaternion, setQuaternion] = useState<Quaternion>([0, 0, 0, 1]);
  const [size, setSize] = useState({ width: 960, height: 600 });
  const [problem, setProblem] = useState<string | null>(null);
  /** Shown when the GPU probe had to downgrade a feature (see Runtime.glStep). */
  const [gpuNote, setGpuNote] = useState<string | null>(null);
  const savedView = useRef<{
    position: THREE.Vector3;
    quaternion: THREE.Quaternion;
    target: THREE.Vector3;
    fov: number;
  } | null>(null);
  const inCameraView = useRef(false);
  /** Fly-camera state: held keys, an in-progress look drag, and the current
   *  yaw/pitch (OrbitControls owns the orientation until fly mode starts). */
  const flyKeys = useRef(new Set<string>());
  const flyLook = useRef<{ x: number; y: number } | null>(null);
  const flyAngles = useRef({ yaw: 0, pitch: 0 });
  const flyModeRef = useRef(flyMode);
  flyModeRef.current = flyMode;

  /** Draped cartography (vector lines, graticule): geometry loaded per stack
   *  change; the group is rebuilt, cameras never refetch. */
  const anchorsRef = useRef<CartographyAnchor[]>([]);
  const labelDivsRef = useRef(new Map<string, HTMLDivElement>());
  /** The camera transform the label occlusion pass last saw. */
  const occlusionCameraRef = useRef(new THREE.Vector3(Infinity, 0, 0));
  const occlusionQuatRef = useRef(new THREE.Quaternion(Infinity, 0, 0));
  const occlusionVersionRef = useRef(-1);
  const anchorsVersionRef = useRef(0);

  const textureCache = useRef(new Map<string, Promise<THREE.Texture>>());
  const loadTexture = useCallback((path: string): Promise<THREE.Texture> => {
    const cache = textureCache.current;
    let pending = cache.get(path);
    if (!pending) {
      pending = bridge
        .readImageAsset(path)
        .then(
          (payload) =>
            new Promise<THREE.Texture>((resolve, reject) => {
              const loader = new THREE.TextureLoader();
              loader.load(
                `data:image/${payload.kind === "orthophoto" && path.toLowerCase().endsWith(".png") ? "png" : "jpeg"};base64,${payload.base64}`,
                (texture) => {
                  texture.colorSpace = THREE.SRGBColorSpace;
                  texture.wrapS = THREE.RepeatWrapping;
                  texture.wrapT = THREE.RepeatWrapping;
                  resolve(texture);
                },
                undefined,
                reject,
              );
            }),
        );
      cache.set(path, pending);
      pending.catch(() => cache.delete(path));
    }
    return pending;
  }, []);

  /** The engine's colormap stops per name (fetched once; the shell has no
   *  palette of its own). */
  const colormapStopsCache = useRef(new Map<string, Promise<[number, number, number][]>>());
  const loadColormapStops = useCallback((name: string) => {
    const cache = colormapStopsCache.current;
    let pending = cache.get(name);
    if (!pending) {
      pending = bridge.colormapStops(name).then((stops) => {
        if (!Array.isArray(stops) || stops.length < 2) {
          throw new Error(`the engine returned no stops for ${name}`);
        }
        return stops.map(
          (stop) => [stop[0], stop[1], stop[2]] as [number, number, number],
        );
      });
      cache.set(name, pending);
      pending.catch(() => cache.delete(name));
    }
    return pending;
  }, [bridge]);

  /** Decoded drape texture for a terrain overlay (shell-side GeoTIFF decode;
   *  cached by path — opacity is applied on the material, not the texture). */
  const overlayTextureCache = useRef(new Map<string, Promise<THREE.Texture>>());
  const loadOverlayTexture = useCallback((path: string): Promise<THREE.Texture> => {
    const cache = overlayTextureCache.current;
    let pending = cache.get(path);
    if (!pending) {
      pending = bridge
        .rasterPreview(path, 2048)
        .then(
          (payload) =>
            new Promise<THREE.Texture>((resolve, reject) => {
              const loader = new THREE.TextureLoader();
              loader.load(
                `data:image/png;base64,${payload.base64}`,
                (texture) => {
                  texture.colorSpace = THREE.SRGBColorSpace;
                  texture.wrapS = THREE.ClampToEdgeWrapping;
                  texture.wrapT = THREE.ClampToEdgeWrapping;
                  resolve(texture);
                },
                undefined,
                reject,
              );
            }),
        );
      cache.set(path, pending);
      pending.catch(() => cache.delete(path));
    }
    return pending;
  }, []);

  const terrainCache = useRef(new Map<string, Promise<TerrainPreview>>());
  const readTerrain = useCallback(
    (path: string): Promise<TerrainPreview> => {
      const cache = terrainCache.current;
      const key = `${path}:${terrainResolution}`;
      let pending = cache.get(key);
      if (!pending) {
        pending = bridge.terrainPreview(path, terrainResolution);
        cache.set(key, pending);
        pending.catch(() => cache.delete(key));
      }
      return pending;
    },
    [terrainResolution],
  );

  const readMesh = useCallback(
    async (path: string): Promise<ArrayBuffer> => {
      const payload = await bridge.readMeshFile(path);
      return base64ToArrayBuffer(payload.base64);
    },
    [],
  );

  /** The project HDRI currently decoded for this viewport (one at a time). */
  const environmentRef = useRef<LoadedEnvironment | null>(null);
  const loadEnvironment = useCallback(
    async (path: string): Promise<LoadedEnvironment> => {
      const runtime = runtimeRef.current;
      if (!runtime?.pmrem) throw new Error("the viewport is not ready");
      const payload = await bridge.readHdr(path);
      const texData = new HDRLoader().parse(base64ToArrayBuffer(payload.base64));
      if (!texData.data || !texData.width || !texData.height) {
        throw new Error("the HDRI has no image data");
      }
      const source = new THREE.DataTexture(
        texData.data as unknown as Uint16Array,
        texData.width,
        texData.height,
        THREE.RGBAFormat,
        texData.type ?? THREE.HalfFloatType,
      );
      source.mapping = THREE.EquirectangularReflectionMapping;
      source.colorSpace = THREE.LinearSRGBColorSpace;
      source.minFilter = THREE.LinearFilter;
      source.magFilter = THREE.LinearFilter;
      source.generateMipmaps = false;
      source.flipY = texData.flipY ?? true;
      source.needsUpdate = true;
      // PMREM is what `scene.environment` expects; the source doubles as the
      // Material background (PMREM output cannot be drawn as a skybox).
      // A display-referred copy for the backdrop: some webviews reject a
      // half-float texture when it is filtered, and the background is a
      // fallback path anyway (values above 1 clip to white).
      const background = new THREE.DataTexture(
        // HDRLoader defaults to HalfFloatType; that is what the clamp reads.
        clampHalfFloatToByte(texData.data as Uint16Array, texData.width, texData.height),
        texData.width,
        texData.height,
        THREE.RGBAFormat,
        THREE.UnsignedByteType,
      );
      background.mapping = THREE.EquirectangularReflectionMapping;
      background.colorSpace = THREE.LinearSRGBColorSpace;
      background.minFilter = THREE.LinearFilter;
      background.magFilter = THREE.LinearFilter;
      background.generateMipmaps = false;
      background.flipY = source.flipY;
      background.needsUpdate = true;
      const target = runtime.pmrem.fromEquirectangular(source);
      return { path, source, background, environment: target.texture, target };
    },
    [],
  );

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        // Ask for the discrete adapter on hybrid hosts instead of letting the
        // browser pick the power-saving one.
        powerPreference: "high-performance",
      });
    } catch (cause) {
      setProblem(
        `WebGL is unavailable in this webview: ${cause instanceof Error ? cause.message : cause}`,
      );
      return;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    // The renderer tone-maps (ACES, exposure 1.35); without this the viewport
    // reads much brighter than the Rendered frame.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    // Material preview only, but the map must exist before any light casts.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SOLID_BACKGROUND);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
    camera.position.set(9.5, 7, 11);

    const orbit = new OrbitControls(camera, canvas);
    orbit.target.set(0, 0.8, 0);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.09;
    orbit.screenSpacePanning = false;
    orbit.minDistance = 0.8;
    orbit.maxDistance = 150;
    // Free orbit, Blender-style: the old 89.5° cap made it impossible to look
    // at the scene from below the ground plane.
    orbit.maxPolarAngle = Math.PI;
    orbit.update();

    const gizmo = new TransformControls(camera, canvas);
    gizmo.setSize(0.85);
    scene.add(gizmo.getHelper());

    const hemi = new THREE.HemisphereLight(0xc8d4e8, 0x1b1f26, 0);
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.set(6, 9, 4);
    // One shadow rig for every mode: Solid and Material differ by the surface
    // they show, not by how they are lit.
    sun.castShadow = false;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.bias = -0.0008;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 80;
    const shadowExtent = TERRAIN_VIEW_SIZE;
    sun.shadow.camera.left = -shadowExtent;
    sun.shadow.camera.right = shadowExtent;
    sun.shadow.camera.top = shadowExtent;
    sun.shadow.camera.bottom = -shadowExtent;
    sun.shadow.camera.updateProjectionMatrix();
    // Sun + environment only — the engine's terrain shader has no fill light,
    // so neither does the preview. `hemi` is the fallback for a device that
    // cannot sample the environment (see `applyGpuStep`).
    scene.add(hemi, sun);

    // The built-in sky is the *same* gradient the render service writes into its
    // studio HDR (`write_sky_hdr` in studio_render.py), prefiltered once at
    // mount and swapped for the project's HDRI when one is set. A device that
    // cannot filter the float target keeps the fallback hemisphere light.
    const pmrem = new THREE.PMREMGenerator(renderer);
    let studioEnvironment: THREE.Texture | null = null;
    try {
      const sky = studioSkyTexture();
      studioEnvironment = pmrem.fromEquirectangular(sky).texture;
      sky.dispose();
    } catch {
      studioEnvironment = null;
    }

    const grid = new THREE.GridHelper(20, 40, 0x2f3744, 0x1d222a);
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.85;
    scene.add(grid);

    const axes = new THREE.AxesHelper(1.1);
    scene.add(axes);

    const overlay = new THREE.Group();
    overlay.name = "cartography-overlay";
    overlay.userData.owned = true;
    scene.add(overlay);
    const labelLayer = document.createElement("div");
    labelLayer.className = "viewport-cartography";
    labelLayer.setAttribute("aria-hidden", "true");
    container.appendChild(labelLayer);

    const runtime: Runtime = {
      renderer,
      scene,
      camera,
      orbit,
      gizmo,
      sun,
      hemi,
      grid,
      objects: new Map(),
      helpers: [],
      helperKey: "",
      studioEnvironment,
      environment: studioEnvironment,
      environmentBackground: null,
      pmrem,
      overlay,
      labelLayer,
      lineMaterials: [],
      shadowKey: "",
      glStep: 0,
      glProbed: false,
      needsRender: true,
      selection: [],
    };
    runtimeRef.current = runtime;

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      runtime.overlay.visible = viewportModeRef.current !== "rendered";
      if (runtime.labelLayer) runtime.labelLayer.style.display = runtime.overlay.visible ? "" : "none";
      for (const line of runtime.lineMaterials) {
        line.resolution.set(width, height);
      }
      runtime.needsRender = true;
      setSize((current) =>
        current.width === width && current.height === height
          ? current
          : { width, height },
      );
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    gizmo.addEventListener("dragging-changed", (event) => {
      // Fly mode owns the camera: re-enabling OrbitControls here would let its
      // own pointer handlers rotate the camera behind the fly-look handler.
      orbit.enabled =
        !(event as unknown as { value: boolean }).value && !flyModeRef.current;
    });
    gizmo.addEventListener("objectChange", () => {
      for (const helper of runtime.helpers) helper.update();
      // The gizmo moves the object directly; the command that records it is
      // applied later, so this frame has to be drawn on its own.
      runtime.needsRender = true;
    });
    // Rendered mode follows the camera. A viewport frame is throttled to one
    // per 120 ms (the SSAO pass, which alone costs ~4.8 s, is reserved for
    // final stills), and the session keeps only the newest pending request, so
    // a drag does not queue frames it cannot finish.
    let lastRenderRequest = 0;
    orbit.addEventListener("change", () => {
      if (!modeRef.current) return;
      const now = performance.now();
      if (now - lastRenderRequest < 120) return;
      lastRenderRequest = now;
      const target = requestSizeRef.current();
      void actionsRef.current.renderViewport(target.width, target.height, "preview");
    });

    orbit.addEventListener("end", () => {
      if (!modeRef.current) return;
      const target = requestSizeRef.current();
      void actionsRef.current.renderViewport(target.width, target.height, "preview");
    });

    gizmo.addEventListener("mouseUp", () => {
      const target = gizmo.object;
      const id = target?.userData.studioId as string | undefined;
      if (!target || !id) return;
      void actionsRef.current.apply({
        type: "setTransform",
        id,
        transform: readTransform(target),
      });
    });

    // Click (not drag) selects; dragging orbits the viewport, or looks around
    // in fly mode.
    let pressed: { x: number; y: number } | null = null;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const onPointerDown = (event: PointerEvent) => {
      pressed = { x: event.clientX, y: event.clientY };
      if (flyModeRef.current && event.button === 0) {
        flyLook.current = { x: event.clientX, y: event.clientY };
      }
    };

    const onPointerMove = (event: PointerEvent) => {
      const look = flyLook.current;
      if (!look) return;
      flyAngles.current.yaw -= (event.clientX - look.x) * 0.0032;
      flyAngles.current.pitch = Math.max(
        -1.55,
        Math.min(1.55, flyAngles.current.pitch - (event.clientY - look.y) * 0.0032),
      );
      flyLook.current = { x: event.clientX, y: event.clientY };
      camera.quaternion.setFromEuler(
        new THREE.Euler(flyAngles.current.pitch, flyAngles.current.yaw, 0, "YXZ"),
      );
      runtime.needsRender = true;
    };

    const onPointerUp = (event: PointerEvent) => {
      flyLook.current = null;
      if (!pressed) return;
      const travelled = Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y);
      pressed = null;
      if (travelled > 4 || gizmo.dragging) return;
      // In Rendered the engine image covers this canvas; a click would select
      // whatever happens to sit under the pointer in the hidden scene. Orbit
      // and zoom still work because the overlay no longer eats them.
      if (modeRef.current) return;

      const rect = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const hits = raycaster.intersectObjects([...runtime.objects.values()], true);
      let node: THREE.Object3D | null = hits[0]?.object ?? null;
      while (node && !node.userData.studioId) node = node.parent;
      const id = node?.userData.studioId as string | undefined;

      const additive = event.shiftKey || event.ctrlKey || event.metaKey;
      if (!id) {
        if (!additive) void actionsRef.current.select([]);
        return;
      }
      if (!additive) {
        void actionsRef.current.select([id]);
        return;
      }
      const next = new Set(runtime.selection);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      void actionsRef.current.select([...next]);
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);

    // Fly camera: WASD in the view plane, Q/E down and up, Shift to go faster.
    const onFlyKeyDown = (event: KeyboardEvent) => {
      if (!flyModeRef.current) return;
      const key = event.key.toLowerCase();
      if (!FLY_KEYS.has(key)) return;
      flyKeys.current.add(key);
      runtime.needsRender = true;
      event.preventDefault();
    };
    const onFlyKeyUp = (event: KeyboardEvent) => {
      flyKeys.current.delete(event.key.toLowerCase());
    };
    const onWindowBlur = () => {
      flyKeys.current.clear();
      flyLook.current = null;
    };
    window.addEventListener("keydown", onFlyKeyDown);
    window.addEventListener("keyup", onFlyKeyUp);
    window.addEventListener("blur", onWindowBlur);

    /**
     * Publish the Solid camera in the renderer's frame every frame, so a
     * render reproduces the view on screen instead of a canned framing.
     * The axis/scale/datum math lives in `viewMath` (unit-tested);
     * camera view aims 10 viewport units ahead, free view aims at the
     * orbit target — any point on the view axis frames the same picture,
     * since the mapping is a similarity transform.
     */
    const updateViewStore = () => {
      const project = stateRef.current?.project;
      const object = project?.scene.objects.find(
        (candidate) => candidate.kind.type === "terrain",
      );
      if (!object || object.kind.type !== "terrain") {
        viewStore.current = null;
        return;
      }
      const aim =
        cameraViewRef.current || flyModeRef.current
          ? camera.position
              .clone()
              .add(
                new THREE.Vector3(0, 0, -1)
                  .applyQuaternion(camera.quaternion)
                  .multiplyScalar(10),
              )
          : orbit.target;
      viewStore.current = computeRenderView({
        camera: [camera.position.x, camera.position.y, camera.position.z],
        aim: [aim.x, aim.y, aim.z],
        fovDeg: camera.fov,
        spanXz: object.kind.worldSpanXz,
        dimensions: object.kind.dimensions,
        elevationRange: object.kind.elevationRange,
        zScale: object.kind.zScale,
      });
    };

    /**
     * Publish the viewport camera as a scene transform. The timeline's path
     * recorder reads it while a take is running, so what gets keyed is exactly
     * what is on screen — the camera's own orientation, not an orbit aim.
     */
    let frame = 0;
    let lastQuaternion: Quaternion = [0, 0, 0, 1];
    let lastTick = performance.now();
    const tick = (now: number = performance.now()) => {
      frame = requestAnimationFrame(tick);
      const dt = Math.min(0.1, (now - lastTick) / 1000);
      lastTick = now;
      // OrbitControls.update() ends with `camera.lookAt(target)`, which would
      // overwrite the camera-view quaternion every frame. While looking through
      // the project camera (or flying), the model owns the view. Its return
      // value also tells us whether the camera actually moved (damping keeps it
      // true until the motion settles) — an idle viewport then costs no draw.
      const flying = flyModeRef.current;
      const moved = inCameraView.current || flying ? false : orbit.update();
      if (flying && flyKeys.current.size > 0) {
        const speed = (flyKeys.current.has("shift") ? 14 : 5) * dt;
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
        if (flyKeys.current.has("w")) camera.position.addScaledVector(forward, speed);
        if (flyKeys.current.has("s")) camera.position.addScaledVector(forward, -speed);
        if (flyKeys.current.has("a")) camera.position.addScaledVector(right, -speed);
        if (flyKeys.current.has("d")) camera.position.addScaledVector(right, speed);
        if (flyKeys.current.has("e")) camera.position.y += speed;
        if (flyKeys.current.has("q")) camera.position.y -= speed;
        runtime.needsRender = true;
      }
      updateViewStore();
      viewStore.cameraTransform = cameraTransform(camera);
      // Cartography labels ride the projection: cheap per frame, and the
      // occlusion pass (a ray per anchor against the terrain) only runs
      // when the camera actually moved.
      runtime.overlay.visible = viewportModeRef.current !== "rendered";
      const labelLayerEl = runtime.labelLayer;
      if (labelLayerEl) {
        if (runtime.overlay.visible && anchorsRef.current.length > 0) {
          const width = container.clientWidth;
          const height = container.clientHeight;
          const cameraMoved =
            Math.abs(camera.position.x - occlusionCameraRef.current.x) > 1e-4 ||
            Math.abs(camera.position.y - occlusionCameraRef.current.y) > 1e-4 ||
            Math.abs(camera.position.z - occlusionCameraRef.current.z) > 1e-4 ||
            Math.abs(camera.quaternion.x - occlusionQuatRef.current.x) > 1e-4 ||
            Math.abs(camera.quaternion.y - occlusionQuatRef.current.y) > 1e-4 ||
            Math.abs(camera.quaternion.z - occlusionQuatRef.current.z) > 1e-4 ||
            Math.abs(camera.quaternion.w - occlusionQuatRef.current.w) > 1e-4 ||
            anchorsVersionRef.current !== occlusionVersionRef.current;
          if (cameraMoved) {
            const terrainObject = stateRef.current?.project.scene.objects.find(
              (object) => object.kind.type === "terrain",
            );
            const terrainGroup = terrainObject
              ? runtime.objects.get(terrainObject.id)
              : undefined;
            let terrainMesh: THREE.Mesh | undefined;
            for (const child of terrainGroup?.children ?? []) {
              if ((child as THREE.Mesh).isMesh) {
                terrainMesh = child as THREE.Mesh;
                break;
              }
            }
            const ray = new THREE.Raycaster();
            for (const anchor of anchorsRef.current) {
              if (!terrainMesh) {
                anchor.occluded = false;
                continue;
              }
              const direction = anchor.world.clone().sub(camera.position);
              const distance = direction.length();
              ray.set(camera.position, direction.normalize());
              ray.far = distance - 0.05;
              anchor.occluded = ray.intersectObject(terrainMesh, false).length > 0;
            }
            occlusionCameraRef.current.copy(camera.position);
            occlusionQuatRef.current.copy(camera.quaternion);
            occlusionVersionRef.current = anchorsVersionRef.current;
          }
          labelLayerEl.style.display = "";
          const screen = new THREE.Vector3();
          for (const anchor of anchorsRef.current) {
            const div = labelDivsRef.current.get(anchor.id);
            if (!div) continue;
            // Behind the camera: a view-space test, not the projected z.
            const view = anchor.world.clone().applyMatrix4(camera.matrixWorldInverse);
            if (view.z >= -0.05 || anchor.occluded) {
              div.style.display = "none";
              continue;
            }
            screen.copy(anchor.world).project(camera);
            const x = (screen.x * 0.5 + 0.5) * width;
            const y = (-screen.y * 0.5 + 0.5) * height;
            if (x < -80 || y < -40 || x > width + 80 || y > height + 40) {
              div.style.display = "none";
              continue;
            }
            div.style.display = "";
            div.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -115%)`;
          }
        } else {
          labelLayerEl.style.display = "none";
        }
      }
      for (const helper of runtime.helpers) helper.update();
      const q = camera.quaternion;
      if (
        Math.abs(q.x - lastQuaternion[0]) > 1e-4 ||
        Math.abs(q.y - lastQuaternion[1]) > 1e-4 ||
        Math.abs(q.z - lastQuaternion[2]) > 1e-4 ||
        Math.abs(q.w - lastQuaternion[3]) > 1e-4
      ) {
        lastQuaternion = [q.x, q.y, q.z, q.w];
        setQuaternion(lastQuaternion);
      }
      if (!moved && !runtime.needsRender) return;
      runtime.needsRender = false;
      renderer.render(scene, camera);
      // Some webviews reject the first draw of a shadow/float-texture pass.
      // Probe once per change; a rejected draw downgrades one step and the
      // next frame re-probes, so the error cannot repeat forever.
      if (!runtime.glProbed) {
        runtime.glProbed = true;
        const error = renderer.getContext().getError();
        if (error !== renderer.getContext().NO_ERROR) {
          setGpuNote(applyGpuStep(runtime));
          runtime.glProbed = false;
          runtime.needsRender = true;
        }
      }
    };
    tick();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("keydown", onFlyKeyDown);
      window.removeEventListener("keyup", onFlyKeyUp);
      window.removeEventListener("blur", onWindowBlur);
      gizmo.detach();
      gizmo.dispose();
      orbit.dispose();
      for (const group of runtime.objects.values()) {
        disposeChildren(group);
        scene.remove(group);
      }
      for (const helper of runtime.helpers) {
        scene.remove(helper);
        helper.geometry.dispose();
        (helper.material as THREE.Material).dispose();
      }
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      runtime.studioEnvironment?.dispose();
      environmentRef.current?.source.dispose();
      environmentRef.current?.background.dispose();
      environmentRef.current?.target.dispose();
      runtime.pmrem?.dispose();
      disposeOverlayObjects(overlay);
      scene.remove(overlay);
      overlay.clear();
      labelLayer.remove();
      renderer.dispose();
      runtimeRef.current = null;
    };
  }, []);

  // Project state → three.js scene. Runs on every edit; the map diff keeps it
  // to what actually changed.
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    const project = state?.project;
    // The viewport hillshade flag (shared for both call sites in this effect).
    const hillshade = project?.viewport.hillshade ?? true;
    if (!project) {
      for (const group of runtime.objects.values()) {
        disposeChildren(group);
        runtime.scene.remove(group);
      }
      runtime.objects.clear();
      runtime.selection = [];
      runtime.gizmo.detach();
      for (const helper of runtime.helpers) {
        runtime.scene.remove(helper);
        helper.geometry.dispose();
        (helper.material as THREE.Material).dispose();
      }
      runtime.helpers = [];
      runtime.helperKey = "";
      // Nothing references these any more: free them rather than keeping the
      // closed project's bytes, scenes and textures for the session.
      clearAssetCaches();
      for (const pending of textureCache.current.values()) {
        void pending.then((texture) => texture.dispose()).catch(() => undefined);
      }
      textureCache.current.clear();
      runtime.needsRender = true;
      return;
    }

    runtime.selection = project.selection;
    const activeCameraId = project.scene.activeCamera;
    const live = new Map(project.scene.objects.map((object) => [object.id, object]));
    for (const [id, group] of runtime.objects) {
      if (live.has(id)) continue;
      disposeChildren(group);
      runtime.scene.remove(group);
      runtime.objects.delete(id);
    }

    for (const object of project.scene.objects) {
      let group = runtime.objects.get(object.id);
      if (!group) {
        group = new THREE.Group();
        group.userData.studioId = object.id;
        runtime.scene.add(group);
        runtime.objects.set(object.id, group);
      }
      group.visible = object.visible && !(cameraView && object.id === activeCameraId);
      applyTransform(group, object.transform);

      // The preview's elevation ramp: the engine's own LUT, sampled by the
      // shell and applied to the mesh, so Material shows the map without a
      // Rendered frame.
      // The preview follows the map's stack: the topmost visible ramp is the
      // one the map shows.
      const ramp = object.kind.type === "terrain" ? topRamp(project) : null;
      const rampName = ramp?.colormap ?? "";
      const rampStrength = ramp?.strength ?? 1;
      const rampKey = rampName ? `${rampName}:${rampStrength}` : "none";
      const signature = contentSignature(object, terrainResolution, rampKey, {
        asset: topImagery(project)?.asset ?? "",
        opacity: topImagery(project)?.opacity ?? 0,
      });
      if (group.userData.signature !== signature) {
        group.userData.signature = signature;
        disposeChildren(group);
      } else {
        // Rebuilt content keeps its geometry; a mode switch (or a material
        // edit) only changes how it is shaded.
        applyShading(runtime, object, viewportModeRef.current, hillshade);
        continue;
      }

      if (object.kind.type === "mesh") {
        const meshKind = object.kind;
        const source = meshKind.source;
        if (source.type === "primitive") {
          const built = buildPrimitive(source.primitive, meshKind.material);
          group.add(built);
          const textureAsset = meshKind.material.texture
            ? project.assets.find((entry) => entry.id === meshKind.material.texture)
            : undefined;
          if (textureAsset && built instanceof THREE.Mesh) {
            const expected = signature;
            const mesh = built;
            loadTexture(textureAsset.path)
              .then((texture) => {
                if (group.userData.signature !== expected) return;
                const material = mesh.material as THREE.MeshStandardMaterial;
                material.map = texture;
                material.needsUpdate = true;
              })
              .catch(() => undefined);
          }
        } else {
          const asset = project.assets.find((entry) => entry.id === source.asset);
          if (!asset) {
            setProblem(
              `Object "${object.name}" references a missing asset — relink it in the Assets panel.`,
            );
            continue;
          }
          group.add(buildLoadingPlaceholder());
          const expected = signature;
          loadAssetScene(asset.path, readMesh, source.node)
            .then((loaded) => {
              if (group.userData.signature !== expected) {
                disposeTree(loaded);
                return;
              }
              disposeChildren(group);
              group.add(loaded);
              // glTF materials are the Material preview's albedo; Solid draws
              // the same geometry as clay.
              setSolidShading(loaded, viewportModeRef.current === "solid");
              setProblem(null);
              runtime.needsRender = true;
            })
            .catch((cause) => {
              setProblem(
                `Could not load ${asset.name}: ${cause instanceof Error ? cause.message : cause}`,
              );
            });
        }
      } else if (object.kind.type === "terrain") {
        const terrainKind = object.kind;
        const asset = project.assets.find((entry) => entry.id === terrainKind.asset);
        if (!asset) {
          setProblem(
            `Object "${object.name}" references a missing DEM — relink it in the Assets panel.`,
          );
          continue;
        }
        // The draped imagery is the topmost visible imagery layer now.
        const imagery = topImagery(project);
        const overlay = imagery
          ? { asset: imagery.asset ?? "", opacity: imagery.opacity ?? 0.82 }
          : null;
        const overlayAsset = overlay
          ? project.assets.find((entry) => entry.id === overlay.asset)
          : undefined;
        group.add(buildLoadingPlaceholder());
        const expected = signature;
        const zScale = terrainKind.zScale;
        Promise.all([
          readTerrain(asset.path),
          rampName
            ? loadColormapStops(rampName).then((stops) => ({
                stops,
                strength: rampStrength,
              }))
            : Promise.resolve(null),
        ])
          .then(([preview, ramp]) => {
            if (group.userData.signature !== expected) return;
            disposeChildren(group);
            const mesh = buildTerrainMesh(preview, zScale, ramp, hillshade);
            group.add(mesh);
            setProblem(null);
            applyTerrainShading(mesh, viewportModeRef.current === "solid", hillshade);
            // The drape is the terrain's albedo: Material shows it, Solid
            // stays clay. The shell decodes the raster; a scene without an
            // overlay stays clay in both.
            if (overlay && overlayAsset?.path) {
              loadOverlayTexture(overlayAsset.path)
                .then((texture) => {
                  if (group.userData.signature !== expected) return;
                  mesh.userData.drape = texture;
                  applyTerrainShading(mesh, viewportModeRef.current === "solid", hillshade);
                  runtime.needsRender = true;
                })
                .catch((cause) => {
                  setProblem(
                    `Could not drape ${overlayAsset.name}: ${cause instanceof Error ? cause.message : String(cause)}`,
                  );
                });
            }
          })
          .catch((cause) => {
            setProblem(
              `Could not sample ${asset.name}: ${cause instanceof Error ? cause.message : cause}`,
            );
          });
      } else if (object.kind.type === "camera") {
        group.add(buildCameraGizmo(object.kind));
      } else {
        group.add(buildLightGizmo(object.kind));
      }
      applyShading(runtime, object, viewportModeRef.current, hillshade);
    }

    // Camera view: the render camera *is* the project's active camera, so an
    // inspector edit moves the view immediately.
    const activeCamera =
      project.scene.objects.find(
        (object) => object.id === project.scene.activeCamera && object.kind.type === "camera",
      ) ?? project.scene.objects.find((object) => object.kind.type === "camera");

    // Fly mode owns the camera while it is on: applying the camera object's
    // transform here would snap the view back on every project edit.
    if (cameraView && !flyMode) {
      if (activeCamera && activeCamera.kind.type === "camera") {
        const transform = activeCamera.transform;
        runtime.camera.up.set(0, 1, 0);
        runtime.camera.position.set(
          transform.translation.x,
          transform.translation.y,
          transform.translation.z,
        );
        runtime.camera.quaternion.setFromEuler(
          new THREE.Euler(
            transform.rotationDeg.x * DEG,
            transform.rotationDeg.y * DEG,
            transform.rotationDeg.z * DEG,
            "ZYX",
          ),
        );
        runtime.camera.fov = activeCamera.kind.fovDeg;
        // Keep the projection usable even if the stored values are odd: a huge
        // near plane or a far plane behind the camera renders a black frame.
        const distance = runtime.camera.position.length();
        runtime.camera.near = Math.max(0.01, Math.min(activeCamera.kind.near, distance * 0.1));
        runtime.camera.far = Math.max(activeCamera.kind.far, distance * 8);
        runtime.camera.updateProjectionMatrix();
        runtime.orbit.enabled = false;
        // Clear a previous camera warning without wiping an asset error.
        setProblem((current) =>
          current && current.startsWith("Camera view") ? null : current,
        );
      } else {
        setProblem("Camera view needs a camera object in the scene.");
      }
    }

    // The scene's directional light drives the actual lighting.
    const solidView = viewportModeRef.current === "solid";
    // One derived rig for the preview and (via the same project) the engine.
    // `lighting.ts` is the single place these numbers come from; the shared
    // fixture test keeps it honest against the renderer payload.
    const rig = lightingRig(project);
    if (rig.sun) {
      const [dx, dy, dz] = rig.sun.direction;
      // Far enough out that the shadow camera encloses the terrain.
      runtime.sun.position.set(-dx * 20, -dy * 20, -dz * 20);
      // The engine multiplies `sun_color * intensity` in linear radiance
      // (`u_shading.light_params`). Matching that means no sRGB conversion
      // here and no intensity fudge: the preview and the still then evaluate
      // the same `albedo/pi * sun * n.l` term.
      const [r, g, b] = rig.sun.color;
      runtime.sun.color.setRGB(r, g, b, THREE.LinearSRGBColorSpace);
      runtime.sun.intensity = rig.sun.intensity;
    }

    // One rig for every mode: the scene sun plus the project environment (or
    // the built-in sky). Solid renders the same light on clay; Material on the
    // albedo drape; Rendered is the engine frame, which uses this same sun,
    // environment and exposure.
    applyEnvironment(
      runtime,
      solidView ? "solid" : "material",
      project.render.iblIntensity,
      project.render.environmentBackgroundIntensity,
    );
    // One exposure for every mode — the project's, applied by the engine in
    // Rendered; the preview now agrees instead of running 0.95 against the
    // engine's 1.35.
    runtime.renderer.toneMappingExposure = rig.exposure;
    applyShadowRig(runtime, rig);

    runtime.grid.visible = project.viewport.gridVisible;

    // Selection boxes follow their objects every tick (playback moves them at
    // the timeline's rate), so only rebuild them when the selection or the
    // camera view changes.
    const helperKey = `${project.selection.join(",")}|${cameraView ? activeCameraId ?? "" : ""}`;
    if (runtime.helperKey === helperKey) {
      for (const helper of runtime.helpers) helper.update();
    } else {
      runtime.helperKey = helperKey;
      for (const helper of runtime.helpers) {
        runtime.scene.remove(helper);
        helper.geometry.dispose();
        (helper.material as THREE.Material).dispose();
      }
      runtime.helpers = [];
      for (const id of project.selection) {
        const group = runtime.objects.get(id);
        if (!group) continue;
        // The box around the camera we are looking through is drawn from inside
        // itself and reads as stray lines across the view.
        if (cameraView && id === activeCameraId) continue;
        const helper = new THREE.BoxHelper(group, ACCENT);
        runtime.scene.add(helper);
        runtime.helpers.push(helper);
      }
    }

    const single =
      project.selection.length === 1
        ? project.scene.objects.find((object) => object.id === project.selection[0])
        : undefined;
    // Terrain carries a measured georeferenced footprint: no gizmo (and the
    // SetTransform command rejects it anyway) — moving it would only desync
    // the Solid view from the renderer, which always draws the DEM centred.
    const editable =
      single && !single.locked && single.visible && single.kind.type !== "terrain"
        ? runtime.objects.get(single.id)
        : undefined;
    if (tool === "select" || !editable) {
      runtime.gizmo.detach();
    } else {
      runtime.gizmo.setMode(tool === "move" ? "translate" : tool === "rotate" ? "rotate" : "scale");
      runtime.gizmo.setSpace(space);
      if (runtime.gizmo.object !== editable) runtime.gizmo.attach(editable);
    }
    // Everything above mutated the scene; draw once on the next tick.
    runtime.needsRender = true;
  }, [
    cameraView,
    flyMode,
    loadColormapStops,
    loadTexture,
    readMesh,
    readTerrain,
    space,
    state,
    terrainResolution,
    tool,
  ]);

  // The map overlay: vector lines, their labels and the graticule, draped
  // on the terrain in the same 0..1 lattice the still uses. Rebuilt only
  // when the relevant layers (or the terrain) change; a camera move never
  // refetches.
  const cartographyLayers = state?.project.cartography;
  const cartographySignature = cartographyLayers
    ? JSON.stringify(
        cartographyLayers
          .filter(
            (layer) =>
              layer.visible &&
              (layer.type === "vector" ||
                layer.type === "polygon" ||
                layer.type === "labels" ||
                layer.type === "graticule"),
          )
          .map((layer) => [
            layer.id,
            layer.type,
            layer.asset ?? "",
            layer.colour?.join(",") ?? "",
            layer.widthPx ?? 0,
            layer.intervalDeg ?? 0,
            layer.labels === false ? "0" : "1",
            layer.source ?? "",
            layer.field ?? "",
            layer.legendField ?? "",
            layer.sizePx ?? 0,
          ]),
      )
    : "";
  const lastCartographySignature = useRef("__unset__");
  useEffect(() => {
    const runtime = runtimeRef.current;
    const container = containerRef.current;
    if (!runtime || !container) return;
    if (lastCartographySignature.current === cartographySignature) return;
    lastCartographySignature.current = cartographySignature;
    let cancelled = false;

    // The overlay always mirrors the stack: whatever it had, it drops.
    disposeOverlayObjects(runtime.overlay);
    runtime.overlay.visible = viewportModeRef.current !== "rendered";
    if (runtime.labelLayer) {
      runtime.labelLayer.style.display = runtime.overlay.visible ? "" : "none";
      runtime.labelLayer.replaceChildren();
    }
    runtime.lineMaterials = [];
    labelDivsRef.current = new Map();
    anchorsRef.current = [];

    const project = stateRef.current?.project;
    const terrainObject = project?.scene.objects.find(
      (object) => object.kind.type === "terrain",
    );
    if (!project || !terrainObject || terrainObject.kind.type !== "terrain") {
      runtime.needsRender = true;
      return;
    }
    const terrainKind = terrainObject.kind;
    const terrainAsset = project.assets.find((asset) => asset.id === terrainKind.asset);
    if (!terrainAsset) {
      runtime.needsRender = true;
      return;
    }
    const zScale = terrainKind.zScale;

    (async () => {
      let preview: TerrainPreview;
      try {
        // Shares the terrain cache with the mesh build — one read per DEM.
        preview = await readTerrain(terrainAsset.path);
      } catch {
        // The terrain's own error path already reports this one.
        return;
      }
      if (cancelled) return;
      const spanXz = preview.spanXz;
      const lift = drapeLift(spanXz, [preview.minElevation, preview.maxElevation], zScale);
      const lineMaterials: LineMaterial[] = [];
      const anchors: CartographyAnchor[] = [];
      const addDraped = (
        lines: [number, number][][],
        layer: CartographyLayer,
        colourOverride?: [number, number, number, number],
      ) => {
        if (lines.length === 0) return;
        const { positions, pairs } = drapePolylines(lines, preview, spanXz, zScale, lift);
        if (pairs.length === 0) return;
        const geometry = new LineSegmentsGeometry();
        geometry.setPositions(expandPairs(positions, pairs));
        const rgba = colourOverride ?? layer.colour ?? [1, 0.62, 0.2, 1];
        const material = new LineMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          linewidth: Math.max(1, layer.widthPx ?? 2),
          transparent: true,
          opacity: rgba[3],
          depthTest: true,
          // Lines drape ON the terrain: the depth test keeps the hidden
          // stretch of a line behind a ridge hidden.
          depthWrite: false,
        });
        material.resolution.set(container.clientWidth, container.clientHeight);
        const line = new LineSegments2(geometry, material);
        // Instanced bounding boxes are unreliable here; the overlay is tiny.
        line.frustumCulled = false;
        line.renderOrder = 4;
        runtime.overlay.add(line);
        lineMaterials.push(material);
      };
      for (const layer of vectorLayers(project)) {
        const assetId = layer.asset;
        const file = assetId
          ? project.assets.find((entry) => entry.id === assetId)
          : undefined;
        if (!file) continue;
        try {
          const legend = (layer.legendField ?? "").trim();
          const result = await bridge.vectorGeometry(
            terrainAsset.path,
            file.path,
            "",
            legend,
          );
          if (cancelled) return;
          if (legend && result.categories && result.palette) {
            const groups = new Map<string, [number, number][][]>();
            for (const [index, line] of result.lines.entries()) {
              const category = result.categories[index] ?? "";
              const bucket = groups.get(category) ?? [];
              bucket.push(line);
              groups.set(category, bucket);
            }
            const alpha = layer.colour?.[3] ?? 1;
            for (const [value, subLines] of groups) {
              const rgb = result.palette[value];
              // A category absent from the palette (uncategorised) keeps the
              // layer colour — the same rule the still applies.
              addDraped(
                subLines,
                layer,
                rgb ? [rgb[0], rgb[1], rgb[2], alpha] : layer.colour,
              );
            }
          } else {
            addDraped(result.lines, layer);
          }
        } catch (cause) {
          if (!cancelled) {
            setProblem(
              `"${layer.name}" could not be drawn: ${cause instanceof Error ? cause.message : String(cause)}`,
            );
          }
        }
      }
      for (const layer of labelLayers(project)) {
        const source = project.cartography.find((entry) => entry.id === layer.source);
        const file = source?.asset
          ? project.assets.find((entry) => entry.id === source.asset)
          : undefined;
        // A label layer whose source vector is gone is skipped in silence
        // (the Inspector shows the dangling reference).
        if (!file) continue;
        try {
          const result = await bridge.vectorGeometry(
            terrainAsset.path,
            file.path,
            layer.field ?? "",
          );
          if (cancelled) return;
          const ink = labelInk(layer.colour);
          for (const [index, point] of result.points.entries()) {
            anchors.push({
              id: `${layer.id}:${index}`,
              text: point.text,
              className: "cartography-label--feature",
              ...ink,
              fontSizePx: layer.sizePx ?? 13,
              world: new THREE.Vector3(
                ...anchorWorld(preview, spanXz, zScale, point.u, point.v),
              ),
              occluded: false,
            });
          }
        } catch (cause) {
          if (!cancelled) {
            setProblem(
              `"${layer.name}" could not be drawn: ${cause instanceof Error ? cause.message : String(cause)}`,
            );
          }
        }
      }
      for (const layer of graticuleLayers(project)) {
        const interval = layer.intervalDeg ?? 1;
        try {
          const result = await bridge.graticuleGeometry(
            terrainAsset.path,
            interval,
            layer.labels ?? true,
          );
          if (cancelled) return;
          addDraped(result.lines, layer);
          const ink = labelInk(layer.colour);
          for (const [index, annotation] of result.labels.entries()) {
            anchors.push({
              id: `${layer.id}:g${index}`,
              text: annotation.text,
              className: `cartography-label--graticule cartography-label--${annotation.kind}`,
              ...ink,
              // The engine's own grid annotation size.
              fontSizePx: 12,
              world: new THREE.Vector3(
                ...anchorWorld(preview, spanXz, zScale, annotation.u, annotation.v),
              ),
              occluded: false,
            });
          }
        } catch (cause) {
          if (!cancelled) {
            setProblem(
              `"${layer.name}" could not be drawn: ${cause instanceof Error ? cause.message : String(cause)}`,
            );
          }
        }
      }
      for (const layer of polygonLayers(project)) {
        const assetId = layer.asset;
        const file = assetId
          ? project.assets.find((entry) => entry.id === assetId)
          : undefined;
        if (!file) continue;
        try {
          const result = await bridge.polygonGeometry(terrainAsset.path, file.path);
          if (cancelled) return;
          const mesh = buildPolygonMesh(
            result.exteriors,
            result.holes,
            preview,
            spanXz,
            zScale,
            lift,
            layer.colour ?? [0.2, 0.55, 0.75, 0.7],
          );
          if (mesh) runtime.overlay.add(mesh);
        } catch (cause) {
          if (!cancelled) {
            setProblem(
              `"${layer.name}" could not be drawn: ${cause instanceof Error ? cause.message : String(cause)}`,
            );
          }
        }
      }
      if (cancelled) return;
      runtime.lineMaterials = lineMaterials;
      anchorsRef.current = anchors;
      anchorsVersionRef.current += 1;
      if (runtime.labelLayer) {
        labelDivsRef.current = rebuildCartographyLabels(runtime.labelLayer, anchors);
      }
      runtime.needsRender = true;
    })().catch((cause) => {
      if (!cancelled) {
        setProblem(
          `the map overlay could not be built: ${cause instanceof Error ? cause.message : String(cause)}`,
        );
      }
    });
    return () => {
      cancelled = true;
    };
  }, [cartographySignature, readTerrain]);

  // A project HDRI replaces the studio sky for Material and Rendered. The
  // viewport decodes and pre-filters it here (Rendered receives the path and
  // lets the engine upload it); a replaced map is freed after the new one is
  // bound, and a failed load keeps whatever was lighting the scene before.
  // The rig owns the project→renderer mapping, including which HDRI file the
  // project points at; keep one lookup.
  const environmentPath = state ? lightingRig(state.project).environment : null;
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    let cancelled = false;
    const settings = stateRef.current?.project.render;
    const apply = () => {
      applyEnvironment(
        runtime,
        viewportModeRef.current === "solid" ? "solid" : "material",
        settings?.iblIntensity ?? 1,
        settings?.environmentBackgroundIntensity ?? 1,
      );
      runtime.needsRender = true;
    };

    if (!environmentPath) {
      const previous = environmentRef.current;
      environmentRef.current = null;
      previous?.source.dispose();
      previous?.background.dispose();
      previous?.target.dispose();
      runtime.environment = runtime.studioEnvironment;
      runtime.environmentBackground = null;
      apply();
      return;
    }

    void loadEnvironment(environmentPath)
      .then((loaded) => {
        if (cancelled) {
          loaded.source.dispose();
          loaded.target.dispose();
          return;
        }
        const previous = environmentRef.current;
        environmentRef.current = loaded;
        runtime.environment = loaded.environment;
        runtime.environmentBackground = loaded.background;
          // The PMREM output is new; let the probe see this texture.
        runtime.glProbed = false;
        apply();
        if (previous && previous.path !== loaded.path) {
          previous.source.dispose();
          previous.background.dispose();
          previous.target.dispose();
        }
        setProblem((current) =>
          current && current.startsWith("Could not load HDRI") ? null : current,
        );
      })
      .catch((cause) => {
        if (cancelled) return;
        setProblem(
          `Could not load HDRI ${environmentPath}: ${cause instanceof Error ? cause.message : cause}`,
        );
      });
    return () => {
      cancelled = true;
    };
  }, [environmentPath, loadEnvironment]);

  // The decoded HDRI belongs to the viewport, not the cache, so free it here.
  useEffect(
    () => () => {
      const custom = environmentRef.current;
      environmentRef.current = null;
      custom?.source.dispose();
      custom?.background.dispose();
      custom?.target.dispose();
    },
    [],
  );

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;

    if (cameraView && !inCameraView.current) {
      savedView.current = {
        position: runtime.camera.position.clone(),
        quaternion: runtime.camera.quaternion.clone(),
        target: runtime.orbit.target.clone(),
        fov: runtime.camera.fov,
      };
      inCameraView.current = true;
      runtime.orbit.enabled = false;
    } else if (!cameraView && inCameraView.current) {
      const saved = savedView.current;
      if (saved) {
        runtime.camera.position.copy(saved.position);
        runtime.camera.quaternion.copy(saved.quaternion);
        runtime.camera.fov = saved.fov;
        runtime.camera.updateProjectionMatrix();
        runtime.orbit.target.copy(saved.target);
      }
      inCameraView.current = false;
      runtime.orbit.enabled = true;
      runtime.orbit.update();
      setProblem((current) =>
        current && current.startsWith("Camera view") ? null : current,
      );
    }
  }, [cameraView]);

  // Fly mode: free look, no orbit target. Leaving it hands the camera back to
  // OrbitControls, aimed ahead so orbiting continues from where you stopped.
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    if (!flyMode) {
      runtime.orbit.enabled = true;
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(runtime.camera.quaternion);
      runtime.orbit.target.copy(runtime.camera.position).addScaledVector(forward, 10);
      runtime.orbit.update();
      runtime.needsRender = true;
      return;
    }
    // Camera view and fly mode cannot both own the camera.
    if (cameraView) setCameraView(false);
    runtime.orbit.enabled = false;
    flyKeys.current.clear();
    const euler = new THREE.Euler().setFromQuaternion(runtime.camera.quaternion, "YXZ");
    flyAngles.current = { yaw: euler.y, pitch: euler.x };
    runtime.needsRender = true;
  }, [flyMode, cameraView, setCameraView]);

  const viewAlong = useCallback((axis: Axis) => {
    if (cameraView || flyModeRef.current) return;
    const runtime = runtimeRef.current;
    if (!runtime) return;
    const direction =
      axis === "x"
        ? new THREE.Vector3(1, 0, 0)
        : axis === "y"
          ? new THREE.Vector3(0, 1, 0)
          : new THREE.Vector3(0, 0, 1);
    const distance = runtime.camera.position.distanceTo(runtime.orbit.target);
    runtime.camera.position.copy(runtime.orbit.target).add(direction.multiplyScalar(distance));
    // Looking straight down needs a different up vector or the view is degenerate.
    runtime.camera.up.set(0, axis === "y" ? 0 : 1, axis === "y" ? -1 : 0);
    runtime.camera.lookAt(runtime.orbit.target);
    runtime.orbit.update();
    runtime.needsRender = true;
  }, []);

  /** Viewport render resolution for Rendered mode. Options: 1×/2×/4× the
   *  viewport, the DEM's own pixel size, or the project's output size. */
  type RenderScale = "1" | "2" | "4" | "project" | "dem";
  const [renderScale, setRenderScale] = useState<RenderScale>("1");

  const terrainObject = state?.project.scene.objects.find((object) => object.kind.type === "terrain");
  const demDims = terrainObject?.kind.type === "terrain" ? terrainObject.kind.dimensions : null;

  const renderScaleOptions: Array<{ value: RenderScale; label: string }> = [
    { value: "1", label: "1× viewport" },
    { value: "2", label: "2× viewport" },
    { value: "4", label: "4× viewport" },
    ...(demDims
      ? [{ value: "dem" as RenderScale, label: `DEM (${demDims[0]} × ${demDims[1]})` }]
      : []),
    {
      value: "project",
      label: `Project (${state?.project.render.width ?? 1920} × ${state?.project.render.height ?? 1080})`,
    },
  ];
  const activeScaleLabel =
    renderScaleOptions.find((option) => option.value === renderScale)?.label ?? "1× viewport";

  // An in-app menu — never the native `<select>` popup, which Gdk maps as a
  // separate window (the "popup with a non-topmost parent" warning, and it
  // lands behind the shell window on some WMs).
  const [sizeMenuOpen, setSizeMenuOpen] = useState(false);
  const sizeMenuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!sizeMenuOpen) return;
    const close = (event: MouseEvent) => {
      if (!sizeMenuRef.current?.contains(event.target as Node)) setSizeMenuOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSizeMenuOpen(false);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", escape);
    };
  }, [sizeMenuOpen]);

  /** Frame size for a Rendered viewport frame. No hidden multipliers — what
   *  the selector shows is the size the engine renders. The cap scales the
   *  whole frame down uniformly so the aspect never changes.
   *
   *  The cap is 2048, not 4096: the engine renders frames natively at the
   *  requested size and the transfer rides IPC as base64, so a frame bigger
   *  than 2048 costs the webview megabytes of decode for no visible gain on
   *  a preview surface. A still render skips both the cap and the base64
   *  (`render_still`); use that for the output file. */
  const requestSize = useCallback(() => {
    const base =
      renderScale === "project"
        ? { width: state?.project.render.width ?? 1920, height: state?.project.render.height ?? 1080 }
        : renderScale === "dem" && demDims
          ? { width: demDims[0], height: demDims[1] }
          : { width: Math.max(320, size.width), height: Math.max(240, size.height) };
    const factor = renderScale === "project" || renderScale === "dem" ? 1 : Number(renderScale);
    const cap = 2048;
    const shrink = Math.min(1, cap / (base.width * factor), cap / (base.height * factor));
    return {
      width: Math.max(64, Math.round(base.width * factor * shrink)),
      height: Math.max(64, Math.round(base.height * factor * shrink)),
    };
  }, [demDims, renderScale, size.height, size.width, state?.project.render.height, state?.project.render.width]);
  // Only Rendered composes an engine frame. Solid and Material are the live
  // three.js scene — that is what keeps Material's orbit free and instant.
  // The plate's crop frame: the still composes at the project's render
  // settings, so the viewport shows that aspect as a dashed letterbox while
  // the Plate layer is visible — the plate's crop shape, at the still's
  // aspect, centred on the canvas.
  const plateVisible = (state?.project.cartography ?? []).some(
    (layer) => layer.type === "plate" && layer.visible,
  );
  const plateSettings = state?.project.render;
  const plateAspect =
    plateSettings && plateSettings.width > 0 && plateSettings.height > 0
      ? plateSettings.width / plateSettings.height
      : null;
  const containerAspect = size.width > 0 && size.height > 0 ? size.width / size.height : null;
  let plateFrame: { width: number; height: number; renderW: number; renderH: number } | null =
    null;
  if (plateVisible && plateAspect && containerAspect && plateSettings) {
    if (containerAspect >= plateAspect) {
      plateFrame = {
        width: size.height * plateAspect,
        height: size.height,
        renderW: plateSettings.width,
        renderH: plateSettings.height,
      };
    } else {
      plateFrame = {
        width: size.width,
        height: size.width / plateAspect,
        renderW: plateSettings.width,
        renderH: plateSettings.height,
      };
    }
  }
  const renderedMode = mode === "rendered";
  const modeRef = useRef(renderedMode);
  modeRef.current = renderedMode;
  const cameraViewRef = useRef(cameraView);
  cameraViewRef.current = cameraView;
  const requestSizeRef = useRef(requestSize);
  requestSizeRef.current = requestSize;
  const sizeRef = useRef(size);
  sizeRef.current = size;
  const stateRef = useRef(state);
  stateRef.current = state;
  const setCameraViewRef = useRef(setCameraView);
  setCameraViewRef.current = setCameraView;

  /** The solid camera as a scene transform, aimed at the terrain centre. */
  const cameraTransformFromView = useCallback((): Transform | null => {
    const runtime = runtimeRef.current;
    return runtime ? cameraTransform(runtime.camera) : null;
  }, []);

  /**
   * Store the current viewport view as the scene's active camera, so the
   * renderer uses exactly what the user is looking at. Blender uses
   * Ctrl+Alt+Numpad0; this is bound to End / Numpad 1 as requested.
   */
  const captureCamera = useCallback(async () => {
    const runtime = runtimeRef.current;
    const project = stateRef.current?.project;
    if (!runtime || !project) return;
    const transform = cameraTransformFromView();
    if (!transform) return;

    const active = project.scene.objects.find(
      (object) => object.id === project.scene.activeCamera,
    );
    if (!active) {
      setProblem("There is no camera object to receive this view — add one from Add.");
      return;
    }
    await actionsRef.current.apply({
      type: "setTransform",
      id: active.id,
      transform,
    });
    await actionsRef.current.apply({ type: "setActiveCamera", id: active.id });
    // Numpad 1 should also put the viewport into camera mode — unless fly mode
    // owns the camera.
    if (!flyModeRef.current) setCameraViewRef.current(true);
    setProblem(null);
    actionsRef.current.notify(`Camera "${active.name}" set from this view`);
  }, [cameraTransformFromView]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "End" || event.code === "Numpad1") {
        event.preventDefault();
        void captureCamera();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [captureCamera]);

  // A frame belongs to the camera and scene that produced it. Entering Rendered
  // drops the previous image so a stale view cannot cover the live scene while
  // the first engine frame is produced.
  const wasRendered = useRef(false);
  useEffect(() => {
    if (renderedMode && !wasRendered.current) {
      actionsRef.current.clearRenderFrame();
    }
    wasRendered.current = renderedMode;
  }, [renderedMode]);

  // Entering Rendered mode renders once at the current viewport size, and a
  // change of that size renders again: a window resize otherwise leaves a frame
  // at the old size covering the new viewport. Orbiting re-renders through the
  // orbit handler, which the session coalesces ("newest wins").
  const renderedOnce = useRef(false);
  const renderedSize = useRef<{ width: number; height: number } | null>(null);
  useEffect(() => {
    if (!renderedMode) {
      renderedOnce.current = false;
      renderedSize.current = null;
      return;
    }
    if (render.error) {
      // A failed frame must not latch: let the next switch/render try again.
      renderedOnce.current = false;
      return;
    }
    const target = requestSizeRef.current();
    const resized =
      renderedSize.current === null ||
      renderedSize.current.width !== target.width ||
      renderedSize.current.height !== target.height;
    if (render.busy || (renderedOnce.current && !resized)) return;
    renderedOnce.current = true;
    renderedSize.current = target;
    void actions.renderViewport(target.width, target.height, "preview");
  }, [actions, render.busy, renderedMode, render.error, size.height, size.width]);

  // Scene edits (sun, exposure, overlay, z-scale, transforms…) re-render the
  // frame after a short pause, so the Rendered view reflects the project
  // instead of a stale PNG until someone presses Re-render. Debounced and
  // serialised in the session, so dragging a slider still costs one frame.
  const revision = state?.revision ?? -1;
  const revisionRef = useRef(revision);
  useEffect(() => {
    const previous = revisionRef.current;
    revisionRef.current = revision;
    if (!renderedMode || revision === previous || previous === -1) return;
    const timer = window.setTimeout(() => {
      const target = requestSizeRef.current();
      void actionsRef.current.renderViewport(target.width, target.height, "preview");
    }, 350);
    return () => window.clearTimeout(timer);
  }, [revision, renderedMode]);

  // Changing the render scale asks for a frame at the new size right away.
  useEffect(() => {
    if (!renderedMode) return;
    const timer = window.setTimeout(() => {
      const target = requestSizeRef.current();
      void actionsRef.current.renderViewport(target.width, target.height, "preview");
    }, 50);
    return () => window.clearTimeout(timer);
    // Only the scale itself; size/mode changes already flow through the
    // effects above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renderScale]);

  const scene = state?.project.scene;
  const camera = scene?.objects.find((object) => object.id === scene.activeCamera);
  // The renderer is a terrain pipeline: say which objects it cannot draw.
  const unrendered = (scene?.objects ?? [])
    .filter((object) => object.kind.type === "terrain")
    .length === 0
    ? ["everything"]
    : (scene?.objects ?? [])
        .filter((object) => object.kind.type === "mesh" && object.visible)
        .map((object) => object.name);

  return (
    <div className="viewport" ref={containerRef}>
      <canvas ref={canvasRef} className="viewport-canvas" />
      {renderedMode && (
        <div className={`viewport-rendered ${render.frame ? "has-frame" : ""}`}>
          {render.frame && (
            <img
              src={`data:image/png;base64,${render.frame.png}`}
              alt="Forge3D render of the current scene"
              className={render.busy ? "is-updating" : undefined}
            />
          )}
          {render.busy && render.frame && (
            <div className="viewport-rendered__updating">Updating…</div>
          )}
          {unrendered.length > 0 && (
            <div className="viewport-rendered__notice">
              Rendered mode draws the terrain only — {unrendered.join(", ")}{" "}
              {unrendered.length === 1 ? "is" : "are"} not in this frame yet
              (geometry export is pending).
            </div>
          )}
          <div className="viewport-rendered__bar">
            <span>
              {render.busy
                ? render.frame
                  ? `Updating… (showing ${render.frame.width} × ${render.frame.height})`
                  : "Rendering with the Forge3D terrain pipeline…"
                : render.frame
                  ? `${render.frame.width} × ${render.frame.height} · ${(render.frame.ms / 1000).toFixed(2)} s`
                  : "No frame rendered yet"}
            </span>
            <div className="viewport-rendered__scale" ref={sizeMenuRef}>
              <span>Size</span>
              <button
                type="button"
                className="viewport-rendered__scale-button"
                aria-haspopup="menu"
                aria-expanded={sizeMenuOpen}
                title="Viewport frames render at this size; DEM renders at the DEM's own pixels"
                onClick={() => setSizeMenuOpen((open) => !open)}
              >
                {activeScaleLabel}
              </button>
              {sizeMenuOpen && (
                <div className="add-menu__dropdown viewport-rendered__menu" role="menu">
                  {renderScaleOptions.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      role="menuitemradio"
                      aria-checked={renderScale === option.value}
                      onClick={() => {
                        setSizeMenuOpen(false);
                        setRenderScale(option.value);
                      }}
                    >
                      {renderScale === option.value ? "✓ " : ""}
                      {option.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              className="button button--small"
              disabled={render.busy}
              onClick={() => {
                const target = requestSizeRef.current();
                void actions.renderViewport(target.width, target.height, "preview");
              }}
            >
              {render.busy ? "Rendering…" : "Re-render"}
            </button>
          </div>
        </div>
      )}
      {plateFrame && !renderedMode && (
        <div
          className="viewport-plate-frame"
          style={{ width: plateFrame.width, height: plateFrame.height }}
          aria-hidden="true"
        >
          <span className="viewport-plate-frame__caption">
            plate · {plateFrame.renderW} × {plateFrame.renderH}
          </span>
        </div>
      )}
      <ViewportToolbar
        onRender={() => {
          const target = requestSizeRef.current();
          void actions.renderViewport(target.width, target.height, "preview");
        }}
      />
      <AxisGizmo quaternion={quaternion} onPick={viewAlong} />
      <div className="viewport-overlay viewport-overlay--bottom">
        <span className="viewport-hint">
          {cameraView
            ? "Camera view · press 0 or the camera button to return to the free view"
            : tool === "select"
              ? "Click to select · drag to orbit · middle-drag to pan · scroll to zoom"
              : `${tool[0].toUpperCase()}${tool.slice(1)} tool · drag the gizmo · G/R/S to switch`}
        </span>
        <span className="viewport-camera">
          {camera ? `Active camera: ${camera.name}` : "No active camera"}
          {mode === "material" ? " · three.js preview" : mode === "rendered" ? " · Forge3D engine" : ""}
        </span>
        {gpuNote && <span className="viewport-gpu-note">{gpuNote}</span>}
      </div>
      {problem && !renderedMode && <div className="viewport-problem">{problem}</div>}
      {renderedMode && render.error && (
        <div className="viewport-problem">{render.error}</div>
      )}
    </div>
  );
}

function buildLoadingPlaceholder(): THREE.Object3D {
  const placeholder = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: 0x3a4150, wireframe: true }),
  );
  placeholder.userData.owned = true;
  return placeholder;
}
