/**
 * The recipes ARE the port of the engine's `forge3d.presets`. Each recipe
 * must emit the preset's look as one Batch of commands — exposure, shadow
 * filter and map size, IBL intensity, sun specs, and the rainier
 * exaggerations — and must never touch the camera, the DEM's identity, the
 * layers or the selection. This test applies every recipe to a starter-like
 * project and asserts the exact command payload, so a recipe that drifts
 * from its engine preset fails here, not in a review.
 *
 * Runs with zero dependencies:
 *   npm test   (node --experimental-strip-types src/state/recipes.test.ts)
 */
import { RECIPES, findRecipe, type RecipeId } from "./recipes.ts";
import type {
  Command,
  LightObject,
  Project,
  RenderSettings,
  SceneObject,
} from "../bridge/types";

const render = (patch: Partial<RenderSettings> = {}): RenderSettings =>
  ({
    width: 960,
    height: 600,
    exposure: 1.35,
    shadows: true,
    sky: true,
    iblIntensity: 1.0,
    environment: null,
    environmentBackgroundIntensity: 1.0,
    stillAov: { albedo: false, normal: false, depth: false },
    stillDenoise: false,
    shadowTechnique: "pcf",
    shadowResolution: 4096,
    sunVisibility: false,
    sunVisibilityDistance: 2000,
    sunVisibilitySteps: 32,
    skyModel: "hosek",
    skyTurbidity: 2.0,
    skyGroundAlbedo: 0.3,
    skyExposure: 0.08,
    water: { enabled: false, level: 0.0, slopeThreshold: 0.02, roughness: 0.85 },
    colormap: "",
    colormapStrength: 1.0,
    legend: false,
    plate: false,
    plateTitle: "",
    fog: { enabled: false, density: 0.0, reach: 0.6, inscatter: [1.0, 1.0, 1.0] },
    clouds: { enabled: false, shadows: true, coverage: 0.5, density: 1.0, shadowStrength: 0.85, quality: "medium" },
    bloom: { enabled: false, threshold: 0.85, softness: 0.25, intensity: 0.35 },
    vignette: 0.0,
    ...patch,
  }) as RenderSettings;

const light = (): LightObject => ({
  kind: "directional",
  color: [1.0, 0.97, 0.92],
  intensity: 3.0,
});

const starterProject = (): Project =>
  ({
    format: "forge3d.project",
    schemaVersion: 1,
    id: "p",
    name: "Recipes",
    createdUnixMs: 0,
    updatedUnixMs: 0,
    scene: {
      objects: [
        {
          id: "terrain-1",
          name: "Terrain",
          parent: null,
          transform: { translation: { x: 0, y: 0, z: 0 }, rotationDeg: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
          visible: true,
          locked: false,
          kind: { type: "terrain", asset: "dem-1", zScale: 1.0, albedo: null, dimensions: null, worldOriginXz: null, worldSpanXz: null, elevationRange: null },
        },
        {
          id: "sun-1",
          name: "Sun",
          parent: null,
          transform: { translation: { x: 4, y: 6, z: 3 }, rotationDeg: { x: -50, y: 30, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
          visible: true,
          locked: false,
          kind: { type: "light", ...light() },
        },
        {
          id: "camera-1",
          name: "Camera",
          parent: null,
          transform: { translation: { x: 6, y: 4, z: 8 }, rotationDeg: { x: -22, y: 37, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
          visible: true,
          locked: false,
          kind: { type: "camera", fovDeg: 45 },
        },
      ] as unknown as SceneObject[],
      activeCamera: "camera-1",
    },
    assets: [],
    selection: [],
    workspace: "studio",
    viewport: {} as Project["viewport"],
    render: render(),
    timeline: {} as Project["timeline"],
    cartography: [],
  }) as unknown as Project;

let failures = 0;
const fail = (where: string, problem: string) => {
  failures += 1;
  console.error(`FAIL ${where}: ${problem}`);
};
const eq = (where: string, actual: unknown, expected: unknown) => {
  if (actual !== expected) fail(where, `got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
};
const ok = (where: string, condition: boolean) => {
  if (!condition) fail(where, "condition not met");
};

interface Applied {
  render?: RenderSettings;
  sun?: LightObject;
  sunAngles?: { azimuthDeg: number; elevationDeg: number };
  terrainZScale?: number;
  commandTypes: string[];
}

const applyRecipe = (id: RecipeId): Applied => {
  const project = starterProject();
  const recipe = findRecipe(id);
  if (!recipe) throw new Error(`recipe ${id} not found`);
  const commands: Command[] = recipe.build(project, null);
  const applied: Applied = { commandTypes: commands.map((command) => command.type) };
  for (const command of commands) {
    if (command.type === "setRenderSettings") applied.render = command.render;
    if (command.type === "setLight") applied.sun = command.light;
    if (command.type === "setSunAngles") {
      applied.sunAngles = { azimuthDeg: command.azimuthDeg, elevationDeg: command.elevationDeg };
    }
    if (command.type === "setTerrainZScale") applied.terrainZScale = command.zScale;
  }
  return applied;
};

const ALLOWED = new Set(["setRenderSettings", "setLight", "setSunAngles", "setTerrainZScale"]);
for (const recipe of RECIPES) {
  const { commandTypes } = applyRecipe(recipe.id);
  for (const type of commandTypes) {
    ok(`${recipe.id}.look-only`, ALLOWED.has(type));
  }
  ok(`${recipe.id}.look-only`, commandTypes.includes("setRenderSettings"));
}

const outdoor = applyRecipe("outdoorSun");
eq("outdoorSun.exposure", outdoor.render!.exposure, 1.0);
eq("outdoorSun.technique", outdoor.render!.shadowTechnique, "pcf");
eq("outdoorSun.map", outdoor.render!.shadowResolution, 2048);
eq("outdoorSun.ibl", outdoor.render!.iblIntensity, 0.0);
eq("outdoorSun.sky", outdoor.render!.sky, true);
eq("outdoorSun.sunIntensity", outdoor.sun!.intensity, 5.0);
eq("outdoorSun.sunColor", JSON.stringify(outdoor.sun!.color), JSON.stringify([1.0, 0.97, 0.92]));
eq("outdoorSun.noSunAngles", outdoor.sunAngles, undefined);
eq("outdoorSun.noExaggeration", outdoor.terrainZScale, undefined);

const showcase = applyRecipe("rainierShowcase");
eq("rainierShowcase.exposure", showcase.render!.exposure, 1.0);
eq("rainierShowcase.technique", showcase.render!.shadowTechnique, "pcss");
eq("rainierShowcase.map", showcase.render!.shadowResolution, 4096);
eq("rainierShowcase.ibl", showcase.render!.iblIntensity, 0.3);
eq("rainierShowcase.sky", showcase.render!.sky, true);
eq("rainierShowcase.sunIntensity", showcase.sun!.intensity, 4.0);
eq("rainierShowcase.sunColor", JSON.stringify(showcase.sun!.color), JSON.stringify([1.0, 0.95, 0.9]));
eq("rainierShowcase.angles", JSON.stringify(showcase.sunAngles), JSON.stringify({ azimuthDeg: 135.0, elevationDeg: 25.0 }));
eq("rainierShowcase.zScale", showcase.terrainZScale, 1.35);

const relief = applyRecipe("rainierRelief");
eq("rainierRelief.exposure", relief.render!.exposure, 1.2);
eq("rainierRelief.technique", relief.render!.shadowTechnique, "pcss");
eq("rainierRelief.map", relief.render!.shadowResolution, 4096);
eq("rainierRelief.ibl", relief.render!.iblIntensity, 0.25);
eq("rainierRelief.sunIntensity", relief.sun!.intensity, 5.0);
eq("rainierRelief.sunColor", JSON.stringify(relief.sun!.color), JSON.stringify([1.0, 0.92, 0.85]));
eq("rainierRelief.angles", JSON.stringify(relief.sunAngles), JSON.stringify({ azimuthDeg: 225.0, elevationDeg: 18.0 }));
eq("rainierRelief.zScale", relief.terrainZScale, 1.5);

const pbr = applyRecipe("studioPbr");
eq("studioPbr.exposure", pbr.render!.exposure, 1.0);
eq("studioPbr.technique", pbr.render!.shadowTechnique, "pcf");
eq("studioPbr.map", pbr.render!.shadowResolution, 2048);
eq("studioPbr.ibl", pbr.render!.iblIntensity, 0.0);
eq("studioPbr.sky", pbr.render!.sky, false);
eq("studioPbr.sunIntensity", pbr.sun!.intensity, 6.0);
eq("studioPbr.sunColor", JSON.stringify(pbr.sun!.color), JSON.stringify([1.0, 0.98, 0.95]));
eq("studioPbr.noSunAngles", pbr.sunAngles, undefined);

const toon = applyRecipe("toonViz");
eq("toonViz.exposure", toon.render!.exposure, 1.0);
eq("toonViz.technique", toon.render!.shadowTechnique, "pcf");
eq("toonViz.map", toon.render!.shadowResolution, 1024);
eq("toonViz.ibl", toon.render!.iblIntensity, 0.0);
eq("toonViz.sky", toon.render!.sky, false);
eq("toonViz.sunIntensity", toon.sun!.intensity, 4.0);
eq("toonViz.sunColor", JSON.stringify(toon.sun!.color), JSON.stringify([1.0, 1.0, 1.0]));

// The render command must carry the whole settings object (the shell
// replaces wholesale) with only the recipe's keys changed.
const whole = applyRecipe("outdoorSun");
eq("whole.unpatchedWidth", whole.render!.width, 960);
eq("whole.unpatchedColormap", whole.render!.colormap, "");
eq("whole.unpatchedResolution", whole.render!.shadowResolution, 2048);

if (failures > 0) {
  throw new Error(`${failures} recipe assertion(s) failed`);
}
console.log("recipes: all assertions passed");
