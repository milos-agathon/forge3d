/**
 * Look recipes: the named lighting packages from the engine's `forge3d.presets`.
 *
 * Each recipe translates one preset the engine ships into a Studio project in
 * one undoable step (`Batch`). Recipes touch the look only — camera, DEM,
 * layers and selection are untouched.
 *
 * Not everything the engine preset carries is expressible in Studio: where the
 * translation is impossible (a toon BRDF) or deliberately drifts (an unportable
 * sun direction vector), the recipe leaves the value alone or applies the
 * closest honest setting, and the tooltip says so.
 */
import type {
  Command,
  LightObject,
  Project,
  RenderSettings,
} from "../bridge/types";

export type RecipeId =
  | "outdoorSun"
  | "rainierShowcase"
  | "rainierRelief"
  | "studioPbr"
  | "toonViz";

export interface Recipe {
  id: RecipeId;
  name: string;
  /** The engine's own name for this look, for the tooltip. */
  enginePreset: string;
  /** One line on what the recipe sets and what it doesn't. */
  note: string;
  build: (project: Project, sun: SceneObjectRef | null) => Command[];
}

/** A minimal stand-in for the directional sun. */
export interface SceneObjectRef {
  id: string;
  light: LightObject;
}

const findSun = (project: Project): SceneObjectRef | null => {
  for (const object of project.scene.objects) {
    if (object.kind.type !== "light") continue;
    if (object.kind.kind !== "directional") continue;
    return { id: object.id, light: object.kind };
  }
  return null;
};

const findTerrain = (project: Project) =>
  project.scene.objects.find((object) => object.kind.type === "terrain") ?? null;

const setRender = (project: Project, patch: Partial<RenderSettings>): Command => ({
  type: "setRenderSettings",
  render: { ...project.render, ...patch },
});

const setSun = (
  sun: SceneObjectRef,
  patch: Partial<LightObject>,
): Command => ({
  type: "setLight",
  id: sun.id,
  light: { ...sun.light, ...patch },
});

const setSunAngles = (
  sun: SceneObjectRef,
  azimuthDeg: number,
  elevationDeg: number,
): Command => ({
  type: "setSunAngles",
  id: sun.id,
  azimuthDeg,
  elevationDeg,
});

const withSun = (project: Project): SceneObjectRef | null => findSun(project);

export const RECIPES: Recipe[] = [
  {
    id: "outdoorSun",
    name: "Outdoor sun",
    enginePreset: "outdoor_sun",
    note:
      "Dish sun, banked-style exposure 1.0, 2×2048 PCF shadows, Hosek-Wilkie sky. " +
      "IBL off (the preset's GI is empty). The preset's sun direction vector is " +
      "engine-frame and is not ported — aim your own sun, or leave it where it is.",
    build: (project) => {
      const sun = withSun(project);
      const render = setRender(project, {
        exposure: 1.0,
        shadows: true,
        shadowTechnique: "pcf",
        shadowResolution: 2048,
        iblIntensity: 0.0,
        sky: true,
        skyModel: "hosek",
      });
      return [
        render,
        ...(sun ? [setSun(sun, { intensity: 5.0, color: [1.0, 0.97, 0.92] })] : []),
      ];
    },
  },
  {
    id: "rainierShowcase",
    name: "Rainier showcase",
    enginePreset: "rainier_showcase",
    note:
      "Cross-light from 135° azimuth at 25° elevation, PCSS 4k shadows, IBL 0.3 so " +
      "shadows dominate, relief ×1.35. Camera moves omitted — recipe sets the look, " +
      "not the framing. Cascades/light-size/ambient occlusion are not ported " +
      "(no terrain-pipe or Rendered-AO control); the preset's builtin clear_sky env maps to Studio's built-in sky at the same 0.3.",
    build: (project) => {
      const terrain = findTerrain(project);
      const sun = withSun(project);
      const terrainCommand = terrain
        ? [
            {
              type: "setTerrainZScale",
              id: terrain.id,
              zScale: 1.35,
            } as Command,
          ]
        : [];
      return [
        setRender(project, {
          exposure: 1.0,
          shadows: true,
          shadowTechnique: "pcss",
          shadowResolution: 4096,
          iblIntensity: 0.3,
          sky: true,
          skyModel: "hosek",
        }),
        ...(sun
          ? [
              setSun(sun, { intensity: 4.0, color: [1.0, 0.95, 0.9] }),
              setSunAngles(sun, 135.0, 25.0),
            ]
          : []),
        ...terrainCommand,
      ];
    },
  },
  {
    id: "rainierRelief",
    name: "Rainier relief",
    enginePreset: "rainier_relief",
    note:
      "Low 18° sun from 225°, PCSS 4k, IBL 0.25 so relief dominates, relief ×1.5. " +
      "Camera offspring of the upstream preset (mesh view, 65° theta/45° phi) is omitted — " +
      "recipe sets the look, not the framing. Cascades/light-size/ambient-occlusion not ported; " +
      "the preset's builtin clear_sky env maps to Studio's built-in sky at the same 0.25.",
    build: (project) => {
      const terrain = findTerrain(project);
      const sun = withSun(project);
      const terrainCommand = terrain
        ? [
            {
              type: "setTerrainZScale",
              id: terrain.id,
              zScale: 1.5,
            } as Command,
          ]
        : [];
      return [
        setRender(project, {
          exposure: 1.2,
          shadows: true,
          shadowTechnique: "pcss",
          shadowResolution: 4096,
          iblIntensity: 0.25,
          sky: true,
          skyModel: "hosek",
        }),
        ...(sun
          ? [
              setSun(sun, { intensity: 5.0, color: [1.0, 0.92, 0.85] }),
              setSunAngles(sun, 225.0, 18.0),
            ]
          : []),
        ...terrainCommand,
      ];
    },
  },
  {
    id: "studioPbr",
    name: "Studio PBR",
    enginePreset: "studio_pbr",
    note:
      "Stage-key look: exposure 1.0, 2×2048 PCF shadows, IBL off, flat backdrop " +
      "(the preset disables atmosphere — the sky toggle maps to Studio's flat off). " +
      "The preset's sun direction vector is engine-frame and is not ported.",
    build: (project) => {
      const sun = withSun(project);
      return [
        setRender(project, {
          exposure: 1.0,
          shadows: true,
          shadowTechnique: "pcf",
          shadowResolution: 2048,
          iblIntensity: 0.0,
          sky: false,
        }),
        ...(sun ? [setSun(sun, { intensity: 6.0, color: [1.0, 0.98, 0.95] })] : []),
      ];
    },
  },
  {
    id: "toonViz",
    name: "Toon viz",
    enginePreset: "toon_viz",
    note:
      "Exposure 1.0, 1×1024 PCF (the preset's hard shadows — Studio's terrain has no hard PCF " +
      "or toon BRDF, so closest-honest PCF at the preset's size), IBL off, sky off. " +
      "The preset's sun direction vector is engine-frame and is not ported.",
    build: (project) => {
      const sun = withSun(project);
      return [
        setRender(project, {
          exposure: 1.0,
          shadows: true,
          shadowTechnique: "pcf",
          shadowResolution: 1024,
          iblIntensity: 0.0,
          sky: false,
        }),
        ...(sun ? [setSun(sun, { intensity: 4.0, color: [1.0, 1.0, 1.0] })] : []),
      ];
    },
  },
];

export function findRecipe(id: RecipeId): Recipe | undefined {
  return RECIPES.find((recipe) => recipe.id === id);
}
