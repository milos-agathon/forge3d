/**
 * The lighting rig both renderers share.
 *
 * Solid and Material draw with three.js, Rendered draws with Forge3D; the only
 * way the three stay the same picture is that they are handed the *same*
 * numbers. This module derives those numbers from the project — one pure
 * function — so the viewport and the node test that pins it against the
 * renderer's payload read the same source.
 *
 * The invariants it encodes (each was a real bug once):
 *  - the sun is the scene's directional light, with its own colour and
 *    intensity — no fudge factor, no sRGB conversion (`u_shading.light_params`
 *    is `colour * intensity` in linear radiance);
 *  - the exposure is the project's, the same one the engine tone-maps with;
 *  - the environment is the project's HDRI path, or nothing for the built-in sky;
 *  - shadows are on unless the project turns them off, with the project's
 *    filter and map size.
 */
import * as THREE from "three";

import type { Project, ShadowTechnique } from "../bridge/types";

const DEG = Math.PI / 180;

export interface LightingSun {
  /** Unit direction the light travels, viewport frame (Y up, north at -Z). */
  direction: [number, number, number];
  /** Linear RGB radiance multiplier, exactly the project's value. */
  intensity: number;
  /** Linear RGB colour, exactly the project's value. */
  color: [number, number, number];
}

export interface LightingShadowRig {
  enabled: boolean;
  technique: ShadowTechnique;
  resolution: number;
}

export interface LightingRig {
  /** The visible directional light, or null when the scene has none. */
  sun: LightingSun | null;
  exposure: number;
  /** HDRI asset path, or null for the built-in studio sky. */
  environment: string | null;
  shadows: LightingShadowRig;
}

export function lightingRig(project: Project): LightingRig {
  const environment =
    project.render.environment !== null
      ? project.assets.find((asset) => asset.id === project.render.environment)?.path ?? null
      : null;

  const light = project.scene.objects.find(
    (object) =>
      object.kind.type === "light" && object.kind.kind === "directional" && object.visible,
  );

  let sun: LightingSun | null = null;
  if (light && light.kind.type === "light") {
    // The light's local -Z is where it travels. three's Euler is the same
    // convention the viewport used inline (Rz·Ry·Rx), so the fixture test
    // compares one implementation against the engine, not two against each
    // other.
    const rotation = light.transform.rotationDeg;
    const direction = new THREE.Vector3(0, 0, -1).applyEuler(
      new THREE.Euler(rotation.x * DEG, rotation.y * DEG, rotation.z * DEG, "ZYX"),
    );
    sun = {
      direction: [direction.x, direction.y, direction.z],
      intensity: Math.max(0, light.kind.intensity),
      color: [...light.kind.color] as [number, number, number],
    };
  }

  return {
    sun,
    exposure: project.render.exposure,
    environment,
    shadows: {
      enabled: project.render.shadows,
      technique: project.render.shadowTechnique,
      resolution: project.render.shadowResolution,
    },
  };
}
