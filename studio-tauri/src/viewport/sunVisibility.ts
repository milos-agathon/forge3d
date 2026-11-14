/**
 * Sun visibility: a baked grid of terrain self-shadowing, for the preview.
 *
 * The still ray-marches every fragment toward the sun (`sun_visibility`
 * distance and steps). The preview cannot match it fragment-wise cheaply,
 * so it bakes the same march once per (DEM, zScale, sun) change at grid
 * resolution: for every grid cell, cast toward the sun and test whether
 * terrain stands in the way. Baked once, sampled linearly by the terrain
 * shader's irradiance term. Pure math, no three.js — testable in node.
 */

export interface SunVisibilityInput {
  /** Row-major heights; row 0 is the north edge (the raster's top). */
  heights: readonly number[];
  resolution: number;
  minElevation: number;
  spanXz: [number, number];
  zScale: number;
  /** Unit direction TOWARD the sun (opposite of the light's travel direction). */
  toward: [number, number, number];
  /** Ray marches to the edge of the grid, in this many steps. */
  steps?: number;
}

/** Bilinear height at fractional cell coordinates (x = west→east, y = north→south). */
function sampleHeight(heights: readonly number[], res: number, x: number, y: number): number {
  const gx = Math.min(Math.max(x, 0), res - 1);
  const gy = Math.min(Math.max(y, 0), res - 1);
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, res - 1);
  const y1 = Math.min(y0 + 1, res - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const at = (cx: number, cy: number): number => {
    const value = heights[cy * res + cx];
    return Number.isFinite(value) ? value : 0;
  };
  const row0 = at(x0, y0) * (1 - tx) + at(x1, y0) * tx;
  const row1 = at(x0, y1) * (1 - tx) + at(x1, y1) * tx;
  return row0 * (1 - ty) + row1 * ty;
}

/**
 * Bake the visibility grid. `out[cy * res + cx]` is 255 where the sun sees
 * the point, 0 where terrain stands in the way (the same rule as the still's
 * ray march, read at texel centers).
 */
export function bakeSunVisibility(input: SunVisibilityInput): Uint8Array {
  const res = input.resolution;
  const steps = Math.max(8, Math.min(64, input.steps ?? 64));
  const [tx, ty, tz] = input.toward;
  const horizontal = Math.hypot(tx, tz);
  const out = new Uint8Array(res * res);
  if (horizontal < 1e-6) {
    // Zenith sun: nothing can occlude anything.
    out.fill(255);
    return out;
  }
  const towardUnit: [number, number, number] = [tx / horizontal, ty / horizontal, tz / horizontal];
  // Metres per cell in world units; the ray marches in metres like the still.
  const stepMeters = Math.max(input.spanXz[0], input.spanXz[1]) / steps;
  const elevationRange = Math.max(1e-6, input.heights.reduce((max, v) => Math.max(max, v), -Infinity)
    - input.heights.reduce((min, v) => Math.min(min, v), Infinity));
  const heightScale = input.zScale;
  // The bias the still uses implicitly per texel; one texel of height.
  const bias = Math.max(0.5, elevationRange * input.zScale * 0.0002);

  for (let v = 0; v < res; v += 1) {
    const y = (v / (res - 1)) * input.spanXz[1];
    for (let u = 0; u < res; u += 1) {
      const x = (u / (res - 1)) * input.spanXz[0];
      const base = sampleHeight(input.heights, res, u, v);
      const baseHeight = (base - input.minElevation) * heightScale;
      let visible = 1;
      for (let step = 1; step <= steps; step += 1) {
        const distance = step * stepMeters;
        const sampleX = x + towardUnit[0] * distance;
        const sampleY = y + towardUnit[2] * distance;
        const gridX = (sampleX / input.spanXz[0]) * (res - 1);
        const gridY = (sampleY / input.spanXz[1]) * (res - 1);
        if (gridX < 0 || gridY < 0 || gridX > res - 1 || gridY > res - 1) break;
        // Upright-ray height above the start point, rising at tan(sun elev).
        const rayY = baseHeight + towardUnit[1] * distance;
        const terrain = (sampleHeight(input.heights, res, gridX, gridY) - input.minElevation) * heightScale;
        if (terrain > rayY + bias) {
          visible = 0;
          break;
        }
      }
      out[v * res + u] = Math.round(visible * 255);
    }
  }
  return out;
}

/** Visibility value for a cell; 0 (occluded) through 255 (lit). */
export function visibilityBytesToRgba(visibility: Uint8Array): Uint8Array {
  const out = new Uint8Array(visibility.length * 4);
  for (let index = 0; index < visibility.length; index += 1) {
    const value = visibility[index];
    out[index * 4] = value;
    out[index * 4 + 1] = value;
    out[index * 4 + 2] = value;
    out[index * 4 + 3] = 255;
  }
  return out;
}
