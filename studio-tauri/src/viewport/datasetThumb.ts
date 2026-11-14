/**
 * Dataset thumbnails for the New Project gallery.
 *
 * The heightfield from the shell's own `terrainPreview` IPC is mapped to a
 * warm ramp — no shading, no second implementation of relief; the gallery
 * just says what the DEM looks like.
 */

export interface ThumbHeights {
  resolution: number;
  heights: number[];
  minElevation: number;
  maxElevation: number;
}

export interface Ramp {
  from: [number, number, number];
  to: [number, number, number];
}

const LOW: [number, number, number] = [30, 28, 22];
const HIGH: [number, number, number] = [236, 230, 214];

/** Row-major heights (row 0 = north) -> RGBA warm ramp, row 0 first. */
export function thumbPixels(preview: ThumbHeights): Uint8ClampedArray<ArrayBuffer> {
  const res = preview.resolution;
  const range = Math.max(1e-6, preview.maxElevation - preview.minElevation);
  const out = new Uint8ClampedArray(new ArrayBuffer(res * res * 4));
  for (let index = 0; index < res * res; index += 1) {
    const t = Math.min(1, Math.max(0, (preview.heights[index] - preview.minElevation) / range));
    const r = Math.round(LOW[0] + (HIGH[0] - LOW[0]) * t);
    const g = Math.round(LOW[1] + (HIGH[1] - LOW[1]) * t);
    const b = Math.round(LOW[2] + (HIGH[2] - LOW[2]) * t);
    const o = index * 4;
    out[o] = r;
    out[o + 1] = g;
    out[o + 2] = b;
    out[o + 3] = 255;
  }
  return out;
}
