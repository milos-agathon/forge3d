/**
 * The thumbnail ramp is a presentation mapping: pin the invariants only.
 */
import { thumbPixels } from "./datasetThumb.ts";

let failures = 0;
function ok(name: string, condition: boolean): void {
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}`);
  }
}

const preview = {
  resolution: 2,
  heights: [100, 400, 400, 100],
  minElevation: 100,
  maxElevation: 400,
};

const out = thumbPixels(preview);
ok("RGBA8 output, full frame", out.length === 2 * 2 * 4);
// every pixel in the 2x2 is opaque (alpha is the 4th byte of each pixel)
ok("floor pixels are opaque", [0, 1, 2, 3].every((i) => out[i * 4 + 3] === 255));
// corners low (dark), centre high (bright)
const lowLuma = out[0] + out[1] + out[2];
const highLuma = out[5] + out[6] + out[7];
ok("high ground reads brighter than low", highLuma > lowLuma);

if (failures > 0) {
  throw new Error(`datasetThumb: ${failures} assertion(s) failed`);
}
console.log("datasetThumb: all assertions passed");
