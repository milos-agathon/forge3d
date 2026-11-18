/**
 * Regression tests for the timeline playback clock.
 *
 * Runs with zero dependencies:
 *   npm test   (node --experimental-strip-types)
 *
 * The bug this pins: the playback effect depended on the project state, so
 * every `setPlayhead` snapshot restarted it and reset `last`/`accumulator`.
 * A frame is 41.7 ms at 24 fps and the IPC round-trip is shorter than that, so
 * the accumulator never reached a step and playback froze after one frame.
 */
import { frameStepMs, nextPlayheadFrame, tickClock } from "./playback.ts";

let failures = 0;

function check(name: string, condition: boolean, detail?: string): void {
  if (condition) return;
  failures += 1;
  console.error(`FAIL ${name}${detail ? `: ${detail}` : ""}`);
}

function equal(name: string, actual: number, expected: number): void {
  check(name, actual === expected, `got ${actual}, want ${expected}`);
}

// --- frame stepping -------------------------------------------------
equal("first frame", nextPlayheadFrame(0, 96), 1);
equal("last frame stays until the end", nextPlayheadFrame(95, 96), 96);
equal("end wraps to the start", nextPlayheadFrame(96, 96), 0);
equal("a past-the-end frame wraps", nextPlayheadFrame(240, 96), 0);
equal("fractional playheads round", nextPlayheadFrame(4.6, 96), 6);
equal("an invalid playhead starts at 1", nextPlayheadFrame(Number.NaN, 96), 1);
equal("a zero-length timeline cannot advance", nextPlayheadFrame(3, 0), 0);

// --- clock ----------------------------------------------------------
const step = frameStepMs(24);
check("24 fps frame step", Math.abs(step - 1000 / 24) < 1e-9, String(step));
equal("invalid frame rate falls back to 1 fps", frameStepMs(0), 1000);

const idle = tickClock(0, 16, step);
check("below one step is not due", !idle.due);
check("time accumulates while below a step", idle.rest === 16, String(idle.rest));

const exact = tickClock(step, 0, step);
check("exactly one step is due", exact.due);
equal("an exact step leaves no backlog", exact.rest, 0);

// A stalled window must not emit a burst: one frame per tick, one step kept.
const burst = tickClock(10_000, 0, step);
check("a long stall is still one frame", burst.due);
check("backlog is capped at one step", burst.rest <= step + 1e-9, String(burst.rest));

// --- the regression: a two-second playback run at 60 Hz --------------
// 120 ticks × 16.67 ms = 2 s; at 24 fps that is 48 frames. The old loop
// restarted on every state update, so it advanced at most a handful.
let frames = 0;
let rest = 0;
let playhead = 0;
for (let tick = 0; tick < 120; tick += 1) {
  const clock = tickClock(rest, 1000 / 60, step);
  rest = clock.rest;
  if (!clock.due) continue;
  playhead = nextPlayheadFrame(playhead, 96);
  frames += 1;
}
check(
  "two seconds at 24 fps advances ~48 frames",
  frames >= 47 && frames <= 48,
  `${frames} frames`,
);
equal("playhead follows the frame count", playhead, frames);

// A rate change mid-run takes effect on the next tick, without a restart.
const slowStep = frameStepMs(12);
const afterChange = tickClock(step * 0.75, 0, slowStep);
check("a slower rate keeps a partial accumulator", !afterChange.due);

if (failures > 0) {
  throw new Error(`${failures} playback assertion(s) failed`);
}
console.log("playback: all assertions passed");
