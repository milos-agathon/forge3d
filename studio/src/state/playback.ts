/**
 * The timeline playback clock.
 *
 * Playback is a fixed-rate loop over `setPlayhead` IPC calls, so its two
 * pieces of arithmetic live here, away from React: the loop must NOT restart
 * when the project state changes (each playhead update produces a new
 * `timeline` object, and an effect keyed on it reset the clock on every frame —
 * playback advanced at most one frame per IPC round-trip).
 *
 * The loop advances at most one frame per animation frame: when the shell is
 * slower than the timeline's frame rate, playback runs slow instead of piling
 * IPC calls up in a queue. Backlog is capped at one step for the same reason.
 */

/** Frames per second → milliseconds per frame. Invalid rates fall back to 1. */
export function frameStepMs(frameRate: number): number {
  const rate = Number.isFinite(frameRate) && frameRate > 0 ? frameRate : 1;
  return 1000 / rate;
}

/**
 * The frame after `current`, wrapping at the end of the timeline (Blender's
 * looping playback). Non-finite input starts from frame 0.
 */
export function nextPlayheadFrame(current: number, length: number): number {
  if (!(length > 0)) return 0;
  const frame = Math.round(Number.isFinite(current) ? current : 0);
  if (frame >= length) return 0;
  return Math.max(0, frame) + 1;
}

/**
 * Add elapsed time to the accumulator and report whether a frame is due.
 *
 * `rest` is the leftover time, capped at one step: a backgrounded window that
 * comes back after ten seconds must not emit a burst of frames at the shell.
 */
export function tickClock(
  accumulatorMs: number,
  elapsedMs: number,
  stepMs: number,
): { due: boolean; rest: number } {
  if (!(stepMs > 0)) return { due: false, rest: 0 };
  const accumulated = Math.max(0, accumulatorMs) + Math.max(0, elapsedMs);
  if (accumulated < stepMs) return { due: false, rest: accumulated };
  return { due: true, rest: Math.min(accumulated - stepMs, stepMs) };
}
