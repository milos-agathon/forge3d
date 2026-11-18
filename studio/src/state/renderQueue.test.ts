/**
 * Regression tests for the offline render queue state machine.
 *
 * Runs with zero dependencies:
 *   npm test   (node --experimental-strip-types)
 *
 * The shell owns the IPC; these tests pin what the queue promises: FIFO order,
 * one running job, honest progress, and a cancel rule that cannot abandon a
 * still already inside the renderer.
 */
import {
  canCancelJob,
  clearFinishedJobs,
  enqueueJob,
  jobProgress,
  newRenderJob,
  nextQueuedJob,
  patchJob,
  queueSummary,
} from "./renderQueue.ts";

let failures = 0;

function check(name: string, condition: boolean, detail?: string): void {
  if (condition) return;
  failures += 1;
  console.error(`FAIL ${name}${detail ? `: ${detail}` : ""}`);
}

function equal(name: string, actual: unknown, expected: unknown): void {
  check(name, actual === expected, `got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
}

function near(name: string, actual: number, expected: number): void {
  check(name, Math.abs(actual - expected) < 1e-9, `got ${actual}, want ${expected}`);
}

// --- construction ---------------------------------------------------
const still = newRenderJob({ kind: "still", label: "Still", total: 1, output: "/r/a.png" });
equal("a still starts queued", still.status, "queued");
equal("a still has one frame", still.total, 1);
equal("a still starts at zero", still.done, 0);

const sequence = newRenderJob({
  kind: "sequence",
  label: "Sequence",
  total: 97,
  output: "/r/frames",
});
equal("a sequence keeps its frame count", sequence.total, 97);

const odd = newRenderJob({ kind: "sequence", label: "Odd", total: 0, output: "" });
equal("a zero total clamps to one frame", odd.total, 1);
const fractional = newRenderJob({ kind: "sequence", label: "Frac", total: 4.8, output: "" });
equal("a fractional total floors", fractional.total, 4);

// --- ordering -------------------------------------------------------
const first = enqueueJob([], still);
const both = enqueueJob(first, sequence);
equal("FIFO: first in, first out", both[0].id, still.id);
equal("FIFO: the second job follows", both[1].id, sequence.id);
equal("enqueue does not mutate its input", first.length, 1);

equal("the next job is the first queued one", nextQueuedJob(both)?.id, still.id);
const running = patchJob(both, still.id, { status: "running" });
equal("a running job is not picked again", nextQueuedJob(running)?.id, sequence.id);
const finished = patchJob(running, sequence.id, { status: "done" });
equal("an empty queue has no next job", nextQueuedJob(finished), null);

// --- patching -------------------------------------------------------
const patched = patchJob(both, still.id, { done: 1, status: "done" });
equal("patching preserves other jobs", patched[1].id, sequence.id);
equal("patching preserves order", patched[0].id, still.id);
equal("patching an unknown id is a no-op", patchJob(both, "nope", { done: 9 }).length, 2);
equal("the original is not mutated", both[0].done, 0);

// --- progress -------------------------------------------------------
near("progress starts at zero", jobProgress(sequence), 0);
near("progress follows done", jobProgress(patchJob([sequence], sequence.id, { done: 24 })[0]), 24 / 97);
near("progress completes", jobProgress(patchJob([sequence], sequence.id, { done: 97 })[0]), 1);
near("progress never exceeds one", jobProgress(patchJob([sequence], sequence.id, { done: 500 })[0]), 1);

// --- cancellation ---------------------------------------------------
check("a queued job can be cancelled", canCancelJob(still));
check("a running sequence can be cancelled", canCancelJob({ ...sequence, status: "running" }));
check(
  "a running still cannot be cancelled mid-frame",
  !canCancelJob({ ...still, status: "running" }),
);
check("a done job has nothing to cancel", !canCancelJob({ ...sequence, status: "done" }));
check("a failed job has nothing to cancel", !canCancelJob({ ...sequence, status: "failed" }));

// --- summary and clearing -------------------------------------------
const mixed = [
  still,
  { ...sequence, status: "running" as const, done: 12 },
  { ...newRenderJob({ kind: "sequence", label: "queued", total: 2, output: "" }) },
  { ...newRenderJob({ kind: "sequence", label: "failed", total: 2, output: "" }), status: "failed" as const },
  { ...newRenderJob({ kind: "still", label: "done", total: 1, output: "" }), status: "done" as const },
];
const summary = queueSummary(mixed);
equal("summary: queued", summary.queued, 2);
equal("summary: running", summary.running, 1);
equal("summary: failed", summary.failed, 1);
equal("summary: finished", summary.finished, 1);

const cleared = clearFinishedJobs(mixed);
equal("clearing keeps pending work", cleared.length, 3);
check(
  "clearing drops finished work",
  cleared.every((job) => job.status === "queued" || job.status === "running"),
);

if (failures > 0) {
  throw new Error(`${failures} renderQueue assertion(s) failed`);
}
console.log("renderQueue: all assertions passed");
