/**
 * Offline render queue — the pure half.
 *
 * The viewport scheduler is "newest wins" (a preview is disposable). Offline
 * work is the opposite: a still or a frame sequence is a job that must run to
 * completion, in order, with visible progress and a cancel that stops between
 * frames. This module owns the job state machine; `session.tsx` owns the IPC.
 *
 * Keeping the rules here means the queue can be tested without a shell: order,
 * the single-running invariant, progress, and what "cancel" means for a job
 * that is already inside a frame.
 */

export type RenderJobKind = "still" | "sequence";

export type RenderJobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface RenderJob {
  id: string;
  kind: RenderJobKind;
  label: string;
  /** Frames to render; 1 for a still. */
  total: number;
  /** Frames finished so far. */
  done: number;
  status: RenderJobStatus;
  /** File for a still, directory for a sequence. */
  output: string;
  error: string | null;
}

export interface RenderJobInput {
  kind: RenderJobKind;
  label: string;
  total: number;
  output: string;
}

/** A queued job, with a stable id and a whole number of frames. */
export function newRenderJob(input: RenderJobInput): RenderJob {
  const total =
    Number.isFinite(input.total) && input.total > 0 ? Math.floor(input.total) : 1;
  return {
    id: crypto.randomUUID(),
    kind: input.kind,
    label: input.label,
    total,
    done: 0,
    status: "queued",
    output: input.output,
    error: null,
  };
}

/** Append a job. The queue is FIFO: the order the user started things in. */
export function enqueueJob(jobs: RenderJob[], job: RenderJob): RenderJob[] {
  return [...jobs, job];
}

/** Replace one job in place, preserving order. Unknown ids are a no-op. */
export function patchJob(
  jobs: RenderJob[],
  id: string,
  patch: Partial<Omit<RenderJob, "id">>,
): RenderJob[] {
  return jobs.map((job) => (job.id === id ? { ...job, ...patch } : job));
}

/** The next job the worker should run: the first queued one, or null. */
export function nextQueuedJob(jobs: RenderJob[]): RenderJob | null {
  return jobs.find((job) => job.status === "queued") ?? null;
}

/** Fraction complete, 0..1. */
export function jobProgress(job: RenderJob): number {
  if (job.total <= 0) return 0;
  return Math.min(1, Math.max(0, job.done / job.total));
}

/**
 * Whether Cancel can stop this job now.
 *
 * A queued job can always be dropped. A running sequence stops between frames,
 * so it can be cancelled. A running still is a single frame already inside the
 * renderer — cancelling would mean killing the renderer session, which would
 * also take down any viewport frame in flight, so the button is disabled and
 * the completion marks it done.
 */
export function canCancelJob(job: RenderJob): boolean {
  if (job.status === "queued") return true;
  return job.status === "running" && job.kind === "sequence";
}

/** Counts for the queue header. */
export function queueSummary(jobs: RenderJob[]): {
  queued: number;
  running: number;
  failed: number;
  finished: number;
} {
  let queued = 0;
  let running = 0;
  let failed = 0;
  let finished = 0;
  for (const job of jobs) {
    if (job.status === "queued") queued += 1;
    else if (job.status === "running") running += 1;
    else if (job.status === "failed") failed += 1;
    else finished += 1;
  }
  return { queued, running, failed, finished };
}

/** Drop finished/failed/cancelled entries, keeping anything still pending. */
export function clearFinishedJobs(jobs: RenderJob[]): RenderJob[] {
  return jobs.filter((job) => job.status === "queued" || job.status === "running");
}
