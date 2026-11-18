import { useSession } from "../state/session-context";
import {
  canCancelJob,
  jobProgress,
  queueSummary,
  type RenderJob,
  type RenderJobStatus,
} from "../state/renderQueue";

const STATUS_LABEL: Record<RenderJobStatus, string> = {
  queued: "queued",
  running: "rendering",
  done: "done",
  failed: "failed",
  cancelled: "cancelled",
};

function headline(job: RenderJob): string {
  if (job.status === "running" && job.kind === "sequence") {
    return `${job.done} / ${job.total} frames`;
  }
  if (job.status === "done" && job.kind === "sequence") {
    return `${job.total} frames written`;
  }
  if (job.status === "done") return "written";
  if (job.status === "queued") return `${job.total} frame${job.total === 1 ? "" : "s"}`;
  return "";
}

/**
 * Offline render queue: stills and frame sequences run FIFO through one worker
 * so the engine never has two jobs fighting over it. The viewport's own preview
 * renders are not here — those are newest-wins and never queue.
 */
export function RenderQueue() {
  const { renderJobs, actions } = useSession();
  if (renderJobs.length === 0) return null;
  const summary = queueSummary(renderJobs);
  const busy = summary.running > 0 || summary.queued > 0;

  return (
    <section className="render-queue" aria-label="Render queue">
      <header className="render-queue__header">
        <span>Render queue</span>
        <span className="render-queue__meta">
          {summary.running > 0
            ? "rendering"
            : busy
              ? `${summary.queued} queued`
              : "idle"}
        </span>
        {!busy && (
          <button
            type="button"
            className="text-button"
            onClick={actions.clearFinishedJobs}
          >
            Clear
          </button>
        )}
      </header>
      <ul className="render-queue__list">
        {renderJobs.map((job) => (
          <li key={job.id} className={`render-job render-job--${job.status}`}>
            <div className="render-job__row">
              <span className="render-job__label" title={job.output || undefined}>
                {job.label}
              </span>
              <span className="render-job__status">{STATUS_LABEL[job.status]}</span>
              <span className="render-job__count mono">{headline(job)}</span>
              {canCancelJob(job) && (
                <button
                  type="button"
                  className="text-button"
                  onClick={() => actions.cancelRenderJob(job.id)}
                >
                  Cancel
                </button>
              )}
            </div>
            {job.kind === "sequence" && (
              <div
                className="render-job__bar"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={job.total}
                aria-valuenow={job.done}
              >
                <span style={{ width: `${Math.round(jobProgress(job) * 100)}%` }} />
              </div>
            )}
            {job.error && <p className="render-job__error">{job.error}</p>}
          </li>
        ))}
      </ul>
    </section>
  );
}
