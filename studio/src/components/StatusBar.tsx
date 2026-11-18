import { useSession } from "../state/session-context";

function time(unixMs: number | null | undefined): string {
  if (!unixMs) return "—";
  return new Date(unixMs).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function StatusBar() {
  const { state, autosave, recovery, busy, engine } = useSession();

  const objects = state?.project.scene.objects.length ?? 0;
  const selected = state?.project.selection.length ?? 0;

  return (
    <footer className="statusbar">
      <span className="status-chip status-chip--tauri">Tauri shell</span>
      <span
        className={`status-engine ${engine?.available ? "is-ready" : ""}`}
        title={[
          engine?.detail,
          engine?.backend && engine.adapter
            ? `${engine.backend} · ${engine.adapter}`
            : null,
          engine?.software
            ? "software fallback: the engine did not get a GPU adapter"
            : null,
        ]
          .filter(Boolean)
          .join("\n") ?? "Probing for the Forge3D engine…"}
      >
        {engine
          ? engine.available
            ? `forge3d ${engine.packageVersion ?? "?"}${
                engine.backend ? ` · ${engine.backend}` : ""
              }${
                engine.adapter
                  ? ` · ${engine.adapter.replace(/^NVIDIA GeForce\s*/, "").slice(0, 44)}`
                  : ""
              }${engine.software ? " · software" : ""}`
            : "engine unavailable"
          : "probing engine…"}
      </span>
      {state ? (
        <>
          <span>{objects} objects</span>
          <span>{selected} selected</span>
          <span className="mono">rev {state.revision}</span>
          <span>{state.historyLen} undo steps</span>
          <span className={state.dirty ? "status-dirty" : "status-clean"}>
            {state.dirty ? "unsaved" : "saved"}
          </span>
        </>
      ) : (
        <span>No project</span>
      )}

      <span className="statusbar__spacer" />

      {busy && <span className="status-busy">working…</span>}
      {recovery.length > 0 && (
        <span className="status-warn">{recovery.length} recovery snapshot(s)</span>
      )}
      {autosave?.lastError ? (
        <span className="status-error" title={autosave.lastError}>
          autosave failed
        </span>
      ) : (
        <span className="status-muted">
          autosave {autosave ? `every ${autosave.intervalSeconds}s` : "off"}
          {autosave?.lastWriteUnixMs
            ? ` · last ${time(autosave.lastWriteUnixMs)}`
            : " · not written yet"}
        </span>
      )}
    </footer>
  );
}
