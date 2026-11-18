import { useSession } from "../state/session-context";
import { FileIcon, FolderIcon } from "./icons";

export function EmptyState() {
  const { actions, info } = useSession();

  return (
    <div className="empty">
      <div className="empty__card">
        <span className="empty__mark" aria-hidden="true">
          F3
        </span>
        <h1>Forge3D Studio</h1>
        <p className="empty__lede">
          One project, two workspaces. Author the scene here, render it with
          Forge3D, and switch between them without losing work.
        </p>

        <div className="empty__actions">
          <button
            type="button"
            className="button button--primary"
            onClick={() => actions.requestNewProject()}
          >
            <FileIcon size={14} /> New project
          </button>
          <button
            type="button"
            className="button"
            onClick={() => void actions.openProject()}
          >
            <FolderIcon size={14} /> Open project…
          </button>
        </div>

        <dl className="empty__facts">
          <div>
            <dt>Shell</dt>
            <dd>{info?.version ?? "—"}</dd>
          </div>
          <div>
            <dt>Bridge</dt>
            <dd>Tauri IPC</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
