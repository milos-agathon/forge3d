import { useSession } from "../state/session-context";
import { CloseIcon } from "./icons";
import { DatasetGallery } from "./DatasetGallery";
import { RemoteDemDialog } from "./RemoteDemDialog";

function metres(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value).toLocaleString()} m`;
}

/**
 * Starter-scene picker. Creating a project is a decision, not a keystroke, and
 * the two bundled DEMs are very different subjects (a 4 km amphitheatre vs a
 * 353 km country), so it earns a dialog.
 */
export function NewProjectDialog() {
  const { newProjectOpen, presets, actions, busy } = useSession();
  if (!newProjectOpen) return null;

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="New project"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) actions.dismissNewProject();
      }}
    >
      <div className="modal">
        <header className="modal__header">
          <h2>New project</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Cancel"
            onClick={actions.dismissNewProject}
          >
            <CloseIcon size={14} />
          </button>
        </header>

        <p className="hint">
          Pick a starter scene. The DEM is read in place; nothing is copied or
          modified.
        </p>

        <div className="preset-grid">
          {presets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className="preset"
              disabled={!preset.available || busy}
              onClick={() => void actions.createProject(preset.id)}
              title={
                preset.available
                  ? preset.description
                  : "This DEM is not present in the checkout"
              }
            >
              <span className="preset__name">{preset.name}</span>
              <span className="preset__desc">{preset.description}</span>
              <span className="preset__meta mono">
                {preset.grid ? `${preset.grid[0]} × ${preset.grid[1]} px` : "—"}
                {preset.elevationRange
                  ? ` · ${metres(preset.elevationRange[0])}–${metres(preset.elevationRange[1])}`
                  : ""}
              </span>
              {!preset.available && (
                <span className="preset__missing">DEM not found</span>
              )}
            </button>
          ))}

          <button
            type="button"
            className="preset preset--empty"
            disabled={busy}
            onClick={() => void actions.createProject("empty")}
          >
            <span className="preset__name">Empty scene</span>
            <span className="preset__desc">
              Camera, sun, and a cube — for testing mesh import and tools
            </span>
            <span className="preset__meta mono">no DEM</span>
          </button>
        </div>

        <DatasetGallery
          onPick={(presetId) => void actions.createProject(presetId)}
          disabled={busy}
        />

        <RemoteDemDialog />

        <footer className="modal__footer">
          <span className="hint">
            {presets.length === 0
              ? "Looking for the bundled DEMs…"
              : `${presets.filter((preset) => preset.available).length} bundled DEMs available`}
          </span>
          <button type="button" className="button" onClick={actions.dismissNewProject}>
            Cancel
          </button>
        </footer>
      </div>
    </div>
  );
}
