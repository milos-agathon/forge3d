import { useSession } from "../state/session-context";
import { CloseIcon } from "./icons";

/**
 * About Forge3D Studio.
 *
 * The shell knows its own version and the engine probe's outcome; everything
 * here is read from those, so the dialog cannot drift from the status bar.
 */
const LICENSE = "Apache-2.0 OR MIT";

const SHORTCUTS: Array<{ keys: string; action: string }> = [
  { keys: "V / G / R / S", action: "Select, move, rotate, scale" },
  { keys: "X or Del", action: "Delete selection" },
  { keys: "Shift+D", action: "Duplicate selection" },
  { keys: "0", action: "Toggle camera view" },
  { keys: "End / Numpad 1", action: "Set the camera from this view" },
  { keys: "N", action: "Toggle the properties sidebar" },
  { keys: "W A S D · Q E", action: "Fly the camera (fly mode)" },
  { keys: "Drag", action: "Look around in fly mode, orbit otherwise" },
  { keys: "Ctrl+Z / Ctrl+Shift+Z", action: "Undo / redo" },
  { keys: "Ctrl+S", action: "Save" },
  { keys: "Ctrl+O · Ctrl+N", action: "Open / new project" },
];

export function AboutDialog({ onClose }: { onClose: () => void }) {
  const { info, engine } = useSession();

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="About Forge3D Studio"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal modal--about">
        <header className="modal__header">
          <h2>About Forge3D Studio</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close"
            onClick={onClose}
          >
            <CloseIcon size={14} />
          </button>
        </header>

        <dl className="about__facts">
          <div>
            <dt>Shell</dt>
            <dd>{info?.name ?? "Forge3D Studio"}</dd>
          </div>
          <div>
            <dt>Version</dt>
            <dd className="mono">{info?.version ?? "—"}</dd>
          </div>
          <div>
            <dt>Engine</dt>
            <dd>
              {engine?.available
                ? `forge3d ${engine.packageVersion ?? "?"}`
                : "probing…"}
            </dd>
          </div>
          <div>
            <dt>Backend</dt>
            <dd>{engine?.backend ?? "—"}</dd>
          </div>
          <div>
            <dt>Adapter</dt>
            <dd>{engine?.adapter ?? "—"}</dd>
          </div>
          <div>
            <dt>Device features</dt>
            <dd>{engine?.features?.length ? engine.features.length : "—"}</dd>
          </div>
          <div>
            <dt>License</dt>
            <dd>{LICENSE}</dd>
          </div>
        </dl>

        <h3 className="about__heading">Keyboard shortcuts</h3>
        <table className="about__shortcuts">
          <tbody>
            {SHORTCUTS.map(({ keys, action }) => (
              <tr key={keys}>
                <td>
                  <kbd>{keys}</kbd>
                </td>
                <td>{action}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <footer className="modal__footer">
          <span className="hint">
            Rendering by the Forge3D engine (Rust + WebGPU) through the Tauri shell.
          </span>
          <button type="button" className="button" onClick={onClose}>
            Close
          </button>
        </footer>
      </div>
    </div>
  );
}
