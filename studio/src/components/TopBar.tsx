import { useEffect, useRef, useState } from "react";

import { useSession } from "../state/session-context";
import {
  FileIcon,
  FolderIcon,
  RedoIcon,
  SaveIcon,
  UndoIcon,
} from "./icons";
import { AboutDialog } from "./AboutDialog";

export function TopBar({
  workspace,
  onWorkspace,
}: {
  workspace: "scene" | "cartography";
  onWorkspace(workspace: "scene" | "cartography"): void;
}) {
  const { state, actions, busy } = useSession();
  const [menuOpen, setMenuOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", escape);
    };
  }, [menuOpen]);

  const project = state?.project;
  const fileName = state?.path?.split(/[\\/]/).pop();

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand__mark" aria-hidden="true">
          F3
        </span>
        <span className="brand__name">Forge3D</span>
        <span className="brand__product">Studio</span>
      </div>

      <div className="menu" ref={menuRef}>
        <button
          type="button"
          className={`menu__button ${menuOpen ? "is-open" : ""}`}
          onClick={() => setMenuOpen((open) => !open)}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
        >
          File
        </button>
        {menuOpen && (
          <div className="menu__dropdown" role="menu">
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                actions.requestNewProject();
              }}
            >
              <FileIcon size={14} /> New project
              <kbd>Ctrl N</kbd>
            </button>
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                void actions.openProject();
              }}
            >
              <FolderIcon size={14} /> Open…
              <kbd>Ctrl O</kbd>
            </button>
            <div className="menu__separator" />
            <button
              type="button"
              role="menuitem"
              disabled={!state}
              onClick={() => {
                setMenuOpen(false);
                void actions.saveProject(false);
              }}
            >
              <SaveIcon size={14} /> Save
              <kbd>Ctrl S</kbd>
            </button>
            <button
              type="button"
              role="menuitem"
              disabled={!state}
              onClick={() => {
                setMenuOpen(false);
                void actions.saveProject(true);
              }}
            >
              <SaveIcon size={14} /> Save as…
              <kbd>Ctrl ⇧ S</kbd>
            </button>
            <div className="menu__separator" />
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                setAboutOpen(true);
              }}
            >
              About Forge3D Studio
            </button>
            <div className="menu__separator" />
            <button
              type="button"
              role="menuitem"
              disabled={!state}
              onClick={() => {
                setMenuOpen(false);
                void actions.closeProject();
              }}
            >
              Close project
            </button>
          </div>
        )}
      </div>

      <div className="segmented" role="tablist" aria-label="Workspace">
        <button
          type="button"
          role="tab"
          aria-selected={workspace === "scene"}
          className={workspace === "scene" ? "is-active" : ""}
          onClick={() => onWorkspace("scene")}
        >
          Scene
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={workspace === "cartography"}
          className={workspace === "cartography" ? "is-active" : ""}
          onClick={() => onWorkspace("cartography")}
        >
          Cartography
        </button>
      </div>

      <div className="project-identity">
        {project ? (
          <>
            <span className="project-name">{project.name}</span>
            <span className={`dot ${state?.dirty ? "dot--dirty" : "dot--saved"}`} aria-hidden="true" />
            <span className="project-meta">
              {state?.dirty ? "unsaved changes" : state?.path ? "saved" : "not saved yet"}
            </span>
            {fileName && <span className="project-path mono">{fileName}</span>}
          </>
        ) : (
          <span className="project-meta">No project open</span>
        )}
      </div>

      <div className="topbar__actions">
        <button
          type="button"
          className="icon-button"
          disabled={!state?.canUndo || busy}
          title={state?.canUndo ? `Undo ${state.undoLabel} (Ctrl+Z)` : "Nothing to undo"}
          onClick={() => void actions.undo()}
        >
          <UndoIcon />
        </button>
        <button
          type="button"
          className="icon-button"
          disabled={!state?.canRedo || busy}
          title={state?.canRedo ? `Redo ${state.redoLabel} (Ctrl+Shift+Z)` : "Nothing to redo"}
          onClick={() => void actions.redo()}
        >
          <RedoIcon />
        </button>
        <button
          type="button"
          className="button button--render"
          disabled={!state || busy}
          title="Render a full-resolution still — choose where to save it"
          onClick={() => void actions.renderStill()}
        >
          Render
        </button>
        <button
          type="button"
          className="button button--primary"
          disabled={!state || busy}
          title="Save (Ctrl+S)"
          onClick={() => void actions.saveProject(false)}
        >
          <SaveIcon size={14} /> Save
        </button>
      </div>

      {aboutOpen && <AboutDialog onClose={() => setAboutOpen(false)} />}
    </header>
  );
}
