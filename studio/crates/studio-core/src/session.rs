//! The shell session: at most one open editor, plus recovery bookkeeping.
//!
//! This lives in the core crate rather than the Tauri crate so the whole
//! open → edit → save → recover flow is testable without a window, a webview, or
//! a GPU. The Tauri layer only adds a mutex and IPC.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::autosave::{AutosaveStore, RecoveryEntry};
use crate::command::Command;
use crate::editor::{Applied, Editor, SessionState};
use crate::error::StudioError;
use crate::ids::ObjectId;
use crate::persist::{load, save_atomic};
use crate::project::{Project, TerrainSeed, UNTITLED};

/// How often the desktop shell considers writing a recovery snapshot.
pub const AUTOSAVE_INTERVAL_SECONDS: u64 = 30;

/// Result of an autosave attempt, surfaced to the status bar.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AutosaveReport {
    pub id: String,
    pub written_unix_ms: u64,
    pub source_path: Option<PathBuf>,
}

pub struct Session {
    editor: Option<Editor>,
    /// Revision of the last snapshot, so an idle session stops writing.
    last_snapshot_revision: Option<u64>,
    last_report: Option<AutosaveReport>,
    /// Last autosave failure, kept so the UI can show it instead of hiding it.
    last_error: Option<String>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            editor: None,
            last_snapshot_revision: None,
            last_report: None,
            last_error: None,
        }
    }

    pub fn state(&self) -> Option<SessionState> {
        self.editor.as_ref().map(Editor::state)
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn last_report(&self) -> Option<&AutosaveReport> {
        self.last_report.as_ref()
    }

    /// Start a new unsaved project, discarding the current session.
    pub fn new_project(&mut self, name: Option<String>) -> SessionState {
        self.new_project_seeded(name, None)
    }

    /// Start a new project with a terrain starter when a DEM was resolved.
    pub fn new_project_seeded(
        &mut self,
        name: Option<String>,
        terrain: Option<TerrainSeed>,
    ) -> SessionState {
        let project = Project::starter(name.unwrap_or_else(|| UNTITLED.to_owned()), terrain);
        self.begin(Editor::new(project))
    }

    /// Load a project file. The current session is only replaced on success.
    pub fn open_project(&mut self, path: &Path) -> Result<SessionState, StudioError> {
        let project = load(path)?;
        Ok(self.begin(Editor::opened(project, path)))
    }

    /// Save to `path`, or to the session's path when omitted.
    pub fn save_project(&mut self, path: Option<PathBuf>) -> Result<SessionState, StudioError> {
        let mut editor = self
            .editor
            .take()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;

        let target = match path.or_else(|| editor.path().map(Path::to_path_buf)) {
            Some(target) => target,
            None => {
                self.editor = Some(editor);
                return Err(StudioError::invalid(
                    "this project has no path yet; choose a location first",
                ));
            }
        };

        if let Err(error) = save_atomic(&target, editor.project()) {
            // The editor goes back exactly as it was: a failed save must not
            // cost the user the session.
            self.editor = Some(editor);
            return Err(error);
        }

        editor.set_path(Some(target));
        editor.mark_saved();
        let state = editor.state();
        self.editor = Some(editor);
        self.last_snapshot_revision = None;
        Ok(state)
    }

    pub fn apply(&mut self, command: Command) -> Result<(Applied, SessionState), StudioError> {
        let editor = self
            .editor
            .as_mut()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;
        let applied = editor.apply(command)?;
        Ok((applied, editor.state()))
    }

    pub fn undo(&mut self) -> Result<(Applied, SessionState), StudioError> {
        let editor = self
            .editor
            .as_mut()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;
        let applied = editor.undo()?;
        Ok((applied, editor.state()))
    }

    pub fn redo(&mut self) -> Result<(Applied, SessionState), StudioError> {
        let editor = self
            .editor
            .as_mut()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;
        let applied = editor.redo()?;
        Ok((applied, editor.state()))
    }

    pub fn set_selection(&mut self, ids: Vec<ObjectId>) -> Result<SessionState, StudioError> {
        let editor = self
            .editor
            .as_mut()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;
        editor.set_selection(ids)?;
        Ok(editor.state())
    }

    /// Move the playhead and evaluate the timeline into the project (view
    /// state — not undoable, not a reason to autosave). Returns the new
    /// snapshot so the UI re-renders with the evaluated transforms, plus the
    /// clamped frame that was applied.
    pub fn set_playhead(&mut self, frame: f32) -> Result<(f32, SessionState), StudioError> {
        let editor = self
            .editor
            .as_mut()
            .ok_or_else(|| StudioError::invalid("no project is open"))?;
        let frame = editor.preview_playhead(frame);
        Ok((frame, editor.state()))
    }

    /// Swap in a recovered project. It stays dirty: the user still has to save.
    pub fn restore_recovery(&mut self, entry: &RecoveryEntry, project: Project) -> SessionState {
        let mut editor = Editor::new(project);
        if let Some(source) = entry.source_path.as_deref() {
            editor.set_path(Some(PathBuf::from(source)));
        }
        self.begin(editor)
    }

    pub fn close_project(&mut self) {
        self.editor = None;
        self.last_snapshot_revision = None;
        self.last_report = None;
        self.last_error = None;
    }

    /// Write a snapshot when the project is dirty and changed since the last
    /// snapshot. Returns `None` when there is nothing new to write.
    pub fn autosave_if_needed(
        &mut self,
        store: &AutosaveStore,
    ) -> Result<Option<AutosaveReport>, StudioError> {
        let Some(editor) = self.editor.as_ref() else {
            return Ok(None);
        };
        if !editor.is_dirty() || self.last_snapshot_revision == Some(editor.revision()) {
            return Ok(None);
        }

        let revision = editor.revision();
        match store.write(editor.path(), editor.project()) {
            Ok(entry) => {
                let report = AutosaveReport {
                    id: entry.id,
                    written_unix_ms: entry.written_unix_ms,
                    source_path: entry.source_path.map(PathBuf::from),
                };
                self.last_snapshot_revision = Some(revision);
                self.last_report = Some(report.clone());
                self.last_error = None;
                Ok(Some(report))
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// Drop snapshots belonging to `path`, called after a successful save.
    pub fn prune_recovery_for(
        &mut self,
        store: &AutosaveStore,
        path: &Path,
    ) -> Result<usize, StudioError> {
        let target = path.to_string_lossy();
        let mut removed = 0;
        for entry in store.list()? {
            if entry.source_path.as_deref() == Some(target.as_ref()) {
                store.discard(&entry.id)?;
                removed += 1;
            }
        }
        if removed > 0 {
            self.last_snapshot_revision = None;
        }
        Ok(removed)
    }

    fn begin(&mut self, editor: Editor) -> SessionState {
        let state = editor.state();
        self.editor = Some(editor);
        self.last_snapshot_revision = None;
        self.last_report = None;
        self.last_error = None;
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;
    use crate::math::Vec3;
    use crate::scene::ObjectKind;

    fn temp_store() -> (tempfile::TempDir, AutosaveStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = AutosaveStore::new(dir.path().join("recovery"));
        (dir, store)
    }

    fn cube_id(session: &Session) -> ObjectId {
        session
            .state()
            .expect("project open")
            .project
            .scene
            .objects
            .iter()
            .find(|object| object.is_mesh())
            .unwrap()
            .id
    }

    #[test]
    fn commands_require_an_open_project() {
        let mut session = Session::new();
        let error = session
            .apply(Command::SetActiveCamera { id: None })
            .unwrap_err()
            .to_string();
        assert!(error.contains("no project is open"), "{error}");
        assert!(session.state().is_none());
    }

    #[test]
    fn saving_without_a_path_is_refused_with_guidance() {
        let mut session = Session::new();
        session.new_project(None);
        let error = session.save_project(None).unwrap_err().to_string();
        assert!(error.contains("choose a location"), "{error}");
        // The session survives the refusal.
        assert!(session.state().is_some());
    }

    #[test]
    fn save_marks_clean_and_records_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("saved.studio3d");
        let mut session = Session::new();
        session.new_project(Some("Saved".to_owned()));
        assert!(session.state().unwrap().dirty);

        let saved = session.save_project(Some(path.clone())).unwrap();
        assert!(!saved.dirty);
        assert_eq!(saved.path.as_deref(), Some(path.to_str().unwrap()));
        assert!(path.exists());
    }

    #[test]
    fn a_failed_save_keeps_the_session_and_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("blocked.studio3d");
        std::fs::create_dir(&blocked).unwrap();

        let mut session = Session::new();
        session.new_project(None);
        assert!(session.save_project(Some(blocked.clone())).is_err());
        assert!(session.state().is_some(), "session was lost on failed save");
        assert!(session.state().unwrap().dirty);
    }

    #[test]
    fn autosave_skips_clean_projects_and_dedupes_revisions() {
        let (_dir, store) = temp_store();
        let mut session = Session::new();
        session.new_project(None);

        let first = session.autosave_if_needed(&store).unwrap();
        assert!(first.is_some(), "a brand new project is dirty");
        assert!(
            session.autosave_if_needed(&store).unwrap().is_none(),
            "same revision must not be written twice"
        );

        let cube = cube_id(&session);
        session
            .apply(Command::SetVisibility {
                id: cube,
                visible: false,
            })
            .unwrap();
        assert!(session.autosave_if_needed(&store).unwrap().is_some());

        let path = store.dir().join("does-not-matter.studio3d");
        std::fs::write(&path, "x").unwrap();
        let _ = path;
    }

    #[test]
    fn saving_prunes_recovery_snapshots_for_that_path() {
        let dir = tempfile::tempdir().unwrap();
        let project_path = dir.path().join("prunable.studio3d");
        let (store_dir, store) = temp_store();
        let mut session = Session::new();
        session.new_project(None);

        session.save_project(Some(project_path.clone())).unwrap();
        let cube = cube_id(&session);
        session
            .apply(Command::SetTransform {
                id: cube,
                transform: crate::math::Transform {
                    translation: Vec3::new(1.0, 0.0, 0.0),
                    ..Default::default()
                },
            })
            .unwrap();
        assert!(session.autosave_if_needed(&store).unwrap().is_some());
        assert_eq!(store.list().unwrap().len(), 1);

        session.save_project(None).unwrap();
        assert_eq!(
            session.prune_recovery_for(&store, &project_path).unwrap(),
            1
        );
        assert!(store.list().unwrap().is_empty());
        drop(store_dir);
    }

    #[test]
    fn restored_projects_are_dirty_and_point_at_their_source() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("original.studio3d");
        let (store_dir, store) = temp_store();
        let mut session = Session::new();
        session.new_project(Some("Recovered".to_owned()));
        session.save_project(Some(source.clone())).unwrap();

        let cube = cube_id(&session);
        session
            .apply(Command::SetVisibility {
                id: cube,
                visible: false,
            })
            .unwrap();
        let project = session.state().unwrap().project;
        let record = store.write(Some(&source), &project).unwrap();
        let loaded = store.load(&record.id).unwrap();
        let state = session.restore_recovery(&record, loaded);

        assert!(state.dirty, "recovered work must not look saved");
        assert_eq!(state.path.as_deref(), Some(source.to_str().unwrap()));
        assert!(!state.can_undo, "recovery starts a fresh history");
        drop(store_dir);
    }

    #[test]
    fn closing_discards_the_session_but_not_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kept.studio3d");
        let mut session = Session::new();
        session.new_project(None);
        session.save_project(Some(path.clone())).unwrap();
        session.close_project();
        assert!(session.state().is_none());
        assert!(path.exists());
    }

    #[test]
    fn opening_a_broken_file_keeps_the_current_project() {
        let dir = tempfile::tempdir().unwrap();
        let broken = dir.path().join("broken.studio3d");
        std::fs::write(&broken, "{ not a project").unwrap();

        let mut session = Session::new();
        session.new_project(Some("Still open".to_owned()));
        let before = session.state().unwrap().project.id;

        assert!(session.open_project(&broken).is_err());
        assert_eq!(session.state().unwrap().project.id, before);
    }

    #[test]
    fn scene_kind_check_survives_the_session_boundary() {
        let mut session = Session::new();
        session.new_project(None);
        let state = session.state().unwrap();
        assert!(matches!(
            state.project.scene.objects[0].kind,
            ObjectKind::Mesh(_)
        ));
    }
}
