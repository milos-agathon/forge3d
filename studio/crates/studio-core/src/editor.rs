//! An editing session: the project, its revision, and the undo/redo stacks.
//!
//! The editor owns the only mutable copy of the project. Views (Tauri commands,
//! adapters, tests) receive [`SessionState`] snapshots and mutate through
//! [`Command`]s, which is what keeps "the renderer is a projection of state"
//! true in practice.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::command::Command;
use crate::error::StudioError;
use crate::ids::ObjectId;
use crate::now_unix_ms;
use crate::project::Project;

/// Undo depth. Bounded so a long editing session cannot grow without limit;
/// the limit is a session policy, not a project-format concern.
pub const DEFAULT_HISTORY_LIMIT: usize = 256;

/// Result of an applied edit, returned to the UI and to adapters.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Applied {
    /// Human-readable label, e.g. `Move Cube`.
    pub label: String,
    /// Monotonic session revision after the edit.
    pub revision: u64,
    /// Objects a viewport adapter must re-sync.
    pub affected: Vec<ObjectId>,
}

/// One undoable step. `inverse` is stored, so undo never reconstructs state.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub label: String,
    pub command: Command,
    pub inverse: Command,
}

/// Everything the desktop shell needs to render the project.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionState {
    pub project: Project,
    pub revision: u64,
    pub dirty: bool,
    /// Saved file path, when the project has one.
    pub path: Option<String>,
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
    pub history_len: usize,
}

#[derive(Debug, Clone)]
pub struct Editor {
    project: Project,
    revision: u64,
    dirty: bool,
    path: Option<PathBuf>,
    undo: Vec<HistoryEntry>,
    redo: Vec<HistoryEntry>,
    limit: usize,
}

impl Editor {
    /// A new, never-saved project.
    pub fn new(project: Project) -> Self {
        Self::with_history_limit(project, DEFAULT_HISTORY_LIMIT)
    }

    pub fn with_history_limit(project: Project, limit: usize) -> Self {
        Self {
            project,
            revision: 0,
            dirty: true,
            path: None,
            undo: Vec::new(),
            redo: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// A project loaded from `path`: clean, with an empty history.
    pub fn opened(project: Project, path: impl Into<PathBuf>) -> Self {
        let mut editor = Self::new(project);
        editor.path = Some(path.into());
        editor.dirty = false;
        editor
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn set_path(&mut self, path: Option<PathBuf>) {
        self.path = path;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|entry| entry.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|entry| entry.label.as_str())
    }

    pub fn state(&self) -> SessionState {
        SessionState {
            project: self.project.clone(),
            revision: self.revision,
            dirty: self.dirty,
            path: self
                .path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            can_undo: self.can_undo(),
            can_redo: self.can_redo(),
            undo_label: self.undo_label().map(str::to_owned),
            redo_label: self.redo_label().map(str::to_owned),
            history_len: self.undo.len(),
        }
    }

    /// Apply a command, push it onto the undo stack, and drop the redo stack.
    ///
    /// The command stored for redo is the *canonical* one, derived by applying
    /// the inverse to a copy of the post-edit state. Without that step a
    /// `DuplicateObject` would mint a fresh UUID every time it re-ran, so redo
    /// would silently produce a different object than undo removed.
    pub fn apply(&mut self, command: Command) -> Result<Applied, StudioError> {
        let label = command.label(&self.project);
        let (inverse, affected) = command.apply_with_effect(&mut self.project)?;

        let mut scratch = self.project.clone();
        let canonical = inverse.apply(&mut scratch)?;
        drop(scratch);

        self.after_edit();
        self.redo.clear();
        self.push_undo(HistoryEntry {
            label: label.clone(),
            command: canonical,
            inverse,
        });
        Ok(Applied {
            label,
            revision: self.revision,
            affected,
        })
    }

    /// Undo the last command by moving its entry to the redo stack and
    /// applying the stored inverse. The entry keeps both directions, so redo
    /// re-applies the original command rather than its inverse.
    pub fn undo(&mut self) -> Result<Applied, StudioError> {
        let entry = self.undo.pop().ok_or(StudioError::NothingToUndo)?;
        let (_, affected) = entry.inverse.apply_with_effect(&mut self.project)?;
        self.after_edit();
        self.redo.push(entry.clone());
        Ok(Applied {
            label: format!("Undo {}", entry.label),
            revision: self.revision,
            affected,
        })
    }

    /// Re-apply the last undone command.
    pub fn redo(&mut self) -> Result<Applied, StudioError> {
        let entry = self.redo.pop().ok_or(StudioError::NothingToRedo)?;
        let (_, affected) = entry.command.apply_with_effect(&mut self.project)?;
        self.after_edit();
        self.undo.push(entry.clone());
        Ok(Applied {
            label: format!("Redo {}", entry.label),
            revision: self.revision,
            affected,
        })
    }

    /// Replace the selection. Selection is view state: it is persisted with the
    /// project but is neither undoable nor a reason to mark the project dirty.
    pub fn set_selection(&mut self, ids: Vec<ObjectId>) -> Result<(), StudioError> {
        let mut unique = Vec::with_capacity(ids.len());
        for id in ids {
            if self.project.scene.object(id).is_none() {
                return Err(StudioError::Command(
                    crate::command::CommandError::ObjectNotFound { id },
                ));
            }
            if !unique.contains(&id) {
                unique.push(id);
            }
        }
        self.project.selection = unique;
        self.project.touch();
        Ok(())
    }

    /// Move the playhead: evaluate the timeline into the project's object
    /// state at `frame` and store it. Like selection this is **view state** —
    /// not an undoable edit, no revision bump, no dirty flag — so scrubbing
    /// never floods the undo stack or triggers autosave. The same
    /// [`crate::timeline::evaluate`] the frame-sequence export uses runs here,
    /// which is what keeps playback and rendering in step.
    ///
    /// Returns the clamped frame that was applied.
    pub fn preview_playhead(&mut self, frame: f32) -> f32 {
        let frame = crate::timeline::evaluate(&mut self.project, frame);
        self.project.touch();
        frame
    }

    fn push_undo(&mut self, entry: HistoryEntry) {
        self.undo.push(entry);
        let overflow = self.undo.len().saturating_sub(self.limit);
        if overflow > 0 {
            self.undo.drain(0..overflow);
        }
    }

    fn after_edit(&mut self) {
        self.project.touch();
        self.project.updated_unix_ms = now_unix_ms();
        self.revision += 1;
        self.dirty = true;
        debug_assert!(
            self.project.validate().is_ok(),
            "a command produced an invalid project: {:?}",
            self.project.validate()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3};
    use crate::project::UNTITLED;

    fn cube_id(editor: &Editor) -> ObjectId {
        editor
            .project()
            .scene
            .objects
            .iter()
            .find(|object| object.is_mesh())
            .expect("starter scene has a cube")
            .id
    }

    #[test]
    fn apply_marks_dirty_and_bumps_revision() {
        let mut editor = Editor::new(Project::new(UNTITLED));
        assert!(editor.is_dirty());
        let applied = editor
            .apply(Command::RenameObject {
                id: cube_id(&editor),
                name: "Crate".to_owned(),
            })
            .unwrap();
        assert_eq!(applied.revision, 1);
        assert_eq!(applied.label, "Rename Cube");
        assert!(editor.can_undo());
        assert!(!editor.can_redo());
    }

    #[test]
    fn undo_and_redo_round_trip_transforms() {
        let mut editor = Editor::new(Project::new(UNTITLED));
        let id = cube_id(&editor);
        let before = editor.project().scene.object(id).unwrap().transform;

        let moved = Transform {
            translation: Vec3::new(1.5, 0.0, -2.0),
            rotation_deg: Vec3::new(0.0, 45.0, 0.0),
            scale: Vec3::new(1.0, 2.0, 1.0),
        };
        editor
            .apply(Command::SetTransform {
                id,
                transform: moved,
            })
            .unwrap();
        assert_eq!(editor.project().scene.object(id).unwrap().transform, moved);

        editor.undo().unwrap();
        assert_eq!(editor.project().scene.object(id).unwrap().transform, before);

        editor.redo().unwrap();
        assert_eq!(editor.project().scene.object(id).unwrap().transform, moved);
    }

    #[test]
    fn new_command_clears_redo_stack() {
        let mut editor = Editor::new(Project::new(UNTITLED));
        let id = cube_id(&editor);
        editor
            .apply(Command::SetVisibility { id, visible: false })
            .unwrap();
        editor.undo().unwrap();
        assert!(editor.can_redo());
        editor
            .apply(Command::SetLocked { id, locked: true })
            .unwrap();
        assert!(!editor.can_redo());
    }

    #[test]
    fn undo_without_history_reports_a_real_error() {
        let mut editor = Editor::new(Project::new(UNTITLED));
        let error = editor.undo().unwrap_err().to_string();
        assert!(error.contains("nothing to undo"), "{error}");
        let error = editor.redo().unwrap_err().to_string();
        assert!(error.contains("nothing to redo"), "{error}");
    }

    #[test]
    fn history_limit_drops_the_oldest_entries() {
        let mut editor = Editor::with_history_limit(Project::new(UNTITLED), 2);
        let id = cube_id(&editor);
        for visible in [false, true, false] {
            editor
                .apply(Command::SetVisibility { id, visible })
                .unwrap();
        }
        assert_eq!(editor.state().history_len, 2);
    }

    #[test]
    fn selection_is_not_undoable_and_not_a_dirtying_edit() {
        let mut editor = Editor::opened(Project::new(UNTITLED), "example.studio3d");
        assert!(!editor.is_dirty());
        let id = cube_id(&editor);
        editor.set_selection(vec![id]).unwrap();
        assert_eq!(editor.project().selection, vec![id]);
        assert!(!editor.is_dirty());
        assert!(!editor.can_undo());
    }

    #[test]
    fn selection_rejects_missing_objects() {
        let mut editor = Editor::new(Project::new(UNTITLED));
        assert!(editor.set_selection(vec![crate::ids::new_id()]).is_err());
    }
}
