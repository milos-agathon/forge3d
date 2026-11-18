//! Crash-recovery snapshots.
//!
//! Autosave does not touch the user's project file. It writes timestamped
//! envelopes into a recovery directory, which the shell lists on startup so a
//! crash mid-edit costs the user at most the autosave interval — never the
//! saved file.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::StudioError;
use crate::project::Project;

/// How many snapshots to keep per source project by default.
pub const DEFAULT_AUTOSAVE_KEEP: usize = 20;

/// Marker written into every recovery envelope.
const ENVELOPE_KIND: &str = "studio3d.autosave";
const EXTENSION: &str = "autosave";

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AutosaveFile {
    kind: String,
    written_unix_ms: u64,
    source_path: Option<String>,
    project: Project,
}

/// A recovery snapshot the user can restore or discard.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecoveryEntry {
    /// Stable handle used by `restore`/`discard`; a file name, not a path.
    pub id: String,
    pub written_unix_ms: u64,
    /// Project file the snapshot came from, when it had been saved before.
    pub source_path: Option<String>,
    pub project_id: String,
    pub project_name: String,
    pub object_count: usize,
    /// Absolute path of the snapshot, for diagnostics only.
    pub path: String,
}

/// A directory of recovery snapshots.
#[derive(Debug, Clone)]
pub struct AutosaveStore {
    dir: PathBuf,
}

impl AutosaveStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write a snapshot. Returns the entry the UI can show immediately.
    pub fn write(
        &self,
        source: Option<&Path>,
        project: &Project,
    ) -> Result<RecoveryEntry, StudioError> {
        project.validate().map_err(StudioError::invalid)?;
        fs::create_dir_all(&self.dir).map_err(|source| StudioError::Write {
            path: self.dir.clone(),
            source,
        })?;

        let written_unix_ms = crate::now_unix_ms();
        let id = format!("{written_unix_ms}-{}.{EXTENSION}", project.id);
        let envelope = AutosaveFile {
            kind: ENVELOPE_KIND.to_owned(),
            written_unix_ms,
            source_path: source.map(|path| path.to_string_lossy().into_owned()),
            project: project.clone(),
        };
        let json = serde_json::to_vec_pretty(&envelope)
            .map_err(|source| StudioError::json("autosave snapshot", source))?;

        // Same discipline as `save_atomic`: temp file, flush, rename.
        let target = self.dir.join(&id);
        let tmp = self.dir.join(format!(".{id}.{}.tmp", std::process::id()));
        let written = (|| -> std::io::Result<()> {
            let mut file = fs::File::create(&tmp)?;
            std::io::Write::write_all(&mut file, &json)?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(source) = written {
            let _ = fs::remove_file(&tmp);
            return Err(StudioError::Write { path: tmp, source });
        }
        if let Err(source) = fs::rename(&tmp, &target) {
            let _ = fs::remove_file(&tmp);
            return Err(StudioError::Write {
                path: target,
                source,
            });
        }

        Ok(RecoveryEntry {
            id,
            written_unix_ms,
            source_path: envelope.source_path,
            project_id: project.id.to_string(),
            project_name: project.name.clone(),
            object_count: project.scene.objects.len(),
            path: target.display().to_string(),
        })
    }

    /// All snapshots, newest first. Unreadable or corrupt files are skipped:
    /// a broken snapshot must not stop the user from recovering a good one.
    pub fn list(&self) -> Result<Vec<RecoveryEntry>, StudioError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(StudioError::Read {
                    path: self.dir.clone(),
                    source,
                })
            }
        };

        let mut found = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some(EXTENSION) {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if is_temporary(name) {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(envelope) = serde_json::from_str::<AutosaveFile>(&text) else {
                continue;
            };
            if envelope.kind != ENVELOPE_KIND || envelope.project.validate().is_err() {
                continue;
            }
            found.push(RecoveryEntry {
                id: name.to_owned(),
                written_unix_ms: envelope.written_unix_ms,
                source_path: envelope.source_path,
                project_id: envelope.project.id.to_string(),
                project_name: envelope.project.name.clone(),
                object_count: envelope.project.scene.objects.len(),
                path: path.display().to_string(),
            });
        }
        found.sort_by_key(|entry| std::cmp::Reverse(entry.written_unix_ms));
        Ok(found)
    }

    pub fn load(&self, id: &str) -> Result<Project, StudioError> {
        let path = self.path_for(id)?;
        let text = fs::read_to_string(&path).map_err(|source| StudioError::Read {
            path: path.clone(),
            source,
        })?;
        let envelope: AutosaveFile = serde_json::from_str(&text)
            .map_err(|source| StudioError::json(format!("autosave {id}"), source))?;
        if envelope.kind != ENVELOPE_KIND {
            return Err(StudioError::invalid(format!(
                "{id} is not a Studio autosave snapshot"
            )));
        }
        envelope.project.validate().map_err(StudioError::invalid)?;
        Ok(envelope.project)
    }

    pub fn discard(&self, id: &str) -> Result<(), StudioError> {
        let path = self.path_for(id)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(StudioError::Write { path, source }),
        }
    }

    /// Drop the oldest snapshots, keeping `keep` per source project.
    pub fn prune(&self, keep: usize) -> Result<usize, StudioError> {
        let mut by_source: std::collections::HashMap<Option<String>, Vec<RecoveryEntry>> =
            std::collections::HashMap::new();
        for entry in self.list()? {
            by_source
                .entry(entry.source_path.clone())
                .or_default()
                .push(entry);
        }
        let mut removed = 0;
        for entries in by_source.values_mut() {
            entries.sort_by_key(|entry| std::cmp::Reverse(entry.written_unix_ms));
            for stale in entries.iter().skip(keep.max(1)) {
                self.discard(&stale.id)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Resolve a snapshot file name inside the store, rejecting anything that
    /// tries to escape the directory.
    fn path_for(&self, id: &str) -> Result<PathBuf, StudioError> {
        // Exactly one normal path component: no separators, no `..`, no root.
        let mut components = Path::new(id).components();
        let single_normal =
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
        if !single_normal || is_temporary(id) {
            return Err(StudioError::invalid(format!(
                "{id:?} is not a recovery snapshot id"
            )));
        }
        Ok(self.dir.join(id))
    }
}

fn is_temporary(name: &str) -> bool {
    name.starts_with('.') || name.ends_with(".tmp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::UNTITLED;

    #[test]
    fn missing_directory_lists_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = AutosaveStore::new(dir.path().join("does-not-exist"));
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn path_traversal_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = AutosaveStore::new(dir.path());
        assert!(store.load("../secrets.autosave").is_err());
        assert!(store.load("/etc/passwd").is_err());
        assert!(store.discard("nested/file.autosave").is_err());
    }

    #[test]
    fn corrupt_snapshots_are_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("broken.autosave"), "{ not json").unwrap();
        let store = AutosaveStore::new(dir.path());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn write_then_list_reports_the_project_identity() {
        let dir = tempfile::tempdir().unwrap();
        let store = AutosaveStore::new(dir.path());
        let project = Project::new(UNTITLED);
        let entry = store.write(None, &project).unwrap();
        assert_eq!(entry.project_name, project.name);
        assert_eq!(entry.object_count, 3);
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, entry.id);
    }

    #[test]
    fn prune_keeps_the_newest_per_source() {
        let dir = tempfile::tempdir().unwrap();
        let store = AutosaveStore::new(dir.path());
        let project = Project::new(UNTITLED);
        for _ in 0..4 {
            store
                .write(Some(Path::new("/tmp/a.studio3d")), &project)
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(store.list().unwrap().len(), 4);
        store.prune(2).unwrap();
        assert_eq!(store.list().unwrap().len(), 2);
    }
}
