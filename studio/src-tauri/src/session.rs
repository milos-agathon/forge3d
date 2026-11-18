//! Shell state shared with the background autosave thread.
//!
//! The session logic itself is `studio_core::Session`; this file only adds the
//! mutex that Tauri commands and the autosave thread contend on.

use std::path::PathBuf;

use studio_core::autosave::AutosaveStore;
use studio_core::Session;

pub struct AppState {
    pub session: std::sync::Mutex<Session>,
    pub autosave: AutosaveStore,
}

impl AppState {
    pub fn new(autosave_dir: PathBuf) -> Self {
        Self {
            session: std::sync::Mutex::new(Session::new()),
            autosave: AutosaveStore::new(autosave_dir),
        }
    }
}
