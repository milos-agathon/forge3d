//! Errors surfaced by the project layer.
//!
//! Every variant carries enough context for the UI to show a real message:
//! users must never see a bare "operation failed", and a failed save must
//! always leave the previous file intact (see [`crate::persist`]).

use std::path::PathBuf;

use crate::command::CommandError;

#[derive(Debug, thiserror::Error)]
pub enum StudioError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{context} is not valid JSON: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("{context} uses schema version {found}, but this build supports {supported}")]
    UnsupportedSchema {
        context: String,
        found: u32,
        supported: u32,
    },

    #[error("{context} is not a Studio project (format tag {found:?}, expected \"{expected}\")")]
    WrongFormat {
        context: String,
        found: String,
        expected: String,
    },

    #[error("invalid project: {reason}")]
    InvalidProject { reason: String },

    #[error("nothing to undo")]
    NothingToUndo,

    #[error("nothing to redo")]
    NothingToRedo,

    #[error(transparent)]
    Command(#[from] CommandError),
}

impl StudioError {
    /// Build a JSON error tied to a display context (file path or "autosave").
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    /// Build an `InvalidProject` error with a formatted reason.
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::InvalidProject {
            reason: reason.into(),
        }
    }
}
