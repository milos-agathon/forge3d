// src/viewer/event_loop/ipc_state.rs
// IPC state management for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use super::super::ipc::ViewerStats;
use super::super::viewer_enums::ViewerCmd;
use crate::picking::PickEvent;

/// Global IPC command queue - static ensures visibility across threads
static IPC_QUEUE: OnceLock<Mutex<VecDeque<ViewerCmd>>> = OnceLock::new();

/// Global picking event queue for polling
static PICK_EVENTS: OnceLock<Mutex<Vec<PickEvent>>> = OnceLock::new();

/// Global lasso state string (simple shared state)
static LASSO_STATE: OnceLock<Mutex<String>> = OnceLock::new();

/// Get the global IPC command queue
pub fn get_ipc_queue() -> &'static Mutex<VecDeque<ViewerCmd>> {
    IPC_QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Get the global picking event queue
pub fn get_pick_events() -> &'static Mutex<Vec<PickEvent>> {
    PICK_EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Get the global lasso state
pub fn get_lasso_state() -> &'static Mutex<String> {
    LASSO_STATE.get_or_init(|| Mutex::new("inactive".to_string()))
}

/// Global viewer stats for IPC queries
static IPC_STATS: OnceLock<Mutex<ViewerStats>> = OnceLock::new();

/// Get the global IPC stats
pub fn get_ipc_stats() -> &'static Mutex<ViewerStats> {
    IPC_STATS.get_or_init(|| Mutex::new(ViewerStats::default()))
}

/// Update IPC stats with current viewer state
pub fn update_ipc_stats(vb_ready: bool, vertex_count: u32, index_count: u32, scene_has_mesh: bool) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.vb_ready = vb_ready;
        stats.vertex_count = vertex_count;
        stats.index_count = index_count;
        stats.scene_has_mesh = scene_has_mesh;
    }
}

/// Update IPC transform stats
pub fn update_ipc_transform_stats(transform_version: u64, transform_is_identity: bool) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.transform_version = transform_version;
        stats.transform_is_identity = transform_is_identity;
    }
}

/// Bundle save request: (path, optional name)
static PENDING_BUNDLE_SAVE: OnceLock<Mutex<Option<(String, Option<String>)>>> = OnceLock::new();

/// Bundle load request: path
static PENDING_BUNDLE_LOAD: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Get the pending bundle save request (path, optional name).
/// Calling this clears the pending request.
pub fn take_pending_bundle_save() -> Option<(String, Option<String>)> {
    let lock = PENDING_BUNDLE_SAVE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        guard.take()
    } else {
        None
    }
}

/// Peek the pending bundle save request without consuming it.
pub fn peek_pending_bundle_save() -> Option<(String, Option<String>)> {
    let lock = PENDING_BUNDLE_SAVE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = lock.lock() {
        guard.clone()
    } else {
        None
    }
}

/// Set a pending bundle save request.
pub fn set_pending_bundle_save(path: String, name: Option<String>) {
    let lock = PENDING_BUNDLE_SAVE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = Some((path, name));
    }
}

/// Get the pending bundle load request (path).
/// Calling this clears the pending request.
pub fn take_pending_bundle_load() -> Option<String> {
    let lock = PENDING_BUNDLE_LOAD.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        guard.take()
    } else {
        None
    }
}

/// Peek the pending bundle load request without consuming it.
pub fn peek_pending_bundle_load() -> Option<String> {
    let lock = PENDING_BUNDLE_LOAD.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = lock.lock() {
        guard.clone()
    } else {
        None
    }
}

/// Set a pending bundle load request.
pub fn set_pending_bundle_load(path: String) {
    let lock = PENDING_BUNDLE_LOAD.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = Some(path);
    }
}
