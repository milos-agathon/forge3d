// src/viewer/event_loop/mod.rs
// Event loop handling for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

mod cmd_parse_init;
mod ipc_state;
mod runner;
mod stdin_reader;

pub use cmd_parse_init::parse_initial_commands;
pub use ipc_state::{
    get_ipc_queue, get_ipc_stats, get_pick_events, get_lasso_state, 
    update_ipc_stats, update_ipc_transform_stats,
    take_pending_bundle_save, set_pending_bundle_save,
    take_pending_bundle_load, set_pending_bundle_load,
};
pub use runner::{run_viewer, run_viewer_with_ipc};
pub use stdin_reader::spawn_stdin_reader;
