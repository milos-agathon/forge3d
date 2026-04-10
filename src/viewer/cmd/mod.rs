// src/viewer/cmd/mod.rs
// Command handling for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring
//
// This module dispatches ViewerCmd variants to specialized handler modules.
// Each handler module focuses on a specific domain (GI, sky/fog, IBL, etc.)

mod effects_command;
mod gi_command;
mod handler;
mod ipc_command;
mod labels_command;
mod legacy_handler;
mod pointcloud_command;
mod scene_command;
mod scene_review_command;
mod terrain_command;
mod vector_overlay_command;

// handler.rs contains the main handle_cmd impl block on Viewer
// The other modules contain standalone helper functions
