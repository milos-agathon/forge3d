// src/viewer/cmd/mod.rs
// Command handling for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring
//
// This module dispatches ViewerCmd variants to specialized handler modules.
// Each handler module focuses on a specific domain (GI, sky/fog, IBL, etc.)

mod capture_handlers;
mod gi_handlers;
mod handler;
mod scene_handlers;
mod ssao_handlers;
mod ssgi_handlers;

// handler.rs contains the main handle_cmd impl block on Viewer
// The other modules contain standalone helper functions
