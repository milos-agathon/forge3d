// src/viewer/state/mod.rs
// State decomposition for the Viewer struct
// Split from mod.rs as part of the viewer refactoring

mod fog_state;
mod gi_state;
mod gpu_state;
mod mesh_upload;
mod resize;
mod scene_state;
mod sky_state;
mod viewer_helpers;

// resize exports Viewer::resize() impl directly
// mesh_upload exports Viewer::upload_mesh() impl directly
