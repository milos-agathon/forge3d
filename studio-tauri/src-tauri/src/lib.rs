//! Forge3D Studio desktop shell.
//!
//! Phase 1 wires the Tauri window to [`studio_core`]: the canonical project
//! model, the command layer, atomic save, and crash recovery. The viewport is
//! a preview until Phase 2 replaces it with a Three.js solid viewport.

mod commands;
mod forge3d;
mod render;
mod session;

use std::time::Duration;

use tauri::{AppHandle, Manager};

use forge3d::Forge3dState;
use render::RenderState;

pub use session::AppState;

pub fn run() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    // `--engine-check` answers the one question a headless stranger has:
    // can this machine run Forge3D Studio? Probe without starting the GUI.
    if std::env::args().any(|arg| arg == "--engine-check") {
        std::process::exit(forge3d::engine_check());
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let autosave_dir = app
                .path()
                .app_data_dir()
                .map_err(|error| format!("no app-data directory available: {error}"))?
                .join("recovery");
            log::info!("recovery snapshots: {}", autosave_dir.display());
            app.manage(AppState::new(autosave_dir));
            app.manage(Forge3dState::new());
            app.manage(RenderState::new());
            spawn_autosave_loop(app.handle().clone());
            Ok(())
        })
        // Closing the window is the one exit path the UI cannot intercept: take
        // a final snapshot so the last few seconds of typing are recoverable.
        .on_window_event(|window, event| {
            if !matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                return;
            }
            let Some(state) = window.try_state::<AppState>() else {
                return;
            };
            let Ok(mut session) = state.session.lock() else {
                return;
            };
            match session.autosave_if_needed(&state.autosave) {
                Ok(Some(report)) => log::info!("final recovery snapshot {}", report.id),
                Ok(None) => {}
                Err(error) => log::warn!("final recovery snapshot failed: {error}"),
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::project_state,
            commands::new_project,
            commands::scene_presets,
            commands::open_project,
            commands::save_project,
            commands::close_project,
            commands::apply_command,
            commands::undo,
            commands::redo,
            commands::set_selection,
            commands::set_playhead,
            commands::set_sun_from_solar,
            commands::render_animation_frame,
            commands::autosave_now,
            commands::autosave_status,
            commands::list_recovery,
            commands::check_assets,
            commands::restore_recovery,
            commands::discard_recovery,
            commands::read_mesh_file,
            commands::file_size,
            commands::terrain_preview,
            commands::vector_geometry,
            commands::polygon_geometry,
            commands::graticule_geometry,
            commands::colormap_stops,
            commands::dataset_list,
            commands::remote_cog_preview,
            commands::dataset_path,
            commands::remote_cog_info,
            commands::import_terrain_dem,
            commands::read_image_asset,
            commands::read_hdr,
            commands::raster_preview,
            commands::add_imagery_layer,
            commands::add_vector_layer,
            commands::add_polygon_layer,
            commands::forge3d_probe,
            commands::render_frame,
            commands::render_still,
            commands::render_status,
            commands::render_stop,
            commands::shell_info,
            commands::capability_table,
        ])
        .run(tauri::generate_context!())
        .expect("failed to start Forge3D Studio");
}

/// Periodically snapshot unsaved work. Runs off the UI thread: a slow disk
/// delays the snapshot, never the editor. The session lock is released before
/// pruning so a directory scan cannot block a keystroke.
fn spawn_autosave_loop(app: AppHandle) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(studio_core::AUTOSAVE_INTERVAL_SECONDS));
        let state = app.state::<AppState>();

        let wrote = {
            let Ok(mut session) = state.session.lock() else {
                log::warn!("autosave skipped: session mutex is poisoned");
                continue;
            };
            match session.autosave_if_needed(&state.autosave) {
                Ok(report) => report,
                Err(error) => {
                    log::warn!("autosave failed: {error}");
                    None
                }
            }
        };

        if let Some(report) = wrote {
            log::info!("autosaved {} at {}", report.id, report.written_unix_ms);
            if let Err(error) = state.autosave.prune(studio_core::DEFAULT_AUTOSAVE_KEEP) {
                log::warn!("failed to prune recovery snapshots: {error}");
            }
        }
    });
}
