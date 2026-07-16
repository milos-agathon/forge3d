// src/viewer/event_loop/runner.rs
// Event loop runner functions for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::io;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use super::super::ipc;
use super::super::viewer_enums::ViewerCmd;
use super::super::Viewer;
use super::super::ViewerConfig;
use super::super::INITIAL_CMDS;
use super::{
    command_priority, get_ipc_queue, get_ipc_stats, order_command_batch, parse_initial_commands,
    spawn_stdin_reader, update_ipc_stats, QueuedIpcCommand,
};

fn apply_command_batch(viewer: &mut Viewer, commands: Vec<ViewerCmd>) -> Result<(), String> {
    viewer.preflight_command_batch(&commands)?;
    for command in order_command_batch(commands) {
        viewer.handle_cmd(command)?;
    }
    Ok(())
}

type EventLoopFatal = Arc<Mutex<Option<String>>>;

fn record_event_loop_fatal(fatal: &EventLoopFatal, message: String) {
    match fatal.lock() {
        Ok(mut slot) => *slot = Some(message),
        Err(_) => eprintln!("[viewer] fatal-error channel is poisoned"),
    }
}

fn propagate_event_loop_fatal(fatal: EventLoopFatal) -> Result<(), Box<dyn std::error::Error>> {
    let message = fatal
        .lock()
        .map_err(|_| io::Error::other("viewer fatal-error channel is poisoned"))?
        .take();
    match message {
        Some(message) => Err(io::Error::other(message).into()),
        None => Ok(()),
    }
}

#[cfg(feature = "extension-module")]
use super::super::INITIAL_TERRAIN_CONFIG;

/// Entry point for the interactive viewer with single-terminal workflow
pub fn run_viewer(config: ViewerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Create an event loop that supports user events (ViewerCmd)
    let event_loop: EventLoop<ViewerCmd> =
        EventLoopBuilder::<ViewerCmd>::with_user_event().build()?;
    let proxy: EventLoopProxy<ViewerCmd> = event_loop.create_proxy();

    // Create window
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                config.width as f64,
                config.height as f64,
            ))
            .build(&event_loop)?,
    );

    // Collect initial commands provided by example CLI
    let mut pending_cmds: Vec<ViewerCmd> = if let Some(cmds) = INITIAL_CMDS.get() {
        parse_initial_commands(cmds)?
    } else {
        Vec::new()
    };

    // Spawn stdin reader thread
    spawn_stdin_reader(proxy);

    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();
    let mut pending_scale_factor_resize = false;
    let fatal_error: EventLoopFatal = Arc::new(Mutex::new(None));
    let fatal_for_loop = Arc::clone(&fatal_error);

    event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed if viewer_opt.is_none() => {
                // Initialize viewer on resume (required for some platforms)
                let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                match v {
                    Ok(v) => {
                        viewer_opt = Some(v);
                        last_frame = Instant::now();
                        // If an initial terrain config was provided (via open_terrain_viewer),
                        // attempt to attach a TerrainScene before applying CLI commands.
                        #[cfg(feature = "extension-module")]
                        if let Some(cfg) = INITIAL_TERRAIN_CONFIG.get() {
                            if let Some(viewer) = viewer_opt.as_mut() {
                                if let Err(e) = viewer.load_terrain_from_config(cfg) {
                                    eprintln!(
                                        "[viewer] failed to load terrain scene from config: {}",
                                        e
                                    );
                                }
                            }
                        }
                        // Apply any pending commands from CLI now that viewer exists
                        if let Some(viewer) = viewer_opt.as_mut() {
                            if let Err(error) =
                                apply_command_batch(viewer, std::mem::take(&mut pending_cmds))
                            {
                                eprintln!("[viewer] initial command batch rejected: {error}");
                            }
                        }
                    }
                    Err(e) => {
                        let message = format!("Failed to create viewer: {e}");
                        eprintln!("{message}");
                        record_event_loop_fatal(&fatal_for_loop, message);
                        elwt.exit();
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if matches!(event, WindowEvent::ScaleFactorChanged { .. }) {
                    pending_scale_factor_resize = true;
                }
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } if key_event.state == ElementState::Pressed => {
                                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                                    elwt.exit();
                                }
                            }
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if pending_scale_factor_resize {
                        let size = viewer.window.inner_size();
                        if size.width != viewer.config.width || size.height != viewer.config.height
                        {
                            viewer.resize(size);
                        }
                        pending_scale_factor_resize = false;
                    }
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    viewer.update(dt);
                    match viewer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            viewer.resize(viewer.window.inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            eprintln!("Surface timeout!");
                            std::thread::sleep(Duration::from_millis(4));
                            window.request_redraw();
                        }
                    }
                }
            }
            Event::UserEvent(cmd) => match cmd {
                ViewerCmd::Quit => {
                    // Process any pending snapshot before exiting
                    if let Some(viewer) = viewer_opt.as_mut() {
                        if viewer.snapshot_request.is_some() {
                            viewer.update(0.0);
                            let _ = viewer.render();
                        }
                    }
                    elwt.exit();
                }
                other => {
                    if let Some(viewer) = viewer_opt.as_mut() {
                        eprintln!("[IPC] Processing command: {:?}", other);
                        if let Err(error) = apply_command_batch(viewer, vec![other]) {
                            eprintln!("[viewer] command rejected without mutation: {error}");
                        }
                    } else {
                        eprintln!("[IPC] Viewer not ready, dropping command");
                    }
                }
            },
            _ => {}
        }
    })?;

    propagate_event_loop_fatal(fatal_error)
}

/// Run the viewer with an IPC server for non-blocking Python control.
/// Prints `FORGE3D_VIEWER_READY port=<PORT>` when the server is listening.
pub fn run_viewer_with_ipc(
    mut config: ViewerConfig,
    ipc_config: ipc::IpcServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    config.vsync = false;

    // Clear any stale commands and stats
    if let Ok(mut q) = get_ipc_queue().lock() {
        q.clear();
    }
    update_ipc_stats(false, 0, 0, false);

    // Create simple event loop (no user events needed)
    let event_loop: EventLoop<()> = EventLoop::new()?;

    // Start IPC server - pushes to global queue, reads from global stats
    let ipc_handle = ipc::start_ipc_server(
        ipc_config,
        move |cmd| {
            let (completion, result) = mpsc::sync_channel(1);
            get_ipc_queue()
                .lock()
                .map_err(|_| "Queue lock failed".to_string())?
                .push_back(QueuedIpcCommand { cmd, completion });
            result
                .recv_timeout(std::time::Duration::from_secs(300))
                .map_err(|error| format!("viewer command completion unavailable: {error}"))?
        },
        || {
            get_ipc_stats()
                .lock()
                .map(|s| s.clone())
                .unwrap_or_default()
        },
    )?;

    // Capture port for printing READY after viewer is initialized
    let ipc_port = ipc_handle.port;

    // Create window
    eprintln!(
        "[viewer-ipc] Creating window {}x{}",
        config.width, config.height
    );
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                config.width as f64,
                config.height as f64,
            ))
            .with_visible(true)
            .build(&event_loop)?,
    );
    eprintln!("[viewer-ipc] Window created, waiting for Resumed event");

    // Collect initial commands
    let mut pending_cmds: Vec<ViewerCmd> = if let Some(cmds) = INITIAL_CMDS.get() {
        parse_initial_commands(cmds)?
    } else {
        Vec::new()
    };

    // Viewer state
    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();
    let mut pending_scale_factor_resize = false;
    let fatal_error: EventLoopFatal = Arc::new(Mutex::new(None));
    let fatal_for_loop = Arc::clone(&fatal_error);

    event_loop.run(move |event, elwt| {
        // IPC needs prompt command handling, but an unthrottled Poll loop can
        // request redraws faster than the swapchain releases presentable images.
        elwt.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(8),
        ));

        match event {
            Event::Resumed => {
                eprintln!("[viewer-ipc] Received Resumed event");
                if viewer_opt.is_none() {
                    eprintln!("[viewer-ipc] Initializing Viewer...");
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(mut v) => {
                            eprintln!("[viewer-ipc] Viewer initialized successfully");
                            if let Err(error) =
                                apply_command_batch(&mut v, std::mem::take(&mut pending_cmds))
                            {
                                eprintln!("[viewer-ipc] initial command batch rejected: {error}");
                            }
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
                            // Print READY line AFTER viewer is initialized
                            println!("FORGE3D_VIEWER_READY port={}", ipc_port);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                            eprintln!("[viewer-ipc] READY message sent, port={}", ipc_port);
                            // Kick off render loop so IPC commands can be processed
                            window.request_redraw();
                        }
                        Err(e) => {
                            let message = format!("Failed to create viewer: {e}");
                            eprintln!("[viewer-ipc] FATAL: {message}");
                            record_event_loop_fatal(&fatal_for_loop, message);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if matches!(event, WindowEvent::ScaleFactorChanged { .. }) {
                    pending_scale_factor_resize = true;
                }
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } if key_event.state == ElementState::Pressed => {
                                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                                    elwt.exit();
                                }
                            }
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                // Poll global IPC queue for commands
                let mut has_pending_snapshot = false;
                if let Some(viewer) = viewer_opt.as_mut() {
                    if let Ok(mut q) = get_ipc_queue().lock() {
                        let mut queued: Vec<QueuedIpcCommand> = q.drain(..).collect();
                        drop(q);
                        let commands = queued
                            .iter()
                            .map(|item| item.cmd.clone())
                            .collect::<Vec<_>>();
                        if let Err(error) = viewer.preflight_command_batch(&commands) {
                            eprintln!("[viewer-ipc] command batch rejected: {error}");
                            for item in queued {
                                let _ = item.completion.send(Err(error.clone()));
                            }
                            window.request_redraw();
                            return;
                        }
                        queued.sort_by_key(|item| command_priority(&item.cmd));
                        for item in queued {
                            match item.cmd {
                                ViewerCmd::Quit => {
                                    if viewer.snapshot_request.is_some() {
                                        viewer.update(0.0);
                                        let _ = viewer.render();
                                    }
                                    let _ = item.completion.send(Ok(()));
                                    elwt.exit();
                                    return;
                                }
                                other => {
                                    let outcome = viewer.handle_cmd(other);
                                    let _ = item.completion.send(outcome);
                                }
                            }
                        }
                    }
                    has_pending_snapshot = viewer.snapshot_request.is_some();
                }
                window.request_redraw();
                // If snapshot is pending, keep requesting redraws until it's captured
                if has_pending_snapshot {
                    window.request_redraw();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if pending_scale_factor_resize {
                        let size = viewer.window.inner_size();
                        if size.width != viewer.config.width || size.height != viewer.config.height
                        {
                            viewer.resize(size);
                        }
                        pending_scale_factor_resize = false;
                    }
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    viewer.update(dt);
                    match viewer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            viewer.resize(viewer.window.inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            eprintln!("Surface timeout!");
                            std::thread::sleep(Duration::from_millis(4));
                            window.request_redraw();
                        }
                    }
                }
            }
            _ => {}
        }
    })?;

    propagate_event_loop_fatal(fatal_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_loop_fatal_is_returned_to_the_binary() {
        let fatal: EventLoopFatal = Arc::new(Mutex::new(None));
        record_event_loop_fatal(&fatal, "adapter rejected".to_string());
        let error = propagate_event_loop_fatal(fatal).expect_err("fatal state must be returned");
        assert!(error.to_string().contains("adapter rejected"));
    }

    #[test]
    fn clean_event_loop_exit_remains_successful() {
        let fatal: EventLoopFatal = Arc::new(Mutex::new(None));
        propagate_event_loop_fatal(fatal).expect("clean exit must remain successful");
    }

    #[test]
    fn frame_establishing_commands_precede_content_and_keep_stable_order() {
        let commands = vec![
            ViewerCmd::AddLabel {
                id: None,
                text: "before".to_string(),
                world_pos: [6_378_137.0, 0.0, 0.0],
                size: None,
                color: None,
                halo_color: None,
                halo_width: None,
                priority: None,
                min_zoom: None,
                max_zoom: None,
                offset: None,
                rotation: None,
                underline: None,
                small_caps: None,
                leader: None,
                horizon_fade_angle: None,
            },
            ViewerCmd::SetCamLookAt {
                eye: [6_378_137.0, 100.0, 100.0],
                target: [6_378_137.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            },
            ViewerCmd::LoadTerrain("terrain.tif".to_string()),
            ViewerCmd::AddLabel {
                id: None,
                text: "after".to_string(),
                world_pos: [6_378_138.0, 0.0, 0.0],
                size: None,
                color: None,
                halo_color: None,
                halo_width: None,
                priority: None,
                min_zoom: None,
                max_zoom: None,
                offset: None,
                rotation: None,
                underline: None,
                small_caps: None,
                leader: None,
                horizon_fade_angle: None,
            },
        ];
        let ordered = order_command_batch(commands);
        assert!(matches!(ordered[0], ViewerCmd::LoadTerrain(_)));
        assert!(matches!(ordered[1], ViewerCmd::SetCamLookAt { .. }));
        assert!(matches!(
            &ordered[2],
            ViewerCmd::AddLabel { text, .. } if text == "before"
        ));
        assert!(matches!(
            &ordered[3],
            ViewerCmd::AddLabel { text, .. } if text == "after"
        ));
    }
}
