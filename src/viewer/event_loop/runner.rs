// src/viewer/event_loop/runner.rs
// Event loop runner functions for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::sync::Arc;
use std::time::Instant;

use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use super::super::ipc;
use super::super::viewer_enums::ViewerCmd;
use super::super::ViewerConfig;
use super::super::Viewer;
use super::super::INITIAL_CMDS;
use super::{get_ipc_queue, get_ipc_stats, parse_initial_commands, spawn_stdin_reader, update_ipc_stats};

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
        parse_initial_commands(cmds)
    } else {
        Vec::new()
    };

    // Spawn stdin reader thread
    spawn_stdin_reader(proxy);

    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                // Initialize viewer on resume (required for some platforms)
                if viewer_opt.is_none() {
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
                            for cmd in pending_cmds.drain(..) {
                                if let Some(viewer) = viewer_opt.as_mut() {
                                    viewer.handle_cmd(cmd);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to create viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) =
                                        key_event.physical_key
                                    {
                                        elwt.exit();
                                    }
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
                        viewer.handle_cmd(other);
                    } else {
                        eprintln!("[IPC] Viewer not ready, dropping command");
                    }
                }
            },
            _ => {}
        }
    });

    Ok(())
}

/// Run the viewer with an IPC server for non-blocking Python control.
/// Prints `FORGE3D_VIEWER_READY port=<PORT>` when the server is listening.
pub fn run_viewer_with_ipc(
    config: ViewerConfig,
    ipc_config: ipc::IpcServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
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
            if let Ok(mut q) = get_ipc_queue().lock() {
                q.push_back(cmd);
                Ok(())
            } else {
                Err("Queue lock failed".to_string())
            }
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
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                config.width as f64,
                config.height as f64,
            ))
            .build(&event_loop)?,
    );

    // Collect initial commands
    let mut pending_cmds: Vec<ViewerCmd> = if let Some(cmds) = INITIAL_CMDS.get() {
        parse_initial_commands(cmds)
    } else {
        Vec::new()
    };

    // Viewer state
    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();

    let _ = event_loop.run(move |event, elwt| {
        // ControlFlow::Poll for IPC mode - responsive command handling
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
        
        match event {
            Event::Resumed => {
                if viewer_opt.is_none() {
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(mut v) => {
                            for cmd in pending_cmds.drain(..) {
                                v.handle_cmd(cmd);
                            }
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
                            // Print READY line AFTER viewer is initialized
                            println!("FORGE3D_VIEWER_READY port={}", ipc_port);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                            // Kick off render loop so IPC commands can be processed
                            window.request_redraw();
                        }
                        Err(e) => {
                            eprintln!("Failed to create viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) =
                                        key_event.physical_key
                                    {
                                        elwt.exit();
                                    }
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
                if let Some(viewer) = viewer_opt.as_mut() {
                    if let Ok(mut q) = get_ipc_queue().lock() {
                        while let Some(cmd) = q.pop_front() {
                            match cmd {
                                ViewerCmd::Quit => {
                                    if viewer.snapshot_request.is_some() {
                                        viewer.update(0.0);
                                        let _ = viewer.render();
                                    }
                                    elwt.exit();
                                    return;
                                }
                                other => {
                                    viewer.handle_cmd(other);
                                }
                            }
                        }
                    }
                }
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
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
                        }
                    }
                }
            }
            _ => {}
        }
    });

    Ok(())
}
