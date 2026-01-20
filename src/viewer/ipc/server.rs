// src/viewer/ipc_split/server.rs
// TCP server for IPC viewer control

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

use super::protocol::{
    ipc_request_to_viewer_cmd, parse_ipc_request, IpcRequest, IpcResponse, ViewerStats, BundleRequest,
};
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::event_loop::{
    get_pick_events, get_lasso_state,
    take_pending_bundle_save, take_pending_bundle_load,
};

/// IPC server configuration
pub struct IpcServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for IpcServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 0, // Let OS choose a free port
        }
    }
}

/// Result of starting the IPC server
pub struct IpcServerHandle {
    pub port: u16,
    pub shutdown_tx: mpsc::Sender<()>,
}

/// Start the IPC server thread that accepts connections and forwards commands
/// to the viewer via the provided sender.
///
/// Returns the actual port the server is listening on (useful when port=0).
pub fn start_ipc_server<F, G>(
    config: IpcServerConfig,
    cmd_sender: F,
    stats_getter: G,
) -> std::io::Result<IpcServerHandle>
where
    F: Fn(ViewerCmd) -> Result<(), String> + Send + Sync + 'static,
    G: Fn() -> ViewerStats + Send + Sync + 'static,
{
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)?;
    let actual_port = listener.local_addr()?.port();

    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

    // Wrap in Arc for sharing across connections
    let cmd_sender = std::sync::Arc::new(cmd_sender);
    let stats_getter = std::sync::Arc::new(stats_getter);

    thread::spawn(move || {
        // Set non-blocking to allow shutdown check
        listener
            .set_nonblocking(true)
            .expect("Cannot set non-blocking");

        loop {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                break;
            }

            match listener.accept() {
                Ok((stream, _addr)) => {
                    // Handle connection in a new thread
                    let cmd_sender_clone = std::sync::Arc::clone(&cmd_sender);
                    let stats_getter_clone = std::sync::Arc::clone(&stats_getter);
                    handle_ipc_connection(
                        stream,
                        move |cmd| cmd_sender_clone(cmd),
                        move || stats_getter_clone(),
                    );
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No connection yet, sleep briefly
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    eprintln!("[IPC] Accept error: {}", e);
                }
            }
        }
    });

    Ok(IpcServerHandle {
        port: actual_port,
        shutdown_tx,
    })
}

/// Handle a single IPC connection (reads NDJSON, sends responses)
fn handle_ipc_connection<F, G>(stream: TcpStream, cmd_sender: F, stats_getter: G)
where
    F: Fn(ViewerCmd) -> Result<(), String>,
    G: Fn() -> ViewerStats,
{
    // IMPORTANT: The stream may inherit non-blocking mode from the listener.
    // We must set it to blocking mode for reliable read_line behavior with large messages.
    if let Err(e) = stream.set_nonblocking(false) {
        eprintln!("[IPC] Failed to set blocking mode: {}", e);
        return;
    }
    
    // Set timeouts to prevent blocking forever
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(300)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));

    // Use larger buffer for handling large JSON messages (e.g., vector overlays with many vertices)
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, stream.try_clone().expect("Failed to clone stream"));
    let mut writer = stream;

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF - client closed connection
                break;
            }
            Ok(n) => {
                // Debug: log large messages
                if n > 100000 {
                    eprintln!("[IPC] Received large message: {} bytes", n);
                }
                
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let response = match parse_ipc_request(trimmed) {
                    Ok(req) => {
                        // Debug: log command type for large messages
                        if line.len() > 100000 {
                            eprintln!("[IPC] Parsed large request: {:?}", std::mem::discriminant(&req));
                        }
                        
                        // Handle special requests that return data directly
                        match req {
                            IpcRequest::GetStats => IpcResponse::with_stats(stats_getter()),
                            IpcRequest::PollPickEvents => {
                                if let Ok(mut events) = get_pick_events().lock() {
                                    let result = events.clone();
                                    events.clear();
                                    IpcResponse::with_pick_events(result)
                                } else {
                                    IpcResponse::error("Failed to lock pick event queue")
                                }
                            }
                            IpcRequest::GetLassoState => {
                                if let Ok(state) = get_lasso_state().lock() {
                                    IpcResponse::with_lasso_state(state.clone())
                                } else {
                                    IpcResponse::error("Failed to lock lasso state")
                                }
                            }
                            IpcRequest::PollPendingBundleSave => {
                                if let Some((path, name)) = take_pending_bundle_save() {
                                    IpcResponse::with_bundle_request(BundleRequest::save(path, name))
                                } else {
                                    IpcResponse::with_bundle_request(BundleRequest::none())
                                }
                            }
                            IpcRequest::PollPendingBundleLoad => {
                                if let Some(path) = take_pending_bundle_load() {
                                    IpcResponse::with_bundle_request(BundleRequest::load(path))
                                } else {
                                    IpcResponse::with_bundle_request(BundleRequest::none())
                                }
                            }
                            _ => {
                                match ipc_request_to_viewer_cmd(&req) {
                                    Ok(Some(cmd)) => match cmd_sender(cmd) {
                                        Ok(()) => IpcResponse::success(),
                                        Err(e) => {
                                            eprintln!("[IPC] Command error: {}", e);
                                            IpcResponse::error(e)
                                        },
                                    },
                                    Ok(None) => {
                                        IpcResponse::error("Internal error: unhandled special request")
                                    }
                                    Err(e) => {
                                        eprintln!("[IPC] Conversion error: {}", e);
                                        IpcResponse::error(e)
                                    },
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[IPC] Parse error (msg len={}): {}", trimmed.len(), e);
                        IpcResponse::error(e)
                    },
                };

                let response_json = serde_json::to_string(&response)
                    .unwrap_or_else(|_| r#"{"ok":false}"#.to_string());
                if let Err(e) = writeln!(writer, "{}", response_json) {
                    eprintln!("[IPC] Write error: {}", e);
                    break;
                }
                if let Err(e) = writer.flush() {
                    eprintln!("[IPC] Flush error: {}", e);
                    break;
                }
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                // Timeout - continue waiting for more data
                continue;
            }
            Err(e) => {
                eprintln!("[IPC] Read error: {}", e);
                break;
            }
        }
    }
}
