// src/viewer/ipc.rs
// IPC protocol for non-blocking viewer control via TCP + NDJSON
// Supports Journey 1 (open populated -> interact -> Python updates -> snapshot)
// and Journey 2 (open blank -> build from Python -> snapshot)

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

use super::viewer_enums::ViewerCmd;

/// IPC request envelope (NDJSON format)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum IpcRequest {
    /// Get viewer stats (geometry readiness, vertex/index counts)
    GetStats,
    /// Load an OBJ file
    LoadObj { path: String },
    /// Load a glTF/GLB file
    LoadGltf { path: String },
    /// Set object transform
    SetTransform {
        #[serde(default)]
        translation: Option<[f32; 3]>,
        #[serde(default)]
        rotation_quat: Option<[f32; 4]>,
        #[serde(default)]
        scale: Option<[f32; 3]>,
    },
    /// Set camera look-at
    CamLookat {
        eye: [f32; 3],
        target: [f32; 3],
        #[serde(default = "default_up")]
        up: [f32; 3],
    },
    /// Set field of view
    SetFov { deg: f32 },
    /// Set sun lighting (azimuth/elevation)
    LitSun {
        azimuth_deg: f32,
        elevation_deg: f32,
    },
    /// Set IBL (environment map)
    LitIbl {
        path: String,
        #[serde(default = "default_intensity")]
        intensity: f32,
    },
    /// Set terrain z-scale (height exaggeration)
    SetZScale { value: f32 },
    /// Take a snapshot
    Snapshot {
        path: String,
        #[serde(default)]
        width: Option<u32>,
        #[serde(default)]
        height: Option<u32>,
    },
    /// Close the viewer
    Close,
    /// Load terrain DEM file for interactive viewing
    LoadTerrain { path: String },
    /// Set terrain camera parameters
    SetTerrainCamera {
        #[serde(default = "default_phi")]
        phi_deg: f32,
        #[serde(default = "default_theta")]
        theta_deg: f32,
        #[serde(default = "default_radius")]
        radius: f32,
        #[serde(default = "default_fov")]
        fov_deg: f32,
    },
    /// Set terrain sun parameters
    SetTerrainSun {
        #[serde(default = "default_sun_azimuth")]
        azimuth_deg: f32,
        #[serde(default = "default_sun_elevation")]
        elevation_deg: f32,
        #[serde(default = "default_sun_intensity")]
        intensity: f32,
    },
    /// Set multiple terrain parameters at once (like rayshader::render_camera)
    SetTerrain {
        #[serde(default)]
        phi: Option<f32>,
        #[serde(default)]
        theta: Option<f32>,
        #[serde(default)]
        radius: Option<f32>,
        #[serde(default)]
        fov: Option<f32>,
        #[serde(default)]
        sun_azimuth: Option<f32>,
        #[serde(default)]
        sun_elevation: Option<f32>,
        #[serde(default)]
        sun_intensity: Option<f32>,
        #[serde(default)]
        ambient: Option<f32>,
        #[serde(default)]
        zscale: Option<f32>,
        #[serde(default)]
        shadow: Option<f32>,
        #[serde(default)]
        background: Option<[f32; 3]>,
        #[serde(default)]
        water_level: Option<f32>,
        #[serde(default)]
        water_color: Option<[f32; 3]>,
    },
    /// Get current terrain parameters
    GetTerrainParams,
}

/// Stats about the currently loaded scene
#[derive(Debug, Clone, Serialize, Default)]
pub struct ViewerStats {
    /// Whether vertex buffer is ready for drawing
    pub vb_ready: bool,
    /// Number of vertices in the current mesh
    pub vertex_count: u32,
    /// Number of indices in the current mesh
    pub index_count: u32,
    /// Whether the scene has any mesh loaded
    pub scene_has_mesh: bool,
    /// Monotonically increasing transform version (incremented on each set_transform)
    pub transform_version: u64,
    /// Whether the current transform is identity (no translation, rotation, or scale applied)
    pub transform_is_identity: bool,
}

fn default_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn default_intensity() -> f32 {
    1.0
}

fn default_phi() -> f32 {
    135.0
}

fn default_theta() -> f32 {
    45.0
}

fn default_radius() -> f32 {
    1000.0
}

fn default_fov() -> f32 {
    55.0
}

fn default_sun_azimuth() -> f32 {
    135.0
}

fn default_sun_elevation() -> f32 {
    35.0
}

fn default_sun_intensity() -> f32 {
    3.0
}

/// IPC response envelope
#[derive(Debug, Clone, Serialize)]
pub struct IpcResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ViewerStats>,
}

impl IpcResponse {
    pub fn success() -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(msg.into()),
            stats: None,
        }
    }

    pub fn with_stats(stats: ViewerStats) -> Self {
        Self {
            ok: true,
            error: None,
            stats: Some(stats),
        }
    }
}

/// Parse an IPC request from a JSON line
pub fn parse_ipc_request(line: &str) -> Result<IpcRequest, String> {
    serde_json::from_str(line).map_err(|e| format!("JSON parse error: {}", e))
}

/// Convert an IpcRequest to a ViewerCmd (if possible)
/// Returns None for requests that are handled specially (like GetStats)
pub fn ipc_request_to_viewer_cmd(req: &IpcRequest) -> Result<Option<ViewerCmd>, String> {
    match req {
        IpcRequest::GetStats => Ok(None), // Handled specially, not a ViewerCmd
        IpcRequest::LoadObj { path } => Ok(Some(ViewerCmd::LoadObj(path.clone()))),
        IpcRequest::LoadGltf { path } => Ok(Some(ViewerCmd::LoadGltf(path.clone()))),
        IpcRequest::SetTransform {
            translation,
            rotation_quat,
            scale,
        } => Ok(Some(ViewerCmd::SetTransform {
            translation: *translation,
            rotation_quat: *rotation_quat,
            scale: *scale,
        })),
        IpcRequest::CamLookat { eye, target, up } => Ok(Some(ViewerCmd::SetCamLookAt {
            eye: *eye,
            target: *target,
            up: *up,
        })),
        IpcRequest::SetFov { deg } => Ok(Some(ViewerCmd::SetFov(*deg))),
        IpcRequest::LitSun {
            azimuth_deg,
            elevation_deg,
        } => Ok(Some(ViewerCmd::SetSunDirection {
            azimuth_deg: *azimuth_deg,
            elevation_deg: *elevation_deg,
        })),
        IpcRequest::LitIbl { path, intensity } => Ok(Some(ViewerCmd::SetIbl {
            path: path.clone(),
            intensity: *intensity,
        })),
        IpcRequest::SetZScale { value } => Ok(Some(ViewerCmd::SetZScale(*value))),
        IpcRequest::Snapshot { path, width, height } => Ok(Some(ViewerCmd::SnapshotWithSize {
            path: path.clone(),
            width: *width,
            height: *height,
        })),
        IpcRequest::Close => Ok(Some(ViewerCmd::Quit)),
        IpcRequest::LoadTerrain { path } => Ok(Some(ViewerCmd::LoadTerrain(path.clone()))),
        IpcRequest::SetTerrainCamera {
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
        } => Ok(Some(ViewerCmd::SetTerrainCamera {
            phi_deg: *phi_deg,
            theta_deg: *theta_deg,
            radius: *radius,
            fov_deg: *fov_deg,
        })),
        IpcRequest::SetTerrainSun {
            azimuth_deg,
            elevation_deg,
            intensity,
        } => Ok(Some(ViewerCmd::SetTerrainSun {
            azimuth_deg: *azimuth_deg,
            elevation_deg: *elevation_deg,
            intensity: *intensity,
        })),
        IpcRequest::SetTerrain {
            phi, theta, radius, fov,
            sun_azimuth, sun_elevation, sun_intensity,
            ambient, zscale, shadow, background, water_level, water_color,
        } => Ok(Some(ViewerCmd::SetTerrain {
            phi: *phi, theta: *theta, radius: *radius, fov: *fov,
            sun_azimuth: *sun_azimuth, sun_elevation: *sun_elevation, sun_intensity: *sun_intensity,
            ambient: *ambient, zscale: *zscale, shadow: *shadow, background: *background,
            water_level: *water_level, water_color: *water_color,
        })),
        IpcRequest::GetTerrainParams => Ok(Some(ViewerCmd::GetTerrainParams)),
    }
}

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
    // Set timeouts to prevent blocking forever
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(300)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
    
    let mut reader = BufReader::new(stream.try_clone().expect("Failed to clone stream"));
    let mut writer = stream;

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF - client closed connection
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let response = match parse_ipc_request(trimmed) {
                    Ok(req) => {
                        // Handle GetStats specially - it returns data directly
                        if matches!(req, IpcRequest::GetStats) {
                            IpcResponse::with_stats(stats_getter())
                        } else {
                            match ipc_request_to_viewer_cmd(&req) {
                                Ok(Some(cmd)) => match cmd_sender(cmd) {
                                    Ok(()) => IpcResponse::success(),
                                    Err(e) => IpcResponse::error(e),
                                },
                                Ok(None) => {
                                    // Should not happen - GetStats is handled above
                                    IpcResponse::error("Internal error: unhandled special request")
                                }
                                Err(e) => IpcResponse::error(e),
                            }
                        }
                    }
                    Err(e) => IpcResponse::error(e),
                };

                let response_json =
                    serde_json::to_string(&response).unwrap_or_else(|_| r#"{"ok":false}"#.to_string());
                if let Err(e) = writeln!(writer, "{}", response_json) {
                    eprintln!("[IPC] Write error: {}", e);
                    break;
                }
                if let Err(e) = writer.flush() {
                    eprintln!("[IPC] Flush error: {}", e);
                    break;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock || 
                         e.kind() == std::io::ErrorKind::TimedOut => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_load_obj() {
        let json = r#"{"cmd":"load_obj","path":"model.obj"}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::LoadObj { path } => assert_eq!(path, "model.obj"),
            _ => panic!("Expected LoadObj"),
        }
    }

    #[test]
    fn test_parse_cam_lookat() {
        let json = r#"{"cmd":"cam_lookat","eye":[0,5,10],"target":[0,0,0],"up":[0,1,0]}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::CamLookat { eye, target, up } => {
                assert_eq!(eye, [0.0, 5.0, 10.0]);
                assert_eq!(target, [0.0, 0.0, 0.0]);
                assert_eq!(up, [0.0, 1.0, 0.0]);
            }
            _ => panic!("Expected CamLookat"),
        }
    }

    #[test]
    fn test_parse_snapshot_with_size() {
        let json = r#"{"cmd":"snapshot","path":"out.png","width":3840,"height":2160}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::Snapshot { path, width, height } => {
                assert_eq!(path, "out.png");
                assert_eq!(width, Some(3840));
                assert_eq!(height, Some(2160));
            }
            _ => panic!("Expected Snapshot"),
        }
    }

    #[test]
    fn test_parse_set_z_scale() {
        let json = r#"{"cmd":"set_z_scale","value":2.5}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::SetZScale { value } => assert_eq!(value, 2.5),
            _ => panic!("Expected SetZScale"),
        }
    }

    #[test]
    fn test_parse_close() {
        let json = r#"{"cmd":"close"}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::Close => {}
            _ => panic!("Expected Close"),
        }
    }

    #[test]
    fn test_parse_get_stats() {
        let json = r#"{"cmd":"get_stats"}"#;
        let req = parse_ipc_request(json).unwrap();
        assert!(matches!(req, IpcRequest::GetStats));
    }

    #[test]
    fn test_ipc_response_success() {
        let resp = IpcResponse::success();
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, r#"{"ok":true}"#);
    }

    #[test]
    fn test_ipc_response_error() {
        let resp = IpcResponse::error("something went wrong");
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, r#"{"ok":false,"error":"something went wrong"}"#);
    }

    #[test]
    fn test_ipc_response_with_stats() {
        let stats = ViewerStats {
            vb_ready: true,
            vertex_count: 100,
            index_count: 300,
            scene_has_mesh: true,
        };
        let resp = IpcResponse::with_stats(stats);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true"#));
        assert!(json.contains(r#""vb_ready":true"#));
        assert!(json.contains(r#""vertex_count":100"#));
        assert!(json.contains(r#""index_count":300"#));
        assert!(json.contains(r#""scene_has_mesh":true"#));
    }
}
