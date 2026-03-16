use super::super::*;

#[pyfunction]
#[pyo3(signature = (
    width=1024, height=768,
    title="forge3d Interactive Viewer".to_string(),
    vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0,
    obj_path=None, gltf_path=None,
    snapshot_path=None,
    snapshot_width=None, snapshot_height=None,
    initial_commands=None,
))]
pub(crate) fn open_viewer(
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
    obj_path: Option<String>,
    gltf_path: Option<String>,
    snapshot_path: Option<String>,
    snapshot_width: Option<u32>,
    snapshot_height: Option<u32>,
    initial_commands: Option<Vec<String>>,
) -> PyResult<()> {
    use crate::viewer::{run_viewer, set_initial_commands, ViewerConfig};

    // Argument validation mirroring the Python wrapper
    if obj_path.is_some() && gltf_path.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obj_path and gltf_path are mutually exclusive; provide at most one",
        ));
    }

    match (snapshot_width, snapshot_height) {
        (Some(w), Some(h)) => {
            if w == 0 || h == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "snapshot_width and snapshot_height, if provided, must be positive",
                ));
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot_width and snapshot_height must be provided together",
            ));
        }
        (None, None) => {}
    }

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
        snapshot_width,
        snapshot_height,
    };

    // Map Python-level configuration into the existing INITIAL_CMDS mechanism so that
    // object loading and snapshots are expressed as viewer commands. This preserves
    // the single-terminal command workflow and keeps all behavior flowing through
    // the ViewerCmd parsing logic in src/viewer/mod.rs.
    let mut cmds: Vec<String> = Vec::new();

    if let Some(path) = obj_path {
        cmds.push(format!(":obj {}", path));
    }
    if let Some(path) = gltf_path {
        cmds.push(format!(":gltf {}", path));
    }
    if let Some(path) = snapshot_path {
        cmds.push(format!(":snapshot {}", path));
    }
    if let Some(extra) = initial_commands {
        // Append extra commands in order, unaltered, as if the user had typed
        // them on stdin.
        cmds.extend(extra);
    }

    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

    run_viewer(config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Viewer error: {}", e)))
}

/// Open an interactive terrain viewer with a preconfigured RendererConfig.
///
/// This mirrors `open_viewer` but takes a `RendererConfig` describing the terrain
/// scene (DEM, HDR/IBL, colormap, material controls, etc.). The viewer setup
/// (window size, FOV, znear/zfar, snapshot options, initial commands) is identical
/// to `open_viewer` so the Python wrapper can share validation logic.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    cfg,
    width=1024, height=768,
    title="forge3d Terrain Interactive Viewer".to_string(),
    vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0,
    snapshot_path=None,
    snapshot_width=None, snapshot_height=None,
    initial_commands=None,
))]
pub(crate) fn open_terrain_viewer(
    cfg: PyObject,
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
    snapshot_path: Option<String>,
    snapshot_width: Option<u32>,
    snapshot_height: Option<u32>,
    initial_commands: Option<Vec<String>>,
) -> PyResult<()> {
    use crate::viewer::{
        run_viewer, set_initial_commands, set_initial_terrain_config, ViewerConfig,
    };

    // cfg is currently unused on the Rust side; keep it to preserve the Python API shape.
    let _ = cfg;

    // Argument validation mirrors open_viewer
    match (snapshot_width, snapshot_height) {
        (Some(w), Some(h)) => {
            if w == 0 || h == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "snapshot_width and snapshot_height, if provided, must be positive",
                ));
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot_width and snapshot_height must be provided together",
            ));
        }
        (None, None) => {}
    }

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
        snapshot_width,
        snapshot_height,
    };

    // Map terrain-specific options into INITIAL_CMDS: snapshot and any extra
    // commands (e.g., GI/fog toggles) are expressed as viewer commands.
    let mut cmds: Vec<String> = Vec::new();
    if let Some(path) = snapshot_path {
        cmds.push(format!(":snapshot {}", path));
    }
    if let Some(extra) = initial_commands {
        cmds.extend(extra);
    }
    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

    // Stash the terrain configuration so the viewer can attach a TerrainScene when
    // it is first constructed inside run_viewer.
    set_initial_terrain_config(RendererConfig::default());

    run_viewer(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Terrain viewer error: {}", e))
    })
}
