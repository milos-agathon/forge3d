use super::super::*;
use crate::py_functions::*;

mod camera;
mod diagnostics;
mod geometry;
mod interactive;
mod io_import;
mod rendering;

#[cfg(feature = "extension-module")]
pub(crate) fn register_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    interactive::register_interactive_py_functions(m)?;
    geometry::register_geometry_py_functions(m)?;
    io_import::register_io_import_py_functions(m)?;
    diagnostics::register_diagnostics_py_functions(m)?;
    camera::register_camera_py_functions(m)?;
    rendering::register_rendering_py_functions(m)?;
    Ok(())
}
