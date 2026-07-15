//! P5: Point cloud rendering for interactive viewer

mod load;
mod shader;
mod state;
mod types;

pub(crate) use load::preflight_laz_bounds;
pub use state::PointCloudState;
pub use types::{ColorMode, PointCloudUniforms, PointInstance3D, PointSource3D};
