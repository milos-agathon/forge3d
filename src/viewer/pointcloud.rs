//! P5: Point cloud rendering for interactive viewer

mod load;
mod shader;
mod state;
mod types;

pub use state::PointCloudState;
pub use types::{ColorMode, PointCloudUniforms, PointInstance3D};
