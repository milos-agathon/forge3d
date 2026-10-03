//! Target dispatch for forge3d.
//!
//! The native renderer remains the full crate. The browser artifact is a
//! deliberately small, PyO3-free ORBIS streaming bridge that reuses the exact
//! bounded ticket state used by native workers.

#[cfg(not(target_arch = "wasm32"))]
include!("lib_native.rs");

#[cfg(target_arch = "wasm32")]
#[path = "wasm_terrain.rs"]
pub mod terrain;

#[cfg(target_arch = "wasm32")]
pub mod core {
    pub mod error {
        #[derive(Debug)]
        pub enum RenderError {
            Device(String),
            Upload(String),
            Budget(String),
        }

        impl RenderError {
            pub fn device(message: impl Into<String>) -> Self {
                Self::Device(message.into())
            }

            pub fn upload(message: impl Into<String>) -> Self {
                Self::Upload(message.into())
            }

            pub fn budget(message: impl Into<String>) -> Self {
                Self::Budget(message.into())
            }
        }

        impl std::fmt::Display for RenderError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Device(message) => write!(f, "Device error: {message}"),
                    Self::Upload(message) => write!(f, "Upload error: {message}"),
                    Self::Budget(message) => write!(f, "Memory budget exceeded: {message}"),
                }
            }
        }

        impl std::error::Error for RenderError {}

        pub type RenderResult<T> = Result<T, RenderError>;
    }

    #[path = "memory_tracker.rs"]
    pub mod memory_tracker;

    // Browser atlas/page-table resources use the exact native RAII ledger and
    // budget wrappers; wasm is not allowed a parallel no-op allocation path.
    #[path = "resource_tracker.rs"]
    pub mod resource_tracker;
}

#[cfg(target_arch = "wasm32")]
mod wasm_orbis;
