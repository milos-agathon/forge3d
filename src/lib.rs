// src/lib.rs
// Rust crate root for forge3d - GPU rendering library with Python bindings
// Provides SDF primitives, CSG operations, hybrid traversal, and path tracing
// RELEVANT FILES:src/sdf/mod.rs,src/path_tracing/mod.rs,python/forge3d/__init__.py

// Core modules
pub mod math {
    /// Orthonormalize a tangent `t` against normal `n` and return (tangent, bitangent).
    ///
    /// Uses simple Gram-Schmidt then computes bitangent as cross(n, t_ortho).
    pub fn orthonormalize_tangent(n: [f32; 3], t: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
        fn norm(v: [f32; 3]) -> f32 { dot(v, v).sqrt() }
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
        fn mul(v: [f32; 3], s: f32) -> [f32; 3] { [v[0]*s, v[1]*s, v[2]*s] }
        fn normalize(v: [f32; 3]) -> [f32; 3] { let l = norm(v); if l > 0.0 { [v[0]/l, v[1]/l, v[2]/l] } else { v } }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]
        }

        let n_n = normalize(n);
        let t_ortho = normalize(sub(t, mul(n_n, dot(n_n, t))));
        let b = cross(n_n, t_ortho);
        (t_ortho, b)
    }
}

// Rendering modules
pub mod accel;
pub mod camera;
pub mod colormap;
pub mod context;
pub mod core;
pub mod device_caps;
pub mod error;
pub mod external_image;
pub mod formats;
pub mod gpu;
pub mod grid;
pub mod loaders;
pub mod mesh;
pub mod path_tracing;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod sdf;  // New SDF module
pub mod terrain;
pub mod terrain_stats;
pub mod textures {}
pub mod transforms;
pub mod vector;

// Re-export commonly used types
pub use error::RenderError;
pub use sdf::{
    SdfScene, SdfSceneBuilder, HybridScene, HybridHitResult,
    SdfPrimitive, SdfPrimitiveType, CsgOperation
};
pub use path_tracing::{TracerParams, TracerEngine};

