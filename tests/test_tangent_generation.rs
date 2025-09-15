// tests/test_tangent_generation.rs
// Rust unit test for tangent orthonormalization helper.
// Exists to validate the simple Gram-Schmidt implementation.
// RELEVANT FILES:src/lib.rs

#[test]
fn tangents_orthonormalize_with_normals() {
    let n = [0.0_f32, 1.0, 0.0];
    let t = [1.0_f32, 0.1, 0.0];
    let (t_ortho, b) = forge3d::math::orthonormalize_tangent(n, t);
    // n dot t_ortho ~= 0
    let ndot = n[0]*t_ortho[0] + n[1]*t_ortho[1] + n[2]*t_ortho[2];
    assert!(ndot.abs() < 1e-5);
    // |t_ortho| ~= 1 and |b| ~= 1
    let t_len = (t_ortho[0]*t_ortho[0] + t_ortho[1]*t_ortho[1] + t_ortho[2]*t_ortho[2]).sqrt();
    let b_len = (b[0]*b[0] + b[1]*b[1] + b[2]*b[2]).sqrt();
    assert!((t_len - 1.0).abs() < 1e-5);
    assert!((b_len - 1.0).abs() < 1e-5);
}

