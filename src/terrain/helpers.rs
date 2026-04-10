use super::*;

// ---------- Geometry (analytic spike) ----------

// T33-BEGIN:build-grid-xyuv
/// Minimal grid that matches T3.1/T3.3 vertex layout: interleaved [x, z, u, v] (Float32x4) => 16-byte stride.
pub(super) fn build_grid_xyuv(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = n.max(2) as usize;
    let (w, h) = (n, n); // base grid resolution (without skirts)

    // Domain: [-1.5, +1.5] in X and Z; we feed (x,z) into position.xy.
    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);

    // Expanded grid with one-vertex skirt ring around the base grid
    let ew = w + 2; // expanded width
    let eh = h + 2; // expanded height

    // Interleaved verts: [x, z, u, v]
    let mut verts = Vec::<f32>::with_capacity(ew * eh * 4);
    for j in 0..eh {
        for i in 0..ew {
            // Convert expanded indices to base grid relative index (can be -1..w)
            let bi = i as isize - 1;
            let bj = j as isize - 1;

            // Position extends one step beyond base domain on each side
            let x = -scale + (bi as f32) * step_x;
            let z = -scale + (bj as f32) * step_z;

            // UVs are in [0,1] for interior; outside ring goes slightly beyond [0,1]
            let u = if bi < 0 {
                -1.0 / (w as f32 - 1.0)
            } else if bi >= w as isize {
                1.0 + 1.0 / (w as f32 - 1.0)
            } else {
                (bi as f32) / (w as f32 - 1.0)
            };
            let v = if bj < 0 {
                -1.0 / (h as f32 - 1.0)
            } else if bj >= h as isize {
                1.0 + 1.0 / (h as f32 - 1.0)
            } else {
                (bj as f32) / (h as f32 - 1.0)
            };

            verts.extend_from_slice(&[x, z, u, v]);
        }
    }

    // Indexed triangles (CCW) over expanded grid
    let mut idx = Vec::<u32>::with_capacity((ew - 1) * (eh - 1) * 6);
    for j in 0..eh - 1 {
        for i in 0..ew - 1 {
            let a = (j * ew + i) as u32;
            let b = (j * ew + i + 1) as u32;
            let c = ((j + 1) * ew + i) as u32;
            let d = ((j + 1) * ew + i + 1) as u32;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    use wgpu::util::DeviceExt;
    let v_usage = wgpu::BufferUsages::VERTEX;
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-xyuv-vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: v_usage,
    });
    let i_usage = wgpu::BufferUsages::INDEX;
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-xyuv-ibuf"),
        contents: bytemuck::cast_slice(&idx),
        usage: i_usage,
    });

    // B15: Track buffer allocations (not host-visible)
    let tracker = global_tracker();
    let vbuf_size = (verts.len() * std::mem::size_of::<f32>()) as u64;
    let ibuf_size = (idx.len() * std::mem::size_of::<u32>()) as u64;
    tracker.track_buffer_allocation(vbuf_size, is_host_visible_usage(v_usage));
    tracker.track_buffer_allocation(ibuf_size, is_host_visible_usage(i_usage));
    (vbuf, ibuf, idx.len() as u32)
}
// T33-END:build-grid-xyuv

// MVP + light
pub(super) fn build_view_matrices(width: u32, height: u32) -> (glam::Mat4, glam::Mat4, glam::Vec3) {
    let aspect = width as f32 / height as f32;
    let proj = crate::camera::perspective_wgpu(45f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(3.0, 2.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let light = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
    (view, proj, light)
}

// A2-END:terrain-module

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_terrain_uniforms_layout() {
        // Verify TerrainUniforms struct is exactly 176 bytes as expected by WGSL shader
        assert_eq!(
            size_of::<TerrainUniforms>(),
            176,
            "TerrainUniforms size must be 176 bytes to match WGSL binding"
        );

        // Verify 16-byte alignment for std140 compatibility
        assert_eq!(
            align_of::<TerrainUniforms>(),
            16,
            "TerrainUniforms must be 16-byte aligned for std140 compatibility"
        );
    }

    #[test]
    fn test_default_proj_is_wgpu_clip() {
        // Verify that build_view_matrices uses WGPU clip space projection
        let (w, h) = (512, 384);
        let aspect = w as f32 / h as f32;
        let fovy_deg = 45.0_f32;
        let fovy_rad = fovy_deg.to_radians();
        let (znear, zfar) = (0.1, 100.0);

        let (_, proj, _) = build_view_matrices(w, h);
        let expected = crate::camera::perspective_wgpu(fovy_rad, aspect, znear, zfar);

        // Assert all 16 elements are approximately equal
        let proj_array = proj.to_cols_array();
        let expected_array = expected.to_cols_array();

        for (i, (&actual, &expected)) in proj_array.iter().zip(expected_array.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Element {} differs: actual={}, expected={}, diff={}",
                i,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
    }
}
