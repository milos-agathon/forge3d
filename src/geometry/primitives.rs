// src/geometry/primitives.rs
// Procedural unit primitives for Forge3D geometry workflows
// Exists to supply reusable base meshes for geometry and IO pipelines
// RELEVANT FILES:src/geometry/mod.rs,tests/test_f9_primitives.py,examples/f9_primitives_demo.py,src/geometry/validate.rs

use std::f32::consts::TAU;

use glam::Vec3;

use super::MeshBuffers;

/// Supported primitive kinds for phase 1 implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Plane,
    Box,
    Sphere,
    Cylinder,
    Cone,
    Torus,
    TextStub,
}

/// Shared tessellation parameters for primitives.
#[derive(Debug, Clone, Copy)]
pub struct PrimitiveParams {
    pub resolution: (u32, u32),
    pub radial_segments: u32,
    pub rings: u32,
    pub height_segments: u32,
    pub tube_segments: u32,
    pub radius: f32,
    pub tube_radius: f32,
    pub include_caps: bool,
}

impl Default for PrimitiveParams {
    fn default() -> Self {
        Self {
            resolution: (1, 1),
            radial_segments: 32,
            rings: 16,
            height_segments: 1,
            tube_segments: 12,
            radius: 0.5,
            tube_radius: 0.15,
            include_caps: true,
        }
    }
}

/// Generate a primitive mesh using the provided type and parameters.
pub fn generate_primitive(kind: PrimitiveType, params: PrimitiveParams) -> MeshBuffers {
    match kind {
        PrimitiveType::Plane => generate_plane(params.resolution.0, params.resolution.1),
        PrimitiveType::Box => generate_unit_box(),
        PrimitiveType::Sphere => {
            generate_sphere(params.rings, params.radial_segments, params.radius)
        }
        PrimitiveType::Cylinder => generate_cylinder(
            params.radial_segments,
            params.height_segments,
            params.radius,
            params.include_caps,
        ),
        PrimitiveType::Cone => generate_cone(
            params.radial_segments,
            params.height_segments,
            params.radius,
            params.include_caps,
        ),
        PrimitiveType::Torus => generate_torus(
            params.radial_segments,
            params.tube_segments,
            params.radius.min(0.35),
            params.tube_radius.min(0.15),
        ),
        PrimitiveType::TextStub => generate_text3d_stub(),
    }
}

/// Generate a unit plane in the XY plane centered at the origin.
pub fn generate_plane(segments_x: u32, segments_y: u32) -> MeshBuffers {
    let sx = segments_x.max(1);
    let sy = segments_y.max(1);
    let vertex_count = ((sx + 1) * (sy + 1)) as usize;
    let index_count = (sx * sy * 6) as usize;

    let mut mesh = MeshBuffers::with_capacity(vertex_count, index_count);

    for y in 0..=sy {
        let v = y as f32 / sy as f32;
        let pos_y = (v - 0.5) as f32;
        for x in 0..=sx {
            let u = x as f32 / sx as f32;
            let pos_x = (u - 0.5) as f32;
            mesh.positions.push([pos_x, pos_y, 0.0]);
            mesh.normals.push([0.0, 0.0, 1.0]);
            mesh.uvs.push([u, 1.0 - v]);
        }
    }

    for y in 0..sy {
        for x in 0..sx {
            let row_start = y * (sx + 1);
            let a = row_start + x;
            let b = a + 1;
            let c = a + sx + 1;
            let d = c + 1;
            mesh.indices
                .extend_from_slice(&[a as u32, d as u32, c as u32, a as u32, b as u32, d as u32]);
        }
    }

    mesh
}

/// Generate a unit cube with per-face normals and UVs.
pub fn generate_unit_box() -> MeshBuffers {
    let mut mesh = MeshBuffers::with_capacity(24, 36);

    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.5, -0.5, -0.5)), // +X
        (-Vec3::X, Vec3::Y, -Vec3::Z, Vec3::new(-0.5, -0.5, 0.5)), // -X
        (Vec3::Y, Vec3::Z, Vec3::X, Vec3::new(-0.5, 0.5, -0.5)), // +Y
        (-Vec3::Y, Vec3::Z, -Vec3::X, Vec3::new(-0.5, -0.5, 0.5)), // -Y
        (Vec3::Z, Vec3::Y, -Vec3::X, Vec3::new(-0.5, -0.5, 0.5)), // +Z
        (-Vec3::Z, Vec3::Y, Vec3::X, Vec3::new(0.5, -0.5, -0.5)), // -Z
    ];

    for &(normal_axis, up_axis, right_axis, origin) in &faces {
        let normal = normal_axis.normalize();
        let right = right_axis.normalize();
        let up = up_axis.normalize();
        let corners = [origin, origin + right, origin + right + up, origin + up];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let base = mesh.vertex_count() as u32;
        for (corner, uv) in corners.iter().zip(uv.iter()) {
            mesh.positions.push([corner.x, corner.y, corner.z]);
            mesh.normals.push([normal.x, normal.y, normal.z]);
            mesh.uvs.push(*uv);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    mesh
}

/// Generate a UV sphere centered at the origin.
pub fn generate_sphere(rings: u32, segments: u32, radius: f32) -> MeshBuffers {
    let rings = rings.max(2);
    let segments = segments.max(3);
    let mut mesh = MeshBuffers::with_capacity(
        ((rings + 1) * (segments + 1)) as usize,
        (rings * segments * 6) as usize,
    );

    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        let y = (theta.cos() * radius).clamp(-radius, radius);
        let r = theta.sin() * radius;
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let phi = u * TAU;
            let x = r * phi.cos();
            let z = r * phi.sin();
            let normal = Vec3::new(x, y, z).normalize_or_zero();
            mesh.positions.push([x, y, z]);
            mesh.normals.push([normal.x, normal.y, normal.z]);
            mesh.uvs.push([u, 1.0 - v]);
        }
    }

    for ring in 0..rings {
        for seg in 0..segments {
            let a = ring * (segments + 1) + seg;
            let b = a + segments + 1;
            mesh.indices.extend_from_slice(&[
                a as u32,
                (a + 1) as u32,
                b as u32,
                (a + 1) as u32,
                (b + 1) as u32,
                b as u32,
            ]);
        }
    }

    mesh
}

/// Generate a cylinder along the Y axis with optional caps.
pub fn generate_cylinder(
    radial_segments: u32,
    height_segments: u32,
    radius: f32,
    include_caps: bool,
) -> MeshBuffers {
    let radial_segments = radial_segments.max(3);
    let height_segments = height_segments.max(1);

    let side_vertex_count = ((height_segments + 1) * (radial_segments + 1)) as usize;
    let cap_vertex_count = if include_caps {
        (radial_segments + 2) as usize * 2
    } else {
        0
    };
    let mut mesh = MeshBuffers::with_capacity(
        side_vertex_count + cap_vertex_count,
        (height_segments * radial_segments * 6) as usize
            + if include_caps {
                radial_segments as usize * 6
            } else {
                0
            },
    );

    for h in 0..=height_segments {
        let v = h as f32 / height_segments as f32;
        let y = (v - 0.5) * 1.0;
        for seg in 0..=radial_segments {
            let u = seg as f32 / radial_segments as f32;
            let phi = u * TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let x = radius * cos_phi;
            let z = radius * sin_phi;
            mesh.positions.push([x, y, z]);
            mesh.normals.push([cos_phi, 0.0, sin_phi]);
            mesh.uvs.push([u, 1.0 - v]);
        }
    }

    for h in 0..height_segments {
        for seg in 0..radial_segments {
            let a = h * (radial_segments + 1) + seg;
            let b = a + radial_segments + 1;
            mesh.indices.extend_from_slice(&[
                a as u32,
                (a + 1) as u32,
                b as u32,
                (a + 1) as u32,
                (b + 1) as u32,
                b as u32,
            ]);
        }
    }

    if include_caps {
        add_cylinder_cap(&mut mesh, radial_segments, radius, 0.5, true);
        add_cylinder_cap(&mut mesh, radial_segments, radius, -0.5, false);
    }

    mesh
}

fn add_cylinder_cap(mesh: &mut MeshBuffers, radial_segments: u32, radius: f32, y: f32, top: bool) {
    let start = mesh.vertex_count() as u32;
    mesh.positions.push([0.0, y, 0.0]);
    mesh.normals.push([0.0, if top { 1.0 } else { -1.0 }, 0.0]);
    mesh.uvs.push([0.5, 0.5]);

    for seg in 0..=radial_segments {
        let u = seg as f32 / radial_segments as f32;
        let phi = u * TAU;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let x = radius * cos_phi;
        let z = radius * sin_phi;
        mesh.positions.push([x, y, z]);
        mesh.normals.push([0.0, if top { 1.0 } else { -1.0 }, 0.0]);
        mesh.uvs.push([0.5 + cos_phi * 0.5, 0.5 + sin_phi * 0.5]);
    }

    for seg in 0..radial_segments {
        if top {
            mesh.indices
                .extend_from_slice(&[start, start + seg + 1, start + seg + 2]);
        } else {
            mesh.indices
                .extend_from_slice(&[start, start + seg + 2, start + seg + 1]);
        }
    }
}

/// Generate a cone along the Y axis with optional base cap.
pub fn generate_cone(
    radial_segments: u32,
    height_segments: u32,
    radius: f32,
    include_caps: bool,
) -> MeshBuffers {
    let radial_segments = radial_segments.max(3);
    let height_segments = height_segments.max(1);

    let mut mesh = MeshBuffers::with_capacity(
        ((height_segments + 1) * (radial_segments + 1)) as usize
            + if include_caps {
                (radial_segments + 2) as usize
            } else {
                0
            },
        (height_segments * radial_segments * 6) as usize
            + if include_caps {
                radial_segments as usize * 3
            } else {
                0
            },
    );

    let slope = radius / 1.0;
    for h in 0..=height_segments {
        let v = h as f32 / height_segments as f32;
        let y = (v - 0.5) * 1.0;
        let ring_radius = radius * (1.0 - v);
        for seg in 0..=radial_segments {
            let u = seg as f32 / radial_segments as f32;
            let phi = u * TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let x = ring_radius * cos_phi;
            let z = ring_radius * sin_phi;
            let normal = Vec3::new(cos_phi, slope, sin_phi).normalize();
            mesh.positions.push([x, y, z]);
            mesh.normals.push([normal.x, normal.y, normal.z]);
            mesh.uvs.push([u, 1.0 - v]);
        }
    }

    for h in 0..height_segments {
        for seg in 0..radial_segments {
            let a = h * (radial_segments + 1) + seg;
            let b = a + radial_segments + 1;
            mesh.indices.extend_from_slice(&[
                a as u32,
                (a + 1) as u32,
                b as u32,
                (a + 1) as u32,
                (b + 1) as u32,
                b as u32,
            ]);
        }
    }

    if include_caps {
        add_cylinder_cap(&mut mesh, radial_segments, radius, -0.5, false);
    }

    mesh
}

/// Generate a torus centered at the origin.
pub fn generate_torus(
    major_segments: u32,
    minor_segments: u32,
    major_radius: f32,
    minor_radius: f32,
) -> MeshBuffers {
    let major_segments = major_segments.max(3);
    let minor_segments = minor_segments.max(3);
    let mut mesh = MeshBuffers::with_capacity(
        ((major_segments + 1) * (minor_segments + 1)) as usize,
        (major_segments * minor_segments * 6) as usize,
    );

    for i in 0..=major_segments {
        let u = i as f32 / major_segments as f32;
        let phi = u * TAU;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let center = Vec3::new(major_radius * cos_phi, 0.0, major_radius * sin_phi);

        for j in 0..=minor_segments {
            let v = j as f32 / minor_segments as f32;
            let theta = v * TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let offset = Vec3::new(
                minor_radius * cos_theta * cos_phi,
                minor_radius * sin_theta,
                minor_radius * cos_theta * sin_phi,
            );
            let pos = center + offset;
            let normal = offset.normalize_or_zero();
            mesh.positions.push([pos.x, pos.y, pos.z]);
            mesh.normals.push([normal.x, normal.y, normal.z]);
            mesh.uvs.push([u, v]);
        }
    }

    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = i * (minor_segments + 1) + j;
            let b = a + minor_segments + 1;
            mesh.indices.extend_from_slice(&[
                a as u32,
                (a + 1) as u32,
                b as u32,
                (a + 1) as u32,
                (b + 1) as u32,
                b as u32,
            ]);
        }
    }

    mesh
}

/// Fallback mesh for text extrusion until font pipeline is implemented.
pub fn generate_text3d_stub() -> MeshBuffers {
    MeshBuffers::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_dimensions() {
        let mesh = generate_plane(2, 2);
        assert_eq!(mesh.vertex_count(), 9);
        assert_eq!(mesh.triangle_count(), 8);
    }

    #[test]
    fn box_has_expected_counts() {
        let mesh = generate_unit_box();
        assert_eq!(mesh.vertex_count(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn torus_not_empty() {
        let mesh = generate_torus(8, 6, 0.35, 0.15);
        assert!(!mesh.is_empty());
    }
}
