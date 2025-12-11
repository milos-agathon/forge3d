// src/viewer/viewer_ssr_scene.rs
// SSR scene building utilities for the interactive viewer
// RELEVANT FILES: src/p5/ssr.rs

use crate::geometry::{generate_plane, generate_sphere};
use crate::p5::ssr::SsrScenePreset;
use glam::{Mat4, Vec3};

use super::viewer_types::SceneMesh;

/// Build SSR albedo texture from preset
pub fn build_ssr_albedo_texture(preset: &SsrScenePreset, size: u32) -> Vec<u8> {
    let dim = size.max(1);
    let mut pixels = vec![0u8; (dim * dim * 4) as usize];

    let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };

    let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };

    let horizon = (preset.floor_horizon.clamp(0.0, 1.0) * dim as f32)
        .round()
        .clamp(0.0, (dim - 1) as f32) as u32;

    for y in 0..dim {
        let color = if y < horizon {
            let denom = horizon.max(1);
            let t = (y as f32 / denom as f32).powf(0.9).clamp(0.0, 1.0);
            lerp3(preset.background_top, preset.background_bottom, t)
        } else {
            let denom = (dim - horizon).max(1);
            let t = ((y - horizon) as f32 / denom as f32).clamp(0.0, 1.0);
            lerp3(preset.floor.color_top, preset.floor.color_bottom, t)
        };
        for x in 0..dim {
            let idx = ((y * dim + x) * 4) as usize;
            pixels[idx] = to_u8(color[0]);
            pixels[idx + 1] = to_u8(color[1]);
            pixels[idx + 2] = to_u8(color[2]);
            pixels[idx + 3] = 255;
        }
    }

    let stripe_center = preset.stripe.center_y * dim as f32;
    let stripe_half = (preset.stripe.half_thickness * dim as f32).max(1.0);
    for y in 0..dim {
        let dy = ((y as f32 - stripe_center) / stripe_half).abs();
        if dy < 1.0 {
            let alpha = (1.0 - dy).powf(2.0) * preset.stripe.glow_strength;
            let glow = lerp3(
                preset.stripe.inner_color,
                preset.stripe.outer_color,
                (y as f32 / dim as f32).clamp(0.0, 1.0),
            );
            for x in 0..dim {
                let idx = ((y * dim + x) * 4) as usize;
                let dst = [
                    pixels[idx] as f32 / 255.0,
                    pixels[idx + 1] as f32 / 255.0,
                    pixels[idx + 2] as f32 / 255.0,
                ];
                let inv = 1.0 - alpha;
                let mixed = [
                    dst[0] * inv + glow[0] * alpha,
                    dst[1] * inv + glow[1] * alpha,
                    dst[2] * inv + glow[2] * alpha,
                ];
                pixels[idx] = to_u8(mixed[0]);
                pixels[idx + 1] = to_u8(mixed[1]);
                pixels[idx + 2] = to_u8(mixed[2]);
            }
        }
    }

    pixels
}

/// Build SSR scene mesh from preset (floor + spheres)
pub fn build_ssr_scene_mesh(preset: &SsrScenePreset) -> SceneMesh {
    let mut scene = SceneMesh::new();

    // Floor plane
    const FLOOR_WIDTH: f32 = 8.0;
    const FLOOR_DEPTH: f32 = 6.0;
    const FLOOR_Y: f32 = -1.0;
    let floor_mesh = generate_plane(32, 32);
    let floor_transform = Mat4::from_translation(Vec3::new(0.0, FLOOR_Y, 0.0))
        * Mat4::from_scale(Vec3::new(FLOOR_WIDTH * 0.5, 1.0, FLOOR_DEPTH * 0.5));
    scene.extend_with_mesh(&floor_mesh, floor_transform, 0.35, 0.0);

    // No extra back wall geometry; only floor + spheres per M2 visual acceptance

    // Glossy spheres
    const SPHERE_RINGS: u32 = 48;
    const SPHERE_SEGMENTS: u32 = 64;
    for (i, sphere) in preset.spheres.iter().enumerate() {
        let mesh = generate_sphere(SPHERE_RINGS, SPHERE_SEGMENTS, 1.0);
        let x = (sphere.offset_x - 0.5) * 6.0;
        let y = sphere.center_y * 3.0;
        let z = 0.5 + (i as f32) * 0.01;
        let radius = (sphere.radius * 3.0).max(0.05);
        let transform =
            Mat4::from_translation(Vec3::new(x, y, z)) * Mat4::from_scale(Vec3::splat(radius));
        scene.extend_with_mesh(&mesh, transform, sphere.roughness, 0.0);
    }

    scene
}
