use crate::util::image_write;
use anyhow::Context;
use glam::Vec3;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const DEFAULT_SCENE_PATH: &str = "reports/p5/p5_ssr_scene.json";
pub const DEFAULT_OUTPUT_NAME: &str = "p5_ssr_glossy_spheres.png";

#[derive(Debug, Clone, Deserialize)]
pub struct SsrScenePreset {
    #[serde(default = "SsrScenePreset::default_render")]
    pub render: RenderPreset,
    #[serde(default = "SsrScenePreset::default_background_top")]
    pub background_top: [f32; 3],
    #[serde(default = "SsrScenePreset::default_background_bottom")]
    pub background_bottom: [f32; 3],
    #[serde(default = "SsrScenePreset::default_floor_horizon")]
    pub floor_horizon: f32,
    #[serde(default = "SsrScenePreset::default_env_tint")]
    pub env_tint: [f32; 3],
    #[serde(default = "SsrScenePreset::default_light_dir")]
    pub light_dir: [f32; 3],
    #[serde(default = "SsrScenePreset::default_light_intensity")]
    pub light_intensity: f32,
    #[serde(default = "SsrScenePreset::default_stripe")]
    pub stripe: StripePreset,
    #[serde(default = "SsrScenePreset::default_floor")]
    pub floor: FloorPreset,
    #[serde(default = "SsrScenePreset::default_spheres")]
    pub spheres: Vec<SpherePreset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenderPreset {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StripePreset {
    pub center_y: f32,
    pub half_thickness: f32,
    pub inner_color: [f32; 3],
    pub outer_color: [f32; 3],
    pub glow_strength: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FloorPreset {
    pub start_y: f32,
    pub color_top: [f32; 3],
    pub color_bottom: [f32; 3],
    pub reflection_strength: f32,
    pub reflection_power: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpherePreset {
    pub offset_x: f32,
    pub center_y: f32,
    pub radius: f32,
    pub roughness: f32,
}

impl SsrScenePreset {
    pub fn load_from(path: &Path) -> anyhow::Result<Self> {
        let txt = std::fs::read_to_string(path)
            .with_context(|| format!("read SSR scene preset {}", path.display()))?;
        Ok(serde_json::from_str(&txt)?)
    }

    pub fn load_or_default(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        match Self::load_from(path.as_ref()) {
            Ok(scene) => Ok(scene),
            Err(err) => {
                eprintln!(
                    "[P5] Falling back to default SSR scene ({}); reason: {err}",
                    path.as_ref().display()
                );
                Ok(Self::default())
            }
        }
    }

    fn default_render() -> RenderPreset {
        RenderPreset {
            width: 1920,
            height: 1080,
        }
    }

    fn default_background_top() -> [f32; 3] {
        [0.04, 0.07, 0.12]
    }

    fn default_background_bottom() -> [f32; 3] {
        [0.24, 0.42, 0.62]
    }

    fn default_floor_horizon() -> f32 {
        0.58
    }

    fn default_env_tint() -> [f32; 3] {
        [0.28, 0.38, 0.55]
    }

    fn default_light_dir() -> [f32; 3] {
        [-0.3, 0.6, 0.74]
    }

    fn default_light_intensity() -> f32 {
        1.0
    }

    fn default_stripe() -> StripePreset {
        StripePreset {
            center_y: 0.32,
            half_thickness: 0.025,
            inner_color: [1.0, 0.94, 0.78],
            outer_color: [0.95, 0.6, 0.35],
            glow_strength: 0.85,
        }
    }

    fn default_floor() -> FloorPreset {
        FloorPreset {
            start_y: 0.63,
            color_top: [0.16, 0.15, 0.13],
            color_bottom: [0.08, 0.08, 0.07],
            reflection_strength: 0.2,
            reflection_power: 3.0,
        }
    }

    fn default_spheres() -> Vec<SpherePreset> {
        let count = 9;
        let base = 0.15;
        let span = 0.7;
        let radius = 0.11;
        let center_y = 0.63;
        (0..count)
            .map(|i| SpherePreset {
                offset_x: base + span * (i as f32 / (count as f32 - 1.0)),
                center_y,
                radius,
                roughness: 0.1 + 0.1 * i as f32,
            })
            .collect()
    }
}

impl Default for SsrScenePreset {
    fn default() -> Self {
        Self {
            render: Self::default_render(),
            background_top: Self::default_background_top(),
            background_bottom: Self::default_background_bottom(),
            floor_horizon: Self::default_floor_horizon(),
            env_tint: Self::default_env_tint(),
            light_dir: Self::default_light_dir(),
            light_intensity: Self::default_light_intensity(),
            stripe: Self::default_stripe(),
            floor: Self::default_floor(),
            spheres: Self::default_spheres(),
        }
    }
}

pub fn default_scene_path() -> PathBuf {
    PathBuf::from(DEFAULT_SCENE_PATH)
}

pub fn write_glossy_png(preset: &SsrScenePreset, output: &Path) -> anyhow::Result<()> {
    let width = preset.render.width.max(1);
    let height = preset.render.height.max(1);
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    let set_pixel = |pixels: &mut [u8], x: u32, y: u32, color: [f32; 3]| {
        if x >= width || y >= height {
            return;
        }
        let idx = ((y * width + x) * 4) as usize;
        pixels[idx] = to_u8(color[0]);
        pixels[idx + 1] = to_u8(color[1]);
        pixels[idx + 2] = to_u8(color[2]);
        pixels[idx + 3] = 255;
    };

    let mut blend_pixel = |pixels: &mut [u8], x: u32, y: u32, color: [f32; 3], alpha: f32| {
        if x >= width || y >= height {
            return;
        }
        let idx = ((y * width + x) * 4) as usize;
        let dst = [
            pixels[idx] as f32 / 255.0,
            pixels[idx + 1] as f32 / 255.0,
            pixels[idx + 2] as f32 / 255.0,
        ];
        let inv = 1.0 - alpha;
        let mixed = [
            dst[0] * inv + color[0] * alpha,
            dst[1] * inv + color[1] * alpha,
            dst[2] * inv + color[2] * alpha,
        ];
        pixels[idx] = to_u8(mixed[0]);
        pixels[idx + 1] = to_u8(mixed[1]);
        pixels[idx + 2] = to_u8(mixed[2]);
    };

    let horizon = (preset.floor_horizon.clamp(0.0, 1.0) * height as f32)
        .round()
        .clamp(0.0, (height - 1) as f32) as u32;

    for y in 0..height {
        let color = if y < horizon {
            let denom = horizon.max(1);
            let t = (y as f32 / denom as f32).powf(0.9).clamp(0.0, 1.0);
            lerp3(preset.background_top, preset.background_bottom, t)
        } else {
            let denom = (height - horizon).max(1);
            let t = ((y - horizon) as f32 / denom as f32).clamp(0.0, 1.0);
            lerp3(preset.floor.color_top, preset.floor.color_bottom, t)
        };
        for x in 0..width {
            set_pixel(&mut pixels, x, y, color);
        }
    }

    let stripe_center = preset.stripe.center_y * height as f32;
    let stripe_half = (preset.stripe.half_thickness * height as f32).max(1.0);
    for y in 0..height {
        let dy = ((y as f32 - stripe_center) / stripe_half).abs();
        if dy < 1.0 {
            let alpha = (1.0 - dy).powf(2.0) * preset.stripe.glow_strength;
            let glow = lerp3(
                preset.stripe.inner_color,
                preset.stripe.outer_color,
                (y as f32 / height as f32).clamp(0.0, 1.0),
            );
            for x in 0..width {
                blend_pixel(&mut pixels, x, y, glow, alpha);
            }
        }
    }

    let env_tint = Vec3::from(preset.env_tint);
    let light_dir = Vec3::from(preset.light_dir);
    let light_intensity = preset.light_intensity.max(0.0);

    for sphere in &preset.spheres {
        let cx = sphere.offset_x.clamp(0.0, 1.0) * width as f32;
        let cy = sphere.center_y * height as f32;
        let radius = (sphere.radius * height as f32).max(1.0);
        draw_sphere(
            &mut pixels,
            width,
            height,
            (cx, cy),
            radius,
            sphere.roughness.min(0.95),
            stripe_center,
            stripe_half,
            env_tint,
            light_dir,
            light_intensity,
        );
    }

    let reflection_plane = (preset.floor.start_y.clamp(0.0, 1.0) * height as f32).round() as i32;
    if reflection_plane >= 0 && reflection_plane < height as i32 {
        for y in reflection_plane.max(0) as u32..height {
            let span = (height as i32 - reflection_plane).max(1) as f32;
            let refl = ((y as i32 - reflection_plane) as f32 / span).clamp(0.0, 1.0);
            let alpha = (1.0 - refl).powf(preset.floor.reflection_power.max(0.1))
                * preset.floor.reflection_strength.clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }
            let mirror_y = (2.0 * reflection_plane as f32 - y as f32).round() as i32;
            if mirror_y < 0 || mirror_y as u32 >= height {
                continue;
            }
            for x in 0..width {
                let idx_src = ((mirror_y as u32 * width + x) * 4) as usize;
                let color = [
                    pixels[idx_src] as f32 / 255.0,
                    pixels[idx_src + 1] as f32 / 255.0,
                    pixels[idx_src + 2] as f32 / 255.0,
                ];
                blend_pixel(&mut pixels, x, y, color, alpha);
            }
        }
    }

    image_write::write_png_rgba8_small(output, &pixels, width, height)?;
    Ok(())
}

fn draw_sphere(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    center: (f32, f32),
    radius: f32,
    roughness: f32,
    stripe_center: f32,
    stripe_half: f32,
    env_tint: Vec3,
    light_dir: Vec3,
    light_intensity: f32,
) {
    let min_x = (center.0 - radius).floor().max(0.0) as u32;
    let max_x = (center.0 + radius).ceil().min(width as f32 - 1.0) as u32;
    let min_y = (center.1 - radius).floor().max(0.0) as u32;
    let max_y = (center.1 + radius).ceil().min(height as f32 - 1.0) as u32;
    let light_dir = light_dir.normalize();
    let stripe_scale = stripe_half.max(1.0);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = (px - center.0) / radius;
            let dy = (py - center.1) / radius;
            let dist2 = dx * dx + dy * dy;
            if dist2 > 1.0 {
                continue;
            }
            let nz = (1.0 - dist2).sqrt();
            let normal = Vec3::new(dx, -dy, nz).normalize();
            let diffuse = normal.dot(light_dir).clamp(0.0, 1.0) * light_intensity;
            let reflectivity = (1.0 - roughness).clamp(0.0, 1.0);
            let highlight_power = normal
                .z
                .clamp(0.0, 1.0)
                .powf(32.0 * (1.0 - roughness) + 4.0);
            let projected_y = py - dy * radius;
            let dist_to_stripe =
                ((projected_y - stripe_center).abs() / (stripe_scale * 0.8)).clamp(0.0, 4.0);
            let stripe_reflection = ((1.0 - dist_to_stripe).max(0.0)).powf(3.0);
            let reflection = reflectivity * stripe_reflection;
            let base = Vec3::splat(0.08) + Vec3::splat(diffuse * 0.6);
            let spec = Vec3::splat(highlight_power * 0.7);
            let stripe_color = Vec3::new(1.0, 0.94, 0.78) * reflection;
            let env = env_tint * (0.25 + normal.z.abs() * 0.5) * reflectivity;
            let final_color = base + spec + stripe_color + env;
            let color = [final_color.x, final_color.y, final_color.z];
            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = to_u8(color[0]);
            pixels[idx + 1] = to_u8(color[1]);
            pixels[idx + 2] = to_u8(color[2]);
            pixels[idx + 3] = 255;
        }
    }
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

pub fn stripe_contrast(
    preset: &SsrScenePreset,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Vec<f32> {
    fn srgb_to_linear(channel: u8) -> f32 {
        let c = channel as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn luminance(px: &[u8]) -> f32 {
        let r = srgb_to_linear(px[0]);
        let g = srgb_to_linear(px[1]);
        let b = srgb_to_linear(px[2]);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    let mut result = Vec::with_capacity(preset.spheres.len());
    for sphere in &preset.spheres {
        let center_x = (sphere.offset_x.clamp(0.0, 1.0)) * width as f32;
        let center_y = sphere.center_y.clamp(0.0, 1.0) * height as f32;
        let radius = (sphere.radius * height as f32).max(1.0);

        let roi_half_w = (radius * 0.25).max(4.0);
        let roi_half_h = (radius * 0.12).max(3.0);
        let roi_center_y = (center_y - radius * 0.25).clamp(0.0, height as f32 - 1.0);

        let mut min_l = f32::MAX;
        let mut max_l = f32::MIN;
        let mut count = 0u32;

        let x0 = (center_x - roi_half_w).floor() as i32;
        let x1 = (center_x + roi_half_w).ceil() as i32;
        let y0 = (roi_center_y - roi_half_h).floor() as i32;
        let y1 = (roi_center_y + roi_half_h).ceil() as i32;

        for y in y0.max(0)..=y1.min(height as i32 - 1) {
            for x in x0.max(0)..=x1.min(width as i32 - 1) {
                let idx = ((y as u32 * width + x as u32) * 4) as usize;
                if idx + 3 >= pixels.len() {
                    continue;
                }
                let luma = luminance(&pixels[idx..idx + 4]);
                min_l = min_l.min(luma);
                max_l = max_l.max(luma);
                count += 1;
            }
        }

        if count == 0 {
            result.push(0.0);
            continue;
        }
        if max_l <= 0.0 {
            result.push(0.0);
            continue;
        }
        let contrast = if max_l + min_l <= 1e-5 {
            0.0
        } else {
            (max_l - min_l) / (max_l + min_l + 1e-5)
        };
        result.push(contrast);
    }
    result
}
