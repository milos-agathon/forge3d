use super::*;

impl Scene {
    /// Apply lightweight runtime post-fx on CPU readback so Scene-level toggles
    /// have observable output impact in offscreen renders.
    pub(super) fn apply_runtime_postfx_cpu(&self, pixels: &mut [u8]) {
        let expected = self.width as usize * self.height as usize * 4;
        if pixels.len() != expected || expected == 0 {
            return;
        }

        let width = self.width as usize;
        let height = self.height as usize;

        if self.ssgi_enabled {
            Self::apply_ssgi_cpu(pixels, &self.ssgi_settings);
        }
        if self.ssr_enabled {
            Self::apply_ssr_cpu(pixels, width, height, &self.ssr_settings);
        }
        if self.bloom_enabled && self.bloom_config.enabled {
            Self::apply_bloom_cpu(pixels, width, height, &self.bloom_config);
        }
    }

    fn apply_ssgi_cpu(pixels: &mut [u8], settings: &crate::lighting::screen_space::SSGISettings) {
        let intensity = (settings.intensity / 8.0).clamp(0.0, 1.0);
        let step_factor = (settings.ray_steps as f32 / 96.0).clamp(0.2, 1.0);
        let radius_factor = (settings.ray_radius / 16.0).clamp(0.1, 1.0);
        let ibl = settings.ibl_fallback.clamp(0.0, 1.0);
        let gi_gain = 0.24 * intensity * step_factor * radius_factor;

        for px in pixels.chunks_exact_mut(4) {
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;
            let luma = (r + g + b) / 3.0;
            let lift = (1.0 - luma) * gi_gain;

            let rr = (r + lift * (1.0 - 0.15 * ibl)).clamp(0.0, 1.0);
            let gg = (g + lift).clamp(0.0, 1.0);
            let bb = (b + lift * (0.9 + 0.1 * ibl)).clamp(0.0, 1.0);

            px[0] = (rr * 255.0) as u8;
            px[1] = (gg * 255.0) as u8;
            px[2] = (bb * 255.0) as u8;
        }
    }

    fn apply_ssr_cpu(
        pixels: &mut [u8],
        width: usize,
        height: usize,
        settings: &crate::lighting::screen_space::SSRSettings,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        let source = pixels.to_vec();
        let base_mix = (settings.intensity / 8.0).clamp(0.0, 1.0) * 0.45;
        let roughness_fade = settings.roughness_fade.clamp(0.0, 1.0);
        let edge_fade = settings.edge_fade.clamp(0.0, 1.0);
        let h1 = height.saturating_sub(1).max(1) as f32;

        for y in 0..height {
            let y_norm = y as f32 / h1;
            let edge_weight = y_norm.powf(1.0 + edge_fade * 2.0);
            let mix = base_mix * edge_weight * (1.0 - roughness_fade * 0.5);
            if mix <= 0.0 {
                continue;
            }
            let mirror_y = height - 1 - y;
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let ridx = (mirror_y * width + x) * 4;

                let cr = source[idx] as f32 / 255.0;
                let cg = source[idx + 1] as f32 / 255.0;
                let cb = source[idx + 2] as f32 / 255.0;

                let rr = source[ridx] as f32 / 255.0;
                let rg = source[ridx + 1] as f32 / 255.0;
                let rb = source[ridx + 2] as f32 / 255.0;

                let out_r = (cr * (1.0 - mix) + rr * mix * 0.85).clamp(0.0, 1.0);
                let out_g = (cg * (1.0 - mix) + rg * mix * 0.90).clamp(0.0, 1.0);
                let out_b = (cb * (1.0 - mix) + rb * mix).clamp(0.0, 1.0);

                pixels[idx] = (out_r * 255.0) as u8;
                pixels[idx + 1] = (out_g * 255.0) as u8;
                pixels[idx + 2] = (out_b * 255.0) as u8;
                pixels[idx + 3] = source[idx + 3];
            }
        }
    }

    fn apply_bloom_cpu(
        pixels: &mut [u8],
        width: usize,
        height: usize,
        config: &crate::core::bloom::BloomConfig,
    ) {
        if width == 0 || height == 0 {
            return;
        }

        let count = width * height;
        let source = pixels.to_vec();
        let mut bright = vec![0.0_f32; count * 3];
        let mut tmp = vec![0.0_f32; count * 3];
        let mut blurred = vec![0.0_f32; count * 3];

        let threshold = config.threshold.max(0.0);
        let softness = config.softness.clamp(0.0, 1.0);
        let knee = (threshold * softness).max(1e-5);
        for i in 0..count {
            let p = i * 4;
            let b = i * 3;
            let r = source[p] as f32 / 255.0;
            let g = source[p + 1] as f32 / 255.0;
            let bl = source[p + 2] as f32 / 255.0;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
            let brightness_factor = if luma < threshold - knee {
                0.0
            } else if luma < threshold + knee {
                let t = ((luma - threshold + knee) / (2.0 * knee)).clamp(0.0, 1.0);
                t * t
            } else {
                1.0
            };
            if brightness_factor > 0.0 {
                bright[b] = r * brightness_factor;
                bright[b + 1] = g * brightness_factor;
                bright[b + 2] = bl * brightness_factor;
            }
        }

        let radius = config.radius.round().clamp(1.0, 6.0) as i32;

        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0_f32; 3];
                let mut wsum = 0.0_f32;
                for dx in -radius..=radius {
                    let sx = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
                    let w = 1.0 / (1.0 + dx.unsigned_abs() as f32);
                    let sidx = (y * width + sx) * 3;
                    sum[0] += bright[sidx] * w;
                    sum[1] += bright[sidx + 1] * w;
                    sum[2] += bright[sidx + 2] * w;
                    wsum += w;
                }
                let didx = (y * width + x) * 3;
                tmp[didx] = sum[0] / wsum;
                tmp[didx + 1] = sum[1] / wsum;
                tmp[didx + 2] = sum[2] / wsum;
            }
        }

        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0_f32; 3];
                let mut wsum = 0.0_f32;
                for dy in -radius..=radius {
                    let sy = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
                    let w = 1.0 / (1.0 + dy.unsigned_abs() as f32);
                    let sidx = (sy * width + x) * 3;
                    sum[0] += tmp[sidx] * w;
                    sum[1] += tmp[sidx + 1] * w;
                    sum[2] += tmp[sidx + 2] * w;
                    wsum += w;
                }
                let didx = (y * width + x) * 3;
                blurred[didx] = sum[0] / wsum;
                blurred[didx + 1] = sum[1] / wsum;
                blurred[didx + 2] = sum[2] / wsum;
            }
        }

        let strength = config.strength.clamp(0.0, 4.0) * (0.2 + 0.8 * softness);
        for i in 0..count {
            let p = i * 4;
            let b = i * 3;
            let r = source[p] as f32 / 255.0;
            let g = source[p + 1] as f32 / 255.0;
            let bl = source[p + 2] as f32 / 255.0;

            let rr = (r + blurred[b] * strength).clamp(0.0, 1.0);
            let gg = (g + blurred[b + 1] * strength).clamp(0.0, 1.0);
            let bb = (bl + blurred[b + 2] * strength).clamp(0.0, 1.0);

            pixels[p] = (rr * 255.0) as u8;
            pixels[p + 1] = (gg * 255.0) as u8;
            pixels[p + 2] = (bb * 255.0) as u8;
            pixels[p + 3] = source[p + 3];
        }
    }
}
