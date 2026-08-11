// src/path_tracing/hybrid_compute/terrain_heightfield.rs
// PROMETHEUS: 2.5D min-max acceleration structure over a DEM heightfield for
// the hybrid path tracer. Each texel of mip L stores (min_height, max_height)
// over its 2x2 children in mip L-1; mip 0 stores the min/max of the four
// corner samples of each bilinear DEM cell. Built on the CPU from the same
// heightfield that feeds the rasterizer (the in-tree HZB producer in
// hzb_build.wgsl is shader-only and not integrated, so `from_heightfield` is
// the supported constructor), uploaded once as an RG32Float mip chain.
// RELEVANT FILES: src/shaders/hybrid_terrain_traversal.wgsl, src/path_tracing/hybrid_compute/render.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue, TextureFormat};

/// World -> texel transform and traversal constants consumed by
/// hybrid_terrain_traversal.wgsl. Deliberately packed as six vec4 rows
/// (96 bytes) so the WGSL uniform layout is alignment-trivial:
///   row 0 origin_spacing: origin_x, origin_z (world xz of DEM texel (0,0)),
///                         spacing_x, spacing_z (world units per texel)
///   row 1 h_params:       h_min, h_max (raw DEM range), exaggeration
///                         (world y = height * exaggeration), env intensity
///   row 2 albedo_pad:     terrain albedo rgb, unused
///   row 3 dims:           width_texels, height_texels, cell_w, cell_h
///   row 4 mips:           mip_count, flags (bit0 = terrain enabled),
///                         env_width, env_height (0 = constant env fallback)
///   row 5 extra:          spp (camera samples per frame), Welford window
///                         (frames per convergence window), unused, unused
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TerrainPtUniforms {
    pub origin_spacing: [f32; 4],
    pub h_params: [f32; 4],
    pub albedo_pad: [f32; 4],
    pub dims: [u32; 4],
    pub mips: [u32; 4],
    pub extra: [u32; 4],
}

/// Curvature parameters consumed by the shared terrain traversal. The two
/// explicit pads match WGSL uniform layout (vec2 alignment = 8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EarthCurvatureUniforms {
    pub inv_two_r_prime: f32,
    pub _pad0: f32,
    pub ray_origin_geodetic: [f32; 2],
    pub enabled: u32,
    pub _pad1: u32,
}

impl EarthCurvatureUniforms {
    pub fn new(
        earth: crate::geo::refraction::EarthModel,
        refraction: crate::geo::refraction::RefractionModel,
        ray_origin_geodetic: [f64; 2],
        azimuth_deg: f64,
    ) -> Result<Self, RenderError> {
        if !ray_origin_geodetic[0].is_finite()
            || !(-90.0..=90.0).contains(&ray_origin_geodetic[0])
            || !ray_origin_geodetic[1].is_finite()
            || !(-180.0..=180.0).contains(&ray_origin_geodetic[1])
        {
            return Err(RenderError::render(
                "ray-origin latitude/longitude must be finite and in [-90,90]/[-180,180]",
            ));
        }
        let effective_radius =
            crate::geo::refraction::effective_radius_m(earth, refraction, azimuth_deg)
                .map_err(RenderError::render)?;
        let enabled = effective_radius.is_finite();
        Ok(Self {
            inv_two_r_prime: if enabled {
                (0.5 / effective_radius) as f32
            } else {
                0.0
            },
            _pad0: 0.0,
            ray_origin_geodetic: [ray_origin_geodetic[0] as f32, ray_origin_geodetic[1] as f32],
            enabled: u32::from(enabled),
            _pad1: 0,
        })
    }
}

/// Exact range of y(t) + d(t)^2/(2R') over a node ray span.
#[cfg(test)]
fn curved_ray_height_range(
    origin_y: f64,
    direction_y: f64,
    horizontal_direction_sq: f64,
    t0: f64,
    t1: f64,
    inv_two_r_prime: f64,
) -> (f64, f64) {
    let a = horizontal_direction_sq * inv_two_r_prime;
    let height = |t: f64| origin_y + direction_y * t + a * t * t;
    let y0 = height(t0);
    let y1 = height(t1);
    let mut minimum = y0.min(y1);
    let maximum = y0.max(y1);
    if a > 0.0 {
        let vertex = -direction_y / (2.0 * a);
        if (t0..=t1).contains(&vertex) {
            minimum = minimum.min(height(vertex));
        }
    }
    (minimum, maximum)
}

/// CPU-side min-max mip chain (kept for unit tests and re-upload).
pub struct MinMaxMips {
    /// levels[0] is the finest (per-cell) level; each entry is [min, max].
    /// Levels are padded to power-of-two dims with (+inf, -inf) sentinel
    /// texels so the wgpu floor-division mip chain and the shader's pure
    /// shift-based node->cell math agree exactly; sentinel nodes always fail
    /// the traversal band test.
    pub levels: Vec<Vec<[f32; 2]>>,
    /// Padded (width, height) per level, same order as `levels`.
    pub dims: Vec<(u32, u32)>,
    /// Logical (unpadded) cell counts of level 0.
    pub cell_w: u32,
    pub cell_h: u32,
}

/// Build the min-max cell pyramid on the CPU.
///
/// Level 0 covers the (w-1) x (h-1) bilinear DEM cells (cell (x, y) stores
/// the min/max of its four corner samples, which bounds the bilinear surface
/// over the cell), padded to power-of-two dims. Every coarser mip reduces
/// 2x2 children, so parents always cover all children.
pub fn build_minmax_mips(heights: &[f32], w: u32, h: u32) -> Result<MinMaxMips, RenderError> {
    if w < 2 || h < 2 {
        return Err(RenderError::Upload(format!(
            "terrain heightfield must be at least 2x2 texels, got {w}x{h}"
        )));
    }
    if heights.len() != (w as usize) * (h as usize) {
        return Err(RenderError::Upload(format!(
            "heightfield length {} does not match {w}x{h}",
            heights.len()
        )));
    }
    if heights.iter().any(|v| !v.is_finite()) {
        return Err(RenderError::Upload(
            "terrain heightfield contains non-finite samples".into(),
        ));
    }

    let cw = w - 1;
    let ch = h - 1;
    let pw = cw.next_power_of_two();
    let ph = ch.next_power_of_two();
    const EMPTY: [f32; 2] = [f32::INFINITY, f32::NEG_INFINITY];
    let mut level0 = vec![EMPTY; (pw as usize) * (ph as usize)];
    for y in 0..ch as usize {
        for x in 0..cw as usize {
            let i00 = y * w as usize + x;
            let i10 = i00 + 1;
            let i01 = i00 + w as usize;
            let i11 = i01 + 1;
            let (a, b, c, d) = (heights[i00], heights[i10], heights[i01], heights[i11]);
            level0[y * pw as usize + x] = [a.min(b).min(c).min(d), a.max(b).max(c).max(d)];
        }
    }

    let mut levels = vec![level0];
    let mut dims = vec![(pw, ph)];
    while dims.last().unwrap().0 > 1 || dims.last().unwrap().1 > 1 {
        let (lw, lh) = *dims.last().unwrap();
        let (nw, nh) = ((lw / 2).max(1), (lh / 2).max(1));
        let prev = levels.last().unwrap();
        let mut next = vec![EMPTY; (nw as usize) * (nh as usize)];
        for y in 0..nh as usize {
            for x in 0..nw as usize {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        // Non-square pot dims collapse one axis early; clamp
                        // keeps full coverage in that case.
                        let sx = (2 * x + dx).min(lw as usize - 1);
                        let sy = (2 * y + dy).min(lh as usize - 1);
                        let v = prev[sy * lw as usize + sx];
                        mn = mn.min(v[0]);
                        mx = mx.max(v[1]);
                    }
                }
                next[y * nw as usize + x] = [mn, mx];
            }
        }
        levels.push(next);
        dims.push((nw, nh));
    }

    Ok(MinMaxMips {
        levels,
        dims,
        cell_w: cw,
        cell_h: ch,
    })
}

/// GPU min-max pyramid + the DEM height texture the leaf test samples.
pub struct TerrainMinMaxPyramid {
    pub height_texture: TrackedTexture,
    pub minmax_texture: TrackedTexture,
    pub mip_count: u32,
    pub cell_w: u32,
    pub cell_h: u32,
    pub h_min: f32,
    pub h_max: f32,
    pub byte_size: u64,
    width: u32,
    height: u32,
    cpu_mips: MinMaxMips,
}

impl TerrainMinMaxPyramid {
    /// Upload the DEM (R32Float, 1 mip) and its min-max pyramid (RG32Float,
    /// full chain) built from the same heightfield. Both allocations are
    /// created through `tracked_create_texture`, so the global memory tracker
    /// and allocation ledger record and release them automatically.
    pub fn from_heightfield(
        device: &Device,
        queue: &Queue,
        heights: &[f32],
        w: u32,
        h: u32,
    ) -> Result<Self, RenderError> {
        let mips = build_minmax_mips(heights, w, h)?;
        let (pot_w, pot_h) = mips.dims[0];
        let (cell_w, cell_h) = (mips.cell_w, mips.cell_h);
        let mip_count = mips.levels.len() as u32;
        let h_min = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let h_max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let height_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-height"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &height_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(heights),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let minmax_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-minmax"),
                size: wgpu::Extent3d {
                    width: pot_w,
                    height: pot_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let mut byte_size = (w as u64) * (h as u64) * 4;
        for (level, ((lw, lh), data)) in mips.dims.iter().zip(mips.levels.iter()).enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &minmax_texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(lw * 8),
                    rows_per_image: Some(*lh),
                },
                wgpu::Extent3d {
                    width: *lw,
                    height: *lh,
                    depth_or_array_layers: 1,
                },
            );
            byte_size += (*lw as u64) * (*lh as u64) * 8;
        }
        log::info!(
            "hybrid-pt-terrain-minmax: {}x{} DEM -> {} mips, {:.2} MiB total",
            w,
            h,
            mip_count,
            byte_size as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            height_texture,
            minmax_texture,
            mip_count,
            cell_w,
            cell_h,
            h_min,
            h_max,
            byte_size,
            width: w,
            height: h,
            cpu_mips: mips,
        })
    }

    /// CPU mirror used to derive conservative tile AABBs for raster HZB
    /// culling. This is the same PROMETHEUS pyramid uploaded above, not a
    /// separately constructed terrain min-max hierarchy.
    pub(crate) fn cpu_mips(&self) -> &MinMaxMips {
        &self.cpu_mips
    }

    /// Uniform block for the traversal kernel; terrain is centered on the
    /// world origin: texel (0,0) sits at (-(w-1)/2*sx, -(h-1)/2*sz).
    #[allow(clippy::too_many_arguments)]
    pub fn uniforms(
        &self,
        spacing_x: f32,
        spacing_z: f32,
        exaggeration: f32,
        albedo: [f32; 3],
        env_intensity: f32,
        env_dims: (u32, u32),
        spp: u32,
        welford_window: u32,
    ) -> TerrainPtUniforms {
        let origin_x = -0.5 * (self.width as f32 - 1.0) * spacing_x;
        let origin_z = -0.5 * (self.height as f32 - 1.0) * spacing_z;
        TerrainPtUniforms {
            origin_spacing: [origin_x, origin_z, spacing_x, spacing_z],
            h_params: [self.h_min, self.h_max, exaggeration, env_intensity],
            albedo_pad: [albedo[0], albedo[1], albedo[2], 0.0],
            dims: [self.width, self.height, self.cell_w, self.cell_h],
            mips: [self.mip_count, 1, env_dims.0, env_dims.1],
            extra: [spp.max(1), welford_window.max(2), 0, 0],
        }
    }
}

/// Complete GPU terrain scene for the hybrid tracer: the min-max pyramid plus
/// the environment map and the shading constants the kernel needs. This is
/// the seam `HybridPathTracer::render` accepts to make terrain a first-class
/// primitive alongside mesh/SDF geometry.
pub struct TerrainPtScene {
    pub pyramid: TerrainMinMaxPyramid,
    pub env_texture: TrackedTexture,
    /// (0, 0) selects the constant-white env fallback in the kernel.
    pub env_dims: (u32, u32),
    spacing: (f32, f32),
    exaggeration: f32,
    albedo: [f32; 3],
    env_intensity: f32,
    env_tracked: (u32, u32),
}

impl TerrainPtScene {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        queue: &Queue,
        heights: &[f32],
        dem_width: u32,
        dem_height: u32,
        spacing: (f32, f32),
        exaggeration: f32,
        albedo: [f32; 3],
        env_map: Option<(&[f32], u32, u32)>,
        env_intensity: f32,
    ) -> Result<Self, RenderError> {
        if !(spacing.0.is_finite() && spacing.0 > 0.0 && spacing.1.is_finite() && spacing.1 > 0.0) {
            return Err(RenderError::Upload(format!(
                "terrain spacing must be finite and > 0, got {spacing:?}"
            )));
        }
        if !(exaggeration.is_finite() && exaggeration > 0.0) {
            return Err(RenderError::Upload(
                "terrain exaggeration must be finite and > 0".into(),
            ));
        }
        if albedo.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(RenderError::Upload(
                "terrain albedo must be finite and >= 0".into(),
            ));
        }
        if !(env_intensity.is_finite() && env_intensity >= 0.0) {
            return Err(RenderError::Upload(
                "env intensity must be finite and >= 0".into(),
            ));
        }
        let pyramid =
            TerrainMinMaxPyramid::from_heightfield(device, queue, heights, dem_width, dem_height)?;

        let (env_data, env_w, env_h, env_dims): (Vec<f32>, u32, u32, (u32, u32)) = match env_map {
            Some((data, w, h)) => {
                if w == 0 || h == 0 || data.len() != (w as usize) * (h as usize) * 3 {
                    return Err(RenderError::Upload(
                        "env map dims do not match data length".into(),
                    ));
                }
                if data.iter().any(|v| !v.is_finite()) {
                    return Err(RenderError::Upload(
                        "env map contains non-finite samples".into(),
                    ));
                }
                (data.to_vec(), w, h, (w, h))
            }
            // 1x1 white placeholder; env_dims (0,0) routes the kernel through
            // the constant fallback so both configurations share one code path.
            None => (vec![1.0, 1.0, 1.0], 1, 1, (0, 0)),
        };
        let env_rgba: Vec<f32> = env_data
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 1.0])
            .collect();
        let env_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-env"),
                size: wgpu::Extent3d {
                    width: env_w,
                    height: env_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &env_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&env_rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(env_w * 16),
                rows_per_image: Some(env_h),
            },
            wgpu::Extent3d {
                width: env_w,
                height: env_h,
                depth_or_array_layers: 1,
            },
        );

        Ok(Self {
            pyramid,
            env_texture,
            env_dims,
            spacing,
            exaggeration,
            albedo,
            env_intensity,
            env_tracked: (env_w, env_h),
        })
    }

    /// Total tracked GPU bytes (pyramid mips + DEM texture + env map).
    pub fn byte_size(&self) -> u64 {
        let (ew, eh) = self.env_tracked;
        self.pyramid.byte_size + (ew as u64) * (eh as u64) * 16
    }

    pub fn uniforms(&self, spp: u32, welford_window: u32) -> TerrainPtUniforms {
        self.pyramid.uniforms(
            self.spacing.0,
            self.spacing.1,
            self.exaggeration,
            self.albedo,
            self.env_intensity,
            self.env_dims,
            spp,
            welford_window,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> Vec<f32> {
        (0..w * h)
            .map(|i| (i % w) as f32 * 0.5 + (i / w) as f32 * 0.25)
            .collect()
    }

    #[test]
    fn minmax_invariant_per_node() {
        let mips = build_minmax_mips(&ramp(256, 256), 256, 256).unwrap();
        for level in &mips.levels {
            for v in level {
                // Sentinel padding is (+inf, -inf); real nodes are ordered.
                if v[0].is_finite() || v[1].is_finite() {
                    assert!(v[0] <= v[1], "min must be <= max");
                }
            }
        }
    }

    #[test]
    fn mip_count_and_dims() {
        let mips = build_minmax_mips(&ramp(256, 256), 256, 256).unwrap();
        assert_eq!((mips.cell_w, mips.cell_h), (255, 255));
        assert_eq!(mips.dims[0], (256, 256)); // padded to power of two
        assert_eq!(*mips.dims.last().unwrap(), (1, 1));
        assert_eq!(mips.levels.len(), 9); // 256 -> 128 -> ... -> 1
                                          // Odd, non-square input pads each axis independently.
        let mips = build_minmax_mips(&ramp(100, 37), 100, 37).unwrap();
        assert_eq!((mips.cell_w, mips.cell_h), (99, 36));
        assert_eq!(mips.dims[0], (128, 64));
        assert_eq!(*mips.dims.last().unwrap(), (1, 1));
        assert_eq!(mips.levels.len(), 8); // max(128, 64) -> 8 levels
    }

    #[test]
    fn parent_covers_children() {
        let mips = build_minmax_mips(&ramp(64, 64), 64, 64).unwrap();
        for l in 1..mips.levels.len() {
            let (pw, ph) = mips.dims[l];
            let (cw, ch) = mips.dims[l - 1];
            for y in 0..ph as usize {
                for x in 0..pw as usize {
                    let p = mips.levels[l][y * pw as usize + x];
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            let sx = (2 * x + dx).min(cw as usize - 1);
                            let sy = (2 * y + dy).min(ch as usize - 1);
                            let c = mips.levels[l - 1][sy * cw as usize + sx];
                            assert!(p[0] <= c[0] && p[1] >= c[1], "parent must cover child");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn root_covers_full_range() {
        let heights = ramp(33, 17);
        let mips = build_minmax_mips(&heights, 33, 17).unwrap();
        let root = mips.levels.last().unwrap()[0];
        let mn = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(root[0], mn);
        assert_eq!(root[1], mx);
    }

    #[test]
    fn flat_dem_is_valid() {
        let mips = build_minmax_mips(&vec![5.0; 16 * 16], 16, 16).unwrap();
        for level in &mips.levels {
            for v in level {
                // Real nodes are exactly flat; padding sentinels are
                // (+inf, -inf) and are skipped by the traversal band test.
                if v[0].is_finite() {
                    assert_eq!(v[0], 5.0);
                    assert_eq!(v[1], 5.0);
                }
            }
        }
        // The root must be real (it covers the whole DEM).
        let root = mips.levels.last().unwrap()[0];
        assert_eq!(root, [5.0, 5.0]);
    }

    #[test]
    fn degenerate_dems_error() {
        assert!(build_minmax_mips(&[1.0], 1, 1).is_err());
        assert!(build_minmax_mips(&[f32::NAN; 4], 2, 2).is_err());
        assert!(build_minmax_mips(&[1.0; 5], 2, 2).is_err());
    }

    fn curvature_fixture() -> Vec<f32> {
        (0..256 * 256)
            .map(|i| {
                let x = (i % 256) as f32;
                let y = (i / 256) as f32;
                900.0
                    + 180.0 * (x * 0.071).sin()
                    + 120.0 * (y * 0.047).cos()
                    + 650.0 * (-((x - 150.0).powi(2) + (y - 126.0).powi(2)) / 900.0).exp()
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    struct ProofRay {
        start_x: f64,
        row: usize,
        origin_y: f64,
        horizontal_x: f64,
        direction_y: f64,
        inv_two_r_prime: f64,
    }

    fn cell_hit(heights: &[f32], ray: ProofRay, cell: usize, t0: f64, t1: f64) -> bool {
        let h0 = f64::from(heights[ray.row * 256 + cell]);
        let h1 = f64::from(heights[ray.row * 256 + cell + 1]);
        let terrain_at =
            |t: f64| h0 + (h1 - h0) * (ray.start_x + ray.horizontal_x * t / 500.0 - cell as f64);
        let deviation = |t: f64| {
            ray.origin_y
                + ray.direction_y * t
                + ray.horizontal_x.powi(2) * ray.inv_two_r_prime * t * t
                - terrain_at(t)
        };
        let mut minimum = deviation(t0).min(deviation(t1));
        let a = ray.horizontal_x.powi(2) * ray.inv_two_r_prime;
        let terrain_slope = (h1 - h0) * ray.horizontal_x / 500.0;
        let vertex = -(ray.direction_y - terrain_slope) / (2.0 * a);
        if (t0..=t1).contains(&vertex) {
            minimum = minimum.min(deviation(vertex));
        }
        minimum <= 0.0
    }

    fn brute_row_hit(heights: &[f32], ray: ProofRay) -> bool {
        let positive = ray.horizontal_x > 0.0;
        let first = ray.start_x.floor() as i32;
        for step in 0..255 {
            let cell = if positive { first + step } else { first - step };
            if !(0..255).contains(&cell) {
                break;
            }
            let xa = if positive {
                ray.start_x.max(cell as f64)
            } else {
                ray.start_x.min((cell + 1) as f64)
            };
            let xb = if positive {
                (cell + 1) as f64
            } else {
                cell as f64
            };
            let t0 = ((xa - ray.start_x) * 500.0 / ray.horizontal_x).max(1e-3);
            let t1 = (xb - ray.start_x) * 500.0 / ray.horizontal_x;
            if cell_hit(heights, ray, cell as usize, t0.min(t1), t0.max(t1)) {
                return true;
            }
        }
        false
    }

    fn descent_row_hit(heights: &[f32], mips: &MinMaxMips, ray: ProofRay) -> bool {
        let mut stack = vec![(mips.levels.len() - 1, 0u32)];
        while let Some((level, nx)) = stack.pop() {
            let cx0 = nx << level;
            if cx0 >= 255 {
                continue;
            }
            let cx1 = ((nx + 1) << level).min(255);
            let ta = (f64::from(cx0) - ray.start_x) * 500.0 / ray.horizontal_x;
            let tb = (f64::from(cx1) - ray.start_x) * 500.0 / ray.horizontal_x;
            let t0 = ta.min(tb).max(1e-3);
            let t1 = ta.max(tb);
            if t0 > t1 || t1 < 0.0 {
                continue;
            }
            let ny = (ray.row as u32) >> level;
            let (lw, _) = mips.dims[level];
            let mm = mips.levels[level][(ny * lw + nx) as usize];
            let (ray_min, ray_max) = curved_ray_height_range(
                ray.origin_y,
                ray.direction_y,
                ray.horizontal_x.powi(2),
                t0,
                t1,
                ray.inv_two_r_prime,
            );
            if ray_min > f64::from(mm[1]) || ray_max < f64::from(mm[0]) {
                continue;
            }
            if level == 0 {
                if cell_hit(heights, ray, cx0 as usize, t0, t1) {
                    return true;
                }
            } else {
                stack.push((level - 1, nx * 2));
                stack.push((level - 1, nx * 2 + 1));
            }
        }
        false
    }

    #[test]
    fn curvature_descent_is_conservative() {
        assert_eq!(std::mem::size_of::<EarthCurvatureUniforms>(), 24);
        let distance = 100_000.0f64;
        let inv_two_r = 1.0 / 14_650_000.0;
        let reference_drop = distance * distance * inv_two_r;
        let gpu_drop = (distance as f32).powi(2) * inv_two_r as f32;
        assert!((reference_drop - f64::from(gpu_drop)).abs() < 1.0);

        let heights = curvature_fixture();
        let mips = build_minmax_mips(&heights, 256, 256).unwrap();
        let mut state = 0x4845_4c49u32;
        let mut false_misses = 0usize;
        let mut false_hits = 0usize;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let row = (state as usize % 254) + 1;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let start_x = 1.0 + f64::from(state % 253) + 0.25;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let positive = state & 1 == 0;
            let elevation = (0.1 + f64::from((state >> 1) % 790) / 100.0).to_radians();
            let horizontal_x = elevation.cos() * if positive { 1.0 } else { -1.0 };
            let cell = start_x.floor() as usize;
            let u = start_x - cell as f64;
            let surface = f64::from(heights[row * 256 + cell]) * (1.0 - u)
                + f64::from(heights[row * 256 + cell + 1]) * u;
            let ray = ProofRay {
                start_x,
                row,
                origin_y: surface + 1.7,
                horizontal_x,
                direction_y: elevation.sin(),
                inv_two_r_prime: 1.0 / 14_650_000.0,
            };
            let brute = brute_row_hit(&heights, ray);
            let descent = descent_row_hit(&heights, &mips, ray);
            false_misses += usize::from(brute && !descent);
            false_hits += usize::from(!brute && descent);
        }

        let mut mask_matches = 0usize;
        let mut mask_pixels = 0usize;
        let mut mask_false_misses = 0usize;
        let mut mask_false_hits = 0usize;
        let elevation = 0.6f64.to_radians();
        for row in 0..255 {
            for cell in 0..255 {
                let surface = 0.75 * f64::from(heights[row * 256 + cell])
                    + 0.25 * f64::from(heights[row * 256 + cell + 1]);
                let ray = ProofRay {
                    start_x: cell as f64 + 0.25,
                    row,
                    origin_y: surface + 1.7,
                    horizontal_x: elevation.cos(),
                    direction_y: elevation.sin(),
                    inv_two_r_prime: 1.0 / 14_650_000.0,
                };
                let brute = brute_row_hit(&heights, ray);
                let descent = descent_row_hit(&heights, &mips, ray);
                mask_matches += usize::from(brute == descent);
                mask_false_misses += usize::from(brute && !descent);
                mask_false_hits += usize::from(!brute && descent);
                mask_pixels += 1;
            }
        }
        let false_hit_rate = false_hits as f64 / 10_000.0;
        let agreement = mask_matches as f64 / mask_pixels as f64;
        println!(
            "HELIOS conservative descent: false_misses={false_misses}, \
             false_hit_rate={false_hit_rate:.6}, shadow_mask_agreement={agreement:.6}, \
             mask_false_misses={mask_false_misses}, mask_false_hits={mask_false_hits}"
        );
        assert_eq!(false_misses, 0);
        assert!(false_hit_rate < 0.001);
        assert!(agreement >= 0.999);

        let shader = include_str!("../../shaders/hybrid_terrain_traversal.wgsl");
        assert!(shader.contains("struct EarthCurvatureUniforms"));
        assert!(shader.contains("terrain_curved_height_range"));
        let hybrid = include_str!("../../shaders/hybrid_traversal.wgsl");
        assert!(hybrid.contains("let terrain_hit = terrain_trace(tray, true);"));
    }
}
