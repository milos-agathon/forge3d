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
use crate::core::memory_tracker::global_tracker;
use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue, Texture, TextureFormat};

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
    pub height_texture: Texture,
    pub minmax_texture: Texture,
    pub mip_count: u32,
    pub cell_w: u32,
    pub cell_h: u32,
    pub h_min: f32,
    pub h_max: f32,
    pub byte_size: u64,
    width: u32,
    height: u32,
}

impl TerrainMinMaxPyramid {
    /// Upload the DEM (R32Float, 1 mip) and its min-max pyramid (RG32Float,
    /// full chain) built from the same heightfield. Both allocations are
    /// registered with the global memory tracker under the
    /// `hybrid-pt-terrain-minmax` labels.
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

        let height_texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
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
        global_tracker().track_texture_allocation(w, h, TextureFormat::R32Float);

        let minmax_texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
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
            global_tracker().track_texture_allocation(*lw, *lh, TextureFormat::Rg32Float);
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
        })
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

    fn free_tracked(&self) {
        global_tracker().free_texture_allocation(self.width, self.height, TextureFormat::R32Float);
        let (mut lw, mut lh) = (
            self.cell_w.next_power_of_two(),
            self.cell_h.next_power_of_two(),
        );
        for _ in 0..self.mip_count {
            global_tracker().free_texture_allocation(lw, lh, TextureFormat::Rg32Float);
            lw = (lw / 2).max(1);
            lh = (lh / 2).max(1);
        }
    }
}

/// Frees the tracker entries on every exit path (success, `?`, or panic) so
/// error returns cannot pollute the global memory metrics.
impl Drop for TerrainMinMaxPyramid {
    fn drop(&mut self) {
        self.free_tracked();
    }
}

/// Complete GPU terrain scene for the hybrid tracer: the min-max pyramid plus
/// the environment map and the shading constants the kernel needs. This is
/// the seam `HybridPathTracer::render` accepts to make terrain a first-class
/// primitive alongside mesh/SDF geometry.
pub struct TerrainPtScene {
    pub pyramid: TerrainMinMaxPyramid,
    pub env_texture: Texture,
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
        let env_texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
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
        global_tracker().track_texture_allocation(env_w, env_h, TextureFormat::Rgba32Float);

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

impl Drop for TerrainPtScene {
    fn drop(&mut self) {
        let (ew, eh) = self.env_tracked;
        global_tracker().free_texture_allocation(ew, eh, TextureFormat::Rgba32Float);
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
}
