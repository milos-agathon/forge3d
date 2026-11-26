//! RG32F uploads carry a finite height and a missing flag. On dense R32F
//! textures the GPU supplies zero for G, which also means valid.
pub(crate) fn masked_texels(data: &[f32]) -> Option<Vec<[f32; 2]>> {
    if data.iter().all(|h| h.is_finite()) {
        return None;
    }
    let floor = data
        .iter()
        .copied()
        .filter(|h| h.is_finite())
        .reduce(f32::min)
        .unwrap_or(0.0);
    Some(
        data.iter()
            .map(|&h| {
                if h.is_finite() {
                    [h, 0.5]
                } else {
                    [floor, 1.0]
                }
            })
            .collect(),
    )
}

/// Partition each axis so odd last rows and columns reach the root.
pub(crate) fn reduce_level(
    data: &[[f32; 2]],
    width: u32,
    height: u32,
    floor: f32,
) -> Vec<[f32; 2]> {
    let (nw, nh) = ((width / 2).max(1), (height / 2).max(1));
    let mut next = Vec::with_capacity(nw as usize * nh as usize);
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0.0f64;
            let mut count = 0;
            let mut total = 0;
            for row in y * height / nh..(y + 1) * height / nh {
                for col in x * width / nw..(x + 1) * width / nw {
                    let [value, absent] = data[(row * width + col) as usize];
                    total += 1;
                    if absent <= 0.5 {
                        sum += f64::from(value);
                        count += 1;
                    }
                }
            }
            next.push(if 2 * count >= total {
                [(sum / f64::from(count)) as f32, 0.5]
            } else {
                [floor, 1.0]
            });
        }
    }
    next
}

/// Bilinear reconstruction with the same minimum covered weight as a COG page.
pub(crate) fn bilinear(h: [f32; 4], tx: f32, ty: f32) -> f32 {
    if h.iter().all(|v| v.is_finite()) {
        let top = h[0] * (1.0 - tx) + h[1] * tx;
        let bottom = h[2] * (1.0 - tx) + h[3] * tx;
        return top * (1.0 - ty) + bottom * ty;
    }
    let weights = [
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    ];
    let mut sum = 0.0;
    let mut weight = 0.0;
    for (v, w) in h.into_iter().zip(weights) {
        if v.is_finite() {
            sum += f64::from(v) * f64::from(w);
            weight += f64::from(w);
        }
    }
    if weight >= 0.5 {
        (sum / weight) as f32
    } else {
        f32::NAN
    }
}

pub(crate) fn sample_height_bilinear(data: &[f32], dims: (u32, u32), uv: [f32; 2]) -> f32 {
    let fx = uv[0].clamp(0.0, 1.0) * dims.0.saturating_sub(1) as f32;
    let fy = uv[1].clamp(0.0, 1.0) * dims.1.saturating_sub(1) as f32;
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(dims.0.saturating_sub(1) as usize);
    let y1 = (y0 + 1).min(dims.1.saturating_sub(1) as usize);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let width = dims.0 as usize;
    crate::terrain::masked_height::bilinear(
        [
            data[y0 * width + x0],
            data[y0 * width + x1],
            data[y1 * width + x0],
            data[y1 * width + x1],
        ],
        tx,
        ty,
    )
}

/// Height levels used to mirror GPU clipmap geometry in CPU diagnostics.
pub(crate) struct MaskedPyramid {
    levels: Vec<((u32, u32), Vec<f32>)>,
    pub(crate) floor: f32,
}

impl MaskedPyramid {
    pub(crate) fn new(data: &[f32], mut dims: (u32, u32)) -> Option<Self> {
        let mut packed = masked_texels(data)?;
        let floor = packed
            .iter()
            .map(|sample| sample[0])
            .reduce(f32::min)
            .unwrap_or(0.0);
        let mut levels = Vec::new();
        loop {
            let values = packed
                .iter()
                .map(|&[h, flag]| if flag > 0.5 { f32::NAN } else { h })
                .collect();
            levels.push((dims, values));
            if dims == (1, 1) {
                break;
            }
            packed = reduce_level(&packed, dims.0, dims.1, floor);
            dims = ((dims.0 / 2).max(1), (dims.1 / 2).max(1));
        }
        Some(Self { levels, floor })
    }

    pub(crate) fn morph(&self, uv: [f32; 2], morph: [f32; 2]) -> f32 {
        let ring = morph[1];
        let (fine, coarse) = if ring < 0.0 {
            ((-ring - 1.0) as usize, (-ring) as usize)
        } else {
            (0, ring as usize + 1)
        };
        let sample = |level: usize| {
            let (dims, data) = &self.levels[level.min(self.levels.len() - 1)];
            sample_height_bilinear(data, *dims, uv)
        };
        let (a, b) = (sample(fine), sample(coarse));
        bilinear([a, b, a, b], morph[0].clamp(0.0, 1.0), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mipmap::{build_mip_chain_rgba32f, MipmapConfig};
    use crate::path_tracing::hybrid_compute::terrain_heightfield::{
        build_minmax_mips, build_traversal_bounds,
    };

    #[test]
    fn vacuum_pyramids_match_independent_masked_references() {
        // A missing quadrant, one isolated hole, and different values in every channel.
        let (w, h) = (8usize, 8usize);
        let valid = |x: usize, y: usize| (x >= 4 || y >= 4) && (x, y) != (6, 6);
        let input: Vec<_> = (0..w * h)
            .map(|i| {
                if valid(i % w, i / w) {
                    i as f32
                } else {
                    f32::NAN
                }
            })
            .collect();
        let mut level = masked_texels(&input).unwrap();
        let rgba: Vec<_> = input
            .iter()
            .flat_map(|v| [*v, 2.0 * v, 128.0 - v, 1.0])
            .collect();
        let colors = build_mip_chain_rgba32f(
            &rgba,
            w as u32,
            h as u32,
            &MipmapConfig {
                gamma_aware: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reference: Vec<Option<f32>> =
            input.iter().map(|&v| v.is_finite().then_some(v)).collect();
        let mut width = w;
        for color in colors {
            assert_eq!(level.len(), reference.len());
            for (i, expected) in reference.iter().enumerate() {
                assert_eq!(level[i][1] <= 0.5, expected.is_some());
                let pixel = &color.data[4 * i..4 * i + 4];
                if let Some(v) = expected {
                    assert_eq!(level[i][0], *v);
                    assert_eq!(pixel, &[*v, 2.0 * v, 128.0 - v, 1.0]);
                } else {
                    assert!(pixel[0].is_nan());
                }
            }
            if width == 1 {
                break;
            }
            let mut next = Vec::new();
            for y in (0..width).step_by(2) {
                for x in (0..width).step_by(2) {
                    let i = y * width + x;
                    let children: Vec<_> = [i, i + 1, i + width, i + width + 1]
                        .into_iter()
                        .filter_map(|j| reference[j])
                        .collect();
                    next.push(if children.len() >= 2 {
                        Some(children.iter().sum::<f32>() / children.len() as f32)
                    } else {
                        None
                    });
                }
            }
            level = reduce_level(&level, width as u32, width as u32, 0.0);
            reference = next;
            width /= 2;
        }

        let bounds = build_minmax_mips(&input, w as u32, h as u32).unwrap();
        let empty = [f32::INFINITY, f32::NEG_INFINITY];
        fn reference_bounds(nodes: &[[f32; 2]]) -> [f32; 2] {
            let covered: Vec<_> = nodes.iter().filter(|n| n[0] <= n[1]).collect();
            if covered.is_empty() || covered.len() * 2 < nodes.len() {
                [f32::INFINITY, f32::NEG_INFINITY]
            } else {
                [
                    covered.iter().map(|n| n[0]).fold(f32::INFINITY, f32::min),
                    covered
                        .iter()
                        .map(|n| n[1])
                        .fold(f32::NEG_INFINITY, f32::max),
                ]
            }
        }
        let mut reference = Vec::new();
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let i = y * w + x;
                let corners = [i, i + 1, i + w, i + w + 1].map(|k| {
                    if input[k].is_finite() {
                        [input[k], input[k]]
                    } else {
                        empty
                    }
                });
                reference.push(reference_bounds(&corners));
            }
        }
        let (mut rw, mut rh) = (w - 1, h - 1);
        for (&(dw, dh), level) in bounds.dims.iter().zip(&bounds.levels) {
            for y in 0..dh as usize {
                for x in 0..dw as usize {
                    let expected = if x < rw && y < rh {
                        reference[y * rw + x]
                    } else {
                        empty
                    };
                    assert_eq!(level[y * dw as usize + x], expected);
                }
            }
            let mut next = Vec::new();
            for y in (0..rh).step_by(2) {
                for x in (0..rw).step_by(2) {
                    let children: Vec<_> = (y..(y + 2).min(rh))
                        .flat_map(|row| &reference[row * rw + x..row * rw + (x + 2).min(rw)])
                        .copied()
                        .collect();
                    next.push(reference_bounds(&children));
                }
            }
            (rw, rh) = (rw.div_ceil(2), rh.div_ceil(2));
            reference = next;
        }
        let bounds = build_traversal_bounds(&input, w as u32, h as u32).unwrap();
        for (lod, (dims, nodes)) in bounds.dims.iter().zip(bounds.levels.iter()).enumerate() {
            let side = 1usize << lod;
            for (i, node) in nodes.iter().enumerate() {
                let x0 = i % dims.0 as usize * side;
                let y0 = i / dims.0 as usize * side;
                let mut expected = [f32::INFINITY, f32::NEG_INFINITY];
                if x0 < w - 1 && y0 < h - 1 {
                    for y in y0..=(y0 + side).min(h - 1) {
                        for x in x0..=(x0 + side).min(w - 1) {
                            if valid(x, y) {
                                let v = (y * w + x) as f32;
                                expected = [expected[0].min(v), expected[1].max(v)];
                            }
                        }
                    }
                }
                assert_eq!(*node, expected);
            }
        }
    }
}
