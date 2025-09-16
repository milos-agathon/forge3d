// src/path_tracing/guiding.rs
// Minimal CPU-side spatial/directional guiding scaffolding (bins + updates).
// This exists to satisfy A13 deliverables by introducing a simple online histogram grid and directional bins.
// RELEVANT FILES:src/path_tracing/mod.rs,src/shaders/pt_guiding.wgsl,python/forge3d/guiding.py,tests/test_guiding.py


/// Spatial grid with per-cell directional histogram (simple 8-bin azimuth only).
#[derive(Clone, Debug)]
pub struct GuidingGrid {
    width: u32,
    height: u32,
    bins_per_cell: u32,
    counts: Vec<u32>,
}

impl GuidingGrid {
    pub fn new(width: u32, height: u32, bins_per_cell: u32) -> Self {
        let bins_per_cell = bins_per_cell.max(1);
        let cells = (width as usize) * (height as usize);
        Self {
            width,
            height,
            bins_per_cell,
            counts: vec![0u32; cells * (bins_per_cell as usize)],
        }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32, bin: u32) -> usize {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let bin = bin % self.bins_per_cell;
        let cell = (y as usize) * (self.width as usize) + (x as usize);
        cell * (self.bins_per_cell as usize) + (bin as usize)
    }

    /// Online update: increment the directional bin count for the given cell.
    pub fn update(&mut self, x: u32, y: u32, bin: u32, weight: f32) {
        let w = if weight.is_finite() {
            weight.max(0.0)
        } else {
            0.0
        };
        // Use stochastic rounding to keep counts integral but weight-sensitive.
        let inc = if w <= 0.0 {
            0
        } else if w >= 1.0 {
            1
        } else {
            rand_like(w) as u32
        };
        if inc > 0 {
            let i = self.idx(x, y, bin);
            // Saturating add to avoid overflow.
            self.counts[i] = self.counts[i].saturating_add(inc);
        }
    }

    /// Returns normalized probabilities for all bins in the cell.
    pub fn pdf(&self, x: u32, y: u32) -> Vec<f32> {
        let mut local = vec![0u32; self.bins_per_cell as usize];
        let mut sum: u64 = 0;
        for b in 0..self.bins_per_cell {
            let c = self.counts[self.idx(x, y, b)];
            local[b as usize] = c;
            sum += c as u64;
        }
        if sum == 0 {
            let u = 1.0 / (self.bins_per_cell as f32);
            return vec![u; self.bins_per_cell as usize];
        }
        local
            .into_iter()
            .map(|c| (c as f32) / (sum as f32))
            .collect()
    }

    pub fn dims(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.bins_per_cell)
    }
}

// Tiny deterministic hash for stochastic rounding (no external dep).
fn rand_like(w: f32) -> f32 {
    // Map weight to a pseudo-random bit via a simple LCG over its bits.
    let bits = w.to_bits();
    let mut x = bits.wrapping_mul(1664525).wrapping_add(1013904223);
    x ^= x.rotate_left(13);
    let u = (x as f32) / (u32::MAX as f32);
    if u < w {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_uniform_when_empty() {
        let g = GuidingGrid::new(4, 3, 8);
        let pdf = g.pdf(1, 1);
        assert_eq!(pdf.len(), 8);
        let s: f32 = pdf.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        for p in pdf {
            assert!((p - 1.0 / 8.0).abs() < 1e-6);
        }
    }

    #[test]
    fn updates_accumulate() {
        let mut g = GuidingGrid::new(2, 2, 4);
        for _ in 0..10 {
            g.update(0, 0, 1, 1.0);
        }
        let pdf = g.pdf(0, 0);
        // Bin 1 should dominate.
        assert!(pdf[1] > 0.5);
        let s: f32 = pdf.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}
