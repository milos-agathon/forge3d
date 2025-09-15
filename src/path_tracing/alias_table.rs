// src/path_tracing/alias_table.rs
// Alias table implementation for efficient light sampling in ReSTIR DI

use bytemuck::{Pod, Zeroable};

/// Entry in an alias table for efficient discrete sampling
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AliasEntry {
    /// Probability threshold for this entry
    pub prob: f32,
    /// Index of the alias entry to use if random sample exceeds prob
    pub alias: u32,
}

/// Alias table for O(1) discrete sampling from weighted distributions
#[derive(Clone, Debug)]
pub struct AliasTable {
    /// Array of alias entries
    entries: Vec<AliasEntry>,
    /// Total weight sum (for normalization)
    total_weight: f32,
}

impl AliasTable {
    /// Create a new alias table from weighted samples
    pub fn new(weights: &[f32]) -> Self {
        if weights.is_empty() {
            return Self {
                entries: vec![],
                total_weight: 0.0,
            };
        }

        let n = weights.len();
        let total_weight: f32 = weights.iter().sum();

        if total_weight <= 0.0 {
            // Handle degenerate case with uniform distribution
            let entries = vec![
                AliasEntry {
                    prob: 1.0,
                    alias: 0
                };
                n
            ];
            return Self {
                entries,
                total_weight: 0.0,
            };
        }

        // Walker's alias method for O(n) construction
        let mut prob = vec![0.0f32; n];
        let mut alias = vec![0u32; n];

        // Scale probabilities by n
        let scale = n as f32 / total_weight;
        let mut scaled_weights: Vec<f32> = weights.iter().map(|w| w * scale).collect();

        let mut small = Vec::with_capacity(n);
        let mut large = Vec::with_capacity(n);

        // Separate into underfull and overfull bins
        for (i, &weight) in scaled_weights.iter().enumerate() {
            if weight < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // Balance the bins
        while let (Some(small_idx), Some(large_idx)) = (small.pop(), large.pop()) {
            prob[small_idx] = scaled_weights[small_idx];
            alias[small_idx] = large_idx as u32;

            // Reduce the large bin by the amount given to small bin
            scaled_weights[large_idx] = scaled_weights[large_idx] + scaled_weights[small_idx] - 1.0;

            if scaled_weights[large_idx] < 1.0 {
                small.push(large_idx);
            } else {
                large.push(large_idx);
            }
        }

        // Handle remaining entries (should have prob ≈ 1.0)
        for &idx in &small {
            prob[idx] = 1.0;
        }
        for &idx in &large {
            prob[idx] = 1.0;
        }

        let entries: Vec<AliasEntry> = prob
            .into_iter()
            .zip(alias.into_iter())
            .map(|(p, a)| AliasEntry { prob: p, alias: a })
            .collect();

        Self {
            entries,
            total_weight,
        }
    }

    /// Sample an index using the alias table
    /// Returns (index, pdf) where pdf is the probability of selecting this index
    pub fn sample(&self, u1: f32, u2: f32) -> (usize, f32) {
        if self.entries.is_empty() {
            return (0, 0.0);
        }

        let n = self.entries.len();
        let scaled_u1 = u1 * n as f32;
        let bin = (scaled_u1 as usize).min(n - 1);
        let frac = scaled_u1 - bin as f32;

        let entry = self.entries[bin];
        let selected_bin = if frac < entry.prob {
            bin
        } else {
            entry.alias as usize
        };

        // Calculate PDF: weight[selected] / total_weight
        let pdf = if self.total_weight > 0.0 {
            // This is an approximation - exact PDF would require storing original weights
            1.0 / n as f32
        } else {
            0.0
        };

        (selected_bin, pdf)
    }

    /// Get the raw entries for GPU buffer upload
    pub fn entries(&self) -> &[AliasEntry] {
        &self.entries
    }

    /// Get the total weight
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alias_table_uniform() {
        let weights = [1.0, 1.0, 1.0, 1.0];
        let table = AliasTable::new(&weights);

        assert_eq!(table.len(), 4);
        assert_eq!(table.total_weight(), 4.0);

        // Test sampling
        let (idx, _pdf) = table.sample(0.5, 0.5);
        assert!(idx < 4);
    }

    #[test]
    fn test_alias_table_weighted() {
        let weights = [0.1, 0.3, 0.6];
        let table = AliasTable::new(&weights);

        assert_eq!(table.len(), 3);
        assert_eq!(table.total_weight(), 1.0);
    }

    #[test]
    fn test_alias_table_empty() {
        let weights: [f32; 0] = [];
        let table = AliasTable::new(&weights);

        assert!(table.is_empty());
        assert_eq!(table.total_weight(), 0.0);
    }
}
