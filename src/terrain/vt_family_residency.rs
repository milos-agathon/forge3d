//! Device-free per-family residency accounting for the terrain material VT.
//!
//! The terrain virtual-texture runtime pages three material families (albedo,
//! normal, mask) through one shared atlas + tile cache. This module owns the
//! CPU-side policy that keeps each family inside its own residency budget:
//! budgets are an even split of the total VT budget across enabled families,
//! and eviction pressure from one family never drains another family's
//! resident set while that family stays under its own budget (within-family
//! LRU evicts first; the shared cache capacity remains the global backstop).
//!
//! Kept free of wgpu/PyO3 so the unit tests run under the curated cargo
//! feature set (which excludes `extension-module`).

use std::collections::VecDeque;

/// Number of terrain VT material families (albedo, normal, mask).
pub(crate) const VT_FAMILY_COUNT: usize = 3;

/// Identity of one virtual-texture tile within a family/material/mip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TileKey {
    pub family_slot: u32,
    pub material_index: u32,
    pub x: u32,
    pub y: u32,
    pub mip_level: u32,
}

/// Residency snapshot for one material family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FamilyResidency {
    pub resident_tiles: u32,
    pub resident_bytes: u64,
    pub budget_bytes: u64,
}

/// Per-family residency budgets and LRU order.
pub(crate) struct FamilyResidencyTracker {
    tile_bytes: u64,
    families: [FamilyResidency; VT_FAMILY_COUNT],
    /// Access order per family; front = least recently used.
    lru: [VecDeque<TileKey>; VT_FAMILY_COUNT],
}

impl FamilyResidencyTracker {
    /// Split `total_budget_bytes` evenly across the families enabled in
    /// `family_mask` (bit `1 << slot`). Disabled families get a zero budget.
    pub fn new(total_budget_bytes: u64, family_mask: u32, tile_bytes: u64) -> Self {
        let enabled = (0..VT_FAMILY_COUNT)
            .filter(|slot| family_mask & (1u32 << slot) != 0)
            .count()
            .max(1) as u64;
        let per_family = total_budget_bytes / enabled;
        let mut families = [FamilyResidency::default(); VT_FAMILY_COUNT];
        for (slot, family) in families.iter_mut().enumerate() {
            if family_mask & (1u32 << slot) != 0 {
                family.budget_bytes = per_family;
            }
        }
        Self {
            tile_bytes: tile_bytes.max(1),
            families,
            lru: Default::default(),
        }
    }

    fn slot_of(key: &TileKey) -> usize {
        key.family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize
    }

    /// Mark a resident tile as most recently used.
    pub fn note_access(&mut self, key: TileKey) {
        let slot = Self::slot_of(&key);
        if let Some(pos) = self.lru[slot].iter().position(|entry| *entry == key) {
            self.lru[slot].remove(pos);
            self.lru[slot].push_back(key);
        }
    }

    /// Record a newly resident tile (most recently used).
    pub fn on_insert(&mut self, key: TileKey) {
        let slot = Self::slot_of(&key);
        if self.lru[slot].iter().any(|entry| *entry == key) {
            self.note_access(key);
            return;
        }
        self.lru[slot].push_back(key);
        self.families[slot].resident_tiles += 1;
        self.families[slot].resident_bytes += self.tile_bytes;
    }

    /// Record an eviction (whether within-family or from the shared cache).
    pub fn on_evict(&mut self, key: &TileKey) {
        let slot = Self::slot_of(key);
        if let Some(pos) = self.lru[slot].iter().position(|entry| entry == key) {
            self.lru[slot].remove(pos);
            self.families[slot].resident_tiles =
                self.families[slot].resident_tiles.saturating_sub(1);
            self.families[slot].resident_bytes = self.families[slot]
                .resident_bytes
                .saturating_sub(self.tile_bytes);
        }
    }

    /// True when inserting one more tile would push the family over its own
    /// budget and it still has tiles that could be evicted.
    pub fn needs_eviction(&self, family_slot: u32) -> bool {
        let slot = family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize;
        let family = &self.families[slot];
        family.resident_tiles > 0 && family.resident_bytes + self.tile_bytes > family.budget_bytes
    }

    /// Least recently used resident tile of a family.
    pub fn lru_tile(&self, family_slot: u32) -> Option<TileKey> {
        let slot = family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize;
        self.lru[slot].front().copied()
    }

    pub fn family(&self, family_slot: u32) -> FamilyResidency {
        self.families[family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize]
    }

    pub fn total_resident_tiles(&self) -> u32 {
        self.families.iter().map(|f| f.resident_tiles).sum()
    }

    pub fn total_resident_bytes(&self) -> u64 {
        self.families.iter().map(|f| f.resident_bytes).sum()
    }
}

/// Decode a shader feedback payload (`logical_material + 1`) into
/// `(family_slot, material_index)`. Returns `None` for the zero sentinel and
/// for family slots outside the supported range.
pub(crate) fn decode_feedback_payload(payload: u32, material_count: u32) -> Option<(u32, u32)> {
    let encoded = payload.checked_sub(1)?;
    let material_count = material_count.max(1);
    let family_slot = encoded / material_count;
    let material_index = encoded % material_count;
    (family_slot < VT_FAMILY_COUNT as u32).then_some((family_slot, material_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILE: u64 = 256 * 256 * 4;

    fn key(family_slot: u32, x: u32, y: u32) -> TileKey {
        TileKey {
            family_slot,
            material_index: 0,
            x,
            y,
            mip_level: 0,
        }
    }

    #[test]
    fn budget_splits_evenly_across_enabled_families() {
        let total = 300 * TILE;
        let three = FamilyResidencyTracker::new(total, 0b111, TILE);
        assert_eq!(three.family(0).budget_bytes, total / 3);
        assert_eq!(three.family(1).budget_bytes, total / 3);
        assert_eq!(three.family(2).budget_bytes, total / 3);
        assert!(three.family(0).budget_bytes * 3 <= total);

        let two = FamilyResidencyTracker::new(total, 0b011, TILE);
        assert_eq!(two.family(0).budget_bytes, total / 2);
        assert_eq!(two.family(1).budget_bytes, total / 2);
        assert_eq!(two.family(2).budget_bytes, 0);

        let one = FamilyResidencyTracker::new(total, 0b010, TILE);
        assert_eq!(one.family(0).budget_bytes, 0);
        assert_eq!(one.family(1).budget_bytes, total);
        assert_eq!(one.family(2).budget_bytes, 0);
    }

    #[test]
    fn insert_and_evict_update_family_accounting() {
        let mut tracker = FamilyResidencyTracker::new(10 * TILE, 0b111, TILE);
        tracker.on_insert(key(1, 0, 0));
        tracker.on_insert(key(1, 1, 0));
        tracker.on_insert(key(2, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 2);
        assert_eq!(tracker.family(1).resident_bytes, 2 * TILE);
        assert_eq!(tracker.family(2).resident_tiles, 1);
        assert_eq!(tracker.total_resident_tiles(), 3);
        assert_eq!(tracker.total_resident_bytes(), 3 * TILE);

        // Duplicate insert must not double-count.
        tracker.on_insert(key(1, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 2);

        tracker.on_evict(&key(1, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 1);
        assert_eq!(tracker.family(1).resident_bytes, TILE);
        // Evicting an unknown tile is a no-op.
        tracker.on_evict(&key(1, 9, 9));
        assert_eq!(tracker.family(1).resident_tiles, 1);
    }

    #[test]
    fn lru_order_follows_access_pattern() {
        let mut tracker = FamilyResidencyTracker::new(9 * TILE, 0b111, TILE);
        tracker.on_insert(key(0, 0, 0));
        tracker.on_insert(key(0, 1, 0));
        tracker.on_insert(key(0, 2, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 0, 0)));

        tracker.note_access(key(0, 0, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 1, 0)));

        tracker.on_evict(&key(0, 1, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 2, 0)));
    }

    #[test]
    fn needs_eviction_respects_per_family_budget_only() {
        // 6 tiles total budget -> 2 tiles per family.
        let mut tracker = FamilyResidencyTracker::new(6 * TILE, 0b111, TILE);
        tracker.on_insert(key(0, 0, 0));
        tracker.on_insert(key(0, 1, 0));
        // Family 0 is at budget; one more tile requires within-family eviction.
        assert!(tracker.needs_eviction(0));
        // Family 1 is empty and under budget: no eviction pressure, and it is
        // never asked to give up tiles on family 0's behalf.
        assert!(!tracker.needs_eviction(1));

        tracker.on_insert(key(1, 0, 0));
        assert!(!tracker.needs_eviction(1));
        assert!(tracker.needs_eviction(0));

        // Draining family 0 clears its pressure without touching family 1.
        tracker.on_evict(&key(0, 0, 0));
        tracker.on_evict(&key(0, 1, 0));
        assert!(!tracker.needs_eviction(0));
        assert_eq!(tracker.family(1).resident_tiles, 1);
    }

    #[test]
    fn within_family_eviction_loop_converges() {
        let mut tracker = FamilyResidencyTracker::new(3 * TILE, 0b111, TILE);
        // Budget = 1 tile per family.
        tracker.on_insert(key(2, 0, 0));
        assert!(tracker.needs_eviction(2));
        let victim = tracker.lru_tile(2).expect("family has a victim");
        tracker.on_evict(&victim);
        assert!(!tracker.needs_eviction(2));
        // Empty family never reports pressure (no infinite loops on tiny budgets).
        assert!(!tracker.needs_eviction(0));
        assert_eq!(tracker.lru_tile(0), None);
    }

    #[test]
    fn feedback_payload_demux_round_trips() {
        let material_count = 4;
        for family_slot in 0..VT_FAMILY_COUNT as u32 {
            for material_index in 0..material_count {
                let payload = family_slot * material_count + material_index + 1;
                assert_eq!(
                    decode_feedback_payload(payload, material_count),
                    Some((family_slot, material_index)),
                );
            }
        }
        // Zero is the "no feedback" sentinel.
        assert_eq!(decode_feedback_payload(0, material_count), None);
        // Payloads past the last family are rejected.
        let out_of_range = VT_FAMILY_COUNT as u32 * material_count + 1;
        assert_eq!(decode_feedback_payload(out_of_range, material_count), None);
        // material_count = 0 is clamped instead of dividing by zero.
        assert_eq!(decode_feedback_payload(1, 0), Some((0, 0)));
    }
}
