use crate::terrain::tiling::TileId;
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    Any,
    PreserveRoot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub slot: (u32, u32),
    pub evicted: Option<TileId>,
    pub already_resident: bool,
}

#[derive(Debug)]
pub struct HeightPageResidency {
    tiles_x: u32,
    tiles_y: u32,
    fixed_lod: Option<u32>,
    slots: HashMap<TileId, (u32, u32)>,
    lru: VecDeque<TileId>,
}

impl HeightPageResidency {
    pub fn new(tiles_x: u32, tiles_y: u32, fixed_lod: Option<u32>) -> Result<Self, String> {
        if tiles_x == 0 || tiles_y == 0 {
            return Err("height residency dimensions must be non-zero".to_string());
        }
        Ok(Self {
            tiles_x,
            tiles_y,
            fixed_lod,
            slots: HashMap::new(),
            lru: VecDeque::new(),
        })
    }

    pub fn capacity(&self) -> usize {
        (u64::from(self.tiles_x) * u64::from(self.tiles_y)).min(usize::MAX as u64) as usize
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn slot_of(&self, id: TileId) -> Option<(u32, u32)> {
        self.slots.get(&id).copied()
    }

    pub fn entries(&self) -> Vec<(TileId, (u32, u32))> {
        let mut entries: Vec<_> = self.slots.iter().map(|(id, slot)| (*id, *slot)).collect();
        entries.sort_unstable_by_key(|(id, _)| (id.lod, id.y, id.x));
        entries
    }

    fn touch(&mut self, id: TileId) {
        self.lru.retain(|candidate| *candidate != id);
        self.lru.push_back(id);
    }

    fn first_free_slot(&self) -> Option<(u32, u32)> {
        (0..self.tiles_y).find_map(|sy| {
            (0..self.tiles_x)
                .find(|sx| !self.slots.values().any(|slot| *slot == (*sx, sy)))
                .map(|sx| (sx, sy))
        })
    }

    pub fn evict_lru(&mut self, policy: EvictionPolicy) -> Option<(TileId, (u32, u32))> {
        let candidates = self.lru.len();
        for _ in 0..candidates {
            let id = self.lru.pop_front()?;
            if policy == EvictionPolicy::PreserveRoot && id.lod == 0 {
                self.lru.push_back(id);
                continue;
            }
            if let Some(slot) = self.slots.remove(&id) {
                return Some((id, slot));
            }
        }
        None
    }

    pub fn place(&mut self, id: TileId, policy: EvictionPolicy) -> Result<Placement, String> {
        if let Some(slot) = self.slot_of(id) {
            self.touch(id);
            return Ok(Placement {
                slot,
                evicted: None,
                already_resident: true,
            });
        }
        if let Some(lod) = self.fixed_lod {
            if id.lod != lod || id.x >= self.tiles_x || id.y >= self.tiles_y {
                return Err("tile id is outside fixed height residency".to_string());
            }
            let slot = (id.x, id.y);
            self.slots.insert(id, slot);
            self.touch(id);
            return Ok(Placement {
                slot,
                evicted: None,
                already_resident: false,
            });
        }
        if let Some(slot) = self.first_free_slot() {
            self.slots.insert(id, slot);
            self.touch(id);
            return Ok(Placement {
                slot,
                evicted: None,
                already_resident: false,
            });
        }
        // A root is always allowed to displace a leaf. A leaf using
        // PreserveRoot fails atomically when the root is the only candidate.
        let effective_policy = if id.lod == 0 {
            EvictionPolicy::Any
        } else {
            policy
        };
        let Some((evicted, slot)) = self.evict_lru(effective_policy) else {
            return Err("height residency has no evictable slot".to_string());
        };
        self.slots.insert(id, slot);
        self.touch(id);
        Ok(Placement {
            slot,
            evicted: Some(evicted),
            already_resident: false,
        })
    }

    pub fn remove(&mut self, id: TileId) -> Option<(u32, u32)> {
        self.lru.retain(|candidate| *candidate != id);
        self.slots.remove(&id)
    }

    pub fn resolve(&self, requested: TileId) -> Option<(TileId, (u32, u32))> {
        let mut candidate = Some(requested);
        while let Some(id) = candidate {
            if let Some(slot) = self.slot_of(id) {
                return Some((id, slot));
            }
            candidate = id.parent();
        }
        None
    }

    pub fn touch_resolved(&mut self, requested: TileId) -> Option<TileId> {
        let (id, _) = self.resolve(requested)?;
        self.touch(id);
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserve_root_uses_true_lru_and_reuses_the_evicted_slot() {
        let mut pages = HeightPageResidency::new(3, 1, None).unwrap();
        let root = TileId::new(0, 0, 0);
        let a = TileId::new(2, 0, 0);
        let b = TileId::new(2, 1, 0);
        let c = TileId::new(2, 2, 0);
        pages.place(root, EvictionPolicy::PreserveRoot).unwrap();
        pages.place(a, EvictionPolicy::PreserveRoot).unwrap();
        let b_slot = pages.place(b, EvictionPolicy::PreserveRoot).unwrap().slot;
        assert_eq!(pages.touch_resolved(a), Some(a));
        let placed = pages.place(c, EvictionPolicy::PreserveRoot).unwrap();
        assert_eq!(placed.evicted, Some(b));
        assert_eq!(placed.slot, b_slot);
        assert_eq!(pages.resolve(TileId::new(3, 2, 0)).unwrap().0, root);
    }

    #[test]
    fn capacity_one_root_rejects_leaf_without_mutation() {
        let mut pages = HeightPageResidency::new(1, 1, None).unwrap();
        let root = TileId::new(0, 0, 0);
        pages.place(root, EvictionPolicy::PreserveRoot).unwrap();
        assert!(pages
            .place(TileId::new(1, 0, 0), EvictionPolicy::PreserveRoot)
            .is_err());
        assert_eq!(pages.entries(), vec![(root, (0, 0))]);
    }
}
