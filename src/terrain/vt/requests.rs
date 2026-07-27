use crate::terrain::vt_family_residency::{TileKey, VT_FAMILY_COUNT};
use std::collections::HashSet;
use std::ops::{Index, IndexMut};

/// Device-independent feedback retention state. A not-ready map is an
/// observation, never a request-set transition.
#[derive(Default)]
pub(crate) struct RetainedRequestSet {
    buckets: [HashSet<TileKey>; VT_FAMILY_COUNT],
}

impl RetainedRequestSet {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &HashSet<TileKey>> {
        self.buckets.iter()
    }

    pub(crate) fn on_not_ready(&mut self) {
        // Intentionally preserve every bucket until a resident upload
        // succeeds and removes the corresponding key.
    }
}

impl Index<usize> for RetainedRequestSet {
    type Output = HashSet<TileKey>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buckets[index]
    }
}

impl IndexMut<usize> for RetainedRequestSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buckets[index]
    }
}

pub(crate) fn retention_probe(not_ready_frames: u32) -> (bool, bool, usize) {
    let key = TileKey {
        family_slot: 0,
        material_index: 0,
        x: 17,
        y: 9,
        mip_level: 3,
    };
    let mut retained = RetainedRequestSet::default();
    retained[0].insert(key);
    for _ in 0..not_ready_frames {
        retained.on_not_ready();
    }
    let preserved = retained[0].contains(&key);
    retained[0].remove(&key);
    (preserved, retained[0].is_empty(), retained[0].len())
}

#[cfg(test)]
mod tests {
    use super::RetainedRequestSet;
    use crate::terrain::vt_family_residency::TileKey;

    #[test]
    fn thirty_not_ready_frames_preserve_requests_until_resident() {
        let key = TileKey {
            family_slot: 0,
            material_index: 0,
            x: 17,
            y: 9,
            mip_level: 3,
        };
        let mut retained = RetainedRequestSet::default();
        retained[0].insert(key);
        for _ in 0..30 {
            retained.on_not_ready();
            assert!(retained[0].contains(&key));
        }
        retained[0].remove(&key);
        assert!(retained[0].is_empty());
    }
}
