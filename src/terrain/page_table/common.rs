use crate::terrain::tiling::TileId;

/// Return true if `child` is a strict descendant of `ancestor` in the quadtree.
pub(crate) fn is_descendant_of(child: TileId, ancestor: TileId) -> bool {
    if child.lod <= ancestor.lod {
        return false;
    }
    let shift = child.lod - ancestor.lod;
    (child.x >> shift) == ancestor.x && (child.y >> shift) == ancestor.y
}

#[derive(Clone, Copy, Debug)]
pub enum CoalescePolicy {
    PreferCoarse,
    PreferFine,
}

impl Default for CoalescePolicy {
    fn default() -> Self {
        Self::PreferCoarse
    }
}
