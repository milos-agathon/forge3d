use crate::terrain::tiling::TileId;

/// Raw ORBIS ABI tile coordinates use u32 axes, so LOD 31 is the largest
/// representable quadtree level (`2^31` tiles per axis).
pub const MAX_GLOBAL_TILE_LOD: u32 = 31;

pub fn validate_global_tile_id(lod: u32, x: u32, y: u32) -> Result<TileId, String> {
    if lod > MAX_GLOBAL_TILE_LOD {
        return Err(format!(
            "global height tile LOD {lod} exceeds {MAX_GLOBAL_TILE_LOD}"
        ));
    }
    let axis = 1u32
        .checked_shl(lod)
        .ok_or_else(|| "height quadtree LOD overflow".to_string())?;
    if x >= axis || y >= axis {
        return Err(format!(
            "global height tile out of range: lod={lod}, x={x}, y={y}"
        ));
    }
    Ok(TileId::new(lod, x, y))
}

/// Geographic footprint of the ORBIS global quadtree. Coordinates are
/// `(west, south, east, north)` in WGS84 degrees with linear latitude.
pub fn global_tile_lonlat_bounds(tile_id: TileId) -> Result<(f64, f64, f64, f64), String> {
    validate_global_tile_id(tile_id.lod, tile_id.x, tile_id.y)?;
    let axis = 1u64 << tile_id.lod;
    let scale = axis as f64;
    Ok((
        tile_id.x as f64 / scale * 360.0 - 180.0,
        90.0 - (u64::from(tile_id.y) + 1) as f64 / scale * 180.0,
        (u64::from(tile_id.x) + 1) as f64 / scale * 360.0 - 180.0,
        90.0 - tile_id.y as f64 / scale * 180.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_quadrants_use_equirectangular_tile_identity() {
        assert_eq!(
            global_tile_lonlat_bounds(TileId::new(1, 0, 0)).unwrap(),
            (-180.0, 0.0, 0.0, 90.0)
        );
        assert_eq!(
            global_tile_lonlat_bounds(TileId::new(2, 2, 1)).unwrap(),
            (0.0, 0.0, 90.0, 45.0)
        );
    }
}
