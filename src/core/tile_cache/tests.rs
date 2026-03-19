use super::{allocator::AtlasAllocator, *};

#[test]
fn test_tile_id_creation() {
    let tile = TileId::new(10, 20, 2);
    assert_eq!(tile.x, 10);
    assert_eq!(tile.y, 20);
    assert_eq!(tile.mip_level, 2);
}

#[test]
fn test_tile_id_parent() {
    let tile = TileId::new(10, 20, 2);
    let parent = tile.parent().unwrap();
    assert_eq!(parent.x, 5);
    assert_eq!(parent.y, 10);
    assert_eq!(parent.mip_level, 1);

    let root = TileId::new(0, 0, 0);
    assert!(root.parent().is_none());
}

#[test]
fn test_tile_id_children() {
    let tile = TileId::new(5, 10, 1);
    let children = tile.children();

    assert_eq!(children[0], TileId::new(10, 20, 2));
    assert_eq!(children[1], TileId::new(11, 20, 2));
    assert_eq!(children[2], TileId::new(10, 21, 2));
    assert_eq!(children[3], TileId::new(11, 21, 2));
}

#[test]
fn test_tile_cache_creation() {
    let cache = TileCache::new(100);
    assert_eq!(cache.capacity(), 100);
    assert_eq!(cache.resident_count(), 0);
    assert!(!cache.is_resident(&TileId::new(0, 0, 0)));
}

#[test]
fn test_tile_cache_allocation() {
    let mut cache = TileCache::new(10);
    cache.configure_atlas(512, 512, 64);

    let tile1 = TileId::new(0, 0, 0);
    let tile2 = TileId::new(1, 0, 0);

    let slot1 = cache.allocate_tile(tile1);
    assert!(slot1.is_some());
    assert!(cache.is_resident(&tile1));
    assert_eq!(cache.resident_count(), 1);

    let slot2 = cache.allocate_tile(tile2);
    assert!(slot2.is_some());
    assert!(cache.is_resident(&tile2));
    assert_eq!(cache.resident_count(), 2);

    let accessed_slot = cache.access_tile(&tile1);
    assert!(accessed_slot.is_some());
}

#[test]
fn test_tile_cache_eviction() {
    let mut cache = TileCache::new(2);
    cache.configure_atlas(256, 256, 64);

    let tile1 = TileId::new(0, 0, 0);
    let tile2 = TileId::new(1, 0, 0);
    let tile3 = TileId::new(2, 0, 0);

    cache.allocate_tile(tile1);
    cache.allocate_tile(tile2);
    assert_eq!(cache.resident_count(), 2);

    cache.allocate_tile(tile3);
    assert_eq!(cache.resident_count(), 2);
    assert!(cache.is_resident(&tile3));
    assert!(!cache.is_resident(&tile1));
}

#[test]
fn test_atlas_allocator() {
    let mut allocator = AtlasAllocator::new_with_dimensions(256, 256, 64);

    assert_eq!(allocator.total_count(), 16);
    assert_eq!(allocator.free_count(), 16);

    let slot = allocator.allocate();
    assert!(slot.is_some());
    assert_eq!(allocator.free_count(), 15);

    allocator.deallocate(slot.unwrap());
    assert_eq!(allocator.free_count(), 16);
}

#[test]
fn test_cache_stats() {
    let mut cache = TileCache::new(5);
    cache.configure_atlas(256, 256, 64);

    let stats = cache.stats();
    assert_eq!(stats.capacity, 5);
    assert_eq!(stats.resident_count, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);

    let tile = TileId::new(0, 0, 0);
    cache.access_tile(&tile);
    assert_eq!(cache.stats().misses, 1);

    cache.allocate_tile(tile);
    cache.access_tile(&tile);
    assert_eq!(cache.stats().hits, 1);
}

#[test]
fn test_reference_counting() {
    let mut cache = TileCache::new(5);
    let tile = TileId::new(0, 0, 0);

    cache.allocate_tile(tile);

    assert!(cache.add_ref(&tile));
    assert!(cache.add_ref(&tile));
    assert!(cache.release(&tile));
    assert!(cache.release(&tile));
    assert!(cache.release(&tile));
}
