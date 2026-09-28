//! P2.1/M5: Clipmap streaming integration with HeightMosaic and PageTable.

use super::geomorph::TileReadiness;
use super::level::ClipmapLevel;
use super::ClipmapConfig;
use crate::terrain::lod::LodConfig;
use crate::terrain::page_table::PageTable;
use crate::terrain::stream::HeightMosaic;
use crate::terrain::tiling::TileId;
#[cfg(feature = "enable-globe")]
use glam::DVec3;
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use wgpu::Queue;

/// Clipmap streamer connecting clipmap mesh to tile streaming infrastructure.
pub struct ClipmapStreamer {
    pub clipmap: ClipmapLevel,
    pending_tiles: Vec<TileId>,
    loaded_tiles: Vec<TileId>,
    required_tiles: Vec<TileId>,
    globe_ring_footprints: Vec<Vec<TileId>>,
}

impl ClipmapStreamer {
    /// Create a new clipmap streamer.
    pub fn new(config: ClipmapConfig, center: Vec2, terrain_extent: f32) -> Self {
        Self {
            clipmap: ClipmapLevel::new(config, center, terrain_extent),
            pending_tiles: Vec::new(),
            loaded_tiles: Vec::new(),
            required_tiles: Vec::new(),
            globe_ring_footprints: Vec::new(),
        }
    }

    #[cfg(feature = "enable-globe")]
    pub fn new_globe(
        config: ClipmapConfig,
        center_ecef: DVec3,
        frame: super::globe::GlobeFrame,
        terrain_extent: f32,
    ) -> Option<Self> {
        Some(Self {
            clipmap: ClipmapLevel::new_globe(config, center_ecef, frame, terrain_extent)?,
            pending_tiles: Vec::new(),
            loaded_tiles: Vec::new(),
            required_tiles: Vec::new(),
            globe_ring_footprints: Vec::new(),
        })
    }

    /// Update clipmap based on camera position and request needed tiles.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        _view_matrix: Mat4,
        _proj_matrix: Mat4,
        _lod_config: &LodConfig,
    ) -> Vec<TileId> {
        // Update clipmap center to camera XZ position
        let new_center = Vec2::new(camera_pos.x, camera_pos.z);
        let required_tiles = self.clipmap.update_center(new_center);
        let required_tiles = if required_tiles.is_empty() {
            self.clipmap.calculate_required_tiles()
        } else {
            required_tiles
        };

        self.queue_required(required_tiles)
    }

    /// Follow a planetary camera's surface subpoint in f64 ECEF.
    #[cfg(feature = "enable-globe")]
    pub fn update_globe(
        &mut self,
        camera_anchor: DVec3,
        target_lod: u32,
        max_leaf_tiles: usize,
    ) -> Result<Vec<TileId>, String> {
        if !camera_anchor.is_finite() || camera_anchor.length_squared() == 0.0 {
            return Err("globe streaming camera anchor must be finite and non-zero".to_string());
        }
        let focus_ecef = camera_anchor.normalize() * self.clipmap.center_ecef().length();
        self.update_globe_focus(camera_anchor, focus_ecef, target_lod, max_leaf_tiles)
    }

    #[cfg(feature = "enable-globe")]
    pub fn update_globe_focus(
        &mut self,
        camera_anchor: DVec3,
        focus_ecef: DVec3,
        target_lod: u32,
        max_leaf_tiles: usize,
    ) -> Result<Vec<TileId>, String> {
        if !camera_anchor.is_finite() || camera_anchor.length_squared() == 0.0 {
            return Err("globe streaming camera anchor must be finite and non-zero".to_string());
        }
        if !focus_ecef.is_finite() || focus_ecef.length_squared() == 0.0 {
            return Err("globe streaming focus must be finite and non-zero".to_string());
        }
        let surface_center = focus_ecef.normalize() * self.clipmap.center_ecef().length();
        // Build and validate the complete candidate demand before touching the
        // live camera frame, center, mesh cache, or residency bookkeeping. A
        // capacity error is therefore a rejected transaction, not a partial
        // camera update.
        let candidate_frame = self
            .clipmap
            .globe_frame()
            .ok_or_else(|| "globe streaming requires a valid globe frame".to_string())?;
        let candidate = super::level::ClipmapLevel::new_globe(
            self.clipmap.config.clone(),
            surface_center,
            candidate_frame,
            self.clipmap.terrain_extent,
        )
        .ok_or_else(|| "globe streaming candidate center is invalid".to_string())?;
        let globe_ring_footprints = candidate.globe_target_lod_footprints(target_lod)?;
        let mut seen = HashSet::new();
        let required_tiles: Vec<_> = globe_ring_footprints
            .iter()
            .flatten()
            .copied()
            .filter(|tile| seen.insert(*tile))
            .collect();
        if required_tiles.is_empty() {
            return Err(format!(
                "globe target-LOD {target_lod} footprint is empty"
            ));
        }
        if required_tiles.len() > max_leaf_tiles {
            return Err(format!(
                "complete globe target-LOD {target_lod} footprint needs {} leaf slots, but only {max_leaf_tiles} remain after reserving local ancestors",
                required_tiles.len()
            ));
        }

        self.clipmap.recenter(camera_anchor)?;
        let _ = self.clipmap.update_globe_center(surface_center);
        self.globe_ring_footprints = globe_ring_footprints;
        Ok(self.queue_required(required_tiles))
    }

    fn queue_required(&mut self, required_tiles: Vec<TileId>) -> Vec<TileId> {
        let required: HashSet<_> = required_tiles.iter().copied().collect();
        self.pending_tiles.retain(|tile| required.contains(tile));
        self.loaded_tiles.retain(|tile| required.contains(tile));
        self.required_tiles = required_tiles.clone();
        let new_tiles: Vec<TileId> = required_tiles
            .into_iter()
            .filter(|t| !self.loaded_tiles.contains(t) && !self.pending_tiles.contains(t))
            .collect();

        self.pending_tiles.extend(new_tiles.iter().cloned());
        new_tiles
    }

    /// Reconcile logical clipmap readiness with exact physical atlas
    /// residency. Returns every currently required tile still missing.
    pub fn reconcile_residency(&mut self, resident: &HashSet<TileId>) -> Vec<TileId> {
        let required: HashSet<_> = self.required_tiles.iter().copied().collect();
        self.loaded_tiles
            .retain(|tile| required.contains(tile) && resident.contains(tile));
        self.pending_tiles
            .retain(|tile| required.contains(tile) && !resident.contains(tile));
        for tile in &self.required_tiles {
            if resident.contains(tile) {
                if !self.loaded_tiles.contains(tile) {
                    self.loaded_tiles.push(*tile);
                }
                self.pending_tiles.retain(|pending| pending != tile);
            } else if !self.pending_tiles.contains(tile) {
                self.pending_tiles.push(*tile);
            }
        }
        self.required_tiles
            .iter()
            .copied()
            .filter(|tile| !resident.contains(tile))
            .collect()
    }

    pub fn required_tiles(&self) -> &[TileId] {
        &self.required_tiles
    }

    pub fn invalidate_loaded(&mut self, tiles: &[TileId]) {
        for tile in tiles {
            self.loaded_tiles.retain(|loaded| loaded != tile);
            if self.required_tiles.contains(tile) && !self.pending_tiles.contains(tile) {
                self.pending_tiles.push(*tile);
            }
        }
    }

    /// Mark tiles as loaded (call after successful upload to mosaic).
    pub fn mark_loaded(&mut self, tiles: &[TileId]) {
        for tile in tiles {
            self.pending_tiles.retain(|t| t != tile);
            if !self.loaded_tiles.contains(tile) {
                self.loaded_tiles.push(*tile);
            }
        }
    }

    /// Get the clipmap mesh, generating if needed.
    pub fn mesh(&mut self) -> Result<&super::level::ClipmapMesh, String> {
        self.clipmap.mesh()
    }

    /// Map ring index to tile LOD level.
    pub fn ring_to_tile_lod(&self, ring_index: u32) -> u32 {
        self.clipmap.ring_lod(ring_index)
    }

    /// Get current clipmap center.
    pub fn center(&self) -> Result<Vec2, String> {
        self.clipmap.render_center()
    }

    /// Get terrain extent.
    pub fn terrain_extent(&self) -> f32 {
        self.clipmap.terrain_extent
    }

    /// Get pending tile count.
    pub fn pending_count(&self) -> usize {
        self.pending_tiles.len()
    }

    /// Get loaded tile count.
    pub fn loaded_count(&self) -> usize {
        self.loaded_tiles.len()
    }

    /// Return whether the fine ring and its next-coarser neighbour are fully
    /// resident for safe boundary morphing. Pending keys are explicitly not
    /// resident, even when an older tile at the same LOD is loaded.
    pub fn ring_readiness(&self, ring_index: u32) -> TileReadiness {
        if !self.globe_ring_footprints.is_empty() {
            let loaded = |region: usize| {
                let required = &self.globe_ring_footprints
                    [region.min(self.globe_ring_footprints.len() - 1)];
                !required.is_empty()
                    && required.iter().all(|tile| self.loaded_tiles.contains(tile))
                    && required.iter().all(|tile| !self.pending_tiles.contains(tile))
            };
            let fine_region = (ring_index as usize + 1)
                .min(self.globe_ring_footprints.len() - 1);
            let coarse_region = (fine_region + 1)
                .min(self.globe_ring_footprints.len() - 1);
            return TileReadiness {
                fine_resident: loaded(fine_region),
                coarse_resident: loaded(coarse_region),
            };
        }
        let loaded_for_lod = |lod: u32| {
            let required: Vec<_> = self
                .required_tiles
                .iter()
                .filter(|tile| tile.lod == lod)
                .collect();
            !required.is_empty()
                && required.iter().all(|tile| self.loaded_tiles.contains(tile))
                && required
                    .iter()
                    .all(|tile| !self.pending_tiles.contains(tile))
        };
        let fine_resident = loaded_for_lod(self.ring_to_tile_lod(ring_index));
        let coarse_ring = ring_index.saturating_add(1);
        let coarse_resident = if coarse_ring >= self.clipmap.config.ring_count {
            fine_resident
        } else {
            loaded_for_lod(self.ring_to_tile_lod(coarse_ring))
        };
        TileReadiness {
            fine_resident,
            coarse_resident,
        }
    }
}

/// Integration helper for uploading clipmap tiles to HeightMosaic.
pub fn upload_clipmap_tiles(
    mosaic: &mut HeightMosaic,
    queue: &Queue,
    tiles: &[(TileId, Vec<f32>)],
) -> Vec<TileId> {
    let mut uploaded = Vec::new();
    for (tile_id, height_data) in tiles {
        match mosaic.upload_tile(queue, *tile_id, height_data) {
            Ok(_slot) => uploaded.push(*tile_id),
            Err(e) => {
                eprintln!("[clipmap] Failed to upload tile {:?}: {}", tile_id, e);
            }
        }
    }
    uploaded
}

/// Sync clipmap tiles to page table.
pub fn sync_clipmap_page_table(page_table: &mut PageTable, queue: &Queue, mosaic: &HeightMosaic) {
    page_table.sync_from_mosaic(queue, mosaic);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streamer_creation() {
        let config = ClipmapConfig::new(4, 64);
        let streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);
        assert_eq!(streamer.center().unwrap(), Vec2::ZERO);
        assert_eq!(streamer.terrain_extent(), 1000.0);
    }

    #[test]
    fn test_streamer_update_requests_tiles() {
        let config = ClipmapConfig::new(4, 64);
        let mut streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);

        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());
        let camera = Vec3::new(100.0, 50.0, 100.0);
        let view = Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.33, 1.0, 1000.0);

        let tiles = streamer.update(camera, view, proj, &lod_config);
        // First update should request tiles
        assert!(!tiles.is_empty() || streamer.pending_count() > 0);
    }

    #[test]
    fn test_mark_loaded_clears_pending() {
        let config = ClipmapConfig::new(4, 64);
        let mut streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);

        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());
        let camera = Vec3::new(100.0, 50.0, 100.0);
        let view = Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.33, 1.0, 1000.0);

        let tiles = streamer.update(camera, view, proj, &lod_config);
        let pending_before = streamer.pending_count();

        if !tiles.is_empty() {
            streamer.mark_loaded(&tiles);
            assert!(streamer.pending_count() < pending_before || pending_before == 0);
            assert!(streamer.loaded_count() > 0);
        }
    }

    #[test]
    fn throttled_arrivals_keep_each_boundary_coarse_snapped_until_both_sides_are_loaded() {
        // This catches inverted readiness semantics that treat pending tiles
        // as resident and permit a fine/coarse transition crack mid-arrival.
        let mut streamer = ClipmapStreamer::new(ClipmapConfig::new(2, 4), Vec2::ZERO, 1000.0);
        let fine = TileId::new(0, 0, 0);
        let coarse = TileId::new(1, 0, 0);
        streamer.queue_required(vec![fine, coarse]);

        assert_eq!(
            streamer.ring_readiness(0),
            TileReadiness {
                fine_resident: false,
                coarse_resident: false,
            }
        );
        streamer.mark_loaded(&[fine]);
        assert_eq!(
            streamer.ring_readiness(0),
            TileReadiness {
                fine_resident: true,
                coarse_resident: false,
            }
        );
        streamer.mark_loaded(&[coarse]);
        assert_eq!(
            streamer.ring_readiness(0),
            TileReadiness {
                fine_resident: true,
                coarse_resident: true,
            }
        );
    }

    #[test]
    fn stationary_update_keeps_loaded_required_keys_ready() {
        // This catches clearing the current required keys when the camera has
        // not crossed a recenter threshold after tiles have arrived.
        let mut streamer = ClipmapStreamer::new(ClipmapConfig::new(2, 4), Vec2::ZERO, 1000.0);
        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());
        let camera = Vec3::new(100.0, 50.0, 100.0);
        let requested = streamer.update(camera, Mat4::IDENTITY, Mat4::IDENTITY, &lod_config);
        streamer.mark_loaded(&requested);
        assert_eq!(
            streamer.ring_readiness(0),
            TileReadiness {
                fine_resident: true,
                coarse_resident: true,
            }
        );

        assert!(streamer
            .update(camera, Mat4::IDENTITY, Mat4::IDENTITY, &lod_config)
            .is_empty());
        assert_eq!(
            streamer.ring_readiness(0),
            TileReadiness {
                fine_resident: true,
                coarse_resident: true,
            }
        );
    }

    #[test]
    fn move_away_evict_and_return_requeues_missing_readiness() {
        let mut streamer = ClipmapStreamer::new(ClipmapConfig::new(2, 4), Vec2::ZERO, 1000.0);
        let first = TileId::new(2, 0, 0);
        let away = TileId::new(2, 3, 3);
        streamer.queue_required(vec![first]);
        streamer.mark_loaded(&[first]);
        assert!(streamer.reconcile_residency(&HashSet::from([first])).is_empty());

        streamer.queue_required(vec![away]);
        streamer.mark_loaded(&[away]);
        assert!(streamer.reconcile_residency(&HashSet::from([away])).is_empty());

        // `first` was evicted while away. Returning must not trust historical
        // loaded state or unrelated feedback; exact residency makes it pending.
        streamer.queue_required(vec![first]);
        assert_eq!(streamer.reconcile_residency(&HashSet::from([away])), vec![first]);
        assert_eq!(streamer.pending_count(), 1);
        assert_eq!(streamer.loaded_count(), 0);
    }

    #[test]
    fn multi_key_arrivals_preserve_boundary_parity_and_topology_at_every_transition() {
        use super::super::geomorph::{
            analyze_depth_discontinuities, analyze_seams, apply_tile_readiness, GeomorphConfig,
        };
        use super::super::{make_ring, ClipmapVertex};

        // Exercise the same producer -> readiness mutation -> height consumer
        // path used by TerrainScene::prepare_geometry. The checkerboard DEM is
        // an independent oracle: fine samples on odd texels are 1, while the
        // next-coarser two-texel lattice interpolates only zero-valued texels.
        const TERRAIN_EXTENT: f32 = 64.0;
        const HEIGHT_SIZE: u32 = 65;
        let heightmap = (0..HEIGHT_SIZE)
            .flat_map(|y| (0..HEIGHT_SIZE).map(move |x| ((x + y) & 1) as f32))
            .collect::<Vec<_>>();
        let boundary_at = |vertices: &[ClipmapVertex], extent: f32| {
            vertices
                .iter()
                .copied()
                .filter(|vertex| {
                    !vertex.is_skirt()
                        && (Vec2::new(vertex.position[0], vertex.position[1])
                            .abs()
                            .max_element()
                            - extent)
                            .abs()
                            < 1.0e-5
                })
                .collect::<Vec<_>>()
        };

        // Both sides come from separate calls to the production mesh
        // generator. Ring 0's real outer boundary meets ring 1's inner edge.
        let (fine_template, _) = make_ring(0, 10.0, 20.0, 20, Vec2::ZERO, TERRAIN_EXTENT, 0.3);
        let (coarse_vertices, _) = make_ring(1, 20.0, 28.0, 8, Vec2::ZERO, TERRAIN_EXTENT, 0.3);
        let coarse_boundary = boundary_at(&coarse_vertices, 20.0);

        // The 18-unit contour is a real set of production ring-0 transition
        // vertices. A separately generated ring-1 inner edge provides the
        // coarse-lattice height oracle at exactly that contour.
        let (transition_coarse_vertices, _) =
            make_ring(1, 18.0, 26.0, 8, Vec2::ZERO, TERRAIN_EXTENT, 0.3);
        let transition_coarse_boundary = boundary_at(&transition_coarse_vertices, 18.0);
        let geomorph = GeomorphConfig {
            max_seam_gap: 1.0e-5,
            ..Default::default()
        };

        let fine_a = TileId::new(0, 0, 0);
        let fine_b = TileId::new(0, 1, 0);
        let coarse_a = TileId::new(1, 0, 0);
        let coarse_b = TileId::new(1, 1, 0);

        for (state, loaded, expected) in [
            (
                "pending",
                &[][..],
                TileReadiness {
                    fine_resident: false,
                    coarse_resident: false,
                },
            ),
            (
                "fine partial",
                &[fine_a][..],
                TileReadiness {
                    fine_resident: false,
                    coarse_resident: false,
                },
            ),
            (
                "fine ready, coarse pending",
                &[fine_a, fine_b][..],
                TileReadiness {
                    fine_resident: true,
                    coarse_resident: false,
                },
            ),
            (
                "coarse partial",
                &[coarse_a][..],
                TileReadiness {
                    fine_resident: false,
                    coarse_resident: false,
                },
            ),
            (
                "fine pending, coarse ready",
                &[coarse_a, coarse_b][..],
                TileReadiness {
                    fine_resident: false,
                    coarse_resident: true,
                },
            ),
            (
                "both partial",
                &[fine_a, coarse_a][..],
                TileReadiness {
                    fine_resident: false,
                    coarse_resident: false,
                },
            ),
            (
                "ready",
                &[fine_a, fine_b, coarse_a, coarse_b][..],
                TileReadiness {
                    fine_resident: true,
                    coarse_resident: true,
                },
            ),
        ] {
            let mut streamer = ClipmapStreamer::new(ClipmapConfig::new(2, 4), Vec2::ZERO, 1000.0);
            streamer.queue_required(vec![fine_a, fine_b, coarse_a, coarse_b]);
            streamer.mark_loaded(loaded);
            let readiness = streamer.ring_readiness(0);
            assert_eq!(readiness, expected, "{state}");

            let mut fine_vertices = fine_template.clone();
            apply_tile_readiness(&mut fine_vertices, readiness);

            let fine_boundary = boundary_at(&fine_vertices, 20.0);
            let seam = analyze_seams(&fine_boundary, &coarse_boundary, &geomorph);
            assert!(seam.seams_valid, "{state}: {readiness:?}: {seam:?}");
            assert_eq!(seam.crack_count, 0, "{state}: {readiness:?}: {seam:?}");
            let depth = analyze_depth_discontinuities(
                &fine_boundary,
                &coarse_boundary,
                &heightmap,
                (HEIGHT_SIZE, HEIGHT_SIZE),
                1.0,
                geomorph.max_seam_gap,
            );
            assert!(depth.sample_count > 0, "{state}: {readiness:?}: {depth:?}");
            assert_eq!(depth.crack_count, 0, "{state}: {readiness:?}: {depth:?}");

            let transition_boundary = boundary_at(&fine_vertices, 18.0);
            let transition_seam =
                analyze_seams(&transition_boundary, &transition_coarse_boundary, &geomorph);
            assert!(
                transition_seam.seams_valid,
                "{state}: {readiness:?}: {transition_seam:?}"
            );
            let transition_depth = analyze_depth_discontinuities(
                &transition_boundary,
                &transition_coarse_boundary,
                &heightmap,
                (HEIGHT_SIZE, HEIGHT_SIZE),
                1.0,
                geomorph.max_seam_gap,
            );
            assert!(
                transition_depth.sample_count > 0,
                "{state}: {readiness:?}: {transition_depth:?}"
            );
            if readiness.fine_resident && readiness.coarse_resident {
                assert!(transition_boundary.iter().all(|vertex| {
                    let weight = vertex.morph_weight();
                    weight > 0.0 && weight < 1.0
                }));
                assert!(
                    transition_depth.crack_count > 0,
                    "ready distance morph was replaced by a coarse snap: {transition_depth:?}"
                );
            } else {
                assert!(
                    transition_boundary
                        .iter()
                        .all(|vertex| vertex.morph_weight() == 1.0),
                    "unavailable transition was not coarse-snapped: {state}: {readiness:?}"
                );
                assert_eq!(
                    transition_depth.crack_count, 0,
                    "unavailable transition opened depth holes: {state}: {readiness:?}: {transition_depth:?}"
                );
            }
        }

        // Negative control: move one complete side of the independently
        // generated fine boundary away from its coarse neighbour. The spatial
        // oracle must reject it, proving zero cracks above is not fail-open.
        let mut open_fine_boundary = boundary_at(&fine_template, 20.0);
        for vertex in &mut open_fine_boundary {
            if (vertex.position[0] - 20.0).abs() < 1.0e-5 {
                vertex.position[0] += 0.25;
            }
        }
        let negative = analyze_seams(&open_fine_boundary, &coarse_boundary, &geomorph);
        assert!(
            !negative.seams_valid,
            "negative control passed: {negative:?}"
        );
        assert!(
            negative.crack_count > 0,
            "negative control passed: {negative:?}"
        );
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_streamer_follows_surface_subpoint() {
        use super::super::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let camera = DVec3::X * (radius + 1_000.0);
        let frame = GlobeFrame::globe(radius, camera).unwrap();
        let mut streamer = ClipmapStreamer::new_globe(
            ClipmapConfig::new(2, 4),
            DVec3::X * radius,
            frame,
            10_000.0,
        )
        .unwrap();
        let next_camera = frame.lonlat_alt_to_ecef(1.0, 0.0, 1_000.0).unwrap();
        let requested = streamer.update_globe(next_camera, 14, 256).unwrap();

        assert!(!requested.is_empty());
        assert!((streamer.clipmap.center_ecef().length() - radius).abs() < 1.0e-6);
        assert_eq!(streamer.clipmap.camera_anchor(), Some(next_camera));
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn rainier_lod14_full_footprint_converges_within_reserved_capacity() {
        use super::super::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let seed = GlobeFrame::globe(radius, DVec3::X * radius).unwrap();
        let center = seed.lonlat_alt_to_ecef(-121.7603, 46.8523, 0.0).unwrap();
        let camera = center + center.normalize() * 1_000.0;
        let frame = GlobeFrame::globe(radius, camera).unwrap();
        let mut streamer = ClipmapStreamer::new_globe(
            ClipmapConfig::new(4, 32),
            center,
            frame,
            1_250.0,
        )
        .unwrap();

        let requested = streamer.update_globe(camera, 14, 242).unwrap();
        assert!(!requested.is_empty());
        assert!(requested.len() <= 242);
        assert!(requested.iter().all(|tile| tile.lod == 14));

        let before_rejection = streamer.required_tiles().to_vec();
        let error = streamer.update_globe(camera, 14, requested.len() - 1).unwrap_err();
        assert!(error.contains("complete globe target-LOD 14 footprint needs"));
        assert_eq!(streamer.required_tiles(), before_rejection);

        streamer.mark_loaded(&requested);
        assert_eq!(streamer.pending_count(), 0);
        for ring in 0..4 {
            let readiness = streamer.ring_readiness(ring);
            assert!(readiness.fine_resident, "ring {ring} fine footprint did not converge");
            assert!(
                readiness.coarse_resident,
                "ring {ring} coarse footprint did not converge"
            );
        }
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_capacity_rejection_is_transactional_across_camera_motion() {
        use super::super::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let seed = GlobeFrame::globe(radius, DVec3::X * radius).unwrap();
        let center = seed.lonlat_alt_to_ecef(-121.7603, 46.8523, 0.0).unwrap();
        let camera = center + center.normalize() * 1_000.0;
        let frame = GlobeFrame::globe(radius, camera).unwrap();
        let mut streamer = ClipmapStreamer::new_globe(
            ClipmapConfig::new(4, 32),
            center,
            frame,
            1_250.0,
        )
        .unwrap();

        let requested = streamer.update_globe(camera, 14, 242).unwrap();
        streamer.mark_loaded(&requested[..requested.len() / 2]);
        let before_center = streamer.clipmap.center_ecef();
        let before_anchor = streamer.clipmap.camera_anchor();
        let before_required = streamer.required_tiles.clone();
        let before_pending = streamer.pending_tiles.clone();
        let before_loaded = streamer.loaded_tiles.clone();
        let (before_vertex_ptr, before_index_ptr, before_vertices, before_indices) = {
            let mesh = streamer.mesh().unwrap();
            (
                mesh.vertices.as_ptr() as usize,
                mesh.indices.as_ptr() as usize,
                bytemuck::cast_slice::<_, u8>(&mesh.vertices).to_vec(),
                bytemuck::cast_slice::<_, u8>(&mesh.indices).to_vec(),
            )
        };

        let moved_camera = seed
            .lonlat_alt_to_ecef(-121.60, 46.90, 1_000.0)
            .unwrap();
        let error = streamer
            .update_globe(moved_camera, 14, requested.len() - 1)
            .unwrap_err();
        assert!(error.contains("complete globe target-LOD 14 footprint needs"));
        assert_eq!(streamer.clipmap.center_ecef(), before_center);
        assert_eq!(streamer.clipmap.camera_anchor(), before_anchor);
        assert_eq!(streamer.required_tiles, before_required);
        assert_eq!(streamer.pending_tiles, before_pending);
        assert_eq!(streamer.loaded_tiles, before_loaded);
        let mesh = streamer.mesh().unwrap();
        assert_eq!(mesh.vertices.as_ptr() as usize, before_vertex_ptr);
        assert_eq!(mesh.indices.as_ptr() as usize, before_index_ptr);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&mesh.vertices), before_vertices);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&mesh.indices), before_indices);
    }
}
