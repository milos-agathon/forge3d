//! P2.1/M5: Complete clipmap level with center block and nested rings.

use super::ring::{make_center_block, make_ring, make_ring_skirts};
use super::vertex::ClipmapVertex;
use super::ClipmapConfig;
use crate::terrain::tiling::TileId;
use glam::{DVec3, Vec2};

#[derive(Debug)]
enum ClipmapFrame {
    Flat(Vec2),
    #[cfg(feature = "enable-globe")]
    Globe {
        camera: super::globe::GlobeFrame,
        center_to_ecef: glam::DMat4,
    },
}

/// Bounds for a mesh region (start index, index count).
#[derive(Debug, Clone, Copy)]
pub struct MeshBounds {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
}

/// Complete clipmap mesh data ready for GPU upload.
#[derive(Debug)]
pub struct ClipmapMesh {
    pub vertices: Vec<ClipmapVertex>,
    pub indices: Vec<u32>,
    pub center_bounds: MeshBounds,
    pub ring_bounds: Vec<MeshBounds>,
    pub triangle_count: u32,
}

impl ClipmapMesh {
    /// Get total vertex count.
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Get total index count.
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Calculate triangle reduction percentage vs a full-resolution grid.
    pub fn triangle_reduction_percent(&self, full_res_triangles: u32) -> f32 {
        if full_res_triangles == 0 {
            return 0.0;
        }
        let reduction =
            (full_res_triangles as f32 - self.triangle_count as f32) / full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }
}

/// Complete clipmap level managing center block and all LOD rings.
#[derive(Debug)]
pub struct ClipmapLevel {
    pub config: ClipmapConfig,
    pub center: DVec3,
    pub terrain_extent: f32,
    pub base_cell_size: f32,
    frame: ClipmapFrame,
    mesh: Option<ClipmapMesh>,
}

impl ClipmapLevel {
    /// Create a new clipmap level centered at the given world position.
    pub fn new(config: ClipmapConfig, center: Vec2, terrain_extent: f32) -> Self {
        let base_cell_size = terrain_extent / (config.center_resolution as f32 * 8.0);
        Self {
            config,
            center: DVec3::new(f64::from(center.x), 0.0, f64::from(center.y)),
            terrain_extent,
            base_cell_size,
            frame: ClipmapFrame::Flat(center),
            mesh: None,
        }
    }

    /// Create a globe clipmap centered in f64 ECEF coordinates.
    #[cfg(feature = "enable-globe")]
    pub fn new_globe(
        config: ClipmapConfig,
        center_ecef: DVec3,
        frame: super::globe::GlobeFrame,
        terrain_extent: f32,
    ) -> Option<Self> {
        let center_distance = center_ecef.length();
        if frame.mode() != super::globe::GlobeMode::Globe
            || !center_ecef.is_finite()
            || !center_distance.is_finite()
            || center_distance == 0.0
            || !terrain_extent.is_finite()
            || terrain_extent <= 0.0
            || config.center_resolution == 0
            || config.ring_resolution == 0
        {
            return None;
        }
        let base_cell_size = Self::new(config.clone(), Vec2::ZERO, terrain_extent).base_cell_size;
        let center_to_ecef = super::globe::GlobeFrame::tangent_to_ecef(center_ecef)?;
        Some(Self {
            config,
            center: center_ecef,
            terrain_extent,
            base_cell_size,
            frame: ClipmapFrame::Globe {
                camera: frame,
                center_to_ecef,
            },
            mesh: None,
        })
    }

    /// Generate or regenerate the clipmap mesh.
    pub fn generate(&mut self) -> &ClipmapMesh {
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut ring_bounds = Vec::new();
        let generation_center = self.generation_center();

        // Generate center block
        let center_half = self.base_cell_size * self.config.center_resolution as f32 * 0.5;
        let (center_verts, center_indices) = make_center_block(
            self.config.center_resolution,
            generation_center,
            center_half,
            self.terrain_extent,
        );

        let center_bounds = MeshBounds {
            vertex_start: 0,
            vertex_count: center_verts.len() as u32,
            index_start: 0,
            index_count: center_indices.len() as u32,
        };

        all_vertices.extend(center_verts);
        all_indices.extend(center_indices);

        // Generate rings from innermost (finest LOD) to outermost (coarsest)
        let mut current_inner = center_half;
        for ring_idx in 0..self.config.ring_count {
            let ring_extent = self.config.ring_extent(ring_idx, self.base_cell_size);
            let current_outer = current_inner + ring_extent;

            let vertex_start = all_vertices.len() as u32;
            let index_start = all_indices.len() as u32;

            let (mut ring_verts, mut ring_indices) = make_ring(
                ring_idx,
                current_inner,
                current_outer,
                self.config.ring_resolution,
                generation_center,
                self.terrain_extent,
                self.config.morph_range,
            );
            let (skirt_verts, skirt_indices) = make_ring_skirts(
                &ring_verts,
                &ring_indices,
                self.config.skirt_depth,
                ring_idx,
                self.config.ring_resolution as usize + 1,
            );
            ring_verts.extend(skirt_verts);
            ring_indices.extend(skirt_indices);

            // Offset indices by current vertex count
            let offset_indices: Vec<u32> = ring_indices.iter().map(|&i| i + vertex_start).collect();

            ring_bounds.push(MeshBounds {
                vertex_start,
                vertex_count: ring_verts.len() as u32,
                index_start,
                index_count: ring_indices.len() as u32,
            });

            all_vertices.extend(ring_verts);
            all_indices.extend(offset_indices);

            current_inner = current_outer;
        }

        let triangle_count = all_indices.len() as u32 / 3;
        self.rebase_globe_vertices(&mut all_vertices);

        self.mesh = Some(ClipmapMesh {
            vertices: all_vertices,
            indices: all_indices,
            center_bounds,
            ring_bounds,
            triangle_count,
        });

        self.mesh.as_ref().unwrap()
    }

    fn generation_center(&self) -> Vec2 {
        match self.frame {
            ClipmapFrame::Flat(center) => center,
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { .. } => Vec2::ZERO,
        }
    }

    fn rebase_globe_vertices(&self, vertices: &mut [ClipmapVertex]) {
        #[cfg(feature = "enable-globe")]
        if let ClipmapFrame::Globe { camera, .. } = self.frame {
            for vertex in vertices {
                let local_x = vertex.position[0];
                let local_y = vertex.position[1];
                let local = DVec3::new(f64::from(local_x), f64::from(local_y), 0.0);
                let world = self
                    .globe_vertex_world(local)
                    .expect("globe frame owns a center tangent transform");
                let render = camera
                    .camera_relative(world)
                    .expect("validated globe geometry stays finite");
                let direction = world.normalize();
                let uv = crate::camera::Anchor::direction_to_render(DVec3::new(
                    (direction.y.atan2(direction.x) / std::f64::consts::TAU + 0.5).rem_euclid(1.0),
                    (0.5 - direction.z.asin() / std::f64::consts::PI).clamp(0.0, 1.0),
                    0.0,
                ));
                vertex.uv = [uv.x, uv.y];
                vertex.set_globe_position(render.position, render.up);
            }
        }
        #[cfg(not(feature = "enable-globe"))]
        let _ = vertices;
    }

    /// Get the generated mesh, generating if needed.
    pub fn mesh(&mut self) -> &ClipmapMesh {
        if self.mesh.is_none() {
            self.generate();
        }
        self.mesh.as_ref().unwrap()
    }

    /// Update the clipmap center position.
    /// Returns list of TileIds that should be requested for streaming.
    pub fn update_center(&mut self, new_center: Vec2) -> Vec<TileId> {
        let current_center = match &mut self.frame {
            ClipmapFrame::Flat(center) => center,
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { .. } => return Vec::new(),
        };
        let delta = new_center - *current_center;

        // Only regenerate if moved significantly (half a cell)
        if delta.length() < self.base_cell_size * 0.5 {
            return Vec::new();
        }

        *current_center = new_center;
        self.center = DVec3::new(f64::from(new_center.x), 0.0, f64::from(new_center.y));
        self.mesh = None; // Force regeneration

        // Calculate which tiles are needed for each LOD level
        self.calculate_required_tiles()
    }

    /// Move a globe clipmap patch to an f64 ECEF center.
    #[cfg(feature = "enable-globe")]
    pub fn update_globe_center(&mut self, new_center_ecef: DVec3) -> Vec<TileId> {
        if !new_center_ecef.is_finite()
            || new_center_ecef.length_squared() == 0.0
            || (new_center_ecef - self.center).length() < f64::from(self.base_cell_size * 0.5)
        {
            return Vec::new();
        }
        let Some(center_to_ecef) = super::globe::GlobeFrame::tangent_to_ecef(new_center_ecef)
        else {
            return Vec::new();
        };
        let ClipmapFrame::Globe { camera, .. } = self.frame else {
            return Vec::new();
        };
        self.center = new_center_ecef;
        self.frame = ClipmapFrame::Globe {
            camera,
            center_to_ecef,
        };
        self.mesh = None;
        self.calculate_required_tiles()
    }

    /// Calculate tiles required for current clipmap position.
    pub fn calculate_required_tiles(&self) -> Vec<TileId> {
        let mut tiles = Vec::new();
        let center = match &self.frame {
            ClipmapFrame::Flat(center) => *center,
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { .. } => return self.calculate_required_globe_tiles(),
        };

        // Center block tiles (LOD 0)
        let center_tile = self.world_to_tile(center, 0);
        tiles.push(center_tile);

        // Ring tiles
        let mut current_inner = self.base_cell_size * self.config.center_resolution as f32 * 0.5;
        for ring_idx in 0..self.config.ring_count {
            let lod = self.config.ring_lod(ring_idx);
            let ring_extent = self.config.ring_extent(ring_idx, self.base_cell_size);
            let current_outer = current_inner + ring_extent;

            // Sample tiles at ring corners and edges
            let corners = [
                center + Vec2::new(-current_outer, -current_outer),
                center + Vec2::new(current_outer, -current_outer),
                center + Vec2::new(-current_outer, current_outer),
                center + Vec2::new(current_outer, current_outer),
            ];

            for corner in &corners {
                let tile = self.world_to_tile(*corner, lod);
                if !tiles.contains(&tile) {
                    tiles.push(tile);
                }
            }

            current_inner = current_outer;
        }

        tiles
    }

    #[cfg(feature = "enable-globe")]
    fn calculate_required_globe_tiles(&self) -> Vec<TileId> {
        let mut tiles = Vec::new();
        let mut outer_extent = self
            .config
            .ring_bounds(0, self.base_cell_size, Vec2::ZERO)
            .0;
        self.append_globe_tiles(0, f64::from(outer_extent), &mut tiles);
        for ring in 0..self.config.ring_count {
            outer_extent += self.config.ring_extent(ring, self.base_cell_size);
            self.append_globe_tiles(self.ring_lod(ring), f64::from(outer_extent), &mut tiles);
        }
        tiles
    }

    #[cfg(feature = "enable-globe")]
    fn append_globe_tiles(&self, lod: u32, patch_radius_m: f64, tiles: &mut Vec<TileId>) {
        let direction = self.center.normalize();
        let longitude = direction.y.atan2(direction.x);
        let latitude = direction.z.asin();
        let angular_radius =
            (patch_radius_m / self.center.length()).clamp(0.0, std::f64::consts::PI);
        let longitude_radius = if latitude.abs() + angular_radius >= std::f64::consts::FRAC_PI_2 {
            std::f64::consts::PI
        } else {
            (angular_radius / latitude.cos().abs().max(1.0e-12)).min(std::f64::consts::PI)
        };
        let tile_count = 1_u32.checked_shl(lod.min(31)).unwrap_or(u32::MAX);
        let count = i64::from(tile_count);
        let scale = f64::from(tile_count);
        let x_min = ((longitude - longitude_radius) / std::f64::consts::TAU + 0.5) * scale;
        let x_max = ((longitude + longitude_radius) / std::f64::consts::TAU + 0.5) * scale;
        let y_min = ((0.5
            - (latitude + angular_radius).min(std::f64::consts::FRAC_PI_2) / std::f64::consts::PI)
            * scale)
            .floor()
            .clamp(0.0, scale - 1.0) as i64;
        let y_max = ((0.5
            - (latitude - angular_radius).max(-std::f64::consts::FRAC_PI_2) / std::f64::consts::PI)
            * scale)
            .floor()
            .clamp(0.0, scale - 1.0) as i64;

        for raw_x in x_min.floor() as i64..=x_max.floor() as i64 {
            let x = raw_x.rem_euclid(count) as u32;
            for y in y_min..=y_max {
                let tile = TileId::new(lod, x, y as u32);
                if !tiles.contains(&tile) {
                    tiles.push(tile);
                }
            }
        }
    }

    /// Authoritative f64 center used by globe and flat modes.
    pub fn center_ecef(&self) -> DVec3 {
        self.center
    }

    /// Camera-relative 2D center consumed by the current clipmap vertex layout.
    pub fn render_center(&self) -> Vec2 {
        match self.frame {
            ClipmapFrame::Flat(center) => center,
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { camera, .. } => {
                let position = camera
                    .camera_relative(self.center)
                    .expect("validated globe center stays finite")
                    .position;
                Vec2::new(position.x, position.y)
            }
        }
    }

    /// Re-anchor globe geometry after more than half a finest cell of motion.
    #[cfg(feature = "enable-globe")]
    pub fn recenter(&mut self, camera_anchor: DVec3) -> bool {
        let ClipmapFrame::Globe {
            camera,
            center_to_ecef,
        } = self.frame
        else {
            return false;
        };
        let displacement = (camera_anchor - camera.camera_anchor()).length();
        if !displacement.is_finite() || displacement <= f64::from(self.base_cell_size * 0.5) {
            return false;
        }
        let Some(camera) = camera.reanchored(camera_anchor) else {
            return false;
        };
        self.frame = ClipmapFrame::Globe {
            camera,
            center_to_ecef,
        };
        self.mesh = None;
        true
    }

    #[cfg(feature = "enable-globe")]
    fn globe_vertex_world(&self, local: DVec3) -> Option<DVec3> {
        match self.frame {
            ClipmapFrame::Flat(_) => None,
            ClipmapFrame::Globe { center_to_ecef, .. } => {
                if !local.is_finite() {
                    return None;
                }
                let center_radius = self.center.length();
                let radius = center_radius + local.z;
                if !center_radius.is_finite() || radius <= 0.0 {
                    return None;
                }
                let horizontal = DVec3::new(local.x, local.y, 0.0);
                let distance = horizontal.length();
                let center_up = self.center / center_radius;
                let up = if distance == 0.0 {
                    center_up
                } else {
                    let tangent = center_to_ecef.transform_vector3(horizontal) / distance;
                    let angle = distance / center_radius;
                    center_up * angle.cos() + tangent * angle.sin()
                };
                Some(up.normalize() * radius)
            }
        }
    }

    #[cfg(feature = "enable-globe")]
    pub fn camera_anchor(&self) -> Option<DVec3> {
        match self.frame {
            ClipmapFrame::Flat(_) => None,
            ClipmapFrame::Globe { camera, .. } => Some(camera.camera_anchor()),
        }
    }

    /// Convert world position to tile ID at given LOD level.
    fn world_to_tile(&self, pos: Vec2, lod: u32) -> TileId {
        let tile_size = self.terrain_extent / (1 << lod) as f32;
        let normalized = (pos + Vec2::splat(self.terrain_extent * 0.5)) / tile_size;
        TileId::new(
            lod,
            normalized.x.floor().max(0.0) as u32,
            normalized.y.floor().max(0.0) as u32,
        )
    }

    /// Get LOD level for a given ring index.
    pub fn ring_lod(&self, ring_index: u32) -> u32 {
        self.config.ring_lod(ring_index)
    }

    /// Calculate triangle count for a full-resolution grid (for reduction comparison).
    pub fn full_resolution_triangle_count(&self) -> u32 {
        // Full terrain at finest LOD
        let total_cells = self.config.center_resolution * 4; // Approximate coverage
        total_cells * total_cells * 2
    }
}

/// Generate a complete clipmap mesh from configuration.
pub fn clipmap_generate(config: &ClipmapConfig, center: Vec2, terrain_extent: f32) -> ClipmapMesh {
    let mut level = ClipmapLevel::new(config.clone(), center, terrain_extent);
    level.generate();
    level.mesh.take().unwrap()
}

/// Calculate triangle reduction percentage.
pub fn calculate_triangle_reduction(full_res_triangles: u32, clipmap_triangles: u32) -> f32 {
    if full_res_triangles == 0 {
        return 0.0;
    }
    ((full_res_triangles as f32 - clipmap_triangles as f32) / full_res_triangles as f32).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_clipmap_level_creation() {
        let config = ClipmapConfig::new(4, 64);
        let level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        assert_eq!(level.config.ring_count, 4);
        assert_eq!(level.center, DVec3::ZERO);
    }

    #[test]
    fn test_clipmap_mesh_generation() {
        let config = ClipmapConfig::new(4, 32);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let mesh = level.generate();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.index_count() > 0);
        assert_eq!(mesh.index_count() % 3, 0);
        assert_eq!(mesh.ring_bounds.len(), 4);
    }

    #[test]
    fn test_triangle_reduction_meets_40_percent() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let full_res = level.full_resolution_triangle_count();
        let mesh = level.generate();

        // Compare against full-res grid
        let reduction = mesh.triangle_reduction_percent(full_res);

        // P2.1 exit criteria: ≥40% reduction
        assert!(
            reduction >= 40.0,
            "Triangle reduction {:.1}% should be >= 40%",
            reduction
        );
    }

    #[test]
    fn test_center_update_triggers_tile_requests() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);

        // Large movement should trigger tile requests
        let tiles = level.update_center(Vec2::new(100.0, 100.0));
        assert!(!tiles.is_empty());
    }

    #[test]
    fn test_small_center_update_no_regeneration() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        level.generate();

        // Small movement should not regenerate
        let tiles = level.update_center(Vec2::new(0.1, 0.1));
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_clipmap_generate_function() {
        let config = ClipmapConfig::new(4, 32);
        let mesh = clipmap_generate(&config, Vec2::ZERO, 1000.0);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count > 0);
    }

    #[test]
    fn flat_mesh_bytes_remain_compatible() {
        let config = ClipmapConfig {
            ring_count: 2,
            ring_resolution: 4,
            center_resolution: 4,
            skirt_depth: 10.0,
            morph_range: 0.3,
        };
        let mesh = clipmap_generate(&config, Vec2::new(125.25, -75.5), 1000.0);
        let mut digest = Sha256::new();
        for vertex in &mesh.vertices {
            digest.update(bytemuck::cast_slice(&[
                vertex.position[0],
                vertex.position[1],
                vertex.uv[0],
                vertex.uv[1],
                vertex.morph_data[0],
                vertex.morph_data[1],
            ]));
        }
        digest.update(bytemuck::cast_slice(&mesh.indices));
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "43ba5df020284c48fd185ed5f7f643234fe47941bcaada047aa753ca1500a9e2"
        );
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_vertices_are_rebased_per_vertex_at_planet_scale() {
        use crate::terrain::clipmap::globe::GlobeFrame;
        use glam::DVec3;

        let seed = GlobeFrame::globe(DVec3::X * GlobeFrame::WGS84_MEAN_RADIUS_M).unwrap();
        let center = seed.lonlat_alt_to_ecef(-121.7603, 46.8523, 0.0).unwrap();
        let camera = center + center.normalize() * 1_000.0;
        let frame = GlobeFrame::globe(camera).unwrap();
        let config = ClipmapConfig {
            ring_count: 0,
            ring_resolution: 2,
            center_resolution: 2,
            ..ClipmapConfig::default()
        };
        let mut level = ClipmapLevel::new_globe(config, center, frame, 0.004).unwrap();
        let mesh = level.generate();

        let expected_center = frame.camera_relative(center).unwrap().position;
        assert!((glam::Vec3::from(mesh.vertices[4].position) - expected_center).length() < 1.0e-7);
        let left = mesh.vertices[3].position[0];
        let right = mesh.vertices[5].position[0];
        assert!((left + 0.000_25).abs() < 1.0e-7, "left={left}");
        assert!((right - 0.000_25).abs() < 1.0e-7, "right={right}");
        assert!(mesh
            .vertices
            .iter()
            .all(|vertex| glam::Vec3::from(vertex.position).is_finite()));
        let expected_uv = crate::camera::Anchor::direction_to_render(DVec3::new(
            (-121.7603 / 360.0 + 0.5_f64).rem_euclid(1.0),
            0.5 - 46.8523 / 180.0,
            0.0,
        ));
        assert!((mesh.vertices[4].uv[0] - expected_uv.x).abs() < 1.0e-6);
        assert!((mesh.vertices[4].uv[1] - expected_uv.y).abs() < 1.0e-6);
        assert!(
            (mesh.vertices[4].geodetic_up() - frame.camera_relative(center).unwrap().up).length()
                < 1.0e-6
        );
        assert_eq!(level.center_ecef(), center);

        let curved = level
            .globe_vertex_world(DVec3::new(1_000.0, 1_000.0, 0.0))
            .unwrap();
        assert!((curved.length() - center.length()).abs() < 1.0e-6);
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_arc_length_uses_elevated_patch_radius() {
        use crate::terrain::clipmap::globe::GlobeFrame;

        let frame =
            GlobeFrame::globe(DVec3::X * (GlobeFrame::WGS84_MEAN_RADIUS_M + 2_000.0)).unwrap();
        let center = DVec3::X * (GlobeFrame::WGS84_MEAN_RADIUS_M + 1_000.0);
        let mut level =
            ClipmapLevel::new_globe(ClipmapConfig::new(1, 4), center, frame, 20_000.0).unwrap();
        let offset = 10_000.0;
        let world = level
            .globe_vertex_world(DVec3::new(offset, 0.0, 0.0))
            .unwrap();
        let angle = (center.normalize().dot(world.normalize()))
            .clamp(-1.0, 1.0)
            .acos();
        assert!((angle * center.length() - offset).abs() < 1.0e-6);
        assert!(level
            .generate()
            .vertices
            .iter()
            .all(|vertex| vertex.is_globe()));
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_recenter_uses_strict_half_cell_threshold() {
        use crate::terrain::clipmap::globe::GlobeFrame;
        use glam::DVec3;

        let camera = DVec3::X * (GlobeFrame::WGS84_MEAN_RADIUS_M + 1_000.0);
        let frame = GlobeFrame::globe(camera).unwrap();
        let config = ClipmapConfig::new(1, 4);
        let mut level =
            ClipmapLevel::new_globe(config, DVec3::X * 6_371_000.0, frame, 1000.0).unwrap();
        let threshold = level.base_cell_size * 0.5;
        let center_index = 12;
        let before = level.generate().vertices[center_index].position;
        let uv_before = level.mesh().vertices[center_index].uv;
        let fixed_local = DVec3::new(125.0, -40.0, 0.0);
        let world_before = level.globe_vertex_world(fixed_local).unwrap();

        assert!(!level.recenter(camera + DVec3::Y * f64::from(threshold)));
        assert_eq!(level.camera_anchor(), Some(camera));
        assert!(level.recenter(camera + DVec3::Y * f64::from(threshold + 0.01)));
        assert_eq!(
            level.camera_anchor(),
            Some(camera + DVec3::Y * f64::from(threshold + 0.01))
        );
        assert_ne!(level.mesh().vertices[center_index].position, before);
        assert_eq!(level.mesh().vertices[center_index].uv, uv_before);
        let world_after = level.globe_vertex_world(fixed_local).unwrap();
        assert!(
            (world_after - world_before).length() < 1.0e-9,
            "fixed patch rotated by {} m",
            (world_after - world_before).length()
        );
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_center_update_moves_patch_and_requests_tiles() {
        use crate::terrain::clipmap::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let camera = DVec3::X * (radius + 1_000.0);
        let frame = GlobeFrame::globe(camera).unwrap();
        let mut level =
            ClipmapLevel::new_globe(ClipmapConfig::new(2, 4), DVec3::X * radius, frame, 10_000.0)
                .unwrap();
        let next_center = frame.lonlat_alt_to_ecef(1.0, 0.0, 0.0).unwrap();
        let tiles = level.update_globe_center(next_center);

        assert_eq!(level.center_ecef(), next_center);
        assert!(!tiles.is_empty());
        assert!(level
            .generate()
            .vertices
            .iter()
            .all(|vertex| vertex.is_globe()));
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_tile_demand_covers_patch_across_boundaries() {
        use crate::terrain::clipmap::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let center = DVec3::X * radius;
        let frame = GlobeFrame::globe(center + DVec3::X * 1_000.0).unwrap();
        let level =
            ClipmapLevel::new_globe(ClipmapConfig::new(2, 4), center, frame, 10_000.0).unwrap();
        let tiles = level.calculate_required_tiles();

        assert!(
            tiles.iter().filter(|tile| tile.lod == 1).count() >= 4,
            "equator/prime-meridian patch must cover all intersecting LOD-1 quadrants"
        );
    }
}
