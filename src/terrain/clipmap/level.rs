//! P2.1/M5: Complete clipmap level with center block and nested rings.

use super::ring::{make_center_block, make_ring, make_ring_skirts};
#[cfg(feature = "enable-globe")]
use super::ring::{globe_ring_bounds, make_ring_with_cell_size};
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

fn snap_center_to_finest_grid(center: Vec2, base_cell_size: f32) -> Vec2 {
    if !base_cell_size.is_finite() || base_cell_size <= 0.0 {
        return center;
    }
    (center / base_cell_size).round() * base_cell_size
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
#[derive(Debug, Clone)]
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
    pub fn generate(&mut self) -> Result<&ClipmapMesh, String> {
        self.validate_generation_inputs()?;
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
            let historical_outer = current_inner + ring_extent;

            #[cfg(feature = "enable-globe")]
            let (ring_inner, ring_outer) = match self.frame {
                ClipmapFrame::Globe { .. } => globe_ring_bounds(
                    ring_idx,
                    self.base_cell_size,
                    self.config.center_resolution,
                    self.config.ring_resolution,
                )?,
                ClipmapFrame::Flat(_) => (current_inner, historical_outer),
            };
            #[cfg(not(feature = "enable-globe"))]
            let (ring_inner, ring_outer) = (current_inner, historical_outer);

            let vertex_start = all_vertices.len() as u32;
            let index_start = all_indices.len() as u32;

            let (mut ring_verts, mut ring_indices) = match self.frame {
                #[cfg(feature = "enable-globe")]
                ClipmapFrame::Globe { .. } => make_ring_with_cell_size(
                    ring_idx,
                    ring_inner,
                    ring_outer,
                    self.config.ring_resolution,
                    generation_center,
                    self.terrain_extent,
                    self.config.morph_range,
                    self.base_cell_size * 2.0_f32.powi(ring_idx as i32 + 1),
                ),
                ClipmapFrame::Flat(_) => make_ring(
                    ring_idx,
                    ring_inner,
                    ring_outer,
                    self.config.ring_resolution,
                    generation_center,
                    self.terrain_extent,
                    self.config.morph_range,
                ),
            };
            let (skirt_verts, skirt_indices) = make_ring_skirts(
                &ring_verts,
                &ring_indices,
                self.config.skirt_depth,
                ring_idx,
                0,
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

            current_inner = historical_outer;
        }

        let triangle_count = all_indices.len() as u32 / 3;
        self.rebase_globe_vertices(&mut all_vertices)?;

        self.mesh = Some(ClipmapMesh {
            vertices: all_vertices,
            indices: all_indices,
            center_bounds,
            ring_bounds,
            triangle_count,
        });

        self.mesh
            .as_ref()
            .ok_or_else(|| "clipmap mesh generation did not retain its result".to_string())
    }

    fn generation_center(&self) -> Vec2 {
        match self.frame {
            ClipmapFrame::Flat(center) => center,
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { .. } => Vec2::ZERO,
        }
    }

    fn validate_generation_inputs(&self) -> Result<(), String> {
        let center = self.generation_center();
        if !center.is_finite() || !self.center.is_finite() {
            return Err("clipmap center must be finite".to_string());
        }
        if !self.terrain_extent.is_finite() || self.terrain_extent <= 0.0 {
            return Err("clipmap terrain extent must be finite and positive".to_string());
        }
        if !self.base_cell_size.is_finite() || self.base_cell_size <= 0.0 {
            return Err("clipmap base cell size must be finite and positive".to_string());
        }
        if self.config.center_resolution == 0 || self.config.ring_resolution == 0 {
            return Err("clipmap center and ring resolutions must be positive".to_string());
        }
        if self.config.ring_count > 31 {
            return Err("clipmap ring count exceeds the checked LOD shift range".to_string());
        }
        if !self.config.skirt_depth.is_finite() || !self.config.morph_range.is_finite() {
            return Err("clipmap skirt depth and morph range must be finite".to_string());
        }

        // Reject address-space overflow before make_center_block/make_ring can
        // assert or reserve. The conservative ring bound also includes skirt
        // duplicates and moving-boundary splice coordinates.
        let center_side = u64::from(self.config.center_resolution)
            .checked_add(1)
            .ok_or_else(|| "clipmap center resolution is out of range".to_string())?;
        let center_vertices = center_side
            .checked_mul(center_side)
            .ok_or_else(|| "clipmap center vertex count is out of range".to_string())?;
        let ring_side = u64::from(self.config.center_resolution)
            .checked_add(u64::from(self.config.ring_resolution).saturating_mul(2))
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| "clipmap ring resolution is out of range".to_string())?;
        let ring_vertices = ring_side
            .checked_mul(ring_side)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(u64::from(self.config.ring_count)))
            .ok_or_else(|| "clipmap ring vertex count is out of range".to_string())?;
        let total_vertices = center_vertices
            .checked_add(ring_vertices)
            .ok_or_else(|| "clipmap total vertex count is out of range".to_string())?;
        if total_vertices > u64::from(u32::MAX) {
            return Err("clipmap configuration exceeds the u32 mesh address space".to_string());
        }

        let center_half =
            f64::from(self.base_cell_size) * f64::from(self.config.center_resolution) * 0.5;
        if !center_half.is_finite() || center_half <= 0.0 {
            return Err("clipmap center extent is out of range".to_string());
        }
        let mut current_inner = center_half;
        for ring in 0..self.config.ring_count {
            let scale = 1_u32
                .checked_shl(ring)
                .ok_or_else(|| "clipmap ring LOD shift is out of range".to_string())?;
            let width = f64::from(self.base_cell_size)
                * f64::from(scale)
                * f64::from(self.config.ring_resolution);
            let outer = current_inner + width;
            if !width.is_finite()
                || width <= 0.0
                || !outer.is_finite()
                || outer <= current_inner
            {
                return Err(format!("clipmap ring {ring} extent is out of range"));
            }
            current_inner = outer;
        }
        Ok(())
    }

    pub(crate) fn rebase_globe_vertices(
        &self,
        vertices: &mut [ClipmapVertex],
    ) -> Result<(), String> {
        #[cfg(feature = "enable-globe")]
        if let ClipmapFrame::Globe { camera, .. } = self.frame {
            for vertex in vertices {
                let local_x = vertex.position[0];
                let local_y = vertex.position[1];
                let local = DVec3::new(f64::from(local_x), f64::from(local_y), 0.0);
                let world = self.globe_vertex_world(local).ok_or_else(|| {
                    "globe clipmap vertex could not be projected into ECEF".to_string()
                })?;
                let render = camera
                    .camera_relative(world)
                    .map_err(|error| format!("globe clipmap vertex rebase failed: {error}"))?;
                let direction = world.normalize();
                let uv = crate::camera::Anchor::direction_to_render(DVec3::new(
                    (direction.y.atan2(direction.x) / std::f64::consts::TAU + 0.5).rem_euclid(1.0),
                    (0.5 - direction.z.asin() / std::f64::consts::PI).clamp(0.0, 1.0),
                    0.0,
                ));
                vertex.uv = [uv.x, uv.y];
                vertex.set_globe_position(render.position, render.up);
                if vertex.is_skirt() {
                    vertex.morph_data[0] =
                        -self.curvature_safe_skirt_depth_for_ring(vertex.ring_index());
                }
            }
        }
        #[cfg(not(feature = "enable-globe"))]
        let _ = vertices;
        Ok(())
    }

    #[cfg(feature = "enable-globe")]
    fn curvature_safe_skirt_depth_for_ring(&self, ring_index: u32) -> f32 {
        let radius = self.center.length() as f32;
        let ClipmapFrame::Globe { camera, .. } = self.frame else {
            unreachable!("curvature-safe skirt depth is globe-only")
        };
        let altitude = (camera.camera_anchor().length() - camera.radius()).max(0.0) as f32;
        let (_, outer_extent) =
            self.config
                .ring_bounds(ring_index, self.base_cell_size, self.generation_center());
        super::ring::curvature_safe_skirt_depth(
            self.config.skirt_depth,
            radius,
            outer_extent.max(0.0) * 2.0,
            altitude,
        )
    }

    /// Get the generated mesh, generating if needed.
    pub fn mesh(&mut self) -> Result<&ClipmapMesh, String> {
        if self.mesh.is_none() {
            self.generate()?;
        }
        self.mesh
            .as_ref()
            .ok_or_else(|| "clipmap mesh is unavailable after generation".to_string())
    }

    /// Update the clipmap center position.
    /// Returns list of TileIds that should be requested for streaming.
    #[allow(clippy::infallible_destructuring_match)]
    pub fn update_center(&mut self, new_center: Vec2) -> Vec<TileId> {
        let new_center = snap_center_to_finest_grid(new_center, self.base_cell_size);
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

    /// Camera-centred cumulative clipmap footprints expressed directly at the
    /// streaming target LOD. The centre region is first and every region is
    /// ordered nearest-to-camera first; no coarse parent is expanded into its
    /// planet-wide descendant set.
    #[cfg(feature = "enable-globe")]
    pub(crate) fn globe_target_lod_footprints(
        &self,
        target_lod: u32,
    ) -> Result<Vec<Vec<TileId>>, String> {
        let mut radii = Vec::with_capacity(self.config.ring_count as usize + 1);
        let mut outer_extent = self
            .config
            .ring_bounds(0, self.base_cell_size, Vec2::ZERO)
            .0;
        radii.push(outer_extent);
        for ring in 0..self.config.ring_count {
            outer_extent += self.config.ring_extent(ring, self.base_cell_size);
            radii.push(outer_extent);
        }
        let direction = self.center.normalize();
        let longitude = direction.y.atan2(direction.x);
        let latitude = direction.z.asin();
        let axis = 1_u32.checked_shl(target_lod).ok_or_else(|| {
            format!("globe target LOD {target_lod} exceeds the u32 tile address space")
        })?;
        let center_x = (((longitude / std::f64::consts::TAU + 0.5) * f64::from(axis)).floor()
            as i64)
            .rem_euclid(i64::from(axis)) as u32;
        let center_y = ((0.5 - latitude / std::f64::consts::PI) * f64::from(axis))
            .floor()
            .clamp(0.0, f64::from(axis - 1)) as u32;
        Ok(radii
            .into_iter()
            .map(|radius| {
                let mut tiles = Vec::new();
                self.append_globe_tiles(target_lod, f64::from(radius), &mut tiles);
                tiles.sort_by_key(|tile| {
                    let direct = tile.x.abs_diff(center_x);
                    let dx = direct.min(axis - direct);
                    let dy = tile.y.abs_diff(center_y);
                    (u64::from(dx) * u64::from(dx) + u64::from(dy) * u64::from(dy), dy, dx, tile.y, tile.x)
                });
                tiles
            })
            .collect())
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
        let Some(tile_count) = 1_u32.checked_shl(lod) else {
            return;
        };
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
    pub fn render_center(&self) -> Result<Vec2, String> {
        match self.frame {
            ClipmapFrame::Flat(center) => Ok(center),
            #[cfg(feature = "enable-globe")]
            ClipmapFrame::Globe { camera, .. } => {
                let position = camera
                    .camera_relative(self.center)
                    .map_err(|error| format!("globe clipmap render center is invalid: {error}"))?
                    .position;
                Ok(Vec2::new(position.x, position.y))
            }
        }
    }

    /// Re-anchor globe geometry after more than half a finest cell of motion.
    #[cfg(feature = "enable-globe")]
    pub fn recenter(&mut self, camera_anchor: DVec3) -> Result<bool, String> {
        if !camera_anchor.is_finite() {
            return Err("globe clipmap camera anchor must be finite".to_string());
        }
        let ClipmapFrame::Globe {
            camera,
            center_to_ecef,
        } = self.frame
        else {
            return Ok(false);
        };
        let displacement = (camera_anchor - camera.camera_anchor()).length();
        if !displacement.is_finite() {
            return Err("globe clipmap camera displacement is out of range".to_string());
        }
        if displacement <= f64::from(self.base_cell_size * 0.5) {
            return Ok(false);
        }
        let camera = camera
            .reanchored(camera_anchor)
            .map_err(|error| format!("globe clipmap reanchor failed: {error}"))?;
        self.frame = ClipmapFrame::Globe {
            camera,
            center_to_ecef,
        };
        self.mesh = None;
        Ok(true)
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

    #[cfg(feature = "enable-globe")]
    pub(crate) fn globe_frame(&self) -> Option<super::globe::GlobeFrame> {
        match self.frame {
            ClipmapFrame::Flat(_) => None,
            ClipmapFrame::Globe { camera, .. } => Some(camera),
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
        full_resolution_triangle_count(&self.config)
    }
}

/// Triangles required to cover the complete clipmap footprint at the finest
/// lattice. This is the meaningful comparator for a nested clipmap; the old
/// `center_resolution * 4` approximation covered only a small central square.
pub fn full_resolution_triangle_count(config: &ClipmapConfig) -> u32 {
    let ring_cells_per_side = (0..config.ring_count)
        .map(|ring| config.ring_resolution.checked_shl(ring).unwrap_or(u32::MAX))
        .fold(0u32, u32::saturating_add);
    let cells_per_side = config
        .center_resolution
        .saturating_add(ring_cells_per_side.saturating_mul(2));
    cells_per_side
        .saturating_mul(cells_per_side)
        .saturating_mul(2)
}

/// Generate a complete clipmap mesh from configuration.
pub fn clipmap_generate(
    config: &ClipmapConfig,
    center: Vec2,
    terrain_extent: f32,
) -> Result<ClipmapMesh, String> {
    let mut level = ClipmapLevel::new(config.clone(), center, terrain_extent);
    level.generate()?;
    level
        .mesh
        .take()
        .ok_or_else(|| "flat clipmap generation did not retain its mesh".to_string())
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
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Frozen `ClipmapVertex` ABI from origin/main at be2d6b7, before ORBIS
    /// added camera-relative Z and octahedral geodetic-up fields. Keeping this
    /// independent type in the test prevents the new layout from defining its
    /// own compatibility oracle.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct LegacyFlatVertex {
        position: [f32; 2],
        uv: [f32; 2],
        morph_data: [f32; 2],
    }

    impl LegacyFlatVertex {
        fn project(vertex: &ClipmapVertex) -> Self {
            Self {
                position: [vertex.position[0], vertex.position[1]],
                uv: vertex.uv,
                morph_data: vertex.morph_data,
            }
        }
    }

    const FLAT_FIXTURE_VIEW_PROJ: [f32; 16] = [
        0.0125, 0.001, 0.0005, 0.0, -0.0008, 0.014, 0.002, 0.0, 0.001, -0.003, 0.009,
        0.0, -0.25, 0.125, 0.5, 1.0,
    ];
    const FLAT_FIXTURE_HEIGHT_RAW: f32 = 0.625;
    const FLAT_FIXTURE_HEIGHT_MIN: f32 = -200.0;
    const FLAT_FIXTURE_HEIGHT_MAX: f32 = 800.0;
    const FLAT_FIXTURE_EXAGGERATION: f32 = 1.25;
    const FLAT_FIXTURE_RING_RESOLUTION: f32 = 4.0;

    fn legacy_flat_vertex_stage(vertex: &LegacyFlatVertex) -> [f32; 12] {
        let uv = glam::Vec2::from(vertex.uv).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
        // Fixed identity height curve and constant non-streamed overview. A
        // constant makes the fine/coarse samples equal while retaining the
        // legacy morph/skirt calculation from origin/main's WGSL.
        let h_disp = FLAT_FIXTURE_HEIGHT_MIN
            + FLAT_FIXTURE_HEIGHT_RAW * (FLAT_FIXTURE_HEIGHT_MAX - FLAT_FIXTURE_HEIGHT_MIN);
        let skirt = if vertex.morph_data[0] < 0.0 {
            FLAT_FIXTURE_RING_RESOLUTION * 0.001
        } else {
            0.0
        };
        let height_center = (FLAT_FIXTURE_HEIGHT_MIN + FLAT_FIXTURE_HEIGHT_MAX) * 0.5;
        let world_z_centered =
            (h_disp - height_center - skirt) * FLAT_FIXTURE_EXAGGERATION;
        let world_z_original = (h_disp - skirt) * FLAT_FIXTURE_EXAGGERATION;
        let centered = glam::Vec4::new(
            vertex.position[0],
            vertex.position[1],
            world_z_centered,
            1.0,
        );
        let clip = glam::Mat4::from_cols_array(&FLAT_FIXTURE_VIEW_PROJ) * centered;
        [
            clip.x,
            clip.y,
            clip.z,
            clip.w,
            vertex.position[0],
            vertex.position[1],
            world_z_original,
            0.0,
            0.0,
            1.0,
            uv.x,
            uv.y,
        ]
    }

    fn extended_flat_vertex_stage(vertex: &ClipmapVertex) -> [f32; 12] {
        let uv = glam::Vec2::from(vertex.uv).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
        let h_disp = FLAT_FIXTURE_HEIGHT_MIN
            + FLAT_FIXTURE_HEIGHT_RAW * (FLAT_FIXTURE_HEIGHT_MAX - FLAT_FIXTURE_HEIGHT_MIN);
        let skirt = if vertex.morph_data[0] < 0.0 {
            FLAT_FIXTURE_RING_RESOLUTION * 0.001
        } else {
            0.0
        };
        let height_center = (FLAT_FIXTURE_HEIGHT_MIN + FLAT_FIXTURE_HEIGHT_MAX) * 0.5;
        let world_z_centered =
            (h_disp - height_center - skirt) * FLAT_FIXTURE_EXAGGERATION;
        let world_z_original = (h_disp - skirt) * FLAT_FIXTURE_EXAGGERATION;
        let base = glam::Vec3::from(vertex.position);
        let up = vertex.geodetic_up();
        let centered = (base + up * world_z_centered).extend(1.0);
        let world = base + up * world_z_original;
        let clip = glam::Mat4::from_cols_array(&FLAT_FIXTURE_VIEW_PROJ) * centered;
        [
            clip.x, clip.y, clip.z, clip.w, world.x, world.y, world.z, up.x, up.y, up.z, uv.x,
            uv.y,
        ]
    }

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
        let mesh = level.generate().unwrap();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.index_count() > 0);
        assert_eq!(mesh.index_count() % 3, 0);
        assert_eq!(mesh.ring_bounds.len(), 4);
    }

    #[test]
    fn public_generation_rejects_invalid_inputs_without_panicking() {
        let assert_rejected = |config: ClipmapConfig, center: Vec2, terrain_extent: f32| {
            let result = catch_unwind(AssertUnwindSafe(|| {
                clipmap_generate(&config, center, terrain_extent)
            }));
            assert!(result.is_ok(), "invalid public generation input panicked");
            assert!(result.unwrap().is_err(), "invalid public generation input succeeded");
        };

        let valid = ClipmapConfig::new(1, 4);
        assert_rejected(valid.clone(), Vec2::splat(f32::NAN), 1_000.0);
        assert_rejected(valid.clone(), Vec2::ZERO, f32::INFINITY);
        assert_rejected(valid.clone(), Vec2::ZERO, 0.0);

        let mut invalid_center = valid.clone();
        invalid_center.center_resolution = 0;
        assert_rejected(invalid_center, Vec2::ZERO, 1_000.0);

        let mut invalid_ring = valid.clone();
        invalid_ring.ring_resolution = 0;
        assert_rejected(invalid_ring, Vec2::ZERO, 1_000.0);

        let mut invalid_morph = valid.clone();
        invalid_morph.morph_range = f32::NAN;
        assert_rejected(invalid_morph, Vec2::ZERO, 1_000.0);

        let mut invalid_skirt = valid.clone();
        invalid_skirt.skirt_depth = f32::INFINITY;
        assert_rejected(invalid_skirt, Vec2::ZERO, 1_000.0);

        let mut overflow = valid;
        overflow.ring_count = 32;
        overflow.ring_resolution = u32::MAX;
        overflow.center_resolution = u32::MAX;
        assert_rejected(overflow, Vec2::ZERO, 1_000.0);
    }

    #[test]
    fn test_triangle_reduction_meets_40_percent() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let full_res = level.full_resolution_triangle_count();
        let mesh = level.generate().unwrap();

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
    fn test_full_resolution_comparator_covers_outermost_ring() {
        let config = ClipmapConfig::new(4, 32);
        // 32 center cells plus two sides of 32*(1+2+4+8) ring cells.
        assert_eq!(full_resolution_triangle_count(&config), 1_968_128);
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
        level.generate().unwrap();

        // Small movement should not regenerate
        let tiles = level.update_center(Vec2::new(0.1, 0.1));
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_center_updates_snap_to_the_finest_grid() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let base_cell = level.base_cell_size;

        let requested = Vec2::new(base_cell * 0.6, -base_cell * 1.6);
        assert!(!level.update_center(requested).is_empty());
        assert_eq!(
            level.center,
            DVec3::new(f64::from(base_cell), 0.0, f64::from(-base_cell * 2.0))
        );

        // Another raw camera position inside the same snapped cell neither
        // changes the mesh lattice nor spuriously requests tiles.
        let same_cell = Vec2::new(base_cell * 0.7, -base_cell * 1.7);
        assert!(level.update_center(same_cell).is_empty());
        assert_eq!(
            level.center,
            DVec3::new(f64::from(base_cell), 0.0, f64::from(-base_cell * 2.0))
        );
    }

    #[test]
    fn test_initial_center_preserves_static_clipmap_semantics() {
        let config = ClipmapConfig::new(4, 64);
        let center = Vec2::new(12.25, -31.75);
        let level = ClipmapLevel::new(config, center, 1000.0);
        assert_eq!(level.center, DVec3::new(12.25, 0.0, -31.75));
    }

    #[test]
    fn test_clipmap_generate_function() {
        let config = ClipmapConfig::new(4, 32);
        let mesh = clipmap_generate(&config, Vec2::ZERO, 1000.0).unwrap();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count > 0);
    }

    #[test]
    fn flat_mesh_projects_to_the_immutable_pre_orbis_fixture() {
        let config = ClipmapConfig {
            ring_count: 2,
            ring_resolution: 4,
            center_resolution: 4,
            skirt_depth: 10.0,
            morph_range: 0.3,
        };
        let mesh = clipmap_generate(&config, Vec2::new(125.25, -75.5), 1000.0).unwrap();
        let legacy_vertices: Vec<LegacyFlatVertex> = mesh
            .vertices
            .iter()
            .map(LegacyFlatVertex::project)
            .collect();
        assert_eq!(std::mem::size_of::<LegacyFlatVertex>(), 24);
        assert_eq!(std::mem::size_of::<ClipmapVertex>(), 36);
        assert!(mesh.vertices.iter().any(ClipmapVertex::is_skirt));
        for (extended, legacy) in mesh.vertices.iter().zip(&legacy_vertices) {
            assert_eq!(extended.position[2].to_bits(), 0.0f32.to_bits());
            assert_eq!(extended.normal_oct.map(f32::to_bits), [0, 0]);
            assert_eq!(
                legacy.position.map(f32::to_bits),
                [
                    extended.position[0].to_bits(),
                    extended.position[1].to_bits(),
                ]
            );
            assert_eq!(legacy.uv.map(f32::to_bits), extended.uv.map(f32::to_bits));
            assert_eq!(
                legacy.morph_data.map(f32::to_bits),
                extended.morph_data.map(f32::to_bits)
            );
            assert!(!extended.is_globe());
            assert!(extended.morph_data[1] >= 0.0);
            assert_eq!(extended.morph_data[1].fract().to_bits(), 0.0f32.to_bits());
            assert_eq!(extended.morph_data[1] as u32, extended.ring_index());
            if extended.is_skirt() {
                assert_eq!(extended.morph_data[0].to_bits(), (-1.0f32).to_bits());
            }
        }
        let mut digest = Sha256::new();
        digest.update(bytemuck::cast_slice(&legacy_vertices));
        digest.update(bytemuck::cast_slice(&mesh.indices));
        // SHA-256 of the complete pre-ORBIS vertex byte stream followed by
        // the complete u32 index stream for this immutable fixture.
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "606d974cc49ef48a5821aee06c5c7c3013b67a0e58e94e2bb54a0ec2fd447b4e"
        );
    }

    #[test]
    fn flat_cpu_vertex_output_matches_pre_orbis_with_no_culling_or_streaming() {
        // Exact limitation: repository history has no pre-ORBIS backend pixel
        // golden rendered with culling="none". This is therefore the permitted
        // CPU/WGSL-output fixture, not a claim about physical raster bytes.
        let config = ClipmapConfig {
            ring_count: 2,
            ring_resolution: 4,
            center_resolution: 4,
            skirt_depth: 10.0,
            morph_range: 0.3,
        };
        let mesh = clipmap_generate(&config, Vec2::new(125.25, -75.5), 1000.0).unwrap();
        let legacy: Vec<_> = mesh
            .vertices
            .iter()
            .map(LegacyFlatVertex::project)
            .collect();
        let mut legacy_outputs = Vec::with_capacity(mesh.indices.len());
        let mut extended_outputs = Vec::with_capacity(mesh.indices.len());
        // `culling="none"`: process every index in the immutable draw order.
        // No streaming state participates; the fixed 0.625 overview sample is
        // encoded above through the origin/main identity height curve.
        for &index in &mesh.indices {
            legacy_outputs.push(legacy_flat_vertex_stage(&legacy[index as usize]));
            extended_outputs.push(extended_flat_vertex_stage(
                &mesh.vertices[index as usize],
            ));
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extended_outputs),
            bytemuck::cast_slice::<_, u8>(&legacy_outputs)
        );
        let mut digest = Sha256::new();
        digest.update(bytemuck::cast_slice(&legacy_outputs));
        digest.update(bytemuck::cast_slice(&mesh.indices));
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "9e51945b62f169f3ec14b9ac6c821de40a1bb99b2d2c216b9a2d449bab18ce25"
        );
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_vertices_are_rebased_per_vertex_at_planet_scale() {
        use crate::terrain::clipmap::globe::GlobeFrame;
        use glam::DVec3;

        let seed = GlobeFrame::globe(
            GlobeFrame::WGS84_MEAN_RADIUS_M,
            DVec3::X * GlobeFrame::WGS84_MEAN_RADIUS_M,
        )
        .unwrap();
        let center = seed.lonlat_alt_to_ecef(-121.7603, 46.8523, 0.0).unwrap();
        let camera = center + center.normalize() * 1_000.0;
        let frame = GlobeFrame::globe(GlobeFrame::WGS84_MEAN_RADIUS_M, camera).unwrap();
        let config = ClipmapConfig {
            ring_count: 0,
            ring_resolution: 2,
            center_resolution: 2,
            ..ClipmapConfig::default()
        };
        let mut level = ClipmapLevel::new_globe(config, center, frame, 0.004).unwrap();
        let mesh = level.generate().unwrap();

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

        let frame = GlobeFrame::globe(
            GlobeFrame::WGS84_MEAN_RADIUS_M,
            DVec3::X * (GlobeFrame::WGS84_MEAN_RADIUS_M + 2_000.0),
        )
        .unwrap();
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
            .unwrap()
            .vertices
            .iter()
            .all(|vertex| vertex.is_globe()));
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_skirts_cover_the_patch_sagitta_without_changing_flat_vertex_encoding() {
        // This catches a globe path that leaves the legacy one-unit skirt in
        // place, which is too shallow to cover a wide curved patch horizon.
        use crate::terrain::clipmap::globe::GlobeFrame;

        let radius = 1_000.0;
        let center = DVec3::X * radius;
        let frame = GlobeFrame::globe(radius, center + DVec3::X * 1_000.0).unwrap();
        let config = ClipmapConfig {
            ring_count: 1,
            ring_resolution: 4,
            center_resolution: 4,
            skirt_depth: 1.0,
            ..ClipmapConfig::default()
        };
        let mut level = ClipmapLevel::new_globe(config, center, frame, 100.0).unwrap();
        let mesh = level.generate().unwrap();
        let max_globe_skirt_depth = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.is_skirt())
            .map(|vertex| -vertex.morph_data[0])
            .fold(0.0_f32, f32::max);

        assert!(max_globe_skirt_depth.is_finite());
        assert!(max_globe_skirt_depth >= 1_000.0);
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn globe_recenter_uses_strict_half_cell_threshold() {
        use crate::terrain::clipmap::globe::GlobeFrame;
        use glam::DVec3;

        let camera = DVec3::X * (GlobeFrame::WGS84_MEAN_RADIUS_M + 1_000.0);
        let frame = GlobeFrame::globe(GlobeFrame::WGS84_MEAN_RADIUS_M, camera).unwrap();
        let config = ClipmapConfig::new(1, 4);
        let mut level =
            ClipmapLevel::new_globe(config, DVec3::X * 6_371_000.0, frame, 1000.0).unwrap();
        let threshold = level.base_cell_size * 0.5;
        let center_index = 12;
        let before = level.generate().unwrap().vertices[center_index].position;
        let uv_before = level.mesh().unwrap().vertices[center_index].uv;
        let fixed_local = DVec3::new(125.0, -40.0, 0.0);
        let world_before = level.globe_vertex_world(fixed_local).unwrap();

        let nan_error = level.recenter(DVec3::splat(f64::NAN)).unwrap_err();
        assert!(nan_error.contains("finite"));
        assert_eq!(level.camera_anchor(), Some(camera));
        let overflow_error = level.recenter(DVec3::splat(f64::MAX)).unwrap_err();
        assert!(overflow_error.contains("out of range"));
        assert_eq!(level.camera_anchor(), Some(camera));

        assert!(!level
            .recenter(camera + DVec3::Y * f64::from(threshold))
            .unwrap());
        assert_eq!(level.camera_anchor(), Some(camera));
        assert!(level
            .recenter(camera + DVec3::Y * f64::from(threshold + 0.01))
            .unwrap());
        assert_eq!(
            level.camera_anchor(),
            Some(camera + DVec3::Y * f64::from(threshold + 0.01))
        );
        assert_ne!(level.mesh().unwrap().vertices[center_index].position, before);
        assert_eq!(level.mesh().unwrap().vertices[center_index].uv, uv_before);
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
        let frame = GlobeFrame::globe(radius, camera).unwrap();
        let mut level =
            ClipmapLevel::new_globe(ClipmapConfig::new(2, 4), DVec3::X * radius, frame, 10_000.0)
                .unwrap();
        let next_center = frame.lonlat_alt_to_ecef(1.0, 0.0, 0.0).unwrap();
        let tiles = level.update_globe_center(next_center);

        assert_eq!(level.center_ecef(), next_center);
        assert!(!tiles.is_empty());
        assert!(level
            .generate()
            .unwrap()
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
        let frame = GlobeFrame::globe(radius, center + DVec3::X * 1_000.0).unwrap();
        let level =
            ClipmapLevel::new_globe(ClipmapConfig::new(2, 4), center, frame, 10_000.0).unwrap();
        let tiles = level.calculate_required_tiles();

        assert!(
            tiles.iter().filter(|tile| tile.lod == 1).count() >= 4,
            "equator/prime-meridian patch must cover all intersecting LOD-1 quadrants"
        );
    }

    #[cfg(feature = "enable-globe")]
    #[test]
    fn rainier_lod14_footprint_is_bounded_and_center_first() {
        use crate::terrain::clipmap::globe::GlobeFrame;

        let radius = GlobeFrame::WGS84_MEAN_RADIUS_M;
        let seed = GlobeFrame::globe(radius, DVec3::X * radius).unwrap();
        let center = seed.lonlat_alt_to_ecef(-121.7603, 46.8523, 0.0).unwrap();
        let frame = GlobeFrame::globe(radius, center + center.normalize() * 1_000.0).unwrap();
        // A fully covered LOD-14 source tile at Rainier is about 1.2 km
        // north/south. Use that real source-tile scale rather than combining
        // a coarse regional extent with an unrelated fine target LOD.
        let level =
            ClipmapLevel::new_globe(ClipmapConfig::new(4, 32), center, frame, 1_250.0).unwrap();
        let footprints = level.globe_target_lod_footprints(14).unwrap();
        let ordered = footprints.iter().flatten().copied().collect::<Vec<_>>();
        let unique = ordered
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert!(!ordered.is_empty());
        assert!(
            unique.len() <= 242,
            "complete local Rainier footprint exceeds a 256-slot atlas after 14 ancestors: {}",
            unique.len()
        );
        assert!(ordered.iter().all(|tile| tile.lod == 14));
        for regions in footprints.windows(2) {
            assert!(regions[0].iter().all(|tile| regions[1].contains(tile)));
        }
        let count = 1_u32 << 14;
        let lon = (-121.7603_f64).to_radians();
        let lat = 46.8523_f64.to_radians();
        let center_tile = TileId::new(
            14,
            (((lon / std::f64::consts::TAU + 0.5) * f64::from(count)).floor() as u32) % count,
            ((0.5 - lat / std::f64::consts::PI) * f64::from(count))
                .floor().clamp(0.0, f64::from(count - 1)) as u32,
        );
        assert_eq!(ordered[0], center_tile);
        assert!(ordered.contains(&center_tile));
        assert!(level.globe_target_lod_footprints(32).is_err());
    }
}
