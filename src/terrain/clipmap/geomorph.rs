//! P2.2: Geo-morphing and seam correctness utilities.
//!
//! Provides vertex blending at LOD boundaries to eliminate T-junction
//! artifacts and visual seams between clipmap rings.

use super::vertex::ClipmapVertex;
use glam::Vec2;

/// Configuration for geo-morphing.
#[derive(Debug, Clone, Copy)]
pub struct GeomorphConfig {
    /// Morph blend range as fraction of ring width [0.0-1.0].
    pub morph_range: f32,
    /// Maximum allowed seam gap in world units.
    pub max_seam_gap: f32,
    /// Enable snapping vertices to coarser grid at boundaries.
    pub snap_to_coarse: bool,
}

impl Default for GeomorphConfig {
    fn default() -> Self {
        Self {
            morph_range: 0.3,
            max_seam_gap: 0.001,
            snap_to_coarse: true,
        }
    }
}

/// Result of seam analysis.
#[derive(Debug, Clone)]
pub struct SeamAnalysis {
    /// Number of boundary vertices analyzed.
    pub boundary_vertex_count: u32,
    /// Maximum gap between adjacent LOD levels.
    pub max_gap: f32,
    /// Average gap between adjacent LOD levels.
    pub avg_gap: f32,
    /// Number of T-junction candidates detected.
    pub t_junction_count: u32,
    /// Whether all seams are within acceptable tolerance.
    pub seams_valid: bool,
}

/// Calculate morph weight for a vertex based on distance from LOD boundary.
///
/// Returns a weight in [0.0, 1.0] where:
/// - 0.0 = use fine (current) LOD height
/// - 1.0 = use coarse (next) LOD height
pub fn calculate_morph_weight(
    distance_from_inner: f32,
    ring_width: f32,
    morph_range: f32,
) -> f32 {
    if ring_width <= 0.0 || morph_range <= 0.0 {
        return 0.0;
    }
    
    let t = (distance_from_inner / ring_width).clamp(0.0, 1.0);
    let morph_start = 1.0 - morph_range;
    
    if t > morph_start {
        ((t - morph_start) / morph_range).min(1.0)
    } else {
        0.0
    }
}

/// Snap a UV coordinate to the coarser LOD grid.
///
/// This ensures that vertices at LOD boundaries align with the coarser grid,
/// eliminating T-junctions where the coarse level samples the heightmap.
pub fn snap_uv_to_coarse_grid(uv: Vec2, ring_index: u32, texture_size: u32) -> Vec2 {
    let lod_scale = 1 << ring_index;
    let coarse_texel_size = lod_scale as f32 / texture_size as f32;
    
    Vec2::new(
        (uv.x / coarse_texel_size).floor() * coarse_texel_size,
        (uv.y / coarse_texel_size).floor() * coarse_texel_size,
    )
}

/// Analyze seams between adjacent clipmap rings for potential artifacts.
pub fn analyze_seams(
    inner_vertices: &[ClipmapVertex],
    outer_vertices: &[ClipmapVertex],
    config: &GeomorphConfig,
) -> SeamAnalysis {
    let mut max_gap = 0.0_f32;
    let mut total_gap = 0.0_f32;
    let mut t_junction_count = 0_u32;
    let mut boundary_count = 0_u32;

    // Find vertices at the boundary between rings
    for outer_v in outer_vertices {
        // Check if this is an inner boundary vertex (morph_weight near 0)
        if outer_v.morph_weight() < 0.1 {
            boundary_count += 1;
            
            // Find closest inner ring vertex
            let outer_pos = Vec2::from(outer_v.position);
            let mut min_dist = f32::MAX;
            
            for inner_v in inner_vertices {
                // Check outer boundary of inner ring (morph_weight near 1)
                if inner_v.morph_weight() > 0.9 {
                    let inner_pos = Vec2::from(inner_v.position);
                    let dist = outer_pos.distance(inner_pos);
                    min_dist = min_dist.min(dist);
                }
            }
            
            if min_dist < f32::MAX {
                max_gap = max_gap.max(min_dist);
                total_gap += min_dist;
                
                // T-junction if gap is non-zero but not aligned
                if min_dist > config.max_seam_gap * 0.1 && min_dist < config.max_seam_gap * 10.0 {
                    t_junction_count += 1;
                }
            }
        }
    }

    let avg_gap = if boundary_count > 0 {
        total_gap / boundary_count as f32
    } else {
        0.0
    };

    SeamAnalysis {
        boundary_vertex_count: boundary_count,
        max_gap,
        avg_gap,
        t_junction_count,
        seams_valid: max_gap <= config.max_seam_gap,
    }
}

/// Correct seam vertices by snapping to coarser grid positions.
///
/// Returns the number of vertices corrected.
pub fn correct_seam_vertices(
    vertices: &mut [ClipmapVertex],
    ring_index: u32,
    texture_size: u32,
    config: &GeomorphConfig,
) -> u32 {
    if !config.snap_to_coarse {
        return 0;
    }

    let mut corrected = 0;
    let morph_threshold = 1.0 - config.morph_range * 0.5;

    for v in vertices.iter_mut() {
        // Only correct vertices near the outer boundary (high morph weight)
        if v.morph_weight() > morph_threshold && (v.ring_index() as u32) == ring_index {
            let old_uv = Vec2::from(v.uv);
            let new_uv = snap_uv_to_coarse_grid(old_uv, ring_index + 1, texture_size);
            
            if old_uv.distance(new_uv) > 0.0001 {
                v.uv = [new_uv.x, new_uv.y];
                corrected += 1;
            }
        }
    }

    corrected
}

/// Blend vertex positions at LOD boundaries for smooth transitions.
pub fn blend_boundary_vertices(
    fine_vertices: &[ClipmapVertex],
    coarse_vertices: &[ClipmapVertex],
    blend_factor: f32,
) -> Vec<ClipmapVertex> {
    let mut blended = Vec::with_capacity(fine_vertices.len());
    
    for fine_v in fine_vertices {
        // Find corresponding coarse vertex
        let fine_pos = Vec2::from(fine_v.position);
        let mut best_coarse: Option<&ClipmapVertex> = None;
        let mut best_dist = f32::MAX;
        
        for coarse_v in coarse_vertices {
            let coarse_pos = Vec2::from(coarse_v.position);
            let dist = fine_pos.distance(coarse_pos);
            if dist < best_dist {
                best_dist = dist;
                best_coarse = Some(coarse_v);
            }
        }
        
        if let Some(coarse_v) = best_coarse {
            if best_dist < 1.0 {
                // Blend position based on morph weight
                let t = blend_factor * fine_v.morph_weight();
                let blended_pos = fine_pos.lerp(Vec2::from(coarse_v.position), t);
                let blended_uv = Vec2::from(fine_v.uv).lerp(Vec2::from(coarse_v.uv), t);
                
                blended.push(ClipmapVertex::new(
                    blended_pos.x,
                    blended_pos.y,
                    blended_uv.x,
                    blended_uv.y,
                    fine_v.morph_weight(),
                    fine_v.ring_index() as u32,
                ));
            } else {
                blended.push(*fine_v);
            }
        } else {
            blended.push(*fine_v);
        }
    }
    
    blended
}

/// Validate that geo-morphing eliminates visible seams.
pub fn validate_geomorph(
    vertices: &[ClipmapVertex],
    config: &GeomorphConfig,
) -> Result<(), String> {
    // Check morph weights are in valid range
    for (i, v) in vertices.iter().enumerate() {
        let mw = v.morph_weight();
        if mw > 1.0 {
            return Err(format!("Vertex {} has morph_weight {} > 1.0", i, mw));
        }
        // Negative morph weight is valid (indicates skirt vertex)
    }

    // Check for discontinuities in morph weights
    let mut prev_morph = 0.0_f32;
    let mut large_jumps = 0;
    
    for v in vertices {
        let mw = v.morph_weight();
        if mw >= 0.0 {  // Skip skirt vertices
            let jump = (mw - prev_morph).abs();
            if jump > config.morph_range {
                large_jumps += 1;
            }
            prev_morph = mw;
        }
    }

    if large_jumps > vertices.len() / 10 {
        return Err(format!(
            "Too many large morph weight discontinuities: {} (threshold: {})",
            large_jumps,
            vertices.len() / 10
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_weight_calculation() {
        // At inner boundary, weight should be 0
        assert_eq!(calculate_morph_weight(0.0, 100.0, 0.3), 0.0);
        
        // At outer boundary, weight should be 1
        assert!((calculate_morph_weight(100.0, 100.0, 0.3) - 1.0).abs() < 0.01);
        
        // In middle of ring (before morph zone), weight should be 0
        assert_eq!(calculate_morph_weight(50.0, 100.0, 0.3), 0.0);
        
        // At start of morph zone (70% through ring with 0.3 morph_range)
        let w = calculate_morph_weight(70.0, 100.0, 0.3);
        assert!(w >= 0.0 && w <= 0.1);
    }

    #[test]
    fn test_snap_uv_to_coarse_grid() {
        // Ring 0 with 256 texture should snap to 1/256 grid
        let uv = Vec2::new(0.123, 0.456);
        let snapped = snap_uv_to_coarse_grid(uv, 0, 256);
        assert!((snapped.x * 256.0).fract() < 0.001);
        assert!((snapped.y * 256.0).fract() < 0.001);
        
        // Ring 1 should snap to 2/256 grid
        let snapped = snap_uv_to_coarse_grid(uv, 1, 256);
        assert!((snapped.x * 128.0).fract() < 0.001);
        assert!((snapped.y * 128.0).fract() < 0.001);
    }

    #[test]
    fn test_geomorph_config_default() {
        let config = GeomorphConfig::default();
        assert!(config.morph_range > 0.0 && config.morph_range <= 1.0);
        assert!(config.max_seam_gap > 0.0);
        assert!(config.snap_to_coarse);
    }
}
