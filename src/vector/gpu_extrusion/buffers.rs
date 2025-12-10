use super::types::{PolygonMeta, RingVertexPacked};
use crate::vector::extrusion::TessellatedPolygon;
use glam::Vec2;

pub struct PolygonBuffers {
    pub metas: Vec<PolygonMeta>,
    pub base_vertices: Vec<[f32; 2]>,
    pub base_indices: Vec<u32>,
    pub ring_vertices: Vec<RingVertexPacked>,
    pub vertex_count: u32,
    pub index_count: u32,
}

pub fn pack_tessellations(tessellated: &[TessellatedPolygon]) -> Result<PolygonBuffers, String> {
    let mut metas = Vec::with_capacity(tessellated.len());
    let mut base_vertices = Vec::new();
    let mut base_indices = Vec::new();
    let mut ring_vertices = Vec::new();

    let mut base_vertex_offset: u32 = 0;
    let mut base_index_offset: u32 = 0;
    let mut ring_offset: u32 = 0;
    let mut output_vertex_offset: u32 = 0;
    let mut output_index_offset: u32 = 0;

    for tess in tessellated {
        let base_v = u32::try_from(tess.base_vertices.len())
            .map_err(|_| "polygon has too many base vertices (u32 overflow)".to_string())?;
        let base_i = u32::try_from(tess.base_indices.len())
            .map_err(|_| "polygon has too many indices (u32 overflow)".to_string())?;
        let ring_count = u32::try_from(tess.ring.len())
            .map_err(|_| "polygon ring too large (u32 overflow)".to_string())?;

        let side_vertex_count = ring_count
            .checked_mul(4)
            .ok_or_else(|| "side vertex count overflow".to_string())?;
        let side_index_count = ring_count
            .checked_mul(6)
            .ok_or_else(|| "side index count overflow".to_string())?;

        base_vertices.extend(tess.base_vertices.iter().map(|v| [v.x, v.y]));
        base_indices.extend(&tess.base_indices);
        ring_vertices.extend(
            tess.ring
                .iter()
                .zip(&tess.ring_u)
                .map(|(pos, u)| RingVertexPacked {
                    position: [pos.x, pos.y],
                    u_coord: *u,
                    _pad: 0.0,
                }),
        );

        metas.push(PolygonMeta {
            base_vertex_offset,
            base_vertex_count: base_v,
            base_index_offset,
            base_index_count: base_i,
            ring_offset,
            ring_count,
            output_vertex_offset,
            output_index_offset,
            bbox_min: [tess.bbox_min.x, tess.bbox_min.y],
            bbox_scale: compute_bbox_scale(tess.bbox_size),
        });

        base_vertex_offset = base_vertex_offset
            .checked_add(base_v)
            .ok_or_else(|| "base vertex offset overflow".to_string())?;
        base_index_offset = base_index_offset
            .checked_add(base_i)
            .ok_or_else(|| "base index offset overflow".to_string())?;
        ring_offset = ring_offset
            .checked_add(ring_count)
            .ok_or_else(|| "ring offset overflow".to_string())?;
        output_vertex_offset = output_vertex_offset
            .checked_add(base_v * 2)
            .and_then(|val| val.checked_add(side_vertex_count))
            .ok_or_else(|| "vertex offset overflow".to_string())?;
        output_index_offset = output_index_offset
            .checked_add(base_i * 2)
            .and_then(|val| val.checked_add(side_index_count))
            .ok_or_else(|| "index offset overflow".to_string())?;
    }

    Ok(PolygonBuffers {
        metas,
        base_vertices,
        base_indices,
        ring_vertices,
        vertex_count: output_vertex_offset,
        index_count: output_index_offset,
    })
}

fn compute_bbox_scale(size: Vec2) -> [f32; 2] {
    let x = if size.x.abs() > crate::vector::extrusion::EPSILON {
        1.0 / size.x
    } else {
        0.0
    };
    let y = if size.y.abs() > crate::vector::extrusion::EPSILON {
        1.0 / size.y
    } else {
        0.0
    };
    [x, y]
}
