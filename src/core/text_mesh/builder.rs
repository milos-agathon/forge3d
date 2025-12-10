use super::types::VertexPN;
use lyon_path::{math, Path};
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, FillVertexConstructor, VertexBuffers,
};
use ttf_parser::{Face, OutlineBuilder};

struct PathSink<'a> {
    builder: &'a mut lyon_path::path::Builder,
    scale: f32,
    offset: glam::Vec2,
}
impl OutlineBuilder for PathSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.begin(math::point(
            self.offset.x + x * self.scale,
            self.offset.y - y * self.scale,
        ));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(math::point(
            self.offset.x + x * self.scale,
            self.offset.y - y * self.scale,
        ));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(
            math::point(
                self.offset.x + x1 * self.scale,
                self.offset.y - y1 * self.scale,
            ),
            math::point(
                self.offset.x + x * self.scale,
                self.offset.y - y * self.scale,
            ),
        );
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            math::point(
                self.offset.x + x1 * self.scale,
                self.offset.y - y1 * self.scale,
            ),
            math::point(
                self.offset.x + x2 * self.scale,
                self.offset.y - y2 * self.scale,
            ),
            math::point(
                self.offset.x + x * self.scale,
                self.offset.y - y * self.scale,
            ),
        );
    }
    fn close(&mut self) {
        self.builder.end(true);
    }
}

#[derive(Clone, Copy)]
struct FrontVertexCtor;
impl FillVertexConstructor<VertexPN> for FrontVertexCtor {
    fn new_vertex(&mut self, v: FillVertex) -> VertexPN {
        let p = v.position();
        VertexPN {
            position: [p.x, p.y, 0.0],
            normal: [0.0, 0.0, 1.0],
        }
    }
}

pub fn build_text_mesh(
    text: &str,
    font_bytes: &[u8],
    size_px: f32,
    depth: f32,
    bevel_width: f32,
    bevel_segments: u32,
) -> anyhow::Result<(Vec<VertexPN>, Vec<u32>)> {
    let face = Face::parse(font_bytes, 0).map_err(|_| anyhow::anyhow!("Invalid font data"))?;
    let units = face.units_per_em() as f32;
    let scale = (size_px / units).max(1e-6);

    // Build combined path for all glyphs
    let mut path_builder = Path::builder();
    let mut x_cursor = 0.0f32;

    for ch in text.chars() {
        if ch == '\n' {
            // simple newline support
            // Move down by ascent (approx) and reset x
            x_cursor = 0.0;
            // skip vertical offset in this simple pass
            continue;
        }
        if let Some(gid) = face.glyph_index(ch) {
            // Outline glyph into builder with current offset
            let mut sink = PathSink {
                builder: &mut path_builder,
                scale,
                offset: glam::Vec2::new(x_cursor, 0.0),
            };
            // outline_glyph returns Option<Rect>; we ignore the bounds and just build when present
            let _ = face.outline_glyph(gid, &mut sink);
            // advance x_cursor
            let adv = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
            x_cursor += adv;
        }
    }

    let path = path_builder.build();

    // Tessellate front face
    let mut buffers: VertexBuffers<VertexPN, u32> = VertexBuffers::new();
    let mut tess = FillTessellator::new();
    tess.tessellate_path(
        path.as_slice(),
        &FillOptions::tolerance(0.1),
        &mut BuffersBuilder::new(&mut buffers, FrontVertexCtor),
    )
    .map_err(|e| anyhow::anyhow!("Tessellation failed: {:?}", e))?;

    let mut vertices = buffers.vertices.clone();
    let mut indices = buffers.indices.clone();

    // Back face (reverse winding)
    let back_offset = vertices.len() as u32;
    for v in &buffers.vertices {
        vertices.push(VertexPN {
            position: [v.position[0], v.position[1], depth],
            normal: [0.0, 0.0, -1.0],
        });
    }
    let mut back_tris: Vec<u32> = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        back_tris.push(back_offset + tri[0]);
        back_tris.push(back_offset + tri[2]);
        back_tris.push(back_offset + tri[1]);
    }
    indices.extend(back_tris);

    // Side walls with geometric bevels: iterate flattened path edges
    let tol = 0.5f32; // px tolerance for flattening
    let mut _first_in_subpath = None;
    let mut _prev = None;
    for e in path.iter() {
        match e {
            lyon_path::Event::Begin { at } => {
                _first_in_subpath = Some(at);
                _prev = Some(at);
            }
            lyon_path::Event::Line { from, to } => {
                add_side_bevel_strips(
                    &mut vertices,
                    &mut indices,
                    from,
                    to,
                    depth,
                    bevel_width,
                    bevel_segments.max(1),
                );
                _prev = Some(to);
            }
            lyon_path::Event::Quadratic { from, ctrl, to } => {
                let seg = lyon_geom::QuadraticBezierSegment { from, ctrl, to };
                seg.for_each_flattened(tol, &mut |ls: &lyon_geom::LineSegment<f32>| {
                    add_side_bevel_strips(
                        &mut vertices,
                        &mut indices,
                        ls.from,
                        ls.to,
                        depth,
                        bevel_width,
                        bevel_segments.max(1),
                    );
                });
                _prev = Some(to);
            }
            lyon_path::Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => {
                let seg = lyon_geom::CubicBezierSegment {
                    from,
                    ctrl1,
                    ctrl2,
                    to,
                };
                seg.for_each_flattened(tol, &mut |ls: &lyon_geom::LineSegment<f32>| {
                    add_side_bevel_strips(
                        &mut vertices,
                        &mut indices,
                        ls.from,
                        ls.to,
                        depth,
                        bevel_width,
                        bevel_segments.max(1),
                    );
                });
                _prev = Some(to);
            }
            lyon_path::Event::End { last, first, close } => {
                if close {
                    add_side_bevel_strips(
                        &mut vertices,
                        &mut indices,
                        last,
                        first,
                        depth,
                        bevel_width,
                        bevel_segments.max(1),
                    );
                }
                _first_in_subpath = None;
                _prev = None;
            }
        }
    }

    Ok((vertices, indices))
}

fn add_side_bevel_strips(
    vertices: &mut Vec<VertexPN>,
    indices: &mut Vec<u32>,
    from: math::Point,
    to: math::Point,
    depth: f32,
    bevel_width: f32,
    bevel_segments: u32,
) {
    let p0 = glam::Vec2::new(from.x, from.y);
    let p1 = glam::Vec2::new(to.x, to.y);
    let edge = p1 - p0;
    let len = edge.length();
    if len < 1e-6 {
        return;
    }
    let n = glam::Vec2::new(edge.y / len, -edge.x / len); // outward approx

    let bw = bevel_width.max(0.0);
    let segs = bevel_segments as usize;
    // Generate front bevel rings from z=0 to z=bw
    let mut rings: Vec<(f32, glam::Vec2)> = Vec::new();
    for k in 0..=segs {
        let t = k as f32 / (segs as f32);
        let z = bw * t;
        let off = n * (bw * t);
        rings.push((z, off));
    }
    // Add middle section (constant offset) if depth allows
    if depth > 2.0 * bw {
        rings.push((depth - bw, n * bw));
    }
    // Back bevel rings from z=depth-bw to z=depth
    for k in (0..=segs).rev() {
        let t = k as f32 / (segs as f32);
        let z = depth - bw * t;
        let off = n * (bw * t);
        rings.push((z, off));
    }

    // Build quad strips between successive rings
    for r in 0..(rings.len().saturating_sub(1)) {
        let (z0, off0) = rings[r];
        let (z1, off1) = rings[r + 1];
        let v_start = vertices.len() as u32;
        let n3 = [n.x, n.y, 0.0];
        // p0, p1 at z0
        vertices.push(VertexPN {
            position: [p0.x + off0.x, p0.y + off0.y, z0],
            normal: n3,
        });
        vertices.push(VertexPN {
            position: [p1.x + off0.x, p1.y + off0.y, z0],
            normal: n3,
        });
        // p1, p0 at z1
        vertices.push(VertexPN {
            position: [p1.x + off1.x, p1.y + off1.y, z1],
            normal: n3,
        });
        vertices.push(VertexPN {
            position: [p0.x + off1.x, p0.y + off1.y, z1],
            normal: n3,
        });
        indices.extend_from_slice(&[
            v_start,
            v_start + 1,
            v_start + 2,
            v_start,
            v_start + 2,
            v_start + 3,
        ]);
    }
}
