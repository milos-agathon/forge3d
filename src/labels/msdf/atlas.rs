use super::distance::{contains, correct_collision, median, signed_pseudo_distance};
use super::edge::{color_edges, flatten_path, Contour, Edge};
use crate::labels::font::{FontCollection, FontGlyph, TextError};
use lyon_path::math::Point;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GlyphMetric {
    pub codepoint: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub advance: f32,
}

#[derive(Clone, Debug)]
pub struct BakedMsdfAtlas {
    pub image: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub font_size: f32,
    pub line_height: f32,
    pub baseline: f32,
    pub px_range: f32,
    pub padding: u32,
    pub bake_ms: f64,
    pub glyphs: Vec<GlyphMetric>,
}

impl BakedMsdfAtlas {
    pub fn byte_count(&self) -> usize {
        self.image.len()
    }
}

struct PreparedGlyph {
    codepoint: u32,
    contours: Vec<Contour>,
    bounds: [f32; 4],
    advance: f32,
    cell_width: u32,
    cell_height: u32,
}

fn bounds(contours: &[Contour]) -> Option<[f32; 4]> {
    contours
        .iter()
        .flat_map(|contour| &contour.points)
        .fold(None, |bounds, point| {
            Some(match bounds {
                None => [point.x, point.y, point.x, point.y],
                Some(mut bounds) => {
                    bounds[0] = bounds[0].min(point.x);
                    bounds[1] = bounds[1].min(point.y);
                    bounds[2] = bounds[2].max(point.x);
                    bounds[3] = bounds[3].max(point.y);
                    bounds
                }
            })
        })
}

fn integer_sqrt_ceiling(value: usize) -> usize {
    let mut root = 1usize;
    while root * root < value {
        root += 1;
    }
    root
}

fn prepare(
    fonts: &FontCollection,
    character: char,
    font_size: f32,
    margin: u32,
) -> Result<PreparedGlyph, TextError> {
    let glyph = fonts.glyph_for(character)?;
    let metrics = fonts.metrics(glyph.font_index)?;
    let scale = font_size / f32::from(metrics.units_per_em);
    let advance = fonts.horizontal_advance(FontGlyph {
        font_index: glyph.font_index,
        glyph_id: glyph.glyph_id,
    })? as f32
        * scale;
    let contours = match fonts.outline(glyph.font_index, glyph.glyph_id) {
        Ok(path) => flatten_path(&path, scale, Point::new(0.0, 0.0)),
        Err(TextError::MissingOutline { .. }) if character.is_whitespace() => Vec::new(),
        Err(error) => return Err(error),
    };
    let bounds = bounds(&contours).unwrap_or([0.0; 4]);
    let cell_width = ((bounds[2] - bounds[0]).ceil() as u32).max(1) + margin * 2;
    let cell_height = ((bounds[3] - bounds[1]).ceil() as u32).max(1) + margin * 2;
    Ok(PreparedGlyph {
        codepoint: character as u32,
        contours,
        bounds,
        advance,
        cell_width,
        cell_height,
    })
}

fn translated(contours: &[Contour], offset: Point) -> Vec<Contour> {
    contours
        .iter()
        .map(|contour| Contour {
            points: contour
                .points
                .iter()
                .map(|point| Point::new(point.x + offset.x, point.y + offset.y))
                .collect(),
        })
        .collect()
}

fn field(edges: &[Edge], point: Point, inside: bool) -> [f32; 3] {
    let mut channels = [f32::INFINITY; 3];
    let mut nearest = f32::INFINITY;
    for edge in edges {
        let distance = signed_pseudo_distance(point, *edge);
        nearest = nearest.min(distance.abs());
        for (channel, value) in channels.iter_mut().enumerate() {
            if edge.color & (1 << channel) != 0 && distance.abs() < value.abs() {
                *value = distance;
            }
        }
    }
    for channel in &mut channels {
        if !channel.is_finite() {
            *channel = nearest;
        }
    }
    let truth = nearest.copysign(if inside { 1.0 } else { -1.0 });
    let corrected = correct_collision(channels, truth);
    debug_assert_eq!(median(corrected) >= 0.0, truth >= 0.0);
    corrected
}

pub fn bake_msdf_atlas(
    fonts: &FontCollection,
    characters: &[char],
    font_size: f32,
    px_range: f32,
    padding: u32,
) -> Result<BakedMsdfAtlas, TextError> {
    let started = Instant::now();
    let margin = padding + px_range.ceil() as u32;
    let prepared = characters
        .iter()
        .copied()
        .map(|character| prepare(fonts, character, font_size, margin))
        .collect::<Result<Vec<_>, _>>()?;
    let cell_width = prepared
        .iter()
        .map(|glyph| glyph.cell_width)
        .max()
        .unwrap_or(1);
    let cell_height = prepared
        .iter()
        .map(|glyph| glyph.cell_height)
        .max()
        .unwrap_or(1);
    let columns = integer_sqrt_ceiling(prepared.len().max(1));
    let rows = prepared.len().max(1).div_ceil(columns);
    let width = cell_width * columns as u32;
    let height = cell_height * rows as u32;
    let mut image = vec![0u8; width as usize * height as usize * 3];
    let mut glyphs = Vec::with_capacity(prepared.len());

    for (index, glyph) in prepared.iter().enumerate() {
        let x = (index % columns) as u32 * cell_width;
        let y = (index / columns) as u32 * cell_height;
        let offset_x = glyph.bounds[0].floor() - margin as f32;
        let offset_y = glyph.bounds[1].floor() - margin as f32;
        let contours = translated(
            &glyph.contours,
            Point::new(x as f32 - offset_x, y as f32 - offset_y),
        );
        let edges: Vec<_> = contours.iter().flat_map(color_edges).collect();
        for pixel_y in y..y + cell_height {
            for pixel_x in x..x + cell_width {
                let rgb = if edges.is_empty() {
                    [0.0; 3]
                } else {
                    let sample = Point::new(pixel_x as f32 + 0.5, pixel_y as f32 + 0.5);
                    field(&edges, sample, contains(&contours, sample))
                };
                let target = (pixel_y as usize * width as usize + pixel_x as usize) * 3;
                for channel in 0..3 {
                    image[target + channel] =
                        ((0.5 + rgb[channel] / px_range).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        glyphs.push(GlyphMetric {
            codepoint: glyph.codepoint,
            x,
            y,
            width: cell_width,
            height: cell_height,
            offset_x,
            offset_y,
            advance: glyph.advance,
        });
    }

    Ok(BakedMsdfAtlas {
        image,
        width,
        height,
        font_size,
        line_height: font_size * 1.2,
        baseline: font_size,
        px_range,
        padding,
        bake_ms: started.elapsed().as_secs_f64() * 1000.0,
        glyphs,
    })
}
