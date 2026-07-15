use super::distance::{contains, correct_collision, edge_distance, median};
use super::edge::{color_edges, flatten_path, Contour, Edge};
use crate::labels::font::{FontCollection, FontGlyph, TextError};
use crate::labels::shape::ShapedText;
use lyon_path::math::Point;
use std::collections::BTreeMap;
use std::time::Instant;
use ttf_parser::GlyphId;

#[derive(Clone, Debug)]
pub struct GlyphMetric {
    pub codepoint: u32,
    pub font_index: usize,
    pub glyph_id: u16,
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
    /// Independently baked all-contour scalar SDF used by fidelity/ablation gates.
    pub sdf_image: Vec<u8>,
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
    font_index: usize,
    glyph_id: u16,
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

fn prepare_glyph(
    fonts: &FontCollection,
    font_index: usize,
    glyph_id: GlyphId,
    codepoint: u32,
    font_size: f32,
    margin: u32,
) -> Result<PreparedGlyph, TextError> {
    let metrics = fonts.metrics(font_index)?;
    let scale = font_size / f32::from(metrics.units_per_em);
    let advance = fonts.horizontal_advance(FontGlyph {
        font_index,
        glyph_id,
    })? as f32
        * scale;
    let contours = match fonts.outline(font_index, glyph_id) {
        Ok(path) => flatten_path(&path, scale, Point::new(0.0, 0.0)),
        // Spaces and other advance-only glyphs legitimately have no outline.
        Err(TextError::MissingOutline { .. }) => Vec::new(),
        Err(error) => return Err(error),
    };
    let bounds = bounds(&contours).unwrap_or([0.0; 4]);
    let cell_width = ((bounds[2] - bounds[0]).ceil() as u32).max(1) + margin * 2;
    let cell_height = ((bounds[3] - bounds[1]).ceil() as u32).max(1) + margin * 2;
    Ok(PreparedGlyph {
        codepoint,
        font_index,
        glyph_id: glyph_id.0,
        contours,
        bounds,
        advance,
        cell_width,
        cell_height,
    })
}

fn prepare_character(
    fonts: &FontCollection,
    character: char,
    font_size: f32,
    margin: u32,
) -> Result<PreparedGlyph, TextError> {
    let glyph = fonts.glyph_for(character)?;
    prepare_glyph(
        fonts,
        glyph.font_index,
        glyph.glyph_id,
        character as u32,
        font_size,
        margin,
    )
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

fn field(edges: &[Edge], point: Point, inside: bool) -> ([f32; 3], f32) {
    let mut channels = [f32::INFINITY; 3];
    let mut channel_true = [f32::INFINITY; 3];
    let mut nearest = f32::INFINITY;
    for edge in edges {
        let distance = edge_distance(point, *edge);
        nearest = nearest.min(distance.true_distance.abs());
        for (channel, value) in channels.iter_mut().enumerate() {
            if edge.color & (1 << channel) != 0
                && distance.true_distance.abs() < channel_true[channel]
            {
                channel_true[channel] = distance.true_distance.abs();
                *value = distance.pseudo_distance;
            }
        }
    }
    let sign = if inside { 1.0 } else { -1.0 };
    for channel in &mut channels {
        if !channel.is_finite() {
            *channel = nearest.copysign(sign);
        }
    }
    let truth = nearest.copysign(sign);
    let corrected = correct_collision(channels, truth);
    debug_assert_eq!(median(corrected) >= 0.0, truth >= 0.0);
    (corrected, truth)
}

fn majority_coverage_inside(contours: &[Contour], pixel_x: u32, pixel_y: u32) -> bool {
    const SUBPIXELS: u32 = 8;
    let mut covered = 0u32;
    for subpixel_y in 0..SUBPIXELS {
        for subpixel_x in 0..SUBPIXELS {
            let sample = Point::new(
                pixel_x as f32 + (subpixel_x as f32 + 0.5) / SUBPIXELS as f32,
                pixel_y as f32 + (subpixel_y as f32 + 0.5) / SUBPIXELS as f32,
            );
            covered += u32::from(contains(contours, sample));
        }
    }
    covered * 2 >= SUBPIXELS * SUBPIXELS
}

fn bake_prepared(
    started: Instant,
    prepared: Vec<PreparedGlyph>,
    font_size: f32,
    px_range: f32,
    padding: u32,
) -> BakedMsdfAtlas {
    let margin = padding + px_range.ceil() as u32;
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
    let mut sdf_image = vec![0u8; width as usize * height as usize];
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
                let (rgb, scalar) = if edges.is_empty() {
                    ([-px_range; 3], -px_range)
                } else {
                    let sample = Point::new(pixel_x as f32 + 0.5, pixel_y as f32 + 0.5);
                    field(
                        &edges,
                        sample,
                        majority_coverage_inside(&contours, pixel_x, pixel_y),
                    )
                };
                let target = (pixel_y as usize * width as usize + pixel_x as usize) * 3;
                for channel in 0..3 {
                    image[target + channel] =
                        ((0.5 + rgb[channel] / px_range).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
                sdf_image[pixel_y as usize * width as usize + pixel_x as usize] =
                    ((0.5 + scalar / px_range).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        glyphs.push(GlyphMetric {
            codepoint: glyph.codepoint,
            font_index: glyph.font_index,
            glyph_id: glyph.glyph_id,
            x,
            y,
            width: cell_width,
            height: cell_height,
            offset_x,
            offset_y,
            advance: glyph.advance,
        });
    }

    BakedMsdfAtlas {
        image,
        sdf_image,
        width,
        height,
        font_size,
        line_height: font_size * 1.2,
        baseline: font_size,
        px_range,
        padding,
        bake_ms: started.elapsed().as_secs_f64() * 1000.0,
        glyphs,
    }
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
        .map(|character| prepare_character(fonts, character, font_size, margin))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(bake_prepared(
        started, prepared, font_size, px_range, padding,
    ))
}

/// Bake the exact `(font_index, glyph_id)` identities emitted by shaping.
///
/// This is the canonical path for GPU text because ligatures, substitutions, and
/// fallback faces cannot be reconstructed from Unicode codepoints after shaping.
pub fn bake_msdf_atlas_from_shaped(
    shaped: &ShapedText,
    font_size: f32,
    px_range: f32,
    padding: u32,
) -> Result<BakedMsdfAtlas, TextError> {
    let started = Instant::now();
    let margin = padding + px_range.ceil() as u32;
    let mut identities = BTreeMap::new();
    for glyph in shaped.runs.iter().flat_map(|run| &run.glyphs) {
        let codepoint = shaped
            .text
            .get(glyph.cluster as usize..)
            .and_then(|tail| tail.chars().next())
            .map_or(0, |character| character as u32);
        identities
            .entry((glyph.font_index, glyph.glyph_id))
            .or_insert(codepoint);
    }
    let prepared = identities
        .into_iter()
        .map(|((font_index, glyph_id), codepoint)| {
            prepare_glyph(
                &shaped.fonts,
                font_index,
                GlyphId(glyph_id),
                codepoint,
                font_size,
                margin,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(bake_prepared(
        started, prepared, font_size, px_range, padding,
    ))
}

#[cfg(test)]
mod tests {
    use super::{bake_msdf_atlas_from_shaped, field, median};
    use crate::labels::font::{FontCollection, FontRequest};
    use crate::labels::msdf::distance::contains;
    use crate::labels::msdf::edge::{color_edges, Contour};
    use crate::labels::shape;
    use lyon_path::math::Point;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    fn contour(points: &[(f32, f32)]) -> Contour {
        Contour {
            points: points.iter().map(|&(x, y)| Point::new(x, y)).collect(),
        }
    }

    fn assert_field_sign(contours: &[Contour], samples: &[(f32, f32)]) {
        let edges: Vec<_> = contours.iter().flat_map(color_edges).collect();
        for &(x, y) in samples {
            let point = Point::new(x, y);
            let inside = contains(contours, point);
            let reconstructed = median(field(&edges, point, inside).0);
            assert_eq!(
                reconstructed >= 0.0,
                inside,
                "wrong MSDF sign at ({x}, {y})"
            );
        }
    }

    #[test]
    fn clockwise_and_counterclockwise_holes_keep_the_same_fill_sign() {
        let outer = contour(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let hole = contour(&[(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0), (3.0, 3.0)]);
        let samples = [(1.0, 1.0), (2.9, 5.0), (5.0, 5.0), (11.0, 5.0)];
        assert_field_sign(&[outer.clone(), hole.clone()], &samples);

        let reversed = [
            Contour {
                points: outer.points.into_iter().rev().collect(),
            },
            Contour {
                points: hole.points.into_iter().rev().collect(),
            },
        ];
        assert_field_sign(&reversed, &samples);
    }

    #[test]
    fn acute_and_obtuse_endpoint_collisions_preserve_scalar_truth() {
        let acute = contour(&[(0.0, 0.0), (8.0, 0.0), (0.5, 8.0), (0.0, 0.0)]);
        let obtuse = contour(&[(0.0, 0.0), (8.0, 0.0), (11.0, 3.0), (0.0, 0.0)]);
        let samples = [
            (-0.1, 0.1),
            (0.1, 0.1),
            (7.9, 0.1),
            (8.1, -0.1),
            (10.8, 2.8),
            (11.2, 3.2),
        ];
        assert_field_sign(&[acute], &samples);
        assert_field_sign(&[obtuse], &samples);
    }

    #[test]
    fn shaped_bake_preserves_ligature_and_fallback_font_identity() {
        let fonts = Arc::new(
            FontCollection::load(&[
                FontRequest::from_bytes(
                    "NotoSansLatin-subset.ttf",
                    include_bytes!("../../../assets/fonts/NotoSansLatin-subset.ttf").to_vec(),
                ),
                FontRequest::from_bytes(
                    "NotoSansHebrew-subset.ttf",
                    include_bytes!("../../../assets/fonts/NotoSansHebrew-subset.ttf").to_vec(),
                ),
            ])
            .unwrap(),
        );
        let shaped = shape::shape("fi בד", fonts, 24.0, None, None, &[]).unwrap();
        let expected = shaped
            .runs
            .iter()
            .flat_map(|run| &run.glyphs)
            .map(|glyph| (glyph.font_index, glyph.glyph_id))
            .collect::<BTreeSet<_>>();
        assert!(expected.iter().any(|(font_index, _)| *font_index == 1));

        let baked = bake_msdf_atlas_from_shaped(&shaped, 24.0, 4.0, 2).unwrap();
        let actual = baked
            .glyphs
            .iter()
            .map(|glyph| (glyph.font_index, glyph.glyph_id))
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
    }
}
