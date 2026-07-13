// src/gis/geometry/antimeridian.rs
// MENSURA M-04: canonical antimeridian (±180°) splitter for geographic geometry.
//
// The pre-MENSURA topology path unwrapped a dateline-crossing geometry onto a
// continuous 360° sheet, ran the op, then wrapped longitudes back — which for a
// polygon leaves a vertex "jump" across ±180 that renders as the world-spanning
// COMPLEMENT rather than the small crossing shape. This splitter instead cuts
// the geometry at the antimeridian into pieces on each side, preserving ring
// closure, holes (assigned by side), orientation, and MultiPolygon/collection
// structure, and normalizes every output longitude into (-180, 180].
//
// Scope: correct for the geometries that occur in practice — lines and polygons
// (with or without holes) that cross the antimeridian at a single meridian copy.
// A ring whose unwrapped span exceeds 360° (physically a >half-globe polygon) is
// left to the caller's wrapping path; `split_at_antimeridian` never fabricates
// geometry it cannot cut cleanly.
// RELEVANT FILES: src/gis/geometry.rs, src/gis/geometry/math.rs

use super::model::{Coord, Geometry};

const EPS: f64 = 1e-9;

/// Normalize a longitude into (-180, 180].
fn wrap_lon(x: f64) -> f64 {
    let r = (x + 180.0).rem_euclid(360.0) - 180.0;
    if r == -180.0 {
        180.0
    } else {
        r
    }
}

fn wrap(c: Coord) -> Coord {
    Coord {
        x: wrap_lon(c.x),
        y: c.y,
    }
}

fn wrap_all(pts: &[Coord]) -> Vec<Coord> {
    pts.iter().copied().map(wrap).collect()
}

/// True if the wrapped edge a→b takes the short path across the antimeridian.
fn edge_crosses(a: Coord, b: Coord) -> bool {
    (b.x - a.x).abs() > 180.0
}

fn line_crosses(pts: &[Coord]) -> bool {
    pts.windows(2).any(|w| edge_crosses(wrap(w[0]), wrap(w[1])))
}

/// Split an open polyline into 1+ pieces at each antimeridian crossing. Each
/// crossing edge is cut at ±180 with the latitude linearly interpolated.
fn split_line(pts: &[Coord]) -> Vec<Vec<Coord>> {
    if pts.len() < 2 {
        return vec![wrap_all(pts)];
    }
    let mut segments: Vec<Vec<Coord>> = Vec::new();
    let mut cur: Vec<Coord> = vec![wrap(pts[0])];
    for i in 1..pts.len() {
        let a = wrap(pts[i - 1]);
        let b = wrap(pts[i]);
        let dlon = b.x - a.x;
        if dlon.abs() > 180.0 {
            // Unwrap b next to a and find where the edge hits ±180.
            let (b_un_x, bound) = if dlon > 180.0 {
                (b.x - 360.0, -180.0)
            } else {
                (b.x + 360.0, 180.0)
            };
            let t = (bound - a.x) / (b_un_x - a.x);
            let lat = a.y + t * (b.y - a.y);
            cur.push(Coord { x: bound, y: lat });
            segments.push(std::mem::take(&mut cur));
            cur.push(Coord { x: -bound, y: lat });
            cur.push(b);
        } else {
            cur.push(b);
        }
    }
    segments.push(cur);
    segments
}

/// Unwrap a ring onto a continuous longitude sheet anchored at its first vertex.
fn unwrap_ring(ring: &[Coord]) -> Vec<Coord> {
    let mut out: Vec<Coord> = Vec::with_capacity(ring.len());
    if ring.is_empty() {
        return out;
    }
    let mut prev = wrap_lon(ring[0].x);
    out.push(Coord {
        x: prev,
        y: ring[0].y,
    });
    let mut offset = 0.0;
    for c in &ring[1..] {
        let mut x = wrap_lon(c.x) + offset;
        if x - prev > 180.0 {
            offset -= 360.0;
            x -= 360.0;
        } else if x - prev < -180.0 {
            offset += 360.0;
            x += 360.0;
        }
        out.push(Coord { x, y: c.y });
        prev = x;
    }
    out
}

/// Sutherland–Hodgman clip of a closed ring against a half-plane at `bound`
/// (keep x ≤ bound if `keep_less`, else x ≥ bound). Returns a closed ring, or
/// None if the clipped result is degenerate (< 3 distinct vertices).
fn clip_ring(ring: &[Coord], bound: f64, keep_less: bool) -> Option<Vec<Coord>> {
    if ring.len() < 4 {
        return None;
    }
    let inside = |x: f64| {
        if keep_less {
            x <= bound + EPS
        } else {
            x >= bound - EPS
        }
    };
    let intersect = |a: Coord, b: Coord| {
        let t = (bound - a.x) / (b.x - a.x);
        Coord {
            x: bound,
            y: a.y + t * (b.y - a.y),
        }
    };
    let mut out: Vec<Coord> = Vec::new();
    for w in ring.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (a_in, b_in) = (inside(a.x), inside(b.x));
        if b_in {
            if !a_in {
                out.push(intersect(a, b));
            }
            out.push(b);
        } else if a_in {
            out.push(intersect(a, b));
        }
    }
    if out.len() < 3 {
        return None;
    }
    if out.first() != out.last() {
        out.push(out[0]);
    }
    // Reject a sliver whose vertices are all on the clip line.
    if out.iter().all(|c| (c.x - bound).abs() < EPS) {
        return None;
    }
    Some(out)
}

/// Shift a clipped piece (in unwrapped space) by a whole multiple of 360° so
/// its longitudes land coherently in [-180, 180] — the boundary vertices at the
/// split meridian become +180 for the western piece and -180 for the eastern
/// one, so neither renders as the world-spanning complement. A naive per-vertex
/// wrap would collapse -180 to +180 and re-introduce the wrap-around edge.
fn normalize_piece(ring: Vec<Coord>) -> Vec<Coord> {
    if ring.is_empty() {
        return ring;
    }
    let mean = ring.iter().map(|c| c.x).sum::<f64>() / ring.len() as f64;
    let shift = -360.0 * ((mean + 180.0) / 360.0).floor();
    ring.into_iter()
        .map(|c| Coord {
            x: c.x + shift,
            y: c.y,
        })
        .collect()
}

#[cfg(test)]
fn signed_area(ring: &[Coord]) -> f64 {
    let mut s = 0.0;
    for w in ring.windows(2) {
        s += w[0].x * w[1].y - w[1].x * w[0].y;
    }
    s * 0.5
}

/// Split a polygon (outer ring + holes) at the antimeridian into a Polygon (no
/// crossing) or a MultiPolygon (crossing). Holes are assigned to the side they
/// fall on. Longitudes are wrapped into (-180, 180] on output.
fn split_polygon(rings: &[Vec<Coord>]) -> Geometry {
    let wrapped = || -> Vec<Vec<Coord>> { rings.iter().map(|r| wrap_all(r)).collect() };
    if rings.is_empty() || !rings.iter().any(|r| line_crosses(r)) {
        return Geometry::Polygon(wrapped());
    }
    let outer = unwrap_ring(&rings[0]);
    let minx = outer.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
    let maxx = outer.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);
    if maxx - minx >= 360.0 {
        // A ring wider than the globe cannot be cut cleanly at one meridian.
        return Geometry::Polygon(wrapped());
    }
    // The single antimeridian copy (180 + 360k) the ring straddles.
    let k = ((minx - 180.0) / 360.0).floor() as i64 + 1;
    let m = 180.0 + 360.0 * k as f64;
    if !(m > minx && m < maxx) {
        return Geometry::Polygon(wrapped());
    }
    let west = clip_ring(&outer, m, true).map(normalize_piece);
    let east = clip_ring(&outer, m, false).map(normalize_piece);
    let mut sides: Vec<(Vec<Vec<Coord>>, f64)> = Vec::new();
    if let Some(w) = west {
        sides.push((vec![w], -1.0)); // west sits below the split meridian
    }
    if let Some(e) = east {
        sides.push((vec![e], 1.0)); // east sits above it
    }
    if sides.len() < 2 {
        return Geometry::Polygon(wrapped());
    }
    // Assign each hole to the side(s) it clips into, preserving hole winding.
    for hole in &rings[1..] {
        let hole_un = unwrap_ring(hole);
        if let Some(hw) = clip_ring(&hole_un, m, true) {
            sides[0].0.push(normalize_piece(hw));
        }
        if let Some(he) = clip_ring(&hole_un, m, false) {
            sides[1].0.push(normalize_piece(he));
        }
    }
    Geometry::MultiPolygon(sides.into_iter().map(|(rings, _)| rings).collect())
}

/// Split a geographic geometry at the antimeridian. Non-crossing geometry is
/// returned with longitudes wrapped into (-180, 180]; crossing lines/polygons
/// become Multi* geometries cut at ±180. Only call for geographic (lon/lat)
/// input.
pub(super) fn split_at_antimeridian(geometry: &Geometry) -> Geometry {
    match geometry {
        Geometry::Empty => Geometry::Empty,
        Geometry::Point(p) => Geometry::Point(wrap(*p)),
        Geometry::MultiPoint(ps) => Geometry::MultiPoint(wrap_all(ps)),
        Geometry::LineString(pts) => {
            let mut segs = split_line(pts);
            if segs.len() == 1 {
                Geometry::LineString(segs.pop().unwrap())
            } else {
                Geometry::MultiLineString(segs)
            }
        }
        Geometry::MultiLineString(lines) => {
            let mut out: Vec<Vec<Coord>> = Vec::new();
            for line in lines {
                out.extend(split_line(line));
            }
            Geometry::MultiLineString(out)
        }
        Geometry::Polygon(rings) => split_polygon(rings),
        Geometry::MultiPolygon(polys) => {
            let mut out: Vec<Vec<Vec<Coord>>> = Vec::new();
            for rings in polys {
                match split_polygon(rings) {
                    Geometry::Polygon(r) => out.push(r),
                    Geometry::MultiPolygon(ps) => out.extend(ps),
                    _ => {}
                }
            }
            Geometry::MultiPolygon(out)
        }
        Geometry::Collection(items) => {
            Geometry::Collection(items.iter().map(split_at_antimeridian).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: f64, y: f64) -> Coord {
        Coord { x, y }
    }

    #[test]
    fn line_crossing_splits_into_two_segments_meeting_at_pm180() {
        // 170E -> 170W (short path crosses +180).
        let g = split_at_antimeridian(&Geometry::LineString(vec![c(170.0, 10.0), c(-170.0, 20.0)]));
        match g {
            Geometry::MultiLineString(parts) => {
                assert_eq!(parts.len(), 2);
                // First piece ends at +180, second starts at -180, same latitude.
                let end = *parts[0].last().unwrap();
                let start = parts[1][0];
                assert!((end.x - 180.0).abs() < 1e-9);
                assert!((start.x + 180.0).abs() < 1e-9);
                assert!((end.y - start.y).abs() < 1e-9);
                // The crossing latitude is the midpoint (equal Δlon each side).
                assert!((end.y - 15.0).abs() < 1e-9);
            }
            other => panic!("expected MultiLineString, got {other:?}"),
        }
    }

    #[test]
    fn non_crossing_line_is_returned_wrapped_unchanged() {
        let g = split_at_antimeridian(&Geometry::LineString(vec![c(10.0, 0.0), c(20.0, 5.0)]));
        assert_eq!(g, Geometry::LineString(vec![c(10.0, 0.0), c(20.0, 5.0)]));
    }

    #[test]
    fn polygon_crossing_dateline_splits_into_two_polygons() {
        // A box spanning 170E..170W (i.e. 170..190 unwrapped), 10S..10N.
        let ring = vec![
            c(170.0, 10.0),
            c(-170.0, 10.0),
            c(-170.0, -10.0),
            c(170.0, -10.0),
            c(170.0, 10.0),
        ];
        let g = split_at_antimeridian(&Geometry::Polygon(vec![ring]));
        match g {
            Geometry::MultiPolygon(polys) => {
                assert_eq!(polys.len(), 2);
                // Every vertex is within [-180, 180], and each piece stays on
                // one side of the dateline (all >= 170 or all <= -170).
                for poly in &polys {
                    let xs: Vec<f64> = poly[0].iter().map(|p| p.x).collect();
                    assert!(xs.iter().all(|x| x.abs() <= 180.0 + 1e-9));
                    let all_east = xs
                        .iter()
                        .all(|x| *x >= 170.0 - 1e-6 || (x - 180.0).abs() < 1e-6);
                    let all_west = xs
                        .iter()
                        .all(|x| *x <= -170.0 + 1e-6 || (x + 180.0).abs() < 1e-6);
                    assert!(all_east || all_west, "piece straddles dateline: {xs:?}");
                }
            }
            other => panic!("expected MultiPolygon, got {other:?}"),
        }
    }

    #[test]
    fn split_polygon_preserves_total_area() {
        // The two split pieces' unsigned planar areas sum to the original box's
        // area (20° lon x 20° lat = 400 deg², measured in unwrapped space).
        let ring = vec![
            c(170.0, 10.0),
            c(190.0, 10.0),
            c(190.0, -10.0),
            c(170.0, -10.0),
            c(170.0, 10.0),
        ];
        let original = signed_area(&ring).abs();
        assert!((original - 400.0).abs() < 1e-6);
        let g = split_at_antimeridian(&Geometry::Polygon(vec![ring]));
        if let Geometry::MultiPolygon(polys) = g {
            // Un-wrap each piece back for a planar area comparison.
            let total: f64 = polys
                .iter()
                .map(|p| {
                    let un = unwrap_ring(&p[0]);
                    signed_area(&un).abs()
                })
                .sum();
            assert!(
                (total - original).abs() < 1e-6,
                "area {total} != {original}"
            );
        } else {
            panic!("expected MultiPolygon");
        }
    }

    #[test]
    fn non_crossing_polygon_stays_a_single_polygon() {
        let ring = vec![
            c(10.0, 10.0),
            c(20.0, 10.0),
            c(20.0, 0.0),
            c(10.0, 0.0),
            c(10.0, 10.0),
        ];
        let g = split_at_antimeridian(&Geometry::Polygon(vec![ring.clone()]));
        assert_eq!(g, Geometry::Polygon(vec![ring]));
    }

    #[test]
    fn point_is_only_wrapped() {
        assert_eq!(
            split_at_antimeridian(&Geometry::Point(c(190.0, 5.0))),
            Geometry::Point(c(-170.0, 5.0))
        );
    }
}
