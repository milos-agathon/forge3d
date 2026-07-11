use crate::gis::error::{GisError, GisResult};

use super::math::{distance, ring_signed_area_and_centroid, unwrap_dateline, wrap_lon};
use super::model::{
    empty_geometry_error, CentroidStats, Coord, Geometry, EPSILON, INVALID_GEOMETRY,
};

pub(super) fn centroid_for_geometries(
    geometries: &[Geometry],
    geographic: bool,
) -> GisResult<Coord> {
    // MENSURA dateline handling: a geographic polygon spanning 179° → -179°
    // must be unwrapped before the shoelace runs, or the centroid lands on
    // the wrong side of the planet. Only applied when the data plausibly is
    // lon/lat degrees AND an antimeridian jump actually exists.
    if geographic {
        let mut unwrapped = geometries.to_vec();
        if unwrap_dateline(&mut unwrapped) {
            let centroid = centroid_for_unwrapped(&unwrapped)?;
            return Ok(Coord {
                x: wrap_lon(centroid.x),
                y: centroid.y,
            });
        }
    }
    centroid_for_unwrapped(geometries)
}

fn centroid_for_unwrapped(geometries: &[Geometry]) -> GisResult<Coord> {
    let mut stats = CentroidStats::default();
    for geometry in geometries {
        accumulate_centroid(geometry, &mut stats)?;
    }
    if stats.polygon_weight > EPSILON {
        return Ok(Coord {
            x: stats.polygon_x / stats.polygon_weight,
            y: stats.polygon_y / stats.polygon_weight,
        });
    }
    if stats.line_weight > EPSILON {
        return Ok(Coord {
            x: stats.line_x / stats.line_weight,
            y: stats.line_y / stats.line_weight,
        });
    }
    if stats.point_count > 0 {
        let count = stats.point_count as f64;
        return Ok(Coord {
            x: stats.point_x / count,
            y: stats.point_y / count,
        });
    }
    Err(GisError::InvalidGeometry(format!(
        "{}: geometry has no centroid",
        super::model::EMPTY_GEOMETRY
    )))
}

fn accumulate_centroid(geometry: &Geometry, stats: &mut CentroidStats) -> GisResult<()> {
    match geometry {
        Geometry::Empty => Err(empty_geometry_error()),
        Geometry::Point(point) => {
            stats.point_count += 1;
            stats.point_x += point.x;
            stats.point_y += point.y;
            Ok(())
        }
        Geometry::MultiPoint(points) => {
            for point in points {
                stats.point_count += 1;
                stats.point_x += point.x;
                stats.point_y += point.y;
            }
            Ok(())
        }
        Geometry::LineString(points) => {
            accumulate_line_centroid(points, stats);
            Ok(())
        }
        Geometry::MultiLineString(lines) => {
            for line in lines {
                accumulate_line_centroid(line, stats);
            }
            Ok(())
        }
        Geometry::Polygon(rings) => accumulate_polygon_centroid(rings, stats),
        Geometry::MultiPolygon(polygons) => {
            for polygon in polygons {
                accumulate_polygon_centroid(polygon, stats)?;
            }
            Ok(())
        }
        Geometry::Collection(geometries) => {
            for item in geometries {
                accumulate_centroid(item, stats)?;
            }
            Ok(())
        }
    }
}

fn accumulate_line_centroid(points: &[Coord], stats: &mut CentroidStats) {
    for segment in points.windows(2) {
        let length = distance(segment[0], segment[1]);
        if length <= EPSILON {
            continue;
        }
        stats.line_weight += length;
        stats.line_x += ((segment[0].x + segment[1].x) * 0.5) * length;
        stats.line_y += ((segment[0].y + segment[1].y) * 0.5) * length;
    }
}

fn accumulate_polygon_centroid(rings: &[Vec<Coord>], stats: &mut CentroidStats) -> GisResult<()> {
    let mut total_area = 0.0;
    let mut total_x = 0.0;
    let mut total_y = 0.0;
    for (index, ring) in rings.iter().enumerate() {
        let (area, centroid) = ring_signed_area_and_centroid(ring).ok_or_else(|| {
            GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: polygon ring has zero area"))
        })?;
        let weight = if index == 0 { area.abs() } else { -area.abs() };
        total_area += weight;
        total_x += centroid.x * weight;
        total_y += centroid.y * weight;
    }
    if total_area <= EPSILON {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: polygon area must be positive"
        )));
    }
    stats.polygon_weight += total_area;
    stats.polygon_x += total_x;
    stats.polygon_y += total_y;
    Ok(())
}
