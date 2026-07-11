use crate::gis::error::{GisError, GisResult};

use super::math::{
    geodesic_line_length, geodesic_polygon_area, geodesic_polygon_perimeter, line_length,
    polygon_area, polygon_perimeter,
};
use super::model::{empty_geometry_error, Geometry, MeasureStats, UNSUPPORTED_OPTION};

/// How lengths and areas are computed (MENSURA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasureMode {
    /// Planar shoelace/hypot over projected coordinates (units of the CRS).
    Planar,
    /// Karney geodesic length and authalic area on the WGS84 ellipsoid;
    /// coordinates are lon/lat degrees, results are metres / m².
    GeodesicWgs84,
}

pub(super) fn validate_metric_names(metrics: &[String]) -> GisResult<(bool, bool)> {
    let mut area = false;
    let mut length = false;
    for metric in metrics {
        match metric.as_str() {
            "area" => area = true,
            "length" => length = true,
            other => {
                return Err(GisError::InvalidArgument(format!(
                    "{UNSUPPORTED_OPTION}: unsupported metric {other:?}"
                )))
            }
        }
    }
    Ok((area, length))
}

pub(super) fn measure_geometries(
    geometries: &[Geometry],
    mode: MeasureMode,
) -> GisResult<MeasureStats> {
    let mut stats = MeasureStats::default();
    for geometry in geometries {
        accumulate_measure(geometry, mode, &mut stats)?;
    }
    Ok(stats)
}

fn measured_line_length(points: &[super::model::Coord], mode: MeasureMode) -> f64 {
    match mode {
        MeasureMode::Planar => line_length(points),
        MeasureMode::GeodesicWgs84 => geodesic_line_length(points),
    }
}

fn measured_polygon(
    rings: &[Vec<super::model::Coord>],
    mode: MeasureMode,
) -> GisResult<(f64, f64)> {
    match mode {
        MeasureMode::Planar => Ok((polygon_area(rings)?, polygon_perimeter(rings))),
        MeasureMode::GeodesicWgs84 => Ok((
            geodesic_polygon_area(rings)?,
            geodesic_polygon_perimeter(rings),
        )),
    }
}

fn accumulate_measure(
    geometry: &Geometry,
    mode: MeasureMode,
    stats: &mut MeasureStats,
) -> GisResult<()> {
    match geometry {
        Geometry::Empty => Err(empty_geometry_error()),
        Geometry::Point(_) | Geometry::MultiPoint(_) => Ok(()),
        Geometry::LineString(points) => {
            stats.length += measured_line_length(points, mode);
            stats.has_length = true;
            Ok(())
        }
        Geometry::Polygon(rings) => {
            let (area, perimeter) = measured_polygon(rings, mode)?;
            stats.area += area;
            stats.length += perimeter;
            stats.has_area = true;
            stats.has_length = true;
            Ok(())
        }
        Geometry::MultiLineString(lines) => {
            for line in lines {
                stats.length += measured_line_length(line, mode);
                stats.has_length = true;
            }
            Ok(())
        }
        Geometry::MultiPolygon(polygons) => {
            for polygon in polygons {
                let (area, perimeter) = measured_polygon(polygon, mode)?;
                stats.area += area;
                stats.length += perimeter;
                stats.has_area = true;
                stats.has_length = true;
            }
            Ok(())
        }
        Geometry::Collection(geometries) => {
            if geometries.is_empty() {
                return Err(empty_geometry_error());
            }
            for item in geometries {
                accumulate_measure(item, mode, stats)?;
            }
            Ok(())
        }
    }
}
