use crate::gis::error::{GisError, GisResult};

use super::math::{line_length, polygon_area, polygon_perimeter};
use super::model::{empty_geometry_error, Geometry, MeasureStats, UNSUPPORTED_OPTION};

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

pub(super) fn measure_geometries(geometries: &[Geometry]) -> GisResult<MeasureStats> {
    let mut stats = MeasureStats::default();
    for geometry in geometries {
        accumulate_measure(geometry, &mut stats)?;
    }
    Ok(stats)
}

fn accumulate_measure(geometry: &Geometry, stats: &mut MeasureStats) -> GisResult<()> {
    match geometry {
        Geometry::Empty => Err(empty_geometry_error()),
        Geometry::Point(_) | Geometry::MultiPoint(_) => Ok(()),
        Geometry::LineString(points) => {
            stats.length += line_length(points);
            stats.has_length = true;
            Ok(())
        }
        Geometry::Polygon(rings) => {
            stats.area += polygon_area(rings)?;
            stats.length += polygon_perimeter(rings);
            stats.has_area = true;
            stats.has_length = true;
            Ok(())
        }
        Geometry::MultiLineString(lines) => {
            for line in lines {
                stats.length += line_length(line);
                stats.has_length = true;
            }
            Ok(())
        }
        Geometry::MultiPolygon(polygons) => {
            for polygon in polygons {
                stats.area += polygon_area(polygon)?;
                stats.length += polygon_perimeter(polygon);
                stats.has_area = true;
                stats.has_length = true;
            }
            Ok(())
        }
        Geometry::GeometryCollection(geometries) => {
            if geometries.is_empty() {
                return Err(empty_geometry_error());
            }
            for item in geometries {
                accumulate_measure(item, stats)?;
            }
            Ok(())
        }
    }
}
