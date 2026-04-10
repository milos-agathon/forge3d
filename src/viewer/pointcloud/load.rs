use crate::viewer::pointcloud::PointInstance3D;

pub(super) struct LoadResult {
    pub points: Vec<PointInstance3D>,
    pub has_rgb: bool,
    pub has_intensity: bool,
}

pub(super) fn load_laz_points(path: &str, max_points: usize) -> Result<LoadResult, String> {
    use las::{Read, Reader};

    eprintln!("[pointcloud] Opening file: {}", path);

    let mut reader =
        Reader::from_path(path).map_err(|e| format!("Failed to open LAS/LAZ file: {}", e))?;

    let header = reader.header();
    let total_points = header.number_of_points() as usize;
    let bounds = header.bounds();

    eprintln!("[pointcloud] File has {} points", total_points);
    eprintln!(
        "[pointcloud] Bounds: X({:.1}, {:.1}) Y({:.1}, {:.1}) Z({:.1}, {:.1})",
        bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y, bounds.min.z, bounds.max.z
    );

    let min_z = bounds.min.z;
    let max_z = bounds.max.z;
    let z_range = max_z - min_z;
    let stride = if total_points > max_points {
        total_points / max_points
    } else {
        1
    };

    let n_read = (total_points / stride).min(max_points);
    eprintln!("[pointcloud] Loading {} points (stride {})", n_read, stride);

    let mut points = Vec::with_capacity(n_read);
    let mut has_rgb = false;
    let mut intensity_min: u16 = u16::MAX;
    let mut intensity_max: u16 = 0;

    for (i, point_result) in reader.points().enumerate() {
        if stride > 1 && i % stride != 0 {
            continue;
        }
        if points.len() >= max_points {
            break;
        }

        let point = point_result.map_err(|e| format!("Error reading point: {}", e))?;
        let px = point.x as f32;
        let py = point.z as f32;
        let pz = point.y as f32;

        let elevation_norm = if z_range > 0.0 {
            ((point.z - min_z) / z_range).clamp(0.0, 1.0) as f32
        } else {
            0.5
        };

        let rgb = if let Some(color) = point.color {
            has_rgb = true;
            [
                color.red as f32 / 65535.0,
                color.green as f32 / 65535.0,
                color.blue as f32 / 65535.0,
            ]
        } else {
            [1.0, 1.0, 1.0]
        };

        intensity_min = intensity_min.min(point.intensity);
        intensity_max = intensity_max.max(point.intensity);

        points.push(PointInstance3D {
            position: [px, py, pz],
            elevation_norm,
            rgb,
            intensity: point.intensity as f32,
            size: 1.0,
            _pad: [0.0; 3],
        });

        if points.len() % 100000 == 0 {
            eprintln!("[pointcloud] Loaded {} points...", points.len());
        }
    }

    println!("[pointcloud] Loaded {} points total", points.len());
    println!("[pointcloud] Has RGB: {}", has_rgb);
    println!(
        "[pointcloud] Intensity range: {} - {}",
        intensity_min, intensity_max
    );

    if let Some(p) = points.first() {
        println!(
            "[pointcloud] Sample point - rgb: {:?}, intensity: {}",
            p.rgb, p.intensity
        );
    }

    let intensity_range = (intensity_max - intensity_min) as f32;
    if intensity_range > 0.0 {
        let intensity_min_f = intensity_min as f32;
        for p in points.iter_mut() {
            p.intensity = (p.intensity - intensity_min_f) / intensity_range;
        }
        println!("[pointcloud] Normalized intensity to 0-1 range");
    } else {
        for p in points.iter_mut() {
            p.intensity = 0.5;
        }
        println!("[pointcloud] All intensities same, using 0.5");
    }

    if let Some(p) = points.first() {
        println!(
            "[pointcloud] Final sample - rgb: {:?}, intensity: {:.3}, elev: {:.3}",
            p.rgb, p.intensity, p.elevation_norm
        );
    }

    Ok(LoadResult {
        points,
        has_rgb,
        has_intensity: intensity_range > 0.0,
    })
}
