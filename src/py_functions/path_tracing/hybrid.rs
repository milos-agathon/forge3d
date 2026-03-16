use super::*;

#[pyfunction]
pub(crate) fn hybrid_render(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: Option<&Bound<'_, PyAny>>,
    camera: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    fn py_any_to_vec3(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
        let (x, y, z): (f32, f32, f32) = obj.extract()?;
        Ok(Vec3::new(x, y, z))
    }

    struct CameraParams {
        origin: Vec3,
        target: Vec3,
        up: Vec3,
        fov_degrees: f32,
    }

    impl Default for CameraParams {
        fn default() -> Self {
            Self {
                origin: Vec3::new(0.0, 0.0, 5.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_degrees: 45.0,
            }
        }
    }

    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("image dimensions must be positive"));
    }

    let sdf_scene = if let Some(scene_obj) = scene {
        let extracted: PyRef<'_, PySdfScene> = scene_obj.extract()?;
        extracted.0.clone()
    } else {
        crate::sdf::SdfScene::new()
    };

    let mut cam = CameraParams::default();
    if let Some(camera_obj) = camera {
        let camera_dict = camera_obj.downcast::<PyDict>().ok();
        let update_vec3 = |key: &str, out: &mut Vec3| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = py_any_to_vec3(&value)?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = py_any_to_vec3(&value)?;
            }
            Ok(())
        };

        let update_f32 = |key: &str, out: &mut f32| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = value.extract()?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = value.extract()?;
            }
            Ok(())
        };

        update_vec3("origin", &mut cam.origin)?;
        update_vec3("target", &mut cam.target)?;
        update_vec3("up", &mut cam.up)?;
        update_f32("fov_degrees", &mut cam.fov_degrees)?;
    }

    let mut forward = (cam.target - cam.origin).normalize_or_zero();
    if forward.length_squared() == 0.0 {
        forward = Vec3::new(0.0, 0.0, -1.0);
    }

    let up_hint = cam.up.normalize_or_zero();
    let up_hint = if up_hint.length_squared() == 0.0 {
        Vec3::Y
    } else {
        up_hint
    };

    let mut right = forward.cross(up_hint).normalize_or_zero();
    if right.length_squared() == 0.0 {
        right = Vec3::X;
    }

    let mut up = right.cross(forward).normalize_or_zero();
    if up.length_squared() == 0.0 {
        up = Vec3::Y;
    }

    let hybrid_scene = crate::sdf::HybridScene::sdf_only(sdf_scene);
    let w = width as usize;
    let h = height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    let aspect = width as f32 / height as f32;
    let half_fov = (cam.fov_degrees.to_radians() * 0.5).tan();
    let half_w = aspect * half_fov;
    let half_h = half_fov;
    let sky_color = [153u8, 178u8, 229u8];

    for y in 0..h {
        let ndc_y = (1.0 - ((y as f32 + 0.5) / height as f32)) * 2.0 - 1.0;
        for x in 0..w {
            let ndc_x = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;

            let mut dir = right * (ndc_x * half_w) + up * (ndc_y * half_h) - forward;
            dir = dir.normalize_or_zero();
            if dir.length_squared() == 0.0 {
                dir = -forward;
            }

            let ray = HybridRay {
                origin: cam.origin,
                direction: dir,
                tmin: 0.001,
                tmax: 100.0,
            };
            let result = hybrid_scene.intersect(ray);

            let pixel_index = (y * w + x) * 4;
            if result.hit {
                let color = match result.material_id {
                    1 => [204u8, 51u8, 51u8],
                    2 => [51u8, 204u8, 51u8],
                    3 => [51u8, 51u8, 204u8],
                    4 => [210u8, 210u8, 210u8],
                    _ => [230u8, 153u8, 76u8],
                };
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&color);
            } else {
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&sky_color);
            }
            pixels[pixel_index + 3] = 255;
        }
    }

    let arr1 = PyArray1::<u8>::from_vec_bound(py, pixels);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}
