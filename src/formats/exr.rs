use super::hdr::HdrImage;
use crate::core::error::{RenderError, RenderResult};
use std::path::Path;

#[cfg(feature = "images")]
pub fn load_exr<P: AsRef<Path>>(path: P) -> RenderResult<HdrImage> {
    use exr::prelude::{read_first_flat_layer_from_file, FlatSamples};

    let path = path.as_ref();
    let image = read_first_flat_layer_from_file(path).map_err(|e| {
        RenderError::io(format!(
            "Failed to read EXR file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let width = image.layer_data.size.width();
    let height = image.layer_data.size.height();
    if width == 0 || height == 0 {
        return Err(RenderError::upload(format!(
            "EXR file '{}' dimensions cannot be zero",
            path.display()
        )));
    }
    if width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(RenderError::upload(format!(
            "EXR image '{}' dimensions {}x{} exceed supported u32 range",
            path.display(),
            width,
            height
        )));
    }
    let pixel_count = width.checked_mul(height).ok_or_else(|| {
        RenderError::upload(format!(
            "EXR file '{}' dimensions {}x{} overflow",
            path.display(),
            width,
            height
        ))
    })?;

    fn samples_to_f32(samples: &FlatSamples) -> Vec<f32> {
        match samples {
            FlatSamples::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            FlatSamples::F32(v) => v.clone(),
            FlatSamples::U32(v) => v.iter().map(|x| *x as f32).collect(),
        }
    }

    let mut r = None;
    let mut g = None;
    let mut b = None;
    for channel in &image.layer_data.channel_data.list {
        match channel.name.to_string().as_str() {
            "R" => r = Some(samples_to_f32(&channel.sample_data)),
            "G" => g = Some(samples_to_f32(&channel.sample_data)),
            "B" => b = Some(samples_to_f32(&channel.sample_data)),
            _ => {}
        }
    }

    let (r, g, b) = match (r, g, b) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => {
            return Err(RenderError::upload(format!(
                "EXR file '{}' is missing one of the required R, G, B channels",
                path.display()
            )))
        }
    };
    if r.len() != pixel_count || g.len() != pixel_count || b.len() != pixel_count {
        return Err(RenderError::upload(format!(
            "EXR file '{}' channel sample counts do not match {}x{} dimensions",
            path.display(),
            width,
            height
        )));
    }

    let data_len = pixel_count.checked_mul(3).ok_or_else(|| {
        RenderError::upload(format!(
            "EXR file '{}' RGB output length overflows for {} pixels",
            path.display(),
            pixel_count
        ))
    })?;
    let mut data = Vec::with_capacity(data_len);
    for i in 0..pixel_count {
        data.push(r[i]);
        data.push(g[i]);
        data.push(b[i]);
    }

    Ok(HdrImage {
        width: width as u32,
        height: height as u32,
        data,
    })
}

#[cfg(not(feature = "images"))]
pub fn load_exr<P: AsRef<Path>>(path: P) -> RenderResult<HdrImage> {
    let _ = path.as_ref();
    Err(RenderError::upload(
        "EXR environment decoding requires the 'images' feature, which was disabled in this build"
            .to_string(),
    ))
}

#[cfg(all(test, feature = "images"))]
mod exr_tests {
    use super::*;
    use exr::prelude::*;
    use half::f16;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_exr(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        p.push(format!(
            "forge3d_exr_{name}_{}_{}.exr",
            std::process::id(),
            nonce
        ));
        p
    }

    fn write_exr_channels(
        path: &Path,
        width: usize,
        height: usize,
        channels: Vec<(&str, FlatSamples)>,
    ) {
        let mut list = SmallVec::<[AnyChannel<FlatSamples>; 4]>::new();
        for (name, samples) in channels {
            list.push(AnyChannel {
                name: Text::new_or_none(name).expect("valid channel name"),
                sample_data: samples,
                quantize_linearly: false,
                sampling: Vec2(1, 1),
            });
        }
        let image = Image::from_channels((width, height), AnyChannels::sort(list));
        image.write().to_file(path).expect("write exr fixture");
    }

    #[test]
    fn f32_shuffled_order_ignores_extra_channels() {
        let r = [0.1f32, 0.2, 0.3, 0.4];
        let g = [0.5f32, 0.6, 0.7, 0.8];
        let b = [0.9f32, 1.0, 1.1, 1.2];
        let path = tmp_exr("f32_shuffled");
        write_exr_channels(
            &path,
            2,
            2,
            vec![
                ("B", FlatSamples::F32(b.to_vec())),
                ("Z", FlatSamples::F32(vec![9.0, 9.0, 9.0, 9.0])),
                ("R", FlatSamples::F32(r.to_vec())),
                ("A", FlatSamples::F32(vec![1.0, 1.0, 1.0, 1.0])),
                ("G", FlatSamples::F32(g.to_vec())),
            ],
        );
        let img = load_exr(&path).expect("decode f32 exr");
        std::fs::remove_file(&path).ok();
        assert_eq!((img.width, img.height), (2, 2));
        let mut expected = Vec::new();
        for i in 0..4 {
            expected.push(r[i]);
            expected.push(g[i]);
            expected.push(b[i]);
        }
        assert_eq!(img.data, expected);
    }

    #[test]
    fn f16_samples_shuffled_order() {
        let r = [0.25f32, 0.5, 0.75, 1.0];
        let g = [0.125f32, 0.25, 0.375, 0.5];
        let b = [1.5f32, 2.0, 2.5, 3.0];
        let half = |xs: &[f32]| FlatSamples::F16(xs.iter().map(|&x| f16::from_f32(x)).collect());
        let path = tmp_exr("f16");
        write_exr_channels(
            &path,
            2,
            2,
            vec![("G", half(&g)), ("B", half(&b)), ("R", half(&r))],
        );
        let img = load_exr(&path).expect("decode f16 exr");
        std::fs::remove_file(&path).ok();
        assert_eq!((img.width, img.height), (2, 2));
        for i in 0..4 {
            assert_eq!(img.data[i * 3], f16::from_f32(r[i]).to_f32());
            assert_eq!(img.data[i * 3 + 1], f16::from_f32(g[i]).to_f32());
            assert_eq!(img.data[i * 3 + 2], f16::from_f32(b[i]).to_f32());
        }
    }

    #[test]
    fn u32_samples_shuffled_order() {
        let r = [0u32, 1, 2, 3];
        let g = [10u32, 20, 30, 40];
        let b = [100u32, 200, 300, 400];
        let path = tmp_exr("u32");
        write_exr_channels(
            &path,
            2,
            2,
            vec![
                ("B", FlatSamples::U32(b.to_vec())),
                ("R", FlatSamples::U32(r.to_vec())),
                ("G", FlatSamples::U32(g.to_vec())),
            ],
        );
        let img = load_exr(&path).expect("decode u32 exr");
        std::fs::remove_file(&path).ok();
        assert_eq!((img.width, img.height), (2, 2));
        for i in 0..4 {
            assert_eq!(img.data[i * 3], r[i] as f32);
            assert_eq!(img.data[i * 3 + 1], g[i] as f32);
            assert_eq!(img.data[i * 3 + 2], b[i] as f32);
        }
    }

    #[test]
    fn missing_channel_is_an_error() {
        let path = tmp_exr("missing_b");
        write_exr_channels(
            &path,
            2,
            2,
            vec![
                ("R", FlatSamples::F32(vec![0.1, 0.2, 0.3, 0.4])),
                ("G", FlatSamples::F32(vec![0.5, 0.6, 0.7, 0.8])),
            ],
        );
        let err = load_exr(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(format!("{err}").contains("missing"));
    }
}

#[cfg(all(test, not(feature = "images")))]
mod exr_feature_off_tests {
    use super::*;

    #[test]
    fn load_exr_requires_images_feature() {
        let err = load_exr("nonexistent.exr").unwrap_err();
        assert!(format!("{err}").contains("images"));
    }
}
