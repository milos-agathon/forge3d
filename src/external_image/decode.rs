//! Image decoding utilities for PNG and JPEG files.

use super::types::{ImageImportConfig, ImageSourceFormat};
use crate::core::error::{RenderError, RenderResult};
use std::fs::File;
use std::path::Path;

/// Decode an image file to RGBA8 data.
pub fn decode_image_file(
    path: &Path,
    config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| RenderError::io("Cannot determine image format from file extension"))?;

    match extension.as_str() {
        "png" => decode_png_file(path, config),
        "jpg" | "jpeg" => decode_jpeg_file(path, config),
        _ => Err(RenderError::io(format!(
            "Unsupported image format: {}",
            extension
        ))),
    }
}

fn decode_file(
    path: &Path,
    config: &ImageImportConfig,
    format: image::ImageFormat,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    let reader =
        image::ImageReader::with_format(std::io::BufReader::new(File::open(path)?), format);
    let image = reader
        .decode()
        .map_err(|e| RenderError::io(format!("Invalid image: {e}")))?;
    let source = match (format, image.color()) {
        (image::ImageFormat::Jpeg, _) => ImageSourceFormat::JpegRgb,
        (_, image::ColorType::L8 | image::ColorType::L16) => ImageSourceFormat::PngGrayscale,
        (_, image::ColorType::Rgb8 | image::ColorType::Rgb16) => ImageSourceFormat::PngRgb,
        _ => ImageSourceFormat::PngRgba,
    };
    let (width, height) = (image.width(), image.height());
    if width > config.max_dimension || height > config.max_dimension {
        return Err(RenderError::Upload(
            "Image exceeds configured maximum dimensions".into(),
        ));
    }
    let mut rgba = image.into_rgba8().into_raw();
    if config.premultiply_alpha {
        for pixel in rgba.chunks_exact_mut(4) {
            let alpha = u16::from(pixel[3]);
            for channel in &mut pixel[..3] {
                *channel = ((u16::from(*channel) * alpha + 127) / 255) as u8;
            }
        }
    }
    Ok((rgba, width, height, source))
}

pub fn decode_png_file(
    path: &Path,
    config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    decode_file(path, config, image::ImageFormat::Png)
}

pub fn decode_jpeg_file(
    path: &Path,
    config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    decode_file(path, config, image::ImageFormat::Jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_png_jpeg_and_invalid_bytes() {
        let dir = std::env::temp_dir().join(format!("forge3d-image-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for extension in ["png", "jpg"] {
            let path = dir.join(format!("small.{extension}"));
            // Neutral grayscale avoids chroma subsampling ambiguity; rows and columns vary.
            let source = image::RgbImage::from_fn(16, 16, |x, y| {
                let v = (32 + x * 8 + y * 4) as u8;
                image::Rgb([v, v, v])
            });
            if extension == "jpg" {
                image::codecs::jpeg::JpegEncoder::new_with_quality(
                    File::create(&path).unwrap(),
                    100,
                )
                .encode_image(&source)
                .unwrap();
            } else {
                source.save(&path).unwrap();
            }
            let (pixels, w, h, _) =
                decode_image_file(&path, &ImageImportConfig::default()).unwrap();
            assert_eq!((w, h), (16, 16));
            for (pixel, expected) in pixels.chunks_exact(4).zip(source.pixels()) {
                for channel in 0..3 {
                    if extension == "png" {
                        assert_eq!(pixel[channel], expected[channel]);
                    } else {
                        assert!(
                            pixel[channel].abs_diff(expected[channel]) <= 2,
                            "JPEG quality-100 grayscale DCT quantization tolerance"
                        );
                    }
                }
                assert_eq!(pixel[3], 255);
            }
            std::fs::write(&path, b"invalid image").unwrap();
            assert!(decode_image_file(&path, &ImageImportConfig::default()).is_err());
        }
        let path = dir.join("alpha.png");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([100, 80, 60, 128]))
            .save(&path)
            .unwrap();
        let config = ImageImportConfig {
            premultiply_alpha: true,
            ..Default::default()
        };
        assert_eq!(
            decode_image_file(&path, &config).unwrap().0,
            [50, 40, 30, 128]
        );
    }
}
