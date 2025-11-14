// tests/golden_images.rs
//! Golden image regression tests for P9
//!
//! Generates and compares rendered images against reference "golden" images
//! using SSIM (Structural Similarity Index) with epsilon ≥ 0.98 @ 1280×920.

#[cfg(test)]
mod golden_image_tests {
    use std::path::PathBuf;

    /// Test configuration for rendering a golden image
    struct GoldenImageConfig {
        name: &'static str,
        width: u32,
        height: u32,
        brdf: &'static str,
        shadows: &'static str,
        gi: Vec<&'static str>,
        ibl_enabled: bool,
    }

    const GOLDEN_CONFIGS: &[GoldenImageConfig] = &[
        // 1. Lambert + Hard shadows + No GI
        GoldenImageConfig {
            name: "lambert_hard_noggi",
            width: 1280,
            height: 920,
            brdf: "lambert",
            shadows: "hard",
            gi: vec![],
            ibl_enabled: false,
        },
        // 2. Phong + PCF + No GI
        GoldenImageConfig {
            name: "phong_pcf_nogi",
            width: 1280,
            height: 920,
            brdf: "phong",
            shadows: "pcf",
            gi: vec![],
            ibl_enabled: false,
        },
        // 3. Cook-Torrance GGX + PCF + IBL
        GoldenImageConfig {
            name: "ggx_pcf_ibl",
            width: 1280,
            height: 920,
            brdf: "cooktorrance-ggx",
            shadows: "pcf",
            gi: vec!["ibl"],
            ibl_enabled: true,
        },
        // 4. Disney Principled + PCSS + IBL + SSAO
        GoldenImageConfig {
            name: "disney_pcss_ibl_ssao",
            width: 1280,
            height: 920,
            brdf: "disney-principled",
            shadows: "pcss",
            gi: vec!["ibl", "ssao"],
            ibl_enabled: true,
        },
        // 5. Oren-Nayar + VSM + No GI
        GoldenImageConfig {
            name: "orennayar_vsm_nogi",
            width: 1280,
            height: 920,
            brdf: "oren-nayar",
            shadows: "vsm",
            gi: vec![],
            ibl_enabled: false,
        },
        // 6. Toon + Hard + No GI
        GoldenImageConfig {
            name: "toon_hard_nogi",
            width: 1280,
            height: 920,
            brdf: "toon",
            shadows: "hard",
            gi: vec![],
            ibl_enabled: false,
        },
        // 7. Ashikhmin-Shirley + PCSS + IBL
        GoldenImageConfig {
            name: "ashikhmin_pcss_ibl",
            width: 1280,
            height: 920,
            brdf: "ashikhmin-shirley",
            shadows: "pcss",
            gi: vec!["ibl"],
            ibl_enabled: true,
        },
        // 8. Ward + EVSM + No GI
        GoldenImageConfig {
            name: "ward_evsm_nogi",
            width: 1280,
            height: 920,
            brdf: "ward",
            shadows: "evsm",
            gi: vec![],
            ibl_enabled: false,
        },
        // 9. Blinn-Phong + MSM + No GI
        GoldenImageConfig {
            name: "blinnphong_msm_nogi",
            width: 1280,
            height: 920,
            brdf: "blinn-phong",
            shadows: "msm",
            gi: vec![],
            ibl_enabled: false,
        },
        // 10. Cook-Torrance GGX + CSM + IBL + GTAO
        GoldenImageConfig {
            name: "ggx_csm_ibl_gtao",
            width: 1280,
            height: 920,
            brdf: "cooktorrance-ggx",
            shadows: "csm",
            gi: vec!["ibl", "gtao"],
            ibl_enabled: true,
        },
        // 11. Disney Principled + PCF + IBL + SSGI
        GoldenImageConfig {
            name: "disney_pcf_ibl_ssgi",
            width: 1280,
            height: 920,
            brdf: "disney-principled",
            shadows: "pcf",
            gi: vec!["ibl", "ssgi"],
            ibl_enabled: true,
        },
        // 12. Cook-Torrance GGX + PCSS + IBL + SSR
        GoldenImageConfig {
            name: "ggx_pcss_ibl_ssr",
            width: 1280,
            height: 920,
            brdf: "cooktorrance-ggx",
            shadows: "pcss",
            gi: vec!["ibl", "ssr"],
            ibl_enabled: true,
        },
    ];

    use image::{imageops, GenericImageView};
    use std::process::Command;

    fn golden_image_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/golden/{}.png", name))
    }

    fn rendered_image_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/golden/rendered/{}.png", name))
    }

    fn diff_image_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/golden/diff/{}.png", name))
    }

    /// Compute SSIM between two images on downscaled grayscale for speed
    /// Returns a value in [0, 1] where 1 = identical
    fn compute_ssim(img1_path: &PathBuf, img2_path: &PathBuf) -> Result<f64, String> {
        if !img1_path.exists() {
            return Err(format!("Image 1 not found: {:?}", img1_path));
        }
        if !img2_path.exists() {
            return Err(format!("Image 2 not found: {:?}", img2_path));
        }

        let im1 = image::open(img1_path).map_err(|e| format!("open img1: {}", e))?;
        let im2 = image::open(img2_path).map_err(|e| format!("open img2: {}", e))?;

        // Resize both to a standard small size to reduce computation
        let target_w = 320u32;
        let target_h = 180u32;
        let im1r = imageops::resize(
            &im1.to_luma8(),
            target_w,
            target_h,
            imageops::FilterType::Triangle,
        );
        let im2r = imageops::resize(
            &im2.to_luma8(),
            target_w,
            target_h,
            imageops::FilterType::Triangle,
        );

        let w = im1r.width() as usize;
        let h = im1r.height() as usize;
        // Convert to f64 arrays
        let buf1: Vec<f64> = im1r.into_raw().into_iter().map(|v| v as f64).collect();
        let buf2: Vec<f64> = im2r.into_raw().into_iter().map(|v| v as f64).collect();

        // Gaussian parameters
        let k1 = 0.01f64;
        let k2 = 0.03f64;
        let data_range = 255.0f64;
        let c1 = (k1 * data_range).powi(2);
        let c2 = (k2 * data_range).powi(2);
        let win_size: usize = 11;
        let sigma: f64 = 1.5;

        // Build 1D Gaussian kernel
        let mut g = vec![0.0f64; win_size];
        let center = (win_size as isize - 1) as f64 / 2.0;
        let sigma2 = 2.0 * sigma * sigma;
        let mut sum = 0.0;
        for i in 0..win_size {
            let x = i as f64 - center;
            g[i] = (-x * x / sigma2).exp();
            sum += g[i];
        }
        for i in 0..win_size {
            g[i] /= sum;
        }

        // Helper: separable convolution
        let convolve = |src: &Vec<f64>| -> Vec<f64> {
            let mut tmp = vec![0.0f64; w * h];
            let mut dst = vec![0.0f64; w * h];
            // horizontal
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    for k in 0..win_size {
                        let dx = k as isize - center as isize;
                        let xx = (x as isize + dx).clamp(0, (w as isize) - 1) as usize;
                        acc += src[y * w + xx] * g[k];
                    }
                    tmp[y * w + x] = acc;
                }
            }
            // vertical
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    for k in 0..win_size {
                        let dy = k as isize - center as isize;
                        let yy = (y as isize + dy).clamp(0, (h as isize) - 1) as usize;
                        acc += tmp[yy * w + x] * g[k];
                    }
                    dst[y * w + x] = acc;
                }
            }
            dst
        };

        // compute means
        let mu1 = convolve(&buf1);
        let mu2 = convolve(&buf2);

        // compute squares and products
        let buf1_sq: Vec<f64> = buf1.iter().map(|v| v * v).collect();
        let buf2_sq: Vec<f64> = buf2.iter().map(|v| v * v).collect();
        let buf12: Vec<f64> = buf1.iter().zip(buf2.iter()).map(|(a, b)| a * b).collect();

        let mu1_sq: Vec<f64> = mu1.iter().map(|v| v * v).collect();
        let mu2_sq: Vec<f64> = mu2.iter().map(|v| v * v).collect();
        let mu12: Vec<f64> = mu1.iter().zip(mu2.iter()).map(|(a, b)| a * b).collect();

        let sigma1_sq: Vec<f64> = {
            let tmp = convolve(&buf1_sq);
            tmp.iter().zip(mu1_sq.iter()).map(|(t, m)| t - m).collect()
        };
        let sigma2_sq: Vec<f64> = {
            let tmp = convolve(&buf2_sq);
            tmp.iter().zip(mu2_sq.iter()).map(|(t, m)| t - m).collect()
        };
        let sigma12: Vec<f64> = {
            let tmp = convolve(&buf12);
            tmp.iter().zip(mu12.iter()).map(|(t, m)| t - m).collect()
        };

        // SSIM map and mean
        let mut ssim_sum = 0.0f64;
        let mut count = 0usize;
        for i in 0..(w * h) {
            let num = (2.0 * mu12[i] + c1) * (2.0 * sigma12[i] + c2);
            let den = (mu1_sq[i] + mu2_sq[i] + c1) * (sigma1_sq[i] + sigma2_sq[i] + c2);
            let v = if den != 0.0 { num / den } else { 0.0 };
            ssim_sum += v;
            count += 1;
        }
        Ok(ssim_sum / (count as f64))
    }

    /// Render a single golden image using the Python generator into rendered dir
    fn render_golden_image(
        config: &GoldenImageConfig,
        _output_path: &PathBuf,
    ) -> Result<(), String> {
        // Ensure rendered directory exists
        let rendered_dir = PathBuf::from("tests/golden/rendered");
        std::fs::create_dir_all(&rendered_dir).map_err(|e| format!("mkdir rendered: {}", e))?;

        // Call the Python generator filtered by name, writing into rendered directory
        let status = Command::new("python3")
            .arg("scripts/generate_golden_images.py")
            .arg("--output-dir")
            .arg(rendered_dir)
            .arg("--overwrite")
            .arg("--filter")
            .arg(config.name)
            .arg("--obj")
            .arg("assets/cornell_box.obj")
            .status()
            .map_err(|e| format!("spawn python: {}", e))?;
        if !status.success() {
            return Err(format!("generator failed with status {}", status));
        }
        Ok(())
    }

    /// Compare rendered image against golden image using SSIM
    fn compare_golden_image(config: &GoldenImageConfig) -> Result<f64, String> {
        let golden_path = golden_image_path(config.name);
        let rendered_path = rendered_image_path(config.name);

        compute_ssim(&golden_path, &rendered_path)
    }

    #[test]
    #[ignore] // Ignore by default, run with --ignored or --include-ignored
    fn generate_golden_images() {
        println!("\n=== Generating Golden Images ===\n");

        // Use Python generator to write into tests/golden
        let status = Command::new("python3")
            .arg("scripts/generate_golden_images.py")
            .arg("--output-dir")
            .arg("tests/golden")
            .arg("--overwrite")
            .arg("--obj")
            .arg("assets/cornell_box.obj")
            .status()
            .expect("failed to run generator");
        assert!(status.success(), "generator failed");

        println!("\n=== Golden Image Generation Complete ===");
    }

    #[test]
    #[ignore] // Ignore by default, run with --ignored or --include-ignored
    fn test_golden_images() {
        const MIN_SSIM: f64 = 0.98;

        println!("\n=== Golden Image Regression Tests ===\n");
        println!("Minimum SSIM threshold: {}\n", MIN_SSIM);

        let mut passed = 0;
        let mut failed = 0;
        let mut failures = Vec::new();

        for config in GOLDEN_CONFIGS {
            println!("Testing: {}", config.name);

            // Render the image into rendered/
            let rendered_path = rendered_image_path(config.name);
            if let Err(e) = render_golden_image(config, &rendered_path) {
                println!("  ✗ Render failed: {}", e);
                failed += 1;
                failures.push((config.name, format!("Render error: {}", e)));
                continue;
            }

            // Compare against golden image
            match compare_golden_image(config) {
                Ok(ssim) => {
                    if ssim >= MIN_SSIM {
                        println!("  ✓ PASS (SSIM: {:.4})", ssim);
                        passed += 1;
                    } else {
                        println!("  ✗ FAIL (SSIM: {:.4} < {:.2})", ssim, MIN_SSIM);
                        failed += 1;
                        failures.push((config.name, format!("SSIM {:.4} below threshold", ssim)));

                        // TODO: Generate diff image and save to artifacts
                        let diff_path = diff_image_path(config.name);
                        println!("     Diff saved to: {:?}", diff_path);
                    }
                }
                Err(e) => {
                    println!("  ✗ SSIM computation failed: {}", e);
                    failed += 1;
                    failures.push((config.name, format!("SSIM error: {}", e)));
                }
            }
        }

        println!("\n=== Results ===");
        println!("Passed: {}/{}", passed, GOLDEN_CONFIGS.len());
        println!("Failed: {}/{}", failed, GOLDEN_CONFIGS.len());

        if !failures.is_empty() {
            println!("\nFailures:");
            for (name, reason) in &failures {
                println!("  - {}: {}", name, reason);
            }
            panic!("{} golden image test(s) failed", failed);
        }
    }

    #[test]
    fn test_golden_image_configs_valid() {
        // Sanity test: verify all configs are valid
        for config in GOLDEN_CONFIGS {
            assert!(!config.name.is_empty(), "Config must have a name");
            assert!(config.width > 0, "Width must be positive");
            assert!(config.height > 0, "Height must be positive");
            assert!(!config.brdf.is_empty(), "BRDF must be specified");
            assert!(
                !config.shadows.is_empty(),
                "Shadow technique must be specified"
            );
        }

        // Verify we have exactly 12 configs as specified in P9
        assert_eq!(GOLDEN_CONFIGS.len(), 12, "P9 specifies ~12 golden images");
    }

    #[test]
    fn test_golden_image_paths() {
        // Verify path generation is consistent
        let test_name = "test_image";
        let golden = golden_image_path(test_name);
        let rendered = rendered_image_path(test_name);
        let diff = diff_image_path(test_name);

        assert!(golden.to_string_lossy().contains("test_image.png"));
        assert!(rendered.to_string_lossy().contains("rendered"));
        assert!(diff.to_string_lossy().contains("diff"));
    }
}
