// examples/p5_ssr_glossy.rs
// Non-interactive P5 harness that renders the SSR glossy spheres baseline (no SSR tracing yet).
// Usage:
//   cargo run --release --example p5_ssr_glossy

use forge3d::p5::{meta, ssr};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let out_dir = Path::new(meta::DEFAULT_REPORT_DIR);
    std::fs::create_dir_all(out_dir)?;

    let preset = ssr::SsrScenePreset::load_or_default(ssr::default_scene_path());
    let preset = preset?;

    let png_path = out_dir.join(ssr::DEFAULT_OUTPUT_NAME);
    ssr::write_glossy_png(&preset, &png_path)?;
    println!("[P5] Wrote {}", png_path.display());

    meta::write_p5_meta(out_dir, |_meta| {
        // M0: no SSR metrics yet – defaults are inserted by the meta helper.
    })?;

    println!("[P5] SSR glossy spheres harness complete");
    Ok(())
}
