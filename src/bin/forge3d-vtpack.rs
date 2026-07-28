//! Offline TESSELLA virtual-texture packer.

use forge3d::terrain::vt::{
    write_packed_store, PackedPage, PageBytes, PageFormat, PageKey, StoreMetadata,
};
use std::path::{Path, PathBuf};

const HELP: &str = "\
forge3d-vtpack --output STORE [--manifest JSON]
  [--albedo IMAGE --normal IMAGE --mask IMAGE]
  [--procedural --virtual-width N --virtual-height N --seed N]
  [--tile-size N --tile-border N]

Packs Morton-ordered, SHA-256-addressed BC pages. Albedo uses BC7-sRGB,
normal uses BC5-UNORM, and mask/roughness/metalness uses BC7-UNORM.
--procedural writes a sparse deterministic descriptor; requested pages are
generated independently without allocating the declared virtual image.
";

#[derive(Debug)]
struct Args {
    output: PathBuf,
    manifest: Option<PathBuf>,
    albedo: Option<PathBuf>,
    normal: Option<PathBuf>,
    mask: Option<PathBuf>,
    procedural: bool,
    virtual_width: u64,
    virtual_height: u64,
    tile_size: u32,
    tile_border: u32,
    seed: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("forge3d-vtpack: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args(std::env::args().skip(1).collect())? else {
        print!("{HELP}");
        return Ok(());
    };
    let (metadata, pages) = if args.procedural {
        let metadata = StoreMetadata {
            virtual_width: args.virtual_width,
            virtual_height: args.virtual_height,
            tile_size: args.tile_size,
            tile_border: args.tile_border,
            family_count: 3,
            procedural: true,
            procedural_seed: args.seed,
        };
        let pages = pack_procedural_pages(&metadata)?;
        (metadata, pages)
    } else {
        pack_images(&args)?
    };
    let manifest = write_packed_store(&args.output, &metadata, pages)?;
    let manifest_path = args
        .manifest
        .unwrap_or_else(|| args.output.with_extension("manifest.json"));
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("manifest serialization failed: {error}"))?;
    std::fs::write(&manifest_path, &json).map_err(|error| {
        format!(
            "failed to write manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    println!(
        "{}",
        String::from_utf8(json).expect("JSON serialization is UTF-8")
    );
    Ok(())
}

fn pack_procedural_pages(metadata: &StoreMetadata) -> Result<Vec<PackedPage>, String> {
    let side = metadata.slot_size();
    let texels = side as usize * side as usize;
    (0..metadata.family_count)
        .map(|family| {
            let family = family as u8;
            let bytes = match family {
                0 => {
                    let mut rgba = Vec::with_capacity(texels * 4);
                    for index in 0..texels {
                        let value = (index as u64)
                            .wrapping_mul(0x9e37_79b9)
                            .wrapping_add(metadata.procedural_seed)
                            as u8;
                        rgba.extend_from_slice(&[value, value.wrapping_add(31), 96, 255]);
                    }
                    PageBytes::new(
                        PageFormat::Bc7Srgb,
                        side,
                        side,
                        forge3d::core::compressed_textures::encode_bc7_rgba8(&rgba, side, side)?,
                    )?
                }
                1 => {
                    let rg = vec![128; texels * 2];
                    PageBytes::new(
                        PageFormat::Bc5Unorm,
                        side,
                        side,
                        forge3d::core::compressed_textures::encode_bc5_rg8(&rg, side, side)?,
                    )?
                }
                _ => {
                    let rgba = [192, 128, 32, 255].repeat(texels);
                    PageBytes::new(
                        PageFormat::Bc7Unorm,
                        side,
                        side,
                        forge3d::core::compressed_textures::encode_bc7_rgba8(&rgba, side, side)?,
                    )?
                }
            };
            Ok(PackedPage {
                key: PageKey {
                    family,
                    mip: 0,
                    x: 0,
                    y: 0,
                },
                bytes,
            })
        })
        .collect()
}

fn parse_args(raw: Vec<String>) -> Result<Option<Args>, String> {
    if raw.is_empty() || raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(None);
    }
    let value = |name: &str| -> Result<Option<String>, String> {
        let Some(index) = raw.iter().position(|arg| arg == name) else {
            return Ok(None);
        };
        raw.get(index + 1)
            .cloned()
            .ok_or_else(|| format!("{name} requires a value"))
            .map(Some)
    };
    let parse_u64 = |name: &str, default: u64| -> Result<u64, String> {
        value(name)?
            .map(|text| {
                text.parse::<u64>()
                    .map_err(|_| format!("{name} must be an integer"))
            })
            .transpose()
            .map(|parsed| parsed.unwrap_or(default))
    };
    let output = value("--output")?
        .map(PathBuf::from)
        .ok_or_else(|| "--output is required".to_string())?;
    let args = Args {
        output,
        manifest: value("--manifest")?.map(PathBuf::from),
        albedo: value("--albedo")?.map(PathBuf::from),
        normal: value("--normal")?.map(PathBuf::from),
        mask: value("--mask")?.map(PathBuf::from),
        procedural: raw.iter().any(|arg| arg == "--procedural"),
        virtual_width: parse_u64("--virtual-width", 1 << 18)?,
        virtual_height: parse_u64("--virtual-height", 1 << 18)?,
        tile_size: u32::try_from(parse_u64("--tile-size", 128)?)
            .map_err(|_| "--tile-size exceeds u32".to_string())?,
        tile_border: u32::try_from(parse_u64("--tile-border", 4)?)
            .map_err(|_| "--tile-border exceeds u32".to_string())?,
        seed: parse_u64("--seed", 19)?,
    };
    if args.procedural && (args.albedo.is_some() || args.normal.is_some() || args.mask.is_some()) {
        return Err("--procedural cannot be combined with image inputs".to_string());
    }
    Ok(Some(args))
}

fn pack_images(args: &Args) -> Result<(StoreMetadata, Vec<PackedPage>), String> {
    let paths = [
        args.albedo
            .as_deref()
            .ok_or_else(|| "--albedo is required outside --procedural mode".to_string())?,
        args.normal
            .as_deref()
            .ok_or_else(|| "--normal is required outside --procedural mode".to_string())?,
        args.mask
            .as_deref()
            .ok_or_else(|| "--mask is required outside --procedural mode".to_string())?,
    ];
    let images = paths
        .iter()
        .map(|path| load_rgba(path))
        .collect::<Result<Vec<_>, _>>()?;
    let (width, height) = (images[0].0, images[0].1);
    if images.iter().any(|(candidate_width, candidate_height, _)| {
        *candidate_width != width || *candidate_height != height
    }) {
        return Err("albedo, normal, and mask images must have identical dimensions".to_string());
    }
    let metadata = StoreMetadata {
        virtual_width: u64::from(width),
        virtual_height: u64::from(height),
        tile_size: args.tile_size,
        tile_border: args.tile_border,
        family_count: 3,
        procedural: false,
        procedural_seed: 0,
    };
    let mut pages = Vec::new();
    for (family, (_, _, base)) in images.into_iter().enumerate() {
        let mut mip_width = width;
        let mut mip_height = height;
        let mut mip_data = base;
        let mut mip = 0u8;
        loop {
            append_mip_pages(
                &mut pages,
                &metadata,
                family as u8,
                mip,
                mip_width,
                mip_height,
                &mip_data,
            )?;
            if mip_width == 1 && mip_height == 1 {
                break;
            }
            let next_width = mip_width.div_ceil(2);
            let next_height = mip_height.div_ceil(2);
            mip_data = downsample_rgba(&mip_data, mip_width, mip_height);
            mip_width = next_width;
            mip_height = next_height;
            mip = mip
                .checked_add(1)
                .ok_or_else(|| "mip count exceeds u8".to_string())?;
        }
    }
    Ok((metadata, pages))
}

fn load_rgba(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let image = image::open(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
        .to_rgba8();
    Ok((image.width(), image.height(), image.into_raw()))
}

fn append_mip_pages(
    pages: &mut Vec<PackedPage>,
    metadata: &StoreMetadata,
    family: u8,
    mip: u8,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), String> {
    let pages_x = width.div_ceil(metadata.tile_size);
    let pages_y = height.div_ceil(metadata.tile_size);
    for y in 0..pages_y {
        for x in 0..pages_x {
            let tile = bordered_tile(
                rgba,
                width,
                height,
                x,
                y,
                metadata.tile_size,
                metadata.tile_border,
            );
            let side = metadata.slot_size();
            let bytes = match family {
                0 => PageBytes::new(
                    PageFormat::Bc7Srgb,
                    side,
                    side,
                    forge3d::core::compressed_textures::encode_bc7_rgba8(&tile, side, side)?,
                )?,
                1 => {
                    let rg = tile
                        .chunks_exact(4)
                        .flat_map(|pixel| [pixel[0], pixel[1]])
                        .collect::<Vec<_>>();
                    PageBytes::new(
                        PageFormat::Bc5Unorm,
                        side,
                        side,
                        forge3d::core::compressed_textures::encode_bc5_rg8(&rg, side, side)?,
                    )?
                }
                _ => PageBytes::new(
                    PageFormat::Bc7Unorm,
                    side,
                    side,
                    forge3d::core::compressed_textures::encode_bc7_rgba8(&tile, side, side)?,
                )?,
            };
            pages.push(PackedPage {
                key: PageKey { family, mip, x, y },
                bytes,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bordered_tile(
    rgba: &[u8],
    width: u32,
    height: u32,
    page_x: u32,
    page_y: u32,
    tile_size: u32,
    border: u32,
) -> Vec<u8> {
    let side = tile_size + 2 * border;
    let mut tile = Vec::with_capacity(side as usize * side as usize * 4);
    for local_y in 0..side {
        for local_x in 0..side {
            let source_x = (i64::from(page_x * tile_size) + i64::from(local_x) - i64::from(border))
                .clamp(0, i64::from(width - 1)) as u32;
            let source_y = (i64::from(page_y * tile_size) + i64::from(local_y) - i64::from(border))
                .clamp(0, i64::from(height - 1)) as u32;
            let offset = ((source_y * width + source_x) * 4) as usize;
            tile.extend_from_slice(&rgba[offset..offset + 4]);
        }
    }
    tile
}

fn downsample_rgba(source: &[u8], width: u32, height: u32) -> Vec<u8> {
    let next_width = width.div_ceil(2);
    let next_height = height.div_ceil(2);
    let mut next = Vec::with_capacity(next_width as usize * next_height as usize * 4);
    for y in 0..next_height {
        for x in 0..next_width {
            for channel in 0..4 {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(width - 1);
                        let sy = (y * 2 + dy).min(height - 1);
                        sum += u32::from(source[((sy * width + sx) * 4 + channel) as usize]);
                        count += 1;
                    }
                }
                next.push(((sum + count / 2) / count) as u8);
            }
        }
    }
    next
}
