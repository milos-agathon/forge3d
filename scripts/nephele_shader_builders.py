"""Execute the renderer's pure Rust shader builders for static Naga evidence.

No GPU objects are constructed. Function bodies and includes come from the
current tree; substitutions only provide their original module/type context.
Both values of the terrain/HZB boolean inputs are emitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile


def execute_builders(root: Path, balanced, assignment) -> dict[str, list[str]]:
    def read(path):
        return (root / path).read_text(encoding="utf-8")

    def absolute_includes(text, path):
        return re.sub(r'include_str!\(\s*"([^"]+)"\s*\)',
                      lambda m: 'include_str!(' + json.dumps((root / path).parent.joinpath(m[1]).resolve().as_posix()) + ')', text)

    def function(path, name):
        text = read(path)
        matches = list(re.finditer(r'\b(?:pub(?:\([^)]*\))?\s+)?fn\s+' + re.escape(name) + r'\s*\(', text))
        if len(matches) != 1:
            raise ValueError(f"expected one pure builder {path}::{name}")
        start = matches[0].start()
        _, after = balanced(text, text.index('(', matches[0].start()))
        opening = text.index('{', after)
        _, end = balanced(text, opening, '{', '}')
        return absolute_includes(text[start:end], path)

    ssao_path = 'src/core/screen_space_effects/ssao/constructor.rs'
    configure = function(ssao_path, 'configure_concordia_gtao')
    filter_prefix = configure[configure.index('let filter_source ='):configure.index('let filter =')]
    ssao = '\n'.join(function(ssao_path, n) for n in ('shared_contacts_source', 'concordia_gtao_source'))
    reconstruction = function('src/core/screen_space_effects.rs', 'concordia_reconstruction')
    hzb = function('src/core/screen_space_effects/hzb.rs', 'linear_depth_source')
    taa = function('src/core/taa.rs', 'shared_geometry_source')
    point = function('src/vector/point/renderer/shared.rs', 'shared_depth_source')
    overlay = function('src/core/overlays.rs', 'shared_depth_source')
    def constant(path, name):
        source=read(path)
        value=assignment(source,name,len(source))
        if value is None: raise ValueError(f'missing builder constant {path}::{name}')
        return absolute_includes(value,path)
    def included(path):
        return 'include_str!(' + json.dumps((root / path).as_posix()) + ')'

    # Keep module nesting so Self/super/crate references execute unchanged.
    harness = '''#![allow(dead_code, unexpected_cfgs)]
#[path = SHADER_SOURCES] mod shader_sources;
#[path = INTERACTIVE] mod interactive;
mod core {
  pub mod error {
    #[derive(Debug)] pub enum RenderError { Render(String) }
    pub type RenderResult<T> = Result<T, RenderError>;
  }
  pub mod screen_space_effects {
    const GTAO_SHADER_SRC: &str = GTAO;
    RECONSTRUCTION
    pub mod hzb { const HZB_BUILD_SOURCE: &str = HZB; HZB_FUNCTION }
    pub mod ssao { pub mod constructor {
      use super::super::*; use crate::core::error::RenderResult;
      pub struct SsaoRenderer;
      impl SsaoRenderer { SSAO_FUNCTIONS }
      pub fn filter_source() -> RenderResult<String> { FILTER_PREFIX Ok(filter_source) }
    }}
  }
  pub mod taa {
    use super::error::RenderResult;
    const TAA_SHADER_SRC: &str = TAA;
    pub struct TaaRenderer; impl TaaRenderer { TAA_FUNCTION }
  }
  pub mod overlays { use super::error::RenderResult; OVERLAY_FUNCTION }
}
mod point {
  use crate::core::error::RenderError;
  pub struct PointRenderer; impl PointRenderer { POINT_FUNCTION }
}
fn main() {
  let out = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
  let records = [
    ("terrain-false", interactive::terrain_source(false).unwrap()),
    ("terrain-true", interactive::terrain_source(true).unwrap()),
    ("scatter", interactive::scatter_source().unwrap()),
    ("atmosphere", interactive::atmosphere_source().unwrap()),
    ("height-ao", interactive::height_ao_source().unwrap()),
    ("froxels", interactive::froxel_source().unwrap()),
    ("hzb-linear", core::screen_space_effects::hzb::linear_depth_source()),
    ("hzb-legacy", HZB.to_owned()),
    ("contacts", core::screen_space_effects::ssao::constructor::SsaoRenderer::shared_contacts_source().unwrap()),
    ("gtao", core::screen_space_effects::ssao::constructor::SsaoRenderer::concordia_gtao_source().unwrap()),
    ("ao-filter", core::screen_space_effects::ssao::constructor::filter_source().unwrap()),
    ("taa", core::taa::TaaRenderer::shared_geometry_source().unwrap()),
    ("points", point::PointRenderer::shared_depth_source().unwrap()),
    ("fog-legacy", FOG.to_owned()),
    ("overlays", core::overlays::shared_depth_source().unwrap()),
    ("overlays-legacy", OVERLAY.to_owned()),
  ];
  for (name, source) in records { std::fs::write(out.join(format!("{name}.wgsl")), source).unwrap(); }
}
'''
    substitutions = dict(SHADER_SOURCES=json.dumps((root/'src/shader_sources.rs').as_posix()),
                         INTERACTIVE=json.dumps((root/'src/terrain/renderer/interactive/shaders.rs').as_posix()),
                         RECONSTRUCTION=reconstruction, HZB_FUNCTION=hzb, SSAO_FUNCTIONS=ssao,
                         FILTER_PREFIX=filter_prefix, TAA_FUNCTION=taa, POINT_FUNCTION=point, OVERLAY_FUNCTION=overlay,
                         GTAO=constant('src/core/screen_space_effects.rs','GTAO_SHADER_SRC'),
                         HZB=constant('src/core/screen_space_effects/hzb.rs','HZB_BUILD_SOURCE'),
                         TAA=constant('src/core/taa.rs','TAA_SHADER_SRC'), FOG=included('src/shaders/volumetric.wgsl'),
                         OVERLAY=included('src/shaders/overlays.wgsl'))
    harness = re.sub(r'\b('+'|'.join(substitutions)+r')\b', lambda m: substitutions[m[0]], harness)
    # Match the repository's already selected anyhow dependency exactly.
    lock = read('Cargo.lock')
    version = re.search(r'\[\[package\]\]\nname = "anyhow"\nversion = "([^"]+)"', lock)[1]
    with tempfile.TemporaryDirectory(prefix='nephele-builders-') as directory:
        temporary = Path(directory)
        (temporary/'src').mkdir()
        (temporary/'src/main.rs').write_text(harness, encoding='utf-8')
        (temporary/'Cargo.toml').write_text('[package]\nname="nephele_builder_sources"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nanyhow="='+version+'"\n', encoding='utf-8')
        completed = subprocess.run(['cargo','run','--offline','--quiet','--manifest-path',str(temporary/'Cargo.toml'),'--',str(temporary)], cwd=temporary, capture_output=True, text=True)
        if completed.returncode:
            raise ValueError('pure Rust shader builders failed: '+completed.stderr)
        sources={p.stem:p.read_text(encoding='utf-8') for p in temporary.glob('*.wgsl')}
    return {name:[sources[k] for k in keys] for name,keys in {
        'terrain_source': ['terrain-false','terrain-true'],
        'scatter_source':['scatter'],
        'atmosphere_source':['atmosphere'], 'height_ao_source':['height-ao'],
        'shared_contacts_source':['contacts'], 'concordia_gtao_source':['gtao'],
        'shared_geometry_source':['taa'], 'shared_depth_source':['points'],
        'hzb_sources':['hzb-linear','hzb-legacy'], 'filter_source':['ao-filter'],
        'fog_sources':['froxels','fog-legacy'],
        'overlay_sources':['overlays','overlays-legacy'],
    }.items()}


def builder_variants(expr, path, text, before, assignment, sources):
    """Resolve pure factory calls and their complete finite runtime variants."""
    bare=expr.strip().lstrip('&').strip()
    if re.fullmatch(r'[A-Za-z_]\w*',bare):
        value=assignment(text,bare,before)
        if value is not None: bare=value.strip()
    factory=re.fullmatch(r'(?:[A-Za-z_]\w*::)*(terrain_source|scatter_source|atmosphere_source|height_ao_source|shared_contacts_source|concordia_gtao_source|shared_geometry_source|shared_depth_source)\s*\((.*)\)\s*\??',bare,re.S)
    if factory:
        owners = {
            'terrain_source':'src/terrain/renderer/pipeline_cache.rs',
            'scatter_source':'src/render/mesh_instanced.rs',
            'atmosphere_source':'src/terrain/renderer/interactive.rs',
            'height_ao_source':'src/terrain/renderer/interactive.rs',
            'shared_contacts_source':'src/core/screen_space_effects/ssao/constructor.rs',
            'concordia_gtao_source':'src/core/screen_space_effects/ssao/constructor.rs',
            'shared_geometry_source':'src/core/taa.rs',
            'shared_depth_source':'src/vector/point/renderer/shared.rs',
        }
        if not path.as_posix().endswith('/'+owners[factory[1]]): return []
        # Only terrain_source accepts an input, and both boolean values run.
        if factory[1]!='terrain_source' and factory[2].strip(): return []
        return sources[factory[1]]
    if path.as_posix().endswith('/src/core/screen_space_effects/hzb.rs') and re.fullmatch(r'if\s+linear\s*\{\s*linear_depth_source\(\)\s*\}\s*else\s*\{\s*HZB_BUILD_SOURCE\.to_owned\(\)\s*\}',bare,re.S):
        return sources['hzb_sources']
    if path.as_posix().endswith('/src/core/overlays.rs') and re.fullmatch(r'if\s+shared_depth\s*\{\s*shared_depth_source\(\)\?\s*\}\s*else\s*\{\s*include_str!\("../shaders/overlays.wgsl"\)\.to_owned\(\)\s*\}',bare,re.S):
        return sources['overlay_sources']
    if path.as_posix().endswith('/src/core/screen_space_effects/ssao/constructor.rs') and re.fullmatch(r'filter_source\.replace\(anchor,\s*"let result = clamp\(sum / weight_sum, 0.0, 1.0\);"\)',bare):
        return sources['filter_source']
    if path.as_posix().endswith('/src/viewer/init/fog_init.rs') and re.fullmatch(r'shared\s*\.map\(\|value\|\s*value\.0\)\s*\.unwrap_or\(include_str!\("../../shaders/volumetric.wgsl"\)\)',bare,re.S):
        return sources['fog_sources']
    return []
