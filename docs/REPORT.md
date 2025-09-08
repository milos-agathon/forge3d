 # forge3d Repository Audit (READ-ONLY)                                                              
                                                                                                      
  ## Summary                                                                                          
                                                                                                      
  - Python shadows module defines get_preset_config twice; later definition overrides earlier (python/
  forge3d/shadows.py:24,518).                                                                         
  - Host-visible GPU memory tracking exists (512 MiB cap) but lacks drop-time decrements for          
  long-lived allocations; risk of stale metrics (src/core/memory_tracker.rs).                         
  - Py package ships compiled artifacts in-tree (python/forge3d/_forge3d.pyd, _vulkan_forge.pyd);     
  these are build outputs that should not be versioned.                                               
  - WGSL PBR shader implements GGX/Smith/Fresnel with IBL hooks; layouts appear consistent, but lacks 
  explicit bind group layout docs; potential mismatch risks.                                          
  - Rust Renderer readback staging can realloc based on size; budget checks present, but error paths  
  use expect strings and synchronous device polling (src/lib.rs).                                     
  - Python pbr CPU BRDF implements metallic/roughness workflow; evaluation relies on perceptual gain; 
  needs clear doc on intended parity with WGSL path (python/forge3d/pbr.py:200–306).                  
  - Packaging via maturin (PyO3 abi3-py38) looks sane; type stubs (__init__.pyi, py.typed) present and
  mostly aligned with public API.                                                                     
  - Tests are extensive and GPU-aware; several modules intentionally emit runtime warnings in         
  pure-Python fallback modes; indicates pending native integration.                                   
  - No top-level CMake configuration found; build relies on Cargo + maturin despite target note       
  referencing CMake ≥ 3.24.                                                                           
  - Shader and pipeline files are modular but lack Sphinx API doc references linking Python API to    
  Rust/WGSL internals.                                                                                
                                                                                                      
  ## Risk Matrix                                                                                      
                                                                                                      
  | Item | Area | Severity (P1–P3) | Likelihood | Impact | Rationale |                                
  | --- | --- | --- | --- | --- | --- |                                                               
  | Duplicate function in shadows.py | Python | P2 | High | Medium | Two get_preset_config definitions
  (lines 24, 518) cause silent override and confusion. |                                              
  | Host-visible memory accounting | Rust | P2 | Medium | High | No Drop path to free tracked         
  allocations at shutdown; metrics can drift. |                                                       
  | Compiled artifacts in repo | Packaging | P2 | High | Medium | _forge3d.pyd/_vulkan_forge.pyd      
  tracked; platform portability and wheels compromised. |                                             
  | Missing CMake | Build | P3 | High | Low | Mismatch with stated build goals; not strictly needed   
  given maturin, but note for cross-build. |                                                          
  | Shader layout documentation | WGSL | P2 | Medium | Medium | Bind groups exist but not documented; 
  future changes risk mismatch with Rust side. |                                                      
  | Synchronous device polling | Rust | P3 | Medium | Medium | render readback uses Maintain::Wait; OK
  for headless tests; perf impact noted. |                                                            
  | Fallback-only warnings | Python | P3 | High | Low | Bundles/shadows fallback indicates features   
  not wired to native; clarity issue. |                                                               
  | pbr CPU vs WGSL parity | API/SIM | P2 | Medium | Medium | CPU BRDF diverges from WGSL (perceptual 
  gain); doc/test parity needed. |                                                                    
  | Input validation | Python | P2 | Medium | Medium | Many validations present; confirm all array    
  shape/dtype paths across modules. |                                                                 
  | Device feature flags | Rust | P3 | Medium | Medium | Several feature-gated modules; CI guidance   
  needed per-platform. |                                                                              
                                                                                                      
  ## Repository Inventory                                                                             
                                                                                                      
  - Top-level                                                                                         
      - src/: Rust crate for core renderer, pipelines, terrain, vector; WGSL shaders under src/       
  shaders/.                                                                                           
      - python/forge3d/: Python API/interop layer (PyO3 binding _forge3d), typed package (py.typed,   
  stubs).                                                                                             
      - examples/: Python + Rust perf examples.                                                       
      - tests/: PyTest suite (GPU-aware, extensive).                                                  
      - assets/, data/: Images, sample data.                                                          
      - bench/: Rust benchmarking tools and binaries.                                                 
      - Config: Cargo.toml, pyproject.toml, pytest.ini, rust-toolchain.toml.                          
      - Artifacts/caches: .pytest_cache/, target/, .benchmarks/, out/, diag_out/, reports/, .venv/.   
      - Artifacts/caches: .pytest_cache/, target/, .benchmarks/, out/, diag_out/, reports/, .venv/.   
  -                                                                                                   
  src (depth ~2–3)                                                                                    
      - lib.rs: PyO3 Renderer, offscreen pipeline, readback, memory tracking hooks.                   
      - error.rs: RenderError types.                                                                  
      - gpu.rs, context.rs: Device/queue/context helpers.                                             
      - core/: core modules (framegraph, HDR, PBR, shadows, memory_tracker.rs, async_compute, etc).   
      - pipeline/: WGSL-based pipelines (pbr, normal_mapping, hdr_offscreen).                         
      - terrain/: terrain tiling, LOD, pipeline.                                                      
      - vector/: vector layers (points/lines/polygons), OIT, batching, indirect.                      
      - shaders/: WGSL (pbr.wgsl, shadows.wgsl, terrain*.wgsl, tonemap, etc), include and perf.       
  -                                                                                                   
  python/forge3d                                                                                      
      - __init__.py / __init__.pyi: public API + stubs; extension loading stubs.                      
      - pbr.py: PBR materials + CPU BRDF.                                                             
      - shadows.py: CSM configs, ShadowRenderer, duplicate get_preset_config definitions.             
      - bundles.py: render bundles (fallback warnings).                                               
      - hdr.py, envmap.py, mesh.py, normalmap.py, texture.py, vector.py, materials.py, _validate.py.  
      - Compiled binaries present: _forge3d.pyd, _vulkan_forge.pyd (should not be tracked).           
  -                                                                                                   
  Duplicates/orphans/dead code                                                                        
      - Duplicate: python/forge3d/shadows.py defines def get_preset_config(...) twice (lines 24 and   
  518); second overrides first.                                                                       
      - Potential orphan: python/forge3d/materials.py not evidently imported from __init__.py or      
  referenced in tests; verify usage.                                                                  
      - Build artifacts: python/forge3d/_forge3d.pyd, python/forge3d/_vulkan_forge.pyd, out/,         
  diag_out/, .pytest_cache/, target/ — safe to delete/rebuild.                                        
      - CLAUDE.md files in src/shaders and repo root: likely documentation/developer notes; not       
  consumed by build.                                                                                  
                                                                                                      
  ## Architecture Review                                                                              
                                                                                                      
  - Layering (textual):                                                                               
      - WGSL shaders ← Rust core/pipeline modules (wgpu) ← PyO3 bindings (src/lib.rs) ← Python API    
  modules (python/forge3d) ← Examples/Tests.                                                          
  - Boundaries:                                                                                       
      - Rust core::memory_tracker provides global singleton memory metrics (src/core/                 
  memory_tracker.rs).                                                                                 
      - Renderer in src/lib.rs encapsulates offscreen render path and readback.                       
      - Python modules add higher-level constructs (PBR CPU eval, shadows placeholder, bundles        
  fallback).                                                                                          
  - Interdependencies:                                                                                
      - Python __init__.py attempts to import extension functions; falls back to stubs if missing.    
      - python/forge3d/pbr.py is currently CPU-only; not wired to WGSL PBR path.                      
      - python/forge3d/shadows.py uses _core (forge3d._core) guarded; falls back.                     
  - Misplaced logic:                                                                                  
      - Duplicate function definition in shadows (python/forge3d/shadows.py:24,518).                  
      - CPU PBR eval diverges from shader (perceptual gain not in WGSL); requires parity strategy.    
  - RAII/ownership:                                                                                   
      - Rust relies on wgpu RAII; explicit tracking uses atomic counters; no explicit Drop to         
  reconcile counts on drop (risk).                                                                    
  - Unsafe:                                                                                           
      - No explicit unsafe blocks observed in sampled files; standard wgpu usage.                     
                                                                                                      
  ## Language-Specific Review                                                                         
                                                                                                      
  ### Rust                                                                                            
                                                                                                      
  - src/lib.rs                                                                                        
      - Copy/readback path uses device.poll(wgpu::Maintain::Wait) twice and map_async sync wait; ok   
  for tests, could block in interactive flows (lines ~216–236).                                       
      - .expect("map_async channel closed"), .expect("MapAsync failed") (readback): consider robust   
  error propagation via RenderError (file: src/lib.rs; see map_async section, around readback slice   
  and channel).                                                                                       
      - Memory tracking increments on create; decrements on reallocation; but not on Drop for all     
  resources (textures/buffers) — consider implementing Drop to update registry (src/lib.rs Renderer   
  struct).                                                                                            
  - src/core/memory_tracker.rs                                                                        
      - Budget set to 512 MiB (line 18); host-visible detection checks MAP_READ/WRITE only; good      
  (lines ~100–120).                                                                                   
      - calculate_texture_size uses conservative default for unknown formats; fine but can skew       
  metrics (lines ~129–144); consider expanding formats used in repo.                                  
  - Error handling:                                                                                   
      - error.rs uses thiserror? (open to confirm specifics); Recommend consistent Result<_,          
  RenderError> returns at FFI boundary.                                                               
  - Feature flags:                                                                                    
      - Cargo features enable-* present; ensure conditional exports match Python stubs.               
                                                                                                      
  ### Python                                                                                          
                                                                                                      
  - python/forge3d/shadows.py                                                                         
      - Duplicate get_preset_config: line 24 (name: str) and line 518 (quality: str) — unify and      
  de-duplicate.                                                                                       
      - Fallback warnings: CSM shadows running in pure-Python fallback mode and ShadowRenderer running
  in pure-Python fallback mode; good for clarity; add docstrings indicating parity vs native.         
  - python/forge3d/pbr.py                                                                             
      - class PbrRenderer at line 200; def evaluate_brdf at line 217; applies gain = float(np.exp(7.0 
  * metallic)) at line 294; document rationale and ensure CPU vs WGSL consistency policy.             
      - Validation helper validate_pbr_material checks ranges and provides stats; good; consider      
  returning dataclass or TypedDict for stronger typing.                                               
  - Packaging                                                                                         
      - pyproject.toml sets bindings = "pyo3", module-name = "forge3d._forge3d", ABI3 ok (Python      
  ≥ 3.8).                                                                                             
      - Type stubs present and useful; verify alignment with public all export list.                  
  - Artifacts                                                                                         
      - _forge3d.pyd and _vulkan_forge.pyd checked in; should be excluded via .gitignore and          
  distributed via wheels.                                                                             
                                                                                                      
  ### WGSL                                                                                            
                                                                                                      
  - src/shaders/pbr.wgsl                                                                              
      - Functions present: distribution_ggx (line ~133), geometry_smith (~145), fresnel_schlick       
  (~155), fresnel_schlick_roughness (~160), fs_main (~195), etc.                                      
      - Material texture channels: metallic from blue, roughness from green (consistent with Python). 
      - Suggest adding bind group layout summary and expected texture/sampler formats in comments at  
  file head.                                                                                          
  - src/shaders/shadows.wgsl                                                                          
      - Not inspected in-depth; ensure PCF kernel parameterization matches Python presets; document   
  cascade indexing conventions.                                                                       
                                                                                                      
  ### CMake                                                                                           
                                                                                                      
  - No root CMakeLists.txt present; build uses Cargo + maturin. If CMake is a target, consider        
  providing a thin CMake wrapper that invokes maturin/cargo and sets RPATHs/DLL copy on Windows.      
                                                                                                      
  ## Performance & Memory                                                                             
                                                                                                      
  - Hot path: offscreen render + readback (src/lib.rs). Synchronous map + poll; acceptable for tests; 
  for interactive flows, consider async staging and ring buffers.                                     
  - Allocation churn: resizing readback buffer when need > readback_size; good; ensure upper bound    
  documented (e.g., cap size growth).                                                                 
  - Zero-copy boundaries:                                                                             
      - Python to Rust heightmap: converts NumPy to Vec; validates C-contiguous (good), but not       
  zero-copy into GPU; acceptable due to GPU upload requirement; document behavior.                    
  - GPU budget:                                                                                       
      - Host-visible tracked; device-local textures not counted toward host-visible cap (by design).  
  Validate total (host + device) stays reasonable for high-res offscreen.                             
  - Shadows memory:                                                                                   
      - Ultra preset: ensure cascade_count × size^2 × 4 B < 256 MB (e.g., 3×4096^2×4 = 201 MB).       
                                                                                                      
  ## Security & Robustness                                                                            
                                                                                                      
  - Input validation:                                                                                 
      - Python validates shapes/dtypes and contiguity in FFI-facing paths (src/lib.rs and python      
  bridge).                                                                                            
  - Error handling:                                                                                   
      - Replace .expect in readback with proper RenderError paths to avoid panics.                    
  - Shader failure handling:                                                                          
      - Current compiled-in shaders via include_str!; runtime compile errors minimal; consider        
  compile-time validation tool in CI.                                                                 
  - File IO:                                                                                          
      - PNG save returns IO error via RenderError; good.                                              
                                                                                                      
  ## Documentation                                                                                    
                                                                                                      
  - README present; change log exists (CHANGELOG.md).                                                 
  - Sphinx docs dir present (docs/) but content not inspected here; ensure API reference includes     
  Python modules and public Rust functions via docstrings and intersphinx.                            
  - Add docs explaining GPU memory budget and how to interpret metrics from memory_tracker.           
                                                                                                      
  ## Tech Debt & Gaps                                                                                 
                                                                                                      
  - Duplicate function in shadows.py (dedupe, add tests).                                             
  - CPU PBR eval vs WGSL parity not guaranteed; add parity tests or doc justification.                
  - No Drop-based registry decrements; implement to avoid stale metrics.                              
  - Missing CMake wrapper vs stated target; provide optional CMake meta-build if required.            
  - Shader bind group layout doc gaps; add summaries for future maintainability.                      
  - Compiled artifacts in repo; update .gitignore and release process.                                
  - Fallback-only modules (bundles/shadows) need native integration alignment notes.                  
  - Expand texture size accounting for other formats (R16F, BCn if used later).                       
  - Centralize error messages (avoid expect) for robust FFI error reporting.                          
                                                                                                      
  ## Parity with rayshader                                                                            
                                                                                                      
  - Missing: full terrain shading stack with hillshade + ambient occlusion compositing; contour       
  extraction GPU path; flexible colormap stack; camera + tone mapping presets; layered vector overlays
  with proper OIT sorting; PBR IBL lighting pipeline integrated.                                      
  - Minimal viable path: integrate WGSL PBR pipeline via Python API, add terrain AO pass, expand      
  colormap assets, wire shadows renderer to native path, add vector overlay compositing over terrain, 
  provide camera presets and tonemap operators.                                                       
                                                                                                      
  ## 10 Advanced Examples                                                                             
                                                                                                      
  - Terrain + Shadows + PBR Materials: Render DEM with CSM and metallic/dielectric materials atop.    
  - Contour Extraction and Overlay: Compute contours from DEM and overlay with anti-aliased vector    
  paths.                                                                                              
  - HDR Offscreen Tone Mapping: Render wide-gamut scene and compare Reinhard vs ACES tone mapping.    
  - Vector OIT Layering: Dense line/point overlays with weighted OIT and performance counters.        
  - Normal Mapping on Terrain: Apply detail normal maps on DEM mesh with TBN validation.              
  - IBL Environment Lighting: Showcase prefiltered envmap and BRDF LUT with rotating HDR maps.        
  - Multi-Threaded Command Recording: Stress test framegraph with parallel pass recording.            
  - Async Compute Prepass: Run compute culling + LOD selection concurrently with graphics.            
  - Large Texture Upload Policies: Compare split vs single bind group strategies (examples/perf).     
  - Device Capability Probe: Enumerate adapters, present caps matrix, and choose optimal backend.     
                                                                                                      
  ## Open Questions / Missing Evidence                                                                
                                                                                                      
  - What is the intended role of python/forge3d/materials.py? Is it used?                             
      - Command: rg -n "forge3d.materials|from forge3d import materials|materials\." python           
  - Are there Drop impls freeing tracked allocations across modules?                                  
      - Command: rg -n "impl Drop for|free_buffer_allocation|free_texture_allocation" src             
  - What bind group layouts and binding indices are used for each pipeline?                           
      - Command: rg -n "@group\(|@binding\(" src/shaders/*.wgsl                                       
  - What features are enabled in release artifacts?                                                   
      - Command: cargo tree -e features                                                               
  - Do Sphinx docs build and include API references?                                                  
      - Command: sphinx-build -b html docs _build/html                                                
  - What’s the wheel footprint and whether compiled binaries are excluded?                            
      - Command: maturin build --release && python -m pip install dist/*.whl && python -c "import     
  forge3d; print(forge3d.version)"                                                                    
  - What’s the current clippy/fmt status?                                                             
      - Command: cargo fmt -- --check && cargo clippy --all-targets --all-features -D warnings        
  - Do shader compile checks pass on all backends?                                                    
      - Command: RUST_LOG=info WGPU_BACKEND=vulkan cargo test -p forge3d -- tests that exercise       
  shaders                                                                                             
  - What is the full memory metrics path in a large render?                                           
      - Command: python -c "import forge3d as f; r=f.Renderer(4096,4096); print('ok')"                
  - Any CI build matrix?                                                                              
      - Command: rg -n "workflow|GitHub Actions|CI" -S                                                
                                                                                                      
  ## Appendix: Evidence                                                                               
                                                                                                      
  - Duplicate shadows preset function                                                                 
      - python/forge3d/shadows.py:24 — def get_preset_config(name: str) -> 'CsmConfig'                
      - python/forge3d/shadows.py:518 — def get_preset_config(quality: str) -> CsmConfig              
  - CPU PBR evaluate_brdf with gain                                                                   
      - python/forge3d/pbr.py:200 — class PbrRenderer:                                                
      - python/forge3d/pbr.py:217 — def evaluate_brdf(…                                               
      - python/forge3d/pbr.py:294 — gain = float(np.exp(7.0 * metallic))                              
  - Memory budget set to 512 MiB                                                                      
      - src/core/memory_tracker.rs:18 — budget_limit: 512 * 1024 * 1024                               
  - WGSL PBR functions                                                                                
      - src/shaders/pbr.wgsl:133 — fn distribution_ggx…                                               
      - src/shaders/pbr.wgsl:145 — fn geometry_smith…                                                 
      - src/shaders/pbr.wgsl:155 — fn fresnel_schlick…                                                
      - src/shaders/pbr.wgsl:195 — @fragment fn fs_main…  