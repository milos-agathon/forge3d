- Is python/forge3d/materials.py used anywhere or should it be merged into pbr.py?                  
      - Command: rg -n "forge3d.materials|from forge3d import materials|materials\." python           
  - Should CPU PBR BRDF match WGSL exactly, or is the current perceptual gain an intentional          
  divergence?                                                                                         
      - Command: rg -n "@fragment|fs_main|fresnel_schlick" src/shaders/pbr.wgsl                       
  - Do we want a minimal CMake wrapper to satisfy environments requiring CMake-based builds?          
      - Command: cmake -LAH -N .                                                                      
  - What is the expected behavior when map_async fails in headless readback? Should we return a Python
  exception instead of panicking?                                                                     
      - Command: rg -n "map_async|Maintain::Wait|expect\(" src/lib.rs                                 
  - Are there Drop implementations elsewhere decrementing memory tracker counters, or should we add   
  one to Renderer?                                                                                    
      - Command: rg -n "impl Drop for|free_buffer_allocation|free_texture_allocation" src             
  - Should compiled artifacts _forge3d.pyd and _vulkan_forge.pyd be ignored and removed from VCS?     
      - Command: git ls-files python/forge3d/*.pyd                                                    
  - Do we plan to integrate native shadows/bundles soon, or keep pure-Python fallbacks with clearer   
  documentation?                                                                                      
      - Command: rg -n "warnings\.warn\(|fallback mode" python/forge3d/*.py                           
  - What backends/adapters are targeted in CI (Vulkan/Metal/DX12), and do we enforce deterministic    
  tests per backend?                                                                                  
      - Command: pytest -q                                                                            
  - Are there additional texture formats we plan to use that should be added to                       
  calculate_texture_size?                                                                             
      - Command: rg -n "TextureFormat::" src | sort                                                   
  - Should Sphinx build include auto-generated Python API docs and a “GPU setup” guide?               
      - Command: sphinx-build -b html docs _build/html      