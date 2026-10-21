// Deterministic view-frustum froxels. Rust FroxelDepthTransform uses the same
// distance(u)=near*exp(log(far/near)*u) mapping.
struct MediaUniforms { view_proj:mat4x4<f32>, inv_view_proj:mat4x4<f32>, previous_view_proj:mat4x4<f32>, view:mat4x4<f32>, camera:vec4<f32>, sun:vec4<f32>, sun_radiance:vec4<f32>, sigma_s:vec4<f32>, sigma_t:vec4<f32>, depth:vec4<f32>, grid:vec4<u32>, viewport:vec4<u32>, scattering:vec4<f32>, terrain_albedo:vec4<f32>, diffuse_ibl:vec4<f32>, world_min:vec4<f32>, world_max:vec4<f32> }
struct NepheleTerrainHit { point_t:vec4<f32>, normal_hit:vec4<f32> }
// Prefix-exact mirror of crate::shadows::CsmUniforms. The trailing fields are
// intentionally omitted because this pass reads only the prefix below.
struct Cascade { light_projection:mat4x4<f32>, light_view_proj:mat4x4<f32>, near_distance:f32, far_distance:f32, texel_size:f32, _pad:f32 }
struct Csm { light_direction:vec4<f32>, light_view:mat4x4<f32>, cascades:array<Cascade,4>, cascade_count:u32, pcf_kernel_size:u32, depth_bias:f32, slope_bias:f32 }
@group(0) @binding(0) var<uniform> media:MediaUniforms;
// Uploaded exclusively through crate::media::Medium::extinction_at.
@group(0) @binding(1) var canonical_extinction:texture_3d<f32>;
@group(0) @binding(2) var blue_noise_ranks:texture_2d<u32>;
@group(0) @binding(3) var terrain_shadow_maps:texture_depth_2d_array;
@group(0) @binding(4) var<uniform> csm:Csm;
// CPU-integrated through the complete canonical density bounds, so terrain-to-
// sun attenuation remains correct when the segment leaves the camera frustum.
@group(0) @binding(5) var canonical_light_transmittance:texture_3d<f32>;
// Narrow renderer-prepared RadianceProvider seam: analytic sky and AETHER
// both arrive as their actually rendered directional radiance texture.
@group(0) @binding(6) var prepared_sky_radiance:texture_2d<f32>;
@group(0) @binding(7) var<storage,read> sun_terrain_hits:array<NepheleTerrainHit>;
@group(0) @binding(8) var<storage,read> phase_terrain_hits:array<NepheleTerrainHit>;
// Full bounded-medium fields shared with the terrain cloud-light prepass.
@group(0) @binding(9) var world_medium:texture_3d<f32>;
@group(0) @binding(10) var world_sun_source:texture_3d<f32>;
@group(1) @binding(0) var froxel_single_scatter:texture_storage_3d<rgba16float,write>;
@group(1) @binding(1) var froxel_in_scatter:texture_storage_3d<rgba16float,write>;
@group(1) @binding(2) var froxel_sun_source_out:texture_storage_3d<rgba16float,write>;
@group(2) @binding(0) var froxel_extinction:texture_3d<f32>;
@group(2) @binding(1) var froxel_scatter:texture_3d<f32>;
// Authoritative WebGPU depth. World distance is reconstructed through the
// inverse view-projection before the shared logarithmic froxel transform.
@group(2) @binding(2) var scene_depth:texture_depth_2d;
@group(2) @binding(3) var history_scatter:texture_2d<f32>;
@group(2) @binding(4) var history_depth:texture_depth_2d;
@group(2) @binding(5) var out_transmittance:texture_storage_2d<rgba16float,write>;
@group(2) @binding(6) var out_in_scatter:texture_storage_2d<rgba16float,write>;
@group(2) @binding(7) var out_cloud_shadow:texture_storage_2d<rgba16float,write>;
@group(2) @binding(8) var out_optical_depth:texture_storage_2d<rgba16float,write>;
@group(2) @binding(9) var froxel_single_source:texture_3d<f32>;
@group(2) @binding(10) var froxel_sun_source:texture_3d<f32>;
@group(3) @binding(0) var terrain_linear_hdr:texture_2d<f32>;
@group(3) @binding(1) var media_integrated_hdr:texture_2d<f32>;
@group(3) @binding(2) var media_transmittance:texture_2d<f32>;
@group(3) @binding(3) var composite_linear_hdr:texture_storage_2d<rgba16float,write>;
fn froxel_distance(u:f32)->f32{return media.depth.x*exp(media.depth.z*clamp(u,0.0,1.0));}
fn froxel_unit_depth(d:f32)->f32{return clamp(log(max(d,media.depth.x)/media.depth.x)/media.depth.z,0.0,1.0);}
fn visible_grid()->vec2<u32>{return media.grid.xy-vec2<u32>(2u);}
fn froxel_world(id:vec3<u32>)->vec3<f32>{let uv=(vec2<f32>(id.xy)+0.5-vec2<f32>(1.0))/vec2<f32>(visible_grid());let ndc=vec2<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0);let h=media.inv_view_proj*vec4<f32>(ndc,1.0,1.0);let ray=normalize(h.xyz/h.w-media.camera.xyz);return media.camera.xyz+ray*froxel_distance((f32(id.z)+0.5)/f32(media.grid.z));}
fn froxel_coord(world:vec3<f32>)->vec3<i32>{let clip=media.view_proj*vec4<f32>(world,1.0);let ndc=clip.xyz/clip.w;let uv=vec2<f32>(ndc.x*0.5+0.5,0.5-ndc.y*0.5);let xy=uv*vec2<f32>(visible_grid())+vec2<f32>(1.0);let z=froxel_unit_depth(length(world-media.camera.xyz))*f32(media.grid.z);return vec3<i32>(vec3<u32>(clamp(vec3<f32>(xy,z),vec3<f32>(0.0),vec3<f32>(media.grid.xyz)-1.0)));}
fn phase_value(cos_theta:f32)->f32{if(media.scattering.w<0.5){return 0.07957747154594767;}let g=media.scattering.z;let d=max(1.0+g*g-2.0*g*clamp(cos_theta,-1.0,1.0),1e-12);return(1.0-g*g)/(12.566370614359172*d*sqrt(d));}
// Bilinear reconstruction across the four froxel columns around a pixel. A
// static per-pixel dither between nearest columns leaves a fixed 8x8 dot
// pattern whenever temporal history does not average it away.
struct FroxelFootprint { base:vec2<i32>, weight:vec2<f32> }
fn froxel_footprint(uv:vec2<f32>)->FroxelFootprint{let centre=uv*vec2<f32>(visible_grid())+vec2<f32>(0.5);let limit=vec2<f32>(media.grid.xy-vec2<u32>(2u));let base=clamp(floor(centre),vec2<f32>(0.0),limit);return FroxelFootprint(vec2<i32>(base),clamp(centre-base,vec2<f32>(0.0),vec2<f32>(1.0)));}
fn bilinear_extinction(f:FroxelFootprint,z:i32)->vec3<f32>{let a=mix(textureLoad(froxel_extinction,vec3<i32>(f.base,z),0).rgb,textureLoad(froxel_extinction,vec3<i32>(f.base+vec2<i32>(1,0),z),0).rgb,f.weight.x);let b=mix(textureLoad(froxel_extinction,vec3<i32>(f.base+vec2<i32>(0,1),z),0).rgb,textureLoad(froxel_extinction,vec3<i32>(f.base+vec2<i32>(1,1),z),0).rgb,f.weight.x);return max(mix(a,b,f.weight.y),vec3<f32>(0.0));}
fn bilinear_scatter(f:FroxelFootprint,z:i32)->vec3<f32>{let a=mix(textureLoad(froxel_scatter,vec3<i32>(f.base,z),0).rgb,textureLoad(froxel_scatter,vec3<i32>(f.base+vec2<i32>(1,0),z),0).rgb,f.weight.x);let b=mix(textureLoad(froxel_scatter,vec3<i32>(f.base+vec2<i32>(0,1),z),0).rgb,textureLoad(froxel_scatter,vec3<i32>(f.base+vec2<i32>(1,1),z),0).rgb,f.weight.x);return max(mix(a,b,f.weight.y),vec3<f32>(0.0));}
fn bilinear_single_source(f:FroxelFootprint,z:i32)->vec3<f32>{let a=mix(textureLoad(froxel_single_source,vec3<i32>(f.base,z),0).rgb,textureLoad(froxel_single_source,vec3<i32>(f.base+vec2<i32>(1,0),z),0).rgb,f.weight.x);let b=mix(textureLoad(froxel_single_source,vec3<i32>(f.base+vec2<i32>(0,1),z),0).rgb,textureLoad(froxel_single_source,vec3<i32>(f.base+vec2<i32>(1,1),z),0).rgb,f.weight.x);return max(mix(a,b,f.weight.y),vec3<f32>(0.0));}
// Depth32Float stores one IEEE-754 f32 code. Reprojection is also evaluated as
// f32, so accepting the immediately adjacent code covers the sole texture
// quantization boundary without admitting a second representable depth step.
fn depth32_history_matches(stored:f32,projected:f32)->bool{if(!(stored>=0.0&&stored<=1.0&&projected>=0.0&&projected<=1.0)){return false;}let a=bitcast<u32>(stored);let b=bitcast<u32>(projected);return max(a,b)-min(a,b)<=1u;}
fn phase_rank(id:vec3<u32>,salt:u32)->f32{let tile=vec2<u32>(textureDimensions(blue_noise_ranks));let p=(id.xy+vec2<u32>(id.z*3u+salt,id.z*5u+salt*7u))%tile;let rank=textureLoad(blue_noise_ranks,vec2<i32>(p),0).r;return(f32(rank)+0.5)/64.0;}
fn trace_index(id:vec3<u32>)->u32{return id.x+media.grid.x*(id.y+media.grid.y*id.z);}
fn direction_uv(direction:vec3<f32>)->vec2<f32>{let d=normalize(direction);return vec2<f32>(atan2(d.z,d.x)*0.15915494309189535+0.5,acos(clamp(d.y,-1.0,1.0))*0.3183098861837907);}
fn directional_radiance(direction:vec3<f32>)->vec3<f32>{let dims=vec2<u32>(textureDimensions(prepared_sky_radiance));let uv=direction_uv(direction);let p=min(vec2<u32>(uv*vec2<f32>(dims)),dims-1u);return max(textureLoad(prepared_sky_radiance,vec2<i32>(p),0).rgb,vec3<f32>(0.0));}
fn phase_direction(incident:vec3<f32>,id:vec3<u32>)->vec3<f32>{let u1=phase_rank(id,0u);let u2=phase_rank(id,1u);var ct=1.0-2.0*u1;if(media.scattering.w>=0.5&&abs(media.scattering.z)>1e-4){let g=media.scattering.z;let ratio=(1.0-g*g)/(1.0-g+2.0*g*u1);ct=clamp((1.0+g*g-ratio*ratio)/(2.0*g),-1.0,1.0);}let st=sqrt(max(0.0,1.0-ct*ct));let phi=6.283185307179586*u2;let helper=select(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),abs(incident.y)<0.999);let tangent=normalize(cross(helper,incident));let bitangent=cross(incident,tangent);return normalize(tangent*(cos(phi)*st)+bitangent*(sin(phi)*st)+incident*ct);}
fn manual_csm_visibility(world:vec3<f32>,normal:vec3<f32>)->f32 {
    let cascade_count=min(csm.cascade_count,4u);
    if(cascade_count==0u){return 1.0;}
    let view_depth=-(media.view*vec4<f32>(world,1.0)).z;
    if(!(view_depth>0.0)){return 1.0;}
    var cascade_index=cascade_count-1u;
    for(var i=0u;i<cascade_count;i=i+1u){if(view_depth<=csm.cascades[i].far_distance){cascade_index=i;break;}}
    let clip=csm.cascades[cascade_index].light_view_proj*vec4<f32>(world,1.0);
    if(abs(clip.w)<1e-8){return 1.0;}
    let ndc=clip.xyz/clip.w;
    let uv=vec2<f32>(ndc.x*0.5+0.5,ndc.y*-0.5+0.5);
    if(any(uv<vec2<f32>(0.0))||any(uv>vec2<f32>(1.0))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
    let dims=vec2<i32>(textureDimensions(terrain_shadow_maps));
    let base=clamp(vec2<i32>(uv*vec2<f32>(dims)),vec2<i32>(0),dims-1);
    let light=normalize(csm.light_direction.xyz);
    let bias=csm.depth_bias+csm.slope_bias*(1.0-max(dot(normal,light),0.0));
    let receiver=ndc.z-bias;
    let kernel=i32(max(csm.pcf_kernel_size,1u));
    let radius=kernel/2;
    var visible=0.0;
    var samples=0.0;
    for(var y=-radius;y<=radius;y=y+1){for(var x=-radius;x<=radius;x=x+1){let p=clamp(base+vec2<i32>(x,y),vec2<i32>(0),dims-1);let depth=textureLoad(terrain_shadow_maps,p,i32(cascade_index),0);visible+=select(0.0,1.0,receiver<=depth);samples+=1.0;}}
    return visible/max(samples,1.0);
}
fn terminal_radiance(hit:NepheleTerrainHit,direction:vec3<f32>,occlusion:bool)->vec3<f32>{if(hit.normal_hit.w<0.5||!occlusion){return directional_radiance(direction);}let n=normalize(hit.normal_hit.xyz);let sun_t=clamp(textureLoad(canonical_light_transmittance,froxel_coord(hit.point_t.xyz),0).rgb,vec3<f32>(0.0),vec3<f32>(1.0));let sun=max(dot(n,normalize(media.sun.xyz)),0.0)*manual_csm_visibility(hit.point_t.xyz,n)*media.sun_radiance.rgb*media.sun_radiance.a*sun_t;let diffuse=max(media.diffuse_ibl.rgb,vec3<f32>(0.0));return max(media.terrain_albedo.rgb,vec3<f32>(0.0))*(sun+diffuse)*0.3183098861837907;}
// Unlike a camera froxel address, this query is zero outside the physical
// medium. Trilinear weights follow the canonical texel-centre convention.
fn world_field(field:texture_3d<f32>,world:vec3<f32>)->vec4<f32> {
    if(any(world<media.world_min.xyz)||any(world>media.world_max.xyz)){return vec4<f32>(0.0);}
    let dimensions=vec3<i32>(textureDimensions(field));
    let coordinate=(world-media.world_min.xyz)/(media.world_max.xyz-media.world_min.xyz)*vec3<f32>(dimensions)-0.5;
    let base=vec3<i32>(floor(coordinate));
    let f=fract(coordinate);
    var result=vec4<f32>(0.0);
    for(var z=0;z<2;z=z+1){for(var y=0;y<2;y=y+1){for(var x=0;x<2;x=x+1){
        let offset=vec3<i32>(x,y,z);
        let w=mix(vec3<f32>(1.0)-f,f,vec3<f32>(offset));
        result+=textureLoad(field,clamp(base+offset,vec3<i32>(0),dimensions-1),0)*w.x*w.y*w.z;
    }}}
    return result;
}
fn continuation_interval(world:vec3<f32>,direction:vec3<f32>,reach:f32)->vec2<f32> {
    if(media.world_min.w<0.5){return vec2<f32>(0.0,reach);}
    var begin=0.0;
    var end=reach;
    for(var axis=0u;axis<3u;axis=axis+1u){
        if(direction[axis]==0.0){
            if(world[axis]<media.world_min[axis]||world[axis]>media.world_max[axis]){return vec2<f32>(0.0);}
        }else{
            let a=(media.world_min[axis]-world[axis])/direction[axis];
            let b=(media.world_max[axis]-world[axis])/direction[axis];
            begin=max(begin,min(a,b));end=min(end,max(a,b));
        }
    }
    return vec2<f32>(begin,max(begin,end));
}
fn continuation_extinction(world:vec3<f32>)->vec3<f32> {
    if(media.world_min.w<0.5){return max(textureLoad(canonical_extinction,froxel_coord(world),0).rgb,vec3<f32>(0.0));}
    return media.sigma_t.rgb*max(world_field(world_medium,world).a,0.0);
}
fn continuation_transmittance(world:vec3<f32>,direction:vec3<f32>,reach:f32)->vec3<f32> {
    let interval=continuation_interval(world,direction,reach);
    let steps=media.grid.z;
    let step=(interval.y-interval.x)/f32(steps);
    var tau=vec3<f32>(0.0);
    for(var i=0u;i<steps;i=i+1u){tau+=continuation_extinction(world+direction*(interval.x+(f32(i)+0.5)*step))*step;}
    return exp(-tau);
}
@compute @workgroup_size(4,4,4)
fn cs_nephele_inject_single(@builtin(global_invocation_id) id:vec3<u32>) {
    if(any(id>=media.grid.xyz)){return;}
    let canonical=textureLoad(canonical_extinction,vec3<i32>(id),0);
    let density=max(canonical.a,0.0);
    if(!(density>0.0)){
        textureStore(froxel_single_scatter,id,vec4<f32>(0.0));
        textureStore(froxel_sun_source_out,id,vec4<f32>(0.0));
        return;
    }
    let world=froxel_world(id);
    let incident=normalize(world-media.camera.xyz);
    let continuation=phase_direction(incident,id);
    let index=trace_index(id);
    let enabled=media.scattering.x>0.5;
    let sun_visible=!enabled||sun_terrain_hits[index].normal_hit.w<0.5;
    let light_t=clamp(textureLoad(canonical_light_transmittance,vec3<i32>(id),0).rgb,vec3<f32>(0.0),vec3<f32>(1.0));
    let sun_source=media.sigma_s.rgb*density*media.sun_radiance.rgb*media.sun_radiance.a*f32(sun_visible)*light_t;
    let camera_phase=phase_value(dot(incident,normalize(media.sun.xyz)));
    let direct=sun_source*camera_phase;
    textureStore(froxel_sun_source_out,id,vec4<f32>(sun_source,camera_phase));
    let hit=phase_terrain_hits[index];
    let reaches_terrain=enabled&&hit.normal_hit.w>0.5;
    let reach=select(media.depth.y,min(hit.point_t.w,media.depth.y),reaches_terrain);
    let phase_pdf=phase_value(dot(incident,continuation));
    let environment=media.sigma_s.rgb*density*(phase_pdf/max(phase_pdf,1e-20))*continuation_transmittance(world,continuation,reach)*terminal_radiance(hit,continuation,enabled);
    textureStore(froxel_single_scatter,id,vec4<f32>(direct+environment,0.0));
}
// Each froxel's single-scatter source comes from one stochastic continuation
// direction drawn from an 8x8 blue-noise tile, so neighbouring froxels differ
// strongly and the periodic tile shows as a crosshatch. A 3x3 tent across
// froxel columns of the same slice, restricted to froxels that hold medium,
// averages nine decorrelated estimates before multiple scattering and
// integration consume the source.
@compute @workgroup_size(4,4,4) fn cs_nephele_filter_single(@builtin(global_invocation_id) id:vec3<u32>){if(any(id>=media.grid.xyz)){return;}let centre=vec3<i32>(id);if(!(max(textureLoad(froxel_extinction,centre,0).r,max(textureLoad(froxel_extinction,centre,0).g,textureLoad(froxel_extinction,centre,0).b))>0.0)){textureStore(froxel_single_scatter,id,vec4<f32>(0.0));return;}let limit=vec2<i32>(media.grid.xy)-vec2<i32>(1);var sum=vec3<f32>(0.0);var weight=0.0;for(var dy=-1;dy<=1;dy=dy+1){for(var dx=-1;dx<=1;dx=dx+1){let c=vec3<i32>(clamp(centre.xy+vec2<i32>(dx,dy),vec2<i32>(0),limit),centre.z);let ext=textureLoad(froxel_extinction,c,0).rgb;if(max(ext.r,max(ext.g,ext.b))>0.0){let w=f32((2-abs(dx))*(2-abs(dy)));sum+=w*max(textureLoad(froxel_single_source,c,0).rgb,vec3<f32>(0.0));weight+=w;}}}textureStore(froxel_single_scatter,id,vec4<f32>(sum/max(weight,1.0),0.0));}
// Match the existing single-source tent exactly, but evaluate its direct sun
// term in the gathered ray's direction. A camera-direction radiance field is
// not a direction-independent source for anisotropic multiple scattering.
fn continuation_source(centre:vec3<i32>, direction:vec3<f32>, world:vec3<f32>)->vec3<f32> {
    let limit=vec2<i32>(media.grid.xy)-vec2<i32>(1);
    let phase=phase_value(dot(direction,normalize(media.sun.xyz)));
    var camera_sun=vec3<f32>(0.0);
    var gathered_sun=vec3<f32>(0.0);
    var cached_density=0.0;
    var weight=0.0;
    for(var dy=-1;dy<=1;dy=dy+1){
        for(var dx=-1;dx<=1;dx=dx+1){
            let c=vec3<i32>(clamp(centre.xy+vec2<i32>(dx,dy),vec2<i32>(0),limit),centre.z);
            let extinction=textureLoad(froxel_extinction,c,0);
            let ext=extinction.rgb;
            if(max(ext.r,max(ext.g,ext.b))>0.0){
                let w=f32((2-abs(dx))*(2-abs(dy)));
                let sun=textureLoad(froxel_sun_source,c,0);
                cached_density+=w*extinction.a;
                camera_sun+=w*sun.rgb*sun.a;
                gathered_sun+=w*sun.rgb*phase;
                weight+=w;
            }
        }
    }
    let filtered=textureLoad(froxel_single_source,centre,0).rgb;
    let environment=max(filtered-camera_sun/max(weight,1.0),vec3<f32>(0.0));
    if(media.world_min.w<0.5){return environment+gathered_sun/max(weight,1.0);}
    // Keep the existing directional environment cache, expressed per density.
    // Sun radiance and extinction use the full world field along this ray.
    let density=max(world_field(world_medium,world).a,0.0);
    var environment_source=vec3<f32>(0.0);
    if(cached_density>0.0){environment_source=environment*(density/(cached_density/max(weight,1.0)));}
    return environment_source+world_field(world_sun_source,world).rgb*phase;
}

@compute @workgroup_size(4,4,4)
fn cs_nephele_inject_multiple(@builtin(global_invocation_id) id:vec3<u32>) {
    if(any(id>=media.grid.xyz)){return;}
    let canonical=textureLoad(canonical_extinction,vec3<i32>(id),0);
    let density=max(canonical.a,0.0);
    let single=max(textureLoad(froxel_single_source,vec3<i32>(id),0).rgb,vec3<f32>(0.0));
    if(!(density>0.0)){textureStore(froxel_in_scatter,id,vec4<f32>(single,0.0));return;}
    let world=froxel_world(id);
    let incident=normalize(world-media.camera.xyz);
    let continuation=phase_direction(incident,id);
    let hit=phase_terrain_hits[trace_index(id)];
    let reaches_terrain=media.scattering.x>0.5&&hit.normal_hit.w>0.5;
    let reach=select(media.depth.y,min(hit.point_t.w,media.depth.y),reaches_terrain);
    let interval=continuation_interval(world,continuation,reach);
    let steps=media.grid.z;
    let step=(interval.y-interval.x)/f32(steps);
    var t=vec3<f32>(1.0);
    var gathered=vec3<f32>(0.0);
    for(var i=0u;i<steps;i=i+1u){
        let sample_world=world+continuation*(interval.x+(f32(i)+0.5)*step);
        let c=froxel_coord(sample_world);
        let ext=continuation_extinction(sample_world);
        let source=continuation_source(c,continuation,sample_world);
        let st=exp(-ext*step);
        gathered+=t*source*(vec3<f32>(1.0)-st)/max(ext,vec3<f32>(1e-6));
        t*=st;
    }
    let pdf=phase_value(dot(incident,continuation));
    let multiple=media.sigma_s.rgb*density*(phase_value(dot(incident,continuation))/max(pdf,1e-20))*gathered;
    textureStore(froxel_in_scatter,id,vec4<f32>(single+multiple,dot(multiple,vec3<f32>(0.2126,0.7152,0.0722))));
}
@compute @workgroup_size(8,8,1)
fn cs_nephele_integrate(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pixel=gid.xy;
    if(any(pixel>=media.viewport.xy)){return;}
    let uv=(vec2<f32>(pixel)+0.5)/vec2<f32>(media.viewport.xy);
    let footprint=froxel_footprint(uv);
    let sampled_depth=textureLoad(scene_depth,vec2<i32>(pixel),0);
    // NaN fails both ordered comparisons. Exact zero is the camera plane and
    // cannot be a physically visible terrain sample, so fail closed to the
    // configured froxel far distance instead of silently integrating no media.
    let raw_depth=select(1.0,sampled_depth,sampled_depth>0.0&&sampled_depth<=1.0);
    let ndc=vec2<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0);
    let scene_h=media.inv_view_proj*vec4<f32>(ndc,raw_depth,1.0);
    let scene_world=scene_h.xyz/scene_h.w;
    let reconstructed_distance=length(scene_world-media.camera.xyz);
    let scene_distance=select(media.depth.y,reconstructed_distance,reconstructed_distance>=media.depth.x&&reconstructed_distance<=media.depth.y);
    let max_slice=min(u32(froxel_unit_depth(scene_distance)*f32(media.grid.z)),media.grid.z-1u);
    var t=vec3<f32>(1.0);
    var s=vec3<f32>(0.0);
    var tau=vec3<f32>(0.0);
    var multiple_scatter_rgb=vec3<f32>(0.0);
    var previous=media.depth.x;
    for(var z=0u;z<=max_slice;z=z+1u){
        let distance=froxel_distance((f32(z)+1.0)/f32(media.grid.z));
        let step=max(min(distance,scene_distance)-previous,0.0);
        previous=distance;
        let ext=bilinear_extinction(footprint,i32(z));
        let source=bilinear_scatter(footprint,i32(z));
        let single_source=bilinear_single_source(footprint,i32(z));
        let multiple_rgb=max(source-single_source,vec3<f32>(0.0));
        let st=exp(-ext*step);
        let segment_weight=t*(vec3<f32>(1.0)-st)/max(ext,vec3<f32>(1e-6));
        s+=segment_weight*source;
        tau+=ext*step;
        multiple_scatter_rgb+=segment_weight*multiple_rgb;
        t*=st;
    }
    // The CPU history key first rejects camera/depth/density/light/resize/resource
    // changes. Accepted history is then reprojected and depth-tested here.
    if(media.viewport.z!=0u){
        let far_h=media.inv_view_proj*vec4<f32>(ndc,1.0,1.0);
        let ray=normalize(far_h.xyz/far_h.w-media.camera.xyz);
        let world=media.camera.xyz+ray*scene_distance;
        let old_clip=media.previous_view_proj*vec4<f32>(world,1.0);
        if(old_clip.w>0.0){
            let old_ndc=old_clip.xy/old_clip.w;
            let old_uv=vec2<f32>(old_ndc.x*0.5+0.5,0.5-old_ndc.y*0.5);
            if(all(old_uv>=vec2<f32>(0.0))&&all(old_uv<vec2<f32>(1.0))){
                let old_pixel=min(vec2<u32>(old_uv*vec2<f32>(media.viewport.xy)),media.viewport.xy-1u);
                let old_depth=textureLoad(history_depth,vec2<i32>(old_pixel),0);
                let projected_old_depth=old_clip.z/old_clip.w;
                if(depth32_history_matches(old_depth,projected_old_depth)){
                    s=mix(s,textureLoad(history_scatter,vec2<i32>(old_pixel),0).rgb,clamp(media.depth.w,0.0,1.0));
                }
            }
        }
    }
    // Dimensionless Beer accounting residual. `t` is the product of the
    // executed per-slice factors; exp(-tau) is independently reconstructed
    // from the executed integrated extinction.
    let energy_residual=max(max(abs(t.x-exp(-tau.x)),abs(t.y-exp(-tau.y))),abs(t.z-exp(-tau.z)));
    textureStore(out_transmittance,vec2<i32>(pixel),vec4<f32>(t,energy_residual));
    textureStore(out_in_scatter,vec2<i32>(pixel),vec4<f32>(s,dot(multiple_scatter_rgb,vec3<f32>(0.2126,0.7152,0.0722))));
    textureStore(out_cloud_shadow,vec2<i32>(pixel),vec4<f32>(clamp(textureLoad(canonical_light_transmittance,froxel_coord(scene_world),0).rgb,vec3<f32>(0.0),vec3<f32>(1.0)),1.0));
    // The public optical-depth AOV consumes RGB only. Keep the production
    // termination slice in its alpha channel so the integration kernel stays
    // within WebGPU's portable four-storage-texture limit.
    textureStore(out_optical_depth,vec2<i32>(pixel),vec4<f32>(tau,f32(max_slice)));
}
@compute @workgroup_size(8,8,1) fn cs_nephele_composite_linear_hdr(@builtin(global_invocation_id) gid:vec3<u32>){let p=gid.xy;if(any(p>=media.viewport.xy)){return;}let c=vec2<i32>(p);let terrain=textureLoad(terrain_linear_hdr,c,0).rgb;let scatter=textureLoad(media_integrated_hdr,c,0).rgb;let t=textureLoad(media_transmittance,c,0).rgb;textureStore(composite_linear_hdr,c,vec4<f32>(terrain*t+scatter,1.0));}
