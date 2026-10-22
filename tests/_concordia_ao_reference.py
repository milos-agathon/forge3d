"""Independent PROMETHEUS depth/ambient reference for native-frame acceptance.

All auxiliary reads follow the one routine beauty read. Reference arrays and
CPU comparison math are validation-only and never enter MapScene rendering.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import forge3d.map_scene as ms
from forge3d._map_scene_projection import TerrainProjector
from forge3d._map_scene_native_inputs import building_world_triangles
from forge3d.path_tracing import hybrid_render_terrain_reference
from _concordia_scene import reference_scene,capture,world_positions
from _ssim import ssim


def _visibility(data):
    color=data["last_composition_hdr"][...,:3].astype(np.float64).sum(axis=-1)
    original=data["geometry_hdr"][...,:3].astype(np.float64).sum(axis=-1)
    return np.divide(color,original,out=np.ones_like(color),where=original>0)


def _depth_reference_gtao(reference,data,radius,bias,num_samples):
    """Evaluate contact visibility on independent PT depths and normals.

    The first native one-shot frame uses the existing 16-sample GTAO budget,
    four azimuths/four radial steps. This comparison tests shared-depth
    participation; the separate SSIM gate tests physical AO quality.
    """
    depth=data["depth"]
    height,width=depth.shape
    yy,xx=np.indices(depth.shape)
    uv=np.stack(((xx+.5)/width,(yy+.5)/height),axis=-1)
    p00,p11=data["proj"][0][0],data["proj"][1][1]
    ray_cos=1/np.sqrt(1+((uv[...,0]*2-1)/p00)**2+((1-uv[...,1]*2)/p11)**2)
    ref_depth=np.where(np.isfinite(reference["depth"]),reference["depth"]*ray_cos,0)
    def position(coordinate,z):
        return np.stack(((coordinate[...,0]*2-1)/p00*z,(1-coordinate[...,1]*2)/p11*z,-z),axis=-1)
    center=position(uv,ref_depth)
    normal=reference["normal"][...,[0,2,1]]@np.asarray(data["view"])[:3,:3]
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),np.finfo(np.float32).tiny)
    visibility=np.zeros_like(ref_depth)
    kind=data["surface_kind"]
    # First-frame IGN integer offsets round to integral f32 values; the
    # measured native phase is zero. Evaluate the same declared azimuths.
    directions=max(num_samples//4,2)
    for direction in range(directions):
        angle=direction/directions*2*np.pi
        axis=np.array([np.cos(angle),np.sin(angle)])
        blocked=np.zeros_like(ref_depth)
        for step in range(1,5):
            fraction=(step/4)**2
            screen_radius=radius*(.5*height*p11/np.maximum(ref_depth,np.finfo(np.float32).tiny))
            lookup=uv+axis*(screen_radius*fraction)[...,None]/[width,height]
            valid=np.all((lookup>=0)&(lookup<=1),axis=-1)
            # Clip only indices after retaining the independent validity mask.
            pixel=np.minimum((np.clip(lookup,0,1)*[width,height]).astype(int),[width-1,height-1])
            sample_depth=ref_depth[pixel[...,1],pixel[...,0]]
            sample=position((pixel+.5)/[width,height],sample_depth)
            delta=sample-center
            distance=np.linalg.norm(delta,axis=-1)
            cosine=np.maximum(np.sum(delta*normal,axis=-1)-bias,0)/np.maximum(distance,1e-4)
            attenuation=1-np.clip(distance/radius,0,1)
            valid&=(sample_depth>0)&~((kind==1)&(kind[pixel[...,1],pixel[...,0]]==1))
            contribution=np.clip(cosine,0,1)**2*attenuation
            blocked=np.maximum(blocked,np.where(valid,contribution,0))
        visibility+=1-blocked
    return np.where(ref_depth>0,np.clip(visibility/directions,.05,1),1)


@cache
def ao_reference_metrics():
    model=reference_scene()
    result,data=capture(model)
    variants={}
    for name,height_enabled,gtao_enabled in (("gtao_only",False,True),("height_only",True,False),("off",False,False)):
        variant=reference_scene()
        variant.recipe.lighting.settings["height_ao"]["enabled"]=height_enabled
        variant.recipe.lighting.settings["screen_space"]["ssao"]["enabled"]=gtao_enabled
        variants[name]=capture(variant)
    projector=TerrainProjector(model.recipe)
    dem=np.asarray(model.recipe.terrain.data,dtype=np.float32)
    low,high=ms._heightmap_domain(dem)
    center=(high-low)*float(projector.params.z_scale)*.5
    inverse=np.asarray(data["inv_view"]).T
    def pt(point): return (float(point[0]),float(point[2])+center,float(point[1]))
    eye=inverse[:3,3];forward=-inverse[:3,2];up=inverse[:3,1]
    camera=dict(origin=pt(eye),look_at=pt(eye+forward),up=(float(up[0]),float(up[2]),float(up[1])),
        fov_y=float(np.degrees(2*np.arctan(1/np.asarray(data["proj"])[1,1]))),exposure=1.0)
    triangles=np.asarray(building_world_triangles(projector),dtype=np.float32)
    vertices=triangles.reshape(-1,3)[:,[0,2,1]].copy();vertices[:,1]+=center
    # XYZ -> XZY reflects handedness. Reverse each triangle so the PT roof
    # normal stays outward instead of tracing its hemisphere inside the box.
    indices=np.ascontiguousarray(np.arange(len(vertices),dtype=np.uint32).reshape(-1,3)[:,[0,2,1]])
    reference=hybrid_render_terrain_reference(np.ascontiguousarray(dem-low),256,192,camera,
        spacing=(projector.span/(dem.shape[1]-1),)*2,exaggeration=float(projector.params.z_scale),
        albedo=(1.,1.,1.),sun_azimuth_deg=0.,sun_elevation_deg=45.,sun_intensity=0.,env_intensity=1.,
        mesh_vertices=vertices,mesh_indices=indices,spp=8,max_frames=512,min_frames=512,
        variance_threshold=1e-3,seed=45)
    assert reference["converged"] and reference["variance"] < 1e-3
    world=world_positions(data)
    terrain=data["surface_kind"]==1
    contact=np.zeros_like(terrain)
    batches=ms._terrain_scatter_building_batches_for_recipe(model.recipe,dem,shared_frame=True)[0]
    cursor=0
    for batch in batches:
        count=np.asarray(batch["levels"][0]["mesh"]["indices"]).size//3*len(batch["transforms"])
        group=triangles[cursor:cursor+count].reshape(-1,3);cursor+=count
        a,b=group[:,:2].min(axis=0),group[:,:2].max(axis=0)
        distance=np.linalg.norm(np.maximum(np.maximum(a-world[...,:2],world[...,:2]-b),0),axis=-1)
        contact|=terrain&(distance<=2.0)  # declared two-metre building contact band
    ridge=terrain&(np.abs(world[...,0])<8.0)  # declared sixteen-metre canyon floor
    ref_hit=np.isfinite(reference["depth"])
    roi=(contact|ridge)&ref_hit
    contact&=ref_hit
    observed={"combined":_visibility(data),**{name:_visibility(values) for name,(_,values) in variants.items()}}
    for _,values in variants.values():
        np.testing.assert_array_equal(data["depth"],values["depth"])
        np.testing.assert_array_equal(data["geometry_hdr"],values["geometry_hdr"])
    # The tracer's mesh material has its own albedo. Recover visibility from
    # the measured albedo AOV rather than interpreting a colored roof as AO.
    reference_albedo=np.asarray(reference["albedo"],dtype=np.float64)
    visibility_rgb=np.divide(reference["radiance"],reference_albedo,
        out=np.zeros_like(reference_albedo),where=reference_albedo>0)
    expected=np.clip(visibility_rgb.mean(axis=-1),0,1)
    def lightness(value):
        # CIE76 on linear neutral visibility patches (a*=b*=0, D65).
        delta=6/29
        f=np.where(value>delta**3,np.cbrt(np.maximum(value,0)),value/(3*delta**2)+4/29)
        return 116*f-16
    scores={name:{"ssim":float(ssim(value,expected,data_range=1,mask=roi)),
        "mae":float(np.abs(value[roi]-expected[roi]).mean()),
        "delta_e_76":float(np.abs(lightness(value[roi])-lightness(expected[roi])).mean())}
        for name,value in observed.items()}
    radius=float(model.recipe.lighting.settings["screen_space"]["ssao"]["radius"])
    parameters=result.metadata["native_pipeline"]["ao_parameters"]
    reference_contact=_depth_reference_gtao(reference,data,radius,float(parameters["bias"]),int(parameters["num_samples"]))
    # Encoded view normals use RGBA16Float. The first-order visibility
    # sensitivity is at most two times the 3-vector normal storage envelope.
    precision=2*np.sqrt(3)*np.finfo(np.float16).eps
    agreement=float(np.mean(np.abs(data["gtao_raw"][contact]-reference_contact[contact])<=precision))
    # Independent reference selects fully clear opaque receivers in this mixed canyon /
    # building scene. Every reference RGB channel must be white within its
    # float32 storage precision, rather than selecting pixels with native AO.
    # Strictly clear terrain facets, independently selected by the PT white
    # furnace. Terrain albedo is exactly one; no native AO value selects ROI.
    nonoccluded=terrain&ref_hit&np.all(reference_albedo==1.0,axis=-1)&np.all(
        np.abs(reference["radiance"]-1.0)<=np.finfo(np.float32).eps,axis=-1)
    if not nonoccluded.any():
        import os
        from pathlib import Path
        directory=os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR")
        if directory:
            folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/"reference-clear-diagnostic.npz",radiance=reference["radiance"],
                albedo=reference_albedo,kind=data["surface_kind"],reference_depth=reference["depth"],
                estimator_variance=reference["luminance_variance"],visibility=expected,
                rgba_combined=result.rgba,rgba_off=variants["off"][0].rgba)
    assert nonoccluded.any()
    luminance=np.asarray([.2126,.7152,.0722])
    clear=variants["off"][0].rgba[...,:3]@luminance
    combined=result.rgba[...,:3]@luminance
    luminance_change=float(np.abs(combined[nonoccluded]-clear[nonoccluded]).mean()/clear[nonoccluded].mean())
    metrics={"adapter":result.metadata["native_pipeline"]["execution_report"]["adapter"],
        "reference":{"frames":int(reference["frames"]),"variance":float(reference["variance"]),"converged":bool(reference["converged"]),"camera":camera},
        "gtao_radius_m":radius,"contact_band_m":2.0,"roi_pixels":int(roi.sum()),
        "contact_pixels":int(contact.sum()),"contact_visibility_precision":float(precision),
        "nonoccluded_pixels":int(nonoccluded.sum()),"nonoccluded_luminance_change":luminance_change,
        "nonoccluded_definition":"strict unit-albedo terrain white-furnace radiance within float32 rounding",
        "nonoccluded_surfaces":{str(kind):int(np.sum(nonoccluded&(data["surface_kind"]==kind))) for kind in (1,2)},
        "contact_depth_reference_agreement":agreement,"scores":scores,
        "ablation_certificates":{name:item.metadata["native_pipeline"] for name,(item,_) in variants.items()}}
    images={"reference":expected,**observed,"roi":roi,"contact":contact,"ridge":ridge,
        "nonoccluded":nonoccluded,"rgba_off":variants["off"][0].rgba,"reference_contact_visibility":reference_contact,"native_contact_visibility":data["gtao_raw"],"rgba":result.rgba}
    return metrics,images


def write_ao_evidence(metrics,data,directory):
    import json
    from pathlib import Path
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/"ao-reference-metrics.json").write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(directory/"ao-reference-data.npz",**data)
    for name in ("reference","combined","gtao_only","height_only"):
        image=Image.fromarray(np.round(np.clip(data[name],0,1)*255).astype(np.uint8))
        image.save(directory/(name+".png"))
        for crop in ("contact","ridge"):
            yy,xx=np.nonzero(data[crop])
            if len(xx):
                image.crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))).save(directory/(name+"-"+crop+".png"))
    Image.fromarray(data["rgba"]).save(directory/"ao-mixed-beauty.png")
    Image.fromarray(data["rgba_off"]).save(directory/"ao-off-beauty.png")
    Image.fromarray(data["nonoccluded"].astype(np.uint8)*255).save(directory/"nonoccluded-reference-roi.png")
