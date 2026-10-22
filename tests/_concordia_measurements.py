"""After-beauty physical measurements; never supplies inputs to rendering."""
from __future__ import annotations
import numpy as np
from _concordia_scene import scene,capture,world_positions


def cloud_coupling():
    frames={}
    reports={}
    for cloud in (False,True):
        for sun in (0.0,1.0):
            model=scene(effects=('taa','cloud_shadows') if cloud else ('taa',),cartography=False,height_ao=False)
            model.recipe.lighting.intensity=sun
            image,data=capture(model)
            frames[cloud,sun]=(image,data)
            reports[f'cloud_{cloud}_sun_{sun}']=image.metadata['native_pipeline']
    data=frames[True,1.0][1]
    world=world_positions(data)
    field=data['cloud_field'][...,0]/255.0
    scale=np.asarray(data['cloud_scale_bias'])
    uv=np.clip(world[...,:2]*scale[:2]+scale[2:],0,1)
    coords=uv*np.array([field.shape[1],field.shape[0]])-.5
    base=np.floor(coords).astype(int); t=coords-base
    x0=np.clip(base[...,0],0,field.shape[1]-1);x1=np.clip(base[...,0]+1,0,field.shape[1]-1)
    y0=np.clip(base[...,1],0,field.shape[0]-1);y1=np.clip(base[...,1]+1,0,field.shape[0]-1)
    expected=(field[y0,x0]*(1-t[...,0])+field[y0,x1]*t[...,0])*(1-t[...,1])+(field[y1,x0]*(1-t[...,0])+field[y1,x1]*t[...,0])*t[...,1]
    clear=frames[False,1.0][1]['geometry_hdr'][...,:3].astype(float)-frames[False,0.0][1]['geometry_hdr'][...,:3].astype(float)
    cloud=frames[True,1.0][1]['geometry_hdr'][...,:3].astype(float)-frames[True,0.0][1]['geometry_hdr'][...,:3].astype(float)
    denominator=clear.sum(axis=-1)
    observed=np.divide(cloud.sum(axis=-1),denominator,out=np.ones_like(denominator),where=denominator!=0)
    # Mid-transmission is an objective contour of the generated field, declared
    # from its complete range before observing terrain/building shading.
    threshold=float((field.min()+field.max())*.5)
    payload={'threshold_mid_transmission':threshold,'cloud_field_range':[float(field.min()),float(field.max())],
        'adapter':reports['cloud_True_sun_1.0']['execution_report']['adapter'],'surfaces':{}}
    masks={}
    for kind,name in ((1,'terrain'),(2,'buildings')):
        roi=(data['surface_kind']==kind)&(denominator>0)
        a=(expected<threshold)&roi;b=(observed<threshold)&roi
        # Displacement is measured in output pixels, independently for each
        # receiver, against the actual world-projected shared GPU cloud field.
        difference=a^b
        displacement=0.0
        if difference.any():
            def directed(points,target):
                if not len(points): return 0.0
                if not len(target): return float("inf")
                return max(float(np.sqrt(np.min(np.sum((target-point)**2,axis=1)))) for point in points)
            displacement=max(directed(np.argwhere(b&~a),np.argwhere(a)),directed(np.argwhere(a&~b),np.argwhere(b)))
        payload['surfaces'][name]={'lit_pixels':int(roi.sum()),'expected_covered':int(a.sum()),
            'observed_covered':int(b.sum()),'coverage_disagreement_pixels':int(difference.sum()),
            'max_displacement_px':displacement,'mean_absolute_transmission_error':float(np.abs(expected[roi]-observed[roi]).mean())}
        masks[name+'_roi']=roi
        masks[name+'_expected_coverage']=a;masks[name+'_observed_coverage']=b
    terrain=data['surface_kind']==1
    changed=np.any(frames[True,1.0][0].rgba[...,:3]!=frames[False,1.0][0].rgba[...,:3],axis=-1)
    payload['changed_terrain_fraction']=float(changed[terrain].mean())
    payload['ablation_certificates']=reports
    return payload,masks,frames


def write_cloud_evidence(metrics,masks,frames,directory):
    import json
    from pathlib import Path
    from PIL import Image
    folder=Path(directory)/"clouds"
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"cloud-coupling.json").write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(folder/"coverage-masks.npz",**masks)
    cloudy=frames[True,1.0][0].rgba
    clear=frames[False,1.0][0].rgba
    Image.fromarray(cloudy).save(folder/"clouds-enabled.png")
    Image.fromarray(clear).save(folder/"clouds-disabled.png")
    field=frames[True,1.0][1]["cloud_field"]
    Image.fromarray(field).save(folder/"shared-cloud-field.png")
    # One output pixel around the objective contour exposes the exact
    # boundary used by the displacement gate, on both kinds of receiver.
    coverage=masks["terrain_expected_coverage"]|masks["buildings_expected_coverage"]
    edge=np.zeros_like(coverage)
    edge[:,1:]|=coverage[:,1:]!=coverage[:,:-1]
    edge[:,:-1]|=coverage[:,1:]!=coverage[:,:-1]
    edge[1:]|=coverage[1:]!=coverage[:-1]
    edge[:-1]|=coverage[1:]!=coverage[:-1]
    yy,xx=np.nonzero(edge)
    if len(xx):
        bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
        Image.fromarray(cloudy).crop(bounds).save(folder/"cloud-boundary.png")
        Image.fromarray(clear).crop(bounds).save(folder/"cloud-boundary-clear.png")
    for name,mask in masks.items():
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+".png"))
