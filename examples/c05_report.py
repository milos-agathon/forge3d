"""Verify the C05 real-data outputs and write their local evidence report."""
from __future__ import annotations
import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import rasterio  # before the checkout shim
from rasterio.features import rasterize
import numpy as np
import shapely
from PIL import Image,ImageDraw
from _import_shim import ensure_repo_import
ensure_repo_import()
from forge3d import PreparedScene,LandscapeLayout,export_landscape
from forge3d.preparation import _sha
from c05_real_data import record,package_check,implementation_identity


def registration(root):
    reference=json.loads((root/'sources/rotterdam-water.geojson').read_text(encoding='utf-8'))
    river=[f for f in reference['features'] if f['properties'].get('naamnl')=='Nieuwe Maas']
    if not river: raise AssertionError('Named independent river is absent')
    prepared=PreparedScene.load(root/'rotterdam/initial/prepared')
    grid=prepared.recipe.target_grid
    geometries=[f['geometry'] for f in river]
    mask=rasterize([(g,1) for g in geometries],out_shape=(grid.height,grid.width),transform=grid.transform,fill=0,dtype='uint8')!=0
    rows,cols=np.where(mask)
    centres=shapely.points(88000+(cols+.5)*10,442000-(rows+.5)*10)
    union=shapely.union_all([shapely.geometry.shape(g) for g in geometries])
    if not np.all(shapely.covers(union,centres)): raise AssertionError('Independent river rasterization misplaced centres')
    valid=prepared.layers['terrain'].valid_mask
    # Cross-product registration: compare independent BRT footprints with AHN
    # voids. A half-pixel shift lattice over the review's +/-30 m workload
    # exposes a displacement; it does not turn AHN voids into water truth.
    from rasterio.transform import Affine
    spacing=abs(grid.geotransform[1])
    shifts=np.arange(-30,30+spacing/2,spacing/2)
    # Use the re-review's +/-60 m shoreline corridor, excluding distant AHN
    # voids (such as buildings) from the more local agreement denominator.
    corridor=rasterize([(shapely.geometry.mapping(union.buffer(60).difference(union.buffer(-60))),1)],
                       out_shape=(grid.height,grid.width),transform=grid.transform,fill=0,dtype='uint8')!=0
    scores=[]
    for dx in shifts:
        for dy in shifts:
            transform=Affine.from_gdal(*(np.asarray(grid.geotransform)+[dx,0,0,dy,0,0]))
            shifted=rasterize([(g,1) for g in geometries],out_shape=(grid.height,grid.width),transform=transform,fill=0,dtype='uint8')!=0
            intersection=int((shifted & ~valid).sum()); combined=int((shifted | ~valid).sum())
            scores.append({'grid_offset_xy_m':[float(dx),float(dy)],'intersection_cells':intersection,
                           'union_cells':combined,'iou':intersection/combined,
                           'shoreline_agreement':float(((shifted==~valid)&corridor).sum()/corridor.sum())})
    best=max(scores,key=lambda s:s['iou'])
    zero=next(s for s in scores if s['grid_offset_xy_m']==[0,0])
    cardinal=[s for s in scores if s['grid_offset_xy_m'] in [[-spacing,0],[spacing,0],[0,-spacing],[0,spacing]]]
    shoreline_best=max(scores,key=lambda s:s['shoreline_agreement'])
    margins=[{'grid_offset_xy_m':s['grid_offset_xy_m'],'zero_minus_shift_iou':zero['iou']-s['iou'],
              'zero_minus_shift_shoreline_agreement':zero['shoreline_agreement']-s['shoreline_agreement']} for s in cardinal]
    if (any(abs(v)>spacing/2 for v in best['grid_offset_xy_m']) or
            not all(zero['iou']>s['iou'] for s in cardinal)):
        raise AssertionError('Independent BRT/AHN displacement check no longer peaks within half an output pixel')
    import rasterio
    with rasterio.open(root/'common/grindelwald-alti-2025-2m.tif') as ds:
        heights=ds.read(1,masked=True)
        row,col=np.unravel_index(np.argmax(heights),heights.shape)
        summit=float(heights[row,col]); peak_xy=ds.xy(row,col)
    position_source=json.loads((root/'sources/swiss-eiger-position-source.json').read_text(encoding='utf-8'))
    position_path=root/'sources/swiss-eiger-position.json'
    if _sha(position_path)!=position_source['sha256']:raise AssertionError('Independent Swiss reference changed')
    positions=json.loads(position_path.read_text(encoding='utf-8'))['results']
    position=next(p for p in positions if p['featureId']==position_source['feature_id'])
    if position['geometry']['spatialReference']['wkid']!=2056:raise AssertionError('Unexpected reference CRS')
    named_xy=np.asarray(position['geometry']['points'][0],dtype=float)
    residual=np.asarray(peak_xy)-named_xy
    published=3967.0  # Swiss Alpine Club Eiger route portal, rounded to metres
    if abs(summit-published)>.5:
        raise AssertionError('Eiger peak is inconsistent with the independently published metre-rounded height')
    official=json.loads((root/'sources/cbs-source.json').read_text(encoding='utf-8'))
    common=json.loads((root/'common/provenance.json').read_text(encoding='utf-8'))
    if official['sha256']!=common['cbs']['sha256']: raise AssertionError('Official and staged CBS archives differ')
    # Registration visualization uses only original footprint coordinates.
    # No fitted shift or invented shoreline is applied to either dataset.
    base=Image.open(root/'rotterdam/initial/display.png').convert('RGB')
    canvas=Image.new('RGB',(1400,1300),'#f7f6f2'); canvas.paste(base,(100,70))
    draw=ImageDraw.Draw(canvas)
    for geometry in geometries:
        polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
        for polygon in polygons:
            for ring in polygon:
                outline_xy=[(100+(x-88000)/10,70+(442000-y)/10) for x,y,*_ in ring]
                draw.line(outline_xy,fill='#123b58',width=2)
    draw.text((100,18),'Rotterdam | Independent BRT Nieuwe Maas outlines over the recorded display',fill='#192a31')
    draw.text((100,40),'RD EPSG:28992; 10 m pixels. Blue hatch = missing AHN height, not a water classification.',fill='#192a31')
    draw.text((100,1275),'E 88000 to 100000 | N 430000 to 442000 | North at top | No registration fit applied',fill='#192a31')
    canvas.save(root/'river-registration.png')
    result={'named_river_features':len(river),'independent_reference_features':len(reference['features']),
        'implementation':implementation_identity(),
        'cross_product_registration':{'reference':'BRT Nieuwe Maas footprints versus AHN missing-height mask',
            'grid_spacing_m':spacing,'shift_step_m':spacing/2,'shift_extent_m':30,
            'shift_extent_basis':'Reproduces the supplied review shift workload, not an accuracy threshold',
            'best':best,'unshifted':zero,'one_pixel_controls':cardinal,'all_scores':scores,
            'zero_minus_one_pixel_margins':margins,
            'shoreline_corridor':{'half_width_m':60,'width_basis':'Supplied re-review workload, not an accuracy tolerance',
                'cells':int(corridor.sum()),'metric':'agreement fraction of BRT water versus AHN voids inside the shoreline corridor',
                'best':shoreline_best,'unshifted_agreement':zero['shoreline_agreement'],
                'peak_minus_zero_agreement':shoreline_best['shoreline_agreement']-zero['shoreline_agreement']},
            'interpretation':'The scores support approximate one-output-pixel registration, but the small margins cannot resolve a possible westward half-pixel offset. No shift is fitted or applied. Different dates and AHN void semantics prevent survey-accuracy inference.'},
        'independent_swiss_position':{'reference':position_source,'feature_id':position['featureId'],
            'reference_xy_lv95':named_xy.tolist(),'dem_peak_xy_lv95':list(peak_xy),
            'dem_minus_reference_xy_m':residual.tolist(),'horizontal_residual_m':float(np.linalg.norm(residual)),
            'source_pixel_spacing_m':2,'output_pixel_spacing_m':8,
            'interpretation':'Observed separation between an independent named-summit point and the measured DEM maximum. The different point definitions and unspecified landmark accuracy do not support a subpixel or survey-accuracy claim; no threshold or fitted offset is invented.'},
        'independent_height':{'reference_url':'https://www.sac-cas.ch/en/huts-and-tours/sac-route-portal/eiger-555/',
            'published_eiger_height_m':published,'reference_rounding_step_m':1,'rounding_half_step_m':.5,
            'source_dem_peak_m':summit,'source_dem_peak_xy_lv95':list(peak_xy),'residual_m':summit-published,
            'interpretation':'Extent maximum at Eiger agrees with the independent metre-rounded summit height; this checks datum/unit plausibility, not a survey-accuracy certificate.'},
        'shapely_centre_checks':len(rows),'river_centres_with_ahn_nodata':int((mask & ~valid).sum()),
        'river_centres_with_ahn_measurements':int((mask & valid).sum()),
        'dates':dict(Counter(str(f['properties'].get('bronactualiteit')) for f in river)),
        'source_accuracy_field_bronnauwkeurigheid':dict(Counter(str(f['properties'].get('bronnauwkeurigheid')) for f in river)),
        'interpretation':'Different products and dates; overlap is reported without an invented acceptance threshold. AHN voids are not water measurements.',
        'coordinate_checks':'Centre-in-polygon is a rasterization self-check only. Cross-product shift residuals above supply registration evidence. CBS cell IDs and independent geometry bounds are also checked during staging.',
        'official_cbs_archive_matches_original_local_bytes':True,'official_cbs_sha256':official['sha256'],
        'sources':json.loads((root/'sources/river-source.json').read_text(encoding='utf-8'))}
    record(root/'registration-checks.json',result)
    print('Independent registration measures and Eiger height recorded:',shoreline_best,'Swiss horizontal residual m:',float(np.linalg.norm(residual)),flush=True)


def reexport(root):
    results={}
    for variant in ['initial','revision']:
        directory=root/'grindelwald'/variant
        original=json.loads((directory/'landscape/landscape.json').read_text(encoding='utf-8'))
        settings=original['import_settings']
        started=time.perf_counter()
        with patch.object(rasterio,'open',side_effect=AssertionError('Raw raster reopened during snapshot reuse')):
            prepared=PreparedScene.load(directory/'prepared')
            export_landscape(prepared,directory/'landscape-reexport',
                layout=LandscapeLayout(settings['quads_per_section'],settings['sections_per_component'],settings['components_x'],settings['components_y']),
                origin=original['geographic']['origin'],categorical_source='landcover',
                material_layers={m['class_id']:m['material_layer'] for m in original['weightmaps']})
        seconds=time.perf_counter()-started
        new=json.loads((directory/'landscape-reexport/landscape.json').read_text(encoding='utf-8'))
        if original!=new: raise AssertionError('Snapshot-only reexport changed the package manifest')
        for file,digest in original['files'].items():
            if _sha(directory/'landscape-reexport'/file)!=digest: raise AssertionError('Reexport changed encoded bytes')
        results[variant]={'snapshot_load_and_export_seconds':seconds,'manifest_exact':True,'all_image_bytes_exact':True,
            'raw_raster_open_forbidden_during_reuse':True,'independent_decode':package_check(prepared,directory/'landscape-reexport')}
    record(root/'landscape-reexport-checks.json',results)
    print('Both real Landscape packages reproduce exactly without opening source rasters',flush=True)


def sequence(directory,encode,output):
    """Read recorded frames; write derived products only to a separate new folder."""
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Report products must be outside the recorded capture directory')
    output.mkdir(parents=True,exist_ok=False)
    evidence=json.loads((directory/'evidence.json').read_text(encoding='utf-8'))
    paths=sorted((directory/'frames').glob('frame_*.png'))
    if len(paths)!=evidence['frames']: raise AssertionError('Missing/extra frame')
    expected=json.loads((directory.parent/'camera-path.json').read_text(encoding='utf-8'))
    if len(paths)!=len(expected['frames']): raise AssertionError('Incomplete camera path')
    previous=None; differences=[]; images=[]
    for i,path in enumerate(paths):
        if _sha(path)!=evidence['frame_sha256'][i]: raise AssertionError('Frame bytes changed')
        image=Image.open(path).convert('RGB')
        if image.size!=(1280,720): raise AssertionError('Wrong output dimensions')
        values=np.asarray(image)
        if previous is not None:
            differences.append(float(np.abs(values.astype(np.int16)-previous.astype(np.int16)).mean()))
        previous=values
        if encode: images.append(image)
        else: image.close()
    if encode:
        started=time.perf_counter()
        with (output/'flyover.webp').open('xb') as stream:
            images[0].save(stream,format='WEBP',save_all=True,append_images=images[1:],duration=1000/expected['fps'],loop=0,lossless=True)
        for image in images: image.close()
        encode_seconds=time.perf_counter()-started
    else: encode_seconds=evidence['encode_seconds']
    animation_path=output/'flyover.webp' if encode else directory/'flyover.webp'
    # Independently decode the entire final lossless animation against raw PNGs.
    with Image.open(animation_path) as animation:
        if animation.n_frames!=len(paths): raise AssertionError('Animation dropped frames')
        for i,path in enumerate(paths):
            animation.seek(i)
            with Image.open(path) as raw:
                if not np.array_equal(np.asarray(animation.convert('RGB')),np.asarray(raw.convert('RGB'))):
                    raise AssertionError('Animation encoding changed pixels')
    indices=[0,len(paths)//2,len(paths)-1]
    sheet=Image.new('RGB',(1280,720*3),'white')
    for n,index in enumerate(indices):
        with Image.open(paths[index]) as image: sheet.paste(image,(0,n*720))
    sheet.save(output/'contact-sheet.png')
    worst=int(np.argmax(differences))+1
    pair=Image.new('RGB',(1280,1440),'white')
    for n,index in enumerate([worst-1,worst]):
        with Image.open(paths[index]) as image: pair.paste(image,(0,n*720))
    pair.save(output/'largest-transition.png')
    result={'frames':len(paths),'all_recorded_png_hashes_match':True,'lossless_animation_all_frames_exact':True,
        'encode_seconds':encode_seconds,'animation_path':str(animation_path.resolve()),
        'animation_sha256':_sha(animation_path),'recorded_animation_sha256':_sha(directory/'flyover.webp'),
        'derived_directory':str(output.resolve()),'mean_absolute_rgb_change':{'min':min(differences),'median':float(np.median(differences)),
        'max':max(differences),'largest_transition_frame':worst},
        'interpretation':'Descriptive temporal measurements, not an invented flicker threshold or aesthetic certificate.'}
    record(output/'sequence-checks.json',result)
    return evidence,result


def report(root,output=None):
    output=root/'report-output' if output is None else output.resolve()
    if output.resolve()==root.resolve():raise ValueError('Choose a separate --output directory for the report')
    if not (root/'registration-checks.json').exists():
        raise ValueError('Run registration first; report generation never changes recorded inputs')
    output.mkdir(parents=True,exist_ok=False)
    implementation=implementation_identity()
    producer={'script_path':str(Path(__file__).resolve()),'script_sha256':_sha(Path(__file__)),
              'input_root':str(root.resolve()),'output_root':str(output.resolve()),
              'implementation':implementation}
    # Keep the exact producer alongside its output, including for later replays.
    (output/'c05_report.executed.py').write_bytes(Path(__file__).read_bytes())
    def link(path):return Path(os.path.relpath(path,output)).as_posix()
    rows=[]; sections=[]; summary=[]
    for case in ['rotterdam','grindelwald']:
        for variant in ['initial','revision']:
            directory=root/case/variant
            checks=json.loads((directory/'checks.json').read_text(encoding='utf-8'))
            qgis=json.loads((directory/'qgis/evidence.json').read_text(encoding='utf-8'))
            f3d,fseq=sequence(directory/'forge3d-render',False,output/case/variant/'forge3d-render')
            other,qseq=sequence(directory/'qgis-3d',True,output/case/variant/'qgis-3d')
            exit_state=json.loads((directory/'qgis-3d/exit-state.json').read_text(encoding='utf-8'))
            if exit_state['native_exit_code']!=0: raise AssertionError('QGIS did not exit successfully')
            if not qgis['display_pixels_exact'] or not all(d['valid_masks_exact'] and d['values_exact'] for d in qgis['differences_from_forge3d'].values()):
                raise AssertionError('Matched preparation differs')
            item={'case':case,'variant':variant,'preparation':checks,'qgis_preparation':qgis,
                  'forge3d_render':f3d,'qgis_render':other,'forge3d_sequence':fseq,'qgis_sequence':qseq,'qgis_exit':exit_state}
            summary.append(item)
            from datetime import datetime,timezone
            command_log=root/f'qgis-3d-{case}-{variant}.log'
            if command_log.stat().st_size!=0: raise AssertionError('QGIS command log needs explicit error review')
            end=datetime.fromisoformat(exit_state['completed_utc'].replace('Z','+00:00'))
            other['command_wall_seconds_from_log_and_exit']=end.timestamp()-command_log.stat().st_mtime
            other['command_timing_basis']='Empty log truncation at launch to observed native exit timestamp; includes startup/shutdown, not GPU timestamps.'
            for tool,prep,render,seq in [('Forge3D',checks,f3d,fseq),('QGIS',qgis,other,qseq)]:
                memory=render.get('peak_working_set_bytes',render.get('viewer_process_sum_of_kernel_peaks_bytes'))
                prep_seconds=prep['total_seconds']-prep['plan_render_seconds'] if tool=='QGIS' else prep['preparation_seconds']
                rows.append(f'<tr><td>{case} {variant}</td><td>{tool}</td><td>{prep_seconds:.3f}</td><td>{render["render_seconds"]:.3f}</td><td>{seq["encode_seconds"]:.3f}</td><td>{memory}</td></tr>')
            links=[]
            for tool,path,seq in [('Forge3D','forge3d-render',fseq),('QGIS','qgis-3d',qseq)]:
                movie=link(Path(seq['animation_path']));still=link(directory/path/'frames/frame_00000.png')
                derived=output/case/variant/path
                links.append(f'<figure><a href="{movie}"><img loading="lazy" src="{still}" alt="{tool} {case} {variant}"></a><figcaption>{tool}: <a href="{movie}">complete lossless flyover</a> · <a href="{link(derived/"contact-sheet.png")}">key views</a> · <a href="{link(derived/"largest-transition.png")}">largest measured transition</a></figcaption></figure>')
            sections.append(f'<section><h2>{case.title()} — {variant}</h2><p>Identical prepared measurements, masks, display mapping, geographic area, camera samples and 1280 × 720 output. Lighting and renderer implementations differ.</p><div class="pair">'+''.join(links)+'</div></section>')
    metadata={name:json.loads((root/name).read_text(encoding='utf-8-sig')) for name in
              ['cpu.json','runtime-identity.json','qgis-opengl.json','registration-checks.json','landscape-reexport-checks.json'] if (root/name).exists()}
    record(output/'summary.json',{'cases':summary,'metadata':metadata,'producer':producer,'implementation':implementation,'qualification':{'unreal_editor':'UNRUN','independent_physical_backend_portability':'ABSENT',
        'qgis_agreement':'Separately authored preparation using the same GDAL warp family; exact equality is agreement, not independent ground truth. See independent BRT/AHN displacement and SAC height checks.',
        'runtime':'Workspace Python 1.34.0 and installed prebuilt viewer 1.44.0 differ. Fresh runs observe loaded modules and code hashes during each frame run; no checkout native-build claim.',
        'operator':'Agent-authored scripts; no human expertise or hands-on timing measurement',
        'timing':'Shared-workstation wall times; setup failures retained; no isolated benchmark or superiority claim',
        'memory':'QGIS process kernel peak versus sum of viewer-process kernel peaks; Python encoding process reported separately in original evidence',
        'report_execution':'This producer verifies existing physical captures on read-only inputs. QGIS movies are re-encoded into this new report directory; original movies are never rewritten. Capture/preparation costs are historical per-phase measurements; encoding costs describe this report execution.',
        'checks':'All eight full sequences and both source/prepared routes are required by this report; no synthetic or fallback render'}})
    document='''<!doctype html><meta charset="utf-8"><title>C05 real-data evidence</title><style>
body{font:17px/1.6 system-ui;margin:40px auto;max-width:1440px;padding:0 24px;background:#f7f6f2;color:#202b30}
h1,h2{line-height:1.2}h1{font-size:42px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd0cb;text-align:left}section{margin:50px 0}a{color:#244c69}figcaption{font-size:14px}@media(max-width:850px){.pair{grid-template-columns:1fr}}
</style><h1>C05 · Real data, reusable scenes</h1><p>Rotterdam ground heights and population; Grindelwald/Eiger terrain and measured land cover. Initial and revised extents in both tools. Unreal Editor acceptance remains unrun.</p><p>Source identities, masks, numerical checks, full frame hashes and qualifications: <a href="summary.json">summary.json</a>. Reproduction: <a href="../../docs/guides/c05_real_data.md">guide</a>. No render is substituted for a failed backend.</p>'''
    document+=''.join(sections)
    document+='<section><h2>Independent geographic checks</h2><img src="river-registration.png" alt="BRT Nieuwe Maas boundaries over AHN/CBS display"><p>BRT water and AHN voids are compared across deliberate shifts, including agreement inside a shoreline corridor. All score margins are recorded: the small difference cannot rule out a westward half-pixel offset and supports only approximate one-pixel registration. The Eiger DEM maximum is compared in XY with an independent official swissNAMES3D summit point and in height with the SAC metre-rounded 3967 m reference. The observed horizontal residual is disclosed rather than converted into an invented accuracy certificate. <a href="registration-checks.json">All margins, coordinates and source identities</a>.</p></section>'
    document+='<section><h2>Observed local costs</h2><p>Seconds of wall time. Capture/preparation costs are reused from the recorded physical runs; QGIS encoding is performed again on copied frames into this report directory. Original movies are unchanged. Shared staging and acquisition are separate. Preparation includes saving and display assets; Landscape export timing is separate. These phase costs are not a new end-to-end benchmark; no speed or quality winner is inferred. Memory scopes differ, as labelled in summary.json.</p><table><tr><th>Scene</th><th>Tool</th><th>Prepare s</th><th>Render s</th><th>Encode s</th><th>Renderer memory bytes</th></tr>'+''.join(rows)+'</table></section>'
    document+='<section><h2>What the evidence does and does not establish</h2><p>All recorded heights, coverage masks and classes match between the preparation routes. Native count cells retain published values and disclosure semantics. Entire lossless animations decode exactly to their raw frames. Camera-pose parameters and physical scales are recorded. Readability and visual preference remain separate from numerical correctness; inspect the linked full sequences and largest transitions.</p><p>Lighting, tone mapping and LOD differ. These are agent-authored workflows on one shared Windows workstation, without a measured human authoring-time comparison, isolated performance benchmark or cross-platform certificate. The existing native bindings and installed viewer were used.</p><p>Both Swiss Landscape packages independently reconstruct within their encoding-derived bounds and reproduce byte-for-byte from saved preparation. Actual Unreal geometry, material assignment, editing, cracks and LOD still require the editor. No editor build is marked qualified.</p></section>'
    document=document.replace('href="../../docs/guides/c05_real_data.md"','href="'+link(Path(__file__).resolve().parents[1]/'docs/guides/c05_real_data.md')+'"')
    document=document.replace('src="river-registration.png"','src="'+link(root/'river-registration.png')+'"')
    document=document.replace('href="registration-checks.json"','href="'+link(root/'registration-checks.json')+'"')
    (output/'report.html').write_text(document,encoding='utf-8')
    if _sha(Path(__file__))!=producer['script_sha256']:
        raise RuntimeError('Report source changed during execution; retain this incomplete output and rerun the current script into a fresh directory')
    presentation(root,output)
    record(output/'producer.json',producer)
    print('Complete C05 evidence report written',output/'report.html',flush=True)


def presentation(root,output=None):
    """Add explanatory legends and an explicitly recorded visual review."""
    output=root if output is None else output
    producer_path=output/'producer.json'
    if producer_path.exists() and json.loads(producer_path.read_text(encoding='utf-8'))['script_sha256']!=_sha(Path(__file__)):
        raise ValueError('Report producer changed; regenerate into a fresh --output directory')
    document=(output/'report.html').read_text(encoding='utf-8')
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    for name in ['workflow-effort.json','cli-authoring/evidence.json','final-checks.json','registration-checks.json']:
        if (root/name).exists(): summary.setdefault('metadata',{})[name]=json.loads((root/name).read_text(encoding='utf-8'))
    common=json.loads((root/'common/provenance.json').read_text(encoding='utf-8'))
    for case in ['rotterdam','grindelwald']:
        for variant in ['initial','revision']:
            directory=root/case/variant
            recipe=json.loads((directory/'recipe.json').read_text(encoding='utf-8'))
            marker=f'<!--C05-LEGEND {case} {variant}-->'
            if marker in document: continue
            if case=='rotterdam':
                prepared=PreparedScene.load(directory/'prepared')
                layer=prepared.layers['population']; values=layer.data[layer.valid_mask]
                entries=[]
                for value in sorted({int(values.min()),int(np.median(values)),int(values.max())}):
                    style=recipe['display_styles']['population']
                    if 'classes' in style: color=style['classes'][str(value)][:3]
                    else:
                        low,high=style['range']; fraction=(np.log1p(value)-np.log1p(low))/(np.log1p(high)-np.log1p(low)) if style.get('scale')=='log1p' else (value-low)/(high-low)
                        colors=np.asarray(style['colors'],dtype=float)
                        color=np.rint(colors[0]+np.clip(fraction,0,1)*(colors[1]-colors[0])).astype(int).tolist()[:3]
                    entries.append(f'<span style="display:inline-block;margin:0 18px 8px 0"><i style="display:inline-block;width:24px;height:16px;background:rgb({",".join(map(str,color))})"></i> {value:,} people / original 100 m cell</span>')
                entries += ['<span style="margin-right:18px"><i style="display:inline-block;width:24px;height:16px;background:rgb(193,192,179)"></i> Absent CBS feature</span>',
                            '<span><i style="display:inline-block;width:24px;height:16px;background:rgb(125,88,148)"></i> Present suppressed/unknown CBS value</span>']
                text='AHN4 ground heights, NAP metres; 2020–2022 acquisition, 0.5 m IDW product delivered at 10 m by the service default nearest-neighbour sampling. Corrected CBS 2024 v2; published counts rounded to five. A reusable log1p range [0, 2010] supports unseen counts; endpoint colours saturate outside that range without altering analysis values. Legend ticks are the observed minimum, median and maximum. The prepared population_state layer preserves absent, suppressed/unknown and published states separately.'
            else:
                entries=[]
                colors=recipe['display_styles']['landcover']['classes']
                for key,name in common['tlm']['classes'].items():
                    entries.append(f'<span style="display:inline-block;margin:0 18px 8px 0"><i style="display:inline-block;width:24px;height:16px;background:rgb({",".join(map(str,colors[key][:3]))})"></i> {escape(name)}</span>')
                text='swissALTI3D 2025 terrain, LN02 metres; measured swissTLM3D 2026-02 footprints. Original class names and feature survey dates are preserved.'
            legend=marker+'<p>'+text+'</p><div>'+''.join(entries)+'</div><p>Swiss gray hatching: unknown or ambiguous thematic data. Rotterdam gray: absent feature; purple: present suppressed/unknown value. Blue hatching takes visual priority for missing terrain height; inspect the retained state layer there. The declared terrain fill is not a measured water level. Legend colours precede terrain lighting.</p>'
            header=f'<h2>{case.title()} — {variant}</h2>'
            if document.count(header)!=1: raise AssertionError('Report section is missing or duplicated')
            document=document.replace(header,header+legend)
    review_path=root/'visual-review.json'
    if review_path.exists() and '<!--C05-VISUAL-REVIEW-->' not in document:
        review=json.loads(review_path.read_text(encoding='utf-8'))
        for path,digest in review['inspected_artifact_sha256'].items():
            if _sha(root/path)!=digest: raise AssertionError('Visual review belongs to changed images')
        paragraphs=''.join('<p><strong>'+escape(key)+':</strong> '+escape(value)+'</p>' for key,value in review['observations'].items())
        document+='<!--C05-VISUAL-REVIEW--><section><h2>Recorded visual review</h2>'+paragraphs+'</section>'
        summary['visual_review']=review
    record(output/'summary.json',summary)
    (output/'report.html').write_text(document,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['registration','reexport','report','presentation']);parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,help='Separate report directory; must be new for report. Recorded captures are read-only inputs.')
    args=parser.parse_args()
    if args.operation in {'report','presentation'}:
        {'report':report,'presentation':presentation}[args.operation](args.directory.resolve(),args.output.resolve() if args.output else None)
    else:
        if args.output:parser.error('--output applies only to report/presentation')
        {'registration':registration,'reexport':reexport}[args.operation](args.directory.resolve())
