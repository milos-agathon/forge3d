from pathlib import Path
import hashlib,json,numpy as np
from scripts import nephele_evidence_report as e
from scripts import nephele_b1_diagnostic as b1
repo=Path.cwd(); artifact=repo/'.tmp/nephele-completion/round1-final-local';head=b1._git(repo,'rev-parse','HEAD')
runtime=e._object(artifact/'installed-wheel-runtime.json')
identity=e._verify_identity(artifact,head,repo,Path(runtime['installed_native_path']),('Windows','X64','windows-nvidia-vulkan'))
context=identity['context'];manifest=e._object(artifact/'fixture-manifest.json')
bundle=e._verify_fixture_manifest(artifact,manifest,repo,context['fixture_commit'])
_,_,camera=e._verify_reference_provenance(artifact,manifest,repo,context['fixture_commit'])
results={}
checks={
 'gate1':lambda:e._gate1(artifact,head,repo,manifest['scene_inputs']['medium']),
 'gate2':lambda:e._gate2(artifact,head,repo),
 'gate6':lambda:e._gate6(artifact,head,context,identity['adapter'],bundle,camera),
}
for name,check in checks.items():
    try:results[name]={'status':'passed','metrics':check()}
    except e.EvidenceError as exc:results[name]={'status':'failed','error':str(exc)}
visual=e._object(artifact/'visual-metrics.json')
for name in ['gate3','gate4']:
    results[name]={'status':'passed' if visual[name.replace('gate','gate')+'_pass'] else 'failed','metrics':visual}
static=e._object(artifact/'gate5-static.json');analyzer=repo/'scripts/nephele_shader_analyzer.py'
recomputed=e.analyze_shaders(repo);recomputed['tool_sha256']=e._sha256(analyzer)
source_matches=hashlib.sha256(e._tracked_blob(repo,head,analyzer)).hexdigest()==e._sha256(analyzer)
static_pass=(source_matches and static==recomputed and not recomputed['unresolved_source_expressions']
 and all(any(call.startswith(wrapper+'<-') for call in recomputed['rust_shader_wrapper_invocations']) for wrapper in recomputed['rust_shader_construction_wrappers'])
 and recomputed['naga_validated_assemblies']==len(recomputed['resolved_source_sha256'])
 and len(recomputed['executed_assembled_naga_contracts'])==recomputed['naga_validated_assemblies']
 and recomputed['compute_entry_texture_sample_compare_calls']==0 and recomputed['stale_disabled_shadow_comments']==0)
roi=e._mask(artifact/'godray-roi-mask.npy',(64,64));shaft=e._mask(artifact/'shaft-mask.npy',(64,64))&roi
reference=e._array(artifact/'reference-terrain-slice.npy');actual=e._array(artifact/'realtime-termination-slice.npy')
if reference.shape!=(64,64) or actual.shape!=(64,64) or not shaft.any():raise ValueError('invalid Gate 5 depth population')
agreement=float(np.mean(np.abs(reference[shaft]-actual[shaft])<=1.0))
results['gate5']={'status':'passed' if static_pass and agreement>=.99 else 'failed','metrics':{
 'tracked_exhaustive_analysis_matches':static_pass,'naga_validated_assemblies':recomputed['naga_validated_assemblies'],
 'ridgeline_within_one_slice_fraction':agreement}}
result={'schema':'forge3d.nephele.bounded_round_gate_diagnostics/1','source_revision':head,
 'physical_acceptance':False,'reason':'Independent local measurements; six-case physical acceptance and the full verifier retain their own failures.',
 'gates':results,'producer_sha256':e._sha256(repo/'.tmp/nephele-completion/measure_all_gates.py')}
(artifact/'independent-gates.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({name:{'status':record['status'],'metrics':record.get('metrics')} for name,record in results.items()},indent=2))
