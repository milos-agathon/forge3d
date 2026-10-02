from pathlib import Path
import argparse,json,numpy as np
from scripts.nephele_evidence_report import _delta_e,_roi_crop,ssim
p=argparse.ArgumentParser(); p.add_argument("artifact",type=Path); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
load=lambda n:np.load(a.artifact/n,allow_pickle=False)
ref,rt=load("reference-rgb.npy"),load("realtime-rgb.npy"); terrain=load("terrain-mask.npy").astype(bool); shadow=load("cloud-shadow-mask.npy").astype(bool)&terrain; roi=load("godray-roi-mask.npy").astype(bool)
d=_delta_e(rt,ref); r,t=_roi_crop(roi,ref,rt); t2,o=_roi_crop(roi,rt,load("terrain-occlusion-disabled-rgb.npy"))
m={"sky_cloud_delta_e_pass_fraction":float(np.mean(d[load("sky-cloud-mask.npy").astype(bool)]<2.5)),"godray_roi_ssim":float(ssim(r,t,data_range=255.0)),"cloud_shadow_delta_e_pass_fraction":float(np.mean(d[shadow]<2.0)),"medium_ablation_changed_fraction":float(np.mean(_delta_e(load("medium-disabled-rgb.npy"),rt)[terrain]>5.0)),"terrain_occlusion_ablation_ssim":float(ssim(t2,o,data_range=255.0))}
m["gate3_pass"]=m["sky_cloud_delta_e_pass_fraction"]>=.95 and m["godray_roi_ssim"]>.95
m["gate4_pass"]=m["cloud_shadow_delta_e_pass_fraction"]>=.95 and m["medium_ablation_changed_fraction"]>=.10 and m["terrain_occlusion_ablation_ssim"]<.80
m["reference_in_scatter_shadow_mean_rgb"]=load("reference-in-scatter.npy")[shadow].mean(axis=0).tolist()
m["realtime_in_scatter_shadow_mean_rgb"]=load("in-scatter.npy")[shadow].mean(axis=0).tolist()
m["physical_acceptance"]=False; m["reason"]="Diagnostic visual recomputation; the full local physical verifier separately binds identity and all six gates."
a.output.write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8"); print(json.dumps(m,indent=2))
