# FB2 collected output helper.
from __future__ import annotations
import os, shutil
from pathlib import Path
from .fb_policy import MODULE_METADATA_FLAT_KEYS, module_metadata_dict
SMALL_FILES=["log.txt","args.json","config.json","fb_resolved_recipe.json"]
DIAG_FILES=[
    "epoch_metrics.csv",
    "per_class_recall.csv",
    "confusion_matrix.csv",
    "fb_split_integrity.csv",
    "fb_class_counts.csv",
    "fb_block_param_map.csv",
    "fb_block_trainable_summary.csv",
    "fb_block_delta_summary.csv",
    "weight_delta_summary.csv",
    "weight_delta_spectrum.csv",
    "boundary_anchor_eval.csv",
    "snapshot_ensemble.csv",
    "adaptive_swa_windows.csv",
    "adaptive_swa_eval.csv",
    "adaptive_swa_forgetting_summary.csv",
    "adaptive_swa_forgetting_by_class.csv",
    "signal_alignment_probe.csv",
    "module_d_sbr_eval.csv",
    "module_e_coverage_audit.csv",
    "module_e_structural_pressure.csv",
    "module_e_pressure_targets.csv",
    "module_e_dynamic_pressure.csv",
]
def _copy(src,dst):
    if os.path.exists(src): os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy2(src,dst); return True
    return False
def run_info_rows(args,model,tag,seed,out_dir):
    target=str(getattr(args,"lora_target",""))
    base_update=str(getattr(args,"lora_base_update",""))
    metadata=module_metadata_dict(args)
    rows=[("model",model),("tag",tag),("seed",seed),("output_dir",out_dir),("dataset",getattr(args,"dataset","")),("subject_mod",getattr(args,"subject_mod","")),("k_shot",getattr(args,"k_shot","")),("epochs",getattr(args,"epochs","")),("finetune_mod",getattr(args,"finetune_mod","")),("lora_target",target),("lora_base_update",base_update)]
    rows.extend((key,metadata[key]) for key in MODULE_METADATA_FLAT_KEYS)
    rows.extend([("fb_recipe",getattr(args,"fb_resolved_recipe",getattr(args,"fb_recipe",""))),("score_is_validation_only",1),("test_used_for_selection",0)])
    return rows
def collect_outputs_if_requested(args):
    if not bool(getattr(args,"fb_collect",False)): return None
    name=str(getattr(args,"fb_collect_name","") or "").strip()
    if not name: return None
    out_dir=str(getattr(args,"output_dir","") or "")
    if not out_dir or not os.path.exists(out_dir): return None
    tag=str(getattr(args,"run_tag","") or Path(out_dir).name); model=str(getattr(args,"model_name","model")); seed=str(getattr(args,"seed","s")); dst_root=os.path.join(name,f"{model}_{tag}_s{seed}"); os.makedirs(dst_root,exist_ok=True)
    copied=[]
    for fn in SMALL_FILES:
        if _copy(os.path.join(out_dir,fn),os.path.join(dst_root,fn)): copied.append(fn)
    diag_src=os.path.join(out_dir,"diagnostics"); diag_dst=os.path.join(dst_root,"diagnostics")
    for fn in DIAG_FILES:
        if _copy(os.path.join(diag_src,fn),os.path.join(diag_dst,fn)): copied.append("diagnostics/"+fn)
    with open(os.path.join(dst_root,"run_info.csv"),"w",encoding="utf-8") as f:
        f.write("key,value\n")
        for k,v in run_info_rows(args,model,tag,seed,out_dir): f.write(f"{k},{v}\n")
    print(f"[FB2] collected {len(copied)} files to: {dst_root}"); return dst_root
