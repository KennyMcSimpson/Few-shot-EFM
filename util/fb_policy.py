# FB2 policy layer. Conservative: profiling first, no automatic LoRA takeover yet.
from __future__ import annotations
import json, os
from typing import Dict
from .fb_registry import parse_blocks, registry_snapshot
from .module_b_signal_alignment import module_b_metadata
from .module_c_lora_search import module_c_metadata, parse_module_ids
from .module_d_semantic_refinement import module_d_metadata
from .module_e_structural_routing import module_e_metadata
def add_fb_args(parser):
    parser.add_argument("--fb_enable", action="store_true", default=False, help="Enable functional-block framework metadata/profiling.")
    parser.add_argument("--fb_probe", action="store_true", default=False, help="Save block-level profiling CSVs. No model core change.")
    parser.add_argument("--fb_recipe", default="profile", type=str, choices=["profile","probe_only","manual","sem_lif","sig_align","cal_readout","str_mix","gram_diag","csb_diag","auto"], help="High-level functional recipe. FB2 records it for profiling; it does not inject LoRA automatically.")
    parser.add_argument("--fb_blocks", default="", type=str, help="Comma-separated functional blocks to emphasize in logs.")
    parser.add_argument("--fb_split_check", action="store_true", default=False, help="Write split/class-count/integrity CSVs.")
    parser.add_argument("--fb_integrity_max_json_records", default=200000, type=int)
    parser.add_argument("--fb_rank_diag", action="store_true", default=False, help="Optional SVD rank diagnostics for small update matrices.")
    parser.add_argument("--fb_svd_max_numel", default=0, type=int, help="0 disables SVD rank diagnostics.")
    parser.add_argument("--fb_signal_probe_batches", default=4, type=int, help="Max validation batches for signal-alignment input-side LoRA probe. 0 disables.")
    parser.add_argument("--fb_collect", action="store_true", default=False, help="Copy small logs/csvs into a short collected folder.")
    parser.add_argument("--fb_collect_name", default="", type=str, help="Collected folder name, e.g. col_fb2s.")
    parser.add_argument("--module_c_enable", action="store_true", default=False, help="Enable Module C LoRA-search metadata. This does not inject LoRA automatically.")
    parser.add_argument("--module_c_candidates", default="B,D,E", type=str, help="Comma-separated candidate adaptation modules for Module C subset selection.")
    parser.add_argument("--module_c_selected", default="", type=str, help="Comma-separated selected modules from a Module C policy/probe run.")
    parser.set_defaults(module_c_preflight=True)
    parser.add_argument("--module_c_no_preflight", action="store_false", dest="module_c_preflight", help="Disable automatic zero-update Module C preflight selection; requires --module_c_selected for lora_target=module_c.")
    parser.add_argument("--module_c_preflight_train_batches", default=0, type=int, help="Module C train preflight batch cap. <=0 scans the full train split; positive values are debug caps.")
    parser.add_argument("--module_c_preflight_val_batches", default=0, type=int, help="Module C validation preflight batch cap. <=0 scans the full validation split; positive values are debug caps.")
    parser.add_argument("--module_c_preflight_hard_k", default=0, type=int, help="Deprecated compatibility flag; Module C now defines focus classes by validation burden >= 1/C.")
    parser.add_argument("--module_c_preflight_min_score", default=0.0, type=float, help="Minimum positive RGFS residual-burden marginal relief required to add a Module C action. Formal default is 0.")
    parser.add_argument("--module_c_preflight_margin", default=0.0, type=float, help="RGFS tie tolerance. Formal default is 0, so complexity only breaks exact ties.")
    parser.add_argument("--module_c_preflight_svd_max_numel", default=1000000, type=int, help="Maximum gradient tensor size used for Module C low-rank SVD fit; larger tensors are skipped.")
    parser.add_argument("--module_c_preflight_max_profile_classes", default=0, type=int, help="Maximum validation classes used for Module C interaction profiles. <=0 uses all validation classes.")
    parser.add_argument("--module_c_preflight_dropout", default=0.0, type=float, help="Dropout for temporary Module C probe LoRA branches.")
    parser.add_argument("--module_c_probe_head_steps", default=3, type=int, help="Temporary head-only calibration steps on the disposable Module C probe model before RGFS gradients are measured.")
    parser.add_argument("--module_c_probe_head_lr", default=1e-3, type=float, help="Learning rate for temporary Module C probe-head calibration.")
    parser.add_argument("--module_c_rgfs_confidence_scale", default=0.0, type=float, help="Optional finite-sample cosine shrinkage scale. Formal full-split Module C default is 0.")
    parser.add_argument("--module_c_rgfs_harm_threshold", default=0.0, type=float, help="Positive reliable harm threshold that blocks an action on focus classes. Formal default is 0.")
    parser.add_argument("--module_c_rgfs_focus_ratio", default=1.0, type=float, help="Class burden ratio relative to uniform that defines RGFS focus classes. Formal default 1.0 means burden >= 1/C.")
    return parser
MODEL_DEFAULT_RECIPE={"BIOT":"sem_lif","LaBraM":"sem_lif","EEGPT":"sig_align","CBraMod":"str_mix","Gram":"gram_diag","CSBrain":"csb_diag","NeurIPT":"probe_only"}
MODULE_B_METADATA_KEYS=(
    "module_b_current","module_b_role","module_b_is_active",
    "module_b_is_pure_isolation","module_b_sites",
    "module_b_input_side_active","module_b_bridge_active",
)
MODULE_D_METADATA_KEYS=("module_d_current","module_d_role","module_d_is_active","module_d_touches_semantic_ffn","module_d_is_pure_isolation","module_d_is_composite","module_d_variant","module_d_reference_metric","module_d_attribution_note")
MODULE_E_METADATA_KEYS=("module_e_current","module_e_role","module_e_is_active","module_e_is_pure_isolation","module_e_is_composite","module_e_variant","module_e_target_blocks","module_e_reference_metrics","module_e_attribution_note")
MODULE_C_METADATA_KEYS=("module_c_current","module_c_role","module_c_is_active","module_c_candidates","module_c_selected_modules","module_c_selection_rule","module_c_no_qv_baseline")
MODULE_METADATA_FLAT_KEYS=MODULE_B_METADATA_KEYS+MODULE_D_METADATA_KEYS+MODULE_E_METADATA_KEYS+MODULE_C_METADATA_KEYS
def resolve_functional_args(args):
    if str(getattr(args,"lora_target","") or "").lower()=="module_c":
        args.module_c_enable=True
    if bool(getattr(args,"module_c_enable",False)):
        args.module_c_resolved_candidates=",".join(parse_module_ids(getattr(args,"module_c_candidates","B,D,E")))
        args.module_c_resolved_selected=",".join(parse_module_ids(getattr(args,"module_c_selected","")))
    if not bool(getattr(args,"fb_enable",False)): return args
    recipe=str(getattr(args,"fb_recipe","profile") or "profile")
    if recipe=="auto": recipe=MODEL_DEFAULT_RECIPE.get(str(getattr(args,"model_name","")),"profile")
    args.fb_resolved_recipe=recipe; args.fb_resolved_blocks=",".join(parse_blocks(getattr(args,"fb_blocks","")))
    if bool(getattr(args,"fb_probe",False)):
        args.monitor_dynamics=True; args.eval_train_set=True
    return args
def module_metadata_dict(args)->Dict:
    target=str(getattr(args,"lora_target",""))
    base_update=str(getattr(args,"lora_base_update",""))
    c_selected=getattr(args,"module_c_resolved_selected",getattr(args,"module_c_selected",""))
    selected_modules=parse_module_ids(c_selected)
    if target.lower()=="module_c":
        b_target="signal_align" if "B" in selected_modules else ""
        d_target="semantic" if "D" in selected_modules else ""
        e_target="struct_mix" if "E" in selected_modules else ""
    else:
        b_target=d_target=e_target=target
    b_meta=module_b_metadata(lora_target=b_target,lora_base_update=base_update)
    d_meta=module_d_metadata(lora_target=d_target,lora_base_update=base_update)
    e_meta=module_e_metadata(lora_target=e_target,lora_base_update=base_update)
    c_meta=module_c_metadata(args=args,selected_modules=parse_module_ids(c_selected))
    metadata={}
    for source,keys in ((b_meta,MODULE_B_METADATA_KEYS),(d_meta,MODULE_D_METADATA_KEYS),(e_meta,MODULE_E_METADATA_KEYS),(c_meta,MODULE_C_METADATA_KEYS)):
        for key in keys:
            metadata[key]=source[key]
    metadata["module_c_recipe"]=c_meta["module_c_recipe"]
    return metadata
def resolved_recipe_dict(args)->Dict:
    model_name=str(getattr(args,"model_name",""))
    target=str(getattr(args,"lora_target",""))
    base_update=str(getattr(args,"lora_base_update",""))
    metadata=module_metadata_dict(args)
    recipe={"framework":"FB2 functional-block profiling","model_name":model_name,"fb_enable":int(bool(getattr(args,"fb_enable",False))),"fb_probe":int(bool(getattr(args,"fb_probe",False))),"fb_recipe":str(getattr(args,"fb_recipe","")),"fb_resolved_recipe":str(getattr(args,"fb_resolved_recipe",getattr(args,"fb_recipe",""))),"fb_blocks":str(getattr(args,"fb_blocks","")),"fb_resolved_blocks":str(getattr(args,"fb_resolved_blocks","")),"selection_rule":"profiling_only_no_test_selection","test_used_for_selection":0,"score_is_validation_only":1,"dataset":str(getattr(args,"dataset","")),"subject_mod":str(getattr(args,"subject_mod","")),"k_shot":getattr(args,"k_shot",""),"seed":getattr(args,"seed",""),"epochs":getattr(args,"epochs",""),"finetune_mod":str(getattr(args,"finetune_mod","")),"loss_type":str(getattr(args,"loss_type","")),"lora_target":target,"lora_base_update":base_update}
    for key in MODULE_B_METADATA_KEYS+MODULE_D_METADATA_KEYS+MODULE_E_METADATA_KEYS:
        recipe[key]=metadata[key]
    recipe["registry_patterns"]=registry_snapshot(model_name)
    for key in MODULE_C_METADATA_KEYS:
        recipe[key]=metadata[key]
    recipe["module_c_recipe"]=metadata["module_c_recipe"]
    return recipe
def write_resolved_recipe(args, output_dir):
    if not bool(getattr(args,"fb_enable",False)): return None
    os.makedirs(output_dir,exist_ok=True); path=os.path.join(output_dir,"fb_resolved_recipe.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(resolved_recipe_dict(args),f,ensure_ascii=False,indent=2)
    print(f"[FB2] resolved recipe saved to: {path}"); return path
