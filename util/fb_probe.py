# FB2 functional-block profiling utilities. Lightweight diagnostics only.
from __future__ import annotations
import csv, hashlib, json, math, os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional
import torch
from .fb_registry import CANONICAL_BLOCKS, classify_param_name
from .module_b_signal_alignment import (
    InputSideLoRAResidual,
    LoRAConv1d1x1,
    infer_input_channels,
    module_b_metadata,
)

def _ensure_dir(path: str):
    if path: os.makedirs(path, exist_ok=True)
def _safe_scalar(v):
    if isinstance(v, torch.Tensor): return v.detach().cpu().item() if v.numel()==1 else str(tuple(v.shape))
    try:
        import numpy as np
        if isinstance(v, np.generic): return v.item()
    except Exception: pass
    if isinstance(v, bool): return int(v)
    if isinstance(v, (str,int,float)) or v is None: return v
    return str(v)
def _write_csv(path: str, rows: List[Dict[str, Any]]):
    _ensure_dir(os.path.dirname(path))
    if not rows: return None
    keys=[]; seen=set()
    for row in rows:
        for k in row.keys():
            if k not in seen: seen.add(k); keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for row in rows: w.writerow({k:_safe_scalar(row.get(k,"")) for k in keys})
    return path
def _append_csv_row(path: str, row: Dict[str, Any]):
    _ensure_dir(os.path.dirname(path)); clean={k:_safe_scalar(v) for k,v in row.items()}; header=not os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(clean.keys()))
        if header: w.writeheader()
        w.writerow(clean)
def _dataset_len(ds):
    try: return len(ds) if ds is not None else 0
    except Exception: return -1
def _read_json_records(path: str, max_records: int):
    if not path or not os.path.exists(path): return []
    try:
        with open(path,"r",encoding="utf-8") as f: obj=json.load(f)
    except Exception: return []
    if isinstance(obj,list): records=obj
    elif isinstance(obj,dict):
        records=[]
        for key in ["data","samples","records","items","annotations"]:
            if isinstance(obj.get(key),list): records=obj[key]; break
        if not records:
            for v in obj.values():
                if isinstance(v,list): records.extend(v)
    else: records=[]
    return records[:max(0,int(max_records))]
def _first_label(obj: Any):
    if isinstance(obj,dict):
        for key in ["label","labels","y","target","class","event","event_label"]:
            if key in obj:
                v=obj[key]; v=v[0] if isinstance(v,list) and v else v
                try: return int(v)
                except Exception: return str(v)
        for v in obj.values():
            got=_first_label(v)
            if got is not None: return got
    elif isinstance(obj,(list,tuple)):
        if len(obj)>=2:
            for cand in [obj[-1],obj[1]]:
                try: return int(cand)
                except Exception: pass
        for v in obj:
            got=_first_label(v)
            if got is not None: return got
    return None
def _strings(obj: Any):
    out=[]
    if isinstance(obj,str): out.append(obj)
    elif isinstance(obj,dict):
        for v in obj.values(): out.extend(_strings(v))
    elif isinstance(obj,(list,tuple)):
        for v in obj: out.extend(_strings(v))
    return out
def _record_id(obj: Any):
    ss=_strings(obj); path_like=[s for s in ss if ("\\" in s or "/" in s or s.endswith((".npy",".npz",".pkl",".edf",".pt",".pth")))]
    if path_like: return path_like[0].replace("\\","/").lower()
    if ss: return hashlib.md5("|".join(ss[:5]).encode("utf-8",errors="ignore")).hexdigest()
    try: return hashlib.md5(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception: return ""
def save_split_integrity(args, dataset_train=None, dataset_val=None, dataset_test=None, dataset_info=None):
    if not bool(getattr(args,"fb_enable",False)) or not bool(getattr(args,"fb_split_check",False)): return None
    diag_dir=os.path.join(args.output_dir,"diagnostics"); _ensure_dir(diag_dir)
    try: root=dataset_info.get("root",{}).get(getattr(args,"subject_mod",""),"")
    except Exception: root=""
    split_to_ds={"train":dataset_train,"val":dataset_val,"test":dataset_test}; split_rows=[]; class_rows=[]; ids_by_split={}; max_records=int(getattr(args,"fb_integrity_max_json_records",200000))
    for split,ds in split_to_ds.items():
        json_path=os.path.join(root,f"{split}.json") if root else ""; records=_read_json_records(json_path,max_records)
        labels=[]; ids=[]
        for r in records:
            # Test identities may be inspected for overlap, but test labels
            # remain sealed until the one final evaluation.
            if split!="test":
                lab=_first_label(r)
                if lab is not None: labels.append(lab)
            rid=_record_id(r)
            if rid: ids.append(rid)
        ids_by_split[split]=set(ids); counts=Counter(labels)
        for lab,cnt in sorted(counts.items(),key=lambda x:str(x[0])): class_rows.append({"split":split,"label":lab,"count":cnt,"source":"json"})
        split_rows.append({"split":split,"dataset_len":_dataset_len(ds),"json_path":json_path,"json_records_read":len(records),"json_label_count_total":"" if split=="test" else sum(counts.values()),"json_unique_ids":len(ids_by_split[split]),"class_count_summary":"withheld_until_final_evaluation" if split=="test" else ";".join([f"{k}:{v}" for k,v in sorted(counts.items(),key=lambda x:str(x[0]))])})
    train_ids=ids_by_split.get("train",set())
    for row in split_rows:
        ids=ids_by_split.get(row["split"],set())
        row["overlap_with_train_ids"]=0 if row["split"]=="train" else (len(train_ids.intersection(ids)) if train_ids and ids else -1)
        row["data_leak_check_status"]="ok_no_exact_id_overlap" if row["overlap_with_train_ids"]==0 else ("unknown_no_ids" if row["overlap_with_train_ids"]<0 else "warning_overlap_found")
    _write_csv(os.path.join(diag_dir,"fb_split_integrity.csv"),split_rows); _write_csv(os.path.join(diag_dir,"fb_class_counts.csv"),class_rows)
    print(f"[FB2] split integrity saved to: {diag_dir}"); return os.path.join(diag_dir,"fb_split_integrity.csv")
def save_block_registry(args, model):
    if not bool(getattr(args,"fb_enable",False)): return None
    diag_dir=os.path.join(args.output_dir,"diagnostics"); _ensure_dir(diag_dir); model_name=getattr(args,"model_name","")
    map_rows=[]; acc=defaultdict(lambda:{"n_tensors":0,"n_trainable_tensors":0,"numel":0,"trainable_numel":0})
    for name,p in model.named_parameters():
        primary,hits=classify_param_name(model_name,name); numel=int(p.numel()); tr=int(bool(p.requires_grad))
        map_rows.append({"model":model_name,"param_name":name,"primary_block":primary,"matched_blocks":";".join(hits),"shape":str(tuple(p.shape)),"numel":numel,"requires_grad":tr})
        a=acc[primary]; a["n_tensors"]+=1; a["numel"]+=numel
        if tr: a["n_trainable_tensors"]+=1; a["trainable_numel"]+=numel
    summary=[]
    for block in CANONICAL_BLOCKS+["other"]:
        a=acc.get(block)
        if a: summary.append({"model":model_name,"block":block,**a,"trainable_ratio":a["trainable_numel"]/max(a["numel"],1)})
    _write_csv(os.path.join(diag_dir,"fb_block_param_map.csv"),map_rows); _write_csv(os.path.join(diag_dir,"fb_block_trainable_summary.csv"),summary)
    print(f"[FB2] block registry saved to: {diag_dir}"); return os.path.join(diag_dir,"fb_block_param_map.csv")
def _state_cpu(model): return {k:v.detach().cpu() for k,v in model.state_dict().items() if torch.is_tensor(v)}
def _rank_stats(delta: torch.Tensor, enabled: bool, max_numel: int):
    if not enabled or max_numel<=0 or delta.ndim!=2 or delta.numel()>max_numel: return {}
    try:
        s=torch.linalg.svdvals(delta.float()); energy=s*s; total=float(energy.sum().item())
        if total<=1e-20: return {"eff_rank":0.0,"rank90":0}
        p=energy/energy.sum(); entropy=float(-(p*(p+1e-12).log()).sum().item()); cumsum=torch.cumsum(energy,dim=0)/energy.sum()
        return {"eff_rank":math.exp(entropy),"rank90":int((cumsum<0.90).sum().item()+1)}
    except Exception: return {}
_METRIC_KEYS=["train_eval_balanced_accuracy","val_balanced_accuracy","test_balanced_accuracy","train_test_gap_balanced_accuracy","val_accuracy","test_accuracy","val_selection_class0","val_selection_class2","test_selection_class0","test_selection_class2","val_worst_class_recall","test_worst_class_recall","val_recall_std","test_recall_std","val_loss","test_loss"]
def _module_b_metadata(args):
    return module_b_metadata(args=args)
def _sample_norm(x: torch.Tensor):
    dims=tuple(range(1,x.ndim))
    return torch.sqrt(torch.clamp(x.float().pow(2).sum(dim=dims), min=0.0))
def _top_channel_energy_text(channel_energy: Optional[torch.Tensor], top_n: int=5) -> str:
    if channel_energy is None or channel_energy.numel()==0:
        return ""
    k=min(int(top_n), int(channel_energy.numel()))
    vals, idx=torch.topk(channel_energy.float(), k=k)
    return ";".join([f"{int(i)}:{float(v):.6g}" for v,i in zip(vals.tolist(),idx.tolist())])
def _module_b_bridge_adapters(model) -> List[tuple[str, Any]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRAConv1d1x1)]
def _top_vector_text(values: Optional[torch.Tensor], top_n: int=5) -> str:
    if values is None or values.numel()==0:
        return ""
    k=min(int(top_n), int(values.numel()))
    vals, idx=torch.topk(values.float(), k=k)
    return ";".join([f"{int(i)}:{float(v):.6g}" for v,i in zip(vals.tolist(),idx.tolist())])
def _top_matrix_pairs_text(weight: torch.Tensor, top_n: int=8) -> str:
    if weight.numel()==0:
        return ""
    flat=weight.detach().float().abs().flatten()
    k=min(int(top_n), int(flat.numel()))
    vals, idx=torch.topk(flat, k=k)
    cols=int(weight.shape[1]) if weight.ndim==2 and weight.shape[1] else 1
    pairs=[]
    for v,i in zip(vals.tolist(),idx.tolist()):
        out_idx=int(i)//cols
        in_idx=int(i)%cols
        pairs.append(f"{out_idx}:{in_idx}:{float(v):.6g}")
    return ";".join(pairs)
def _matrix_rank_stats(weight: torch.Tensor) -> Dict[str, Any]:
    if weight.ndim!=2 or weight.numel()==0:
        return {"effective_rank":"","rank90":"","top_singular_values":""}
    try:
        s=torch.linalg.svdvals(weight.float())
        energy=s*s
        total=float(energy.sum().item())
        if total<=1e-20:
            return {"effective_rank":0.0,"rank90":0,"top_singular_values":""}
        p=energy/energy.sum()
        entropy=float(-(p*(p+1e-12).log()).sum().item())
        cumsum=torch.cumsum(energy,dim=0)/energy.sum()
        top=";".join(f"{float(v):.6g}" for v in s[: min(8, int(s.numel()))].tolist())
        return {"effective_rank":math.exp(entropy),"rank90":int((cumsum<0.90).sum().item()+1),"top_singular_values":top}
    except Exception:
        return {"effective_rank":"","rank90":"","top_singular_values":""}
def _module_b_matrix_row(args, site: str, name: str, adapter: Any) -> Dict[str, Any]:
    weight=adapter.effective_delta_weight().detach().float().cpu()
    sq=weight.pow(2)
    total=float(sq.sum().item())
    diag_n=min(int(weight.shape[0]), int(weight.shape[1])) if weight.ndim==2 else 0
    diag_energy=float(torch.diagonal(sq[:diag_n, :diag_n]).sum().item()) if diag_n>0 else 0.0
    off_energy=max(total-diag_energy, 0.0)
    row={
        "site":site,
        "module_name":name,
        "model":getattr(args,"model_name",""),
        "dataset":getattr(args,"dataset",""),
        "subject_mod":getattr(args,"subject_mod",""),
        "k_shot":getattr(args,"k_shot",""),
        "seed":getattr(args,"seed",""),
        "rank":int(getattr(adapter,"r",-1)),
        "alpha":float(getattr(adapter,"alpha",0.0)),
        "alpha_over_rank":float(getattr(adapter,"scaling",0.0)),
        "runtime_scale":float(getattr(adapter,"lora_runtime_scale",1.0)),
        "out_channels":int(weight.shape[0]) if weight.ndim==2 else "",
        "in_channels":int(weight.shape[1]) if weight.ndim==2 else "",
        "delta_fro_norm":float(torch.linalg.vector_norm(weight).item()) if weight.numel() else 0.0,
        "delta_spectral_norm":float(torch.linalg.matrix_norm(weight,ord=2).item()) if weight.ndim==2 and weight.numel() else 0.0,
        "diagonal_energy_ratio":diag_energy/(total+1e-12),
        "off_diagonal_energy_ratio":off_energy/(total+1e-12),
        "top_input_channels":_top_vector_text(sq.sum(dim=0) if weight.ndim==2 else None),
        "top_output_channels":_top_vector_text(sq.sum(dim=1) if weight.ndim==2 else None),
        "top_channel_pairs":_top_matrix_pairs_text(weight),
    }
    row.update(_matrix_rank_stats(weight))
    row.update(_module_b_metadata(args))
    return row
def save_module_b_config(args, model):
    meta=_module_b_metadata(args)
    if not meta["module_b_is_active"]:
        return None
    diag_dir=os.path.join(args.output_dir,"diagnostics")
    _ensure_dir(diag_dir)
    bridge_adapters=_module_b_bridge_adapters(model)
    input_adapter=getattr(model,"input_side_lora",None)
    bridge_base_frozen=""
    if bridge_adapters:
        bridge_base_frozen=int(all(not p.requires_grad for _,m in bridge_adapters for p in m.base.parameters()))
    payload={
        **meta,
        "model":getattr(args,"model_name",""),
        "dataset":getattr(args,"dataset",""),
        "subject_mod":getattr(args,"subject_mod",""),
        "k_shot":getattr(args,"k_shot",""),
        "seed":getattr(args,"seed",""),
        "lora_rank":getattr(args,"lora_rank",""),
        "lora_alpha":getattr(args,"lora_alpha",""),
        "alpha_over_rank":float(getattr(args,"lora_alpha",0.0))/max(float(getattr(args,"lora_rank",1)),1.0),
        "lora_dropout":getattr(args,"lora_dropout",""),
        "input_channels":infer_input_channels(model),
        "input_side_lora_present":int(isinstance(input_adapter,InputSideLoRAResidual)),
        "bridge_lora_count":len(bridge_adapters),
        "bridge_lora_modules":";".join(name for name,_ in bridge_adapters),
        "bridge_in_channels":";".join(str(int(m.in_channels)) for _,m in bridge_adapters),
        "bridge_out_channels":";".join(str(int(m.out_channels)) for _,m in bridge_adapters),
        "wrapped_base_conv_frozen":bridge_base_frozen,
        "trainable_param_count":sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "total_param_count":sum(int(p.numel()) for p in model.parameters()),
    }
    path=os.path.join(diag_dir,"module_b_config.json")
    with open(path,"w",encoding="utf-8") as f:
        json.dump({k:_safe_scalar(v) for k,v in payload.items()},f,indent=2,ensure_ascii=False)
    print(f"[FB2] Module B config saved to: {path}")
    return path
def save_module_b_matrix_summary(args, model):
    meta=_module_b_metadata(args)
    if not meta["module_b_is_active"]:
        return None
    rows=[]
    input_adapter=getattr(model,"input_side_lora",None)
    if isinstance(input_adapter,InputSideLoRAResidual):
        rows.append(_module_b_matrix_row(args,"input_side","input_side_lora",input_adapter))
    for name,adapter in _module_b_bridge_adapters(model):
        rows.append(_module_b_matrix_row(args,"bridge",name,adapter))
    if not rows:
        return None
    path=os.path.join(args.output_dir,"diagnostics","module_b_matrix_summary.csv")
    _write_csv(path,rows)
    print(f"[FB2] Module B matrix summary saved to: {path}")
    return path
def save_signal_alignment_probe(args, model, data_loader, device, split: str="val_lifecycle_selected", selection_row: Optional[Dict[str, Any]]=None):
    max_batches=int(getattr(args,"fb_signal_probe_batches",4))
    if max_batches<=0 or data_loader is None:
        return None
    meta=_module_b_metadata(args)
    if not meta["module_b_is_active"]:
        return None
    adapter=getattr(model,"input_side_lora",None)
    if adapter is None or not hasattr(adapter,"delta"):
        return None

    diag_dir=os.path.join(args.output_dir,"diagnostics")
    csv_path=os.path.join(diag_dir,"signal_alignment_probe.csv")
    was_training=bool(model.training)
    ratios=[]; input_norms=[]; delta_norms=[]; delta_abs=[]; labels=[]
    channel_energy_sum=None; channel_delta_abs_sum=None; channel_input_abs_sum=None
    channel_batches=0; batches_seen=0; samples_seen=0
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx,batch in enumerate(data_loader):
                if batch_idx>=max_batches:
                    break
                x=batch[0]
                target=batch[-1]
                if str(getattr(args,"norm_method",""))=="mv":
                    x=x.float().to(device,non_blocking=True)*float(getattr(args,"mv_norm_value",1.0))
                else:
                    x=x.float().to(device,non_blocking=True)
                if x.dim()!=3:
                    continue
                delta=adapter.delta(x)
                if delta.shape!=x.shape:
                    continue
                in_norm=_sample_norm(x)
                de_norm=_sample_norm(delta)
                ratio=de_norm/(in_norm+1e-12)
                ratios.append(ratio.detach().cpu())
                input_norms.append(in_norm.detach().cpu())
                delta_norms.append(de_norm.detach().cpu())
                delta_abs.append(delta.detach().float().abs().flatten(1).mean(dim=1).cpu())
                try:
                    y=target.detach().view(-1).cpu().long()
                    if y.numel()==ratio.numel():
                        labels.append(y)
                except Exception:
                    pass
                ce=delta.detach().float().pow(2).mean(dim=(0,2)).cpu()
                channel_energy_sum=ce if channel_energy_sum is None else channel_energy_sum+ce
                cda=delta.detach().float().abs().mean(dim=(0,2)).cpu()
                cia=x.detach().float().abs().mean(dim=(0,2)).cpu()
                channel_delta_abs_sum=cda if channel_delta_abs_sum is None else channel_delta_abs_sum+cda
                channel_input_abs_sum=cia if channel_input_abs_sum is None else channel_input_abs_sum+cia
                channel_batches+=1
                batches_seen+=1
                samples_seen+=int(x.shape[0])
    finally:
        model.train(was_training)
    if not ratios:
        return None

    all_ratio=torch.cat(ratios)
    all_input=torch.cat(input_norms)
    all_delta=torch.cat(delta_norms)
    all_abs=torch.cat(delta_abs)
    finite=torch.isfinite(all_ratio)
    all_ratio=all_ratio[finite]; all_input=all_input[finite]; all_delta=all_delta[finite]; all_abs=all_abs[finite]
    if all_ratio.numel()==0:
        return None
    all_labels=torch.cat(labels)[finite] if labels and torch.cat(labels).numel()==finite.numel() else None
    channel_energy=(channel_energy_sum/float(max(channel_batches,1))) if channel_energy_sum is not None else None
    channel_delta_abs=(channel_delta_abs_sum/float(max(channel_batches,1))) if channel_delta_abs_sum is not None else None
    channel_input_abs=(channel_input_abs_sum/float(max(channel_batches,1))) if channel_input_abs_sum is not None else None
    channel_delta_input_ratio=(channel_delta_abs/(channel_input_abs+1e-12)) if channel_delta_abs is not None and channel_input_abs is not None else None
    selection_row=selection_row or {}

    def make_row(row_type: str, mask: Optional[torch.Tensor]=None, class_id: str=""):
        if mask is None:
            mask=torch.ones_like(all_ratio,dtype=torch.bool)
        if int(mask.sum().item())<=0:
            return None
        r=all_ratio[mask]; inn=all_input[mask]; den=all_delta[mask]; ab=all_abs[mask]
        row={
            "split":split,
            "row_type":row_type,
            "class_id":class_id,
            "n_samples":int(mask.sum().item()),
            "probe_batches":int(batches_seen),
            "probe_samples_seen":int(samples_seen),
            "delta_input_ratio_mean":float(r.mean().item()),
            "delta_input_ratio_median":float(r.median().item()),
            "delta_input_ratio_p95":float(torch.quantile(r.float(),0.95).item()) if r.numel()>1 else float(r.max().item()),
            "delta_input_ratio_max":float(r.max().item()),
            "input_norm_mean":float(inn.mean().item()),
            "delta_norm_mean":float(den.mean().item()),
            "delta_abs_mean":float(ab.mean().item()),
            "top_channel_delta_energy":_top_channel_energy_text(channel_energy),
            "per_channel_delta_abs_mean":_top_vector_text(channel_delta_abs, top_n=9999),
            "per_channel_delta_input_ratio":_top_vector_text(channel_delta_input_ratio, top_n=9999),
            "top_channel_delta_input_ratio":_top_vector_text(channel_delta_input_ratio, top_n=5),
            "fb_recipe":getattr(args,"fb_resolved_recipe",getattr(args,"fb_recipe","")),
            "dataset":getattr(args,"dataset",""),
            "subject_mod":getattr(args,"subject_mod",""),
            "k_shot":getattr(args,"k_shot",""),
            "seed":getattr(args,"seed",""),
        }
        row.update(meta)
        for key in ("start_epoch","end_epoch","length","selection_score","val_balanced_accuracy","test_balanced_accuracy"):
            if key in selection_row:
                row[f"selection_{key}"]=selection_row[key]
        return row

    rows=[]
    overall=make_row("overall")
    if overall is not None:
        rows.append(overall)
    if all_labels is not None:
        for cls in sorted(set(int(x) for x in all_labels.tolist())):
            row=make_row("class", all_labels==cls, str(cls))
            if row is not None:
                rows.append(row)
    _write_csv(csv_path,rows)
    print(f"[FB2] signal alignment probe saved to: {csv_path}")
    return csv_path
def save_block_delta_summary(args, model, init_state, trainable_names: Iterable[str] | None, epoch: int, metrics_row: Optional[Dict[str, Any]]=None):
    if init_state is None: return None
    if not (bool(getattr(args,"fb_enable",False)) and bool(getattr(args,"fb_probe",False))): return None
    diag_dir=os.path.join(args.output_dir,"diagnostics"); _ensure_dir(diag_dir); csv_path=os.path.join(diag_dir,"fb_block_delta_summary.csv")
    model_name=getattr(args,"model_name",""); trainable=set(trainable_names or []); current=_state_cpu(model); rank_enabled=bool(getattr(args,"fb_rank_diag",False)); max_numel=int(getattr(args,"fb_svd_max_numel",0))
    acc=defaultdict(lambda:{"n_tensors":0,"n_trainable_tensors":0,"numel":0,"trainable_numel":0,"delta_sq":0.0,"base_sq":0.0,"param_sq":0.0,"rel_sum":0.0,"rel_max":-1.0,"top_param":"","top_param_delta_norm":0.0,"rank_n":0,"eff_rank_sum":0.0,"rank90_sum":0.0})
    for name,cur in current.items():
        base=init_state.get(name)
        if base is None or cur.shape!=base.shape or not torch.is_floating_point(cur): continue
        c=cur.float(); b=base.float(); d=c-b; delta_sq=float((d*d).sum().item()); base_sq=float((b*b).sum().item()); param_sq=float((c*c).sum().item()); delta_norm=math.sqrt(max(delta_sq,0.0)); rel=delta_norm/(math.sqrt(max(base_sq,0.0))+1e-12)
        primary,hits=classify_param_name(model_name,name); is_trainable=int(name in trainable); a=acc[primary]
        a["n_tensors"]+=1; a["numel"]+=int(cur.numel()); a["delta_sq"]+=delta_sq; a["base_sq"]+=base_sq; a["param_sq"]+=param_sq; a["rel_sum"]+=rel
        if is_trainable: a["n_trainable_tensors"]+=1; a["trainable_numel"]+=int(cur.numel())
        if rel>a["rel_max"]: a["rel_max"]=rel; a["top_param"]=name; a["top_param_delta_norm"]=delta_norm
        rs=_rank_stats(d,rank_enabled,max_numel)
        if rs: a["rank_n"]+=1; a["eff_rank_sum"]+=float(rs.get("eff_rank",0.0)); a["rank90_sum"]+=float(rs.get("rank90",0.0))
    metric_part={}; metrics_row=metrics_row or {}
    for k in _METRIC_KEYS:
        if k in metrics_row: metric_part[k]=metrics_row[k]
    for block in CANONICAL_BLOCKS+["other"]:
        a=acc.get(block)
        if not a or a["n_tensors"]<=0: continue
        delta_norm_l2=math.sqrt(max(a["delta_sq"],0.0)); base_norm_l2=math.sqrt(max(a["base_sq"],0.0))
        row={"epoch":int(epoch),"model":model_name,"run_tag":getattr(args,"run_tag",""),"fb_recipe":getattr(args,"fb_resolved_recipe",getattr(args,"fb_recipe","")),"block":block,"n_tensors":a["n_tensors"],"n_trainable_tensors":a["n_trainable_tensors"],"numel":a["numel"],"trainable_numel":a["trainable_numel"],"delta_norm_l2":delta_norm_l2,"base_norm_l2":base_norm_l2,"param_norm_l2":math.sqrt(max(a["param_sq"],0.0)),"relative_delta_norm_l2":delta_norm_l2/(base_norm_l2+1e-12),"mean_relative_delta_norm":a["rel_sum"]/max(a["n_tensors"],1),"max_relative_delta_norm":a["rel_max"],"top_param":a["top_param"],"top_param_delta_norm":a["top_param_delta_norm"],"rank_diag_tensors":a["rank_n"],"mean_update_eff_rank":(a["eff_rank_sum"]/a["rank_n"]) if a["rank_n"] else "","mean_update_rank90":(a["rank90_sum"]/a["rank_n"]) if a["rank_n"] else ""}
        row.update(metric_part); _append_csv_row(csv_path,row)
    print(f"[FB2] block delta summary appended: {csv_path}"); return csv_path
