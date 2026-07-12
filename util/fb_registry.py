# FB2 functional block registry for AdaBrain-Bench EEG foundation models.
from __future__ import annotations
import re
from typing import Dict, Iterable, List, Sequence, Tuple
CANONICAL_BLOCKS = ["input_front", "spatial", "temporal", "spectral", "semantic", "mixing", "readout", "restoration"]
BLOCK_ALIASES = {"front":"input_front","input":"input_front","signal":"input_front","signal_alignment":"input_front","spatial":"spatial","topology":"spatial","temporal":"temporal","time":"temporal","spectral":"spectral","freq":"spectral","frequency":"spectral","amplitude":"spectral","semantic":"semantic","sem":"semantic","ffn":"semantic","mlp":"semantic","mixing":"mixing","fusion":"mixing","cross":"mixing","readout":"readout","calibration":"readout","cal":"readout","head":"readout","restoration":"restoration","denoise":"restoration"}
MODEL_BLOCK_PATTERNS: Dict[str, Dict[str, Sequence[str]]] = {
"BIOT":{"input_front":["embedding","token","tok","segment","conv"],"spatial":["channel","ch_embedding","chan_embed"],"temporal":["pos","position","relative"],"spectral":["fft","fcn","freq","spect"],"semantic":["ffn","mlp","linear1","linear2","w1","w2","fc1","fc2"],"mixing":["attn","attention","qkv","query","key","value","proj","to_q","to_k","to_v","to_out"],"readout":["task_head","classifier","head"]},
"LaBraM":{"input_front":["patch_embed","patch_embedding","temporal_conv","conv"],"spatial":["spatial","channel","ch_embed","chan"],"temporal":["temporal","time","pos_embed","position"],"spectral":["spectrum","spectral","fourier","vq","quant"],"semantic":["mlp","ffn","fc1","fc2","intermediate","output.dense"],"mixing":["attn","attention","qkv","query","key","value","proj"],"readout":["task_head","classifier","head"]},
"EEGPT":{"input_front":["chan_conv","input_adapter","patch_embed","summary_token","bridge"],"spatial":["spatial","channel","chan","spatial_filter","asg"],"temporal":["temporal","time","pos","position"],"spectral":["spectral","freq","fourier"],"semantic":["mlp","ffn","fc1","fc2","encoder","predictor","reconstructor"],"mixing":["attn","attention","qkv","query","key","value","proj","alignment"],"readout":["task_head","classifier","head","calib"]},
"CBraMod":{"input_front":["patch_embedding","patch_embed","time_conv","input_projection"],"spatial":["self_attn_s","s_attention","spatial","acpe"],"temporal":["self_attn_t","t_attention","temporal","time"],"spectral":["spectral","frequency","fft","freq"],"semantic":["linear1","linear2","ffn","mlp","feed_forward"],"mixing":["criss","cross","fusion","attn","attention","proj"],"readout":["task_head","classifier","head","reconstruction_head"]},
"CSBrain":{"input_front":["chan_conv","patch_embedding","preliminary","embedding"],"spatial":["brain","region","inter_region","spatial"],"temporal":["temporal","window","inter_window","temembed","time"],"spectral":["spectral","freq","fft"],"semantic":["linear1","linear2","ffn","mlp","encoder"],"mixing":["cst","ssa","fusion","attn","attention","inter_"],"readout":["task_head","classifier","head"]},
"Gram":{"input_front":["patch_embed","patch_embedding","input_embed"],"spatial":["ch_embed","channel","spatial","pos_embed"],"temporal":["temporal","time","pos_embed","tokenizer"],"spectral":["spectral","mimic","freq","fourier"],"semantic":["blocks","encoder","mlp","ffn","fc1","fc2"],"mixing":["fusion","layer_fusion","attn","attention","qkv","proj"],"readout":["head","cls","classifier","task_head"],"restoration":["decoder","vqgan","quantize","codebook","base_class","tokenizer_decoder"]},
"NeurIPT":{"input_front":["input_adapter","embed","point","patch"],"spatial":["electrode","3d","iilp","lobe","spatial"],"temporal":["temporal","tsa","time"],"spectral":["amplitude","aamp","spectral","freq"],"semantic":["expert","ffn","mlp","shared_expert"],"mixing":["pmoe","router","gate","attn","attention","merge","fusion"],"readout":["classification_head","classifier","head","task_head"]}}
GENERIC_PATTERNS = {"readout":["task_head","classification_head","classifier"],"semantic":["mlp","ffn","fc1","fc2","linear1","linear2"],"mixing":["attn","attention","qkv","query","key","value"],"input_front":["patch_embed","patch_embedding","chan_conv","input_side_lora","input_side","signal_align","front_align","signal_alignment","channel_adapter","input_adapter"]}
PRIMARY_PRIORITY = ["readout","restoration","spatial","temporal","spectral","mixing","semantic","input_front"]
def normalize_model_name(model_name: str) -> str:
    name=str(model_name or "").strip(); low=name.lower()
    for known in MODEL_BLOCK_PATTERNS:
        if low==known.lower(): return known
    return name
def normalize_block(block: str) -> str:
    key=str(block or "").strip().lower().replace("-","_").replace("/","_"); return BLOCK_ALIASES.get(key,key)
def parse_blocks(blocks: str | Iterable[str] | None) -> List[str]:
    if blocks is None: return []
    raw=[x.strip() for x in blocks.replace(";",",").split(",")] if isinstance(blocks,str) else [str(x).strip() for x in blocks]
    out=[]
    for item in raw:
        if item:
            nb=normalize_block(item)
            if nb not in out: out.append(nb)
    return out
def get_block_patterns(model_name: str, block: str) -> Sequence[str]:
    model=normalize_model_name(model_name); nb=normalize_block(block); patterns=[]
    patterns.extend(MODEL_BLOCK_PATTERNS.get(model,{}).get(nb,[])); patterns.extend(GENERIC_PATTERNS.get(nb,[]))
    seen=set(); uniq=[]
    for p in patterns:
        lp=p.lower()
        if lp and lp not in seen: seen.add(lp); uniq.append(lp)
    return uniq
def _pattern_matches(model_name: str, block: str, pattern: str, name: str) -> bool:
    if normalize_model_name(model_name) == "BIOT" and normalize_block(block) == "mixing":
        segments = tuple(part for part in re.split(r"[.\[\]/\\]+", name) if part)
        pattern_segments = tuple(part for part in pattern.split(".") if part)
        width = len(pattern_segments)
        return width > 0 and any(segments[index:index + width] == pattern_segments for index in range(len(segments) - width + 1))
    return bool(pattern and pattern in name)
def classify_param_name(model_name: str, param_name: str) -> Tuple[str, List[str]]:
    name=str(param_name or "").lower(); hits=[]
    for block in CANONICAL_BLOCKS:
        if any(_pattern_matches(model_name, block, p, name) for p in get_block_patterns(model_name, block)): hits.append(block)
    if not hits: return "other", ["other"]
    for block in PRIMARY_PRIORITY:
        if block in hits: return block, hits
    return hits[0], hits
def registry_snapshot(model_name: str) -> Dict[str, Sequence[str]]:
    return {block:list(get_block_patterns(model_name, block)) for block in CANONICAL_BLOCKS}
