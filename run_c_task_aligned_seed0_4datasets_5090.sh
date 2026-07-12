#!/usr/bin/env bash
set -uo pipefail

REPO="/home/mingkai/eeg_remote/Few-shot-EFM"
PY="${PYTHON_BIN:-/home/mingkai/miniforge3/envs/fewshot-efm/bin/python}"
RUN_ROOT="${1:?run root required}"
WORKER_ID="${2:?worker id 0, 1, or 2 required}"
MODE="${3:-full}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

case "$WORKER_ID" in
  0) CPU_RANGE="8-12" ;;
  1) CPU_RANGE="13-17" ;;
  2) CPU_RANGE="18-23" ;;
  *) echo "invalid worker id: $WORKER_ID" >&2; exit 2 ;;
esac

case "$MODE" in
  full|preflight) ;;
  *) echo "invalid mode: $MODE (expected full or preflight)" >&2; exit 2 ;;
esac

cd "$REPO"
mkdir -p "$RUN_ROOT/collections" "$RUN_ROOT/runner_logs"

STATUS_FILE="$RUN_ROOT/run_status.csv"
LOCK_FILE="$RUN_ROOT/run_status.lock"
GIT_COMMIT="$(git rev-parse HEAD)"

append_status() {
  local row="$1"
  (
    flock -x 9
    if [[ ! -s "$STATUS_FILE" ]]; then
      echo 'lane,model,seed,dataset,mode,run_tag,status,return_code,selected_modules,evidence_strength,branch_count,preflight_seconds,started_at,finished_at,git_commit' > "$STATUS_FILE"
    fi
    echo "$row" >> "$STATUS_FILE"
  ) 9>>"$LOCK_FILE"
}

dataset_tag() {
  case "$1" in
    "SEED-IV") echo "seediv" ;;
    "Sleep-EDF") echo "sleepedf" ;;
    "BCI-IV-2A") echo "bci2a" ;;
    "TUEV") echo "tuev" ;;
  esac
}

model_tag() {
  case "$1" in
    "EEGPT") echo "eeg" ;;
    "BIOT") echo "bi" ;;
    "LaBraM") echo "la" ;;
    "CBraMod") echo "cb" ;;
    "Gram") echo "gr" ;;
    "CSBrain") echo "cs" ;;
  esac
}

read_decision() {
  local run_tag="$1"
  local decision
  decision="$(find -L finetuning_results -type f -path "*/${run_tag}/module_c_preflight_decision.json" -print -quit 2>/dev/null || true)"
  if [[ -z "$decision" ]]; then
    echo 'NA NA NA NA'
    return
  fi
  "$PY" - "$decision" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
selected = ";".join(payload.get("selected_modules", ())) or "NA"
strength = str(payload.get("final_evidence_strength", payload.get("primary_evidence_strength", "NA")))
runtime = payload.get("runtime", {})
print(selected, strength, runtime.get("branch_count", "NA"), runtime.get("total_seconds", "NA"))
PY
}

run_one() {
  local dataset="$1"
  local model="$2"
  local dtag mtag run_tag collect_dir cmd_log stdout_log stderr_log
  local started finished rc status selected evidence branches seconds
  local -a cmd extra

  dtag="$(dataset_tag "$dataset")"
  mtag="$(model_tag "$model")"
  run_tag="${mtag}_c_task_aligned_${dtag}_s0_${RUN_ROOT##*/}"
  collect_dir="$RUN_ROOT/collections/$run_tag"
  cmd_log="$RUN_ROOT/runner_logs/${run_tag}.cmd.txt"
  stdout_log="$RUN_ROOT/runner_logs/${run_tag}.stdout.log"
  stderr_log="$RUN_ROOT/runner_logs/${run_tag}.stderr.log"
  extra=()

  if [[ "$model" == "EEGPT" ]]; then
    extra+=(--sampling_rate 256)
  elif [[ "$model" == "Gram" ]]; then
    extra+=(--gram_ckpt checkpoints/base.pth --gram_vqgan_ckpt checkpoints/base_class_quantization.pth --gram_root external/Gram)
  fi

  cmd=(
    "$PY" run_finetuning.py
    --finetune_mod lora
    --lora_base_update full
    --lora_rank 4
    --lora_alpha 8
    --lora_dropout 0.1
    --lora_target module_c
    --module_c_candidates B,D,E
    --module_c_preflight_train_batches 0
    --module_c_preflight_val_batches 0
    --model_name "$model"
    --subject_mod fewshot
    --dataset "$dataset"
    --task_mod Classification
    --k_shot 0.05
    --epochs 50
    --batch_size 16
    --lr 1e-4
    --weight_decay 0.05
    --num_workers 4
    --loader_prefetch_factor 2
    --seed 0
    --loss_type sqrt_balanced_ce
    --best_metric balanced_accuracy
    --selection_worst_alpha 0.35
    --selection_min02_alpha 0.40
    --selection_std_gamma 0.16
    --monitor_dynamics
    --eval_train_set
    --diag_freq 5
    --save_epoch_ckpt_freq 999
    --short_output_tag_only
    --run_tag "$run_tag"
    --no_auto_resume
    --fb_enable
    --fb_probe
    --fb_recipe manual
    --fb_split_check
    --fb_collect
    --fb_collect_name "$collect_dir"
    --module_b_sites both
    --module_e_mode dynamic_pressure_gate
    --module_e_warmup_steps 0
    --adaptive_swa_eval
    --adaptive_swa_epoch_min 1
    --adaptive_swa_epoch_max 50
    --adaptive_swa_min_len 3
    --adaptive_swa_max_len 8
    --adaptive_swa_stride 1
    --adaptive_swa_select_metric selection_bacc_min02_std
    --adaptive_swa_profile generic
    --adaptive_swa_balance_lambda 0.10
    --adaptive_swa_hard_classes 0,2
    --adaptive_swa_hard_floor 0.05
    --adaptive_swa_hard_floor_lambda 0.20
    --adaptive_swa_std_lambda 0.04
    --adaptive_swa_tie_mode hard_stable
    --adaptive_swa_tie_eps 0.002
    "${extra[@]}"
  )
  if [[ "$MODE" == "preflight" ]]; then
    cmd+=(--module_c_preflight_only)
  fi

  mkdir -p "$collect_dir"
  printf '%q ' "${cmd[@]}" > "$cmd_log"
  printf '\n' >> "$cmd_log"
  echo "[lane $WORKER_ID] START model=$model dataset=$dataset mode=$MODE"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] taskset -c %q ' "$CPU_RANGE"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi

  started="$(date -Is)"
  taskset -c "$CPU_RANGE" "${cmd[@]}" > "$stdout_log" 2> "$stderr_log"
  rc=$?
  finished="$(date -Is)"
  if [[ "$rc" == "0" ]]; then status="done"; else status="failed"; fi
  read -r selected evidence branches seconds < <(read_decision "$run_tag")
  append_status "$WORKER_ID,$model,0,$dataset,$MODE,$run_tag,$status,$rc,$selected,$evidence,$branches,$seconds,$started,$finished,$GIT_COMMIT"
  echo "[lane $WORKER_ID] FINISH model=$model dataset=$dataset mode=$MODE status=$status rc=$rc selected=$selected evidence=$evidence"
}

models=(EEGPT BIOT LaBraM CBraMod Gram CSBrain)
datasets=(TUEV Sleep-EDF BCI-IV-2A SEED-IV)

index=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if (( index % 3 == WORKER_ID )); then
      run_one "$dataset" "$model"
    fi
    ((index+=1))
  done
done

echo "[lane $WORKER_ID] QUEUE_FINISHED mode=$MODE"
