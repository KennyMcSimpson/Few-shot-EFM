#!/usr/bin/env bash
set -euo pipefail

# Dataset choices:
# SEED SEED-IV SEED-VIG BCI-IV-2A SHU EEGMAT HMC SHHS Sleep-EDF Siena TUAB TUEV Things-EEG

data_root="${1:-${DATA_ROOT:-}}"
dataset="${2:-${DATASET:-SEED}}"
split_mode="${3:-${SPLIT_MODE:-cross}}"

if [[ -z "$data_root" ]]; then
  echo "Usage: bash preprocessing/json_process.sh <data_root> [dataset] [cross|multi]" >&2
  echo "Example: bash preprocessing/json_process.sh /data/eeg TUEV cross" >&2
  exit 2
fi

case "$split_mode" in
  cross)
    json_process_script="preprocessing/${dataset}/cross_json_process.py"
    ;;
  multi)
    json_process_script="preprocessing/${dataset}/multi_json_process.py"
    ;;
  *)
    echo "Unknown split mode: $split_mode. Expected 'cross' or 'multi'." >&2
    exit 2
    ;;
esac

if [[ ! -f "$json_process_script" ]]; then
  echo "Dataset split script not found: $json_process_script" >&2
  exit 2
fi

python "$json_process_script" "$data_root"
