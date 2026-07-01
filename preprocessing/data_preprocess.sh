#!/usr/bin/env bash
set -euo pipefail

# Dataset choices:
# SEED SEED-IV SEED-VIG BCI-IV-2A SHU EEGMAT HMC SHHS Sleep-EDF Siena TUAB TUEV Things-EEG

data_root="${1:-${DATA_ROOT:-}}"
dataset="${2:-${DATASET:-SEED}}"

if [[ -z "$data_root" ]]; then
  echo "Usage: bash preprocessing/data_preprocess.sh <data_root> [dataset]" >&2
  echo "Example: bash preprocessing/data_preprocess.sh /data/eeg TUEV" >&2
  exit 2
fi

data_process_script="preprocessing/${dataset}/data_process.py"
if [[ ! -f "$data_process_script" ]]; then
  echo "Dataset preprocessing script not found: $data_process_script" >&2
  exit 2
fi

python "$data_process_script" "$data_root"
