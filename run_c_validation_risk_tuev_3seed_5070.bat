@echo off
setlocal enabledelayedexpansion

cd /d D:\code\codepy\EEG_code\AdaBrain-Bench-main

rem Formal Module C validation-risk TUEV 3-seed runner.
rem Module C always selects a nonempty B/D/E subset from support and
rem validation data before the final LoRA model is created.

set "PY=C:\Users\Kenny\.conda\envs\EEG\python.exe"
set "DATASET=TUEV"
set "KSHOT=0.05"
set "EPOCHS=30"
set "BATCH_SIZE=16"
set "LR=1e-4"
set "WEIGHT_DECAY=0.05"
set "NUM_WORKERS=4"
set "LOADER_PREFETCH=2"
set "SEEDS=0 1 2"

set "C_PREFLIGHT_TRAIN_BATCHES=0"
set "C_PREFLIGHT_VAL_BATCHES=0"

rem Seed-specific fewshot Full-FT references are used only if present.
rem Keep this at 0 for the cleanest 3-seed protocol. Set to 1 only if you
rem intentionally want seed1/2 Module-D SBR diagnostics to reuse seed0 refs.
set "USE_SEED0_REF_FOR_ALL=0"
set "ABD_REF_COL=col_abd_metric_20260625_215700"

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_MODULE_LOADING=LAZY"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "OMP_NUM_THREADS=4"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "COL_C=col_c_validation_risk_tuev_3seed_5070_%STAMP%"

set "DRYRUN=0"
if /I "%~1"=="--dry-run" set "DRYRUN=1"

set "SELECT_ARGS=--best_metric balanced_accuracy --selection_worst_alpha 0.35 --selection_min02_alpha 0.40 --selection_std_gamma 0.16"
set "DIAG_ARGS=--monitor_dynamics --eval_train_set --diag_freq 5 --save_epoch_ckpt_freq 999"
set "ASWA_ARGS=--adaptive_swa_eval --adaptive_swa_epoch_min 1 --adaptive_swa_epoch_max 30 --adaptive_swa_min_len 3 --adaptive_swa_max_len 8 --adaptive_swa_stride 1 --adaptive_swa_select_metric selection_bacc_min02_std --adaptive_swa_profile generic --adaptive_swa_balance_lambda 0.10 --adaptive_swa_hard_classes 0,2 --adaptive_swa_hard_floor 0.05 --adaptive_swa_hard_floor_lambda 0.20 --adaptive_swa_std_lambda 0.04 --adaptive_swa_tie_mode hard_stable --adaptive_swa_tie_eps 0.002"
set "LORA_ARGS=--finetune_mod lora --lora_base_update full --lora_rank 4 --lora_alpha 8 --lora_dropout 0.1"
set "C_ARGS=--lora_target module_c --module_c_candidates B,D,E --module_c_preflight_train_batches %C_PREFLIGHT_TRAIN_BATCHES% --module_c_preflight_val_batches %C_PREFLIGHT_VAL_BATCHES%"
set "FB_ARGS=--fb_enable --fb_probe --fb_recipe manual --fb_split_check --fb_collect --fb_collect_name %COL_C%"
set "MODULE_BE_ARGS=--module_b_sites both --module_e_mode dynamic_pressure_gate --module_e_warmup_steps 0"

echo [RUN] Formal Module C validation-risk TUEV 3-seed runner
echo [RUN] Root: %CD%
echo [RUN] Dataset: %DATASET%
echo [RUN] GPU: RTX 5070 profile, CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [RUN] seeds=%SEEDS%
echo [RUN] num_workers=%NUM_WORKERS% loader_prefetch_factor=%LOADER_PREFETCH% batch_size=%BATCH_SIZE%
echo [RUN] Module C preflight train/val batch caps=%C_PREFLIGHT_TRAIN_BATCHES%/%C_PREFLIGHT_VAL_BATCHES% ^(0/0 means full train/val split^)
echo [RUN] Module C candidates=B,D,E; the selector never emits an empty set.
echo [RUN] Collection: %COL_C%
if "%DRYRUN%"=="1" echo [RUN] DRY RUN: commands will be printed only.
echo.

if "%DRYRUN%"=="0" (
  if exist "%COL_C%" (
    echo [ERROR] Existing collection folder found: %COL_C%
    exit /b 1
  )
  if exist "%COL_C%.zip" (
    echo [ERROR] Existing zip found: %COL_C%.zip
    exit /b 1
  )
  mkdir "%COL_C%"
  echo model,seed,dataset,phase,subject_mod,finetune_mod,lora_target,module_c_candidates,preflight_train_batches,preflight_val_batches,reference_csv,run_tag,status,return_code>"%COL_C%\run_status.csv"
  if not exist "runner_logs" mkdir "runner_logs"
  echo %COL_C%>runner_logs\c_validation_risk_tuev_3seed_5070_latest.txt
) else (
  echo [DRYRUN] would create %COL_C%
)

set /a FAIL_COUNT=0

for %%S in (%SEEDS%) do (
  call :run_c EEGPT eeg %%S
  call :run_c BIOT bi %%S
  call :run_c LaBraM la %%S
  call :run_c CBraMod cb %%S
  call :run_c Gram gr %%S
  call :run_c CSBrain cs %%S
)

if "%DRYRUN%"=="0" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%COL_C%') { Compress-Archive -Path '%COL_C%' -DestinationPath '%COL_C%.zip' -Force }"
)

echo.
echo [RUN] Formal Module C validation-risk TUEV 3-seed runner finished. Collection: %COL_C%
echo [RUN] Failed runs: %FAIL_COUNT%
if not "%FAIL_COUNT%"=="0" exit /b 1
exit /b 0

:set_extra
set "MODEL=%~1"
set "EXTRA_ARGS="
if /I "%MODEL%"=="EEGPT" (
  set "EXTRA_ARGS=--sampling_rate 256"
)
if /I "%MODEL%"=="Gram" (
  set "EXTRA_ARGS=--gram_ckpt checkpoints\base.pth --gram_vqgan_ckpt checkpoints\base_class_quantization.pth --gram_root external\Gram"
)
exit /b 0

:set_reference_args
set "PREFIX=%~1"
set "SEED=%~2"
set "REF="
set "REF_ARGS="
set "REF_SEED=%ABD_REF_COL%\references\%PREFIX%_fewshot_full_ref_seed%SEED%.csv"
set "REF_SEED0=%ABD_REF_COL%\references\%PREFIX%_fewshot_full_ref.csv"
if exist "%REF_SEED%" (
  set "REF=%REF_SEED%"
) else (
  if "%SEED%"=="0" (
    if exist "%REF_SEED0%" set "REF=%REF_SEED0%"
  ) else if "%USE_SEED0_REF_FOR_ALL%"=="1" (
    if exist "%REF_SEED0%" set "REF=%REF_SEED0%"
  )
)
if not "%REF%"=="" (
  set "REF_ARGS=--module_d_sbr_eval --module_d_reference_csv %REF% --module_d_reference_name fewshot_full_seed%SEED% --module_d_hard_k 2"
)
exit /b 0

:run_c
set "MODEL=%~1"
set "PREFIX=%~2"
set "SEED=%~3"
set "TAG=%PREFIX%_c_validation_risk_tuev_s%SEED%_%STAMP%"
call :set_extra "%MODEL%"
call :set_reference_args "%PREFIX%" "%SEED%"
set "COMMON_ARGS=--dataset %DATASET% --task_mod Classification --k_shot %KSHOT% --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --weight_decay %WEIGHT_DECAY% --num_workers %NUM_WORKERS% --loader_prefetch_factor %LOADER_PREFETCH% --seed %SEED%"

echo.
echo [C-Risk+A] model=%MODEL% seed=%SEED% tag=%TAG%
if "%REF%"=="" (
  echo [C-Risk+A] no seed-specific reference CSV found; skipping Module D SBR eval for this run.
) else (
  echo [C-Risk+A] reference=%REF%
)

if "%DRYRUN%"=="1" (
  echo "%PY%" run_finetuning.py %LORA_ARGS% %C_ARGS% --model_name %MODEL% --subject_mod fewshot %COMMON_ARGS% --loss_type sqrt_balanced_ce %SELECT_ARGS% %DIAG_ARGS% --short_output_tag_only --run_tag %TAG% --no_auto_resume %FB_ARGS% %MODULE_BE_ARGS% %ASWA_ARGS% %REF_ARGS% !EXTRA_ARGS!
  exit /b 0
)

call :set_log_paths "%MODEL%" "%TAG%"
echo "%PY%" run_finetuning.py %LORA_ARGS% %C_ARGS% --model_name %MODEL% --subject_mod fewshot %COMMON_ARGS% --loss_type sqrt_balanced_ce %SELECT_ARGS% %DIAG_ARGS% --short_output_tag_only --run_tag %TAG% --no_auto_resume %FB_ARGS% %MODULE_BE_ARGS% %ASWA_ARGS% %REF_ARGS% !EXTRA_ARGS!>"!CMD_LOG!"
"%PY%" run_finetuning.py ^
  %LORA_ARGS% ^
  %C_ARGS% ^
  --model_name %MODEL% ^
  --subject_mod fewshot ^
  %COMMON_ARGS% ^
  --loss_type sqrt_balanced_ce ^
  %SELECT_ARGS% ^
  %DIAG_ARGS% ^
  --short_output_tag_only ^
  --run_tag %TAG% ^
  --no_auto_resume ^
  %FB_ARGS% ^
  %MODULE_BE_ARGS% ^
  %ASWA_ARGS% ^
  %REF_ARGS% ^
  !EXTRA_ARGS! >"!STDOUT_LOG!" 2>"!STDERR_LOG!"
set "RC=!ERRORLEVEL!"
echo !RC!>"!RC_LOG!"
if not "!RC!"=="0" (
  echo %MODEL%,%SEED%,%DATASET%,C,fewshot,lora,module_c,B;D;E,%C_PREFLIGHT_TRAIN_BATCHES%,%C_PREFLIGHT_VAL_BATCHES%,%REF%,%TAG%,failed,!RC!>>"%COL_C%\run_status.csv"
  set /a FAIL_COUNT+=1
  exit /b 0
)
echo %MODEL%,%SEED%,%DATASET%,C,fewshot,lora,module_c,B;D;E,%C_PREFLIGHT_TRAIN_BATCHES%,%C_PREFLIGHT_VAL_BATCHES%,%REF%,%TAG%,done,0>>"%COL_C%\run_status.csv"
exit /b 0

:set_log_paths
set "MODEL=%~1"
set "TAG=%~2"
if not exist "runner_logs" mkdir "runner_logs"
set "STEM=%MODEL%_%TAG%"
set "CMD_LOG=runner_logs\%STEM%.cmd.txt"
set "STDOUT_LOG=runner_logs\%STEM%.stdout.log"
set "STDERR_LOG=runner_logs\%STEM%.stderr.log"
set "RC_LOG=runner_logs\%STEM%.returncode.txt"
echo [LOG] stdout=!STDOUT_LOG!
echo [LOG] stderr=!STDERR_LOG!
exit /b 0
