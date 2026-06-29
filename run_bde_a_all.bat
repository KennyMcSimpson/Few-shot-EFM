@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not defined PYTHON_EXE set "PYTHON_EXE=python"
set "PY=%PYTHON_EXE%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "COL=col_bde_a_%STAMP%"
set "DRYRUN=0"
if /I "%~1"=="--dry-run" set "DRYRUN=1"

echo [BDE+A] Root: %CD%
echo [BDE+A] Collection: %COL%
echo [BDE+A] Protocol: TUEV fewshot k=0.05, Full FT + LoRA, Module A enabled.
echo [BDE+A] Modules: B=signal_align, D=semantic, E=struct_mix.
if "%DRYRUN%"=="1" echo [BDE+A] DRY RUN: commands will not be launched.
echo.

if exist "%COL%" (
    echo [BDE+A][ERROR] Existing collection folder found: %COL%
    exit /b 1
)
if exist "%COL%.zip" (
    echo [BDE+A][ERROR] Existing zip found: %COL%.zip
    exit /b 1
)

mkdir "%COL%"
echo model,module,lora_target,fb_recipe,run_tag,status>"%COL%\run_status.csv"

call :run_model EEGPT eeg
if errorlevel 1 exit /b 1

call :run_model BIOT bi
if errorlevel 1 exit /b 1

call :run_model LaBraM la
if errorlevel 1 exit /b 1

call :run_model CBraMod cb
if errorlevel 1 exit /b 1

call :run_model Gram gr
if errorlevel 1 exit /b 1

call :run_model CSBrain cs
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%COL%' -DestinationPath '%COL%.zip' -Force"
echo.
echo [BDE+A] Done.
echo [BDE+A] Collected folder: %COL%
echo [BDE+A] Zip package: %COL%.zip
exit /b 0

:run_model
set "MODEL=%~1"
set "PREFIX=%~2"

call :run_one "%MODEL%" "%PREFIX%" B b signal_align sig_align
if errorlevel 1 exit /b 1

call :run_one "%MODEL%" "%PREFIX%" D d semantic sem_lif
if errorlevel 1 exit /b 1

call :run_one "%MODEL%" "%PREFIX%" E e struct_mix str_mix
if errorlevel 1 exit /b 1

exit /b 0

:run_one
set "MODEL=%~1"
set "PREFIX=%~2"
set "MODULE_ID=%~3"
set "SUFFIX=%~4"
set "LORA_TARGET=%~5"
set "FB_RECIPE=%~6"
set "TAG=%PREFIX%_%SUFFIX%_a_%STAMP%"
set "EXTRA_ARGS="

if /I "%MODEL%"=="EEGPT" (
    set "EXTRA_ARGS=--sampling_rate 256"
)
if /I "%MODEL%"=="Gram" (
    set "EXTRA_ARGS=--gram_ckpt checkpoints\base.pth --gram_vqgan_ckpt checkpoints\base_class_quantization.pth --gram_root external\Gram"
)

echo.
echo [BDE+A] Run %MODEL% Module %MODULE_ID% target=%LORA_TARGET% tag=%TAG%

if "%DRYRUN%"=="1" (
    echo %MODEL%,%MODULE_ID%,%LORA_TARGET%,%FB_RECIPE%,%TAG%,dry-run>>"%COL%\run_status.csv"
    exit /b 0
)

"%PY%" run_finetuning.py ^
  --dataset TUEV ^
  --model_name %MODEL% ^
  --task_mod Classification ^
  --subject_mod fewshot ^
  --finetune_mod lora ^
  --k_shot 0.05 ^
  --epochs 30 ^
  --batch_size 16 ^
  --lr 1e-4 ^
  --weight_decay 0.05 ^
  --num_workers 4 ^
  --seed 0 ^
  --loss_type sqrt_balanced_ce ^
  --best_metric balanced_accuracy ^
  --selection_worst_alpha 0.35 ^
  --selection_min02_alpha 0.40 ^
  --selection_std_gamma 0.16 ^
  --lora_target %LORA_TARGET% ^
  --lora_base_update full ^
  --lora_rank 4 ^
  --lora_alpha 8 ^
  --lora_dropout 0.1 ^
  --monitor_dynamics ^
  --eval_train_set ^
  --diag_freq 5 ^
  --save_epoch_ckpt_freq 999 ^
  --adaptive_swa_eval ^
  --adaptive_swa_epoch_min 1 ^
  --adaptive_swa_epoch_max 30 ^
  --adaptive_swa_min_len 3 ^
  --adaptive_swa_max_len 8 ^
  --adaptive_swa_stride 1 ^
  --adaptive_swa_select_metric selection_bacc_min02_std ^
  --adaptive_swa_profile generic ^
  --adaptive_swa_balance_lambda 0.10 ^
  --adaptive_swa_hard_classes 0,2 ^
  --adaptive_swa_hard_floor 0.05 ^
  --adaptive_swa_hard_floor_lambda 0.20 ^
  --adaptive_swa_std_lambda 0.04 ^
  --adaptive_swa_tie_mode hard_stable ^
  --adaptive_swa_tie_eps 0.002 ^
  --short_output_tag_only ^
  --run_tag %TAG% ^
  --no_auto_resume ^
  --fb_enable ^
  --fb_probe ^
  --fb_recipe %FB_RECIPE% ^
  --fb_split_check ^
  --fb_collect ^
  --fb_collect_name %COL% ^
  %EXTRA_ARGS%

if errorlevel 1 (
    echo %MODEL%,%MODULE_ID%,%LORA_TARGET%,%FB_RECIPE%,%TAG%,failed>>"%COL%\run_status.csv"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%COL%') { Compress-Archive -Path '%COL%' -DestinationPath '%COL%_partial.zip' -Force }"
    echo [BDE+A][ERROR] %MODEL% Module %MODULE_ID% failed.
    echo [BDE+A][ERROR] Partial package: %COL%_partial.zip
    exit /b 1
)

echo %MODEL%,%MODULE_ID%,%LORA_TARGET%,%FB_RECIPE%,%TAG%,done>>"%COL%\run_status.csv"
exit /b 0
