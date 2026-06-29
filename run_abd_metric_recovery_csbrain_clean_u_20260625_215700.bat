@echo off
setlocal

cd /d "%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [ABD-CS] Resume collection: col_abd_metric_20260625_215700
echo [ABD-CS] Model: CSBrain only
echo [ABD-CS] U phase: clean Ada-style Full FT, no fb/probe, no LoRA, no Module A, CE loss.
echo [ABD-CS] R/REF/A/B/D phases: original ABD metric protocol.
echo [ABD-CS] DataLoader default: CSBrain workers=2, prefetch=1. You can override with --num-workers 4.
echo.

"%PYTHON_EXE%" util\run_abd_metric_recovery_csbrain_clean_u.py --resume-stamp 20260625_215700 %*

set ABD_EXIT_CODE=%ERRORLEVEL%
echo.
echo [ABD-CS] Finished with exit code %ABD_EXIT_CODE%.
exit /b %ABD_EXIT_CODE%
