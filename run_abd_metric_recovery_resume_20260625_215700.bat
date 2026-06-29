@echo off
setlocal

cd /d "%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [ABD] Resume collection: col_abd_metric_20260625_215700
echo [ABD] Protocol: U/R/REF/A/B/D, no C/E in this run.
echo [ABD] DataLoader: all models workers=4, prefetch=1.
echo.

"%PYTHON_EXE%" util\run_abd_metric_recovery.py --resume-stamp 20260625_215700 %*

set ABD_EXIT_CODE=%ERRORLEVEL%
echo.
echo [ABD] Finished with exit code %ABD_EXIT_CODE%.
exit /b %ABD_EXIT_CODE%
