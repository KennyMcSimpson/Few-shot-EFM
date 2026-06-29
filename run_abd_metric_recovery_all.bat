@echo off
setlocal

cd /d "%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" util\run_abd_metric_recovery.py %*
exit /b %ERRORLEVEL%
