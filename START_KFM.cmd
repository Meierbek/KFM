@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "KFM_OPEN_BROWSER=1"
if exist ".tools\ollama\ollama.exe" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_local_runtime.ps1"
set "KFM_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%KFM_PYTHON%" goto bundled
where py >nul 2>nul
if not errorlevel 1 goto pylauncher
where python >nul 2>nul
if not errorlevel 1 goto pythonpath
echo Python 3.10 or newer is required. Download it from https://www.python.org/downloads/
pause
exit /b 1
:bundled
"%KFM_PYTHON%" server.py
goto finish
:pylauncher
py -3 server.py
goto finish
:pythonpath
python server.py
:finish
pause
