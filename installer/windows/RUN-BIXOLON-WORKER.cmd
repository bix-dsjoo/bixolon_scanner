@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\start-bixolon-worker.ps1"
if errorlevel 1 (
  echo.
  echo BIXOLON Worker start failed. Check that the Intel graphics driver is installed.
  pause
)
endlocal
