@echo off
setlocal

set "IMAGE_DIR=%~1"
if not defined IMAGE_DIR set "IMAGE_DIR=C:\easy"

set "POWERSHELL=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PACKAGE_DIR=%~dp0"

echo BIXOLON Bakery AI Scanner 0.1.14 N100 OpenVINO CPU 1xauto test
echo Image directory: %IMAGE_DIR%
"%POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -File "%PACKAGE_DIR%N100-STAGE-TEST.ps1" -ImageDirectory "%IMAGE_DIR%" -OutputPath "%PACKAGE_DIR%n100-0.1.14-result.json"
if errorlevel 1 goto :failed

echo.
echo N100 test completed.
echo Result: %PACKAGE_DIR%n100-0.1.14-result.json
pause
exit /b 0

:failed
echo.
echo N100 test failed. Capture this screen and send it with any generated JSON file.
pause
exit /b 1
