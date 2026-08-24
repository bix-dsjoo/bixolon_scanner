@echo off
setlocal

set "IMAGE_DIR=%~1"
if not defined IMAGE_DIR set "IMAGE_DIR=C:\easy"

set "POWERSHELL=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PACKAGE_DIR=%~dp0"
set "RESULT_PATH=%PACKAGE_DIR%n100-0.1.1-openvino-device-matrix.json"

echo BIXOLON Scanner 0.1.1 N100 OpenVINO CPU vs CPU+GPU test
echo Image directory: %IMAGE_DIR%
echo.

if not exist "%IMAGE_DIR%" goto :missing_images

"%POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PACKAGE_DIR%N100-GPU-BENCHMARK.ps1" -ImageDirectory "%IMAGE_DIR%" -OutputPath "%RESULT_PATH%"
if errorlevel 1 goto :failed

echo.
echo N100 OpenVINO device test completed.
echo Result: %RESULT_PATH%
echo Send this JSON file back for review.
pause
exit /b 0

:missing_images
echo.
echo Image directory was not found: %IMAGE_DIR%
echo Put at least 30 JPEG or PNG images in C:\easy, or drag an image folder onto this CMD.
pause
exit /b 2

:failed
echo.
echo N100 GPU test failed. Capture this screen and send it with any generated JSON file.
pause
exit /b 1
