@echo off
setlocal

set "IMAGE_DIR=%~1"
if not defined IMAGE_DIR set "IMAGE_DIR=C:\easy"

set "POWERSHELL=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PACKAGE_DIR=%~dp0"
set "RESULT_PATH=%PACKAGE_DIR%n100-ssdlite-0.1.7-openvino-device-matrix.json"

echo BIXOLON SSDLite 0.1.7 candidate N100 CPU vs CPU+iGPU benchmark
echo Image directory: %IMAGE_DIR%
echo.

if not exist "%IMAGE_DIR%" goto :missing_images

"%POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%PACKAGE_DIR%N100-GPU-BENCHMARK.ps1" ^
  -ImageDirectory "%IMAGE_DIR%" ^
  -OutputPath "%RESULT_PATH%" ^
  -ExpectedVersion "0.1.7" ^
  -AllowNoCountVerifier ^
  -ConfidenceTolerance 0.02 ^
  -MaximumWorkingSetBytes 2415919104 ^
  -MaximumFullPathLatencyMs 1000.0
if errorlevel 1 goto :failed

echo.
echo N100 test completed.
echo Result: %RESULT_PATH%
echo Send this JSON file back for final review.
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
echo N100 test failed. Send this screen and any generated JSON file for review.
pause
exit /b 1
