param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PortableSource = "artifacts\portable\rpc200-v18-benchmark",
    [string]$OutputDirectory = "artifacts\portable\rpc200-v18-jetson-trt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = [IO.Path]::GetFullPath($RepoRoot)
$source = if ([IO.Path]::IsPathRooted($PortableSource)) {
    [IO.Path]::GetFullPath($PortableSource)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo $PortableSource))
}
$output = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo $OutputDirectory))
}

foreach ($required in @("source", "model-package", "context", "dataset", "manifest.jsonl")) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $required))) {
        throw "Portable source input does not exist: $(Join-Path $source $required)"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists. Choose a new path or remove it explicitly: $output"
}

New-Item -ItemType Directory -Path $output | Out-Null
foreach ($directory in @("source", "model-package", "context", "dataset")) {
    Copy-Item -LiteralPath (Join-Path $source $directory) -Destination (Join-Path $output $directory) -Recurse
}
Copy-Item -LiteralPath (Join-Path $source "manifest.jsonl") -Destination (Join-Path $output "manifest.jsonl")

$experimentSource = Join-Path $repo "src\bixolon_scanner\experiments\rpc200"
$experimentTarget = Join-Path $output "source\src\bixolon_scanner\experiments\rpc200"
foreach ($filename in @("validation_benchmark.py", "jetson_provider_parity.py", "tensorrt_native.py")) {
    $sourcePath = Join-Path $experimentSource $filename
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Jetson experiment source does not exist: $sourcePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $experimentTarget $filename)
}

$runScript = @'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: This bundle requires Linux aarch64." >&2
  exit 2
fi
if ! grep -q '^# R39 (release), REVISION: 2' /etc/nv_tegra_release 2>/dev/null; then
  echo "ERROR: This bundle is locked to Jetson Linux R39.2." >&2
  exit 2
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libcudart.so'; then
  cat >&2 <<'EOF'
ERROR: CUDA runtime was not found.
Install the JetPack components first, then reboot:
  sudo apt update
  sudo apt install nvidia-jetpack
  sudo reboot
EOF
  exit 2
fi
if ! nvpmodel -q 2>/dev/null | grep -q 'MAXN_SUPER'; then
  cat >&2 <<'EOF'
ERROR: MAXN_SUPER mode is not active.
Run this once before the benchmark:
  sudo ./SET_MAX_PERFORMANCE.sh
EOF
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv --system-site-packages .venv; then
    echo "Install python3-venv first: sudo apt install python3-venv" >&2
    exit 2
  fi
fi

PYTHON="$ROOT/.venv/bin/python"
if [[ "${BIXOLON_SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  "$PYTHON" -m pip install -e "$ROOT/source" \
    "numpy==2.4.4" "pillow==12.2.0" "scikit-learn>=1.6,<2"
  if [[ -n "${BIXOLON_ORT_WHEEL:-}" ]]; then
    "$PYTHON" -m pip install --force-reinstall --no-deps "$BIXOLON_ORT_WHEEL"
  else
    "$PYTHON" -m pip install --pre \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
      onnxruntime-gpu==1.25.0.dev20260331005
  fi
fi

"$PYTHON" - <<'PY'
import onnxruntime as ort
import platform
import tensorrt as trt

providers = ort.get_available_providers()
print(f"Python: {platform.python_version()}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"TensorRT: {trt.__version__}")
print(f"Providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ERROR: CUDAExecutionProvider is unavailable")
PY

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$ROOT/results/$STAMP"
mkdir -p "$RESULT_DIR/cuda" "$RESULT_DIR/tensorrt"

TRTEXEC="$(command -v trtexec || true)"
if [[ -z "$TRTEXEC" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [[ -z "$TRTEXEC" ]]; then
  echo "ERROR: trtexec was not found. Install the JetPack TensorRT components." >&2
  exit 2
fi

mapfile -t MODEL_CONTRACT < <("$PYTHON" - "$ROOT/model-package/metadata.json" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
detector = metadata["detector"]
classifier = metadata["classifier"]
print(detector["filename"])
print(detector["input_name"])
print("x".join(str(value) for value in detector["input_size"]))
print(classifier["filename"])
print(classifier["input_name"])
print("x".join(str(value) for value in classifier["input_size"]))
print(max(int(value) for value in classifier["warmup_batch_sizes"]))
PY
)
DETECTOR_ONNX="$ROOT/model-package/${MODEL_CONTRACT[0]}"
DETECTOR_INPUT="${MODEL_CONTRACT[1]}"
DETECTOR_SIZE="${MODEL_CONTRACT[2]}"
CLASSIFIER_ONNX="$ROOT/model-package/${MODEL_CONTRACT[3]}"
CLASSIFIER_INPUT="${MODEL_CONTRACT[4]}"
CLASSIFIER_SIZE="${MODEL_CONTRACT[5]}"
CLASSIFIER_MAX_BATCH="${MODEL_CONTRACT[6]}"
CLASSIFIER_OPT_BATCH=$(((CLASSIFIER_MAX_BATCH + 1) / 2))

ENGINE_DIR="$ROOT/engines"
DETECTOR_ENGINE="$ENGINE_DIR/detector-fp16.plan"
CLASSIFIER_ENGINE="$ENGINE_DIR/classifier-fp16.plan"
mkdir -p "$ENGINE_DIR"

build_engine() {
  local onnx_path="$1"
  local engine_path="$2"
  local input_name="$3"
  local minimum_shape="$4"
  local optimum_shape="$5"
  local maximum_shape="$6"
  if [[ -s "$engine_path" ]]; then
    echo "Reusing TensorRT engine: $engine_path"
    return
  fi
  local partial_path="${engine_path}.partial"
  if [[ -e "$partial_path" ]]; then
    echo "ERROR: incomplete TensorRT engine exists: $partial_path" >&2
    exit 2
  fi
  "$TRTEXEC" \
    --onnx="$onnx_path" \
    --saveEngine="$partial_path" \
    --fp16 \
    --skipInference \
    --builderOptimizationLevel=5 \
    --memPoolSize=workspace:1024 \
    --minShapes="${input_name}:${minimum_shape}" \
    --optShapes="${input_name}:${optimum_shape}" \
    --maxShapes="${input_name}:${maximum_shape}"
  mv "$partial_path" "$engine_path"
}

echo "Building TensorRT FP16 engines..."
build_engine \
  "$DETECTOR_ONNX" "$DETECTOR_ENGINE" "$DETECTOR_INPUT" \
  "1x3x$DETECTOR_SIZE" "1x3x$DETECTOR_SIZE" "1x3x$DETECTOR_SIZE"
build_engine \
  "$CLASSIFIER_ONNX" "$CLASSIFIER_ENGINE" "$CLASSIFIER_INPUT" \
  "1x3x$CLASSIFIER_SIZE" "${CLASSIFIER_OPT_BATCH}x3x$CLASSIFIER_SIZE" \
  "${CLASSIFIER_MAX_BATCH}x3x$CLASSIFIER_SIZE"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "model=$(tr -d '\0' </proc/device-tree/model)"
  echo "architecture=$(uname -m)"
  echo "kernel=$(uname -r)"
  echo "jetson_linux=$(head -n 1 /etc/nv_tegra_release)"
  command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n 1 || true
  "$TRTEXEC" --version 2>&1 | tail -n 1 || true
  command -v nvpmodel >/dev/null 2>&1 && nvpmodel -q 2>/dev/null || true
  sha256sum "$DETECTOR_ENGINE" "$CLASSIFIER_ENGINE"
} >"$RESULT_DIR/environment.txt"

COMMON=(
  -m bixolon_scanner.experiments.rpc200.validation_benchmark
  --package-dir "$ROOT/model-package"
  --context-onnx "$ROOT/context/logistic.onnx"
  --manifest "$ROOT/manifest.jsonl"
  --dataset-root "$ROOT/dataset"
  --warmup 30
  --images-per-level 200
  --class-aware-nms-threshold 0.55
  --context-threshold 0.00225
  --duplicate-overlap-threshold 0.452
  --duplicate-overlap-max-score 0.864
  --duplicate-overlap-max-quality 0.194
  --duplicate-low-quality-max-quality 0.0058
  --duplicate-low-quality-min-score 0.93
  --duplicate-min-rank 0.857
  --assignment-conflict-top-k 2
  --assignment-mutual-pair 160:161
)

echo "Checking CUDA/TensorRT FP16 parity..."
"$PYTHON" -m bixolon_scanner.experiments.rpc200.jetson_provider_parity \
  --package-dir "$ROOT/model-package" \
  --manifest "$ROOT/manifest.jsonl" \
  --dataset-root "$ROOT/dataset" \
  --detector-engine "$DETECTOR_ENGINE" \
  --classifier-engine "$CLASSIFIER_ENGINE" \
  --images-per-level 30 \
  --output "$RESULT_DIR/provider-parity.json"

for provider in cuda tensorrt; do
  for level in easy medium hard; do
    echo "Running RPC200 v18 $provider $level benchmark..."
    provider_args=()
    if [[ "$provider" == "tensorrt" ]]; then
      provider_args=(
        --detector-engine "$DETECTOR_ENGINE"
        --classifier-engine "$CLASSIFIER_ENGINE"
      )
    fi
    "$PYTHON" "${COMMON[@]}" --provider "$provider" "${provider_args[@]}" \
      --levels "$level" --output "$RESULT_DIR/$provider/$level.json"
  done
done

"$PYTHON" - "$RESULT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print(f"\nCompleted: {root}")
for level in ("easy", "medium", "hard"):
    rows = {}
    for provider in ("cuda", "tensorrt"):
        report = json.loads((root / provider / f"{level}.json").read_text())
        rows[provider] = report["difficulty"][level]["all_images"]
    cuda = rows["cuda"]
    trt = rows["tensorrt"]
    speedup = cuda["p95_ms"] / trt["p95_ms"]
    print(
        f"{level:<6} CUDA p95={cuda['p95_ms']:8.3f} ms  "
        f"TRT-FP16 p95={trt['p95_ms']:8.3f} ms  speedup={speedup:.2f}x"
    )
PY
'@

$performanceScript = @'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: Jetson aarch64 is required." >&2
  exit 2
fi

nvpmodel -m 2
/usr/bin/jetson_clocks --fan
nvpmodel -q
/usr/bin/jetson_clocks --show
'@

$readme = @'
# RPC200 v18 Jetson Orin Nano benchmark

Target: NVIDIA Jetson Orin Nano Developer Kit Super, aarch64, Jetson Linux R39.2.

## Transfer from Windows

```powershell
scp .\rpc200-v18-jetson-trt.tar.gz <jetson-user>@<jetson-host>:~/
```

## Run on Jetson

```bash
tar -xzf rpc200-v18-jetson-trt.tar.gz
cd rpc200-v18-jetson-trt
chmod +x RUN_JETSON_BENCHMARK.sh
chmod +x SET_MAX_PERFORMANCE.sh
sudo ./SET_MAX_PERFORMANCE.sh
./RUN_JETSON_BENCHMARK.sh
```

The first run downloads Python dependencies and Microsoft's CUDA 13 Linux aarch64
ONNX Runtime test package. JetPack supplies TensorRT and `trtexec`; the script builds
native FP16 engines using input names, dimensions and classifier batch limits from the
model package metadata. It first gates TensorRT against CUDA on 90 fixed images, then
benchmarks CUDA EP and native TensorRT. It never falls back to CPU. Results are written
below `results`, and generated engines remain below `engines` for subsequent runs.

If CUDA is missing, install the JetPack components and reboot before running:

```bash
sudo apt update
sudo apt install nvidia-jetpack
sudo reboot
```

For stable measurements, select the intended power mode, cool the device adequately,
close other workloads, and run each benchmark more than once.
'@

$utf8NoBom = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $output "RUN_JETSON_BENCHMARK.sh"), $runScript.Replace("`r`n", "`n") + "`n", $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $output "SET_MAX_PERFORMANCE.sh"), $performanceScript.Replace("`r`n", "`n") + "`n", $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $output "README.md"), $readme.Replace("`r`n", "`n") + "`n", $utf8NoBom)

$checksumPath = Join-Path $output "checksums.sha256"
$checksumLines = Get-ChildItem -LiteralPath $output -Recurse -File |
    Where-Object { $_.FullName -ne $checksumPath } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($output, $_.FullName).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
[IO.File]::WriteAllLines($checksumPath, $checksumLines, [Text.Encoding]::ASCII)

$archive = "$output.tar.gz"
if (Test-Path -LiteralPath $archive) {
    throw "Archive already exists: $archive"
}
$parent = Split-Path -Parent $output
$leaf = Split-Path -Leaf $output
& tar.exe -czf $archive -C $parent $leaf
if ($LASTEXITCODE -ne 0) {
    throw "tar archive creation failed"
}
$archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$archive.sha256", "$archiveHash  $leaf.tar.gz`n", [Text.Encoding]::ASCII)

$totalBytes = (Get-ChildItem -LiteralPath $output -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Jetson benchmark prepared: $output"
Write-Host ("Bundle size: {0:N2} GB; archive: {1:N2} GB" -f ($totalBytes / 1GB), ((Get-Item $archive).Length / 1GB))
Write-Host "SHA256: $archiveHash"
