# Detector 목표 모드 `0.2.5` 재실행 가이드

## 목적과 현재 결론

이 문서는 `0.2.5 detector 안전성 우선 목표 모드`를 다시 실행하거나 다음 버전을 설계할 때 필요한 절차, 데이터 조건, 산출물 위치와 장애 대응을 보존한다. 선택 원칙은 위험과 coverage의 가중합이 아니라 다음 사전식 순서다.

1. Natural `Detector PASS Risk U95 ≤ 0.5%`
2. Natural `E2E APPROVED Risk U95 ≤ 0.5%`
3. Hard `Error Catch Recall ≥ 99%`
4. 적격 후보 중 `Safe Auto-Pass Rate` 최대
5. worst-group coverage, AUGRC, seed와 canonical policy JSON 순 tie-break

`2026-08-13` 실행은 코드·선택·lock·test·ONNX parity·RTX 5080 benchmark까지 완료했지만 정확도와 독립 데이터 gate를 통과하지 못했다. 최종 상태는 `experiment_only`이며 운영 기본 package `bread-worker-0.1.1`은 유지한다. 실제 수치는 [0.2.5 결과 보고서](../reports/detector-target-0.2.5.md)에 기록한다.

## 재실행 전 데이터 조건

다음 조건을 만족하지 못하면 모델 성능과 무관하게 production 승격을 인증할 수 없다.

- Natural/Hard/Shift manifest를 분리한다.
- 이미지마다 `image_sha256`, `perceptual_group_id`, `capture_session_id`, `physical_target_group_id`를 채운다.
- 동일 이미지, perceptual duplicate, 동일 촬영 세션과 동일 물리 대상이 development/test를 넘지 않게 한다.
- Natural은 실제 운영 prevalence를 유지하고 Hard를 coverage 분모에 섞지 않는다.
- 오류 0건에서 0.5% 단측 95% 상한을 인증하려면 detector PASS와 E2E APPROVED가 각각 최소 598개 필요하다.
- Hard test에는 누락, background FP, duplicate, localization, count mismatch, small, border, overlap, blur/exposure 사례를 독립 표본으로 충분히 넣는다.
- Shift test는 다른 날짜·매장·카메라·조명을 사용한다.

현재 데이터 감사의 제한은 다음과 같다.

- Natural 341장: `perceptual_group_id` 없음
- Natural 299장: `physical_target_group_id` 없음
- Hard 309장, Shift 303장: `perceptual_group_id` 없음
- Natural test는 94장뿐이므로 zero-error여도 0.5% U95 인증 불가

## 표준 실행 순서

Python은 학습 환경의 동일 interpreter를 사용한다. 아래 예시는 현재 검증한 Windows 경로다.

```powershell
$detectorTargetCommon = @(
  "--config", "configs\detector_target_0.2.5.json",
  "--training-manifest", "manifests\bread-ops-v2\manifest.jsonl",
  "--training-dataset-root", "C:\workspace\raw_data",
  "--natural-manifest", "manifests\bread-ops-v2\manifest.jsonl",
  "--hard-manifest", "C:\path\to\independent-hard.jsonl",
  "--shift-manifest", "C:\path\to\independent-shift.jsonl",
  "--evaluation-dataset-root", "C:\workspace\raw_data",
  "--classifier-package", "artifacts\packages\bread-worker-0.2.4",
  "--baseline-detector-checkpoint", "artifacts\detector\ops-0.1.1-final\best",
  "--classifier-manifest-metadata", "manifests\bread-10shot-v1\metadata.json",
  "--output-dir", "artifacts\experiments\detector-target-0.2.5",
  "--detector-image-cache", "artifacts\cache\detector-bread-v1",
  "--provider", "cuda",
  "--resume"
)

$detectorTargetPython = "C:\Users\OMEN\AppData\Local\Programs\Python\Python311\python.exe"

foreach ($phase in @(
  "prepare", "train", "cache", "select", "lock", "test", "export-package"
)) {
  & $detectorTargetPython -m bixolon_scanner.training.detector_target `
    @detectorTargetCommon --phase $phase
  if ($LASTEXITCODE -ne 0) { throw "detector-target phase failed: $phase" }
}
```

단계를 건너뛰지 않는다. `test`는 pre-test lock이 생성된 다음에만 연다. test 결과를 본 뒤 epoch, threshold, NMS 또는 uncertainty 정책을 바꾸면 새 실험 디렉터리와 새 lock으로 처음부터 다시 실행한다.

## CPU와 GPU 역할

| 단계 | 주 실행 장치 | 메모 |
|---|---|---|
| `prepare` | CPU | manifest hash·duplicate·group leakage 검사 |
| `train` | CUDA | 세 seed × capture-session 3-fold OOF |
| `cache` | CUDA | detector raw prediction과 고정 classifier ROI 출력 생성 |
| `select` | CPU | 캐시된 결과의 NMS·assignment·risk 통계; 모델 재실행 없음 |
| `lock` | CUDA 또는 복사 | RT-DETR 우승 시 전체 development 재학습; baseline이면 복사 |
| `test` | CUDA | 잠긴 Natural/Hard/Shift 평가 |
| export/parity | CPU+CUDA | PyTorch/ONNX 및 provider parity |
| benchmark | CUDA | RTX 5080 full-path 30 warm-up + 1,000회 |

`select`는 GPU inference가 아니라 캐시 후처리다. `PolicyEvaluationCache`가 동일 image/model의 score/NMS 결과와 동일 detection set의 Hungarian assignment를 재사용한다. 이 캐시를 제거하면 14,896개 후보 sweep이 수 시간 이상 느려질 수 있다.

## Parity 실행

동결 classifier는 `0.2.4` ten-shot checkpoint schema를 사용하므로 classifier 전체 tensor strict parity와 detector parity를 분리한다.

```powershell
$package = "artifacts\experiments\detector-target-0.2.5\package"
$reports = "artifacts\experiments\detector-target-0.2.5\reports"
$backbone = "C:\Users\OMEN\.cache\torch\hub\checkpoints\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

& $detectorTargetPython -m bixolon_scanner.training.ten_shot_parity `
  --package-dir $package `
  --pretest-lock artifacts\experiments\bread-10shot-0.2.4\lock\pretest-lock.json `
  --config configs\bread_10shot_0.2.4.json `
  --manifest manifests\bread-10shot-v1\manifest.jsonl `
  --manifest-metadata manifests\bread-10shot-v1\metadata.json `
  --checkpoint artifacts\experiments\bread-10shot-0.2.4\classifier\best.pt `
  --backbone-weights $backbone `
  --calibration artifacts\experiments\bread-10shot-0.2.4\reports\calibration.json `
  --evaluation-tensors artifacts\experiments\bread-10shot-0.2.4\prepared\evaluation_tensors.npy `
  --output "$reports\classifier-parity-strict.json"

foreach ($provider in @("cpu", "cuda")) {
  $cpuFlag = if ($provider -eq "cpu") { @("--cpu") } else { @() }
  & $detectorTargetPython -m bixolon_scanner.training.parity `
    --package-dir $package `
    --detector-checkpoint artifacts\experiments\detector-target-0.2.5\detector\final\best `
    --classifier-checkpoint artifacts\experiments\bread-10shot-0.2.4\classifier\best.pt `
    --image artifacts\benchmark_images\bread_000022.jpg `
    --output "$reports\detector-parity-$provider.json" `
    --detector-only @cpuFlag
}

& $detectorTargetPython -m bixolon_scanner.training.detector_target `
  @detectorTargetCommon --phase parity `
  --parity-report "$reports\classifier-parity-strict.json" `
  --parity-report "$reports\detector-parity-cpu.json" `
  --parity-report "$reports\detector-parity-cuda.json"
```

최종 parity gate는 세 보고서의 `metadata.json` SHA-256이 같아야 하며, classifier tolerance·Top-1·정렬 Top-3·최종 상태와 detector CPU/CUDA count·IoU·좌표·score가 모두 통과해야 한다.

## RTX 5080 benchmark

ONNX Runtime을 독립 실행할 때는 CUDA DLL bundle을 절대 경로로 전달한다. PyTorch가 먼저 DLL을 preload한 parity 성공만으로 Worker CUDA 초기화를 증명할 수 없다.

```powershell
$cudaRuntime = "C:\workspace\bixolon_scanner\apps\product_scanner\build\windows\x64\runner\Release\worker\cuda-runtime"

& $detectorTargetPython -m bixolon_scanner.benchmark `
  --package-dir $package `
  --images artifacts\benchmark_images `
  --provider cuda `
  --cuda-dll-dir $cudaRuntime `
  --warmup 30 `
  --runs 1000 `
  --output "$reports\benchmark-cuda-1000.json"

& $detectorTargetPython -m bixolon_scanner.training.detector_target `
  @detectorTargetCommon --phase benchmark `
  --benchmark-report "$reports\benchmark-cuda-1000.json"

& $detectorTargetPython -m bixolon_scanner.training.detector_target `
  @detectorTargetCommon --phase finalize
```

`--provider cuda`가 실패하면 CPU로 재측정하지 않는다. `cublasLt64_13.dll` 누락 오류는 CUDA runtime 절대 경로부터 확인한다.

## 주요 산출물

Git에는 원본 이미지, checkpoint, ONNX와 prediction cache를 커밋하지 않는다.

| 산출물 | 경로 |
|---|---|
| 데이터 감사 | `prepared/audit.json` |
| 전체 개발 sweep | `reports/development-policy-sweep.json` |
| 선택 결과 | `reports/development-selection.json` |
| pre-test lock | `lock/pretest-lock.json` |
| 잠긴 test | `reports/locked-test.json` |
| parity gate | `reports/parity-gate.json` |
| RTX 5080 benchmark | `reports/benchmark-cuda-1000.json` |
| 최종 판정 | `reports/final-promotion.json` |
| 실험 package | `package/metadata.json`, `detector.onnx`, `classifier.onnx` |

모든 상대 경로의 기준은 `artifacts/experiments/detector-target-0.2.5`다. `locked-test.json`의 `sets.*.metrics.groups`에는 그룹별 N, gate 네 칸, detector/E2E risk U95, coverage와 Safe Auto-Pass가 있다. `object_diagnostics.error_types`에는 localization, duplicate, background FP, missed GT, count mismatch, `border_related`, `size_related`가 있으며 0건도 키를 유지한다.

## 확인된 장애와 대응

| 증상 | 원인 | 대응 |
|---|---|---|
| `select`가 수 시간 동안 끝나지 않음 | 정책마다 같은 NMS·Hungarian assignment 반복 | `PolicyEvaluationCache` 유지, 캐시 경로와 기존 경로의 결과 동일성 테스트 실행 |
| held-out test에서 `int(None)` | test manifest의 `fold: null` | `_prediction_fold()`가 `-1` sentinel로 정규화 |
| generic parity에서 `pretrained_name` 누락 | 0.2.4 ten-shot checkpoint schema 차이 | classifier는 `ten_shot_parity`, detector는 `--detector-only` 사용 |
| benchmark CUDA provider 초기화 실패 | `cublasLt64_13.dll` 검색 경로 없음 | bundled runtime을 `--cuda-dll-dir` 절대 경로로 전달 |
| 오류 0건인데 risk gate 실패 | PASS/APPROVED 분모 부족 | 독립 표본을 최소 598개 이상 확보 |
| package는 development인데 final은 experiment_only | package schema의 비운영 상태와 평가 판정 역할 차이 | production metadata로 변경하지 말고 `final-promotion.json`을 최종 판정으로 사용 |

## 다음 재도전 체크리스트

- [ ] Natural PASS와 APPROVED 독립 표본을 각각 598개 이상 확보
- [ ] 모든 independence field와 perceptual duplicate group 보완
- [ ] Hard detector 오류 사례를 독립적으로 확충하고 Error Catch Recall 99% 검증
- [ ] Shift를 다른 날짜·매장·카메라·조명으로 재구성
- [ ] 새 데이터 버전과 세 manifest SHA-256 기록
- [ ] development에서만 모델·epoch·정책 선택
- [ ] pre-test lock 이후 입력/artifact hash 변경 없음 확인
- [ ] Natural risk U95 두 항목, Silent Failure 0, Hard recall, UNKNOWN Top-3 확인
- [ ] detector/classifier CPU/CUDA parity와 package SHA 일치 확인
- [ ] RTX 5080 full-path p95 100ms 이하 확인
- [ ] 하나라도 실패하면 `experiment_only`, 수동 waiver 금지
- [ ] production 통과 전 앱 기본 package 변경 금지

## 최종 검증

```powershell
& $detectorTargetPython -m pytest -q
& $detectorTargetPython -m ruff check src tests
git diff --check
```

이번 실행에서는 Python 테스트 256개, ruff와 diff 검사가 모두 통과했다.
