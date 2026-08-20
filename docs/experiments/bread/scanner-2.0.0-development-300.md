# Scanner 2.0.0 RC.10 DINOv3 ViT-B/16 개발 300장 평가

- 평가일: 2026-08-19
- 후보: `2.0.0-rc.10`
- 상태: `development_accuracy_and_parity_passed — production_performance_blocked`
- evidence role: `development_regression`, `promotion_evidence=false`
- 현재 production 기본값: `1.1.0`
- 설계 계약: [Scanner 2.0.0 전체 설계](../../architecture/scanner-2.0.0.md)

## 결론

현재 DINOv3 후보는 `2.0.0-rc.10`이다. 기존 `2.0.0-rc.8`은 이미 DINOv2로 잠긴 별도
release이므로 재사용하지 않는다. 최초 DINOv3 `RC.9`는 CUDA와 CPU에서 임계값 경계 객체 한 건의
최종 상태가 달라 반려했다. `RC.10`은 개발 데이터에서 선택한 승인 임계값에 0.005 provider guard를
더해 이 경계 객체를 보수적으로 비승인 처리한다.

RC.10은 300장 정확도 gate, DINOv3 PyTorch/ONNX parity, 300장 CPU/CUDA 최종 판정 parity와 실제
Worker smoke를 통과했다. 그러나 운영 계약인 요청 시작 간격 1,000ms에서 NVIDIA 프로그램별
`Prefer maximum performance`를 적용해도 full-path mean/p95/p99가
119.08/159.12/190.02ms로 100/100/150ms gate를 실패했다. 연속 warm에서는
84.74/95.38/98.76ms로 통과하지만 이 수치로 1 IPS 운영 성능을 대체하지 않는다. 따라서 RC.10은
정확도·안전성 후보이며 아직 `independent_test_pending` 또는 production 후보가 아니다.

300장은 backbone, 파생 view 수, ridge alpha, threshold와 provider guard를 선택한 개발 데이터다.
아래 수치는 회귀·정책 선택 근거이지 독립 일반화 성능이나 production 승격 증거가 아니다. 소유자의
비공개 test는 실행하지 않았다.

## 잠긴 입력과 개발 산출물

| 항목 | 값 |
|---|---|
| DINOv3 weight | `C:/workspace/trash/ai_scanner/weights/dinov3_vitb16.pt` |
| DINOv3 weight SHA-256 | `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` |
| DINOv3 source revision | `6876159a11b4df116f30f667f8c9888617df0751` |
| ONNX SHA-256 | `55ddf018978d2842645cd2ecff37a303b211323047732e9cfba1e55da1536b64` |
| RC.10 Runtime metadata SHA-256 | `107273695761e3f0f44f845fe7fa38554b6260aff9dee493b4f517605b5a36b5` |
| RC.10 Catalog SHA-256 | `e4cd73f7888d739d64a718d4ed102e2201ba7ad91e04a3a9a40ae853310cef33` |
| Catalog source manifest SHA-256 | `f28444b3e349d29e8e9981db5567670b6eaee0a0d04f08c69246ae8929db565e` |
| 300장 detector manifest SHA-256 | `309a7b8a3f5072ced17106f26f1a5004b13218ed310b6bd012b5c29b6fcce406` |
| 평가 규모 | 이미지 300장, judgeable GT 객체 1,410개 |

Runtime은 고정 4-model D-FINE Detector를 사용한다. Catalog는 SKU마다 `single_objects_2` 원본
10장과 원본별 결정적 파생 view 7개, 총 1,600 feature만으로 fit했다. 300장 ROI pixel은 ridge
fitting에 사용하지 않았지만 개발 query 결과가 모델·정책 선택에 사용됐으므로 독립 evidence가 아니다.

- backbone: `dinov3_vitb16`, foundation parameter update 없음
- embedding: 768차원 frozen global representation
- `ridge_alpha`: `0.1`
- 개발 선택 approval threshold: `0.1501057893037796`
- CPU/CUDA provider guard: `0.005`
- RC.10 effective approval threshold: `0.1551057893037796`
- Top-3 safety threshold: `-2.960296869277954`
- Catalog 자동 제한 SKU/pair: 0/0
- support 최소 짧은 변: 96px, 현재 입력 최소 112px

개발 Catalog는 알려진 개발 전용 key로 서명했다. production-control key로 다시 서명하기 전에는
배포 가능한 Catalog가 아니다.

## 전체 요청 지표

| 지표 | 분자 / 분모 | 결과 |
|---|---:|---:|
| `SEGMENTATION` 비율 | 294 / 300 이미지 | **98.0000%** |
| `IMAGE_RECAPTURE` 비율 | 6 / 300 이미지 | **2.0000%** |
| `APPROVED` 비율 | 1,338 / 1,375 segmentation | **97.3091%** |
| `UNKNOWN` Top-3 비율 | 9 / 1,375 segmentation | **0.6545%** |
| `SEGMENT_RECAPTURE` 비율 | 28 / 1,375 segmentation | **2.0364%** |
| `SEGMENTATION` 이미지 FN 비율 | 0 / 294 이미지 | **0.0000%** |
| `SEGMENTATION` 이미지 FP 비율 | 0 / 294 이미지 | **0.0000%** |
| `APPROVED` 객체 오인율 | 1 / 1,410 GT 객체 | **0.0709%** |
| `UNKNOWN` Top-3 Candidate out 비율 | 0 / 1,410 GT 객체 | **0.0000%** |
| 연속 warm 평균속도 | 300 요청 | **84.2277 ms/image** |
| 1 IPS 평균속도 | 300 요청 | **118.4126 ms/image** |

`APPROVED`, `UNKNOWN` Top-3와 `SEGMENT_RECAPTURE`는 실제 `SEGMENTATION` 응답의
`segmentations[]` 1,375개를 공통 분모로 사용하므로 합이 100%다. 승격용 all-GT
`APPROVED` coverage는 1,338/1,410 = 94.8936%, 올바른 승인 coverage는
1,337/1,410 = 94.8227%다. `APPROVED` 출력만 분모로 한 오인율은 1/1,338 = 0.0747%다.

## 난이도별 요청 지표

각 난이도는 100장이다. 객체 상태의 분모는 각 난이도의 `SEGMENTATION` 응답이 반환한
`segmentations[]` 수다.

| 난이도 | `SEGMENTATION` | `IMAGE_RECAPTURE` | 상태 객체 N | `APPROVED` | `UNKNOWN` Top-3 | `SEGMENT_RECAPTURE` | warm 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EASY | 99/100 (99.0000%) | 1/100 (1.0000%) | 405 | 405/405 (**100.0000%**) | 0/405 (**0.0000%**) | 0/405 (**0.0000%**) | **80.7777ms** |
| MEDIUM | 99/100 (99.0000%) | 1/100 (1.0000%) | 493 | 484/493 (**98.1744%**) | 2/493 (**0.4057%**) | 7/493 (**1.4199%**) | **85.6845ms** |
| HARD | 96/100 (96.0000%) | 4/100 (4.0000%) | 477 | 449/477 (**94.1300%**) | 7/477 (**1.4675%**) | 21/477 (**4.4025%**) | **86.2208ms** |
| 전체 | 294/300 (98.0000%) | 6/300 (2.0000%) | 1,375 | 1,338/1,375 (**97.3091%**) | 9/1,375 (**0.6545%**) | 28/1,375 (**2.0364%**) | **84.2277ms** |

| 난이도 | GT N | FN 포함 `SEGMENTATION` 이미지 | FP 포함 `SEGMENTATION` 이미지 | `APPROVED` 오인 / GT | Candidate out / GT | warm p50 / p95 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| EASY | 410 | 0/99 (0%) | 0/99 (0%) | 0/410 (0%) | 0/410 (0%) | 80.46 / 91.97 / 94.36ms |
| MEDIUM | 500 | 0/99 (0%) | 0/99 (0%) | 0/500 (0%) | 0/500 (0%) | 86.12 / 97.00 / 98.66ms |
| HARD | 500 | 0/96 (0%) | 0/96 (0%) | 1/500 (**0.2000%**) | 0/500 (0%) | 86.32 / 95.36 / 102.61ms |
| 전체 | 1,410 | 0/294 (0%) | 0/294 (0%) | 1/1,410 (**0.0709%**) | 0/1,410 (0%) | 82.93 / 95.36 / 98.62ms |

HARD의 0.2%는 난이도별 진단값이다. 고정 승격 gate는 전체 GT 1,410개의 0.0709%에 적용한다.
이 개발 point metric은 별도 locked test의 단측 95% 상한을 대신하지 않는다.

## 후보 이력과 parity

| 후보 | Embedder | 결과 | 판정 |
|---|---|---|---|
| RC.8 | DINOv2 ViT-B/14 | 기존 immutable pre-private release | DINOv3에 재사용 금지 |
| RC.9 | DINOv3 ViT-B/16, base threshold | CPU/CUDA 최종 상태 mismatch 1/300 | **REJECTED** |
| RC.10 | DINOv3 ViT-B/16, 0.005 guard | 최종 상태·class rank mismatch 0/300 | accuracy/parity PASS |

RC.10의 CPU/CUDA 300장 parity는 segmentation 1,375개, bbox mismatch 0, minimum bbox IoU 1.0,
maximum confidence error 0.0041305423, 최종 상태와 class rank mismatch 0이다. DINOv3 Embedder
PyTorch↔ONNX CUDA maximum absolute error는 `1.0878e-5`, ONNX CPU↔CUDA maximum absolute error는
`8.2329e-6`, minimum cosine similarity는 `0.99999988`이다.

DINOv3 ConvNeXt-Tiny A/B의 안전 승인율은 76.4364%였으므로 ViT-B/16을 선택했다. RC.7
DINOv2 대비 RC.10의 `APPROVED / segmentation`은 93.1636%에서 97.3091%로 +4.1455%p,
`APPROVED / all GT`는 90.8511%에서 94.8936%로 +4.0425%p 증가했다.

## 성능과 blocker

측정 범위는 파일 읽기를 제외한 API 내부 decode, 전처리, Detector ensemble·선택적 refinement,
Embedder와 decision이다. provider는 ONNX Runtime CUDA EP, 요청 동시성 1, warm-up 100회다.

| 부하 조건 | 경로 | N | mean | p50 | p95 | p99 | gate |
|---|---|---:|---:|---:|---:|---:|---|
| 연속 warm | 전체 | 300 | 84.23ms | 82.93ms | 95.36ms | 98.62ms | PASS |
| 연속 warm | full path | 294 | 84.74ms | 83.10ms | 95.38ms | 98.76ms | PASS |
| 연속 warm | detector early exit | 6 | 59.22ms | 58.64ms | 62.33ms | 62.77ms | PASS |
| 1,000ms cadence + Python max performance | 전체 | 300 | 118.41ms | 123.90ms | 158.96ms | 189.34ms | **FAIL** |
| 1,000ms cadence + Python max performance | full path | 294 | 119.08ms | 124.08ms | 159.12ms | 190.02ms | **FAIL** |
| 1,000ms cadence + Python max performance | detector early exit | 6 | 85.92ms | 86.45ms | 106.00ms | 107.23ms | **FAIL** |

기본 프로그램 정책에서는 GPU가 `P8 / 645MHz`로 내려가 기존 full-path mean/p95가
347.37/569.48ms였다. Python 3.11 실행 파일에만 NVIDIA `Prefer maximum performance`를 적용하자
유휴 구간이 `P3~P5 / 1,582~967MHz`로 개선됐지만 100ms SLO에는 부족했다. 관리자 권한 없는
`nvidia-smi --lock-gpu-clocks`는 권한 거부됐으므로 GPU clock floor를 고정한 동일 조건의 재측정이
남아 있다. 전역 GPU 설정은 변경하지 않았다.

ONNX-only keep-alive, detector 병렬 실행과 TF32도 개발 진단했다. keep-alive와 병렬 실행은 p95를
통과하지 못했고, TF32는 FN/FP와 승인 오인을 늘려 정확도 parity를 깨므로 반려했다. 실패한 최적화로
성능 gate를 우회하지 않는다.

TensorRT 10.16.1 CUDA 13은 격리된 feasibility probe로만 확인했다. 기본 FP32/TF32 engine은 실제
이미지에서 CUDA EP 대비 detector logits/boxes 최대 절대 오차가 각각 1.7247/0.6993이었고,
`NVIDIA_TF32_OVERRIDE=0`으로 새로 build한 engine도 각각 1.4783/0.05654였다. 이는 기존 수치 parity
허용 범위를 크게 벗어나므로 300장 최종 상태 평가 전에 반려했다. engine cache는 GPU·TensorRT에
종속되고 해당 환경 변수는 production metadata 계약에도 포함되어 있지 않으므로 RC.10 Runtime이나
Worker 의존성에는 추가하지 않았다.

측정 환경은 Windows 11, Core Ultra 9 285K, RAM 64GB, RTX 5080 16GB, driver 591.86,
Python 3.11.9, ONNX Runtime 1.28.0, CUDA toolkit 12.8, cuDNN 9.20.0.48, Pillow 12.3.0이다.

## gate 판정과 남은 작업

| gate | 기준 | 결과 | 판정 |
|---|---:|---:|---|
| `SEGMENTATION` | ≥90% | 98.0000% | PASS |
| `APPROVED / all GT` | ≥90% | 94.8936% | PASS |
| FN 포함 `SEGMENTATION` 이미지 | ≤0.1% | 0.0000% | PASS |
| FP 포함 `SEGMENTATION` 이미지 | ≤0.1% | 0.0000% | PASS |
| wrong `APPROVED / all GT` | ≤0.1% | 0.0709% | PASS |
| Top-3 Candidate out / all GT | ≤0.1% | 0.0000% | PASS |
| CPU/CUDA 최종 상태·순위 parity | mismatch 0 | 0/300 | PASS |
| CUDA 1 IPS full-path mean/p95/p99 | ≤100/100/150ms | 119.08/159.12/190.02ms | **FAIL** |

실제 Worker smoke는 `/ready` 200, 정상 scan 200, 누락 multipart 422 `ERROR`, 손상 이미지 422,
미지원 형식 415, 로그·응답의 원본 경로/바이트 비노출을 통과했다. 성능 gate가 실패했으므로
production-control 재서명, packaged Worker, 10,000회 reliability, SBOM·취약점 scan과 새 release
lock은 시작하지 않는다. 이 작업들은 성능을 통과한 동일 hash 후보에서만 유효하다.

다음 순서로만 진행한다.

1. 관리자 권한으로 배포 executable의 GPU clock floor를 고정하거나 모델 구조를 다시 최적화한다.
2. 같은 canonical manifest로 1,000ms cadence full-path mean/p95/p99 gate를 재측정한다.
3. 통과한 동일 hash Runtime과 production-control 서명 Catalog를 새 immutable 경로로 만든다.
4. CPU/CUDA parity, source/packaged Worker smoke, 10,000회 reliability와 공급망 gate를 재실행한다.
5. 새 pre-private release lock을 만든 뒤에만 소유자 비공개 test를 1회 실행한다.

## 재현 아티팩트

- backbone A/B: `artifacts/experiments/scanner-2.0.0/catalog-dinov3-vitb16-ab/report.json`
- ONNX export: `artifacts/experiments/scanner-2.0.0/embedder-dinov3-vitb16/export.json`
- Embedder parity: `artifacts/evaluations/scanner-2.0.0/rc.8-dinov3-vitb16-embedder-parity.json`
- RC.9 rejected parity: `artifacts/evaluations/scanner-2.0.0/rc.9-dinov3-vitb16-cpu-cuda-parity.json`
- RC.10 warm CUDA: `artifacts/evaluations/scanner-2.0.0/development-300-rc.10-dinov3-vitb16-cuda.json`
- RC.10 warm trace: `artifacts/evaluations/scanner-2.0.0/development-300-rc.10-dinov3-vitb16-cuda-trace.jsonl`
- RC.10 난이도: `artifacts/evaluations/scanner-2.0.0/development-300-rc.10-dinov3-vitb16-cuda-breakdown.json`
- RC.10 CPU: `artifacts/evaluations/scanner-2.0.0/development-300-rc.10-dinov3-vitb16-cpu.json`
- RC.10 CPU/CUDA parity: `artifacts/evaluations/scanner-2.0.0/rc.10-dinov3-vitb16-cpu-cuda-parity.json`
- RC.10 실제 Worker smoke: `artifacts/evaluations/scanner-2.0.0/rc.10-dinov3-vitb16-real-worker-smoke.json`
- RC.10 canonical 1 IPS: `artifacts/evaluations/scanner-2.0.0/development-300-rc.10-dinov3-vitb16-cuda-1ips-maxperf-canonical.json`

generated model, trace와 실행 package는 Git에 넣지 않는다. DINOv3 weight·ONNX·개발 Catalog와
개발 전용 서명도 Git에 포함하지 않는다.
