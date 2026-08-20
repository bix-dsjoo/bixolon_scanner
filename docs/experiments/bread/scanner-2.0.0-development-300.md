# Scanner 2.0.0 RC.8 개발 300장 평가

- 평가일: 2026-08-19
- 후보: `2.0.0-rc.8`
- 결론: `development_gates_passed — owner_private_test_pending`
- evidence role: `development_regression`, `promotion_evidence=false`
- 설계 계약: [Scanner 2.0.0 전체 설계](../../architecture/scanner-2.0.0.md)

> 2026-08-20 후속 결정: 프로젝트 소유자가 private test 미실행과 영구 무키 Catalog를 명시적으로
> waiver하고 같은 RC.8을 `2.0.0` production으로 승격했다. 이 보고서의 역할과 지표는 계속
> development evidence이며 독립 인증으로 바뀌지 않는다.

## 결론

고정 4-model D-FINE Detector, frozen DINOv2 Base Embedder와 SKU별 10장으로 만든 ridge Store
Catalog 조합에 Top-2 pair probability와 retrieval OOD 방어를 추가했다. 300장 개발 gate, CPU/CUDA
최종 판정 parity, PyTorch/ONNX Embedder parity와 source·standalone Worker smoke를 통과했다.

300장과 [2026-08-18 운영 수집본 115장](scanner-2.0.0-operational-2026-08-18-115.md)은 RC.8
정책 선택에 함께 사용됐다. 두 데이터의 합계 415장·판정 가능 GT 1,914개는 모두 개발 계보다.
아래 수치는 회귀와 후보 고정 근거이지 독립 일반화 성능이나 production 승격 증거가 아니다.

## 고정한 정책과 데이터

| 항목 | 값 |
|---|---|
| Runtime | `artifacts/packages/bread-scanner-2.0.0-rc.8-runtime` |
| Store Catalog | `artifacts/catalogs/bread-project-2/2.0.0-rc.8` |
| 300장 manifest | `artifacts/verification/scan-source-cleanup-20260819/detector_manifest.jsonl` |
| 300장 manifest SHA-256 | `309a7b8a3f5072ced17106f26f1a5004b13218ed310b6bd012b5c29b6fcce406` |
| 115장 manifest | `datasets/bread_dataset/operational_collections/2026-08-18/annotations/instances.json` |
| 115장 manifest SHA-256 | `7909a8fdb31850b5af1cb4e95aa007b6b9e3d2ec5da06e6f8c9f74d3ed2bb56f` |
| 정책 설정 | `configs/experiments/bread/scanner_2_0_rc8_policy.json` |
| Catalog support | SKU별 10장, 총 200장과 결정적 파생 view만 사용 |
| approval score | `sigmoid(Top-1 ridge logit - Top-2 ridge logit)` |
| approval threshold | `0.54923` |
| retrieval/OOD minimum similarity | `0.414268881082535` |
| RC.8 평가 당시 상태 | `independent_test_pending`(2026-08-20 owner waiver로 종료) |

`0.54923`은 개발 오류 score `0.5491052`와 다음 올바른 승인 score `0.5493576` 사이에서
CPU/CUDA 수치 여유를 둔 값이다. 특정 SKU pair 예외를 두지 않는다. retrieval 유사도가 OOD 경계보다
낮으면 `SEGMENT_RECAPTURE`, Top-1/Top-2가 애매하거나 Ridge·retrieval이 충돌하면
`UNKNOWN`+Top-3, 나머지 안전 조건을 통과한 객체만 `APPROVED`한다.

## 요청 지표

| 지표 | 분자 / 분모 | 결과 |
|---|---:|---:|
| `SEGMENTATION` 비율 | 294 / 300 이미지 | **98.0000%** |
| `IMAGE_RECAPTURE` 비율 | 6 / 300 이미지 | **2.0000%** |
| `APPROVED` 비율 | 1,269 / 1,410 GT 객체 | **90.0000%** |
| `UNKNOWN` Top-3 비율 | 98 / 1,410 GT 객체 | **6.9504%** |
| `SEGMENT_RECAPTURE` 비율 | 8 / 1,410 GT 객체 | **0.5674%** |
| `SEGMENTATION` 이미지 FN 비율 | 0 / 294 이미지 | **0.0000%** |
| `SEGMENTATION` 이미지 FP 비율 | 0 / 294 이미지 | **0.0000%** |
| `APPROVED` 객체 오인율 | 0 / 1,410 GT 객체 | **0.0000%** |
| `UNKNOWN` Top-3 Candidate out 비율 | 1 / 1,410 GT 객체 | **0.0709%** |
| 평균속도 | 300 요청 | **84.9796 ms/image** |

객체 상태의 분모는 모든 판정 가능 GT 1,410개다. 예측이 `IMAGE_RECAPTURE`인 이미지의 GT도
분모에서 제외하지 않으므로 상태 분자 합은 1,375개다. 장기 목표 `APPROVED ≥99%`는 미달이고
운영 개발 gate `APPROVED ≥90%`는 경계를 포함해 통과했다. 두 판정을 섞지 않는다.

## 성능

측정 범위는 decode, 전처리, Detector ensemble·선택적 refinement, Embedder와 최종 decision이다.

| 경로 | N | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| 전체 요청 | 300 | 84.9796 ms | 82.9625 ms | 98.0651 ms | 105.2764 ms |
| full path | 294 | 85.4377 ms | 83.3139 ms | 98.1018 ms | 105.3441 ms |
| Detector 조기 종료 | 6 | 62.5308 ms | 61.0069 ms | 67.1572 ms | 67.9487 ms |
| selective refinement | 1 | 300.0502 ms | 300.0502 ms | 300.0502 ms | 300.0502 ms |

- 환경: Windows 11, Core Ultra 9 285K, RAM 64GB, RTX 5080 16GB
- provider: ONNX Runtime CUDA Execution Provider
- 요청 동시성: 1
- warm-up: 200회
- 성능 gate: full-path mean·p95 ≤100ms, p99 ≤150ms

CPU 300장 전체 평균은 2,423.47ms였다. CPU에는 latency gate를 적용하지 않는다.

## 개발 gate

| gate | 기준 | 결과 | 판정 |
|---|---:|---:|---|
| `SEGMENTATION` | ≥90% | 98.0000% | PASS |
| `APPROVED / all GT` | ≥90% | 90.0000% | PASS |
| FN 포함 `SEGMENTATION` 이미지 | ≤0.1% | 0.0000% | PASS |
| FP 포함 `SEGMENTATION` 이미지 | ≤0.1% | 0.0000% | PASS |
| wrong `APPROVED / all GT` | ≤0.1% | 0.0000% | PASS |
| Top-3 Candidate out / all GT | ≤0.1% | 0.0709% | PASS |
| CUDA full-path mean / p95 / p99 | ≤100 / 100 / 150ms | 85.44 / 98.10 / 105.34ms | PASS |

## Parity·Worker·운영 gate

- CPU↔CUDA 300장: 1,375 segmentation, 최종 상태·class rank·Top-3 mismatch 0, bbox mismatch 0,
  최소 bbox IoU 1.0, 최대 confidence 차이 0.0004085 ≤0.005, PASS
- PyTorch↔ONNX CUDA Embedder: 최대 절대오차 0.00002074, 최소 cosine 0.99999994, PASS
- ONNX CPU↔CUDA Embedder: 최대 절대오차 0.00002408, 최소 cosine 0.99999988, PASS
- source FastAPI smoke: readiness, 정상 scan, multipart 누락·손상·미지원 입력의 4xx `ERROR`, PASS
- locked build environment의 frozen PyInstaller Worker smoke: CUDA readiness와 정상/손상 이미지,
  금지된 PyTorch·TorchVision·SciPy·pytest runtime path 및 lock 밖 bundled distribution 0개, PASS
- 실제 Worker 10,000회 순차 요청: non-200 0건, 판정 mismatch 0건, readiness 실패 0건,
  RSS 증가율 -0.1421%, PASS
- 10,000회 가속 안정성 지연 mean / p50 / p95 / p99: 93.97 / 92.09 / 107.00 / 120.12ms.
  이 수치는 장시간 soak를 대체하지 않으며 100ms 모델 성능 gate에는 사용하지 않는다.
- Windows/Python/CUDA exact lock, CycloneDX SBOM과 known vulnerability 0건: 기존 동일 dependency
  lock을 RC.8 Worker에 재검증

## 재현 아티팩트

- CUDA report: `artifacts/evaluations/scanner-2.0.0/development-300-rc.8-final-cuda.json`
- CUDA trace: `artifacts/evaluations/scanner-2.0.0/development-300-rc.8-final-cuda-trace.jsonl`
- CPU report: `artifacts/evaluations/scanner-2.0.0/development-300-rc.8-final-cpu.json`
- CPU/CUDA parity: `artifacts/evaluations/scanner-2.0.0/rc.8-final-cpu-cuda-parity.json`
- Embedder parity: `artifacts/evaluations/scanner-2.0.0/rc.8-final-embedder-parity.json`
- Worker smoke: `artifacts/evaluations/scanner-2.0.0/rc.8-final-real-worker-smoke.json`
- packaged Worker smoke: `artifacts/evaluations/scanner-2.0.0/rc.8-final-packaged-worker-smoke.json`
- 10,000회 reliability: `artifacts/evaluations/scanner-2.0.0/rc.8-final-reliability-10000.json`
- release lock: `artifacts/releases/scanner-2.0.0-rc.8-pre-private/release-lock.json`

release lock SHA-256은 `564d804a598e4a749a5b02c29007f9fea059808411e20cdb864ab2e28205f4cf`다.
lock에 기록된 Runtime directory hash는
`18574a22ca1cf59db69b624f9d72ab765f026469fbd9c0f86c7bdae4974d91e2`, Catalog directory
hash는 `001661a9e3efb38660520a971ea7f010382d0f1aa592cb2b7a30a1cbfca72c12`다.

이 보고서 작성 시점에는 새 owner-private locked test가 남아 있었다. 2026-08-20 소유자 예외로
해당 gate를 실행하지 않고 승격했으며 Runtime model graph·weight·policy hash는 변경하지 않았다.
