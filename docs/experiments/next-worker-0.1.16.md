# 0.1.16: 220장 학습 모델과 배경 검출 개선

기준일: 2026-09-07. 제품 버전은 `0.1.16`, Flutter build는 `0.1.16+19`이며
선택한 source candidate는 `limited220-surfaces`다. 원본 단일 200장과 고정 멀티 20장을 유지했다.
운영 2026-08-18의 115장은 학습에 사용하지 않았다. 승인 margin 0.80, 입력 해상도,
전역 primary/detail/verifier routing과 공개 상태·오류 계약도 유지했다.

## 개선과 선택 이유

기존 `limited220-context`는 빈 트레이와 종이·배경을 세 객체로 검출했다. 승인되지는 않았지만
불필요한 UNKNOWN과 객체 재촬영을 만들었다. 합성 학습 배경이 단순 그라데이션에 치우친 점을
보완해 절차적으로 만든 무늬·각진 종이·둥근 트레이 형태를 양성·음성 장면 모두에 적용했다.
외부 배경 사진을 읽지 않으며 양성과 음성의 배경 종류를 분리하는 지름길도 만들지 않았다.

고정 seed 220202616의 1,200장 합성 장면(빈 장면 확률 0.2)을 만들고 기존 Detector checkpoint에서
학습률 0.00003으로 4 epoch 추가 학습했다. 빈 이미지의 hard negative loss를 유지했다.
마지막 epoch를 사용하며 평가로 checkpoint·threshold를 선택하지 않았다. Classifier와 Catalog
payload는 context 후보에서 그대로 재사용했다. 별도 seed 220202617로 160장의 합성 진단을 만들었다.

| 진단 | context | surfaces |
|---|---:|---:|
| 원본 멀티 20장 정답 승인 / GT 136개 | 134 | 135 (99.26%) |
| 원본 멀티 오승인 / 검출 누락 / Top-3 누락 | 0 / 0 / 0 | 0 / 0 / 0 |
| 합성 160장 배경 추가 검출 | 8 | 0 |
| 합성 160장 GT 대응 / GT 344개 | 340 | 343 |
| 합성 진단 오승인 | 0 | 0 |

합성 진단의 GT 미대응은 이미지 전체 재촬영에 포함된 객체도 포함해 읽어야 한다. 기존 후보의
미대응 4개 중 3개는 전체 재촬영 이미지에 있었다. surfaces의 미대응은 1개다. 합성은 같은 원본에서
파생됐으므로 독립 validation이 아니다. 이 개발 비교로 후보를 고른 후 운영 자료를 회귀 확인했다.

## 운영 115장 정답 대조

COCO 정답 504개와 class-agnostic bbox IoU ≥ 0.5의 일대일 대응을 사용했다. 모든 원본 이미지의
SHA-256을 기존 추론 trace와 대조했다. 승인 클래스 오류와 정답에 대응하지 않는 승인을 구분했다.

| 결과 | context | surfaces |
|---|---:|---:|
| 정답 승인 | 465 | 465 (504개 중 92.26%) |
| 승인 오분류 / 배경 오승인 | 0 / 0 | 0 / 0 |
| 정답 객체 검출 누락 | 0 | 0 |
| 배경 추가 검출 | 3 | 0 |
| 실제 객체 UNKNOWN | 39 | 38 |
| 실제 UNKNOWN의 Top-3 정답 포함 | 39 / 39 | 38 / 38 |
| 실제 객체 SEGMENT_RECAPTURE | 0 | 1 |
| 빈 이미지 IMAGE_RECAPTURE | 2 | 4 |

085번의 하단 경계 bread_16은 검출 box가 이미지 끝에 닿고 승인 점수가 낮아 UNKNOWN에서
SEGMENT_RECAPTURE로 바뀌었다. 이는 강제 승인을 피하는 기존 경계 정책의 결과이며 공개 정책을
우회하지 않았다. 092번 빈 트레이와 095번 종이의 배경 검출은 제거됐다. 전체 재촬영 4장에는
정답 객체가 없다. IoU 0.75 민감도 진단에서는 승인 4개의 box가 대응 기준에 미달하지만 IoU 0.5에서
같은 정답 class에 대응한다. 이를 클래스 오승인으로 혼합하지 않는다.

운영 115장에는 원본 220장과 동일 SHA-256 이미지가 없지만 같은 실물·촬영 조건의 독립성은
입증하지 않았다. 운영 결과를 관찰하고 배경 개선 방향을 정했으므로 이 자료는 회귀 자료이며
독립 test가 아니다. 새 실물·미등록 객체의 오승인률이나 SLA를 보장하지 않는다.

## 반복 실행과 배포

같은 멀티 20장을 provider별 3회 측정했다. full-path p95는 CPU 343.65 / 329.47 / 341.67ms,
CUDA 71.26 / 75.35 / 68.11ms다. 실행별 p95 중앙값은 CPU 341.67ms, CUDA 71.26ms다.
각 실행의 p50/p95/p99·표본 수·환경은 `diagnostics-cpu4`의 원본 보고서에 보존했다.
CPU/CUDA 상태와 class rank mismatch 0, bbox IoU 1.0, 최대 confidence 차이 0.000251이다.
목표 CPU p95 250ms, CUDA p95 70ms는 반복 중앙값 기준으로 아직 미달이다.

작업 기록과 재현 명령은 `artifacts/retraining/limited220-surfaces/experiment.json`, `commands.json`,
`detector/report.json`, `stress-comparison.json`, `diagnostics-cpu4/summary.json`,
`operational-20260818/ground-truth-comparison.json`에 있다. 후보는 최대 6개 계획 중 다섯 번째다.
단일 배포 구성은 `configs/versions/0.1.16.json`이며 원본·모델·평가 증빙 해시를 고정한다.
실제 빌드 및 packaged Worker 검증은 [최종 검증 기록](../diagnostics/scanner-0.1.16-final-verification.json)에
완료 상태와 제한을 기록한다. native inference의 영구 hang을 강제 종료하는 supervisor는 이번 버전에
추가하지 않았다. 기존 deadline·readiness 차단과 작업 종료 후 복구를 유지한다.

## 실제 EXE의 운영 115장 확인

CUDA 앱 bundle과 범용 CPU 설치 payload의 실제 `bixolon-worker.exe`를 HTTP로 검사했다.
양쪽 모두 115장 중 SEGMENTATION 111장·IMAGE_RECAPTURE 4장, 정답 승인 465개, UNKNOWN 38개,
객체 재촬영 1개다. 정답 객체 504개와 모두 대응하며 오승인·검출 누락·배경 추가 검출·UNKNOWN
Top-3 누락과 응답 계약 오류는 0이다. CPU/CUDA 상태·class rank가 같고 bbox IoU 1.0,
최대 confidence 차이는 0.000159다.

CPU의 기존 embedder 자동 스레드 설정은 이 PC에서 HTTP p95 2,104.89ms였다. detector 4 threads를
유지하고 embedder도 4 threads로 제한하자 p95 384.31ms로 줄었다. 115장의 상태·box·confidence는
자동 설정과 완전히 같았다. 설치 앱과 Worker ZIP 실행기는 이 4/4 설정을 사용한다.

| 실제 EXE / full-path 111회 | HTTP mean | HTTP p50 | HTTP p95 | HTTP p99 | Worker p95 |
|---|---:|---:|---:|---:|---:|
| CUDA | 52.09ms | 47.97ms | 83.86ms | 104.55ms | 72.79ms |
| CPU 4/4 | 186.05ms | 166.87ms | 384.31ms | 458.28ms | 366.73ms |

각 측정은 warmup 10회 후 운영 115장을 한 번 순회했으며 전체 재촬영 4장은 이 표에서 제외했다.
CUDA 측정 당시 설치 압축 등 호스트 부하가 있었으므로 단일 측정을 보편적 성능으로 해석하지
않는다. CPU HTTP p95 300ms 진단 목표는 미달이며 원본 보고서의 `passes=false`를 보존한다.
이는 오류·정답·parity 실패를 뜻하지 않으며 별도 배포 수명주기를 만들지 않는다.
