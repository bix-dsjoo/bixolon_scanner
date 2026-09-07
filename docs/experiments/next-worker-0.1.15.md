# 0.1.15 데이터 재학습·안전 승인 최적화

기준일: 2026-09-04

## 결론

`datasets`의 실촬영·운영·단일 객체 자료를 다시 감사하고, Detector·분류기·threshold 후보를
비교한 결과 `ssdlite320-v11-conservative-cosine-v1`을 0.1.15로 선택했습니다. 공개 API와
`DecisionPipeline`의 상태 우선순위는 바꾸지 않았습니다.

- Detector: class-agnostic SSDLite320, score `0.50`, uncertainty `0.45`, NMS IoU `0.40`
- primary: DINOv3 ConvNeXt-Tiny 192, 모든 정상 ROI를 한 batch로 실행
- detail/verifier: 기존 전역 위험·ambiguity 조건에 해당하는 ROI만 224/ViT-B/16 160 실행
- 승인: 정규화 classifier margin `0.80`; 미달은 `UNKNOWN`+Top-3

score `0.50`은 BIX 다중 객체 81개를 모두 회수했습니다. classifier margin `0.50`과 `0.70`은
빈 트레이를 `bread_15`로 승인한 사례가 있어 기각했습니다. `0.80`은 확인한 빈 트레이의 승인
오류를 모두 차단하면서 2026-08-27 실객체 138개의 승인을 유지했습니다. 그 집합에서 정답 객체의
최저 일치 confidence는 `0.8474`였습니다.

## 데이터 감사와 분리

감사 대상은 3,394개 이미지였습니다.

| root | 구성 |
|---|---:|
| `bix_bakery_dataset` | 단일 200, 빈 배경 10, 다중 20 |
| `bread_dataset` | fixed multi-object 1,410, 운영 2026-08-18 115/504, 08-27 69/138, 08-28 10/62 |
| `raw_data` | 20 class × 84장 = 1,680 |

SHA-256 중복은 342개 hash/1,122개 occurrence였고 모두 root 사이 중복이었습니다. manifest 생성기는
동일 물리 대상·capture session 계열을 group-aware로 다루고, 이미지 확장자만 읽도록 했습니다.
2026-08-27은 최종 Detector/분류기 학습에서 제외했지만 같은 SKU와 촬영 lineage를 공유하므로
독립 test가 아닌 시간 분리 진단으로만 해석합니다. 나머지 평가도 학습 또는 정책 비교와 겹치는
개발 진단입니다.

## 후보와 선택

분류기는 기존 head, raw-only, BIX scene, temporal operational, all-scene fine-tune의 192/224 경로를
비교했습니다. 최종 `classifier-all-scenes-v4b`는 고정 4 epoch의 마지막 checkpoint를 사용했고,
평가 결과로 epoch를 고르지 않았습니다. Detector도 v7~v11의 운영 recall·빈 배경 precision을
비교한 뒤, fixed 4 epoch 마지막 checkpoint인 `detector-all-scenes-v11`을 사용했습니다.

모델·정책 선택의 우선순위는 다음과 같았습니다.

1. 승인된 오분류와 빈 배경 오승인을 0으로 만든다.
2. 그 조건에서 실제 객체의 안전 승인율과 recall을 최대화한다.
3. 같은 판정이면 full-path 지연과 패키징 복잡도가 낮은 쪽을 선택한다.

OpenVINO CPU/GPU는 새 ConvNeXt ONNX의 동적 RoPE reshape를 컴파일하지 못해 배포 후보에서
제외했습니다. 앱 번들은 ONNX Runtime CUDA, KIOSK/POS 외부 SDK와 설치본은 범용 ONNX Runtime
CPU를 사용합니다. 둘은 같은 ONNX·metadata·Catalog·판정 정책을 사용합니다.

## 최종 개발 진단

| 집합 | 결과 | 지연 |
|---|---|---|
| BIX 20장/81객체, CPU | 81 승인, FP/FN/오승인 0 | mean 85.08ms, p95 87.73ms, p99 99.52ms |
| BIX 20장/81객체, CUDA | 81 승인, FP/FN/오승인 0 | mean 19.11ms, p95 21.04ms, p99 25.01ms |
| 운영 08-27 69장/138객체, CPU | 실제 객체 138 승인·오분류 0; 빈 이미지 3장의 raw false box 6개는 모두 승인 차단 | mean 62.09ms, p95 158.52ms |
| 운영 08-28 10장/62객체, CPU | 61 승인·오분류 0, FN 1 | mean 198.48ms, p95 217.86ms |
| fixed 300장/1,410객체, CPU | 승인 1,324·승인 오분류 0, UNKNOWN 22, FN 1, raw FP 1은 UNKNOWN | mean 178.50ms, p95 256.01ms |

08-27과 fixed300의 generic evaluator는 raw unmatched box 때문에 전체 target를 실패로 표시합니다.
이는 승인 오분류와 다른 지표입니다. trace를 별도로 확인한 결과 빈 배경·unmatched box의
`APPROVED`는 0이었습니다. raw detector 오류를 숨기거나 `ERROR`를 재촬영으로 바꾸지 않습니다.

CPU/CUDA 20장 parity는 최종 상태·class rank·bbox mismatch 0, 최소 bbox IoU `1.0`, 최대
confidence 차이 `2.98e-7`로 통과했습니다. 패키징된 CUDA Worker의 HTTP 왕복은 mean
`32.68ms`, p95 `46.19ms`; Worker 내부는 mean `21.87ms`, p95 `23.21ms`였습니다.

## 배포 검증

- CUDA 앱 bundle: manifest 201파일, Runtime/Catalog/CUDA source hash와 전체 파일 hash 검증
- packaged CUDA Worker: ready·정상 scan·손상/누락/미지원 입력·버전·privacy·dependency lock 통과
- CPU 외부 SDK와 Store Model ZIP: manifest와 실제 payload Worker smoke 통과
- CPU Setup EXE와 Worker ZIP: Inno Setup 6.7.3 컴파일, payload Worker smoke 통과
- Flutter 앱과 외부 SDK: analyze/test 통과 여부는 최종 검증 결과에 기록

Setup은 Authenticode 서명이 없습니다. SHA-256은 손상·파일 변경을 검출하지만 발행자 진위를
증명하지 않습니다.

## 한계

현재 자료는 독립 매장·독립 촬영 session의 blind test가 아닙니다. 따라서 오승인 0이라는 관측을
일반화 성능, 인증 또는 SLA로 표현하지 않습니다. 배포 후에는 실제 매장별 빈 트레이·혼동 SKU·조명
변화 자료를 별도 capture session으로 수집하고, 기존 threshold를 고정한 채 승인 오분류·승인율·
p50/p95/p99를 진단해야 합니다.
