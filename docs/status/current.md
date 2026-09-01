# 현재 버전

기준일: 2026-09-01

현재 실행 조합은 `0.1.12` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.
공식 제품명은 `BIXOLON Bakery AI Scanner`, Flutter 내부 빌드는 `0.1.12+15`입니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.12` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.12` |
| Store Catalog | `0.1.12`, `CHECKSUM-SHA256` |
| Detector | class-aware SSDLite320 MobileNetV3-Large, torchvision BSD-3-Clause |
| Detector 가중치 | 외부 pretrained weight 없이 프로젝트 데이터로 학습 |
| classifier | 검증된 Detector class는 직접 판정, 위험·신규·저신뢰·중복 ROI만 DINOv3 cascade |
| source candidate | `ssdlite320-detector-primary-selective-dino-v2` |
| Runtime source manifest | `08b70842f64aa8647f835288fb84b2a496f588e426b2fb3b28d54f831a240462` |
| Catalog source manifest | `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7` |

## 0.1.12 변경

class-aware SSDLite320이 이미 출력하는 SKU class를 1차 분류로 사용합니다. 검증된 13개 class의
단독·고신뢰 ROI는 Detector score로 바로 판정하고, 혼동 이력이 있는 class, 신규 class, detector score
`0.98` 미만과 동일 class 중복 ROI만 기존 DINOv3 cascade로 전달합니다. 모델 graph·weight와 Catalog
payload는 `0.1.11`에서 바꾸지 않았으며 공개 API와 `IMAGE_RECAPTURE`, `APPROVED`, `UNKNOWN`+Top-3,
`SEGMENT_RECAPTURE`, `ERROR` 계약을 유지합니다.

## 정확도·성능 진단

| 표본 | 결과 |
|---|---|
| 기존 개발 회귀 415장 | GT/prediction/matched `1,914/1,914/1,914`, FP/FN 0, 승인 `1,905/1,905` 정답, 오승인·Top-3 실패·ERROR 0 |
| 2026-08-27 운영 69장 | GT/prediction/matched `138/138/138`, FP/FN·오승인·ERROR 0 |
| multi-object 300장 | GT/prediction/matched `1,410/1,410/1,410`, 승인 정답 1,401, UNKNOWN 4·Top-3 실패 0, SEGMENT_RECAPTURE 5, FP/FN·오승인·ERROR 0 |
| 개발 PC OpenVINO CPU 415장 | full-path p95 `99.139ms` (`0.1.11` 143.532ms 대비 -30.9%) |
| RTX 5080 CUDA 415장 | full-path p95 `66.131ms`, CPU/CUDA decision mismatch 0 |
| 개발 PC OpenVINO CPU multi300 | full-path p95 `96.934ms` (`0.1.11` 132.087ms 대비 -26.6%) |

415장 ROI 1,914개 중 671개(35.06%), multi300 ROI 1,410개 중 498개(35.32%)만 DINOv3를
실행했습니다. 현재 결과는 threshold와 직접 승인 class 선택에 사용한 개발·비열화 방지 자료이며 독립
test나 일반화 성능 인증이 아닙니다.

## N100 진단

동일 모델 binary로 모든 ROI에 DINOv3를 실행한 실제 Intel N100 100장 실측에서 CPU-only p95는
`676.568ms`, CPU detector + Intel iGPU embedder는 `469.471ms`였습니다. 동일 SHA-256 100장에
`0.1.12` 선택 정책을 적용하면 410개 ROI 중 143개(34.88%)만 DINOv3를 실행하며 개발 PC p95는
`97.644→72.857ms`였습니다. 모델 추론을 추가하지 않고 DINO batch를 줄이므로 기존 N100 실측에는
각각 `323.432ms`, `530.529ms`의 1,000ms 목표 여유가 있습니다.

이 근거는 후보 자체의 직접 N100 실측이 아니며 SLA나 인증으로 표현하지 않습니다. 원본은
[N100 device matrix](../diagnostics/n100-0.1.11-openvino-device-matrix.json), 판정과 한계는
[0.1.12 assessment](../diagnostics/n100-detector-primary-0.1.12-assessment.json)에 고정합니다. 상세 실패
분석과 재현 경로는 [Detector-first 선택 DINO 실험](../experiments/detector-primary-selective-dino-0.1.11.md)을
참조하십시오.
