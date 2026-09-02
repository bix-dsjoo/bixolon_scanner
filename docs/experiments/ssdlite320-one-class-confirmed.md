# 1-class SSDLite320 확정 실험 결과

## 범위

이 문서는 1-class objectness `SSDLite320 MobileNetV3-Large`에서 실제 실행으로 확인된 결과만
기록한다. 다른 detector 후보, 예상 성능, 후속 계획과 구조 권고는 포함하지 않는다.

이 모델은 실험 후보이며 활성 `0.1.12` Runtime 변경이나 독립 일반화 성능을 의미하지 않는다.

## 실행 구성

| 항목 | 값 |
|---|---|
| 구조 | torchvision SSDLite320 MobileNetV3-Large |
| 입력 | RGB float32 `[1, 3, 320, 320]` |
| 출력 | objectness `[1, 3234, 1]`, box `[1, 3234, 4]` |
| foreground class | 1 (`bread/object`) |
| 외부 pretrained detector weight | 사용하지 않음 |
| 초기값 | 프로젝트 class-aware checkpoint의 shape-compatible tensor 464개 전이 |
| 학습 epoch | 10 |
| batch | 32 |
| score threshold | 0.98 |
| NMS IoU | 0.4 |
| 모델 파일 | `detector.onnx`, 8,913,430 bytes |

## 학습 데이터 구성

| 데이터 | 원본 표본 | epoch 반복 |
|---|---:|---:|
| 실제 scene | 415장, 객체 1,914개 | 4회 |
| 운영 촬영 | 69장, 객체 138개 | 4회 |
| 합성 scene | 1,500장, 객체 5,835개 | 1회 |
| 빈 hard-negative | 10장 | 16회 |

## 검출 결과

IoU 0.5 기준이다.

| 평가 | 이미지 | 정답 객체 | 예측 객체 | 일치 | Recall | Precision | FP | FN | exact image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 실제415 | 415 | 1,914 | 1,914 | 1,914 | 100% | 100% | 0 | 0 | 415/415 |
| 운영69 | 69 | 138 | 138 | 138 | 100% | 100% | 0 | 0 | 69/69 |
| multi300 재확인 | 300 | 1,410 | 1,410 | 1,410 | 100% | 100% | 0 | 0 | 300/300 |

## ONNX Runtime CPU

개발 PC에서 고정 320, batch 1, `CPUExecutionProvider`, intra-op 4 threads, inter-op 1 thread,
warmup 20회 후 실제415 캐시 이미지를 측정했다. JPEG decode와 전체 분류 파이프라인은 포함하지
않는다.

| 표본 | mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 415 | 4.854ms | 4.851ms | 5.102ms | 5.307ms |

## 고정 식별자

| 파일 | SHA-256 |
|---|---|
| `detector.onnx` | `44a82f75e15bfe3bf8e59eafed5fb2deab3d6ff49667ca6725bf0ec2691511a5` |
| 학습 `report.json` | `be8788698a7de15be7065607e217a6f6848e892cc17296c593753ef9b41df5f3` |
| 실제415 manifest | `63582af7a010096d79056dbf959d7c3d912e8e765b4c750be19dd8d57027f58d` |
| 운영69 manifest | `43a9a6dca695b41d8832e33f3ae870ded395785f00a67b721004044927c88ba8` |
| CPU 측정 report | `e86f2af5fd4faa9e77bf6d1a5b87340a67874d1a724088ef4f442c299f9a4a77` |

## 증빙

- 학습 결과:
  `artifacts/experiments/class-agnostic-detector-0.1.11/ssdlite320-v9-objectness-from-v7/report.json`
- ONNX graph:
  `artifacts/experiments/class-agnostic-detector-0.1.11/ssdlite320-v9-objectness-from-v7/detector.onnx`
- multi300 재확인:
  `artifacts/experiments/class-agnostic-detector-0.1.12-recheck/multi300-cpu-p95-300.json`
- 운영69 재확인:
  `artifacts/experiments/class-agnostic-detector-0.1.12-recheck/operational69-cpu-p95-300.json`
- CPU 측정:
  `artifacts/experiments/one-class-detector-family-same-data-corrected-20260902/ssdlite320-onnx-cpu-report.json`

## 제한

- 실제415와 운영69는 학습 및 threshold 선택에 사용됐다.
- multi300은 실제415에 포함된 중복 데이터이므로 별도 holdout이 아니다.
- 모든 정확도 수치는 memorization/regression 확인이며 독립 일반화, 인증 또는 SLA 증빙이 아니다.
- CPU 수치는 개발 PC의 고정 입력 detector forward 범위이며 N100, 이미지 decode, classifier와 전체
  Worker latency를 대표하지 않는다.
