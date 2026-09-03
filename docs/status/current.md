# 현재 버전

기준일: 2026-09-03

현재 실행 조합은 `0.1.14` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.
공식 제품명은 `BIXOLON Bakery AI Scanner`, Flutter 내부 빌드는 `0.1.14+17`입니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.14` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.14` |
| Store Catalog | `0.1.14`, `CHECKSUM-SHA256` |
| Detector | 1-class SSDLite320 MobileNetV3-Large, torchvision BSD-3-Clause |
| 주 분류 | DINOv3 ConvNeXt-Tiny 192, 정상 ROI 전체 batch |
| 선택 상세 | 같은 ConvNeXt-Tiny 224, 전역 위험 ROI만 |
| 선택 검증 | Frozen DINOv3 ViT-B/16 160, 전역 ambiguity ROI만 |
| source candidate | `ssdlite320-consistent-evidence-v1` |
| Runtime source manifest | `b4df35e975541485845b196521e44cb540814e47002191ed728bd2171615dc48` |
| Catalog source manifest | `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7` |

## 0.1.14 변경

0.1.13의 판정 구조, 모델, 전처리, threshold와 API를 그대로 유지합니다. Windows 기본 실행은
OpenVINO CPU detector + Intel GPU classifier로 고정하며, GPU 초기화·warm-up 실패 시 전체
classifier session을 OpenVINO CPU로 다시 만들고 시작합니다. 설치 빌드는 N100 device matrix를
요구하지 않으며 요청 처리 중 provider 전환은 하지 않습니다.

## 정확도·성능 진단

| 표본 | 결과 |
|---|---|
| 개발 회귀 415장 | GT/prediction/matched `1,914/1,914/1,914`, FP/FN 0, 승인 `1,898/1,898` 정답, 오승인·Top-3 누락·ERROR 0 |
| 운영 촬영 69장 | GT/prediction/matched `138/138/138`, 승인 138 정답, FP/FN·오승인·Top-3 누락·ERROR 0 |
| 동일 정책 multi-object 300장 | GT/prediction/matched `1,410/1,410/1,410`, 승인 1,398 정답, UNKNOWN 8·Top-3 누락 0, SEGMENT_RECAPTURE 4, FP/FN·오승인·ERROR 0 |
| 개발 PC CPU 415장 | full-path p95 `440.164ms`, p99 `574.080ms`, 411 samples |
| 개발 PC CPU 운영 69장 | full-path p95 `354.287ms`, p99 `469.849ms`, 63 samples |
| 개발 PC OpenVINO CPU 415장 | Worker full-path p95 `191.085ms`, HTTP p95 `211.098ms`, 411 samples |
| 개발 PC OpenVINO CPU 운영 69장 | Worker full-path p95 `143.720ms`, HTTP p95 `167.376ms`, 63 samples |

415장 정답 승인율은 `99.164%`, 운영 69장은 `100%`입니다. 전수 ViT 후보는 정답 승인율을
`95.40%` 또는 `97.86%`로 낮추고 p95를 `894.7~1,122.7ms`로 늘려 기각했습니다. 선택 정책은
목표 다섯 개를 현재 진단에서 만족하지만 학습·정책 선택과 겹치는 데이터이므로 독립 일반화 성능,
인증 또는 SLA로 표현하지 않습니다.

상세 구조와 새 매장 검증 계획은 [0.1.13 일관 추론 결정](../experiments/consistent-evidence-0.1.13.md),
동일 판정을 유지한 provider·thread 비교는
[0.1.12·0.1.13 운영 일관성 성능 최적화](../experiments/consistent-performance-0.1.13.md),
공개 상태와 null 규칙은 [API 계약](../contracts/api.md)을 참조하십시오.
