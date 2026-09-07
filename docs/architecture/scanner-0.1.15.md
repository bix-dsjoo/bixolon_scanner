# BIXOLON Bakery AI Scanner 0.1.15 번들

`configs/archive/versions/0.1.15.json`이 배포 조합의 유일한 기준입니다. Python·Worker·Detector·Embedder·
판정 정책·Catalog와 Windows ProductVersion은 `0.1.15`, Flutter 내부 빌드는 `0.1.15+18`입니다.

0.1.15는 공개 상태 계약과 추론 순서를 유지하며, 매장과 SKU가 달라도 같은 모델 순서와 전역
threshold를 사용합니다. Detector·분류기 weight와 threshold는 `datasets` 재학습·비교 결과로
갱신했습니다.
매장별·SKU별 우회 규칙은 없으며 Detector class를 상품 승인에 사용하지 않습니다. Python
`DecisionPipeline.scan()`만 최종 상태를 결정하고 Worker는 HTTP·예외·동시성·provider 조립만
담당합니다. 실행 경로는 PyTorch를 import하지 않고 ONNX Runtime만 사용합니다.

![0.1.15 추론 파이프라인](../assets/scanner-0.1.15-inference-pipeline.svg)

## 고정 실행 순서

1. 1-class SSDLite320 MobileNetV3-Large가 객체 box와 프레임 촬영 품질을 판단합니다.
2. hard 품질 실패는 classifier를 호출하지 않고 `IMAGE_RECAPTURE`로 조기 종료합니다.
3. 정상 ROI 전체를 DINOv3 ConvNeXt-Tiny 192 primary에 한 batch로 전달합니다.
4. `UNKNOWN`, unsafe, dense scene 저신뢰, 긴 ROI 저신뢰의 전역 조건에 해당하는 ROI만 같은
   ConvNeXt-Tiny의 224 detail path로 재분류하고 Top-3 증거를 병합합니다.
5. 전역 ambiguity 범위에 든 경계 승인 후보만 Frozen DINOv3 ViT-B/16 160으로 독립 검증합니다.
6. `DecisionPipeline`의 단일 Safety Arbiter가 `APPROVED`, `UNKNOWN`+Top-3,
   `SEGMENT_RECAPTURE`를 결정합니다. 입력·모델·시스템 실패는 별도 `ERROR`입니다.

선택된 Detector·classifier graph·weight와 Catalog payload는 source manifest로 고정합니다.
Runtime metadata에 고정된 crop, 정규화, NMS, threshold와 입력 크기를 모든 매장에 동일하게
적용합니다. 새 SKU는 Catalog support를 추가하되 Detector를 SKU 분류기로 바꾸지 않습니다.

## 고정 실행 프로필

CUDA 앱 bundle은 Detector·classifier·detail·verifier를 ONNX Runtime CUDA로 실행합니다.
KIOSK/POS 외부 SDK와 Windows 설치본은 장비 호환성을 위해 모두 ONNX Runtime CPU로 실행합니다.
두 프로필은 같은 ONNX, 전처리, metadata, Catalog와 상태 정책을 사용하며 CPU/CUDA parity를
진단했습니다. 요청 처리 중에는 provider를 바꾸거나 결과를 섞지 않습니다.

`GET /health/ready`의 `provider`는 각각 `cuda` 또는 `cpu`입니다. 새 ConvNeXt ONNX의 동적 RoPE
reshape를 OpenVINO가 컴파일하지 못했으므로 OpenVINO profile은 0.1.15 배포에 포함하지 않습니다.

## 전수 verifier를 사용하지 않는 이유

415장 1,914개 ROI에 ViT를 전수 실행한 진단은 FP/FN·오승인·Top-3 누락 0을 유지했지만 다음과
같이 운영 목표를 훼손했습니다.

| 전수 verifier 정책 | 정답 승인율 | `SEGMENT_RECAPTURE` | CPU full-path p95 |
|---|---:|---:|---:|
| 단일 verifier 품질 거부를 재촬영 | 95.40% | 68 | 1,122.7ms |
| class 불일치만 승인 차단 | 97.86% | 5 | 894.7ms |

따라서 ViT 존재 자체가 아니라 동일한 전역 위험 선택 규칙이 운영 일관성의 기준입니다. 전수
검증은 `RECAPTURE`·`UNKNOWN`을 늘리고 99% 정답 승인 목표를 깨므로 활성화하지 않습니다. 이
판단은 현재 개발 회귀 데이터에 한정되며 독립 일반화 성능이나 SLA 인증이 아닙니다.

## 패키징

`scripts/build_app.ps1 -Version 0.1.15`은 Runtime/Catalog/CUDA와 평가 증빙의 고정 해시를 확인하고
binary payload를 바꾸지 않은 채 공개 실행 구성요소 version만 `0.1.15`으로 맞춥니다. 최종 번들은
`version.json`, `provenance.json`, `bundle-manifest.json`과 Runtime metadata가 선언한 license 파일을
포함합니다.

API와 null 규칙은 [Worker 연동 명세](../archive/contracts/worker-integration-0.1.15.md), 후보 비교와 평가
한계는 [0.1.15 데이터 재학습·안전 승인 최적화](../experiments/next-worker-0.1.15.md)를 따릅니다.
