# BIXOLON Bakery AI Scanner 0.1.16 번들

`configs/versions/0.1.16.json`이 배포 조합의 유일한 기준입니다. Python·Worker·Detector·Embedder·
판정 정책·Catalog와 Windows ProductVersion은 `0.1.16`, Flutter 내부 빌드는 `0.1.16+19`입니다.

0.1.16는 공개 상태 계약과 추론 순서를 유지하며, 매장과 SKU가 달라도 같은 모델 순서와 전역
threshold를 사용합니다. 학습 원본은 단일 200장과 멀티 20장으로 고정하며 승인 threshold 0.80을 유지합니다.
매장별·SKU별 우회 규칙은 없으며 Detector class를 상품 승인에 사용하지 않습니다. Python
`DecisionPipeline.scan()`만 최종 상태를 결정하고 Worker는 HTTP·예외·동시성·provider 조립만
담당합니다. 실행 경로는 PyTorch를 import하지 않고 ONNX Runtime만 사용합니다.

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
reshape를 OpenVINO가 컴파일하지 못했으므로 OpenVINO profile은 0.1.16 배포에 포함하지 않습니다.

## 선택 검증

전역 ambiguity 조건에 해당하는 경계 승인만 verifier로 검증합니다. 전수 verifier나 단일
verifier 품질 실패에 의한 재촬영 확대는 비활성입니다. 원본 220장은 같은 실물을 반복 촬영한
자료이므로 학습 진단을 독립 일반화 성능으로 표현하지 않습니다.

## 패키징

`scripts/build_app.ps1 -Version 0.1.16`은 Runtime/Catalog/CUDA와 평가 증빙의 고정 해시를 확인하고
binary payload를 바꾸지 않은 채 공개 실행 구성요소 version만 `0.1.16`으로 맞춥니다. 최종 번들은
`version.json`, `provenance.json`, `bundle-manifest.json`과 Runtime metadata가 선언한 license 파일을
포함합니다.

API와 null 규칙은 [Worker 연동 명세](../archive/contracts/worker-integration-0.1.16.md), 후보 비교와 평가
한계는 [0.1.16 데이터 재학습·안전 승인 최적화](../experiments/next-worker-0.1.16.md)를 따릅니다.
