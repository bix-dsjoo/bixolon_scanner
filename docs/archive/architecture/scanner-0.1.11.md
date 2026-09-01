# BIXOLON Bakery AI Scanner 0.1.11 번들

`configs/versions/0.1.11.json`이 배포 조합의 유일한 기준입니다. Python·Worker·Detector·Embedder·
판정 정책·Catalog와 Windows ProductVersion은 `0.1.11`, Flutter 내부 빌드는 `0.1.11+14`입니다.

Python `DecisionPipeline.scan()`이 입력 검증, detector 조기 종료, ROI batch, classifier fallback과
segmentation 조립 순서를 소유합니다. Worker는 HTTP·예외·동시성·provider 조립만 담당하고 실행
경로에서 PyTorch를 import하지 않습니다.

```mermaid
flowchart LR
    CONFIG["configs/versions/0.1.11.json"] --> VERIFY["source manifest·증빙 SHA-256 검증"]
    VERIFY --> META["Runtime/Catalog metadata 0.1.11"]
    META --> DETECTOR["class-aware SSDLite320 detector"]
    DETECTOR --> HARD{"프레임/검출 hard 품질 실패?"}
    HARD -->|예| RECAPTURE["IMAGE_RECAPTURE; classifier 미실행"]
    HARD -->|아니오| PRIMARY["모든 ROI 192 batch classifier"]
    PRIMARY --> RISK["위험 ROI index 계산"]
    RISK --> FALLBACK["선택 ROI만 224 재분류"]
    FALLBACK --> CONSENSUS["detector/classifier 안전 합의"]
    CONSENSUS --> RESULT["APPROVED / UNKNOWN+Top-3 / SEGMENT_RECAPTURE"]
```

Detector는 torchvision SSDLite320 MobileNetV3-Large 구조이며 외부 detector pretrained weight 없이
프로젝트 데이터로 학습했습니다. detector ensemble과 count verifier는 구성하지 않습니다. classifier
fallback은 전체 detection 문맥으로 neighbor mask crop을 만들되 위험 index의 tensor만 실행하고,
재검증하지 않은 ROI의 192 결과를 보존합니다.

Runtime의 detector 파일은 BSD-3-Clause torchvision 고지, classifier 파일은 DINOv3 License와 함께
배포합니다. YOLO 코드·weight와 AGPL 고지 파일은 `0.1.11` Runtime에 포함하지 않습니다.

`scripts/build_app.ps1 -Version 0.1.11`은 Runtime/Catalog/CUDA와 평가 증빙의 고정 해시를 확인하고
binary payload를 바꾸지 않은 채 실행 구성요소 version만 `0.1.11`로 맞춥니다. 최종 번들은
`version.json`, `provenance.json`, 전체 `bundle-manifest.json`과 Runtime metadata가 선언한 모든
license 파일을 포함합니다.

API와 null 규칙은 [Worker 연동 명세](../contracts/worker-integration-0.1.11.md), 평가 범위와 한계는
[YOLO-free 0.1.7 진단](../../experiments/yolo-free-0.1.7.md)을 따릅니다.
