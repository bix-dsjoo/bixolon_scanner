# Bixolon Multi-Bread Scanner Worker

한 장의 JPEG/PNG에서 여러 빵을 검출하고 20개 등록 클래스 중 하나로 분류하는 Windows용 Worker입니다. JPEG에 MPF/MPO 정보가 포함된 경우 첫 프레임을 입력 이미지로 사용합니다. 학습은 PyTorch, 배포는 ONNX Runtime만 사용합니다.

현재 운영 패키지 버전은 `0.1.1`, 데이터셋 버전은 `bread-43093242294f`, 승격 상태는 `production`입니다. `0.1.1`은 `0.1.0`과 동일한 ONNX 가중치를 사용하고 detector 불확실 객체 후처리만 추가합니다. 300장 단일 실행에서 `UNKNOWN` Top-3가 90%로 기준에 미달하고 299장이 정책 적합 세트와 중복된다는 사실을 보존한 채 프로젝트 책임자 요청에 따라 수동 예외 승격했습니다.

## 판정 계약

```mermaid
flowchart LR
    INPUT["JPEG/PNG"] --> DETECTOR["RT-DETRv2 bread detector"]
    DETECTOR -->|"hard 품질 실패"| RECAPTURE["RECAPTURE / items=[]"]
    DETECTOR -->|"정상 또는 경계 접촉 ROI batch"| CLASSIFIER["DINOv3 ConvNeXt Tiny classifier"]
    CLASSIFIER -->|"경계 접촉 + 낮은 신뢰도"| RECAPTURE
    CLASSIFIER --> ITEMS["객체별 APPROVED 또는 UNKNOWN"]
    ITEMS -->|"모두 승인"| APPROVED["최상위 APPROVED"]
    ITEMS -->|"하나 이상 보류"| UNKNOWN["최상위 UNKNOWN"]
```

- `APPROVED`: 모든 item의 Top-1 신뢰도가 승인 임계값 이상입니다.
- `UNKNOWN`: 미등록/OOD 판정이 아닙니다. 20종 중 Top-1 신뢰도가 임계값 미만인 선택적 보류이며 `prediction=null`, 점수 내림차순 Top-3와 Top-1 `confidence`를 제공합니다.
- `RECAPTURE`: 무검출, query 포화, 최소 크기 미달, detector 불확실 후보, 활성화된 count verifier/blur/exposure gate 실패입니다. 이 detector hard gate는 classifier를 호출하지 않고 `items=[]`, `model_versions.classifier=null`을 반환합니다.
- `DETECTOR_BORDER_CLIPPED`는 schema `1.1`의 `classifier_confidence` 정책에서 경계 접촉만으로 즉시 재촬영하지 않습니다. ROI를 분류한 뒤 경계 접촉 item의 Top-1 신뢰도가 기존 classifier 승인 임계값 미만일 때만 `RECAPTURE`를 반환합니다. 이 경우 classifier가 실행되었으므로 `model_versions.classifier`는 null이 아닙니다. schema `1.0`은 기존 `always_recapture` 동작을 유지합니다.
- `ERROR`: 입력·패키지·provider·추론 장애입니다. 모델 판정인 `RECAPTURE`로 바꾸지 않습니다.

item은 원본 픽셀 bbox를 가지며 `(y, x)` 순으로 정렬됩니다. 객체 수에 업무상 상한을 두지 않습니다. detector query 포화가 의심되면 `DETECTOR_CAPACITY_EXCEEDED`로 안전하게 재촬영을 요구합니다.

Detector 관련 `RECAPTURE` reason code는 다음과 같습니다.

| reason code | 의미 | classifier 실행 |
|---|---|---:|
| `DETECTOR_NO_OBJECT` | 승인 threshold 이상의 객체가 없음 | 아니요 |
| `DETECTOR_CAPACITY_EXCEEDED` | detector query 포화 가능성 | 아니요 |
| `DETECTOR_OBJECT_TOO_SMALL` | 최소 면적 미달 객체 | 아니요 |
| `DETECTOR_UNCERTAIN_OBJECT` | 낮은 detector score의 독립 후보가 존재 | 아니요 |
| `DETECTOR_COUNT_MISMATCH` | 선택적 count verifier와 detector 객체 수가 불일치 | 아니요 |
| `DETECTOR_COUNT_UNCERTAIN` | 선택적 count verifier 신뢰도 미달 | 아니요 |
| `DETECTOR_UNDEREXPOSED` / `DETECTOR_OVEREXPOSED` / `DETECTOR_BLUR` | 활성화된 촬영 품질 gate 실패 | 아니요 |
| `DETECTOR_BORDER_CLIPPED` | 경계 접촉 객체가 정책상 재촬영 대상 | `always_recapture`: 아니요, `classifier_confidence`: 예 |

## API

`POST /v1/scan`은 `image` multipart 필드에 JPEG/PNG 한 장을 받습니다. 스마트폰 JPEG가 MPO 컨테이너로 식별되면 첫 프레임만 처리합니다.

아래는 중간 `APPROVED` item 세 개를 생략한 축약 예시입니다. 전체 실측 응답은 `artifacts/reports/api-sample-unknown.json`에 있습니다.

```json
{
  "request_id": "196b91998cbd47c0abfdbe9f5bc0c2d6",
  "status": "UNKNOWN",
  "reason_codes": ["ITEM_BELOW_APPROVAL_THRESHOLD"],
  "items": [
    {
      "item_id": "item_001",
      "bbox": {"x": 241, "y": 666, "width": 1179, "height": 1457},
      "status": "APPROVED",
      "reason_codes": [],
      "prediction": {"class_id": "bread_19", "class_name": "Pastry Bread"},
      "top3": [],
      "confidence": 0.9999773502349854
    },
    {
      "item_id": "item_005",
      "bbox": {"x": 976, "y": 2110, "width": 1204, "height": 1011},
      "status": "UNKNOWN",
      "reason_codes": ["BELOW_APPROVAL_THRESHOLD"],
      "prediction": null,
      "top3": [
        {"class_id": "bread_13", "class_name": "Muffin", "confidence": 0.7369906306266785},
        {"class_id": "bread_06", "class_name": "Croissant", "confidence": 0.213413804769516},
        {"class_id": "bread_03", "class_name": "Waffle", "confidence": 0.03721670061349869}
      ],
      "confidence": 0.7369906306266785
    }
  ],
  "processing_time_ms": 142.7919,
  "model_versions": {"detector": "0.1.0", "classifier": "0.1.0"}
}
```

공개 필드는 `request_id`, `status`, `reason_codes`, `items`, `processing_time_ms`, `model_versions`입니다. 내부 예외, tensor, 경로와 전체 logits는 응답이나 기본 로그에 포함하지 않습니다.

버전 관리되는 JSON Schema는 `schemas/scan-response.schema.json`입니다. 상태별 null/빈 배열과 집계 의미는 Pydantic validator 및 소비자 계약 테스트가 추가로 강제합니다.

Health endpoint:

- `GET /health/live`
- `GET /health/ready`

## Flutter Windows 앱

작업자용 Product Scanner는 [`apps/product_scanner`](apps/product_scanner)에 있습니다. 단일 2-pane 화면에서 Windows 카메라 촬영 또는 JPEG/PNG 선택, `/v1/scan` 분석, Bounding Box와 상품 목록 연동, Top-3/로컬 상품 검색, 최종 확정과 로컬 Scan Log 저장을 처리합니다.

```powershell
cd apps\product_scanner
flutter pub get
flutter run -d windows
```

기본 Worker 주소는 `http://127.0.0.1:8000`이며 다른 호스트를 사용할 때는 `--dart-define=SCANNER_API_BASE_URL=http://host:port`를 전달합니다. Flutter 설치, 테스트, release build와 로컬 저장 위치는 앱 디렉터리의 README를 참고하십시오.

## 설치와 Worker 실행

```powershell
python -m pip install -e ".[cuda]"

$env:BIXOLON_PACKAGE_DIR = "artifacts\packages\bread-worker-0.1.1"
$env:BIXOLON_PROVIDER = "cuda"
$env:BIXOLON_CUDA_DLL_DIR = "C:\path\to\CUDA-and-cuDNN-bin"
bixolon-worker
```

Worker 런타임 의존성은 FastAPI, Pillow, NumPy, ONNX Runtime입니다. PyTorch, Transformers와 DINOv3 코드는 운영 Worker에 필요하지 않습니다.

두 ONNX session에는 동일 provider가 적용됩니다. `cuda` 강제 모드의 CUDA 초기화 실패는 시작 실패이며 조용히 CPU로 전환하지 않습니다. `auto`에서만 두 session을 함께 CPU로 다시 생성합니다. checksum 검증, session 생성, detector warm-up과 classifier ROI batch 1~7 warm-up이 끝난 뒤 readiness가 열립니다.

예시 환경 변수는 `configs/worker.env.example`에 있습니다.

## 데이터 manifest

원본 데이터는 저장소 밖 `C:\workspace\raw_data\bread_project`에 유지합니다.

```powershell
python -m pip install -e ".[training,test]"

bixolon-manifest `
  --dataset-root C:\workspace\raw_data\bread_project `
  --output-dir manifests\bread-v1
```

manifest는 상대 이미지 경로, 이미지 SHA-256, annotation/클래스, 촬영시각·카메라, `capture_session_id`, split, fold를 기록합니다. 현재 1,979 records이며 분류 보조 이미지 1,680장과 COCO 이미지 299장·객체 1,406개를 포함합니다.

`2026-07-21` 촬영분 94장·511개 객체는 최종 test로 격리했습니다. 이전 촬영분 205장·895개 객체는 시간 단위 촬영 session을 유지한 3-fold development 평가에 사용합니다. 독립 분류 이미지는 학습 보조에만 포함되고 평가는 COCO ROI에서 수행합니다.

원본, cache, checkpoint, ONNX와 benchmark 산출물은 `.gitignore` 대상입니다.

## 버전 관리 설정

재현 seed와 기본 hyperparameter는 `configs/training.json`에서 읽을 수 있습니다. 필수 데이터·출력 경로는 CLI에서 명시합니다.

```powershell
bixolon-train-detector --config configs\training.json `
  --manifest manifests\bread-v1\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\bread_project `
  --output-dir artifacts\detector\fold0 `
  --fold 0
```

각 학습 디렉터리의 `run.json`에는 seed, arguments, dependency 버전, device, 데이터 수, dataset version과 manifest checksum을 기록합니다.

## Classifier 학습·calibration

주력 backbone은 사용자가 라이선스를 승인받은 공식 `dinov3_convnext_tiny`입니다. 공식 코드는 revision `6876159a11b4df116f30f667f8c9888617df0751`로 고정했고, 승인 URL은 저장·로그하지 않습니다. 모델 package에는 architecture, revision, 가중치 파일명과 SHA-256만 기록합니다. DINOv2 Small은 비교 baseline으로 유지합니다.

```powershell
$dinoV3Weights = "C:\workspace\raw_data\model_cache\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

bixolon-cache-classifier `
  --manifest manifests\bread-v1\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\bread_project `
  --output-dir artifacts\cache\classifier-bread-v1-runtime `
  --train-margin-ratio 0.08 `
  --eval-margin-ratio 0.05

0..2 | ForEach-Object {
  bixolon-train-classifier --config configs\training.json `
    --manifest manifests\bread-v1\manifest.jsonl `
    --dataset-root C:\workspace\raw_data\bread_project `
    --fold $_ `
    --weights $dinoV3Weights `
    --cache-dir artifacts\cache\classifier-bread-v1-runtime `
    --output-dir "artifacts\classifier\dinov3-fold$_"
}
```

먼저 frozen backbone linear probe를 30 epochs 학습하고 마지막 두 stage를 20 epochs 미세조정합니다. 미세조정 모델은 validation accuracy가 유지되고 승인 coverage가 증가할 때만 fold checkpoint로 채택합니다.

fold별 runtime-crop logits를 합쳐 global temperature와 승인 threshold를 OOF에서만 정합니다.

```powershell
bixolon-evaluate `
  --predictions `
    artifacts\predictions\dinov3-fold0-runtime-validation.npz `
    artifacts\predictions\dinov3-fold1-runtime-validation.npz `
    artifacts\predictions\dinov3-fold2-runtime-validation.npz `
  --dataset-metadata manifests\bread-v1\metadata.json `
  --output artifacts\reports\classifier-dinov3-oof-runtime-crop.json
```

OOF 895개에서 temperature는 `0.3549453438`, approval threshold는 `0.9611232767`입니다. 승인 coverage `97.6536%`, 승인 precision `100%`, 95% 단측 false-approval upper bound `0.3422%`로 risk-control 조건을 만족했습니다.

최종 classifier는 모든 development ROI와 보조 분류 이미지로 다시 학습합니다.

```powershell
bixolon-train-classifier --config configs\training.json `
  --manifest manifests\bread-v1\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\bread_project `
  --weights $dinoV3Weights `
  --cache-dir artifacts\cache\classifier-bread-v1-runtime `
  --output-dir artifacts\classifier\dinov3-final `
  --final-training
```

## Detector 학습·OOF threshold

COCO의 20개 category를 단일 `bread` detector class로 통합해 Apache-2.0 `PekingU/rtdetr_v2_r18vd`를 640×640으로 미세조정합니다.

```powershell
bixolon-cache-detector `
  --manifest manifests\bread-v1\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\bread_project `
  --output-dir artifacts\cache\detector-bread-v1

0..2 | ForEach-Object {
  bixolon-train-detector --config configs\training.json `
    --manifest manifests\bread-v1\manifest.jsonl `
    --dataset-root C:\workspace\raw_data\bread_project `
    --fold $_ `
    --cache-dir artifacts\cache\detector-bread-v1 `
    --output-dir "artifacts\detector\fold$_"
}
```

각 validation 예측을 JSONL로 저장한 뒤 공통 OOF threshold 하나를 선택합니다.

```powershell
bixolon-aggregate-detector `
  --manifest manifests\bread-v1\manifest.jsonl `
  --predictions `
    artifacts\predictions\detector-fold0-validation.jsonl `
    artifacts\predictions\detector-fold1-validation.jsonl `
    artifacts\predictions\detector-fold2-validation.jsonl `
  --output artifacts\reports\detector-oof.json
```

공통 threshold는 `0.56`이며 OOF recall `99.7765%`, precision `99.8881%`, count accuracy `99.5122%`입니다. fold 1 best에서 시작해 development 전체를 20 epochs, learning rate `1e-6`로 최종 fitting한 checkpoint가 `artifacts/detector/final/best`입니다.

## ONNX export·검증

```powershell
bixolon-export --config configs\training.json `
  --detector-checkpoint artifacts\detector\final\best `
  --classifier-checkpoint artifacts\classifier\dinov3-final\best.pt `
  --calibration-report artifacts\reports\classifier-dinov3-oof-runtime-crop.json `
  --detector-evaluation-report artifacts\reports\detector-oof.json `
  --manifest-metadata manifests\bread-v1\metadata.json `
  --output-dir artifacts\packages\bread-worker-0.1.1

bixolon-parity `
  --package-dir artifacts\packages\bread-worker-0.1.1 `
  --detector-checkpoint artifacts\detector\final\best `
  --classifier-checkpoint artifacts\classifier\dinov3-final\best.pt `
  --image C:\path\to\parity.jpg `
  --cuda-dll-dir C:\path\to\CUDA-and-cuDNN-bin `
  --output artifacts\reports\parity-cuda.json
```

detector ONNX는 고정 640 입력, classifier ONNX는 동적 ROI batch입니다. JPEG는 metadata의 `jpeg_draft_size=1500`으로 DCT 단계에서 축소 디코딩하되 응답 bbox는 원본 픽셀 좌표로 복원합니다. `metadata.json`은 이 입력 정책, crop, 다단계 downscale, threshold, label map, semantic version, 데이터셋 버전, 라이선스·출처와 ONNX SHA-256을 포함합니다. 필수 metadata 누락, 지원하지 않는 schema 또는 checksum 불일치는 시작 실패입니다.

모델 패키지 schema `1.1`은 다음 선택적 안전 정책을 추가합니다. schema `1.0` 패키지는 기존 동작으로 계속 로드됩니다.

- `quality.border_policy=classifier_confidence`: 경계 접촉 bbox를 classifier 신뢰도까지 확인합니다.
- `detector.uncertainty_score_threshold`: 운영 detector threshold 아래에 있으면서 기존 검출과 겹치지 않는 독립 후보를 `DETECTOR_UNCERTAIN_OBJECT`로 차단합니다. `null`이면 비활성입니다.
- `detector.uncertainty_min_area_ratio`: 작은 저점수 노이즈를 제외하는 원본 이미지 대비 최소 후보 면적입니다. 후보 `0.1.1`은 score `0.20`, 면적 `0.039`, 기존 검출과의 IoU `0.5` 미만을 사용합니다.
- `count_verifier`: 별도 ONNX count model의 파일명, 입출력, 신뢰도 threshold를 정의합니다. 모델이 패키지에 없으면 비활성입니다. 299장 이미지는 이 모델의 학습에 사용하지 않습니다.

CPU와 CUDA parity는 모두 통과했습니다. CUDA의 NMS 후 detection 최소 IoU는 `0.99927`, 정규화 좌표 최대 오차는 `8.83e-5`, score 최대 오차는 `0.00256`이며 classifier Top-3 순위와 승인 상태가 일치했습니다.

## 최종 test와 benchmark

package metadata의 OOF threshold를 고정한 test 명령입니다. test에서 threshold를 다시 선택하지 않습니다.

```powershell
bixolon-evaluate-worker `
  --package-dir artifacts\packages\bread-worker-0.1.0 `
  --manifest manifests\bread-v1\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\bread_project `
  --mode test `
  --provider cuda `
  --cuda-dll-dir C:\path\to\CUDA-and-cuDNN-bin `
  --output artifacts\reports\worker-final-test-cuda.json

# 원본 COCO detection 전체를 파일명의 E/M/H 난이도로 분리 진단
bixolon-evaluate-difficulty `
  --package-dir artifacts\packages\bread-worker-0.1.1 `
  --dataset-root C:\workspace\bixolon_bakery_scanner\datasets\detection `
  --provider cuda `
  --match-iou-threshold 0.5 `
  --output artifacts\reports\worker-difficulty-0.1.1-cuda.json `
  --details-output artifacts\reports\worker-difficulty-errors-0.1.1-cuda.csv

bixolon-benchmark `
  --package-dir artifacts\packages\bread-worker-0.1.1 `
  --images artifacts\benchmark_images `
  --provider cuda `
  --cuda-dll-dir C:\path\to\CUDA-and-cuDNN-bin `
  --warmup 30 `
  --runs 1000 `
  --output artifacts\reports\benchmark-0.1.1-cuda.json
```

### 299장 정책 적합 평가

`C:\workspace\bixolon_bakery_scanner\datasets\detection`의 299장은 모델 학습에는 사용하지 않았지만 이번 low-score/면적 후처리 정책 선택에는 사용했습니다. 따라서 아래 결과는 독립 일반화 성능이나 최종 test KPI가 아니라 정책 적합 결과입니다. 후보 `0.1.1`의 결과는 다음과 같습니다.

| 난이도 | 이미지 | APPROVED | UNKNOWN | RECAPTURE |
|---|---:|---:|---:|---:|
| E | 100 | 99 | 0 | 1 |
| M | 99 | 91 | 5 | 3 |
| H | 100 | 86 | 12 | 2 |
| 전체 | 299 | 276 | 17 | 6 |

Detector 자체는 정답 bbox 1,406개 중 1,401개를 매칭했고 누락 5개, 오검출 1개로 기존 가중치와 동일합니다. 새 guard는 누락 5개가 포함된 4개 이미지를 모두 `DETECTOR_UNCERTAIN_OBJECT`로 차단했습니다. 정확히 검출된 이미지 중 `g20_b01_m_0710.jpg`, `g20_b02_e_0323.jpg` 2개도 추가 `RECAPTURE`됩니다. 기존 경계 오탐 3건은 계속 정상 처리되며, `RECAPTURE`를 제외한 `APPROVED` 박스 정답은 1,350/1,351입니다. 별도 count model은 없어 count verifier는 비활성입니다.

### 300장 운영 승격 평가

`C:\workspace\raw_data\bread_project_2`의 E/M/H 각 100장을 CUDA에서 이미지당 한 번씩 실행했습니다. 아래 결과 비율은 난이도별 전체 정답 객체 박스(E 410, M 500, H 500)를 공통 분모로 사용합니다. `인식 성공 + Top-3 Candidate + Candidate out + APPROVED 오인 + RECAPTURE 박스 + Segmentation 누락 = 100%`입니다. 속도는 파일 읽기를 제외한 디코딩부터 후처리까지의 이미지당 단일 실행 평균입니다.

| 난이도 | 전체 객체 | 인식 성공 | Top-3 Candidate | Candidate out | APPROVED 오인 | RECAPTURE 박스 | Segmentation 누락 | 추가 오검출 | 평균 속도 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 410 | 407 (99.2683%) | 0 (0%) | 0 (0%) | 0 (0%) | 3 (0.7317%) | 0 (0%) | 0 (0%) | 77.611ms |
| M | 500 | 474 (94.8000%) | 6 (1.2000%) | 0 (0%) | 0 (0%) | 20 (4.0000%) | 0 (0%) | 0 (0%) | 78.080ms |
| H | 500 | 472 (94.4000%) | 12 (2.4000%) | 2 (0.4000%) | 1 (0.2000%) | 13 (2.6000%) | 0 (0%) | 0 (0%) | 78.625ms |
| 전체 | 1,410 | 1,353 (95.9574%) | 18 (1.2766%) | 2 (0.1418%) | 1 (0.0709%) | 36 (2.5532%) | 0 (0%) | 0 (0%) | 78.105ms |

Detector 원시 결과에는 누락 5개와 오검출 1개가 있지만 해당 이미지가 전부 `RECAPTURE`됐습니다. 따라서 최종 운영 결과의 차단되지 않은 Segmentation 누락과 추가 오검출은 모두 0입니다. `RECAPTURE 박스` 36개에는 이 원시 누락 5개도 포함됩니다.

Segmentation 실패 4장은 모두 `DETECTOR_UNCERTAIN_OBJECT`로 `RECAPTURE`되어 차단되지 않은 누락·오검출은 0장입니다. 상태 분포는 `APPROVED 276 / UNKNOWN 18 / RECAPTURE 6`입니다. 단, 300장 중 299장은 앞선 정책 적합 세트와 SHA-256이 동일하고 유일한 신규 이미지 `M/M_100.jpg`도 생성 이미지이므로 독립 일반화 성능으로 해석하지 않습니다. 원본 보고서는 `artifacts/reports/worker-difficulty-bread-project-2-0.1.1-cuda.json`, 오류 상세는 `artifacts/reports/worker-difficulty-bread-project-2-errors-0.1.1-cuda.csv`입니다.

| 지표 | 최종 test/benchmark | 기준 | 결과 |
|---|---:|---:|---|
| Detector recall | 99.0215% | ≥99% | 통과 |
| Detector precision | 99.8028% | 보고 | - |
| Detector count accuracy | 95.7447% | 보고 | - |
| APPROVED precision | 99.7947% | ≥99.5% | 통과 |
| APPROVED precision 95% CI | 98.8462–99.9637% | 보고 | - |
| Approval coverage | 96.0552% | 보고 | - |
| UNKNOWN Top-3 accuracy | 84.2105% (16/19) | ≥95% | 수동 예외 승인(원시 기준 미달) |
| Overall Top-1 / Top-3 | 98.2213% / 99.2095% | 보고 | - |
| 3~7 객체 CUDA p50/p95/p99 | 68.02 / 90.61 / 99.44ms | p95 ≤100ms | 통과 |
| `0.1.1` 후보 full-path CUDA p50/p95/p99 | 70.80 / 83.48 / 86.74ms | p95 ≤100ms | 통과 |

Benchmark 환경은 Windows 11, Core Ultra 9 285K, RAM 64GB, RTX 5080 16GB, ONNX Runtime GPU `1.28.0`, NVIDIA driver `591.86`입니다. 30회 warm-up 후 4·5·6개 검출 이미지를 1,000회 재생했습니다. `0.1.1` 후보의 full-path는 800건이며 p95 `83.48ms`입니다. 전체 stage 기준 decode p95 `40.19ms`, detector p95 `29.40ms`, classifier p95 `16.35ms`, 판정 overhead p95 `0.28ms`입니다. 나머지 200건은 `DETECTOR_UNCERTAIN_OBJECT` 조기 종료였고 p95 `51.99ms`입니다. 원본 실측 보고서는 `artifacts/reports/benchmark-0.1.1-cuda.json`입니다.

CPU와 CUDA의 전체 test 집계 결과는 동일했습니다. CPU는 기능·상태 호환만 요구하며 지연 기준은 없습니다.

## 승격 결정 및 제한

`2026-08-10` 프로젝트 책임자 요청에 따라 `0.1.1`을 `promotion_status=production`으로 승격했습니다. 300장 평가의 `UNKNOWN` Top-3 accuracy는 20개 중 18개 정답 포함으로 `90%`이며 원시 95% 기준 미달입니다. 평가 독립성도 1/300에 불과하므로 두 항목 모두 측정값을 변경하지 않고 `manual_waiver`로 기록했습니다.

다음 제한사항은 여전히 유효합니다.

- `UNKNOWN` Top-3 accuracy는 18/20(90%)로 95% 기준에 미달합니다.
- 운영 승격 평가 300장 중 299장이 후처리 정책 적합 세트와 중복되며, 유일한 신규 이미지도 생성 이미지입니다.
- 실제 blur, exposure, 잘림 등 `RECAPTURE` 정답 데이터가 충분하지 않아 `RECAPTURE recall ≥99%`를 인증할 수 없습니다.
- 새로운 생산 개체, 매장, 카메라와 조명 분포 일반화를 검증하지 않았습니다.

새 환경과 촬영불량 데이터를 추가하고 dataset version을 올린 뒤 동일한 OOF calibration, 고정 test, ONNX parity와 benchmark 순서로 예외를 해소해야 합니다.

## 연구 근거

- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- [DINOv3 공식 저장소와 라이선스](https://github.com/facebookresearch/dinov3)
- [DINOv3 논문](https://arxiv.org/abs/2508.10104)
- [DINOv2 baseline](https://arxiv.org/abs/2304.07193)
- [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052)
- [ONNX Runtime CUDA Execution Provider와 I/O Binding](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
