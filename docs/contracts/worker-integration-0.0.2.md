# BIXOLON Worker 연동 명세

- 적용 제품 버전: `0.0.2`
- 기본 주소: `http://127.0.0.1:8000`
- 응답 형식: `application/json`

이 문서는 BIXOLON Worker가 식별하는 빵 목록과 이미지 판정 API인
`POST /v1/scan`의 클라이언트 연동 계약을 설명합니다.

CPU 전달 패키지는 같은 PC의 Windows x64 환경만 지원합니다. `start-worker.ps1`은
`BIXOLON_PROVIDER=cpu`, 요청 제한시간 60초, localhost 전용 바인딩을 설정합니다. 모델 warm-up이
끝날 때까지 `/health/ready`를 polling한 뒤 scan을 전송하십시오. Worker는 동시에 하나의
inference만 실행하므로 클라이언트도 scan 요청을 직렬화해야 합니다.

### 실행 및 상태 endpoint

| Method | Path | 용도 |
|---|---|---|
| `GET` | `/health/live` | HTTP process 생존 확인. 정상일 때 `{"status":"alive"}`를 반환합니다. |
| `GET` | `/health/ready` | 모델·Catalog 검증과 warm-up 완료 확인. 준비 전에는 503, 완료 후에는 provider와 version을 반환합니다. |
| `POST` | `/v1/scan` | 이미지 한 장 판정. 아래의 공개 계약을 따릅니다. |
| `GET` | `/docs` | 실행 중인 Worker의 OpenAPI UI. 개발·수동 확인용입니다. |

Flutter 클라이언트는 process 시작 후 `/health/ready`를 최대 180초 polling하고, 개별 scan의 HTTP
timeout을 65초로 설정합니다. 이 65초는 Worker 내부 요청 제한시간 60초보다 길어야 합니다.

## 1. 빵 목록

Worker 응답의 상품 식별자는 `class_id`입니다. 클라이언트의 저장·조회·상품 매핑에는
`class_name` 대신 `class_id`를 사용해야 합니다. `class_name`은 Worker가 반환하는 영문 이름이며,
한국어 표시명은 클라이언트에서 매핑해 사용합니다. Worker 응답에는 `display_name_ko`가 포함되지
않습니다.

| `class_id` | Worker `class_name` | 한국어 표시명 |
|---|---|---|
| `bread_01` | `Walnut Donut` | 호두 도넛 |
| `bread_02` | `Croffle` | 크로플 |
| `bread_03` | `Waffle` | 와플 |
| `bread_04` | `Scone` | 스콘 |
| `bread_05` | `Half-moon Croissant` | 반달 크루아상 |
| `bread_06` | `Croissant` | 크루아상 |
| `bread_07` | `Flower Bread` | 꽃빵 |
| `bread_08` | `Almond Scone` | 아몬드 스콘 |
| `bread_09` | `Dinner Roll` | 디너 롤 |
| `bread_10` | `Sugar Donut` | 설탕 도넛 |
| `bread_11` | `Bagel` | 베이글 |
| `bread_12` | `Egg Tart` | 에그 타르트 |
| `bread_13` | `Muffin` | 머핀 |
| `bread_14` | `Burger` | 버거 |
| `bread_15` | `Sandwich` | 샌드위치 |
| `bread_16` | `Grain Campagne` | 곡물 깜빠뉴 |
| `bread_17` | `Almond Campagne` | 아몬드 깜빠뉴 |
| `bread_18` | `Mini Bread` | 미니 브레드 |
| `bread_19` | `Pastry Bread` | 페이스트리 브레드 |
| `bread_20` | `Plain Bread` | 플레인 브레드 |

## 2. `POST /v1/scan`

JPEG 또는 PNG 이미지 한 장을 받아 이미지 전체의 촬영 적합성과 이미지에 포함된 각 빵을
판정합니다.

### 2.1 요청

| 항목 | 값 |
|---|---|
| Method | `POST` |
| Path | `/v1/scan` |
| Content-Type | `multipart/form-data` |
| Multipart 필드명 | `image` |
| 허용 이미지 | JPEG, PNG |
| 최대 파일 크기 | 20 MiB |
| 최대 이미지 크기 | 50,000,000 pixels (`width × height`) |
| Worker 처리 제한시간 | CPU 전달 패키지 60초, 설정 미지정 시 기본 30초 |

요청 본문은 JSON이 아닙니다. 이미지 파일을 `image`라는 multipart 필드로 전송해야 합니다.
클라이언트 HTTP 라이브러리가 multipart boundary를 생성하도록 두고 `Content-Type` 헤더의 boundary를
직접 고정하지 마십시오.

```bash
curl -X POST "http://127.0.0.1:8000/v1/scan" \
  -H "accept: application/json" \
  -F "image=@scan.jpg;type=image/jpeg"
```

### 2.2 최상위 응답

| 필드 | 타입 | 설명 |
|---|---|---|
| `request_id` | string | 요청 추적 ID. 로그 문의 시 이 값을 사용합니다. |
| `status` | string | `SEGMENTATION`, `IMAGE_RECAPTURE`, `ERROR` 중 하나입니다. |
| `reason_codes` | string[] | 이미지 전체 또는 하위 판정의 집계 사유입니다. |
| `segmentations` | object[] | 검출된 빵별 판정입니다. 재촬영·오류 응답에서는 빈 배열입니다. |
| `processing_time_ms` | number | 네트워크 왕복시간을 제외한 Worker 처리시간입니다. |
| `worker_version` | string | Worker 제품 버전입니다. |
| `detector_version` | string 또는 null | Detector를 실행하지 못한 오류에서는 `null`일 수 있습니다. |
| `classifier_version` | string 또는 null | Detector 조기 종료에서는 `null`입니다. |
| `embedder_version` | string 또는 null | Detector 조기 종료에서는 `null`입니다. |
| `detector_policy_version` | string 또는 null | 실행된 Detector policy 버전입니다. |
| `classifier_policy_version` | string 또는 null | Detector 조기 종료에서는 `null`입니다. |
| `catalog_version` | string 또는 null | Detector 조기 종료에서는 `null`입니다. |

공개된 non-null 버전 값은 모두 같은 제품 버전이어야 합니다. `0.0.2` Worker에서는 실행된
구성요소의 버전도 모두 `0.0.2`입니다.

### 2.3 이미지 `status`

| `status` | HTTP | 의미 | 클라이언트 처리 |
|---|---:|---|---|
| `SEGMENTATION` | 200 | 하나 이상의 빵을 검출했습니다. | `segmentations`를 순서대로 처리합니다. |
| `IMAGE_RECAPTURE` | 200 | 이미지 전체의 재촬영이 필요합니다. | `reason_codes`를 안내하고 새 이미지를 요청합니다. |
| `ERROR` | 4xx 또는 5xx | 입력, 구성, 모델 또는 시스템 오류입니다. | 재촬영 판정으로 취급하지 말고 오류로 처리합니다. |

`IMAGE_RECAPTURE`와 `ERROR`는 항상 `segmentations: []`를 반환합니다.

### 2.4 `segmentations[]`

| 필드 | 타입 | 설명 |
|---|---|---|
| `segmentation_id` | string | 응답 내부 식별자입니다. 예: `segmentation_001` |
| `bbox.x` | integer | 원본 이미지 기준 좌측 상단 X 좌표입니다. |
| `bbox.y` | integer | 원본 이미지 기준 좌측 상단 Y 좌표입니다. |
| `bbox.width` | integer | 원본 이미지 기준 폭입니다. |
| `bbox.height` | integer | 원본 이미지 기준 높이입니다. |
| `status` | string | `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE` 중 하나입니다. |
| `reason_codes` | string[] | 해당 빵의 판정 사유입니다. |
| `prediction` | object 또는 null | 승인된 빵입니다. `class_id`, `class_name`을 포함합니다. |
| `top3` | object[] | `UNKNOWN`일 때 제공되는 최대 3개 후보입니다. |
| `confidence` | number | `0.0` 이상 `1.0` 이하의 판정 점수입니다. |

`confidence`는 클라이언트가 승인 임계값을 다시 계산하기 위한 값이 아닙니다. 클라이언트는
Worker가 반환한 `status`를 최종 판정으로 사용해야 합니다.

### 2.5 빵별 `status`

| `status` | `prediction` | `top3` | `reason_codes` | 의미 |
|---|---|---|---|---|
| `APPROVED` | 객체 | 빈 배열 | 빈 배열 | Worker가 빵을 자동 승인했습니다. |
| `UNKNOWN` | `null` | 1~3개 후보 | 정확히 1개 | 사용자가 후보를 확인해야 합니다. |
| `SEGMENT_RECAPTURE` | `null` | 빈 배열 | 1개 이상 | 해당 빵 영역을 안전하게 판정할 수 없어 재촬영이 필요합니다. |

`top3`는 `confidence` 내림차순으로 정렬됩니다. 각 후보는 다음 필드를 가집니다.

```json
{
  "class_id": "bread_02",
  "class_name": "Croffle",
  "confidence": 0.521
}
```

### 2.6 정상 승인 응답 예시

```json
{
  "request_id": "30f02474f43b4b8eab76290015df17da",
  "status": "SEGMENTATION",
  "reason_codes": [],
  "segmentations": [
    {
      "segmentation_id": "segmentation_001",
      "bbox": {
        "x": 120,
        "y": 84,
        "width": 310,
        "height": 246
      },
      "status": "APPROVED",
      "reason_codes": [],
      "prediction": {
        "class_id": "bread_06",
        "class_name": "Croissant"
      },
      "top3": [],
      "confidence": 0.987
    }
  ],
  "processing_time_ms": 72.143,
  "worker_version": "0.0.2",
  "detector_version": "0.0.2",
  "classifier_version": "0.0.2",
  "embedder_version": "0.0.2",
  "detector_policy_version": "0.0.2",
  "classifier_policy_version": "0.0.2",
  "catalog_version": "0.0.2"
}
```

### 2.7 사용자 확인 응답 예시

```json
{
  "request_id": "996d5d459ac242668b66245d4af796c3",
  "status": "SEGMENTATION",
  "reason_codes": ["SEGMENT_BELOW_APPROVAL_THRESHOLD"],
  "segmentations": [
    {
      "segmentation_id": "segmentation_001",
      "bbox": {
        "x": 74,
        "y": 51,
        "width": 280,
        "height": 231
      },
      "status": "UNKNOWN",
      "reason_codes": ["BELOW_APPROVAL_THRESHOLD"],
      "prediction": null,
      "top3": [
        {
          "class_id": "bread_02",
          "class_name": "Croffle",
          "confidence": 0.521
        },
        {
          "class_id": "bread_03",
          "class_name": "Waffle",
          "confidence": 0.312
        },
        {
          "class_id": "bread_19",
          "class_name": "Pastry Bread",
          "confidence": 0.167
        }
      ],
      "confidence": 0.521
    }
  ],
  "processing_time_ms": 75.804,
  "worker_version": "0.0.2",
  "detector_version": "0.0.2",
  "classifier_version": "0.0.2",
  "embedder_version": "0.0.2",
  "detector_policy_version": "0.0.2",
  "classifier_policy_version": "0.0.2",
  "catalog_version": "0.0.2"
}
```

### 2.8 이미지 재촬영 응답 예시

Detector가 이미지 전체의 촬영 부적합을 판정하면 Classifier를 실행하지 않습니다. 따라서
Classifier, Embedder, Classifier policy 및 Catalog 버전은 `null`입니다.

```json
{
  "request_id": "fceee521b42f460bbfc09f5979cf4e89",
  "status": "IMAGE_RECAPTURE",
  "reason_codes": ["DETECTOR_BLUR"],
  "segmentations": [],
  "processing_time_ms": 31.246,
  "worker_version": "0.0.2",
  "detector_version": "0.0.2",
  "classifier_version": null,
  "embedder_version": null,
  "detector_policy_version": "0.0.2",
  "classifier_policy_version": null,
  "catalog_version": null
}
```

### 2.9 오류 응답과 HTTP 상태 코드

| HTTP | `reason_codes` | 의미 |
|---:|---|---|
| 413 | `IMAGE_TOO_LARGE` | 파일 크기 또는 이미지 픽셀 수 제한을 초과했습니다. |
| 415 | `UNSUPPORTED_IMAGE_FORMAT` | JPEG/PNG가 아닌 형식입니다. |
| 422 | `MISSING_IMAGE_FIELD` | multipart의 `image` 필드가 없습니다. |
| 422 | `CORRUPT_IMAGE` | 이미지가 손상됐거나 디코딩할 수 없습니다. |
| 500 | `MODEL_EXECUTION_FAILED` | 모델 실행 또는 요청 처리시간 초과입니다. |
| 500 | `WORKER_ERROR` | 공개 가능한 상세 사유가 없는 Worker 내부 오류입니다. |

`MODEL_PACKAGE_INVALID`는 Runtime/Catalog 구성 또는 checksum 검증 실패, `PROVIDER_INITIALIZATION_FAILED`는
CPU provider 초기화 실패를 뜻합니다. 둘 다 시작 실패 reason code이며 정상적으로 시작되지 않은
Worker에는 scan 요청을 보내지 않습니다. `PROVIDER_INITIALIZATION_FAILED`가 HTTP 응답 가능한 시점에
발생하면 503을 사용합니다. `CATALOG_CONFUSABLE_PAIR`는 Catalog provenance에만 쓰이며 scan 응답의
reason code가 아닙니다.

오류 응답은 가능한 경우 아래의 공통 JSON 구조를 사용합니다. 내부 예외, stack trace, 로컬 경로는
응답에 포함하지 않습니다.

```json
{
  "request_id": "2ebd0edfe04c48af9e71c69f2c83ab45",
  "status": "ERROR",
  "reason_codes": ["CORRUPT_IMAGE"],
  "segmentations": [],
  "processing_time_ms": 2.314,
  "worker_version": "0.0.2",
  "detector_version": null,
  "classifier_version": null,
  "embedder_version": null,
  "detector_policy_version": null,
  "classifier_policy_version": null,
  "catalog_version": null
}
```

### 2.10 판정 reason code

이미지 전체 재촬영 사유:

- `DETECTOR_CAPACITY_EXCEEDED`
- `DETECTOR_NO_OBJECT`
- `DETECTOR_OBJECT_TOO_SMALL`
- `DETECTOR_BORDER_CLIPPED`
- `DETECTOR_UNDEREXPOSED`
- `DETECTOR_OVEREXPOSED`
- `DETECTOR_BLUR`
- `DETECTOR_UNCERTAIN_OBJECT`
- `DETECTOR_COUNT_UNCERTAIN`
- `DETECTOR_COUNT_MISMATCH`

`UNKNOWN` 사유:

- `BELOW_APPROVAL_THRESHOLD`
- `CLASSIFIER_AMBIGUOUS_TOP2`
- `CLASSIFIER_CATALOG_CONFLICT`
- `DETECTOR_CONTAINED_DUPLICATE`

`SEGMENT_RECAPTURE` 사유:

- `CLASSIFIER_QUALITY_CLASS`
- `CLASSIFIER_OUT_OF_CATALOG`
- `CLASSIFIER_TOP3_UNSAFE`
- `DETECTOR_BORDER_CLIPPED`

`SEGMENTATION` 최상위 집계 사유:

- `SEGMENT_BELOW_APPROVAL_THRESHOLD`
- `SEGMENT_DUPLICATE_REVIEW_REQUIRED`
- `SEGMENT_RECAPTURE_REQUIRED`

클라이언트가 알지 못하는 새 reason code를 받더라도 응답 전체를 실패 처리하지 않아야 합니다.
판정 분기는 reason code 문자열이 아니라 이미지 `status`와 빵별 `status`를 기준으로 구현해야 합니다.

정식 응답 JSON Schema는 [scan-response.schema.json](../../schemas/scan-response.schema.json)을
참조하십시오.
