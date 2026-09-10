# BIXOLON Worker 연동 명세

- 적용 제품 버전: `0.2.0`
- 기본 주소: `http://127.0.0.1:8000`
- 기본 endpoint: `POST /v1/scan`
- multipart 필드: `image`

클라이언트는 Worker 시작 후 `/health/ready`를 polling하고, 반환된 모든 non-null 구성요소 version이
`0.2.0`인지 확인한 뒤 scan을 직렬로 전송합니다. JPEG/PNG 한 장만 허용하며 multipart boundary는
HTTP 라이브러리가 생성하도록 둡니다.

## 응답 상태

| 최상위 `status` | 의미 | `segmentations` |
|---|---|---|
| `SEGMENTATION` | 하나 이상의 객체 판정 | 1개 이상 |
| `IMAGE_RECAPTURE` | detector가 이미지 전체 재촬영을 요구 | 빈 배열 |
| `ERROR` | 입력·구성·모델·시스템 오류 | 빈 배열 |

`ERROR`를 재촬영으로 변환하지 마십시오. `IMAGE_RECAPTURE`는 classifier를 실행하지 않으므로
`classifier_version`, `embedder_version`, `classifier_policy_version`, `catalog_version`이
`null`입니다. `worker_version`, 실행한 `detector_version`과 `detector_policy_version`은
`0.2.0`입니다.

`segmentations[]`의 `status`는 `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE` 중 하나입니다.
`APPROVED`는 `prediction`을, `UNKNOWN`은 score 내림차순 Top-3를 반환합니다.
`SEGMENT_RECAPTURE`는 `prediction: null`, 빈 `top3`를 반환합니다. 클라이언트는 `confidence`로
threshold를 다시 계산하지 않고 Worker 상태를 최종 판정으로 사용합니다.

## 0.2.0 일관 classifier

단일 class-agnostic SSDLite320 detector가 모든 객체 위치와 프레임 품질을 판단합니다. hard 품질 조건이
실패하면 classifier를 호출하지 않고 `IMAGE_RECAPTURE`를 반환합니다. detector ensemble과 별도
`object_presence` 또는 count verifier는 구성하지 않습니다.

정상 ROI 전체를 DINOv3 ConvNeXt-Tiny 192 한 batch로 분류하고 전역 위험 조건에 해당하는 ROI만
224 detail path로 재분류합니다. 전역 ambiguity 범위에 든 경계 승인만 Frozen ViT-B/16 160으로
검증합니다. 매장별·SKU별 routing 예외와 Detector SKU 직접 승인은 없습니다. 공개 `confidence`는
classifier approval score이며 남은 저신뢰 결과는 `UNKNOWN`+Top-3 또는 `SEGMENT_RECAPTURE`입니다.
공개 reason code와 schema는 이전 계약과 같습니다.

정식 필드와 enum은 [API 계약](api.md), JSON Schema는
[scan-response.schema.json](../../schemas/scan-response.schema.json), 상태별 payload는
[0.2.0 예시](examples/0.2.0/)를 참조하십시오. 빵 식별에는 표시명 대신 `class_id`를 사용합니다.
