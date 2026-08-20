# Worker API 계약

제품 `0.0.2` 외부 개발자용 빵 목록과 요청·응답 예시는
[Worker 연동 명세](worker-integration-0.0.2.md)를 참조하십시오.

## Endpoint

- `POST /v1/scan`: `image` 필드에 JPEG/PNG 한 장을 담은 multipart 요청
- `GET /health/live`: 프로세스 생존 확인
- `GET /health/ready`: 모델 package와 provider 준비 상태 확인. 준비 완료 응답은 `status`, `provider`, `worker_version`, `detector_version`, `classifier_version`을 포함하며 2.0 runtime은 `embedder_version`, `detector_policy_version`, `classifier_policy_version`, `catalog_version`도 포함합니다.

## 응답

최상위 응답은 `request_id`, `status`, `reason_codes`, `segmentations`, `processing_time_ms`, `worker_version`, `detector_version`, `classifier_version`을 포함합니다. 2.0의 additive nullable 필드는 `embedder_version`, `detector_policy_version`, `classifier_policy_version`, `catalog_version`입니다.

- 이미지 `status`: `SEGMENTATION`, `IMAGE_RECAPTURE`, `ERROR`
- segmentation `status`: `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE`
- `segmentations[]`: `segmentation_id`, 원본 픽셀 기준 `bbox`, `status`, `reason_codes`, `prediction`, `top3`, `confidence`
- `UNKNOWN`은 `prediction=null`이고 점수 내림차순 Top-3를 제공합니다. 승인 임계값 미만이면 `BELOW_APPROVAL_THRESHOLD`, Top-1/Top-2가 모호하면 `CLASSIFIER_AMBIGUOUS_TOP2`, Catalog의 제한 SKU/pair이면 `CLASSIFIER_CATALOG_CONFLICT`, 활성화된 포함 중복 검토 정책에 걸리면 `DETECTOR_CONTAINED_DUPLICATE`를 reason code로 사용합니다.
- `SEGMENT_RECAPTURE`는 `prediction=null`, 빈 `top3`, 하나 이상의 reason code를 가집니다. 선택적 분류 정책이 안전한 Top-3를 보장하지 못한 경우 `CLASSIFIER_TOP3_UNSAFE`를 사용합니다.
- `IMAGE_RECAPTURE`와 `ERROR`는 빈 `segmentations`를 반환합니다.

정식 기계 판독 schema는 [scan-response.schema.json](../../schemas/scan-response.schema.json)입니다.

## 판정 순서

1. 입력을 검증하고 decode합니다.
2. Detector가 모든 segmentation 위치와 프레임 품질을 판단합니다.
3. detector hard gate가 실패하면 classifier를 호출하지 않고 `IMAGE_RECAPTURE`를 반환합니다.
4. 정상 ROI와 `classifier_confidence` 경계 ROI를 한 batch로 분류합니다.
5. classifier 품질 클래스는 해당 ROI를 `SEGMENT_RECAPTURE`로 만듭니다.
6. 경계 ROI의 Top-1 신뢰도가 승인 임계값 미만이면 해당 ROI를 `DETECTOR_BORDER_CLIPPED` `SEGMENT_RECAPTURE`로 만듭니다.
7. 패키지에서 포함 중복 검토 정책을 활성화한 경우, 거의 완전히 포함되고 같은 Top-1을 가진 ROI 쌍에서 detector 점수가 낮은 고신뢰 ROI는 `DETECTOR_CONTAINED_DUPLICATE` `UNKNOWN`과 Top-3입니다. ROI를 삭제하거나 재촬영으로 바꾸지 않습니다.
8. 나머지 segmentation은 승인 임계값 이상이면 `APPROVED`입니다.
9. 승인 임계값 미만이고 활성화된 선택적 분류 정책이 안전한 Top-3를 보장하지 못하면 `CLASSIFIER_TOP3_UNSAFE` `SEGMENT_RECAPTURE`입니다.
10. 그 밖의 승인 임계값 미만 segmentation은 `BELOW_APPROVAL_THRESHOLD` `UNKNOWN`과 점수 내림차순 Top-3입니다.
11. 하나 이상의 segmentation이 있으면 이미지 상태는 `SEGMENTATION`입니다. 포함 중복 `UNKNOWN`이 있으면 최상위 reason code에 `SEGMENT_DUPLICATE_REVIEW_REQUIRED`를 포함합니다.

Detector hard gate 조기 종료는 실행하지 않은 `classifier_version`, `embedder_version`,
`classifier_policy_version`, `catalog_version`을 `null`로 표시합니다. Detector policy는 실행됐으므로
`detector_policy_version`을 유지합니다. 현재 번들의 실행된 non-null Worker·모델·정책·Catalog
version은 모두 하나의 제품 version과 일치해야 합니다.

## 오류와 보안

- 입력 오류는 4xx, Worker·모델·시스템 장애는 5xx입니다.
- `ERROR`를 모델 판정인 `IMAGE_RECAPTURE`나 `SEGMENT_RECAPTURE`로 변환하지 않습니다.
- raw tensor, 전체 logits, 내부 예외, stack trace, 로컬 경로를 응답에 노출하지 않습니다.
- API 필드, enum, reason code 또는 의미를 바꿀 때 schema, README와 소비자 테스트를 함께 변경합니다.
