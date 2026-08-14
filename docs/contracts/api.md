# Worker API 계약 1.0

## Endpoint

- `POST /v1/scan`: `image` 필드에 JPEG/PNG 한 장을 담은 multipart 요청
- `GET /health/live`: 프로세스 생존 확인
- `GET /health/ready`: 모델 package와 provider 준비 상태 확인. 준비 완료 응답은 `status`, `provider`, `worker_version`, `detector_version`, `classifier_version`을 포함합니다.

## 응답

최상위 응답은 `request_id`, `status`, `reason_codes`, `segmentations`, `processing_time_ms`, `worker_version`, `detector_version`, `classifier_version`을 포함합니다.

- 이미지 `status`: `SEGMENTATION`, `IMAGE_RECAPTURE`, `ERROR`
- segmentation `status`: `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE`
- `segmentations[]`: `segmentation_id`, 원본 픽셀 기준 `bbox`, `status`, `reason_codes`, `prediction`, `top3`, `confidence`
- `UNKNOWN`은 `prediction=null`이고 점수 내림차순 Top-3를 제공합니다. 승인 임계값 미만이면 `BELOW_APPROVAL_THRESHOLD`, 활성화된 포함 중복 검토 정책에 걸리면 `DETECTOR_CONTAINED_DUPLICATE`를 reason code로 사용합니다.
- `SEGMENT_RECAPTURE`는 `prediction=null`, 빈 `top3`, 하나 이상의 reason code를 가집니다.
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
8. 나머지 segmentation은 임계값 이상이면 `APPROVED`, 미만이면 `BELOW_APPROVAL_THRESHOLD` `UNKNOWN`과 Top-3입니다.
9. 하나 이상의 segmentation이 있으면 이미지 상태는 `SEGMENTATION`입니다. 포함 중복 `UNKNOWN`이 있으면 최상위 reason code에 `SEGMENT_DUPLICATE_REVIEW_REQUIRED`를 포함합니다.

Detector hard gate 조기 종료는 실행하지 않은 `classifier_version`을 `null`로 표시합니다. `worker_version`은 전체 조합, `detector_version`과 `classifier_version`은 개별 모델 버전입니다. 세 축은 독립적으로 semantic versioning합니다.

## 오류와 보안

- 입력 오류는 4xx, Worker·모델·시스템 장애는 5xx입니다.
- `ERROR`를 모델 판정인 `IMAGE_RECAPTURE`나 `SEGMENT_RECAPTURE`로 변환하지 않습니다.
- raw tensor, 전체 logits, 내부 예외, stack trace, 로컬 경로를 응답에 노출하지 않습니다.
- API 필드, enum, reason code 또는 의미를 바꿀 때 schema, README와 소비자 테스트를 함께 변경합니다.
