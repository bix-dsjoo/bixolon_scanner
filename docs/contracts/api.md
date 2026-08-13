# Worker API 계약

## Endpoint

- `POST /v1/scan`: `image` 필드에 JPEG/PNG 한 장을 담은 multipart 요청
- `GET /health/live`: 프로세스 생존 확인
- `GET /health/ready`: 모델 package와 provider 준비 상태 확인

## 응답

최상위 응답은 `request_id`, `status`, `reason_codes`, `items`, `processing_time_ms`, `model_versions`를 포함합니다. `status`는 `APPROVED`, `UNKNOWN`, `RECAPTURE`, `ERROR` 중 하나입니다.

각 `items[]`는 `item_id`, 원본 픽셀 기준 `bbox`, `status`, `reason_codes`, `prediction`, `top3`, `confidence`를 포함하며 item 상태는 `APPROVED` 또는 `UNKNOWN`입니다. `UNKNOWN`의 `top3`는 점수 내림차순입니다.

정식 기계 판독 schema는 [scan-response.schema.json](../../schemas/scan-response.schema.json)입니다.

## 판정 순서

1. 입력을 검증하고 decode합니다.
2. Detector가 객체 위치와 프레임 품질을 판단합니다.
3. hard gate가 실패하면 classifier를 호출하지 않고 `RECAPTURE`를 반환합니다.
4. 정상 ROI와 `classifier_confidence` 경계 ROI를 한 batch로 분류합니다.
5. classifier 품질 클래스는 전역 `RECAPTURE`입니다.
6. 경계 ROI의 Top-1 신뢰도가 승인 임계값 미만이면 `DETECTOR_BORDER_CLIPPED` 전역 `RECAPTURE`입니다.
7. 일반 item은 임계값 이상이면 `APPROVED`, 미만이면 `UNKNOWN`과 Top-3입니다.
8. 모든 item이 승인일 때만 최상위 상태가 `APPROVED`입니다.

Detector hard gate 조기 종료 응답은 실행하지 않은 classifier 버전을 `null`로 표시합니다. 분류 후 경계 정책이 실패한 응답은 실행한 classifier 버전을 표시합니다. `RECAPTURE`와 `ERROR`는 빈 `items`를 반환합니다.

## 오류와 보안

- 입력 오류는 4xx, Worker·모델·시스템 장애는 5xx입니다.
- `ERROR`를 모델 판정인 `RECAPTURE`로 변환하지 않습니다.
- raw tensor, 전체 logits, 내부 예외, stack trace, 로컬 경로를 응답에 노출하지 않습니다.
- API 필드, enum, reason code 또는 의미를 바꿀 때 schema, README와 소비자 테스트를 함께 변경합니다.
