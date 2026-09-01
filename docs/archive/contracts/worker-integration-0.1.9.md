# BIXOLON Worker 연동 명세

- 적용 제품 버전: `0.1.9`
- 기본 주소: `http://127.0.0.1:8000`
- 기본 endpoint: `POST /v1/scan`
- multipart 필드: `image`

클라이언트는 Worker 시작 후 `/health/ready`를 polling하고, 반환된 모든 non-null 구성요소 version이
`0.1.9`인지 확인한 뒤 scan을 직렬로 전송합니다. JPEG/PNG 한 장만 허용하며 multipart boundary는
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
`0.1.9`입니다.

`segmentations[]`의 `status`는 `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE` 중 하나입니다.
`APPROVED`는 `prediction`을, `UNKNOWN`은 score 내림차순 Top-3를 반환합니다.
`SEGMENT_RECAPTURE`는 `prediction: null`, 빈 `top3`를 반환합니다. 클라이언트는 `confidence`로
threshold를 다시 계산하지 않고 Worker의 상태를 최종 판정으로 사용합니다.

## 0.1.9 detector 검증

큰 detector proposal은 크기만으로 재촬영하지 않습니다. proposal 내부의 raw query containment
surplus 또는 제한된 선택 detection의 복수 중심점이 확인되거나, 기존 근접·query 중복 hard 조건이
충족되거나, 선택적 landscape 회전 입력이 기존 bbox를 모두 다시 찾으면서 객체를 하나 이상 추가
복구하면 `DETECTOR_UNCERTAIN_OBJECT` 진단으로 classifier 전에 `IMAGE_RECAPTURE`가 됩니다. 회전
검사는 기본 검출 5개와 bbox 면적·간격 조건을 모두 만족할 때만 실행합니다. 겹침 자체는 재촬영
조건이 아니며 기본 결과가 완전하면 정상 분류를 계속합니다. 외부 API의 공개 사유는 기존과 같은
`IMAGE_RECAPTURE_REQUIRED`이므로 클라이언트 분기 계약은 바뀌지 않습니다.

detector가 정상 detection을 하나 이상 반환하면 전체 프레임 `object_presence` verifier가 별도로
빵 존재 여부를 확인합니다. verifier confidence가 `0.54` 미만이거나 빈 장면 판정과 detection 존재가
불일치해도 classifier 전에 `IMAGE_RECAPTURE`가 됩니다. 이는 주변 집기·영수증 등을 detector가
빵으로 오검출한 빈 장면을 차단하는 내부 진단입니다. 외부 reason, 빈 `segmentations`, classifier·
Catalog 계열 version의 `null` 규칙은 다른 detector 조기 종료와 같습니다.

정식 필드와 enum은 [API 계약](../../contracts/api.md), JSON Schema는
[scan-response.schema.json](../../../schemas/scan-response.schema.json), 상태별 payload는
[0.1.9 예시](examples/0.1.9/)를 참조하십시오. 빵 식별에는 표시명 대신 `class_id`를 사용합니다.
