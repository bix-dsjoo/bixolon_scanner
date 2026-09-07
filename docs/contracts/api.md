# Worker API 계약

제품 `0.1.16` 외부 개발자용 빵 목록과 요청·응답 예시는
[Worker 연동 명세](worker-integration-0.1.16.md)를 참조하십시오.

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
- `SEGMENT_RECAPTURE`는 `prediction=null`, 빈 `top3`, 공통 reason code `SEGMENT_RECAPTURE_REQUIRED` 하나만 반환합니다.
- `IMAGE_RECAPTURE`는 빈 `segmentations`와 공통 reason code `IMAGE_RECAPTURE_REQUIRED` 하나만 반환합니다.
- `ERROR`는 빈 `segmentations`와 입력·시스템 오류 reason code를 반환하며 RECAPTURE로 변환하지 않습니다.

정식 기계 판독 schema는 [scan-response.schema.json](../../schemas/scan-response.schema.json)입니다.

## 판정 순서

1. 입력을 검증하고 decode합니다.
2. Detector가 모든 segmentation 위치와 프레임 품질을 판단합니다.
3. detector hard gate가 실패하면 classifier를 호출하지 않고 `IMAGE_RECAPTURE`와 공개 공통 reason `IMAGE_RECAPTURE_REQUIRED`를 반환합니다. `0.1.16`는 프레임 품질과 raw query의 근접·중복, 보강 증거가 있는 큰 제안을 이 단계에서 검사합니다. 큰 제안의 크기만으로는 재촬영하지 않습니다. 구체적인 detector 진단은 구조화 로그에만 남깁니다.
4. 활성 `0.1.16`의 class-agnostic Detector는 SKU를 직접 승인하지 않습니다. 정상 ROI와
   `classifier_confidence` 경계 ROI 전체를 DINOv3 ConvNeXt-Tiny 192에 한 batch로 전달합니다.
5. Runtime이
   `classifier_resolution_fallback`을 활성화하면 안전 경계 밖 ROI만 고해상도로 다시 분류하며,
   `selective_roi_only=true`이면 나머지 ROI의 결과를 보존합니다. 선택 crop도 전체 detection 문맥을
   사용하므로 neighbor-mask 의미는 바뀌지 않습니다. 선택 규칙은 객체 수뿐 아니라
   `minimum_box_aspect_ratio`로 극단적인 ROI 형태를 제한할 수 있습니다.
   `maximum_approval_score_decrease`를 지정한 규칙은 저해상도보다 고해상도 승인 점수가 크게
   떨어진 ROI를 `UNKNOWN`으로 낮춥니다. `fuse_unapproved_top3=true`이면 이 ROI의 Top-3 순위는 두
   해상도의 class-agnostic 후보 점수를 결합하며, `minimum_fallback_approval_score` 미만의 승인
   경계는 provider와 관계없이 `BELOW_APPROVAL_THRESHOLD`로 정규화합니다.
6. 전역 ambiguity 범위에 든 경계 승인 후보만 Frozen DINOv3 ViT-B/16 160 verifier로 검증합니다.
   활성 Runtime은 승인 후보 전수 검증과 단일 verifier 품질 거부 재촬영을 사용하지 않습니다.
7. 실험 metadata에서 adaptive `UNKNOWN` detector refinement를 활성화한 경우에만, 큰 이미지의 초기 분류가 승인되지 않았고 품질 재촬영도 아닌 ROI를 전체 detector ensemble로 한 번 재검출한 뒤 다시 한 batch로 분류합니다. 이 단계는 detector hard gate를 우회하거나 `ERROR`를 판정 상태로 바꾸지 않습니다. 활성 `0.1.16` Runtime은 이 옵션을 사용하지 않습니다.
8. classifier 품질 클래스는 해당 ROI를 `SEGMENT_RECAPTURE`로 만들고 공개 공통
   reason `SEGMENT_RECAPTURE_REQUIRED`를 사용합니다.
9. 경계 ROI의 Top-1 신뢰도가 승인 임계값 미만이면 해당 ROI를 `SEGMENT_RECAPTURE`로 만들고 같은 공통 reason을 사용합니다.
10. 패키지에서 포함 중복 검토 정책을 활성화한 경우, 거의 완전히 포함되고 같은 Top-1을 가진 ROI 쌍에서 detector 점수가 낮은 고신뢰 ROI는 `DETECTOR_CONTAINED_DUPLICATE` `UNKNOWN`과 Top-3입니다. ROI를 삭제하거나 재촬영으로 바꾸지 않습니다.
11. 나머지 segmentation은 승인 임계값 이상이면 `APPROVED`입니다.
12. 승인 임계값 미만이고 활성화된 선택적 분류 정책이 안전한 Top-3를 보장하지 못하면 공통 reason의 `SEGMENT_RECAPTURE`입니다.
13. 그 밖의 승인 임계값 미만 segmentation은 `BELOW_APPROVAL_THRESHOLD` `UNKNOWN`과 점수 내림차순 Top-3입니다.
14. 하나 이상의 segmentation이 있으면 이미지 상태는 `SEGMENTATION`입니다. 포함 중복 `UNKNOWN`이 있으면 최상위 reason code에 `SEGMENT_DUPLICATE_REVIEW_REQUIRED`를 포함합니다.

Detector hard gate 조기 종료는 실행하지 않은 `classifier_version`, `embedder_version`,
`classifier_policy_version`, `catalog_version`을 `null`로 표시합니다. Detector policy는 실행됐으므로
`detector_policy_version`을 유지합니다. 현재 번들의 실행된 non-null Worker·모델·정책·Catalog
version은 모두 하나의 제품 version과 일치해야 합니다.

Runtime metadata의 `detector_class_mode=class_agnostic`은 `detector_class_count=1`인 objectness
Detector를 뜻합니다. 이 모드에서는 Detector 출력 channel을 Catalog class ID로 해석하지 않으며,
Detector-class consensus·corroboration·class disagreement fallback을 함께 구성할 수 없습니다.
Classifier label과 per-class threshold 수는 Detector class 수와 독립적입니다. 따라서 새 SKU 추가는
Catalog/Classifier 확장으로 처리하고, 새 SKU도 기존 `bread/object` 시각 범위에 속하는 한 Detector
재학습을 요구하지 않습니다. 빵이 아닌 새 객체 종류나 촬영 분포가 추가돼 objectness recall이
변하는 경우에는 별도 Detector 재검증이 필요합니다.

`classifier_verification.verify_all_approved_candidates`는 기본값이 `false`입니다. `true`이면 승인
후보 전체를 verifier 대상으로 만듭니다. 과거 415장 진단에서 정답 승인율과 지연이 악화됐으며,
활성 `0.1.16`도 선택 검증을 유지합니다. 이 옵션은 공개 schema나 상태 종류를 바꾸지 않습니다.

`classifier_verification.unknown_recapture_on_dual_verifier_rejection`은 기본값이 `false`인 Runtime
metadata 옵션입니다. 활성화된 후보에서만 primary 승인 차단 ROI의 회전·독립 verifier가 모두
품질 실패를 반환하면 12번의 기존 공통 `SEGMENT_RECAPTURE` 경로를 사용합니다. 공개 reason code나
응답 schema는 추가하지 않습니다.

`classifier_verification.unknown_recapture_on_any_verifier_rejection`도 기본값이 `false`입니다.
활성화하면 primary가 `UNKNOWN`이거나 verifier 불일치로 승인이 차단된 ROI에서 회전·독립
verifier 중 하나만 품질 실패를 반환해도 `CLASSIFIER_TOP3_UNSAFE`
`SEGMENT_RECAPTURE`로 반환합니다. dual 옵션과 any 옵션을 동시에 활성화하지 않으며,
두 옵션 모두 활성 `0.1.16`의 판정 결과를 바꾸지 않습니다. 공개 reason code와 응답 schema는
변경되지 않습니다.

## 오류와 보안

- 입력 오류는 4xx, Worker·모델·시스템 장애는 5xx입니다.
- `ERROR`를 모델 판정인 `IMAGE_RECAPTURE`나 `SEGMENT_RECAPTURE`로 변환하지 않습니다.
- Detector/Classifier의 NaN·Inf 및 Detector 출력 shape 오류는 기존
  `MODEL_EXECUTION_FAILED` 5xx `ERROR`입니다. 비정상 출력을 빈 검출로 처리하지 않습니다.
- 소스 Worker는 디코딩과 추론을 단일 executor에서 실행합니다. 요청 시작 기준 deadline에서
  multipart 파싱에 이미 소비한 시간을 차감하고, 슬롯 대기·파일 읽기·디코딩·추론에 남은 시간을
  적용합니다. multipart 업로드 스트림 자체를 중단하는 제한은 ASGI 서버/상위 계층의 책임입니다.
- 응답 시간 초과 뒤에도 실행 중인 작업이 끝나기 전에는 슬롯을 재사용하지 않습니다.
  deadline을 넘긴 작업이 남아 있으면 `/health/ready`는 503 `not_ready`이고 `/health/live`는
  계속 생존 상태를 반환합니다. 작업 종료 후 readiness가 복구됩니다. 영구 정지한 native 호출의
  강제 종료·프로세스 재시작은 아직 구현하지 않았습니다.
- raw tensor, 전체 logits, 내부 예외, stack trace, 로컬 경로를 응답에 노출하지 않습니다.
- API 필드, enum, reason code 또는 의미를 바꿀 때 schema, README와 소비자 테스트를 함께 변경합니다.
