# BIXOLON Bakery AI Scanner 0.1.8 개발 검증

`0.1.8`은 `0.1.7`의 모델 graph·weight, Catalog payload, detector threshold와 공개 API를 유지하고
검증된 classifier 안전 정책을 활성화한 patch입니다. 승인 차단 `UNKNOWN` ROI의 회전·독립 verifier가
모두 품질 실패를 반환하면 불안전한 Top-3를 노출하지 않고 해당 segmentation을
`SEGMENT_RECAPTURE`로 처리합니다. API 필드, enum과 공개 reason code는 변경하지 않았습니다.

## 고정 입력과 payload

- Runtime source manifest SHA-256:
  `6d61d3fd15601af8891e920fbb7a091272229f40dfbb7c2f5eaf429a8b717915`
- Catalog source manifest SHA-256:
  `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7`
- CUDA source manifest SHA-256:
  `5e50ec41d94f6963b62249bfd8f0e38762e629dd5914db61424a43faed7cd1a5`
- source candidate:
  `yolo26-objectness-single3-consensus-presence-verifier-dual-verifier-recapture`

`scripts/build_app.ps1`의 source/변환 binary SHA 검증과 `bixolon bundle verify`를 통과했습니다.
Runtime과 Catalog의 실행 구성요소 version을 `0.1.8`로 다시 작성했습니다. Runtime metadata의
`unknown_recapture_on_dual_verifier_rejection`은 `true`이며 model graph·weight, adapter,
prototype와 support payload는 바꾸지 않았습니다.

## packaged Worker 회귀

동일 CUDA provider와 HTTP 경로에서 `0.1.7` baseline과 활성 정책 후보를 비교했습니다.

| 표본 | 결과 | 0.1.7 대비 |
|---|---|---|
| 기존 개발 회귀 415장 | GT 1,914, match 1,914, FP 0, FN 0, APPROVED wrong 0, Top-3 miss 0 | semantic prediction diff 0 |
| 2026-08-27 주석 69장 | GT 138, match 138, FP 0, FN 0, APPROVED wrong 0, Top-3 miss 0 | semantic prediction diff 0 |
| 2026-08-28 주석 10장 | GT 62, 수용 prediction 48, FP 0, detector FN 8 | 7번 `bread_04` Top-3 miss 1→0, `SEGMENT_RECAPTURE` |

비교 항목은 status, reason code, segmentation 수, bbox, 객체 상태, prediction, Top-3,
confidence와 detector 조기 종료의 version null pattern입니다. 보호 데이터 415+69장에서는
confidence를 제외한 비교 항목의 차이가 0이었습니다. confidence-only 차이는 415장에서 13장
(`3.05e-7` 이하), 69장에서 62장(`3.40e-6` 이하)이었습니다. 2026-08-28 7번만 의도대로 객체
상태와 Top-3가 `SEGMENT_RECAPTURE`로 바뀌었습니다.

415장 CUDA Worker 처리 p95는 baseline `105.559ms`, 후보 `100.098ms`였습니다. baseline은
packaged HTTP, 후보는 source FastAPI TestClient여서 절대 지연에는 harness 차이가 있습니다.
10% 증가 조건에는 걸리지 않았지만 단일 현재 PC 측정이며 SLA나 독립 성능 근거가 아닙니다.

## 앱·오류·호환 검증

- readiness와 정상 scan의 모든 non-null version이 `0.1.8`임을 확인했습니다.
- 손상 이미지 `422 CORRUPT_IMAGE`, 누락 필드 `422 MISSING_IMAGE_FIELD`, 미지원 형식
  `415 UNSUPPORTED_IMAGE_FORMAT`의 공통 `ERROR` 응답을 확인했습니다.
- Flutter 전체 테스트에서 촬영·파일 입력, 검수 undo/redo, 로그 저장·export와 기존 golden을
  변경 없이 확인했습니다.
- Python 전체 계약 테스트에서 detector 조기 종료, ROI batch, border/contained duplicate,
  Top-3, 네 응답 형태, provider fallback과 version null 규칙을 확인했습니다.

## 한계

415장, 69장과 2026-08-28 10장은 정책 선택·진단에 사용한 개발 자료이므로 독립 일반화 성능을
주장하지 않습니다. 2026-08-28의 detector 누락 8개는 `0.1.8`에서도 남으며 현재 object-presence
verifier는 부분 누락을 검출하지 못합니다. 고정 N100 source-candidate 진단은 confidence tolerance와
p95 진단 기준을 넘겨 원본 `passes=false`를 유지합니다. Setup은 Authenticode 서명이 없고 SHA-256은
손상·파일 변경만 탐지하며 발행자 진위를 인증하지 않습니다.

이전 `0.1.5` N100 결과는 참고용이며 변경된 `0.1.8` Runtime metadata와 동일한 입력을 측정하지
않았습니다. 따라서 `0.1.8` 설치물 생성에는 해당 파일을 재사용하지 않고, 현재 버전 N100 진단이
없으면 deployment provenance에 `NOT_RUN_FOR_SELECTED_VERSION`을 기록합니다.
