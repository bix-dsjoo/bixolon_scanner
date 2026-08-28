# BIXOLON Bakery AI Scanner 0.1.7 개발 검증

`0.1.7`은 `0.1.6`의 모델 graph·weight, Catalog payload, threshold와 판정 우선순위를 그대로
사용하면서 Python pipeline/runtime/Worker와 Flutter 검수·로그 코드의 책임을 분리한 유지보수
patch입니다. API 필드, enum, reason code와 화면 구조·golden은 변경하지 않았습니다.

## 고정 입력과 payload

- Runtime source manifest SHA-256:
  `4030730b71213bf30f65e370d585560df7d43b60fbe7346adbef3f582ed7f316`
- Catalog source manifest SHA-256:
  `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7`
- CUDA source manifest SHA-256:
  `5e50ec41d94f6963b62249bfd8f0e38762e629dd5914db61424a43faed7cd1a5`
- source candidate: `yolo26-objectness-single3-consensus-presence-verifier`

`scripts/build_app.ps1`의 source/변환 binary SHA 검증과 `bixolon bundle verify`를 통과했습니다.
Runtime과 Catalog의 실행 구성요소 version만 `0.1.7`로 다시 작성했으며 model graph·weight,
adapter, prototype와 support payload는 바꾸지 않았습니다.

## packaged Worker 회귀

OpenVINO CPU fallback 경로에서 최종 `0.1.7` Worker를 HTTP로 실행했습니다.

| 표본 | 결과 | 0.1.6 대비 |
|---|---|---|
| 2026-08-27 주석 69장 | `SEGMENTATION` 63, `IMAGE_RECAPTURE` 6, 오류 0 | semantic diff 0, 최대 confidence 차이 0 |
| 기존 개발 회귀 415장 | `SEGMENTATION` 411, `IMAGE_RECAPTURE` 4, 오류 0 | semantic diff 0, 최대 confidence 차이 0 |
| 69장 COCO | GT 138, match 138, FP 0, FN 0, 빈 장면 재촬영 6/6 | 동일 |

비교 항목은 status, reason code, segmentation 수, bbox, 객체 상태, prediction, Top-3,
confidence와 detector 조기 종료의 version null pattern입니다. 69장과 415장 모두 모든 비교 항목의
차이가 0이었습니다.

415장 full path Worker 처리시간은 평균/p50/p95/p99
`163.782/156.672/222.283/281.546ms`, 69장은
`87.787/70.403/154.987/185.276ms`였습니다. 단일 현재 PC 측정이며 SLA나 독립 성능 근거가
아닙니다.

## 앱·오류·호환 검증

- readiness와 정상 scan의 모든 non-null version이 `0.1.7`임을 확인했습니다.
- 손상 이미지 `422 CORRUPT_IMAGE`, 누락 필드 `422 MISSING_IMAGE_FIELD`, 미지원 형식
  `415 UNSUPPORTED_IMAGE_FORMAT`의 공통 `ERROR` 응답을 확인했습니다.
- Flutter 전체 테스트에서 촬영·파일 입력, 검수 undo/redo, 로그 저장·export와 기존 golden을
  변경 없이 확인했습니다.
- Python 전체 계약 테스트에서 detector 조기 종료, ROI batch, border/contained duplicate,
  Top-3, 네 응답 형태, provider fallback과 version null 규칙을 확인했습니다.

## 한계

415장은 `object_presence` verifier 학습에 사용된 개발 자료이고, 69장도 정책 진단에 사용된 운영
표본이므로 독립 일반화 성능을 주장하지 않습니다. 고정 N100 source-candidate 진단은 confidence
tolerance와 p95 진단 기준을 넘겨 원본 `passes=false`를 유지합니다. Setup은 Authenticode 서명이
없고 SHA-256은 손상·파일 변경만 탐지하며 발행자 진위를 인증하지 않습니다.
