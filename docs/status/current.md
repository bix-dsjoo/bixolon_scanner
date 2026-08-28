# 현재 버전

기준일: 2026-08-28

현재 실행 조합은 `0.1.8` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.
공식 제품명은 `BIXOLON Bakery AI Scanner`, Flutter 내부 빌드는 `0.1.8+11`입니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.8` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.8` |
| Store Catalog | `0.1.8`, `CHECKSUM-SHA256` |
| Detector | class-agnostic YOLO26 objectness, score `0.65`, NMS IoU `0.4` |
| 객체 존재 검증 | DINOv3 ViT-S/16 `object_presence`, confidence `0.54` |
| classifier 안전 정책 | dual verifier 품질 실패 시 `SEGMENT_RECAPTURE` |
| source candidate | `yolo26-objectness-single3-consensus-presence-verifier-dual-verifier-recapture` |
| Runtime source manifest | `6d61d3fd15601af8891e920fbb7a091272229f40dfbb7c2f5eaf429a8b717915` |
| Catalog source manifest | `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7` |

## 0.1.8 변경

`0.1.7`의 모델 graph·weight, Catalog payload와 detector threshold를 유지했습니다. 활성 Runtime의
`classifier_verification.unknown_recapture_on_dual_verifier_rejection`만 `true`로 바꿔, 승인 차단
ROI의 회전·독립 verifier가 모두 품질 실패인 경우 `UNKNOWN` Top-3 대신 `SEGMENT_RECAPTURE`를
반환합니다. 공개 API 필드·enum·reason code와 화면 구조는 바꾸지 않았습니다.

## 고정 회귀

| 표본 | 결과 |
|---|---|
| 2026-08-27 주석 69장 | GT 138, match 138, FP 0, FN 0, 빈 장면 6/6 재촬영 |
| 기존 415장 | GT 1,914, FP 0, FN 0, APPROVED wrong 0, Top-3 miss 0 |
| 415+69장 | status·reason·bbox·item status·prediction·Top-3 차이 0; confidence-only 최대 `3.40e-6` |
| 2026-08-28 10장 | Top-3 miss 1→0, 7번 `SEGMENT_RECAPTURE`; detector FN 8은 유지 |

위 자료는 threshold와 정책 선택에 사용한 개발·비열화 방지 자료이며 독립 test나 일반화 성능
근거가 아닙니다.

## N100 진단

공통 모델 payload를 사용한 이전 `0.1.5` N100 100장 실측에서 하이브리드 worker 평균/p95는
`440.767/729.284ms`, CPU-only는 `637.597/1003.835ms`였습니다. 속도비는 평균 `1.443배`,
p95 `1.375배`이고 semantic mismatch와 오류는 0건입니다.

최대 confidence 차이는 `0.0085055232`로 strict tolerance `1e-5`를 넘었고, 하이브리드 p95도
500ms 진단 기준을 넘으므로 원본 결과의 `passes=false`를 유지합니다. 이는 한 대의 N100 진단이며
SLA, 인증 또는 배포 승인으로 표현하지 않습니다.

원본은 [N100 device matrix](../diagnostics/n100-0.1.5-openvino-device-matrix.json), 후보와 source
해시는 [측정 패키지](../diagnostics/n100-0.1.5-measurement-package.json)에 고정합니다.

상세 결과와 한계는 [0.1.8 개발 검증 보고서](../evaluation/scanner-0.1.8.md), 과거 판단은
[버전 이력](../archive/version-history.md)에 기록합니다.
