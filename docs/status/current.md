# 현재 버전

기준일: 2026-08-28

현재 실행 조합은 `0.1.6` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.
공식 제품명은 `BIXOLON Bakery AI Scanner`, Flutter 내부 빌드는 `0.1.6+9`입니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.6` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.6` |
| Store Catalog | `0.1.6`, `CHECKSUM-SHA256` |
| Detector | class-agnostic YOLO26 objectness, score `0.65`, NMS IoU `0.4` |
| 객체 존재 검증 | DINOv3 ViT-S/16 `object_presence`, confidence `0.54` |
| source candidate | `yolo26-objectness-single3-consensus-presence-verifier` |
| Runtime source manifest | `4030730b71213bf30f65e370d585560df7d43b60fbe7346adbef3f582ed7f316` |
| Catalog source manifest | `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7` |

## 0.1.6 변경

큰 proposal은 크기만으로 재촬영하지 않고 raw-query surplus 또는 복수 detection 중심점이 보강할
때만 crowding hard gate를 적용합니다. 정상 detection이 있으면 전체 프레임
`object_presence` verifier가 빈 장면 오검출을 classifier 전에 차단합니다.

Windows OpenVINO 구성은 CPU Detector와 Intel GPU verifier를 병렬 실행하고 모든 Embedder를 Intel
GPU에 배치합니다. GPU 초기화가 실패하면 이미 생성한 CPU Detector·verifier·Embedder 세션을
사용하는 명시적 CPU fallback으로 재조립합니다.

## 고정 회귀

| 표본 | 결과 |
|---|---|
| 2026-08-27 주석 69장 | GT 138, match 138, FP 0, FN 0, 빈 장면 6/6 재촬영 |
| 기존 415장 | status·reason·bbox·prediction·Top-3·confidence·version null pattern 차이 0 |
| packaged CPU fallback 415+69장 | 기준 응답 대비 semantic/confidence 차이 0 |

위 자료는 threshold와 정책 선택에 사용한 개발·비열화 방지 자료이며 독립 test나 일반화 성능
근거가 아닙니다.

## N100 진단

동일 모델·정책 후보의 N100 100장 실측에서 하이브리드 worker 평균/p95는
`440.767/729.284ms`, CPU-only는 `637.597/1003.835ms`였습니다. 속도비는 평균 `1.443배`,
p95 `1.375배`이고 semantic mismatch와 오류는 0건입니다.

최대 confidence 차이는 `0.0085055232`로 strict tolerance `1e-5`를 넘었고, 하이브리드 p95도
500ms 진단 기준을 넘으므로 원본 결과의 `passes=false`를 유지합니다. 이는 한 대의 N100 진단이며
SLA, 인증 또는 배포 승인으로 표현하지 않습니다.

원본은 [N100 device matrix](../diagnostics/n100-0.1.5-openvino-device-matrix.json), 후보와 source
해시는 [측정 패키지](../diagnostics/n100-0.1.5-measurement-package.json)에 고정합니다.

상세 결과와 한계는 [0.1.6 개발 검증 보고서](../evaluation/scanner-0.1.6.md), 과거 판단은
[버전 이력](../archive/version-history.md)에 기록합니다.
