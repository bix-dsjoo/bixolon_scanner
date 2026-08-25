# 현재 버전

기준일: 2026-08-25

현재 실행 조합은 `0.1.3` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.3` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.3` |
| Store Catalog | `0.1.3`, `CHECKSUM-SHA256` |
| Flutter 내부 빌드 | `0.1.3+6` |
| Detector | class-agnostic YOLO26 objectness, score `0.65`, NMS IoU `0.4` |
| Classifier | DINOv3 ConvNeXt-Tiny soup + 180° 검증 + DINOv3 ViT-B/16 독립 검증 |
| N100 실행 | Detector `OpenVINO CPU`, 모든 Embedder `OpenVINO Intel GPU`, 실패 시 명시적 CPU fallback |
| source candidate | `yolo26-objectness-single3-consensus-openvino` |
| Runtime 원본 manifest SHA-256 | `d5150a515563e990dd39824f8519c8b64b3b8fccf51763d863a54db441067400` |
| Catalog 원본 manifest SHA-256 | `4c35abb9d03798c7df34983b6e401c1af6595cf1a71834efa944c85021e245ab` |
| 전체 유효 데이터 report SHA-256 | `d3ec056a414eac236642a1e33a3617849c2e59d3b4073eeefb9fa95401307438` |
| N100 참조 측정 | `0.1.3` 동일 모델·Catalog 100장, 하이브리드 HTTP 평균 `445.480ms`, p95 `751.534ms`, `ERROR` 0 |

`bread_dataset`의 전체 유효 이미지 415장과 객체 1,914개를 별도 고정 test set 없이 개발 회귀로
평가했습니다. `IMAGE_RECAPTURE` 4장, Detector FP/FN 0, `APPROVED` 1,895개, 승인 오분류 0,
`UNKNOWN` 14개와 Top-3 이탈 0, `SEGMENT_RECAPTURE` 5개였습니다. RECAPTURE를 분모에 포함한
올바른 `APPROVED` 비율은 99.0073%입니다. 현재 개발 PC의 OpenVINO full-path 411장 측정은 평균
112.29ms, p95 183.56ms였습니다. 실제 standalone EXE의 multipart HTTP 왕복시간은 같은 411장
기준 평균 144.95ms, p95 210.23ms였고 `ERROR`와 응답 계약 위반은 0건이었습니다.

Classifier 소스는 동일한 grouped-fold 조건으로 `single_objects`와 `single_objects_3`을 비교해
`single_objects_3`만 선택했습니다. multi-object 상품 label은 Classifier 학습·보정에 사용하지
않았습니다. Detector는 상품 ID가 없는 단일 objectness class로 학습했고 새 SKU 추가 시 Detector
모델, threshold와 후처리를 변경하지 않습니다. Classifier 모델도 Catalog와 분리된 고정 embedder를
사용합니다.

신규 SKU를 모사한 20개 class-holdout에서는 기존 19개 클래스 adapter 열을 bit-stable하게 유지하고
새 출력 열만 독립 Ridge로 추가했습니다. 주 분류기, 180° view와 독립 DINOv3 verifier가 모두 새
SKU Top-1에 동의할 때만 증분 출력을 사용한 결과 기존 클래스 판단 36,366건의 출력 변화와
correct-to-incorrect 전환이 모두 0건이었습니다. multi-object label은 이 규칙의 학습·threshold
선택에 쓰지 않고 사후 평가에만 사용했습니다.

이 평가는 같은 데이터 자산을 group-aware train/validation과 전체 개발 회귀에 사용한 결과입니다.
별도 test set 또는 데이터셋 외부의 신규 SKU 독립 표본이 없으므로 독립 일반화 성능,
외부 SKU 일반화, SLA 또는 인증으로 표현하지 않습니다. Catalog에는 `signature.json`, signing key,
HMAC 또는 lifecycle 메타데이터가 없습니다. 파일별 checksum 불일치는 Worker 시작 오류입니다.

`0.1.3+6` 앱은 실행 직후 카메라 초기화와 Worker readiness polling을 병렬로 시작하고, Worker는
OpenVINO `OPTIMIZE_SPEED` 컴파일 캐시를 사용자별 version 디렉터리에 유지합니다. 촬영 후 Flutter
이미지 디코딩과 HTTP 요청도 겹쳐 실행합니다. 선택된 승인 상품을 마우스로 누르는 동안에는 승인
초록 상태와 충돌하는 Orange overlay 대신 중립 pressed overlay를 사용합니다. Runtime graph,
Classifier Catalog와 판정 threshold는 변경하지 않았습니다.

현재 packaged Worker의 415장 multipart HTTP 개발 회귀는 `ERROR`와 응답 계약 위반 0건이었고,
411장 full-path 평균/p95는 `144.95/210.23ms`였습니다. 이 값은 N100 실측이나 SLA가 아닙니다.
`0.1.3+6` 재빌드 뒤 installer Worker의 readiness, 정상 이미지, 손상·누락·미지원 입력 계약도
[packaged Worker smoke](../diagnostics/packaged-worker-0.1.3-build6-smoke.json)에서 통과했습니다.

실험과 한계는 [0.1.3 개발 검증 보고서](../evaluation/scanner-0.1.3.md), 과거 판단은
[버전 이력](../archive/version-history.md)에 기록합니다.
