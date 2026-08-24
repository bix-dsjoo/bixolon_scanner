# 현재 버전

기준일: 2026-08-24

현재 실행 조합은 `0.1.2` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.2` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.2` |
| Store Catalog | `0.1.2`, `CHECKSUM-SHA256` |
| Flutter 내부 빌드 | `0.1.2+4` |
| 사용자 표시·Windows ProductVersion | `0.1.2` |
| Detector | D-FINE 단일 모델 + DINOv3 ViT-S/16 192 입력 객체-presence verifier |
| Embedder | DINOv3 ConvNeXt-Tiny last-stage fine-tuned, 111.4MB |
| N100 실행 | `CPUExecutionProvider`, detector `1×4`, embedder `4 threads` |
| N100 `0.1.2` 측정 | 현장 JSON 대기 중, 평균·p95 목표 `1,000ms` |
| source candidate | `detector415-single3-bias004-detector-corroboration-cpu` |
| Runtime 원본 manifest SHA-256 | `6bc192b3d8fadc13051b0ae98f96d9536d34c0b350a01de790b273b659bff000` |
| Catalog 원본 manifest SHA-256 | `ef72b19dcfe9a3d1fc9469d333f24b1ba14afcf45cc06c7617013ad80fdbe5e0` |
| 현재 PC CPU 평가 report SHA-256 | `09674fd7d3b8517db86332234e8b4b1f9492fb66d6dad2aad5ffcb70a6ed1374` |

`0.1.2`은 `0.0.2`의 모델을 단순 재패키징한 버전이 아닙니다. Detector 학습·선정 데이터는
`multi_object_scenes` 300장과 `operational_collections/2026-08-18` 115장뿐이고, Classifier 학습은
`single_objects_3` 240장뿐입니다. SKU·상품쌍·객체수·난이도별 실행 예외는 없습니다. 같은 415장
개발 회귀에서는 이미지 `IMAGE_RECAPTURE` 4건, prediction 1,912개, detector FN 6/FP 4,
`APPROVED` 1,894개(98.9551%), 승인 오분류 5개, `UNKNOWN` 9개, `SEGMENT_RECAPTURE` 5개,
Candidate out 0개였습니다. 주변 객체 ownership mask 거리 bias `0.04`와 클래스·상품쌍을 보지 않는
두 detector corroboration confidence 조건을 개발 데이터에서 선택했습니다. 정확도 gate는 아직
통과하지 않았고 독립 테스트에서 이 정책을 다시 검증해야 합니다.

현재 개발 PC의 CPU `1×4` 전체 415장 최종 측정은 평균 `212.63ms`, p95 `296.66ms`, p99
`380.27ms`였습니다. 목표로 정한 평균·p95 300ms는 통과했지만 p99는 300ms를 넘었고, 이 장비는
N100이 아니므로 N100의
1초 목표 달성을 의미하지 않습니다. 최종 N100 판단은
`artifacts/handoff/n100-0.1.2-candidate-20260824.zip`에서 생성되는
`n100-0.1.2-result.json`으로만 수행합니다.

Catalog에는 `signature.json`, signing key, HMAC 또는 lifecycle 메타데이터가 없습니다. 파일별
checksum 불일치는 Worker 시작 오류입니다. 이 버전의 평가 결과는 기존 개발 데이터 재평가이며
독립 일반화 성능, SLA 또는 인증으로 표현하지 않습니다. 전체 과거 판단과 제한은
[버전 이력](../archive/version-history.md)에 남겨 둡니다.
