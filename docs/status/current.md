# 현재 운영·실험 상태

기준일: 2026-08-13

| 축 | 버전 | 상태 |
|---|---|---|
| Python | `0.2.0` | 현재 배포 버전 |
| 모델 package | `bread-worker-0.1.1` | `production`, 앱 기본 |
| 데이터셋 | `bread-43093242294f` | 운영 모델 provenance |
| Flutter | `1.0.0+1` | 현재 작업자 앱 |
| Detector 후보 | `0.2.5` | `experiment_only`, 미승격 |

Detector `0.2.5`는 코드·선택·lock·test·ONNX parity·RTX 5080 latency까지 검증했으나 독립 데이터, detector/E2E risk, Hard recall과 `UNKNOWN` Top-3 gate를 모두 만족하지 못했습니다. 따라서 운영 package와 Flutter release의 기본 package는 `bread-worker-0.1.1`을 유지합니다.

세부 결과는 [Detector 0.2.5 보고서](../experiments/detector/detector-target-0.2.5.md), 재실행 조건은 [runbook](../guides/detector-target-0.2.5-runbook.md)을 참고하십시오. 운영 승격은 [모델 승격 가이드](../guides/model-promotion.md)의 모든 gate가 통과되고 보고서 상태가 `promoted`로 바뀐 뒤에만 가능합니다.
