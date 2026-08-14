# 현재 운영·릴리스 상태

기준일: 2026-08-14

| 축 | 버전/후보 | 상태 |
|---|---|---|
| Python | `1.0.0` | 정식 API 계약 코드 |
| 현재 운영 package | `bread-worker-1.0.0` | schema 1.1 production, 바이너리 불변 |
| schema 2.1 hardening 후보 | `bread-worker-1.0.0-hardened-candidate-recovered` | `not_promoted`, 개선 후 재검증 |
| Worker / Detector / Classifier | `1.0.0` | 독립 버전 |
| Detector 학습 파이프라인 | `1.0.0` | recovered 10-shot provenance |
| Classifier 학습 파이프라인 | `1.0.0` | recovered 12-shot provenance |
| release 데이터셋 | `bread-1.0-a52b4faa3e20` | `single_objects_3`, 종류별 12장 |
| Flutter | `1.0.0+2` | Worker·Detector·Classifier readiness 확인 |
| 사용자 독립 test | pending | 1.0.0 조정에 사용하지 않음 |
| 이전 Detector 후보 | `0.2.5` | `experiment_only`, 미승격 |
| Rollback package | `bread-worker-0.1.1` | 장애 시 수동 복구용 |

schema 2.1 hardening 후보는 기존 운영 Detector와 Classifier ONNX를 그대로 사용한다.

- Detector ONNX: `f0d2eaf8e67821627957c3eed1462812063c32c4ad17028dda869addc5371b09`
- Classifier ONNX: `93a9d92c6fd63f5a6aef65e11e3d0acecfffd7c6cf5ac2bfdba732f4e543ab8f`

`multi_object_scenes` 300장 RTX 5080 검증 결과는 인식률 `99.0780%`, 승인 오인율
`0%`(0/1,121), segmentation recall/precision `99.2908%`/`99.7151%`, 평균/P50/P95/P99
`69.89/69.71/84.20/90.92ms`다. `IMAGE_RECAPTURE`와 `SEGMENT_RECAPTURE`는 모두
0건이며 RECAPTURE로 성능을 우회하지 않았다. `scan_log_samples`는 읽거나 집계하지 않았다.

point KPI, segmentation, RECAPTURE, 평균/P95 속도는 모두 통과했다. 다만 승인 오인율의
단측 95% 상한은 표본 부족으로 `0.2669%`이며 목표 `0.1%`를 넘는다. 사용자는 waiver 대신
개선 후 재검증을 선택했으므로 hardening 후보를 production으로 승격하지 않는다. 최소 2,995개
승인 샘플에서 오인 0건인 사용자 독립 test가 완료되면 별도 immutable attestation을 추가한다.
결정 기록은 `configs/releases/hardening_decision_1.0.0.json`에 고정했다.

12-shot Detector 재학습 후보는 잠긴 benchmark를 실패했으므로 rejected 상태이며 운영에
반영하지 않았다. 자세한 학습 provenance와 제한은
[학습 파이프라인 1.0.0](../guides/training-pipeline-1.0.0.md)을 참고한다.
