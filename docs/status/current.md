# 현재 운영·릴리스 상태

기준일: 2026-08-18

| 축 | 버전/후보 | 상태 |
|---|---|---|
| Python | `1.0.0` | 정식 API 계약 코드 |
| 현재 운영 package | `bread-worker-1.0.0` | schema 1.1 production, 바이너리 불변 |
| schema 2.1 hardening 후보 | `bread-worker-1.0.0-hardened-candidate-recovered` | `not_promoted`, 개선 후 재검증 |
| Worker / Detector / Classifier | `1.0.0` | 독립 버전 |
| Detector 학습 파이프라인 | `1.0.0` | recovered 10-shot provenance |
| Classifier 학습 파이프라인 | `1.0.0` | recovered 12-shot provenance |
| Active 1.1 목표 | `bread-zero-error-1.1.0-domain-lda-fixed-four-v3` | 개발 정확도·CUDA 성능 통과, `promotion_ready=false` |
| release 데이터셋 | `bread-1.0-a52b4faa3e20` | `single_objects_3`, 종류별 12장 |
| Flutter | `1.0.0+2` | Worker·Detector·Classifier readiness 확인 |
| 사용자 독립 test | pending | 1.0.0 조정에 사용하지 않음 |
| 이전 Detector 후보 | `0.2.5` | `experiment_only`, 미승격 |
| Rollback package | `bread-worker-0.1.1` | 장애 시 수동 복구용 |

`bread-zero-error-1.1.0`의 공식 gate는 `SEGMENTATION ≥90%`, 전체 판정 가능 GT 대비
end-to-end `APPROVED ≥90%`, 그리고 `SEGMENTATION` 이미지 FP/FN·승인 오인·Candidate out
각각 ≤0.1%다. 최종 개선 목표는 같은 all-GT 분모의 `APPROVED ≥99%`이며 여섯 운영 gate와
별도로 판정한다. `UNKNOWN + Top-3` 비율과 `SEGMENT_RECAPTURE` 비율은 진단 전용이다.

v3 development package를 실제 Worker 경로로 E/M/H 300장에 실행한 결과는
`SEGMENTATION` 300/300, `APPROVED` 1,410/1,410, FP/FN/승인 오인/Candidate out 0건이다.
CUDA full-path 평균/P95는 91.71/90.99ms다. v2에서 한 번 열어 반려한 운영 115장은 v3의
개발 데이터로만 재사용했고, 그 범위에서도 빈 트레이 4장 외 `SEGMENTATION` 111/115,
`APPROVED` 504/504, 네 오류 지표 0건, 평균/P95 74.33/87.09ms였다.

E/M/H 300장의 CPU/CUDA 공개 decision trace는 SHA-256이 모두
`c42eec94614677d4d1ecaebc4a98a7a1ab6d97a301750adb59cb73465c98326c`로 같았다. 최종 상태·
클래스·Top-3 순위, 1,410개 bbox와 confidence가 모두 정확히 일치했다. CPU 지연은 기능 호환
진단일 뿐 승격 지연 gate가 아니다.

이 결과는 개발 목표와 CUDA 성능·provider parity 통과이지 독립 일반화 증거가 아니다. v2 운영
수집본을 본 뒤 v3 정책을 선택했으므로 새 촬영 세션의 잠금 test가 필요하다. Ruff lint/format,
전체 Python, Flutter analyze와 164개 Flutter test, diff check는 통과했다. 따라서 v3는
`active_development`, `promotion_eligible=false`이고 운영
기본 package는 계속 `bread-worker-1.0.0`이다. 분모와 최신 재판정은
[1.1.0 실험 문서](../experiments/bread/bread-zero-error-1.1.0.md)를 따른다.

이전에 검토한 all-data D-FINE final-soup와 RF-DETR Large 후보는 모두 반려됐다. RF-DETR
Large는 group-held-out fold 0의 최대 recall이 97.944%(524/535)에 그쳤고 그 설정에서 FP가
7,223건이어서 남은 fold 학습을 중단했다. 실패 기록은 v3의 same-domain LDA와 선택적
multi-resolution proposal 검증을 채택한 근거로 유지한다.

schema 2.1 hardening 후보는 기존 운영 Detector와 Classifier ONNX를 그대로 사용한다.

- Detector ONNX: `f0d2eaf8e67821627957c3eed1462812063c32c4ad17028dda869addc5371b09`
- Classifier ONNX: `93a9d92c6fd63f5a6aef65e11e3d0acecfffd7c6cf5ac2bfdba732f4e543ab8f`

`multi_object_scenes` 300장 RTX 5080 검증 결과는 인식률 `99.0780%`, 승인 오인율
`0%`(0/1,121), segmentation recall/precision `99.2908%`/`99.7151%`, 평균/P50/P95/P99
`69.89/69.71/84.20/90.92ms`다. `IMAGE_RECAPTURE`와 `SEGMENT_RECAPTURE`는 모두
0건이며 RECAPTURE로 성능을 우회하지 않았다.

point KPI, segmentation, RECAPTURE, 평균/P95 속도는 모두 통과했다. 다만 승인 오인율의
단측 95% 상한은 표본 부족으로 `0.2669%`이며 목표 `0.1%`를 넘는다. 사용자는 waiver 대신
개선 후 재검증을 선택했으므로 hardening 후보를 production으로 승격하지 않는다. 최소 2,995개
승인 샘플에서 오인 0건인 사용자 독립 test가 완료되면 별도 immutable attestation을 추가한다.
결정 기록은 `configs/releases/hardening_decision_1.0.0.json`에 고정했다.

12-shot Detector 재학습 후보는 잠긴 benchmark를 실패했으므로 rejected 상태이며 운영에
반영하지 않았다. 자세한 학습 provenance와 제한은
[학습 파이프라인 1.0.0](../guides/training-pipeline-1.0.0.md)을 참고한다.
