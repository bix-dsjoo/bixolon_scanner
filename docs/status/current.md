# 현재 운영·릴리스 상태

기준일: 2026-08-20

| 축 | 버전/후보 | 상태 |
|---|---|---|
| Python | `1.0.0` | 정식 API 계약 코드 |
| 현재 운영 release | `scanner-2.0.1` | rc.3 기반 owner-waiver production |
| Worker / Detector / Embedder / Policy / Catalog | `2.0.1` | 원자적 운영 조합 |
| Store Catalog | `2.0.1` | `single_objects` 200장, 영구 무키 `CHECKSUM-SHA256` |
| 승격 원본 | `2.0.1-rc.3` | 300장 point gate 통과, 남은 gate는 waiver에 고정 |
| schema 2.1 hardening 후보 | `bread-worker-1.0.0-hardened-candidate-recovered` | `not_promoted`, 개선 후 재검증 |
| Bridge rollback | `bread-worker-1.1.0` | 고정 4-model Detector + domain LDA |
| Detector 학습 파이프라인 | `1.0.0` | recovered 10-shot provenance |
| Classifier 학습 파이프라인 | `1.0.0` | recovered 12-shot provenance |
| 승격 원본 | `bread-zero-error-1.1.0-domain-lda-fixed-four-v3` | 두 알려진 제한을 waiver로 보존 |
| release 데이터셋 | `bread-1.1-development-plus-rejected-operational-v2` | E/M/H LDA fit과 반려 운영본 재사용 계보 |
| Flutter | `2.0.1+5` | 번들 내부 2.0.1 조합 CUDA readiness 확인 |
| 다음 Classifier | `1.1.1+` | `single_objects` 200장 전용 계획 |
| 사용자 독립 test | pending | 1.1.1+ successor 승격용 새 세션 필요 |
| 이전 운영 release | `scanner-2.0.0` | 첫 rollback 기준 |
| Scanner 2.0 RC | `2.0.0-rc.10` | 과거 반려 후보, 1 IPS production 성능 gate 실패 |
| 이전 Detector 후보 | `0.2.5` | `experiment_only`, 미승격 |
| Legacy rollback package | `bread-worker-1.0.0` | 수동 복구용 |
| 비상 rollback package | `bread-worker-0.1.1` | 테스트 계열 최종 복구용 |

Scanner 2.0 RC.10은 fixed four-model D-FINE Detector, frozen DINOv3
ViT-B/16 Embedder와 SKU별 원본 10장 기반 Store Catalog ridge adapter 조합이다. 300장 개발
회귀에서 `SEGMENTATION` 294/300, 반환 segmentation 1,375개 중 `APPROVED`
1,338(97.3091%), `UNKNOWN` 9(0.6545%), `SEGMENT_RECAPTURE` 28(2.0364%), 이미지 FN/FP
각각 0건, 승인 오인 1/1,410, Candidate out 0건을 기록했다. all-GT 승인 coverage는 94.8936%다.

DINOv3 PyTorch↔ONNX CUDA, Embedder CPU↔CUDA와 300장 최종 상태·class rank parity는 통과했다.
연속 warm CUDA full-path 평균/P95/P99도 84.74/95.38/98.76ms로 통과했지만, 운영 계약인 요청
시작 간격 1,000ms에서는 Python 전용 NVIDIA 최대 성능 정책을 적용해도
119.08/159.12/190.02ms로 실패했다. 따라서 RC.10은 pre-private lock 또는 owner-private test
단계가 아니다. RC.9는 임계값 경계의 CPU/CUDA 최종 상태 mismatch 한 건으로 반려했고 RC.10은
0.005 provider guard로 보수적으로 고정했다. 관리자 권한의 GPU clock floor 또는 모델 구조
재최적화, 동일 cadence 재평가, packaged Worker·reliability·supply-chain gate와 새 release lock을
먼저 완료해야 한다. 기존 RC.8 lock은 DINOv2 과거 evidence로만 보존한다.

300장은 backbone·Catalog와 threshold 선택에 사용됐으므로 독립 인증 증거가 아니다. 상세 계약은
[Scanner 2.0 설계](../architecture/scanner-2.0.0.md)와
[300장 평가](../experiments/bread/scanner-2.0.0-development-300.md)를 따른다.

`2.0.1-rc.3`는 원본 `single_objects` 200장 Catalog에 ridge/retrieval 합의 guard를 추가한
승격 원본이다. 300장 재평가에서 정답 승인 1,311/1,410(92.9787%), 승인 오인 0건,
Candidate out 0건, 평균/P95/P99 84.89/95.57/100.63ms를 기록했다. 개발 point gate는 모두
통과했지만 0/1,311의 단측 95% 상한은 0.22825%이며, CPU/CUDA 전체 parity·1 IPS·비공개 test도
완료되지 않았다. 프로젝트 소유자는 2026-08-20 이 제한과 reliability·supply-chain 미완료를
`configs/releases/scanner_2.0.1_owner_waiver.json`에 고정하고 rc.3를 최종 `2.0.1`로
승격했다. CPU/CUDA packaged Worker smoke와 앱 EXE의 CUDA readiness는 통과했다. 운영 승격은
독립 certification 통과를 뜻하지 않는다. 상세 기록은
[RC.3 결과](../experiments/bread/scanner-2.0.1-rc.3-single-objects.md)를 따른다.

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
수집본을 본 뒤 v3 정책을 선택했고 최종 LDA도 E/M/H ROI 1,410개로 fit됐다. 프로젝트 소유자는
2026-08-19 두 제한을 숨기지 않는 `manual_waiver`로 v3를 `bread-worker-1.1.0` 운영 기준선에
승격했다. waiver는 `configs/releases/bread_1.1.0_owner_waiver.json`에 고정한다.

이 예외는 1.1.0에만 적용된다. 1.1.1부터 Classifier의 파라미터, head/statistic fitting,
calibration과 threshold는 `single_objects` 200장만 사용한다. E/M/H와 운영 115장은 개발 회귀
전용이며, successor 승격에는 새 촬영 세션의 잠금 test가 다시 필요하다. 세부 반복 계획은
[200장 전용 1.1.1+ 계획](../experiments/bread/bread-classifier-200-only-1.1.1-plan.md)을 따른다.

추가 raw storage preflight에서도 적격 독립 데이터는 0개였다. `bread_project/group` 299장은
기존 E/M/H 개발 이미지와 dHash≤2 중복이었다. model inference는
실행하지 않았다. 새 촬영 세션은 review 완료 COCO GT와 capture-session provenance를 갖추고
independent preflight를 통과한 뒤에만 단 한 번 평가한다.

사용자가 지정한 `operational_collections/2026-08-18`도 확인했다. 이 115장/504 GT는 기존 v2
locked test와 SHA-256이 정확히 같고 v2 결과를 본 뒤 v3 개발에 재사용한 세트다. 2026-08-19
CUDA 재실행에서 `APPROVED` 504/504, 네 오류 0건, `SEGMENTATION` 96.522%, 평균/P95
75.67/88.48ms를 재현했지만 독립 승격 증거는 아니다. 전체 개발 계보 preflight는 115/115 exact
중복으로 거부했고, Runtime gate도 부적격 데이터를 `independent`로 지정하면 모델 실행 전에
실패하도록 강화했다.

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
