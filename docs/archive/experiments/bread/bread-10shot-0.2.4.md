# 엄격한 10-shot classifier `0.2.4` 최종 평가

## 결론

`0.2.4`는 `experiment_only`입니다. 두 seed의 strict 10-shot checkpoint를 균등 평균한 single-model parameter soup로 `0.2.3`보다 development와 두 회귀셋 성능을 개선했지만, `test_94`와 false-approval risk 및 clean latency 증거가 승격 계약을 충족하지 못했습니다. 운영 package와 fallback은 계속 `bread-worker-0.1.1`입니다.

## 고정 recipe와 lock

- source: 20 class × 10장, 총 200장만 가중치 학습에 사용
- soup member seed: `20260813`, `20260814`
- averaging: 전체 floating tensor의 float32 산술평균, runtime model 1개
- inference crop: 중앙 `0.855`
- 안정화: quantum `0.24`, phase `0.192`, class bias span `0.012`, divisor `50`
- provider threshold guard: `0.000001`, 적용 후 development risk 재계산
- checkpoint SHA-256: `2b4255794899186b33b63d5fc27ebe3f4ada365cf32fd1acd12e8c6082aae29b`
- pre-test lock SHA-256: `4eec25e58a9c176479337805c30dcdc82b6ea88b4cf5389957be28b9dcb5de71`
- classifier ONNX SHA-256: `f0af021c271721f702061cb8d7042f39bf35b2050e6acace9e7dc386b133cbee`
- detector ONNX SHA-256: `0.1.1`과 동일한 `635dd93ccde8a244692ad4cc14aaf259790d59d05fbde20e00588d47f3edefdc`

Detector, input/quality metadata, `DETECTOR_UNCERTAIN_OBJECT`, classifier 미호출/null 규칙과 `/v1/scan` 계약은 변경하지 않았습니다.

## 결과

| 평가 | Top-1 | 전체 Top-3 | 승인 precision | 승인 coverage | false approval 95% 상한 | UNKNOWN Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| development ROI 886개 | 97.856% | 100.000% | 100.000% | 89.391% | 0.378% | 100.000% |
| test 94장, active ROI 478개 | 94.979% | 98.745% | 99.254% | 84.100% | 1.917% | 92.105% |
| `bread_project_2` 300장, active ROI 1,374개 | 96.943% | ≥99.199% | 99.583% | 87.263% | 0.875% | 96.571% |

`bread_project_2`의 전체 Top-3는 APPROVED row에 Top-3가 없는 기존 진단 schema 때문에 보수적 하한입니다. `0.2.3`과 비교하면 test 94장 Top-1은 `+1.883%p`, coverage는 `+6.276%p`; `bread_project_2` Top-1은 `+0.976%p`, coverage는 `+4.639%p` 개선됐습니다.

## 통계 검정력 제한

한쪽 95% Clopper–Pearson false-approval 상한을 `0.5%` 이하로 증명하려면 오류가 0건이어도 승인 표본이 최소 598개 필요합니다. `test_94`는 classifier가 실행된 전체 ROI가 478개뿐이므로 모델이 478개를 모두 정확히 승인하더라도 이 조건을 증명할 수 없습니다. 최종화 도구는 이를 `test_94:approval_risk_certification_feasible=false`로 명시합니다.

현재 test 결과 자체도 하한에 미달합니다. Top-1은 `95%`에 한 ROI 부족하고, 승인 오류 3건과 coverage `84.100%`, UNKNOWN Top-3 `92.105%`가 실패입니다. `bread_project_2`는 Top-1·precision·coverage·UNKNOWN Top-3를 통과했지만 승인 오류 5/1,199건으로 risk 상한이 실패했습니다.

## Parity와 latency

- PyTorch CUDA/CPU ONNX/CUDA ONNX 886개
- 상태, Top-1, Top-3 set 및 순서 mismatch: 모두 `0`
- 최대 절대 logit 차이: `0.00480008`
- RTX 5080 warm-up 30회, 총 1,000회 중 full-path 800건
- full-path p50 `77.743ms`, p95 `110.948ms`, p99 `126.616ms`

Benchmark 당시 별도의 `rpc_data_scale detector` GPU 학습이 실행 중이었습니다. 따라서 p95는 gate를 통과하지 못한 실측값으로 보존하되 clean exclusive benchmark 증거로 취급하지 않습니다. 해당 학습을 임의로 중단하지 않았습니다.

## 승격 판단

실패 gate는 full-path p95, `bread_project_2` risk 상한, `test_94` Top-1·precision·risk·coverage, 그리고 `test_94` risk 검정력 부족입니다. 정규화 결과는 `artifacts/experiments/bread-10shot-0.2.4/reports/final_promotion.json`에 있습니다. package 전환과 rollback은 실행하지 않았습니다.

목표를 검증 가능하게 만들려면 최소 598개의 classifier-active 승인 가능 ROI가 있는 새 잠긴 회귀/독립 평가 세트가 필요합니다. 성능은 동시 GPU 학습이 없는 exclusive 환경에서 같은 1,000회 benchmark로 다시 측정해야 합니다.
