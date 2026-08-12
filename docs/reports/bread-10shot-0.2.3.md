# 엄격한 10-shot classifier `0.2.3` 최종 평가

## 결론

`0.2.3`은 `experiment_only`입니다. 개발 ROI에서는 목표 정확도를 넘고 PyTorch/CPU ONNX/CUDA ONNX 판정 parity도 통과했지만, 잠금 후 처음 접근한 `test_94` 회귀셋의 Top-1·승인 precision·coverage와 RTX 5080 full-path p95가 배포 하한을 통과하지 못했습니다. `bread-worker-0.1.1`을 운영 및 fallback package로 유지하며 package 전환이나 rollback 동작은 실행하지 않았습니다.

`test_94`와 `bread_project_2`는 독립 평가셋이 아니라 잠긴 회귀셋입니다. 아래 결과는 `0.2.3`의 threshold나 가중치를 다시 맞추는 데 사용하지 않습니다.

## 잠금과 모델 계약

- 학습 데이터: `bread_project_3`의 20개 class × 정확히 10장, 총 200장
- 학습에 사용하지 않은 데이터: development/test ROI, `bread_project_2`, 기존 classifier logit·가중치·증류 결과
- recipe: `last_stage_l2sp_center_crop_088_stable_logits`
- checkpoint SHA-256: `09cdd2d06a4a4c5c6b0966a1838b54e9472633496c97ef45b4c8f8bf566b84b8`
- pre-test lock SHA-256: `990c89d2d16bb9ba26f1d2c84ad51c22f75d0f74a13c528fdb72025d8d75fb75`
- classifier ONNX SHA-256: `2f9b24f3ada4c500bfd66d963af9ee6fb31c4303af0389f472fddc575834fc18`
- detector ONNX SHA-256: 기준/후보 모두 `635dd93ccde8a244692ad4cc14aaf259790d59d05fbde20e00588d47f3edefdc`
- detector/input/quality metadata: `0.1.1`과 동일
- API 및 `DETECTOR_UNCERTAIN_OBJECT` hard gate: 변경 없음

Classifier ROI는 detector bbox의 기존 7% padding 후 중앙 88%를 사용합니다. ONNX에 포함된 고정 logit 안정화 정책은 quantum `0.44`, phase `0.066`, class tie-break span `0.044`, divisor `50`이며 CPU/CUDA 순위가 달라지지 않도록 development에서 잠근 뒤 회귀셋에 접근했습니다.

## 평가 결과

| 평가 | Top-1 | 전체 Top-3 | 승인 precision | 승인 coverage | false approval 95% 상한 | UNKNOWN Top-3 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| development ROI, 886개 | 97.630% | 99.774% | 100.000% | 85.102% | 0.397% | 98.485% | 개발 목표 통과 |
| 기존 test 94장, active ROI 478개 | 93.096% | 98.536% | 99.462% | 77.824% | 1.683% | 94.340% | 실패 |
| `bread_project_2` 300장, active ROI 1,364개 | 95.968% | ≥99.047% | 99.823% | 82.625% | 0.558% | 95.359% | 실패 |

`bread_project_2`의 전체 Top-3는 APPROVED row에 Top-3가 저장되지 않는 기존 진단 schema 때문에, Top-1 정답 승인 1,125개와 UNKNOWN Top-3 정답 226개만 센 보수적 하한입니다. 난이도별 Top-1은 E `98.526%`, M `97.263%`, H `92.531%`로 H 촬영 도메인이 주요 약점입니다.

기존 `0.1.1` development approval coverage `97.654%` 대비 `0.2.3`은 `12.552%p` 낮아, 운영 대비 하락 `≤5%p` 목표도 실패했습니다.

## Parity와 성능

- PyTorch CUDA, CPU ONNX, CUDA ONNX 886개에서 최종 상태, Top-1, Top-3 set 및 순서가 모두 동일
- mismatch: Top-1 `0`, Top-3 `0`, 최종 상태 `0`
- 최대 절대 logit 차이: `0.00880003`
- RTX 5080, ONNX Runtime `1.28.0`, CUDA `13.1`, driver `591.86`, warm-up 30회, 총 1,000회
- full-path 800건: p50 `80.923ms`, p95 `117.874ms`, p99 `129.524ms`
- p95 `≤100ms` 실패. 동일 실행에서 decode p95 `49.386ms`, detector p95 `54.544ms`, classifier p95 `24.088ms`여서 classifier만의 회귀로 단정할 수 없습니다.

## 최종 실패 gate

- full-path p95 `117.874ms > 100ms`
- `bread_project_2`: false approval 상한 `0.558% > 0.5%`, coverage `82.625% < 85%`
- `test_94`: Top-1 `93.096% < 95%`, precision `99.462% < 99.5%`, false approval 상한 `1.683% > 0.5%`, coverage `77.824% < 85%`

정규화된 판정은 `artifacts/experiments/bread-10shot-0.2.3/reports/final_promotion.json`에 있습니다. 원본 parity, regression, benchmark JSON은 같은 `reports` 디렉터리에 보존합니다.

## 재현 명령

아래 최종화 명령은 모델이나 threshold를 변경하지 않고 잠긴 산출물만 읽습니다.

```powershell
bixolon-ten-shot-finalize `
  --development-decision artifacts\experiments\bread-10shot-0.2.3\reports\development_decision.json `
  --parity-report artifacts\experiments\bread-10shot-0.2.3\reports\parity.json `
  --benchmark-report artifacts\experiments\bread-10shot-0.2.3\reports\benchmark_cuda_1000.json `
  --test-94-report artifacts\experiments\bread-10shot-0.2.3\reports\regression_test_94_cuda.json `
  --bread-project-2-report artifacts\experiments\bread-10shot-0.2.3\reports\regression_bread_project_2_300_cuda.json `
  --baseline-package artifacts\packages\bread-worker-0.1.1 `
  --candidate-package artifacts\packages\bread-worker-0.2.3 `
  --output artifacts\experiments\bread-10shot-0.2.3\reports\final_promotion.json
```

다음 classifier 후보는 test 결과에 맞춘 threshold 재조정이 아니라 새 버전과 새 pre-test lock으로 시작해야 합니다. 우선순위는 H 도메인의 foreground 촬영 다양성 확보와 crop/배경 불변성 개선이며, latency는 동일 benchmark 이미지로 `0.1.1`과 후보를 연속 측정해 환경 변동과 classifier 비용을 분리하는 것입니다.
