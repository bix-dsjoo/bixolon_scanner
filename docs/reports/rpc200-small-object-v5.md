# RPC200 소형 상품 복구 v5 검증 결과

## 결론

`context-logistic-v4`의 detector/classifier/context 모델과 두 승인 임계값은 그대로
고정하고, Worker의 `min_object_area_ratio`만 `0.005`에서 `0.002`로 낮췄다.
기존에는 실제 상품인 작은 검출 박스 때문에 `OBJECT_TOO_SMALL` 이미지 전체가
`IMAGE_RECAPTURE`가 됐으나, v5에서는 validation selection의 32개 실패 이미지를
모두 정상 경로로 복구했다.

## 요청 형식 결과표

아래 네 실패 열은 각 난이도에서
`Segmentation 실패 이미지 + Top-3 Candidate + Candidate Out + 오인율* = 100%`가
되도록 실패 건수 기준으로 정규화했다. `오인율*`은 이 구성비의 잘못된
`APPROVED` 비중이며, 모델의 실제 승인 오인율은 다음 절에 별도로 기록한다.

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.235% | 0.000% (0) | 73.913% (51) | 2.899% (2) | 23.188% (16) | 52.32 / 61.11ms |
| Medium | 99.590% | 0.000% (0) | 59.756% (49) | 0.000% (0) | 40.244% (33) | 64.34 / 73.80ms |
| Hard | 99.360% | 0.000% (0) | 60.227% (106) | 0.000% (0) | 39.773% (70) | 71.72 / 84.09ms |

실제 승인 오인율은 Easy `0.228%`, Medium `0.271%`, Hard `0.416%`이며 모두
허용 기준 `0.5%` 이내다. `UNKNOWN Top-3` 포함률은 Easy `96.226%`, Medium과
Hard는 각각 `100.000%`다.

## v4 대비 변화

| 난이도 | v4 실패 이미지 | v5 실패 이미지 | v4 실제 오인율 | v5 실제 오인율 | v4 E2E | v5 E2E |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 14 | 0 | 0.217% | 0.228% | 96.642% | 98.013% |
| Medium | 8 | 0 | 0.265% | 0.271% | 97.764% | 98.517% |
| Hard | 10 | 0 | 0.420% | 0.416% | 96.723% | 97.703% |

오인율은 Easy와 Medium에서 각각 `+0.011pp`, `+0.006pp`로 소폭 증가했지만
허용 범위 안이며, Hard는 `-0.004pp` 개선됐다. 전체 이미지 재촬영 32건이
사라져 E2E는 세 난이도 모두 개선됐다. classifier나 임계값을 selection에 맞춰
재선정하지 않았으므로 변화 원인은 소형 객체 gate 하나로 제한된다.

## 속도 측정 조건

- 장비: Windows 11, RTX 5080 16GB
- 경로: 실제 Worker `detector → ROI batch classifier → context ONNX → decision`
- Runtime: ONNX Runtime `1.28.0`, CUDA Execution Provider
- warm-up: 30장
- 측정: 난이도별 200장, 총 600장
- 결과: 모든 난이도의 평균과 P95가 `100ms` 이내

## 재현 명령

```powershell
$env:PYTHONPATH = "src"
python -m bixolon_scanner.training.rpc_small_object_recovery `
  --config configs\rpc_data_scale.json `
  --dataset-root C:\workspace\raw_data\archive\retail_product_checkout `
  --output-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated `
  --min-object-area-ratio 0.002

python -m bixolon_scanner.training.rpc_validation_benchmark `
  --package-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated\validation-candidate-package-small-object-v5 `
  --context-onnx artifacts\experiments\rpc-data-scale-diverse-worker-gated\runs\full\seed20260810\context-small-object-v5\logistic.onnx `
  --manifest artifacts\experiments\rpc-data-scale-diverse-worker-gated\detector\manifest\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\archive\retail_product_checkout `
  --output artifacts\experiments\rpc-data-scale-diverse-worker-gated\reports\validation-context-small-object-v5-benchmark.json `
  --provider cuda `
  --cuda-dll-dir C:\workspace\bixolon_scanner\apps\product_scanner\build\windows\x64\runner\Release\worker\cuda-runtime `
  --warmup 30 `
  --images-per-level 200
```

## Artifact

- selection 보고서: `runs/full/seed20260810/context-small-object-v5/report.json`
- context ONNX: `runs/full/seed20260810/context-small-object-v5/logistic.onnx`
- Worker 검증 패키지: `validation-candidate-package-small-object-v5`
- CUDA benchmark: `reports/validation-context-small-object-v5-benchmark.json`
- 기준 경로: `artifacts/experiments/rpc-data-scale-diverse-worker-gated`

이 결과는 validation candidate이며 production 승격이나 봉인된 `test2019` 평가를
의미하지 않는다.
