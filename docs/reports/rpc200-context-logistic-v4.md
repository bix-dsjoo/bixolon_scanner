# RPC200 context-logistic-v4 validation 실험 보고서

## 1. 목적과 현재 상태

이 문서는 `retail_product_checkout` 200개 클래스에 운영 Worker와 같은
detector → ROI batch → DINOv3 classifier → 상태 결정 흐름을 적용한 validation
개선 실험을 재현하고 검토하기 위한 기록이다.

최종 validation 목표는 다음과 같다.

- Easy, Medium, Hard 각각 인식률 99% 이상
- Easy, Medium, Hard 각각 오인율 0.5% 이하
- RTX 5080에서 난이도별 이미지당 평균 및 P95 100ms 이하
- `RECAPTURE`를 인식률 분모에서 제외하되 별도로 보고
- `UNKNOWN` 증가로 목표를 우회하지 않도록 End-to-End 성공률을 함께 보고

`context-logistic-v4`는 위 validation 목표를 모두 통과했다. 이 결과는 운영
승격이나 최종 test 결과가 아니다. `prepared/experiment.json`의
`test_accessed`는 최종 확인 시점에도 `false`였으며 `test2019`는 봉인되어 있다.

## 2. 데이터와 분리 계약

- 데이터 루트: `C:\workspace\raw_data\archive\retail_product_checkout`
- 클래스: COCO category ID `1..200`
- classifier backbone: DINOv3 ConvNeXt Tiny
- validation: `val2019`
- 촬영 그룹: 파일명 마지막 ID를 capture group으로 사용
- classifier adaptation: calibration group 중 2/3만 checkout adaptation에 사용
- risk calibration: adaptation에 사용하지 않은 나머지 calibration group
- 최종 비교: untouched `selection` 36,329개 detector ROI
- test: `test2019`, validation 목표 통과 전 접근 금지

동일 촬영 그룹이 adaptation과 risk calibration에 동시에 들어가지 않으며,
selection은 classifier와 context validator 학습에 사용하지 않았다.

## 3. Worker 실험 상태 계약

실험 결과는 다음 의미로 집계했다.

- 이미지: `SEGMENTATION` 또는 `IMAGE_RECAPTURE`
- 각 segmentation: `APPROVED`, `UNKNOWN+TOP3`, `SEGMENT_RECAPTURE`
- 인식률: 올바른 `APPROVED / 비-RECAPTURE 인식 대상`
- 오인율: 잘못된 `APPROVED / 전체 APPROVED`
- End-to-End 성공률: 올바른 `APPROVED / 전체 정답 영역`
- `IMAGE_RECAPTURE`는 segmentation 놓침에 중복 집계하지 않음

공개 bread Worker/API enum과 20종 label map은 이 실험에서 변경하지 않았다.

## 4. 개선 이력

### v1: 기존 classifier 기준선

Frozen classifier의 Top-1은 약 60%, partial fine-tuning은 약 78%였다. stage
선정 동률 처리 문제를 고쳐 partial checkpoint를 선택했지만 목표에는 크게
미달했다.

### v2: recapture-positive full fine-tuning

OOF detector 예측 bbox 중 COCO GT와 일치한 ROI만 정상 양성으로 사용하고,
unmatched detector ROI를 hard negative로 추가했다. Top-1이 약 90.58%로 개선됐지만
목표에는 미달해 폐기했다.

### v3: checkout group-safe adaptation

calibration capture group을 group-aware로 분리해 checkout adaptation을 수행했다.
classifier matched ROI Top-1은 99.86%까지 상승했다. 남은 잘못된 `APPROVED`는
상품 분류 오류가 아니라 detector의 unmatched 추가 박스였다.

v3 Worker 지표의 인식률은 Easy 95.95%, Medium 96.41%, Hard 96.19%였다.
classifier threshold만 변경한 sweep은 오인율 0.5% 제한에서 세 난이도 최저
인식률이 96.65%에 머물렀다.

### 폐기한 변경

- 일괄 NMS 하향: Hard 정답 박스 recall이 먼저 감소
- DINO segment rejector: `SEGMENT_RECAPTURE` 약 98.6%, E2E 약 1.4%로
  재촬영을 통한 지표 우회
- class별 confidence/margin threshold: selection 인식률 및 unmatched 승인이 악화
- ExtraTrees validator: offline 정확도는 더 높았지만 Worker 런타임에
  scikit-learn을 추가해야 하므로 최종 배포 가능 후보에서 제외

### v4: context logistic validator

최종 후보는 각 detector segment에 대해 22개 feature를 계산한다.

- detector score, box area/aspect/center/border distance
- 이미지 내 detection 개수와 score rank
- 다른 박스와 IoU, containment, overlap count
- classifier confidence, margin, entropy, Top-3 probability mass, logit margin

calibration 112개 촬영 그룹을 5-fold로 나누어 각 calibration ROI가 자신을
학습하지 않은 logistic validator 점수를 받게 했다. 세 난이도 모두 인식률
99% 이상, 오인율 0.5% 이하인 정책 중 End-to-End 성공률을 우선해 다음 정책을
고정했다.

- classifier approval threshold: `0.9892165785617951`
- segment quality threshold: `0.00513339033649039`
- context model: `rpc-context-validator-logistic-v1`

## 5. 최종 selection 결과

| 난이도 | 인식률 | 오인율 | UNKNOWN Top-3 포함률 | IMAGE_RECAPTURE | SEGMENT_RECAPTURE | 놓침률 | 오검출률 | E2E 성공률 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 99.2242% | 0.2167% | 96.2264% | 1.3917% | 1.0482% | 0.8089% | 0.9915% | 96.6419% |
| Medium | 99.5873% | 0.2645% | 100.0000% | 0.8000% | 1.0585% | 0.6697% | 0.9689% | 97.7636% |
| Hard | 99.3650% | 0.4202% | 100.0000% | 1.0091% | 1.2362% | 1.1309% | 1.1833% | 96.7230% |

## 6. 요청 형식: 실패 outcome 구성비 100% 표

아래 네 열은 일반적인 전체 데이터 비율이 아니다. 실패·후속처리 outcome의
구성을 보기 위해 다음 원시 건수를 합한 뒤 각 항목을 그 합계로 나누었다.

1. Segmentation 실패 이미지: detector hard gate로 `IMAGE_RECAPTURE`된 이미지 수
2. Top-3 Candidate: `UNKNOWN` 중 정답이 Top-3에 포함된 segment 수
3. Candidate Out: `UNKNOWN` 중 정답이 Top-3에 없는 segment 수
4. 오인율 구성: 잘못 `APPROVED`된 segment 수

서로 다른 단위(이미지와 segment)를 하나의 outcome event 표에 넣은 요청형
요약이므로, 모델 KPI에는 5절의 개별 분모 지표를 사용해야 한다.

| 버전 | 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율 구성 | 합계 | 속도 평균 / P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| context-logistic-v4 | Easy | 99.2242% | 17.0732% (14건) | 62.1951% (51건) | 2.4390% (2건) | 18.2927% (15건) | 100.0000% (82건) | 53.00 / 62.06ms |
| context-logistic-v4 | Medium | 99.5873% | 8.9888% (8건) | 55.0562% (49건) | 0.0000% (0건) | 35.9551% (32건) | 100.0000% (89건) | 63.32 / 73.97ms |
| context-logistic-v4 | Hard | 99.3650% | 5.4348% (10건) | 56.5217% (104건) | 0.0000% (0건) | 38.0435% (70건) | 100.0000% (184건) | 68.91 / 76.69ms |

## 7. RTX 5080 지연 검증

측정 환경과 범위:

- GPU: NVIDIA GeForce RTX 5080 16GB
- driver: 591.86
- ONNX Runtime GPU: 1.28.0, CUDA Execution Provider
- 난이도별 selection 이미지 200장, 총 600장
- 고정 SHA-256 order로 표본 선택
- end-to-end warm-up 30회
- JPEG decode → detector ONNX → ROI crop/batch → classifier ONNX → context ONNX → 판정

| 난이도 | 표본 | 평균 | P50 | P95 | P99 | full-path P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 200 | 53.0002ms | 52.5376ms | 62.0568ms | 65.5952ms | 62.0597ms |
| Medium | 200 | 63.3225ms | 62.6380ms | 73.9685ms | 79.4132ms | 73.9724ms |
| Hard | 200 | 68.9080ms | 68.7591ms | 76.6890ms | 80.5745ms | 76.7001ms |

## 8. ONNX parity와 코드 검증

- context logistic ONNX 최대 절대 오차: `7.14986678806984e-7`
- selection threshold 판정 불일치: `0건`
- 전체 pytest: 통과
- `ruff check src tests`: 통과
- `git diff --check`: 통과

## 9. 재현 명령

기본 detector·학습·selection 흐름:

```powershell
bixolon-rpc-data-scale `
  --config configs\rpc_data_scale.json `
  --dataset-root C:\workspace\raw_data\archive\retail_product_checkout `
  --weights C:\workspace\raw_data\model_cache\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth `
  --output-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated `
  --phase all `
  --resume
```

단계별 phase는 `detector`, `adapt-detector`, `prepare`, `train`, `select`,
`test`, `all`이다. 기존 artifact를 검토할 때 `test`는 실행하지 않는다.

context validator 재계산:

```powershell
$env:PYTHONPATH = "src"
python -m bixolon_scanner.training.rpc_context_rejector `
  --config configs\rpc_data_scale.json `
  --output-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated
```

RTX validation benchmark:

```powershell
$env:PYTHONPATH = "src"
python -m bixolon_scanner.training.rpc_validation_benchmark `
  --package-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated\validation-candidate-package `
  --context-onnx artifacts\experiments\rpc-data-scale-diverse-worker-gated\runs\full\seed20260810\context-rejector\logistic.onnx `
  --manifest artifacts\experiments\rpc-data-scale-diverse-worker-gated\detector\manifest\manifest.jsonl `
  --dataset-root C:\workspace\raw_data\archive\retail_product_checkout `
  --output artifacts\experiments\rpc-data-scale-diverse-worker-gated\reports\validation-context-logistic-benchmark.json `
  --provider cuda `
  --cuda-dll-dir C:\workspace\bixolon_scanner\apps\product_scanner\build\windows\x64\runner\Release\worker\cuda-runtime `
  --warmup 30 `
  --images-per-level 200
```

## 10. Artifact 위치와 SHA-256

대형 모델·예측·원본 보고서는 Git에 커밋하지 않는다. 아래 로컬 artifact의
SHA-256으로 나중에 동일 실험인지 확인한다.

| Artifact | SHA-256 |
|---|---|
| `runs/full/seed20260810/partial.pt` | `928cf167c3c9028a0f46fa2b1f64b7935b25c152f61921fedd240d322abfc0ea` |
| `runs/full/seed20260810/context-rejector/logistic.onnx` | `bb8ecb42b9e2beac7bba3500ed451e8dc3de5f619ae2f664981a3d62fd08327b` |
| `runs/full/seed20260810/context-rejector/report.json` | `d4e01878f47b2b60f690624679337d039ad4e2e52b986219496b3b1294f58c4f` |
| `reports/validation-context-logistic-benchmark.json` | `e8fbf2a3b711f6424e24d6555f67183b45811f841cf6ad9de512bb66a5b292e3` |
| `detector/threshold.json` | `262618ab5e78650478d1e984e61a35840db941c01e0e0f112374e8605d68cd7f` |
| `prepared/experiment.json` | `28085fa70cae89fa28e7c71ebdaefa8c7900a063f161fc5eed4c59620b667757` |
| `runs/full/seed20260810/selection_predictions.npz` | `c1c21dcd584da19e706778b4d3b33330967651ee0c4a540bd744653376b78b0c` |

모든 상대 경로의 기준은
`artifacts/experiments/rpc-data-scale-diverse-worker-gated`이다.

## 11. 운영 승격 전에 남은 일

1. context feature 계산과 ONNX session을 정식 Worker pipeline contract에 통합
2. package metadata에 context model, threshold, checksum, schema를 추가
3. PyTorch/ONNX CPU/CUDA 전체 parity 검증
4. validation 통과 model lock 생성 후 `test2019`를 한 번만 평가
5. test 실패 시 다른 threshold나 모델을 test에 맞춰 재선택하지 않음
6. 고정 test 통과 후 production package와 API 회귀 검증

현재 단계는 `validation_candidate`이며 production 승격 상태가 아니다.
