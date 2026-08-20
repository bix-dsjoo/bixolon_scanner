# Scanner 2.0.0 RC.8 운영 수집본 115장 개발 평가

- 평가일: 2026-08-19
- 후보: `2.0.0-rc.8`
- 데이터: `datasets/bread_dataset/operational_collections/2026-08-18`
- 규모: 이미지 115장, 판정 가능 GT 객체 504개
- 데이터 구성: 객체 이미지 111장, 빈 트레이 4장
- evidence role: `development_regression`, `promotion_evidence=false`

> 2026-08-20 후속 결정: 소유자 waiver로 RC.8이 `2.0.0` production에 승격됐지만, 이 115장은
> 계속 development evidence이며 독립 승격 test로 재분류하지 않는다.

프로젝트 소유자 결정에 따라 이 115장은 더 이상 독립 승격 test가 아니다. RC.7의 승인 오인 원인을
분석하고 RC.8 ambiguity·OOD policy를 선택하는 데 사용했으므로 300장과 함께 개발 계보로 잠근다.
이 결과를 독립 일반화 성능이나 production 승격 증거로 재분류할 수 없다.

## 변경한 정책

RC.7은 정규화된 Ridge logit margin 하나로 승인해 `bread_02 Croffle`을 `bread_03 Waffle`로
승인한 두 건을 방어하지 못했다. 두 객체 모두 Detector box는 정확했고 정답이 Classifier Top-2와
retrieval Top-1에 있었다. RC.8은 특정 클래스 pair 예외 없이 다음 순서로 판정한다.

1. retrieval Top-1 유사도 `< 0.414268881082535`이면 높은 Ridge score와 무관하게
   `CLASSIFIER_OUT_OF_CATALOG` `SEGMENT_RECAPTURE`로 보낸다.
2. Ridge Top-1/Top-2 logit gap을 `sigmoid(gap)`으로 바꾼 pair probability가 `< 0.54923`이면
   `CLASSIFIER_AMBIGUOUS_TOP2` `UNKNOWN`+Top-3로 보낸다.
3. Ridge와 retrieval Top-1이 다를 때도 같은 안전 경계를 넘지 못하면 `UNKNOWN`으로 보낸다.
4. 위 방어를 통과한 객체만 `APPROVED`한다.

`0.54923`은 오인 score와 다음 올바른 승인 score 사이에서 CPU/CUDA 수치 여유를 둔 값이다.
매장별로 재선택하지 않으며 새 비공개 test를 본 뒤에도 변경하지 않는다.

## 결과

| 지표 | 분자 / 분모 | 결과 | 개발 Gate | 판정 |
|---|---:|---:|---:|---|
| `SEGMENTATION` | 110 / 115 이미지 | **95.6522%** | ≥90% | PASS |
| `IMAGE_RECAPTURE` | 5 / 115 이미지 | **4.3478%** | 진단 | - |
| `APPROVED` | 468 / 504 GT | **92.8571%** | ≥90% | PASS |
| `UNKNOWN` Top-3 | 26 / 504 GT | **5.1587%** | 진단 | - |
| `SEGMENT_RECAPTURE` | 4 / 504 GT | **0.7937%** | 진단 | - |
| `SEGMENTATION` 이미지 FN | 0 / 110 이미지 | **0%** | ≤0.1% | PASS |
| `SEGMENTATION` 이미지 FP | 0 / 110 이미지 | **0%** | ≤0.1% | PASS |
| `APPROVED` 객체 오인 | 0 / 504 GT | **0%** | ≤0.1% | PASS |
| `UNKNOWN` Candidate out | 0 / 504 GT | **0%** | ≤0.1% | PASS |
| 전체 평균 / p95 / p99 | 115장 | **75.32 / 94.52 / 95.37ms** | 진단 | - |
| full-path 평균 / p95 / p99 | 110장 | **76.53 / 94.54 / 95.37ms** | ≤100 / 100 / 150ms | PASS |

빈 트레이 4장은 `DETECTOR_NO_OBJECT` `IMAGE_RECAPTURE`, 객체 이미지 1장은
`DETECTOR_UNCERTAIN_OBJECT` `IMAGE_RECAPTURE`였다. `SEGMENTATION`으로 진행한 이미지에서는
Detector IoU@0.5 FN/FP가 없었다.

기존 승인 오인 두 건은 모두 `CLASSIFIER_AMBIGUOUS_TOP2` `UNKNOWN`으로 바뀌었다. 이미지 83의
해당 box는 approval score `0.5354593`, 이미지 90은 `0.5488229`이며 두 Top-3 모두 정답
`bread_02 Croffle`을 포함한다. 따라서 틀린 `APPROVED`를 줄이면서 사용자가 제안한 Top-3 방어
의미를 유지한다.

## 결론과 증거 역할

115장 개발 회귀는 모든 point gate와 CUDA 성능 gate를 통과했다. 이 통과는 RC.8 정책 선택의
개발 근거일 뿐 독립 인증 통과가 아니다. 별도 private test 요구는 2026-08-20 소유자 예외로
면제됐으며, 이 결과 자체의 evidence 역할은 바뀌지 않는다.

재현 evidence:

- 집계 report:
  `artifacts/evaluations/scanner-2.0.0/operational-2026-08-18-115-rc.8-final-cuda.json`
- trace:
  `artifacts/evaluations/scanner-2.0.0/operational-2026-08-18-115-rc.8-final-cuda-trace.jsonl`
