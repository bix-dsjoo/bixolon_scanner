# Scanner 2.0.0 RC.7 Bread Project 5 승격 판정

- 평가일: 2026-08-19
- 후보: `2.0.0-rc.7`
- 데이터: `C:/workspace/raw_data/bread_project_5`
- 규모: 이미지 3,000장, GT 객체 13,861개
- 소유자 지시: 파생 데이터라는 제한을 인지한 상태에서 승격 판정에 사용
- 최종 판정: **`rejected`**

> 이 문서는 RC.7 반려 기록이다. 후속 RC.8은 2026-08-20 별도 소유자 waiver로 `2.0.0`
> production에 승격됐으며 RC.7 실패 판정을 뒤집지 않는다.

## 데이터 계보와 manual waiver

COCO 메타데이터상 3,000장은 `bread_project_2`와 2026-08-18 운영 source의 415개 원본 장면을
반전·회전·밝기·명암·이동한 8개 augmentation profile로 구성한다. 파일 SHA-256은 모두 고유했고
잠긴 개발 계보와 exact 중복 및 dHash Hamming distance 2 이하 근접 중복은 각각 0장이었다. 그러나
같은 원본의 파생 이미지는 통계적으로 독립된 촬영 trial이 아니다.

프로젝트 소유자는 이 제한에도 불구하고 세트를 승격 판정에 쓰도록 지시했다. 이에 따라
`owner_private_unseen_capture_provenance`와 최소 독립 certification group 수만 `manual_waiver`로
남겼다. waiver는 point metric gate를 면제하지 않으며, 결과를 독립 일반화 성능 또는 3,000개 독립
trial로 표현할 수 없다.

## 고정 RC.7 결과

100회 warm-up 뒤 ONNX Runtime CUDA EP로 threshold 변경 없이 3,000장을 한 번 평가했다.

| 지표 | 분자 / 분모 | 결과 | Gate | 판정 |
|---|---:|---:|---:|---|
| `SEGMENTATION` | 2,910 / 3,000 이미지 | 97.0000% | ≥90% | PASS |
| `IMAGE_RECAPTURE` | 90 / 3,000 이미지 | 3.0000% | 진단 | - |
| `APPROVED` | 12,634 / 13,861 GT | 91.1478% | ≥90% | PASS |
| `UNKNOWN` Top-3 | 107 / 13,861 GT | 0.7720% | 진단 | - |
| `SEGMENT_RECAPTURE` | 719 / 13,861 GT | 5.1872% | 진단 | - |
| `SEGMENTATION` 이미지 FN | 27 / 2,910 이미지 | **0.9278%** | ≤0.1% | **FAIL** |
| `SEGMENTATION` 이미지 FP | 25 / 2,910 이미지 | **0.8591%** | ≤0.1% | **FAIL** |
| `APPROVED` 객체 오인 | 19 / 13,861 GT | **0.1371%** | ≤0.1% | **FAIL** |
| `APPROVED` 출력 중 오인 | 19 / 12,634 출력 | 0.1504% | 진단 | - |
| `UNKNOWN` Candidate out | 2 / 13,861 GT | 0.0144% | ≤0.1% | PASS |
| 전체 평균 속도 | 3,000장 | 58.54ms | - | - |
| full-path 평균 / p95 / p99 | 2,910장 | 59.20 / 73.73 / 77.77ms | ≤100 / 100 / 150ms | PASS |

Detector는 객체 FN 32개와 FP 26개를 기록했다. 실패는 H에서 집중됐다. H 1,000장 중 FN 포함
15장, FP 포함 15장과 승인 오인 15건이 발생했고, M은 각각 12장, 6장, 4건이었다. E에서는 FN
포함 이미지와 승인 오인이 없고 FP 포함 4장만 있었다. 승인 오인 19건 중 가장 큰 confusion은
`bread_02 → bread_17` 8건과 `bread_02 → bread_03` 3건이었다.

## 승격 결정

파생 provenance waiver를 적용해도 FN 이미지, FP 이미지, 승인 오인 point gate 세 개가 실패했다.
따라서 이 시험을 RC.7 승격 판정으로 사용한 결과는 `rejected`다. `2.0.0` production 승격,
당시 production 변경은 수행하지 않았고 운영 기본값은 `1.1.0`이었다. 후속 RC.8 승격 결정은
`configs/releases/scanner_2.0.0.json`을 따른다.

재현 evidence:

- 집계 report: `artifacts/evaluations/scanner-2.0.0/bread-project-5-3000-rc.7-cuda.json`
- trace: `artifacts/evaluations/scanner-2.0.0/bread-project-5-3000-rc.7-cuda-trace.jsonl`
- owner-waiver decision:
  `artifacts/evaluations/scanner-2.0.0/bread-project-5-3000-rc.7-owner-waiver-promotion-decision.json`
