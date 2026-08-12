# 엄격한 10-shot classifier `0.2.1` 실험 보고서

## 결론

`0.2.1`은 `experiment_only`이다. 전체 development calibration에서는 false-approval risk control을 처음 만족했지만 approval coverage `85%` 하한과 capture-session 교차 calibration을 통과하지 못했다. test split, ONNX export, parity와 benchmark는 pre-test gate 규칙에 따라 실행하지 않았다. 운영 package는 `bread-worker-0.1.1`을 유지한다.

## 변경 범위

- detector ONNX와 pipeline/API 계약은 변경하지 않았다.
- 학습 입력은 `bread_project_3`의 정확한 20×10장만 사용했다.
- white-threshold alpha 대신 border-connected background mask를 사용했다.
- foreground에 7% padding, 약한 조명 gradient와 그림자를 적용했다.
- DINOv3 ConvNeXt Tiny의 global feature와 49개 local patch feature를 결합했다.
- 클래스마다 `normal`, `flipped` 두 proxy를 사용하되 proxy는 ONNX 가중치 후보에 포함하고 runtime support cache는 사용하지 않았다.
- 원본 ROI와 93% center crop을 batch 추론해 TTA disagreement로 logit confidence를 낮췄다.
- 마지막 ConvNeXt stage만 L2-SP로 한 번 미세조정했다.

## 데이터와 재현성

| 항목 | 값 |
|---|---:|
| 학습 source | 200장 |
| 증강 ROI | 6,400개 |
| source당 증강 | 32개 |
| local patch | ROI당 49×768 |
| seed | 20260812, 20260813, 20260814 |
| development 이미지 | 205장 |
| CUDA detector 이후 matched ROI | 889개 |
| test 접근 | 안 함 |

모든 증강은 200개 source SHA 중 하나를 provenance로 가진다. development/test 이미지는 학습 batch에 포함되지 않았다.

## 결과

### Frozen local/global 2-proxy head

| seed | fold 0 | fold 1 | fold 2 | 평균 Top-1 |
|---:|---:|---:|---:|---:|
| 20260812 | 93.2927% | 92.8058% | 94.3463% | 93.4816% |
| 20260813 | 92.9878% | 93.1655% | 92.5795% | 92.9109% |
| 20260814 | 92.9878% | 91.7266% | 92.9329% | 92.5491% |

### Last-stage L2-SP challenger

| seed | fold 0 | fold 1 | fold 2 | 평균 Top-1 |
|---:|---:|---:|---:|---:|
| 20260812 | 95.4268% | 95.6835% | 94.3463% | **95.1522%** |
| 20260813 | 95.4268% | 94.6043% | 93.2862% | 94.4391% |
| 20260814 | 94.8171% | 93.5252% | 91.8728% | 93.4050% |

선택 checkpoint SHA-256은 `f618ed21b40c273a66b2fd6c427558c203922613d2b1da7f0ef7f28ac9f374e2`이다.

### Calibration과 승격 판단

| 지표 | 결과 | 하한/상한 | 판정 |
|---|---:|---:|---|
| Top-1 | 95.1631% | ≥95% | 통과 |
| Top-3 | 99.3251% | ≥95% | 통과 |
| 승인 수 | 680/889 | 1건 이상 | 통과 |
| 승인 precision | 100% | ≥99.5% | 통과 |
| false-approval 95% 상한 | 0.4396% | ≤0.5% | 통과 |
| approval coverage | 76.4904% | ≥85% | **실패** |
| capture-session 3-fold risk control | 승인 0건×3 | 각 fold 유효 승인 필요 | **실패** |

coverage 85%인 756개까지 승인하면 오분류가 5개 발생해 precision `99.3386%`, false-approval 95% 상한 `1.3855%`가 된다. 첫 번째 고신뢰 오분류 순위는 681위이다.

CPU detector로 준비했던 기존 `0.2.0`의 동일한 886개 ROI에서 다시 비교하면 Top-1 `95.2596%`, Top-3 `99.2099%`, risk-control coverage `75.0564%`이다. 따라서 CUDA detector의 3개 ROI 차이를 제거해도 승격 결론은 바뀌지 않는다.

## 취약 클래스

낮은 Top-1 클래스는 `Dinner Roll` 84.78%, `Mini Bread` 85.71%, `Half-moon Croissant` 86.96%, `Plain Bread` 88.10%, `Sandwich` 88.57%이다. 주요 혼동은 `Plain Bread → Pastry Bread` 5건, `Mini Bread → Plain Bread` 4건, `Half-moon Croissant → Flower Bread` 4건이다.

local/2-proxy 구조는 Bread18 앞·뒤 변화 표현에는 합리적이었지만 전체 정확도는 `0.2.0` 최선 96.0497%보다 낮았다. 다음 실험은 local/2-proxy를 동시에 적용하지 않고, 기존 single-proxy L2-SP 구조에 개선된 crop만 적용하는 단일 요인 ablation부터 시작해야 한다.

## Detector 관찰

detector ONNX SHA-256은 `635dd93ccde8a244692ad4cc14aaf259790d59d05fbde20e00588d47f3edefdc`로 유지됐다. 다만 같은 detector에서 CPU는 `DETECTOR_UNCERTAIN_OBJECT` 3장, CUDA는 2장을 반환했고 image 95가 경계에서 달라졌다. classifier 변경과 별개인 기존 provider parity 문제이며 이 실험에서는 detector를 수정하지 않았다.

## 검증

- 전체 pytest: 154개 통과
- Ruff: 통과
- `git diff --check`: 오류 없음
- test 접근: false
- ONNX export: 미실행
- RTX 5080 full-path benchmark: 미실행
