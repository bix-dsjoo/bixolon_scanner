# Bread Classifier 200장 전용 1.1.1+ 계획

## 목표와 현재 기준선

Worker·Detector·Classifier `1.1.0`은 프로젝트 소유자의 2026-08-19 명시적 결정으로 운영
기준선에 승격한다. 이 버전은 최종 LDA head가 E/M/H 300장의 ROI 1,410개로 학습됐고 새 독립
잠금 test가 없다는 두 제한을 package waiver에 그대로 보존하는 bridge release다.

`1.1.1`부터는 Classifier의 학습 가능한 파라미터와 fitted statistic을
`manifests/bread-zero-error-1.1/classifier_manifest.jsonl`의 `single_objects` 200장만으로
만든다. 먼저 여섯 운영 gate를 모두 만족하는 최소 후보를 만들고, 같은 제한을 유지하면서
all-GT `APPROVED ≥99%` 장기 목표까지 patch version을 순차적으로 올린다.

## 변경할 수 없는 200장 데이터 계약

허용 범위는 다음과 같다.

- 원본은 `single_objects` 20종×10장, 총 200장과 고정 SHA-256 manifest뿐이다.
- resize, crop, flip, color·exposure 변화, box jitter, CutMix/MixUp, 절차적 배경과 같은 파생
  입력은 200장 픽셀 또는 난수만으로 결정적으로 생성한다.
- backbone, adapter, projection, prototype, linear/LDA/SVM head, class prior, normalization
  statistic, calibration, approval threshold, Top-3 ranking weight와 TTA 선택을 모두 200장 내부
  fold에서만 fit한다.
- 모든 run evidence에 실제 읽은 이미지 SHA 집합, manifest SHA, augmentation seed, checkpoint와
  ONNX SHA를 기록한다. 허용 manifest 밖의 이미지나 feature cache가 하나라도 섞이면 학습 시작
  전에 실패시킨다.

금지 범위는 다음과 같다.

- `multi_object_scenes`, `operational_collections`, GT crop과 detector ROI를
  loss, head fitting, prototype, normalization, calibration 또는 threshold 선택에 사용하지 않는다.
- 기존 `domain-lda-final` feature cache와 E/M/H logits를 successor 학습에 재사용하지 않는다.
- 평가 결과를 보고 같은 version의 threshold를 다시 맞추지 않는다. 새 가설은 다음 patch
  version에서만 실행한다.

200장은 클래스마다 동일한 물리 대상의 여러 view이므로 물리 대상 독립 validation을 만들 수
없다. 내부 fold는 view·perceptual group 누수만 차단하는 개발 선택용이며 독립 일반화 증거로
표시하지 않는다.

## 데이터 역할

| 데이터 | 1.1.1+ 역할 | 학습·fitting 허용 |
| --- | --- | --- |
| `single_objects` 200장 | 내부 train/calibration/nested validation | 허용 |
| E/M/H 300장·GT 1,410개 | 고정 end-to-end 개발 회귀와 버전 판정 | 금지 |
| 운영 수집본 115장·GT 504개 | 이미 공개된 운영 분포 개발 회귀 | 금지 |
| 새 촬영 세션 | 후보를 잠근 뒤 단 한 번 독립 test | 금지 |

E/M/H 또는 운영 수집본 실패를 분석해 새 가설을 만들 수는 있지만, 그 변경은 반드시 다음 patch
version으로 시작한다. 따라서 이 두 세트는 계속 development이고 독립 test로 복구되지 않는다.

## 버전별 실행 계획

### 1.1.1 — 계약 강제와 정직한 기준선

1. 학습 loader와 cache에 200장 allowlist 검증을 추가한다.
2. E/M/H로 fit한 LDA와 class threshold를 제거한다.
3. 200장만 사용한 DINOv3 ConvNeXt-Tiny checkpoint와 200장 내부 nested fold에서 선택한 head,
   calibration, threshold를 하나의 ONNX로 export한다.
4. 동일 Classifier가 최종 ROI 판정뿐 아니라 Detector proposal class verifier에도 사용되므로
   Detector box 결과까지 전체 회귀한다.
5. E/M/H와 운영 수집본을 한 번 평가하고 통과·실패와 원인을 기록한다. 실패하면 `rejected`로
   잠그고 1.1.2로 이동한다.

### 1.1.2 — ROI·배경 강건성

Detector box의 scale/translation/aspect 오차를 200장 crop에서 합성하고, 이웃 객체와 tray
배경은 200장 foreground 및 절차적 배경만으로 만든다. 한 번에 augmentation 가설 하나만 바꾸며
1.1.1 대비 class별 승인 오인, Candidate out과 coverage 변화를 기록한다.

### 1.1.3 — 소표본 표현과 head

frozen/last-stage adapter, cosine prototype, regularized linear head, shrinkage head를 200장 내부
nested fold에서 비교한다. 같은 fold의 metric으로 head와 threshold를 동시에 고르지 않고 inner
selection과 outer diagnostic을 분리한다.

### 1.1.4 — 선택적 추론과 속도

필요할 때만 TTA 또는 checkpoint ensemble을 실행하는 staged policy를 검토한다. 승인 오인과
Top-3 안전성을 유지하면서 CUDA full-path 평균·p95를 각각 100ms 이하로 맞춘다. 속도 때문에
threshold를 낮추거나 정확도 때문에 E/M/H에 threshold를 맞추지 않는다.

### 1.1.n — 반복 규칙

각 patch는 하나의 가장 영향력 있는 가설만 포함한다. 기준선 재현 → 실패 행 고정 → 가설 → 최소
실험 → 전체 E/M/H·운영 회귀 → parity·성능 순서로 실행한다. 실패 patch도 manifest, config,
checkpoint hash, 보고서와 원인을 `rejected`로 보존하고 다음 patch로 넘어간다.

## 최소 합격과 최종 종료 조건

첫 최소 합격 후보는 다음을 모두 만족해야 한다.

- `SEGMENTATION` 이미지 비율 ≥90%
- all-GT `APPROVED` 비율 ≥90%
- `SEGMENTATION` 이미지 FN·FP 비율 각각 ≤0.1%
- all-GT `APPROVED` 객체 오인율 ≤0.1%
- all-GT `UNKNOWN` Top-3 Candidate out 비율 ≤0.1%
- CPU/CUDA 최종 상태·Top-1·Top-3 parity와 package checksum 검증 통과
- RTX 5080 CUDA full-path 평균·p95 각각 ≤100ms
- Classifier 학습·fitting 입력이 200장 allowlist와 정확히 일치

최소 gate를 통과해도 `APPROVED <99%`이면 같은 데이터 계약으로 다음 patch 개선을 계속한다.
`APPROVED ≥99%`와 여섯 gate를 모두 만족한 후보를 고정한 뒤 새 촬영 세션을 잠그고 단 한 번
독립 평가한다. 독립 test까지 통과해야 200장 전용 successor를 `promoted`로 표시한다.

## 버전별 필수 산출물

- `configs/experiments/bread/...1.1.x.json`: 가설, seed, 중단 조건
- 200장 학습 allowlist audit와 실제 접근 SHA report
- PyTorch checkpoint 및 ONNX checksum과 source provenance
- 200장 nested-fold 선택 보고서
- E/M/H·운영 수집본 end-to-end 여섯 gate와 오류 행
- CPU/CUDA parity, RTX 5080 p50/p95/p99와 표본 수
- `promoted` 또는 `rejected` 결정 및 다음 patch 가설

## 2026-08-19 실행 상태

`1.1.1` 정직한 기준선은 DINOv3 ConvNeXt-Tiny 공개 사전학습 가중치에서 시작해 backbone을
고정하고, `single_objects` 200장의 원본과 결정적 파생 입력만으로 regularized linear head와
승인·Top-3 안전 threshold를 선택했다. 실제 읽은 200개 SHA 집합은 잠긴 allowlist와 정확히
일치했고 E/M/H, 운영 수집본, GT crop, detector ROI는 fitting 전에 접근하지 않았다.

200장 내부 nested OOF 파생 입력 600개에서 Top-1 95.667%, Top-3 98.333%, 승인 오인 0건을
유지한 승인 coverage는 83.667%였다. 후보를 잠근 뒤 E/M/H와 반려 운영 수집본을 각각 한 번
실행했다.

| `1.1.1` 지표 | E/M/H 300장 | 반려 운영 개발 115장 | gate |
| --- | ---: | ---: | ---: |
| `SEGMENTATION` | 300/300 (100%) | 111/115 (96.522%) | ≥90% |
| all-GT `APPROVED` | 942/1,410 (66.809%) | 293/504 (58.135%) | ≥90% |
| `SEGMENTATION` 이미지 FN | 5/300 (1.667%) | 1/111 (0.901%) | ≤0.1% |
| `SEGMENTATION` 이미지 FP | 2/300 (0.667%) | 0/111 | ≤0.1% |
| `APPROVED` 오인 | 1/1,410 (0.0709%) | 0/504 | ≤0.1% |
| Candidate out | 0/1,410 | 0/504 | ≤0.1% |
| CUDA 평균 / p95 | 95.63 / 93.44ms | 77.33 / 89.38ms | 각각 ≤100ms |

따라서 `1.1.1`은 `rejected`다. proposal class verifier도 같은 Classifier를 사용하므로 교체만으로
E/M/H 오류 이미지 `153, 155, 204, 259, 296, 298`과 운영 오류 이미지 `60`이 생겼다.
고정 판정과 checksum은
`manifests/bread-zero-error-1.1/classifier_200_only_1.1.1_rejected_2026-08-19.json`에 보존한다.

`1.1.2`는 head family와 backbone을 그대로 두고 augmentation 가설 하나만 바꾼다. 200장
foreground와 절차적 배경으로 detector box의 경계 이웃 clutter를 만들고 운영
neighbor-ownership mask를 동일하게 적용한다. E/M/H와 운영 수집본은 후보를 다시 잠근 뒤에만
한 번 평가하며 fitting에는 계속 사용하지 않는다.

`1.1.2` 실행 결과 내부 OOF Top-1/Top-3는 96.167%/99.333%로 개선됐지만 무오류 승인
coverage는 78.667%로 낮아졌다. 잠긴 end-to-end 결과도 E/M/H `APPROVED` 873/1,410
(61.915%), 운영 개발 재사용본 288/504(57.143%)로 `1.1.1`보다 낮았다. E/M/H FN/FP 포함
이미지는 각각 5/300, 1/300이고 운영 FN 포함 이미지는 1/111이었다. 따라서 `1.1.2`도
`rejected`로 고정하고 `1.1.3`에서는 입력 recipe를 유지한 채 소표본 head family만 비교한다.
