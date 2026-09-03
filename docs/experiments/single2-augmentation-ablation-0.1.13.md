# `single_objects_2` 200장 증강 절제 실험

## 결론

`single_objects_2` 200장만 SKU 학습 원본으로 사용한 frozen DINOv3 ConvNeXt-Tiny 절제 실험에서
외관, 시점, 주변 객체 문맥을 모두 포함한 `balanced_5000`이 원본-시점 분리 probe의 Top-1과
Top-3 안전성을 함께 고려할 때 가장 좋았다.

이 결과는 0.1.13 Runtime 교체나 운영 일반화 인증이 아니다. 변형 규칙을 고르는 저비용 선행
실험이며, 다중 객체 데이터는 fitting, head 선택, threshold 선택에 사용하지 않았다.

## 데이터 계약

- 학습 원본: `datasets/bread_dataset/single_objects_2`, 20 class × 10장 = 200장
- `single_objects_2`의 200장은 `single_objects_3`에도 내용 해시가 모두 동일하게 존재한다.
- 같은 원본에서 나온 모든 파생 이미지는 항상 같은 fold를 상속한다.
- 세 fold는 촬영 시점 단위로 구성했다. fold 0은 `ground_0`과 `vertical`, fold 1은
  `ground_30_dir_01`과 normal `ground_60_dir_01`, fold 2는 `ground_30_dir_02`와 flipped
  `ground_60_dir_01`이다.
- 클래스마다 같은 물리 샘플 계열만 존재하므로 이 fold는 미관측 시점 검증이지, 미관측 물리 제품
  검증은 아니다.
- `multi_object_scenes`에서 미리 추출된 1,371 detector ROI feature는 모델 선택 후 개발 진단에만
  사용했다.

## 독립 변형 세트

| 세트 | 장수 | 학습 역할 | 핵심 변형 |
|---|---:|---|---|
| clean | 200 | SKU positive | 원본 |
| appearance | 1,600 | SKU positive | 조명, 명암, 채도, 약한 blur/JPEG, 중립 배경, 그림자 |
| geometry | 1,600 | SKU positive | ±180° 회전, 크기, perspective, 제한된 side compression |
| context | 1,600 | SKU positive | 다른 SKU 1~3개, 경계 문맥, 최대 16% 가림, neighbor ownership mask |
| probe | 600 | validation only | 원본당 appearance/geometry/context 각 1개, 별도 seed |
| reject | 800 | quality safety only | 강한 blur, 심한 노출, 큰 가림, edge content loss |

실제 생성 파일은 6,200장이다. SKU positive 학습 풀은 clean을 포함해 5,000장이고, probe 600장과
reject 800장은 SKU positive 학습에 넣지 않는다. 모든 파생 SHA-256 6,200개는 서로 달랐고, 원본
200개 중 서로 다른 fold로 유출된 원본은 0개였다.

## 결과

backbone은 고정하고 small-sample head만 학습했다. head와 규제값은 공통 probe의 nested
source-grouped OOF로 선택했다. 다중 객체 결과는 아래 후보 순위를 결정하는 데 사용하지 않았다.

| 후보 | 유효 학습 표본 | probe Top-1 | probe Top-3 | multi ROI Top-1 | multi ROI Top-3 miss |
|---|---:|---:|---:|---:|---:|
| clean | 200 | 74.00% | 89.00% | 87.96% | 30/1,371 |
| appearance | 1,800 | 83.00% | 94.00% | 92.49% | 33/1,371 |
| geometry | 1,800 | 84.50% | 94.67% | 91.03% | 34/1,371 |
| context | 1,800 | 84.17% | 94.17% | 90.66% | 24/1,371 |
| balanced half | 2,600 | 86.00% | 95.67% | 92.49% | 16/1,371 |
| balanced half + clean 재표집 | 3,200 | 86.83% | 95.67% | 93.51% | 17/1,371 |
| **balanced full** | **5,000** | **86.83%** | **96.17%** | **94.53%** | **15/1,371** |
| balanced full + clean 재표집 | 6,400 | 87.67% | 96.00% | 94.53% | 11/1,371 |

clean 재표집은 probe Top-1은 올렸지만 우선 KPI인 probe Top-3 miss를 23건에서 24건으로 1건
늘렸다. 따라서 다중 객체 진단의 11건만 보고 weighted 후보를 선택하지 않고 `balanced_5000`을
유지했다.

`balanced_5000`의 다중 객체 진단 Top-1 오류는 75건, Top-3 miss는 15건이다. 가장 큰 Top-1 혼동은
`Plain Bread → Mini Bread` 12건, `Muffin → Grain Campagne` 8건, `Dinner Roll → Egg Tart` 4건,
`Croffle → Waffle` 4건이었다. 최종 0건 목표에는 도달하지 않았으므로 이 synthetic 확장만으로
0 FP/FN 또는 Top-3 miss 0을 주장할 수 없다.

## 학습 권고

1. 저장할 SKU positive 데이터는 원본 200장과 세 변형 bank 각 1,600장, 합계 5,000장으로 고정한다.
2. 변형 bank는 하나로 섞어 다시 생성하지 않고 각각 manifest와 seed를 유지한다. 새 규칙의 효과는
   bank 단위로 제거해 재검증할 수 있어야 한다.
3. `probe`는 학습하지 않고 augmentation/head/epoch 선택에만 사용한다. test 또는 multi-object
   결과에 맞춰 규칙이나 threshold를 바꾸지 않는다.
4. `reject`는 SKU positive label로 학습하지 않는다. classifier margin만으로 RECAPTURE를 만들지도
   않는다. 실제로 balanced head의 median Top-1/Top-2 margin은 clean 0.808, probe 0.800,
   synthetic reject도 0.552여서 일부 품질 불량은 여전히 높은 SKU 확신을 보였다. detector quality
   또는 별도 quality head의 명시적 negative로만 사용한다.
5. 다음 단계는 `balanced_5000` 규칙으로 ConvNeXt-Tiny last stage를 짧게 fine-tune하고, 동일한
   source-grouped probe로 epoch와 threshold를 고정하는 것이다. 0.1.13 현재 모델과의 비교는 같은
   240-source 데이터와 섞지 말고 별도 후보로 수행한다.
6. 운영 일반화를 확정하려면 우선 혼동 쌍을 중심으로 각 SKU마다 다른 물리 제품, 다른 매장,
   카메라 높이, 조명, 트레이/배경을 최소 2개 독립 capture session에서 추가한다. 그 세트는 학습과
   threshold 선택에서 격리한다.

## 이론적 근거

- AugMix는 학습/운영 분포가 달라질 때 corruption robustness와 uncertainty가 악화되는 문제를
  다루므로 appearance bank를 별도로 검증하는 근거가 된다.
- Simple Copy-Paste는 단순 객체 합성도 데이터 효율을 높일 수 있음을 보이지만, perspective-aware
  연구는 장면의 위치와 원근을 무시한 합성이 비현실적일 수 있음을 지적한다. 그래서 context bank는
  무제한 합성이 아니라 중립 배경, 제한된 가림, ROI 경계 distractor로 제한했다.
- DINOv3는 다양한 downstream task에 재사용 가능한 frozen feature를 제공하므로, 비싼 fine-tune
  전에 증강 bank 효과를 빠르게 비교하는 proxy backbone으로 사용했다.

참고: [AugMix](https://arxiv.org/abs/1912.02781),
[Simple Copy-Paste](https://arxiv.org/abs/2012.07177),
[Content and Perspective-aware Cut-and-Paste](https://arxiv.org/abs/2406.18586),
[DINOv3](https://arxiv.org/abs/2508.10104)

## 재현 경로

- 설정: `configs/experiments/bread/single2_augmentation_ablation_0.1.13.json`
- 실행 코드: `src/bixolon_scanner/experiments/bread/single2_augmentation_ablation.py`
- 원본 audit: `artifacts/experiments/single2-augmentation-0.1.13/source-audit`
- 생성물·manifest·feature·보고서:
  `artifacts/experiments/single2-augmentation-0.1.13/ablation`
