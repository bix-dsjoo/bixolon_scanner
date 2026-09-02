# Store 2 `single_objects_4` 완전 초기화 실험 기록

이 문서는 2026-09-02에 중단한 Store 2 완전 초기화 실험의 데이터, 모델 계보, 평가 결과와
재개 지점을 보존한다. 기록된 결과는 격리된 실험 산출물이며 활성 제품 `0.1.12` 또는 기존 배포
번들의 성능을 대체하지 않는다.

## 한눈에 보는 현재 상태

| 구분 | 올바른 `APPROVED` | 잘못된 `APPROVED` | FN | FP | 상태 |
| --- | ---: | ---: | ---: | ---: | --- |
| 실제 `DecisionPipeline` 최고 기록 | 1,398/1,410 (99.1489%) | 9 | 3 | 15 | 99%만 통과, 최종 목표 미달 |
| 후속 ONNX detector bbox 평가 | 해당 없음 | 해당 없음 | 0 | 0 | detector 구성요소 목표 달성 |
| 후속 detector + 기존 fresh classifier 진단 | 1,397/1,410 (99.0780%) | 13 | 0 | 0 | 실제 `DecisionPipeline` 미실행, 분류 목표 미달 |
| 인공 옆면 fine-tune classifier 진단 | 1,327/1,410 (94.1135%) | 83 | 0 | 0 | 폐기 |

요청된 최종 조건은 실제 `DecisionPipeline`에서 잘못된 `APPROVED` 0, FN 0, FP 0이다. 후속
detector가 FN/FP 0을 달성했지만 분류 오류가 남았고, 이 detector와 classifier 조합으로 실제
`DecisionPipeline`을 끝까지 실행하지 않았으므로 전체 목표는 완료되지 않았다.

## 고정 범위와 불변성

- 허용 학습 원천은 `datasets/bread_dataset/single_objects_4`의 상품 이미지 200장과 빈 배경
  이미지 10장뿐이다.
- 기존 매장 모델, checkpoint, cache, ONNX 및 Catalog는 초기값이나 학습 입력으로 사용하지 않았다.
- 사전학습 가중치는 공식 DINOv3 ConvNeXt-Tiny와 ViT-B/16만 사용했다.
- 평가는 `datasets/bread_dataset/multi_object_scenes` 300장과 GT 객체 1,410개를 사용했다.
- bbox 평가는 기존 greedy matching과 IoU 0.5 기준을 사용했다.
- 외부 `0.1.12` 추론 단계와 정책 순서는 변경하지 않았다. 후속 detector의 포함 fragment 제거는
  ONNX graph 내부 score 억제로 export했다.
- 활성 `configs/versions/0.1.12.json`과 기존 배포 번들은 수정하지 않았다.
- 문서 작성 시 활성 설정 SHA-256은
  `91cbf30877e249f9147ffc96975deaf86419999c97901f8c06e27186b83ffe58`이다.

## 원천 annotation 검수

최종 사용 원천은
`artifacts/experiments/bread-store2-single4-0.1.12/prepared-source-sam-v3`다.

- 20개 class contact sheet에서 상품 200개의 mask와 bbox overlay를 모두 직접 확인했다.
- 자동 prompt가 실패한 7개는 원본 이미지에 한정된 point/box prompt로 보정하고 overlay를 다시
  확인했다.
- 최종 review 상태는 `APPROVED`다.
- 상품 원본 image set SHA-256:
  `155af9f0aa3eb6a130f78f8a0586f7785a7ff9dec715e105565352b85f897b87`
- classifier manifest SHA-256:
  `1c51ce2bcec7829941f444db7ad694c7668a3cda8955586c9f65dc8472ea37c8`
- detector manifest SHA-256:
  `be45d5a14310d7d0844c7a9262833d99dd15f70005e93147eb8c24524bec3eb3`
- 검수 원문: `prepared-source-sam-v3/annotation-review.json`
- class별 overlay: `prepared-source-sam-v3/qa/bread_01.jpg`부터 `bread_20.jpg`

실제 원본 bbox에서 긴 변/짧은 변 비율이 2 이상인 표본은 200장 중 5장뿐이었다.

| class | 원본 표본 수 | aspect ratio |
| ---: | ---: | ---: |
| 6 | 1 | 2.075 |
| 9 | 1 | 2.012 |
| 15 | 2 | 3.191, 2.087 |
| 17 | 1 | 2.160 |

따라서 옆면 오류가 집중된 class 2에는 실제 긴 옆면 support가 없다. 정면 cutout을 가로 또는 세로로
강하게 압축하는 합성은 detector recall에는 도움이 됐지만, 실제 3차원 옆면 질감과 윤곽을 재현하지
못해 classifier에는 부작용이 컸다.

향후 실제 옆면을 추가로 확보할 수 있다면 손이 일부 포함되더라도 실제 옆면을 우선한다. annotation
단계에서 상품 mask는 손을 제외하고, skin/hand segmentation과 GrabCut 또는 직접 보정으로 경계를
정리한다. 학습 시에는 손이 제거된 상품 cutout을 기본으로 사용하고, 실전 가림 대응용으로 손 모양
occluder를 별도 확률로 다시 합성하는 편이 인공 압축보다 적절하다.

## 실험 계보

### 1. 초기 source-only 합성 및 전체 파이프라인

`synthetic-v1`은 seed `20260901`로 640×640 장면 6,000장을 만들었다. 객체 annotation은
25,059개, 빈 장면은 502장이다. 객체 수 0~7개, scale 0.10~0.44, 최대 180도 회전, 경계 배치,
동일 class 중복, 가림, 조명·색·blur·JPEG 변화를 포함한다. 첫 100개 합성 overlay를 확인했다.

초기 detector는 공식 DINOv3 ConvNeXt-Tiny frozen backbone과 새 FPN/FCOS objectness head를
20 epoch 학습했다. classifier는 공식 DINOv3 ViT-B/16 frozen backbone, 원본 support 200개,
2-view/4-view rotation, diagonal LDA와 prototype/Top-3 exemplar 및 고정 pair rule을 사용했다.
support 200개 재식별은 200/200이다.

실제 `DecisionPipeline` 최고 결과는
`evaluation-runtime-pipeline-v3/report.json`이며 객체별 결과는 같은 디렉터리의 `trace.jsonl`에
있다.

| 항목 | 결과 |
| --- | ---: |
| 이미지 / GT | 300 / 1,410 |
| prediction / bbox 매칭 | 1,422 / 1,407 |
| 올바른 `APPROVED` | 1,398 |
| 올바른 `APPROVED` 비율 | 99.1489% |
| 잘못된 `APPROVED` | 9 |
| FN / FP | 3 / 15 |
| `UNKNOWN` / `SEGMENT_RECAPTURE` | 0 / 0 |
| `IMAGE_RECAPTURE` 이미지 | 0 |
| CPU p95 | 1,467.6 ms |

`evaluation-runtime-pipeline-v2`는 잘못된 `APPROVED`가 13개였고 pair rule 수정 후 v3에서 9개로
줄었다. detector 오류는 그대로 남았다.

### 2. Faster R-CNN v2와 completeness verifier

`synthetic-v2`는 10,000장, annotation 45,454개, 빈 장면 1,021개로 확대했다. Faster R-CNN v2는
기준 threshold에서 bbox 매칭 1,409, FN 1, FP 2였다. 낮은 threshold로 recall을 넓히면 FN 0,
FP 8이었고 containment 및 DINO completeness verifier를 적용해도 최선이 FN 1, FP 5 수준이라
최종 후보로 채택하지 않았다.

근거는 다음 경로에 있다.

- `evaluation-dinov3-fasterrcnn-v2/report.json`
- `evaluation-containment-fasterrcnn-v2/report.json`
- `evaluation-completeness-verifier-v1/report.json`
- `evaluation-completeness-verifier-v2/report.json`

### 3. 옆면 합성 v3와 FN/FP 0 detector

`synthetic-v3-sideview`는 detector의 옆면 recall을 보강하기 위한 합성 세트다.

| 항목 | 값 |
| --- | ---: |
| 이미지 | 12,000 |
| annotation | 53,902 |
| 빈 장면 | 1,183 |
| 객체 수 | 0~8 |
| scale | 0.08~0.50 |
| hard scene 확률 | 0.50 |
| 경계 배치 확률 | 0.12 |
| 동일 class 중복 확률 | 0.30 |
| 옆면 압축 적용 객체 | 22,745 (42.1970%) |
| 압축 비율 | 0.28~0.58 |
| seed | 20260907 |

- manifest SHA-256:
  `3642eba8fea3ac6a34288c25c6fe706986fc3dc46da51bcb616e24a750e07713`
- provenance SHA-256:
  `fbd43b4cc6aec684bad70425355fe830821da94ac1909275b3895d61e4ccc37d`
- 외부 학습 이미지 접근: `false`
- QA 대상: 200장

새 Faster R-CNN은 공식 DINOv3 ConvNeXt-Tiny에서 시작해 FPN, RPN, ROI object classifier와 ROI box
regressor를 새로 만들었다. 마지막 backbone stage 하나만 풀고 12 epoch 학습했다. 초기 detector
checkpoint는 `null`이다.

- 공식 ConvNeXt-Tiny SHA-256:
  `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`
- checkpoint: `detector-dinov3-fasterrcnn-v3-sideview/last.pt`
- checkpoint SHA-256:
  `6424226e9d246dd3d59448990e1e93ca86e7cbd8ec2d3b709eafb968198ca7a7`
- 최종 loss: box 0.031207, classifier 0.014065, objectness 0.000159, RPN box 0.000592

PyTorch 평가에서 score threshold 0.735의 결과는 매칭 1,410, FN 0, FP 1이었다. 남은 FP는 image
33의 실제 객체 내부에 생긴 좁은 fragment였다. 후보가 다른 후보에 70% 이상 포함되고 후보 면적이
포함한 후보의 35% 이하인 경우 낮은 score 후보를 억제하자 FN 0, FP 0이 됐다. 이 처리는 외부
pipeline 정책을 바꾸지 않고 ONNX graph에 포함했다.

- ONNX: `detector-dinov3-fasterrcnn-v3-sideview/detector-contained-v2.onnx`
- ONNX SHA-256:
  `596a9e7d240a9ebc0f44e277e348a0f7581c1acf92b897e12dc27ca97aea2502`
- CPU ONNX Runtime 예측: `evaluation-onnx-fasterrcnn-v3-sideview/detector-predictions.jsonl`
- bbox 증빙: `evaluation-onnx-fasterrcnn-v3-sideview/bbox-report.json`
- 결과: 300장, GT 1,410, prediction 1,410, 매칭 1,410, FN 0, FP 0

첫 export인 `detector-contained.onnx`는 `EyeLike` 연산 때문에 ONNX Runtime에서 로드하지 못한 실패
산출물이다. 사용 가능한 것은 반드시 `detector-contained-v2.onnx`다.

### 4. 원본 중심 ViT-B classifier와 남은 13개 오류

후속 detector crop에 `fresh-runtime-candidate-v3/embedder.onnx`를 적용한 결과는
`evaluation-onnx-fasterrcnn-v3-sideview/classification-current.json`에 있다.

- 공식 ViT-B/16 SHA-256:
  `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- embedder SHA-256:
  `534424dbfb071dd575954ebc55679cf5c73ec2d7e29f8d4b7906cf445ba51808`
- 학습/support 원천: 검수된 상품 원본 200장
- 기존 checkpoint 또는 Catalog 사용: `false`
- bbox 매칭: 1,410, FN 0, FP 0
- 올바른 분류: 1,397/1,410 (99.0780%)
- 잘못된 분류: 13

이 embedder는 class one-hot을 출력하므로 13개가 모두 높은 확신의 잘못된 `APPROVED`가 된다.

| image | annotation | GT → 예측 | IoU |
| ---: | ---: | ---: | ---: |
| 209 | 935 | 8 → 16 | 0.94 |
| 222 | 990 | 20 → 19 | 0.98 |
| 226 | 1022 | 2 → 17 | 0.97 |
| 227 | 1028 | 5 → 4 | 0.93 |
| 245 | 1100 | 8 → 20 | 0.94 |
| 248 | 1116 | 2 → 17 | 0.81 |
| 250 | 1124 | 17 → 16 | 0.88 |
| 257 | 1168 | 8 → 16 | 0.93 |
| 267 | 1232 | 1 → 4 | 0.91 |
| 267 | 1233 | 2 → 17 | 0.93 |
| 288 | 1329 | 18 → 20 | 0.86 |
| 299 | 1402 | 10 → 9 | 0.80 |
| 299 | 1399 | 9 → 17 | 0.74 |

가장 반복적인 오류는 class 2 → 17 세 건과 class 8 → 16 두 건이다. image 248의 class 2는 매우 긴
옆면 bbox였지만 class 2의 원본 support에는 대응하는 실제 옆면이 없다.

### 5. 인공 옆면 classifier 실험과 폐기 근거

`classifier-finetune-v2-sideview`는 검수된 source에서 새 classifier head를 만들고, 직접 ROI 합성에
확률 0.42와 압축 0.28~0.58의 인공 옆면을 넣었다. 공식 DINOv3 이외의 초기 checkpoint는 사용하지
않았다.

- checkpoint SHA-256:
  `fac8090137b2bcdd240c25eca485722b03b135a994cb2531a2071e0e4ca3eb6a`
- ONNX SHA-256:
  `91ee54dd1517406072f49d1d9432d24cbfd4f6e4abe78d6a562b2b2311ec3c`
- 평가: 1,327/1,410 정분류, 83개 오분류, FN 0, FP 0
- 근거: `evaluation-onnx-fasterrcnn-v3-sideview/classification-sideview-v2.json`

정면 형태를 기하학적으로 찌그러뜨린 영상이 상품 고유 형태까지 훼손해 원본 중심 classifier보다
70개 더 틀렸다. 이 모델은 재사용하지 않는다.

원본 ViT-B support마다 인공 옆면 4개를 추가하려던 `fresh-runtime-candidate-v4-sideview`도 원본
support 재식별이 96.5%로 떨어져 exporter가 중단됐다. 유효 ONNX와 실험 디렉터리는 생성되지 않았다.

## 산출물 지도

모든 상대 경로의 기준은
`artifacts/experiments/bread-store2-single4-0.1.12`다.

| 목적 | 경로 | 사용 여부 |
| --- | --- | --- |
| 최종 검수 source | `prepared-source-sam-v3` | 사용 |
| 초기 합성 | `synthetic-v1` | 과거 전체 pipeline 증빙 |
| 확대 합성 | `synthetic-v2` | 비교용 |
| 옆면 detector 합성 | `synthetic-v3-sideview` | 후속 detector 학습 |
| FN/FP 0 detector | `detector-dinov3-fasterrcnn-v3-sideview` | 가장 좋은 detector |
| ONNX bbox 평가 | `evaluation-onnx-fasterrcnn-v3-sideview/bbox-report.json` | detector 최종 증빙 |
| 원본 중심 classifier | `fresh-runtime-candidate-v3` | 분류 기준선 |
| detector+classifier 진단 | `evaluation-onnx-fasterrcnn-v3-sideview/classification-current.json` | 후속 구성요소 증빙 |
| 인공 옆면 classifier | `classifier-finetune-v2-sideview` | 폐기 |
| 실제 pipeline 최고 평가 | `evaluation-runtime-pipeline-v3` | 전체 pipeline 최고 증빙 |
| 과거 격리 Runtime | `fresh-runtime-package-v5` | 활성 배포 아님 |
| 과거 격리 Catalog | `fresh-catalog-v2` | 활성 배포 아님 |

`cache-*`, `classifier-diagnostic-*`, `prepared-source-v2`~`v7`와 이름에 `failed`가 포함된 디렉터리는
중간 진단 또는 실패 산출물이다. 완전 초기화 재학습의 초기값으로 사용하지 않는다. 결과 비교가
필요할 때만 읽기 전용 증빙으로 본다.

## 재개 시 권장 순서

1. 활성 `configs/versions/0.1.12.json` SHA-256과 기존 bundle checksum을 먼저 기록한다.
2. `prepared-source-sam-v3/annotation-review.json`과 class별 QA overlay를 다시 확인한다. 실제 옆면을
   추가할 권한이 생기면 손을 상품 mask에서 제거해 source provenance를 새로 만든다.
3. strict fresh run에서는 이 문서의 checkpoint, cache, ONNX 및 Catalog를 초기값으로 재사용하지
   않고 공식 DINOv3 가중치에서 다시 시작한다. 기존 산출물은 비교 기준으로만 사용한다.
4. detector는 `synthetic-v3-sideview` 설정을 출발점으로 삼되 인공 압축은 detector에만 제한한다.
   ONNX export 뒤 CPU ONNX Runtime에서 300장 bbox 평가를 다시 수행한다.
5. classifier는 검수된 실제 원본 200장을 중심으로 유지한다. 인공 압축 support는 사용하지 않는다.
   실제 옆면이 없는 class는 2-view/4-view, LDA, prototype의 불일치와 margin을 출력해 보수적으로
   `UNKNOWN`으로 보내는 방식을 먼저 검토한다.
6. 특히 2→17, 8→16, 8→20, 20→19, 17→16, 18→20 confusion을 분석한다. 잘못된 13개를 모두
   `APPROVED`에서 제외하면서 올바른 객체를 1개보다 많이 제외하면 99% 기준을 잃으므로 단순 전역
   threshold보다 confusion별 disagreement가 필요하다.
7. 마지막 평가는 구성요소 합산이 아니라 fresh Runtime/Catalog를 실제 `DecisionPipeline`에 로드해
   수행한다. 완료 조건은 올바른 `APPROVED` 1,396개 이상과 동시에 wrong `APPROVED` 0, FN 0,
   FP 0이다.
8. 평가 후 활성 설정과 기존 bundle hash가 시작 시 값과 같은지 다시 검증한다.

관련 실행 진입점은 다음과 같다.

- source 준비/검수: `scripts/approve_store2_annotations.py`
- 합성 생성: `scripts/generate_store2_synthetic.py`
- detector 학습: `scripts/train_store2_dinov3_fasterrcnn.py`
- detector export: `scripts/export_store2_dinov3_fasterrcnn.py`
- ONNX 후보 평가: `scripts/evaluate_store2_onnx_candidate.py`
- classifier export: `scripts/export_store2_vitb16_lda_embedder.py`
- 격리 Runtime/Catalog 조립: `scripts/build_store2_fresh_runtime.py`
- 실제 pipeline 평가: `scripts/evaluate_store2_combined.py`

## 해석 제한

- 300장 세트는 요청에 따라 반복 오류 분석과 pair-rule 선택에도 사용된 development regression
  set이다. 이 수치를 독립 test 일반화 성능, 인증 또는 SLA로 해석하지 않는다.
- 후속 detector의 FN 0 / FP 0은 고정 300장 세트의 bbox evaluator 결과이며 실제
  `DecisionPipeline` 전체 성공을 뜻하지 않는다.
- 99.1489%는 당시 실제 pipeline 최고 기록이고, 99.0780%는 후속 detector crop에 classifier를
  적용한 구성요소 진단이다. 서로 다른 실행 범위이므로 한 결과처럼 합치지 않는다.
- CPU p95는 RTX 5080 또는 N100 성능 증빙이 아니다.
- 이 문서에 기록한 checkpoint와 ONNX는 배포 승인을 받은 자산이 아니며 활성 구성에 연결하지 않았다.
