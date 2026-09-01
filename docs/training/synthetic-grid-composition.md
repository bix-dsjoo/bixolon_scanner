# 20종 빵 위치 인지 합성 데이터 가이드

## 목적과 범위

이 도구는 실제 다중 객체 사진을 확보하기 어려운 조건에서 고정 카메라로 촬영한 단일 빵 200장과
빈 트레이 1장을 사용해 detector 학습 전용 합성 장면 1,500장을 생성합니다. 합성 결과는 실제
환경의 독립 성능 증빙이나 인증 자료가 아니며, 실제 운영 사진을 이용할 수 있게 되면 별도 진단에
사용해야 합니다.

입력 계약은 다음과 같습니다.

- 빵 20종
- `normal`, `flipped` 두 실제 면
- 각 면을 `top_left`, `top_right`, `center`, `bottom_left`, `bottom_right`에서 촬영
- 빵 사진 `20 × 2 × 5 = 200`장
- 같은 카메라·트레이의 빈 배경 1장
- 총 201장, 모든 사진은 같은 해상도와 EXIF 정규화 방향

## 연구 근거와 구현 선택

- [Simple Copy-Paste](https://arxiv.org/abs/2012.07177)는 저데이터 환경에서 Copy-Paste가 box와
  mask 학습 효율을 높이고, 객체 subset과 scale jitter가 유용하다는 실험을 제공합니다.
- [Perspective-Aware Copy-Paste](https://arxiv.org/abs/2406.18586)는 위치와 perspective를 무시한
  무작위 붙여넣기의 한계를 지적합니다. 이 구현은 목표 위치에서 가장 가까운 5점 촬영 원본을
  선택하며 임의의 큰 perspective warp를 적용하지 않습니다.
- [Synthetic Object Compositions, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Synthetic_Object_Compositions_for_Scalable_and_Accurate_Learning_in_Detection_CVPR_2026_paper.html)은
  객체 중심 구성, 기하학적 layout 다양화, 정확한 mask·box, 조화된 blending의 중요성을 보여줍니다.
  이 구현은 실제 전경 mask, 가림 후 visible mask와 local luminance harmonization을 사용합니다.
- [NVIDIA visual-inspection synthetic-data workflow](https://developer.nvidia.com/blog/how-to-train-an-object-detection-model-for-visual-inspection-with-synthetic-data/)의
  structured domain randomization 원칙에 따라 위치·크기·조명·그림자·밀집도·가림을 명시적 범위
  안에서만 변형하고 모든 sampled parameter를 provenance에 저장합니다.
- [Albumentations bbox guide](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/)의
  bbox clipping과 minimum visibility 원칙에 맞춰 프레임 밖 영역과 다른 객체에 가려진 영역을
  반영한 visible bbox를 다시 계산합니다.

Diffusion 기반 generative harmonization은 제품의 토핑·표면·모양 정체성을 바꾸고 결과 재현을 어렵게
할 수 있어 사용하지 않습니다. 대신 실제 빈 배경의 위치별 luminance와 source 위치 luminance를
비교한 제한적 색 조화, feathered alpha와 일관된 장면 그림자를 사용합니다.

## 촬영 폴더 계약

`bread_01`부터 `bread_20`까지 category id를 고정합니다. 디렉터리 이름 뒤 slug는 자유롭게 지정할
수 있지만, 각 파일 stem은 아래 이름과 정확히 일치해야 합니다.

```text
capture_root/
  background.jpg
  bread_01_walnut_donut/
    normal_top_left.jpg
    normal_top_right.jpg
    normal_center.jpg
    normal_bottom_left.jpg
    normal_bottom_right.jpg
    flipped_top_left.jpg
    flipped_top_right.jpg
    flipped_center.jpg
    flipped_bottom_left.jpg
    flipped_bottom_right.jpg
  ...
  bread_20_plain_bread/
    같은 10개 파일
```

JPEG, PNG를 지원하지만 같은 stem의 파일은 하나만 존재해야 합니다. 누락 파일, 추가 이미지,
해상도 불일치, decode 실패와 동일 pixels 복사는 생성 전에 오류로 중단합니다. `flipped`는 좌우
mirror가 아니라 빵을 실제로 뒤집어 촬영한 면입니다. 각 위치에서는 큰 회전 변형을 피하도록
기본 방향을 유지하고, 합성기가 위치 perspective를 손상하지 않는 범위인 ±18도만 jitter합니다.

## 기본 1,500장 구성

| scenario | 수량 | 생성 조건 |
|---|---:|---|
| `NORMAL` | 600 | 빵 2~4개, tray 내부, mask overlap 없음 |
| `DENSE` | 300 | 빵 5~8개, 중앙 근처 밀집, 최대 1.8%의 접촉 허용 |
| `OCCLUSION` | 225 | 빵 3~6개, 적어도 한 객체가 10~30% 가려짐 |
| `EDGE` | 150 | `TRAY_EDGE_TOUCH` 또는 `FRAME_CLIPPED` |
| `LIGHTING` | 125 | 위치 gradient, 색온도, glare, 약한 blur의 제한 변형 |
| `RECAPTURE` | 100 | 빈 장면 25, blur 25, 과노출 20, 저노출 15, 심한 가림 15 |

20종 category 등장 횟수는 최대 차이 2회 이내로 균형을 맞추고 `normal`과 `flipped` 사용 횟수도
category별로 균형을 맞춥니다. `RECAPTURE`는 모델 판정 영역이며 `ERROR`로 변환하지 않습니다.
프레임 clipping과 객체 가림을 구분해 `frame_visibility_fraction`, `occlusion_fraction`, 최종
`visible_fraction`을 annotation과 provenance에 기록합니다.

## 실행

```powershell
bixolon-compose-grid-dataset `
  --capture-root D:\bread-grid-captures `
  --output-root artifacts\synthetic\bread-grid-1500-seed-20260831 `
  --seed 20260831
```

출력 디렉터리는 비어 있어야 하며 기존 산출물을 덮어쓰지 않습니다.

이미 흰 배경으로 잘라낸 객체가 `normal/flipped × (ground_30_dir_01~04 + vertical)` 이름으로
종류별 10장씩 준비된 경우에는 외부 빈 트레이를 지정합니다. 이 모드는 흰 배경을 alpha로 제거하고
네 방향과 수직 자세를 무작위로 선택하되 실제 면과 자세 자체에는 perspective 변형을 적용하지
않습니다.

```powershell
bixolon-compose-grid-dataset `
  --capture-root datasets\bread_dataset\single_objects `
  --background-image datasets\bread_dataset\operational_collections\2026-08-27\images\041_a65820e8740744f8a9984ab467fd7f95.jpg `
  --output-root artifacts\synthetic\single-objects-1500-seed-20260831 `
  --seed 20260831
```

- `images/`: 1,500개 JPEG
- `instances.json`: COCO bbox와 visibility metadata
- `manifest.jsonl`: 기존 detector exporter가 읽는 training manifest
- `source-manifest.json`: 입력 201장 경로와 SHA-256 lock
- `provenance.jsonl`: 장면별 source, 위치, 면, 변형, shadow와 품질 조건
- `metadata.json`: 분포·class/side/source 사용 횟수와 전체 artifact hash
- `preview.jpg`: 최초 12장 contact sheet

동일 입력과 seed는 동일 JPEG, annotation과 manifest hash를 생성합니다. 합성 파생물은 동일한 실제
물리 빵을 공유하므로 독립 validation fold로 나누지 않고 전부 `train_synthetic`, `fold=0`으로
기록합니다. 별도의 synthetic seed 평가는 생성기 회귀 진단일 뿐 실제 일반화 성능이 아닙니다.

## 생성 후 검수

생성 완료 후 최소한 다음을 확인합니다.

1. `preview.jpg`에서 mask halo, 잘린 토핑, 떠 있는 그림자와 비현실적 크기가 없는지 확인합니다.
2. `metadata.json`의 scenario 수량이 `600/300/225/150/125/100`인지 확인합니다.
3. category별 등장 횟수 최대·최소 차이가 2 이하인지 확인합니다.
4. `OCCLUSION` annotation이 10~30% 가림을 포함하고 visible bbox가 보이는 영역을 감싸는지 확인합니다.
5. `FRAME_CLIPPED`와 `TRAY_EDGE_TOUCH`가 별도 condition으로 기록되는지 확인합니다.
6. Runtime·Catalog 제품 버전은 학습 반복으로 변경하지 않습니다.
