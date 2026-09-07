# 스캐너 카메라 설치 각도·높이 선정 검토안

작성 기준일: 2026-09-04
문서 성격: 설치값 확정 전 검토·실험 계획

## 1. 결론 요약

카메라 각도는 **트레이 법선(완전한 수직 하향)을 `0°`로 정의**한다. 이 정의에서 `30°`는 캔의
측면을 많이 보여 주지만, 한 장의 영상 안에서 근거리·원거리 상품의 크기 차이, 키스톤 왜곡, 상품 간
가림과 반사 조건도 함께 키운다. 따라서 현재 단계에서 `30°`를 바로 양산값으로 확정할 근거는 부족하다.

권고안은 다음과 같다.

1. 1차 기구 시제품은 `15°`, `20°`, `25°`를 재현할 수 있게 만들고, 명목 시작점은 **약 `18°`**로 둔다.
2. 최종값은 `15°·20°·25°`의 동일 조건 실험에서 선택한다. 필요할 때만 `10°`와 `30°`를 대조군으로
   추가한다.
3. 문헌만으로 각도를 정하지 않고, **동일한 상부 모양을 가진 캔 SKU의 오인식 감소량**과 전체 장면의
   `APPROVED` 오판, `UNKNOWN`, `RECAPTURE`, 검출 누락, 가장자리 성능을 함께 측정한다.
4. 모든 필수 조건을 통과한 후보 중 **가장 작은 각도**를 채택한다. 최고 평균 정확도 하나만 보고 큰
   각도를 고르지 않는다.
5. 높이는 각도와 별개의 고정값이 아니다. 실제 렌즈의 유효 화각, 트레이 폭·깊이, 최대 상품 높이,
   여유 영역을 넣어 계산해야 한다. 현재 앱은 1920×1080 전체가 아니라 중앙 **960×960 크롭**을
   저장하므로 이 유효 화면을 기준으로 계산한다.
6. 공간이 허용되면 짧은 초점거리 광각 렌즈를 낮게 다는 구성보다 **카메라를 높이고 더 긴 초점거리
   렌즈를 사용해 같은 검사 영역을 맞추는 구성**을 우선 검토한다. 상품 높이로 생기는 원근·시차 변화와
   광각 렌즈 왜곡을 줄이기 쉽다.
7. 캔의 반사는 각도를 더 기울여 해결하지 않는다. 확산 조명과 필요 시 교차 편광을 별도 실험한다.

> 각도를 테이블 면 기준으로 말하는 조직도 있다. 이 문서의 `18°`는 테이블 면 기준 `72°`와 같다.
> 만약 기존의 “30°”가 테이블 면 기준이었다면 수직 기준으로는 `60°`이므로 본 검토 범위보다 훨씬
> 비스듬한 구성이다. 도면과 회의 자료에는 반드시 기준면을 함께 표기해야 한다.

## 2. 왜 하나의 이론값으로 결정할 수 없는가

2025년 ICCV 연구는 여러 비전 파운데이션 모델이 작은 시점 변화에도 특징 표현이 크게 달라질 수 있고,
특정 시점에서는 3차원 구조가 우연히 가려져 오분류가 생기는 `accidental viewpoint`가 존재한다고
보고했다. 2024년 NeurIPS의 능동 시점 선택 연구도 모든 장면에 같은 시점을 적용하기보다, 기존 관측과
가장 다른 정보를 주는 시점을 선택할 때 분할과 복원 성능이 좋아짐을 보였다. 즉 “비스듬할수록 정보가
항상 많다”가 아니라, **새로운 측면 정보의 이득과 왜곡·가림의 손실 사이에 최적 구간이 있다**.

리테일 상품은 특히 SKU 간 모양과 색이 비슷한 세립도 분류 문제다. RPC 데이터셋 연구는 통제된
단품 사진과 실제 다품목 계산대 사진 사이의 차이를 핵심 난제로 다루며, 리테일 다중 카메라 연구는
시점별 특징과 서로 보완적인 관측이 유용하다고 보고한다. 이 결과들은 다각도 정보의 필요성을
지지하지만, 특정 단일 카메라의 설치 각도를 제시하지는 않는다. 설치값은 대상 SKU, 트레이, 렌즈,
조명과 현재 모델을 사용한 현장 실험으로 정해야 한다.

### 2.1 추가 문헌 조사 결과

| 문헌 | 촬영·실험 구성 | 확인된 결과 | 이번 설치 결정에 쓸 수 있는 근거 | 그대로 적용할 수 없는 이유 |
|---|---|---|---|---|
| RPC, 2019 및 IncreACO, WACV 2021 | 단품 exemplar는 상부·30°·45°·수평의 4개 고정 카메라와 회전판으로 수집하고, 다품목 checkout 이미지는 80cm×80cm 판의 상부 카메라로 촬영 | 다각도 단품 데이터와 실제 상부 checkout 장면 사이의 cross-domain 문제를 정의 | `30°·45°` 시점은 측면 외관을 학습 데이터에 포함할 가치가 있다는 근거 | 30°나 45°가 단일 checkout 카메라의 최적 설치각이라는 비교 실험은 아님 |
| MVTec D2S, ECCV 2018 | 1920×1440 산업용 카메라를 회전판 위에 두되 중심에서 약간 벗어나게 설치; 장면별 36° 간격 10회 회전, 조명 3종 | 회전·조명·가림·배경 변화가 있는 21,000장 데이터셋 구성 | 완전한 정중앙 수직만 쓰기보다 작은 off-center 관측을 시험하고, 각도와 조명을 분리 평가해야 함 | 카메라 경사각의 수치와 각도별 인식 성능을 제공하지 않음 |
| Jeon et al., IEEE Sensors Journal 2022 | 여러 카메라의 view-specific 검출 결과를 결합; 실제 구매 동작 142,420장 | view-aware 방법이 기존 방법보다 F1 기준 33.67% 개선 | 비슷한 리테일 상품은 시점마다 식별력이 다르고 보완 시점이 효과적임 | 다중 카메라 결과이므로 단일 카메라를 크게 기울였을 때의 효과로 환산 불가 |
| Yao et al., AutoRetail Digital Twins, 2023 | 3D 상품의 camera distance·height, 자세와 조명을 실제 checkout 영상에 맞게 최적화 | ablation에서 카메라 위치(distance, height)가 domain dissimilarity에 지배적인 영향을 보였고, 실제 분포에 맞춘 합성 데이터가 counting 성능을 개선 | 각도뿐 아니라 높이·거리와 실제 운영 시점 분포를 함께 맞춰야 하며, 설치 후 해당 분포로 재학습·평가해야 함 | 손으로 상품을 통과시키는 영상형 checkout으로 현재 정지 트레이와 동작 분포가 다름 |
| Multi-View Active Fine-Grained Visual Recognition, ICCV 2023 | 세립도 객체를 여러 시점에서 관측하고 다음 식별 시점을 능동적으로 선택 | 현재 시점에 식별 단서가 없으면 단일 영상의 이론적 한계가 있으며, 추가 시점을 선택하는 방법이 다중 시점 인식을 개선 | 동일 상부 캔처럼 단서가 특정 면에만 있으면 “조금 더 기울이기”가 아니라 보완 시점이 필요할 수 있음 | 차량 데이터 기반이며 고정 리테일 카메라 각도를 제시하지 않음 |
| Learning to Select Views, CVPR 2024 | 다중 시점 분류·검출에서 유용한 view만 선택 | 전체 `N`개 중 2~3개 시점만 사용해 계산량을 줄이면서 성능을 유지 | 단일 카메라 한계를 넘을 때 많은 카메라보다 상부+보조 측면 2개 구성이 실용적 후보가 될 수 있음 | 단일 고정 카메라 최적화 연구가 아님 |
| Spatial Resolution Metric for Optimal Viewpoints, ICVS 2023 | 센서 모델과 3D mesh를 ray tracing하여 표면 가시성과 공간 해상도를 함께 최적화 | 표면 coverage와 요구 spatial resolution을 동시에 만족하는 viewpoint 생성 | 높이·각도는 화각 충족만이 아니라 가장 먼 SKU의 px/mm 및 식별 면 coverage로 평가해야 함 | 결함 검사·3D 모델 기반이며 SKU 분류 정확도는 다루지 않음 |
| ABO, CVPR 2022 | 실제 상품 3D 모델을 upper icosphere의 91개 시점과 3개 조명에서 렌더링해 multi-view cross-domain retrieval 평가 | 일반적인 catalog 시점에 없는 관측과 배경 변화가 retrieval을 어렵게 함 | 상품 인식 평가에 방위각·고도각·조명을 체계적으로 샘플링할 필요가 있음 | 가정용 상품 catalog/retrieval 데이터이며 checkout 밀집 장면이 아님 |
| FGPR, Pattern Recognition 2026 | 85,733개 SKU, 357,622개 상품 이미지에 SKU triplet과 OCR text·bounding box를 제공하고 vision/OCR fusion 평가 | OCR text가 세립도 상품 retrieval에서 중요한 역할을 함 | 캔 측면의 “면적”만 볼 것이 아니라 상품명·용량 등 식별 문자의 가독 픽셀 수를 각도 선택 지표로 넣어야 함 | 전자상거래 고품질 이미지 중심이므로 checkout 가림·반사를 직접 재현하지 않음 |
| MIMEX fine-grained retail benchmark, 2024 | 28개 세립도 리테일 상품에서 VLM zero-shot과 CLIP·DINOv2 ensemble을 평가 | 범용 VLM의 zero-shot 세립도 분류가 만족스럽지 않았고, 소수의 visual prototype 적응이 개선을 보임 | 파운데이션 모델이라는 이유만으로 새 설치 시점에 강건하다고 가정하지 말고 실제 캔 prototype을 등록·평가해야 함 | 규모가 작고 카메라 설치각 비교가 아님 |
| Not all Views are Created Equal, ICCV 2025 | ABO·CO3D에서 DINO, DINOv2, ConvNeXt 등 9개 모델의 시점 안정성 분석 | 안정/불안정 시점을 특징으로 분리할 수 있었고, DINO는 ABO에서 94.08%, CO3D에서 77.08% 분리 정확도를 보임; accidental view는 구조 단서를 가려 downstream 성능을 저해 | 현재 분류기가 강하더라도 설치 시점의 blind spot을 별도 탐색해야 함 | 이 수치는 상품 분류 정확도나 특정 설치각의 우수성을 뜻하지 않음 |
| NVIDIA Retail Object Detection 기술 문서 | 학습 데이터 일부에 약 10ft 높이, 수직에서 45°인 checkout view와 상부 view 등을 포함 | 여러 실제 환경·높이·화각을 fine-tuning 데이터에 포함 | 산업 예시에서도 한 시점만 가정하지 않고 설치 분포를 학습 데이터에 반영함 | peer-reviewed 비교가 아니며 45° 최적성을 증명하지 않음; 현재 장비보다 훨씬 높은 설치 |

### 2.2 문헌에서 실제로 도출할 수 있는 결론

추가 문헌을 종합해도 `18°`, `20°` 또는 `30°` 중 하나를 보편적 최적값으로 지목할 수는 없다. 다만
다음 네 가지는 비교적 일관되게 지지된다.

1. **30°·45° 영상은 가치가 있다.** 다만 RPC에서는 주로 isolated SKU의 외관을 학습하기 위한
   exemplar 시점이고, 다품목 checkout 설치각의 최적성 증거가 아니다.
2. **운영 카메라 분포와 학습 데이터 분포가 같아야 한다.** 높이·거리·각도를 바꾸면 모델의 입력
   분포가 바뀌므로, 기존 모델 성능이 그대로 유지된다고 가정할 수 없다.
3. **각도만 최대화하면 안 된다.** viewpoint planning 연구는 coverage뿐 아니라 spatial resolution,
   focus, FOV와 occlusion을 동시에 제약한다.
4. **한 시점에서 식별 단서가 물리적으로 보이지 않으면 모델만으로 해결할 수 없다.** 단일 카메라의
   완만한 경사로 충분하지 않은 동일 상부 SKU가 많다면, 극단적인 단일 경사보다 상부 주 카메라와 작은
   보조 측면 카메라의 2-view 구성을 별도 비용안으로 비교하는 편이 근거가 강하다.

따라서 `15~25°`를 1차 단일 카메라 후보로 두는 권고는 논문에서 직접 얻은 정답 각도가 아니라, 원통
투영 기하와 문헌이 제시한 다목적 제약 사이의 **검증 가능한 공학 가설**이다. 보고서에는 이 표현을
사용해 과도한 근거 주장을 피한다.

## 3. 캔에서 `30°`가 과하게 느껴지는 기하학적 이유

원근을 무시한 1차 근사에서, 세워 둔 원통의 높이를 `H`, 지름을 `D`, 수직 기준 카메라 각도를
`θ`라고 하면 영상에서 추가로 보이는 측면 높이와 윗면의 세로 지름 비는 다음과 같다.

`측면 노출 / 윗면 세로 지름 ≈ (H / D) × tan(θ)`

일반적인 긴 캔을 `H/D = 1.8`로 놓으면 다음과 같다.

| 수직 기준 각도 | 측면/윗면 비(근사) | 윗면 세로축 축소 `cos(θ)` |
|---:|---:|---:|
| 10° | 0.32 | 0.985 |
| 15° | 0.48 | 0.966 |
| 18° | 0.58 | 0.951 |
| 20° | 0.66 | 0.940 |
| 25° | 0.84 | 0.906 |
| 30° | 1.04 | 0.866 |

따라서 `30°`에서는 보이는 측면 높이가 윗면의 투영 세로 지름과 거의 같아진다. 사용자가 느낀
“너무 넓고 길게 보임”은 자연스러운 결과다. 반면 `15~20°`에서는 측면 라벨 정보가 이미 윗면 세로
지름의 약 절반 이상 노출되므로, 동일 상부 캔을 구분할 가능성을 만들면서 형상 변화는 더 작게 유지한다.

이 표는 각도 후보를 줄이기 위한 근사일 뿐이다. 실제 영상에는 핀홀 원근, 렌즈 방사 왜곡, 상품 위치와
높이, 렌즈 주점, 조명이 추가로 작용한다.

## 4. “왜곡”을 두 종류로 나누어 보고해야 하는 이유

### 4.1 렌즈 왜곡

배럴·핀쿠션 왜곡은 렌즈 설계 때문에 화면 위치별 배율이 달라지는 현상이다. 넓은 화각과 짧은
초점거리에서 커지기 쉽다. 체커보드 또는 도트 타깃으로 내부 파라미터와 왜곡 계수를 추정해 보정할 수
있다.

### 4.2 원근·키스톤 및 시차

기울어진 카메라에서 트레이의 먼 쪽과 가까운 쪽이 서로 다른 배율로 보이는 현상이다. 평평한 트레이
면은 homography로 펴 보일 수 있지만, 높이가 있는 캔의 측면과 서로 가리는 관계까지 하나의 2차원
변환으로 없앨 수는 없다. 상품 판정용 원본에 무조건 원근 보정을 적용하기보다, 보정 전·후를 실제
모델로 비교해야 한다.

보고 시에는 다음과 같이 표현하는 것이 정확하다.

> 렌즈 왜곡은 캘리브레이션으로 상당 부분 보정할 수 있으나, 경사 촬영에서 높이가 있는 상품에 생기는
> 원근과 시차는 정보 자체가 달라지는 현상이므로 완전히 복원할 수 없다.

## 5. 높이 산정 방법

### 5.1 먼저 필요한 입력값

- 검사 영역의 실제 폭 `W`와 깊이 `D` (mm)
- 허용할 최대 상품 높이 `Hmax`와 최대 상품 폭
- 카메라 모델, 센서 크기 또는 실제 1920×1080 수평·수직 화각
- 렌즈 초점거리와 왜곡 사양
- 카메라 기구물이 아니라 **렌즈 입사동/주점에서 트레이 면까지의 높이**
- 앱이 저장하는 중앙 960×960에서의 실측 유효 화각 `φsquare`
- 가장 작은 상품이 분류에 필요한 최소 원본 픽셀 수
- 조명, 조리개, 노출시간에서 확보되는 피사계 심도

카메라 데이터시트 화각만 사용할 수 있다면 중앙 크롭의 1차 근사는 다음과 같다.

`φsquare ≈ 2 × atan(0.5 × tan(HFOV1920 / 2))`

이는 가로 1920픽셀 중 중앙 960픽셀만 쓰기 때문이다. 실제 렌즈 왜곡과 디지털 처리 여부가 있으므로
최종값은 트레이 면의 캘리브레이션 타깃을 촬영해 실측한다.

### 5.2 수직 카메라의 첫 계산

수직 하향 카메라가 평면에서 한 변 `L`을 보고, 전체 여유율을 `m`으로 둘 때 첫 추정 높이는 다음과
같다.

`h0 ≈ L × (1 + m) / (2 × tan(φsquare / 2))`

여유율은 초기 시제품에서 `10~15%`로 시작하되, 상품이 가장자리에 놓이는 실제 분포로 조정한다.

### 5.3 기울어진 카메라의 첫 계산

광축이 트레이 중심을 향하고 수직 기준 `θ`만큼 기울었을 때, 유효 세로 화각을 `φ`라 하면 트레이
면에서 광축 교점부터 가까운 쪽과 먼 쪽 경계까지의 거리는 각각 다음과 같이 근사할 수 있다.

- 가까운 쪽: `h × [tan(θ) - tan(θ - φ/2)]`
- 먼 쪽: `h × [tan(θ + φ/2) - tan(θ)]`

트레이 중심에 광축을 맞출 경우 두 값 중 작은 쪽이 필요한 반 깊이보다 커야 한다. 가로 방향의 중심부
폭은 `2h × tan(φ/2) / cos(θ)`로 근사할 수 있다. 다만 실제 높이는 평면 네 모서리만이 아니라
`Hmax`를 포함한 **3차원 검사 체적의 8개 모서리**를 카메라 모델로 투영해 모두 960×960 안전 영역
안에 드는지 확인하여 정한다.

중앙 트레이 면까지 광축상 working distance는 `WDcenter = h / cos(θ)`이다. 초점은 단일 수치가
아니라 가까운 높은 상품 표면부터 먼 낮은 표면까지의 거리 범위를 만족해야 한다.

### 5.4 높이 결정의 실무 기준

1. 960×960 화면 안에 3차원 검사 체적과 여유 영역이 모두 들어오는 최소 높이를 계산한다.
2. 그 높이에서 가장 먼 위치의 최소 상품이 충분한 원본 픽셀을 갖는지 확인한다.
3. 부족하면 단순히 더 낮추지 말고 센서 해상도·렌즈·검사 영역을 함께 재검토한다.
4. 체커보드 캘리브레이션의 재투영 오차, 중심/네 모서리의 mm/px, 배럴 왜곡을 기록한다.
5. 가장 가까운 높은 상품과 가장 먼 낮은 상품 모두에서 초점·노출·흔들림을 확인한다.

## 6. 권장 실험 설계

### 6.1 후보와 통제 조건

- 각도: `10°`, `15°`, `20°`, `25°`, `30°`; 자원이 부족하면 `15°`, `20°`, `25°`부터 수행
- 각 후보에서 960×960의 검사 체적 점유율과 여유가 같도록 높이 또는 렌즈를 조정
- 카메라, 해상도, JPEG 설정, 모델·Catalog 버전, 배경, 조명, 노출을 고정
- 촬영 순서를 무작위화해 시간대·발열·주변광 변화를 특정 각도와 분리
- 각 각도마다 캘리브레이션을 다시 수행

각도와 조명을 한꺼번에 바꾸면 원인을 분리할 수 없다. 1차는 동일 조명으로 각도만 선별하고, 2차에서
선정 후보 `2개 × 조명 2종`의 작은 요인 실험을 수행한다. NIST의 실험계획법처럼 요인과 수준, 반응값을
미리 정하고 교호작용을 확인하는 방식이 적절하다.

### 6.2 반드시 포함할 장면

- 윗면이 같거나 매우 비슷하고 측면 라벨만 다른 캔 SKU 쌍
- 짧은 캔·긴 캔, 유광 캔, 병, 박스, 파우치 등 형상 그룹
- 상품 회전: 최소 5방향
- 위치: 중앙, 네 변, 네 모서리
- 단품과 실제 운영 밀도의 다품목 장면
- 서로 붙음, 부분 포함 중복, 경계 접촉, 높은 상품이 낮은 상품을 가리는 장면
- 서로 다른 날짜/설치 재조정/조명 세션

같은 물리 배치에서 각도만 바꾼 쌍을 만들면 후보 간 비교력이 높아진다. 데이터 분리는 이미지 단위
무작위가 아니라 물리 상품·배치·촬영 세션 단위로 수행한다. validation에서 각도를 선택하고 test는
최종 확인에 한 번만 사용한다.

### 6.3 평가 지표

사업 위험 순서로 다음을 본다.

| 우선순위 | 지표 | 판단 목적 |
|---:|---|---|
| 1 | 잘못된 SKU의 `APPROVED` 비율 | 가장 위험한 오승인 방지 |
| 2 | 동일 상부 캔 혼동률과 pair별 Top-1/Top-3 | 측면 노출의 실제 이득 확인 |
| 3 | 측면 상품명·용량 문자의 최소 높이(px), OCR exact/character accuracy | 노출된 측면이 실제 식별 가능한 품질인지 확인 |
| 4 | detector recall, 누락·중복·경계 접촉률 | 큰 경사에서 생기는 가림·화면 이탈 확인 |
| 5 | 올바른 `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE`, `IMAGE_RECAPTURE` 비율 | 현재 제품 정책 전체 영향 확인 |
| 6 | 중앙·변·모서리별 성능과 최소 ROI 픽셀 | 위치에 따른 원근 차이 확인 |
| 7 | 과노출 픽셀 비율, 선명도, 재투영 오차, 중심/가장자리 mm/px | 광학 품질과 원인 진단 |
| 8 | p50·p95·p99 처리시간과 표본 수 | 해상도·ROI 변화의 실행 영향 확인 |

각 비율에는 상품/배치 단위 bootstrap 95% 신뢰구간을 함께 제시한다. 전체 평균만 내지 말고 동일 상부
캔, 일반 SKU, 형상, 위치, 밀도별 최악 그룹을 별도로 표시한다.

### 6.4 사전에 합의할 선택 규칙

실험 후 유리한 기준을 만드는 것을 피하기 위해 다음을 촬영 전에 합의한다.

1. 잘못된 `APPROVED`의 허용 상한
2. detector 누락과 화면 경계 접촉의 허용 상한
3. 가장 먼 모서리의 최소 ROI 픽셀과 중앙 대비 배율 차이의 허용 범위
4. 동일 상부 캔 혼동률이 수직 기준보다 얼마나 줄어야 “의미 있는 개선”으로 볼지
5. 조명 반사·초점·처리시간의 허용 범위

모든 필수 조건을 통과한 각도 중, 성능 신뢰구간이 최상 후보와 실질적으로 구분되지 않으면 더 작은
각도를 선택한다. 예를 들어 `20°`가 `25°`와 오차 범위 내에서 같은 판정 성능이고 가장자리 왜곡은 더
작다면 `20°`가 합리적이다.

## 7. 조명과 보정에 대한 병행 권고

캔은 곡면 금속이라 방향성 조명에서 밝은 띠와 hotspot이 생기기 쉽다. 머신비전 조명 자료는 곡면·유광
표면에 확산 돔/터널 조명이 반사를 균일하게 만들 수 있고, 광원과 렌즈의 교차 편광이 glare 감소에
도움이 된다고 설명한다. 다만 편광은 광량을 크게 줄일 수 있으므로 노출시간 증가와 흔들림을 함께
측정한다.

- 1차 권장: 넓은 확산광, 주변광 차광, 수동 노출·화이트밸런스 고정
- 2차 후보: 광원 편광판 + 렌즈 편광판의 교차 편광
- 피해야 할 방식: 반사 문제를 카메라 경사 증가 하나로 해결
- 보정: 각 후보에서 OpenCV 카메라 캘리브레이션을 수행하되, 원본과 렌즈 왜곡 보정본을 모두 평가

## 8. 보고용 문장 예시

> 카메라 각도는 캔 측면 정보량을 늘리지만, 경사가 커질수록 위치별 배율 차이와 상품 간 가림도
> 증가한다. 최신 시점 강건성 연구 역시 모든 물체에 공통인 최적 시점이 없고 특정 시점에서 인식이
> 불안정할 수 있음을 보여 준다. 원통 근사상 긴 캔은 수직 기준 30°에서 측면 투영 높이가 윗면 투영
> 지름과 거의 같아져 현재 관찰된 과도한 형상 변화와 일치한다. 이에 30°를 즉시 확정하지 않고
> 15°·20°·25°를 비교하며, 초기 기구 기준은 약 18°로 제안한다. 최종값은 동일 상부 캔의 혼동 감소,
> 오승인, 검출 누락, 가장자리 성능과 반사를 함께 평가해 필수 조건을 만족하는 가장 작은 각도로
> 결정한다. 높이는 중앙 960×960 유효 화각과 3차원 검사 체적을 기준으로 계산하고 실측 캘리브레이션으로
> 검증한다.

문헌 질문이 나왔을 때는 다음을 덧붙인다.

> 대표 리테일 벤치마크인 RPC도 30°와 45° 영상을 사용하지만, 이는 단품 상품의 여러 외관을 학습하기
> 위한 exemplar 촬영각이다. 실제 다품목 checkout 이미지는 상부 카메라로 수집했으므로 RPC를 근거로
> “계산대 카메라의 최적 설치각은 30°”라고 말할 수는 없다. 문헌이 일관되게 뒷받침하는 것은 특정
> 숫자보다 학습·운영 시점의 일치, 식별 면의 가시성, 공간 해상도와 가림을 함께 검증해야 한다는 점이다.

## 9. 최종 확정 전에 채울 값

| 항목 | 값 |
|---|---|
| 검사 영역 `W × D` | 미정 mm × 미정 mm |
| 최대 상품 높이 `Hmax` | 미정 mm |
| 카메라 모델/센서 | 미정 |
| 렌즈 초점거리 | 미정 mm |
| 1920×1080 실측 HFOV/VFOV | 미정° / 미정° |
| 중앙 960×960 실측 유효 화각 | 미정° |
| 최소 상품 크기 및 최소 ROI 픽셀 | 미정 mm / 미정 px |
| 기구상 가능한 수직 높이 범위 | 미정 mm |
| 현재 30°의 기준면 | 수직 기준 / 테이블 면 기준 확인 필요 |
| 조명 방식과 주변광 차광 | 미정 |

이 값들이 채워져야 높이를 mm 단위로 제안할 수 있다. 값 없이 특정 높이를 제시하면 렌즈 화각과 앱의
중앙 크롭을 무시한 임의값이 된다.

## 10. 근거 자료

- Michalkiewicz et al., **Not all Views are Created Equal: Analyzing Viewpoint Instabilities in Vision
  Foundation Models**, ICCV 2025. 작은 시점 변화와 accidental viewpoint에 따른 특징·분류 불안정성을
  분석한다.
  <https://openaccess.thecvf.com/content/ICCV2025/html/Michalkiewicz_Not_all_Views_are_Created_Equal_Analyzing_Viewpoint_Instabilities_in_ICCV_2025_paper.html>
- Huang et al., **Improving Viewpoint-Independent Object-Centric Representations through Active Viewpoint
  Selection**, NeurIPS 2024. 장면별 정보 이득을 기준으로 다음 시점을 고르는 접근을 제시한다.
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2360da01c2ed6592bb691326424de184-Abstract-Conference.html>
- Ruan et al., **Towards Viewpoint-Invariant Visual Recognition via Adversarial Training**, ICCV 2023.
  3차원 시점 변화가 같은 물체의 예측을 크게 바꿀 수 있음을 다룬다.
  <https://openaccess.thecvf.com/content/ICCV2023/html/Ruan_Towards_Viewpoint-Invariant_Visual_Recognition_via_Adversarial_Training_ICCV_2023_paper.html>
- Jeon et al., **A Retail Object Classification Method Using Multiple Cameras for Vision-Based Unmanned
  Kiosks**, IEEE Sensors Journal, 2022. 리테일 객체의 시점별 특징과 다중 시점 결합을 다룬다.
  <https://doi.org/10.1109/JSEN.2022.3210699>
- Wei et al., **RPC: A Large-Scale Retail Product Checkout Dataset**, 2019/2022. 통제 단품과 실제 다품목
  계산대 장면의 차이 및 세립도 SKU 판정 문제를 제시한다.
  <https://arxiv.org/abs/1901.07249>
- Yang et al., **IncreACO: Incrementally Learned Automatic Check-Out With Photorealistic Exemplar
  Augmentation**, WACV 2021. RPC exemplar의 상부·30°·45°·수평 촬영과 실제 checkout target domain의
  차이를 명시한다.
  <https://openaccess.thecvf.com/content/WACV2021/html/Yang_IncreACO_Incrementally_Learned_Automatic_Check-Out_With_Photorealistic_Exemplar_Augmentation_WACV_2021_paper.html>
- Follmann et al., **MVTec D2S: Densely Segmented Supermarket Dataset**, ECCV 2018. 상부에서 약간
  off-center인 카메라, 10개 회전과 3개 조명 조건으로 리테일 장면의 강건성을 평가한다.
  <https://openaccess.thecvf.com/content_ECCV_2018/html/Patrick_Follmann_D2S_Densely_Segmented_ECCV_2018_paper.html>
- Yao et al., **Training with Product Digital Twins for AutoRetail Checkout**, 2023. 실제 checkout 분포에
  맞춘 viewpoint·illumination 최적화와 camera distance·height의 영향을 분석한다.
  <https://arxiv.org/abs/2308.09708>
- Du et al., **Multi-View Active Fine-Grained Visual Recognition**, ICCV 2023. 세립도 분류에서 식별 단서가
  있는 추가 시점을 능동적으로 찾는 문제를 다룬다.
  <https://openaccess.thecvf.com/content/ICCV2023/html/Du_Multi-View_Active_Fine-Grained_Visual_Recognition_ICCV_2023_paper.html>
- Hou et al., **Learning to Select Views for Efficient Multi-View Understanding**, CVPR 2024. 전체 시점 중
  유용한 2~3개만 선택하는 분류·검출 실험을 제시한다.
  <https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Learning_to_Select_Views_for_Efficient_Multi-View_Understanding_CVPR_2024_paper.html>
- Staderini et al., **Spatial Resolution Metric for Optimal Viewpoints Generation in Visual Inspection
  Planning**, ICVS 2023. ray tracing으로 가시 표면 coverage와 요구 공간 해상도를 함께 제약한다.
  <https://doi.org/10.1007/978-3-031-44137-0_17>
- Collins et al., **ABO: Dataset and Benchmarks for Real-World 3D Object Understanding**, CVPR 2022. 실제
  상품을 다양한 방위각·고도각·조명에서 평가하는 multi-view cross-domain retrieval benchmark다.
  <https://openaccess.thecvf.com/content/CVPR2022/html/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.html>
- Zhang et al., **FGPR: A Large-Scale Dataset and Benchmark for Fine-Grained Product Retrieval**, Pattern
  Recognition 172, 2026. 85,733개 SKU에 OCR text와 SKU 정보를 결합해 세립도 retrieval에서 문자의
  중요성을 평가한다.
  <https://doi.org/10.1016/j.patcog.2025.112523>
- Tur et al., **Exploring Fine-grained Retail Product Discrimination with Zero-shot Object Classification
  Using Vision-Language Models**, 2024. MIMEX에서 범용 VLM과 CLIP·DINOv2 기반 방법을 비교한다.
  <https://arxiv.org/abs/2409.14963>
- NVIDIA, **Retail Object Detection Model Card**. 실제·합성 fine-tuning 데이터의 카메라 높이·화각·시점
  범위를 공개한 산업 자료다. 비교 논문이나 설치 최적화 결과로 해석하지 않는다.
  <https://docs.api.nvidia.com/nim/re/reference/nvidia-retail-object-detection>
- Edmund Optics, **Distortion**. 렌즈 왜곡과 parallax를 구분하고 넓은 FOV에서 왜곡이 커지는 경향을
  설명한다.
  <https://www.edmundoptics.com/knowledge-center/application-notes/imaging/distortion/>
- Edmund Optics, **Imaging System Parameter Calculator**. 센서, working distance, FOV, 초점거리와
  물체 공간 해상도의 관계를 제공한다.
  <https://www.edmundoptics.com/knowledge-center/tech-tools/imaging-system-parameter-calculator/>
- OpenCV, **Camera Calibration**. 핀홀 투영, 내부·외부 파라미터와 카메라 캘리브레이션의 기술 근거다.
  <https://docs.opencv.org/5.0/main_modules/calib.html>
- Advanced Illumination, **A Practical Guide to Machine Vision Lighting**. 곡면·반사체에 대한 확산광,
  편광과 조명 기하를 설명한다.
  <https://advancedillumination.com/a-practical-guide-to-machine-vision-lighting/>
- NIST/SEMATECH, **Engineering Statistics Handbook: Choosing an Experimental Design**. 요인·수준과
  실험계획, response surface 설계의 근거다.
  <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm>

## 11. 해석의 한계

- 위 `18°`는 제작과 실험을 시작하기 위한 중심점이지 성능 인증값이 아니다.
- 공개 연구는 대상 카메라·렌즈·SKU·조명과 동일하지 않으므로 논문 결과를 설치 각도로 직접 변환할 수
  없다.
- 캔 원통 식은 후보 구간을 설명하기 위한 orthographic 근사이며 최종 기구 계산을 대체하지 않는다.
- 현재 제품 `0.1.15`가 새 각도에서 독립 일반화 성능이나 SLA를 달성했다고 표현해서는 안 된다. 선택된
  설치조건으로 별도 촬영·평가해야 한다.
