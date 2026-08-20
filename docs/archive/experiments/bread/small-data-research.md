# 빵 DINO 소량 학습 개선 근거

## 적용 결론

이번 실험은 운영 detector, ROI 전처리, Worker 상태 정책을 바꾸지 않고 classifier 학습만 다음과 같이 교체한다.

1. 동일한 DINOv3 ConvNeXt Tiny checkpoint의 backbone을 고정한다.
2. 마지막 spatial patch feature(`x_prenorm`)를 cache한다.
3. patch channel별 최소·최대 범위에서 brightness `c²FroFA`를 적용한다.
4. 원본 1개와 seed `20260810`으로 생성한 feature view 16개를 LayerNorm과 L2 정규화한다.
5. L2 규제 max-margin linear SVM head(`C=1`)만 학습한다.
6. ONNX 모델 안에도 L2 정규화를 포함해 PyTorch/ONNX 및 CPU/CUDA 입력 계약을 일치시킨다.

`N=5·10·15·20` 선택 집합, detector, development fold, 승인 위험 상한 정책은 기존 실험과 동일하다. development/test ROI를 classifier 학습에 넣지 않으며 최종 test 94장은 열지 않는다.

## 논문과 공식 기술 자료

- [DINOv3 논문](https://arxiv.org/abs/2508.10104)과 [Meta AI 공식 기술 블로그](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)는 고정된 backbone에서도 다양한 downstream task로 전이되는 범용 feature와 경량 adapter 사용을 강조한다. 소량의 빵 사진으로 backbone 전체를 계속 움직이는 대신 pretrained 표현을 보존하는 근거로 사용했다.
- [Frozen Feature Augmentation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.html)와 [공식 프로젝트 페이지](https://frozen-feature-augmentation.github.io/)는 frozen spatial feature에 적용한 brightness `c²FroFA`가 5~25-shot에서 linear probe와 weight-decayed lightweight baseline을 일관되게 보완한다고 보고한다. 논문의 고정값 `v=1.0`을 사용했고, 입력 이미지 증강을 더 강하게 만드는 방식은 채택하지 않았다.
- [Rethinking Few-Shot Image Classification](https://arxiv.org/abs/2003.11539)은 강한 pretrained/self-supervised embedding과 단순한 base learner가 복잡한 meta-learning보다 강할 수 있고, 정규화된 규제 선형 분류기가 유효함을 보인다. 따라서 20종을 모두 포함한 max-margin linear head를 사용했다. Logistic head도 먼저 검증했지만 CUDA 경계 표본에서 class rank가 뒤집혀 채택하지 않았다.
- [SimpleShot](https://arxiv.org/abs/1911.04623)은 mean subtraction과 L2 정규화 같은 단순한 feature 변환이 few-shot nearest-centroid 성능에 큰 영향을 준다는 것을 보인다. 현재 DINOv3 ConvNeXt 출력은 LayerNorm만 적용되고 L2 norm은 일정하지 않으므로, L2 정규화를 학습과 ONNX 추론 양쪽에 명시했다.
- [AAAI 2025 fine-grained few-shot 연구](https://ojs.aaai.org/index.php/AAAI/article/view/32645)는 local alignment와 query 관계 모델링의 이점을 보여준다. 다만 query-dependent classifier는 현재 단일 이미지 Worker의 고정 ONNX 계약과 latency를 바꾸고 별도 episodic 학습 기반도 필요하므로 이번 비교 범위에서는 제외했다.

## 현재 데이터에서의 사전 검증

동일한 selection lock과 development ROI 889개를 사용한 classifier-only 사전 검증 결과는 다음과 같다. 이 값은 방법을 구현하기 전 방향 확인용이며, 정식 결과는 실험 디렉터리의 3-fold 보정·Worker 평가 보고서를 기준으로 한다.

| 조건 | 기존 partial fine-tune Top-1 / Top-3 | 개선 recipe 사전 Top-1 / Top-3 |
|---|---:|---:|
| N=5 | 75.82% / 90.66% | 79.42% / 92.69% |
| N=10 | 80.99% / 92.69% | 82.00% / 94.26% |
| N=15 | 81.89% / 94.60% | 82.68% / 95.50% |
| N=20 | 84.03% / 95.16% | 86.05% / 95.95% |

이 확인에서는 `N`이나 승인 threshold를 자동 선택하지 않았다. linear SVM `C=1`은 네 조건에 공통으로 고정했고, FroFA brightness magnitude `1.0`은 논문의 공개 설정을 그대로 사용했다. Logistic 후보는 development 정확도가 높았지만 N=15·20에서 CPU/CUDA Top-1 parity가 각 1건 깨져 제외했다.

정식 ONNX parity에서 PyTorch↔CPU ONNX는 네 조건 모두 통과했고 CPU/CUDA Worker 상태도 모두 같았다. CUDA 절대 logit 오차는 최대 `0.00975`로 설정 허용치 `0.02` 안이다. 다만 엄격한 후보 순위 비교에서는 N=5/10/15에 각각 일부 Top-3 순서 차이가 있었고 N=20은 889 ROI 중 1건의 Top-1 경계 표본이 달라 strict parity는 실패다. threshold나 허용치를 결과에 맞춰 변경하지 않았으며 production 승격 차단 사유로 보존한다.

## 해석상 제한

- seed 한 번의 결과이므로 seed 간 분산을 추정하지 않는다.
- `bread_project_2` 300장 중 299장은 기존 정책 적합 세트와 hash가 같으므로 독립 test가 아니라 운영 흐름 회귀 평가다.
- 작은 데이터에서 feature-space augmentation은 분산을 보완하지만 새로운 촬영 환경, 포장, 배경, 심한 가림에 대한 실제 정보량을 만들지는 못한다. 다음 데이터 수집은 오류 군집별 새 촬영 세션을 우선해야 한다.
- production package 승격, 앱 기본 package 변경, 최종 test 접근은 이 실험에 포함하지 않는다.
