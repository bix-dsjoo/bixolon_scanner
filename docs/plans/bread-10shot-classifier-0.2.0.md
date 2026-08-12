# 빵 종류당 10장 classifier `0.2.0`

## 고정 범위

- classifier 가중치 학습 입력은 `bread_project_3`의 클래스별 정확히 10장뿐이다.
- 기존 classifier 가중치, logit, 증류, 다른 빵 이미지와 운영 배경은 사용하지 않는다.
- 범용 DINOv3 ConvNeXt Tiny 사전학습 가중치만 초기 backbone으로 허용한다.
- detector ONNX와 detector/input/quality metadata, pipeline, API schema 및 `DETECTOR_UNCERTAIN_OBJECT` hard gate는 `0.1.1`에서 변경하지 않는다.
- development/test ROI와 `bread_project_2`는 가중치 학습에 넣지 않는다.

## 데이터 계약

label metadata가 클래스 수와 순서를 결정한다. 각 클래스 디렉터리는 `normal/flipped × vertical/ground30_dir1..4` 10개 슬롯을 정확히 한 번씩 포함해야 한다.

누락·중복 슬롯, label 불일치, 손상 JPEG, source SHA-256 중복 및 클래스별 10장 위반은 즉시 실패한다. 짧은 변 224px 미만 이미지는 audit에 경고와 원본 SHA를 남기되 초해상도를 적용하지 않는다.

현재 audit 결과는 20개 클래스, 200장, 짧은 변 224px 미만 79장, exact SHA 중복 0장이다. dataset version은 `bread-10shot-9df0df1d32c5`다.

## 학습 후보

1. 기준선: 정확한 200장의 DINOv3 `x_prenorm`으로 c²FroFA 16 views를 만들고 linear SVM을 학습한다.
2. 주 후보: frozen DINOv3 ConvNeXt Tiny + `768→192→768` residual adapter + normalized cosine head를 학습한다. CosFace margin `0.2`, scale `30`, cross-entropy와 supervised contrastive loss를 사용한다.
3. 주 후보의 development 3-fold 평균 Top-1이 95% 미만일 때만 마지막 ConvNeXt stage와 norm을 낮은 LR로 한 번 더 학습한다. L2-SP로 사전학습 가중치 이탈을 제한하며 전체 backbone fine-tune은 금지한다.

모든 후보는 seed `20260812`, `20260813`, `20260814`로 실행한다. 동률은 평균 Top-1, 최저 fold Top-1, 낮은 seed 순으로 결정한다.

학습 증강은 정답 foreground에서 classifier ROI를 직접 만든다. 회전·원근·크기·위치·색상·JPEG·약한 blur와 procedural neutral background만 사용한다. 합성 full-frame을 detector에 통과시키지 않는다. provenance의 모든 sample은 200개 source SHA 중 하나를 가리켜야 한다.

support prototype은 cosine class weight 초기화에만 사용한다. 배포 head와 ONNX에는 support tensor, Top-K, cache fusion이 존재하지 않는다.

## 선택, lock과 평가

development capture-session 3-fold에서 global temperature와 단일 approval threshold를 선택한다. false approval rate의 95% 상한이 `0.5%` 이하인 값 중 coverage가 가장 높은 threshold를 사용하며 승인 0건은 실패다.

config, manifest, manifest metadata, checkpoint와 calibration SHA-256을 `pretest-lock.json`에 기록한 후에만 test에 접근하고 package를 export할 수 있다. test 결과로 재학습하거나 threshold를 바꾸지 않는다.

| 판정 | 조건 |
|---|---|
| `production` | Top-1 ≥97%, Top-3 ≥95%, APPROVED precision ≥99.5%, 운영 대비 coverage 하락 ≤5%p, UNKNOWN Top-3 ≥95% 및 공통 gate 통과 |
| `manual_waiver` | Top-1 ≥95%, Top-3 ≥95%, APPROVED precision ≥99.5%, coverage ≥85% 및 공통 gate 통과 |
| `experiment_only` | 수동 예외 하한 또는 공통 gate 미달 |

공통 gate는 승인 1건 이상, false approval 95% 상한 ≤0.5%, 전체 재촬영 recall ≥99%, PyTorch/CPU ONNX/CUDA ONNX의 상태·Top-1·Top-3 순서 일치, RTX 5080 full-path p95 ≤100ms, detector 동결이다. 기존 test 94장과 `bread_project_2` 300장도 각각 수동 예외 하한을 넘어야 한다. 두 세트는 독립 평가셋이 아니라는 제한을 promotion metadata에 보존한다.

`manual_waiver` 결과는 기본적으로 development package로 export된다. 사람이 명시적으로 승인할 때만 기존 schema의 `manual_waiver` promotion record와 함께 production package가 된다.

## 패키징과 rollback

`bread-worker-0.2.0`은 `0.1.1` detector 파일을 byte-for-byte 복사하고 classifier만 교체한다. export 후 detector SHA-256을 다시 계산하며 detector/input/quality metadata의 구조적 동일성을 검사한다. classifier ONNX batch 축은 동적이다.

운영 전환은 candidate와 `0.1.1` fallback을 모두 로드하고 smoke test한 뒤 package pointer를 원자적으로 교체한다. rollback도 fallback smoke test 후 같은 방식으로 pointer를 되돌린다.

## 구현 상태

- strict manifest/audit 및 임의 label map: 구현·검증 완료
- direct ROI augmentation과 source provenance: 구현·단위 검증 완료
- c²FroFA/SVM, adapter/cosine head, 3-seed 선택, 조건부 last-stage L2-SP: 구현 완료
- calibration, pre-test hash lock, 3단계 승격 판정: 구현·단위 검증 완료
- 동적 batch ONNX export, detector byte-copy, strict parity 판정, 원자적 전환/rollback helper: 구현·단위 검증 완료
- 전체 PyTorch 학습, CPU/CUDA ONNX parity, test 두 세트, RTX 5080 benchmark: 실행 전이며 실제 산출물 없이는 승격할 수 없음
