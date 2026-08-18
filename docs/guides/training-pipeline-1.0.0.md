# Detector·Classifier 학습 파이프라인 1.0.0

Detector와 Classifier 학습 파이프라인은 통합 버전을 만들지 않고 독립적으로 관리한다.

| 구성요소 | 파이프라인 버전 | 계약 SHA-256 | provenance |
|---|---|---|---|
| Detector | `1.0.0` | `2ea28345e03b44143cf09f6b5e52c01937345a79fed5b27a9dd092f4a9ff4c48` | `recovered` |
| Classifier | `1.0.0` | `4c1f11b687d1c14a51cac6a7b72c5767f4c6063be805a904458152d6e12f9592` | `recovered` |

계약은 각각 `configs/training/bread_detector_pipeline_1.0.0.json`과
`configs/training/bread_classifier_pipeline_1.0.0.json`에 있다. 파이프라인 버전은
Worker, 모델 package, 데이터셋, Flutter 앱 및 Python 배포 버전과 별개다.

`recovered`는 기존 불변 checkpoint와 주변 산출물에서 증거를 복구했다는 의미다. 원본
native run record로 위장하지 않는다. 새 native run은 모든 단계가 이어진 hash-chain ledger를
남겨야 한다.

## 데이터 계약

두 구성요소의 실제 학습 데이터는 서로 다르며 package schema 2.1 source provenance에 각각
기록한다.

- Detector: `manifests/bread-1.0.0`, `single_objects` 20종×10장, 총 200장
- Classifier: `manifests/bread-1.0-a52b4faa3e20`, `single_objects_3` 20종×12장, 총 240장
- release composition 데이터셋: `bread-1.0-a52b4faa3e20`

`multi_object_scenes`는 학습에 사용할 수 없으며 development selection 및 운영 합격
benchmark로만 사용한다. 독립 최종 test는 사용자 이미지로 진행하며 아직 pending이다.

## 잠긴 단계 순서

```text
dataset audit
→ group split
→ train
→ validation selection
→ checkpoint lock
→ ONNX export
→ CPU/CUDA parity
```

이후 package 조합, `multi_object_scenes` benchmark, RTX 5080 성능 측정과 promotion은
학습 버전이 아니라 Worker/model 승격 절차가 담당한다. test 결과로 threshold를 다시 맞추지
않는다.

## Detector 1.0.0

- D-FINE-N HGNetv2 B0, 입력 `640×640`
- source revision `267a6da6d04c8ad52e54120692896515b9e55981`
- `single_objects` 200장에서 합성 2,000장, empty 698장, annotation 5,235개
- synthetic seed `20260815`, validation fold `2`
- train/validation 1,334/666장, annotation 3,499/1,736개
- 30 epochs, batch 8, backbone LR `4e-4`, head LR 2배, weight decay `1e-4`
- 선택 checkpoint epoch 27, operating threshold `0.49`
- checkpoint SHA-256 `7a6e5d531bb464a985958468e9b36466225eca5b2258230f60c8381de8ac0555`
- ONNX SHA-256 `f0d2eaf8e67821627957c3eed1462812063c32c4ad17028dda869addc5371b09`

threshold `0.49`는 기존 `multi_object_scenes` development 기록에서 복구됐다. 따라서
selection과 benchmark가 겹친다는 제한을 계약에 `selection_benchmark_overlap=true`로
명시한다. 사용자 독립 test 전에는 독립 일반화 성능으로 해석하지 않는다.

`single_objects_3`로 새로 학습한 12-shot Detector 후보는 parity를 통과했지만
`multi_object_scenes`에서 인식률 94.7518%, 오인율 3.4735%, segmentation
recall/precision 97.8014%/81.0700%, P95 115.08ms로 실패했다. 이 후보는
`configs/archive/bread/bread_1_0_0/detector_12shot_rejected.json`에 거절 증거로 보존하며
운영 모델이나 threshold에 반영하지 않는다.

## Classifier 1.0.0

- DINOv3 ConvNeXt-Tiny, 입력 `224×224`
- source revision `6876159a11b4df116f30f667f8c9888617df0751`
- `single_objects_3` 240장
- `multi_object_scenes` development ROI의 `group_id` 기반 3-fold, leakage 없음
- seed `20260812`, `20260813`, `20260814` 비교
- 마지막 stage L2-SP challenger의 seed `20260813` 선택
- staged affine-view TTA export와 `top3_vote` ranking
- checkpoint SHA-256 `1be1781c2ece6e8ac12aa9ac9915fd5f81f115d093f1e08f7c9c3b9622e3b104`
- ONNX SHA-256 `93a9d92c6fd63f5a6aef65e11e3d0acecfffd7c6cf5ac2bfdba732f4e543ab8f`

## 검증 명령

```powershell
bixolon train verify-pipeline --component detector --contract configs/training/bread_detector_pipeline_1.0.0.json --repository-root . --dry-run
bixolon train verify-pipeline --component classifier --contract configs/training/bread_classifier_pipeline_1.0.0.json --repository-root . --dry-run

bixolon train verify-pipeline --component detector --contract configs/training/bread_detector_pipeline_1.0.0.json --repository-root . --smoke
bixolon train verify-pipeline --component classifier --contract configs/training/bread_classifier_pipeline_1.0.0.json --repository-root . --smoke
```

`--dry-run` 결과는 `passed`가 아니라 `verification_scope=contract_and_data_only`와
`passed=null`을 기록한다. `--framework-smoke`는 toy framework 경로이고, `--smoke`는 실제
pinned source, checkpoint, forward/backward, 임시 ONNX export와 production ONNX 실행을
검사한다. Detector raw query 순서는 export에서 바뀔 수 있으므로 실제 300장 final detection
CPU/CUDA parity report를 권위 증거로 사용한다.

현재 schema 1.1 production package는 변경하지 않고 두 계약의 외부 lock으로 검증한다.
향후 schema 2.1 package의 각 model source에는 다음 네 필드가 필수다.

- `training_pipeline_version`
- `training_contract_sha256`
- `training_dataset_version`
- `training_manifest_sha256`

## 버전 상승 기준

- `MAJOR`: 계약 schema, 입력 의미 또는 공식 단계 순서의 비호환 변경
- `MINOR`: architecture, recipe, augmentation, split·selection 또는 export 의미 변경
- `PATCH`: 모델 선택과 출력 의미를 바꾸지 않는 검증·기록·오류 수정

한 구성요소의 변경은 다른 구성요소, Worker, 데이터셋 또는 앱 버전을 자동으로 올리지 않는다.
