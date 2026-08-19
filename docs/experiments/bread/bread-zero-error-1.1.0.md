# Bread zero-error 목표 실험 1.1.0

## 상태와 목표

수명주기는 `active`다. 별도 잠금 test set 없이 `bread_dataset`의 선택된 입력을
중복 방지 3-fold OOF 학습·검증에 사용한다. 공식 승격 평가 범위는
`multi_object_scenes` EASY/MEDIUM/HARD 300장이다. 2026-08-18 프로젝트 소유자가 아래 비율을
Worker·Detector·Classifier `1.1.0`의 최종 개선 목표와 운영 기준으로 지정했다. 99%는 장기
개선 목표이며 운영 승격은 아래 여섯 gate로 판정한다. 운영 gate 하나라도 실패하면
`promoted`로 바꾸지 않는다.

| 지표 | 1.1.0 기준 | 분모 |
| --- | ---: | --- |
| end-to-end `APPROVED` 장기 개선 목표 | ≥ 99% | 판정 가능 이미지의 GT 객체 전체 |
| `SEGMENTATION` | ≥ 90% | E/M/H 유효 요청 이미지 300장 전체 |
| end-to-end `APPROVED` 운영 기준 | ≥ 90% | 판정 가능 이미지의 GT 객체 전체 |
| `SEGMENTATION` 이미지 FN | ≤ 0.1% | 최종 `SEGMENTATION` 이미지 중 IoU@0.5 FN이 하나 이상인 이미지 수 / 최종 `SEGMENTATION` 이미지 수 |
| `SEGMENTATION` 이미지 FP | ≤ 0.1% | 최종 `SEGMENTATION` 이미지 중 IoU@0.5 FP가 하나 이상인 이미지 수 / 최종 `SEGMENTATION` 이미지 수 |
| `APPROVED` 객체 오인율 | ≤ 0.1% | 판정 가능 이미지의 GT 객체 전체 |
| `UNKNOWN` Top-3 Candidate out | ≤ 0.1% | 판정 가능 이미지의 GT 객체 전체 |

판정 가능 이미지는 정답 상태가 객체 판정 대상인 이미지다. 그 이미지가 예측에서
`IMAGE_RECAPTURE`가 되거나 detector가 GT를 놓쳐 classifier에 도달하지 못해도 해당 GT는
분모에 남고 미승인으로 계산한다. 정답상 이미지 전체 재촬영 대상은 객체 분모에서 제외하고
recapture recall로 별도 평가한다. 99% 목표 달성과 여섯 운영 gate 통과는 분리해 보고한다.

`UNKNOWN + Top-3` 비율과 `SEGMENT_RECAPTURE` 비율은 상태 분포 및 실패 원인 분석용으로 계속
집계하지만 1.1 승격 합격 여부에는 사용하지 않는다. Detector FP가 만든 unmatched segmentation은
객체 상태 분모에 넣지 않고 별도의 이미지 FP gate가 잡는다. Detector 목표는 파이프라인이
`SEGMENTATION`으로 수용한 이미지에서 측정하며, gate 적용 전 raw FP/FN도 별도 진단으로
반드시 공개한다.

### 새 기준에 따른 현재 재판정

아래 개발 OOF와 실제 Worker 후보를 같은 1.1.0 공식 gate로 다시 계산했다. 개발 OOF는
group-aware 검증 결과이고, Worker 후보는 Detector `1.0.0`을 포함한 300장 실행 결과이므로
둘 중 어느 것도 독립 잠금 test 통과나 deployable Detector `1.1.0`을 뜻하지 않는다.

| 지표 | 개발 OOF | Worker 후보 | 목표 |
| --- | ---: | ---: | ---: |
| `SEGMENTATION` | 296/300 (98.667%, 통과) | 300/300 (100%, 통과) | ≥ 90% |
| end-to-end `APPROVED` | 1,270/1,410 (90.071%, 운영 통과·목표 미달) | 1,168/1,410 (82.837%, 운영 실패) | 운영 ≥ 90%, 목표 ≥ 99% |
| `SEGMENTATION` 이미지 FN | 0/296 (0%, 통과) | 9/300 (3.000%, 실패) | ≤ 0.1% |
| `SEGMENTATION` 이미지 FP | 0/296 (0%, 통과) | 4/300 (1.333%, 실패) | ≤ 0.1% |
| `APPROVED` 객체 오인율 | 0/1,410 (0%, 통과) | 0/1,410 (0%, 통과) | ≤ 0.1% |
| `UNKNOWN` Top-3 Candidate out | 1/1,410 (0.0709%, 통과) | 1/1,410 (0.0709%, 통과) | ≤ 0.1% |

개발 OOF는 여섯 운영 gate를 통과하지만 end-to-end `APPROVED` 99% 목표에는 미달한다.
실제 Worker 후보는 아직 과거 단일 Classifier와 Detector `1.0.0`을 포함하므로 `APPROVED` 하한과
목표, `SEGMENTATION` 이미지 FN/FP gate를
실패한다. 각 Candidate out 1건과 개발 승인 오인 1건은 별도 예외가 아니라 공식 ≤0.1% gate
자체를 통과한 것으로 판정한다. 현재 수명주기는 `active`이고 `promotion_ready=false`다.
새 정책을 versioned ONNX package와 Worker에 통합하고 잠긴 test, parity와 성능을 통과하기
전에는 `1.1.0`으로 승격하지 않는다.

이는 공식 범위 E/M/H 300장에 대한 유한 표본 결과다. 독립 test가 없으므로 모집단 오류율이나 통계적
상한을 증명한다고 해석하지 않는다.

## 데이터 계약과 분할

Classifier 지원 원본은 과거 같은 조건의 비교에서 `single_objects_2`보다 높은 Top-1을 보인
`single_objects` 200장만 선택했다. `single_objects_1`, `single_objects_2`,
`single_objects_3`는 혼용하지 않는다. 공식 1.1 평가와 승격 판정에는
`multi_object_scenes` E/M/H 300장만 사용한다.

canonical registry는 `manifests/bread-zero-error-1.1`이며 데이터셋 버전은
`bread-1.1-171f25171307`이다. 64-bit dHash Hamming distance 2 이하를 근접 중복으로 묶은 뒤
그룹 단위로 fold를 배정했다. Detector fold는 100/101/99장이고 근접 중복 그룹은 없다.
각 outer fold에서는 해당 fold를 학습에 사용하지 않는다. 모든 300장은
정확히 한 번 OOF 검증 대상이 되며, 나머지 두 fold의 학습 대상이 된다.

단일 객체 원본은 클래스마다 동일 물리 대상을 여러 각도에서 촬영한 자료라 물리 대상 수준의
완전 독립 분할은 불가능하다. 파일 SHA와 근접 중복 누수는 차단하지만 이 제한은 최종 보고서에
계속 남긴다.

## 모델 선택

Detector 1차 경로는 현재 Windows ONNX Runtime CPU/CUDA parity와 지연이 검증된
[D-FINE-N](https://github.com/Peterande/D-FINE)을 실제 이미지에 미세조정하는 것이다.
초기에는 class-agnostic `bread_object` 한 클래스를 시도하고, 전이 안정성 비교를 위해 기존
20-class head를 유지하는 후보도 평가한다. D-FINE은 box 분포 정제와 localization
self-distillation을 사용한다([논문](https://arxiv.org/abs/2410.13842)).

최신 challenger로 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)를 조사했다.
DINOv3 특징과 Spatial Tuning Adapter를 쓰는 최신 실시간 DETR 계열이다
([논문](https://arxiv.org/abs/2509.20787)). 다만 현재 저장소의 Windows ONNX export·parity
경로가 없으므로 D-FINE OOF가 실패할 때만 다음 proposal로 활성화한다.

Classifier는 기존 [DINOv3](https://github.com/facebookresearch/dinov3) ConvNeXt-Tiny
사전학습 모델과 `single_objects` 전용 10-shot 후보를 사용한다. 승인과 Top-3 안전성은
grouped OOF를 기본 평가 단위로 사용한다. nested OOF를 함께 진단하고, 별도 test를 잠그지
않는다는 사용자 조건에 따라 최종 개발 정책은 전체 OOF safety score에서 pooled threshold를
고른다. pooled 결과는 같은 행을 정책 선택에도 사용하므로 독립 일반화 결과가 아니다.
Top-3 set이 안전하지 않은 경우를 `SEGMENT_RECAPTURE`로 보내는 선택 정책은 selective
conformal risk control의 아이디어를 참고하되, 현재 표본에서는 보장식이 아니라 명시적인
유한 count gate로 평가한다([관련 논문](https://arxiv.org/abs/2512.12844)).

## 지표와 선택 순서

Detector는 annotated 이미지만 raw FP/FN 분모에 포함하고, 기대 recapture 이미지는 별도
recapture recall로 집계한다. raw FP/FN을 `IMAGE_RECAPTURE`로 숨기지 않고 raw와 수용 경로
지표를 함께 기록한다. 신뢰할 수 없는 이미지가 낮은 비율의 `IMAGE_RECAPTURE`로 격리된 뒤
수용 경로에서 FP/FN 0을 요구한다.

Classifier에 도달한 matched GT는 `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE` 중 하나가 된다.
파이프라인 선택은 모든 판정 가능 GT를 분모로 end-to-end `APPROVED` 90% 하한과 99% 목표,
승인 오인과 Candidate out 각각 ≤0.1%를 함께 평가한다. `UNKNOWN` 및 `SEGMENT_RECAPTURE` 비율은 진단용으로 함께
보고한다. 이미지 수준에서는 `SEGMENTATION` ≥90%와 수용 이미지 FP/FN 각각 ≤0.1%를 동시에
요구하므로, detector 오류를 `IMAGE_RECAPTURE`로 대량 우회할 수 없다.

OOF 정책을 잠근 뒤에만 모든 허용 학습 행으로 final model을 학습한다. final model 결과는
OOF 검증을 대체하지 않으며 독립 test 통과로 표시하지 않는다.

## 반복 순서

1. 1.0.0 raw query와 classifier view logits에서 실패 유형을 고정한다.
2. D-FINE-N 3-fold OOF를 학습하고 FP/FN·exact-image·recapture를 집계한다.
3. 실패가 남으면 localization, duplicate, background, missed-small/border로 분해한다.
4. augmentation, class-agnostic head, 해상도 또는 DEIMv2 challenger를 한 번에 하나씩 바꾼다.
5. Classifier nested 진단과 pooled OOF를 모두 기록하고, 승인 0오류와 Top-3 0누락을 만족하는
   최소 recapture 개발 정책을 고른다.
6. PyTorch/CPU ONNX/CUDA ONNX parity, Worker E/M/H 300장, RTX 5080 p95를 검증한다.

### 반복 1 메모

fold 0 최초 실행은 COCO 기본 500-iteration warm-up을 그대로 상속해 267장 규모에서 epoch 0
학습률이 `3.6e-7`에 머물렀고 validation AP50이 0.36%였다. 이 실행은 중단하고 실패
증거로 보존했다. 실제 batch 수에 맞춰 warm-up을 50 iteration, EMA warm-up을 100,
총 epoch를 20으로 줄인 뒤 동일 fold를 처음부터 다시 실행한다.

class-agnostic head는 기존 20-class score head와 shape가 달라 random initialization이 되었고,
epoch 1 AP50이 1.97%에 머물렀다. 이 경로는 중단했다. 두 번째 개선은 기존 bread detector의
20-class head와 localization 가중치를 모두 보존하고, 실제 이미지에 backbone `1e-6`, 나머지
`1e-5`, warm-up 10 iteration, 10 epoch로 보수적으로 미세조정하는 것이다. Detector의 class
출력은 최종 품목 확정에 사용하지 않고 기존처럼 object score에만 사용한다.

## 2026-08-18 bread_project_2 난이도별 승격 검토

`C:\workspace\raw_data\bread_project_2`의 E/M/H 각 100장을 난이도별로 재집계했다.
300장 모두 기존 `multi_object_scenes`와 SHA가 일치하므로 독립 test 결과가 아니라 grouped
OOF 개발 결과다.

| 지표 | E | M | H |
| --- | ---: | ---: | ---: |
| 이미지 | 100 | 100 | 100 |
| `SEGMENTATION` | 100 (100%) | 97 (97%) | 99 (99%) |
| `IMAGE_RECAPTURE` | 0 (0%) | 3 (3%) | 1 (1%) |
| 수용 GT | 410 | 480 | 493 |
| raw Detector FP/FN | 0/0 | 0/2 | 0/1 |
| 수용 Detector FP/FN | 0/0 | 0/0 | 0/0 |
| `APPROVED` | 365 (89.02%) | 389 (81.04%) | 373 (75.66%) |
| `UNKNOWN + Top-3` | 31 (7.56%) | 65 (13.54%) | 54 (10.95%) |
| `SEGMENT_RECAPTURE` | 14 (3.41%) | 26 (5.42%) | 66 (13.39%) |
| `APPROVED` 오인 | 0 | 0 | 0 |
| Candidate out | 0 | 0 | 0 |

H의 `SEGMENT_RECAPTURE`는 13.39%, 전체 multi-object 기준은 106/1,383, 7.66%다. 이 비율은
현재 1.1 승격 gate가 아니며 난이도별 실패 원인 분석용으로만 기록한다.

1.1.0 실행 패키지가 없으므로 난이도별 실제 속도는 아직 측정할 수 없다. 같은 300장을
production Worker 1.0.0 CUDA로 직접 실행한 비교 기준은 다음과 같다. 이 값은 1.1.0 속도로
인용하지 않는다.

| CUDA baseline | E | M | H |
| --- | ---: | ---: | ---: |
| 평균 | 65.38ms | 73.52ms | 73.30ms |
| P95 | 80.07ms | 87.89ms | 88.23ms |

승격 요청은 기록했지만 수명주기는 `active`로 유지한다. `bread-worker-1.1.0` 패키지 생성,
runtime 정책의 Worker 통합, 최종 모델 직렬화, CPU/CUDA 상태 parity,
1.1.0 E/M/H latency가 끝나지 않아 `promotion_ready=false`다. 세부 기계 판독 보고서는
`artifacts/evaluations/bread_project_2/bread-zero-error-1.1.0-oof-difficulty.json`이다.

## 2026-08-18 실행 패키지 v16 개발 통과

### 연구 조사와 채택 범위

최근 공개 후보로 [SigLIP 2](https://arxiv.org/abs/2502.14786),
[DINOv3 공식 구현](https://github.com/facebookresearch/dinov3),
[DEIMv2](https://arxiv.org/abs/2509.20787)를 검토했다. 기존 RF-DETR Large의 held-out
proposal coverage가 부족했고, 실제 실행 실패는 더 큰 backbone의 표현력보다 JPEG draft에
따른 proposal 선택·프레임 박스·도메인별 classifier margin에서 발생했다. 따라서 원인과 직접
연결되지 않은 신규 대형 모델 재학습은 채택하지 않았다. 이미 사용 권한과 provenance가 고정된
DINOv3 ConvNeXt-Tiny feature에 same-domain grouped OOF LDA를 적용하고, 네 D-FINE 모델의
proposal consensus와 필요한 경우에만 원본 해상도를 재검사하는 정책을 채택했다.

Classifier는 전체 로그 크기에 의존하는 raw margin 대신 L2-normalized logit margin을 사용한다.
개발 OOF와 v2에서 반려된 운영 수집본을 합친 개발 진단은 1,914개 중 1,906개 승인,
오인 1개, `UNKNOWN` 8개, Candidate out 1개였다. class 4/15/16/18에만 각각
`0.0283131184`, `0.0376056238`, `0.0646190058`, `0.0833921244`의 승인 margin을 두고,
나머지는 0 경계를 유지했다. 최종 ONNX SHA-256은
`8e1fc0f2b3241c771ef2116cb921c346ae80ebb83f1d5467d03ac27b86615a4c`다.

Detector는 1,000px JPEG draft에서 네 모델을 순차 CUDA graph로 실행한다. 프레임의 30%를
초과하는 box는 consensus와 ambiguity 계산 전에 제거한다. 선택 수 4 이하의 consensus와 선택
수 7 이하의 unanimous union은 최저 score가 0.75 이상일 때 추가 검사를 생략한다. 그 외
불확실한 작은 집합은 1,500px draft, 선택 수 변화 또는 미해결 소집합은 원본 해상도로 단계적으로
재검사한다. 이 fast path는 개발 오류를 해결하는 데 필요했던 낮은 score·불일치 장면의 검사를
유지하면서, 명백한 장면의 네 모델 재실행을 제거한다.

### 버전별 실행 결과

| 버전 | 변경과 결과 | 판정 |
| --- | --- | --- |
| v2 | 최초 locked 운영 115장: `APPROVED` 473/504, FP 포함 이미지 7/115 | `rejected_locked_test` |
| v12 | 고정 4-model+LDA, 개발 1,409/1,410; `hard_098` IoU 누락 | 실패 보존 |
| v13 | 단계적 원본 재검사로 개발 1,410/1,410, p95 316ms | 정확도 통과·성능 실패 |
| v14 | 선택적 재검사로 개발 1,410/1,410, p95 312ms | 정확도 통과·성능 실패 |
| v15 | unanimous bypass로 개발 p95 87.56ms, 운영 개발 재사용본 p95 415.71ms | 운영 분포 성능 실패 |
| v16 | area-first와 score 0.75 fast path, 두 개발 범위 모두 정확도·성능 통과 | `active_development` |

v16 공식 분모 평가는 Worker 응답에서 직접 집계한다. `IMAGE_RECAPTURE` 또는 detector FN으로
도달하지 못한 GT도 전체 판정 가능 GT 분모에 남기고, unmatched detector FP는 객체 분모에서
제외한 뒤 `SEGMENTATION` 이미지 FP gate로 검출한다.

```powershell
bixolon evaluate bread-1.1-runtime `
  --package-dir artifacts/packages/bread-zero-error-1.1.0-candidate `
  --dataset-root datasets/bread_dataset `
  --annotation-name multi_object_instances.json `
  --dataset-version bread-1.1-development-emh-300 `
  --output artifacts/experiments/bread-zero-error-1.1.0/reports/runtime-six-gate-development-v16-area-first-fast-path.json `
  --evidence-role development --provider cuda `
  --cuda-dll-dir artifacts/portable/rpc200-v18-benchmark/cuda-runtime `
  --warmup-count 20
```

| v16 지표 | E/M/H 개발 300장 | 반려 locked test의 개발 재사용 115장 | gate |
| --- | ---: | ---: | ---: |
| `SEGMENTATION` | 300/300 (100%) | 111/115 (96.522%) | ≥90% |
| end-to-end `APPROVED` | 1,410/1,410 (100%) | 504/504 (100%) | 운영 ≥90%, 목표 ≥99% |
| `SEGMENTATION` 이미지 FN | 0/300 | 0/111 | ≤0.1% |
| `SEGMENTATION` 이미지 FP | 0/300 | 0/111 | ≤0.1% |
| `APPROVED` 오인 | 0/1,410 | 0/504 | ≤0.1% |
| Candidate out | 0/1,410 | 0/504 | ≤0.1% |
| CUDA 평균 / p95 | 91.71 / 90.99ms | 74.33 / 87.09ms | 둘 다 ≤100ms |
| CUDA p50 / p99 | 77.22 / 524.51ms | 71.37 / 95.11ms | 진단 |

CPU/CUDA parity는 E/M/H 300장 전체의 request ID와 처리시간을 제외한 공개 응답 trace로
비교했다. 양쪽 trace SHA-256은 모두
`c42eec94614677d4d1ecaebc4a98a7a1ab6d97a301750adb59cb73465c98326c`였다. 최종 상태·reason,
클래스·Top-3 순위 mismatch는 0/300, bbox mismatch는 0/1,410, 최소 bbox IoU는 1.0,
최대 confidence 차이는 0.0이었다. parity report SHA-256은
`2a5188daa05a19a3c2fecd1d18841effada88cfcf18c0faf61a38acaaa05ecfb`다. CPU 평균/P95
2,325.99/2,749.38ms는 기능 호환 진단이며 CPU에는 지연 gate가 없다.

E/M/H 개발 report SHA-256은
`ecf2142b3ce584d5e210fa57647c31573677e74c385bbceb8d7962a7e7e2862c`, 운영 개발 재사용
report SHA-256은 `9920466e850bdde9e399b627233978021b50e2a8db24e46f95ccde7daf157cf3`다.
후자의 빈 트레이 4장은 GT 0개이며 `DETECTOR_NO_OBJECT` `IMAGE_RECAPTURE`가 맞다.

v2에서 한 번 열어 반려한 운영 수집본은 v3의 독립 test가 아니다. 실패를 확인한 뒤 프레임 제거와
fast path threshold를 선택했으므로 `development_only`로 영구 전환했다. 현재 로컬에는 v3가
접근하지 않은 독립 annotated test가 없다. 따라서 v16은 개발 목표와 성능 gate를 통과했지만
`promotion_eligible=false`이며, 새 촬영 세션을 잠근 뒤 단 한 번 전체 평가하기 전까지
`promoted` 또는 운영 기본 package로 표시하지 않는다. 고정 계약은
`manifests/bread-zero-error-1.1/final_candidate_v3_2026-08-18.json`에 기록한다.

package 자체는 대형 ONNX를 Git에 넣지 않고 다음 명령으로 독립 재조립한다. manifest와 metadata
template, 각 입력 ONNX checksum이 하나라도 다르면 실패하며 기존의 다른 파일은 덮어쓰지 않는다.

```powershell
bixolon model bread-1.1-candidate-package `
  --manifest manifests/bread-zero-error-1.1/final_candidate_v3_2026-08-18.json `
  --output-dir artifacts/packages/bread-zero-error-1.1.0-candidate-reproduced `
  --report artifacts/experiments/bread-zero-error-1.1.0/reports/package-v3-reproduction.json
```

별도 `bread-zero-error-1.1.0-candidate-reproduced` 디렉터리에 metadata와 ONNX 5개를 모두 새로
복사해 package loader 검증을 통과했다. 재현 report SHA-256은
`720bb31c2300e8de4930cb515cc7f45fd600ebf9e034e0d08ffe76fce00d9663`다.

최종 저장소 회귀는 `ruff check`, `ruff format --check`, 전체 Python 테스트,
`flutter analyze`, Flutter 164개 전체 테스트와 `git diff --check`를 모두 통과했다. 따라서
코드·package·provider·회귀 측면의 개발 검증은 끝났고, 남은 단일 blocker는 v3가 보지 않은 새
독립 촬영 세션의 잠금 평가다.

## 2026-08-18 추가 로컬 독립 데이터 preflight

v3 commit `adfae95`를 push한 뒤 로컬 raw storage를 다시 전수 조사했다. 모델 inference 전에
COCO 구조, 실제 이미지 SHA-256과 크기, review 완료 여부, capture-session provenance, 기존
Detector/Classifier manifest의 exact SHA 및 dHash≤2 중복을 검사하는
`bixolon evaluate bread-1.1-independent-preflight`를 추가했다.

`bread_project/group`의 299장도 E/M/H 개발 이미지와 nearest dHash 거리가 모두 2 이하였고,
269장은 dHash가 완전히 같았다. `bread_project_2`는 이미 사용한 E/M/H export,
`bread_project_3`은 canonical dataset source와 review archive다. `train`, `train2`, `train3`,
`train4`는 단일 객체 분류 자료라 end-to-end multi-object segmentation/FN/FP/재촬영 gate를
검증할 수 없다. 따라서 현재 로컬의 적격 multi-object 독립 데이터는 0개이며 새 촬영 세션이
필요하다는 결론은 유지한다. 전체 inventory는
`manifests/bread-zero-error-1.1/raw_data_independent_inventory_2026-08-18.json`에 고정한다.

## 2026-08-19 사용자 지정 operational collection 재검증

사용자가 `datasets/bread_dataset/operational_collections/2026-08-18`을 추가 평가 대상으로
지정했다. 현재 annotation SHA-256
`7909a8fdb31850b5af1cb4e95aa007b6b9e3d2ec5da06e6f8c9f74d3ed2bb56f`와 115장 image
manifest SHA-256 `dac22864668b8ba008db988201f5c851c2d323c798f704c40d3abf10cf243f6e`는
기존 v2 locked test와 정확히 같다. v2 실패 결과가 공개된 뒤 v3의 FP 제거와 fast path를
선택할 때 사용했으므로 이 데이터의 영구 역할은 `development_only`다.

현재 v3 CUDA package를 다시 실행한 결과 `SEGMENTATION` 111/115(96.522%), 빈 tray
`IMAGE_RECAPTURE` 4/115, all-GT `APPROVED` 504/504(100%), FN·FP·승인 오인·Candidate out
각 0건이었다. 평균/p50/p95/p99는 75.67/73.09/88.48/92.15ms로 정확도·성능 재현에
성공했다. report SHA-256은
`99d048600edf9e911ccc8efdc8780a7fcf8d60e64bfd5734d182f0fcdf2229d7`이며 보고서의
`promotion_eligible`은 의도대로 `false`다.

후속 후보가 본 반려 test를 독립성 검사에서 빠뜨리지 않도록 115장 identity manifest
`rejected_operational_v3_development_identity.jsonl`을 추가했다. 전체 v3 source 715행으로
preflight한 결과 115/115 exact SHA 중복을 검출해 독립 잠금을 거부했다. 지정 경로에는 별도
review metadata와 per-image capture-session manifest도 없다. 반려 보고서 SHA-256은
`aabecff19033729f5c9c816fcd18592f193304496020851daff583b259f2589e`이다.

Runtime gate의 `independent` 역할은 이제 적격 preflight와 candidate manifest를 필수로 받고,
annotation·image manifest·candidate commit·전체 source lineage·package metadata checksum을
모두 재검증한 뒤에만 ONNX session을 만든다. 이 operational collection을 independent로 실행한
부정 테스트는 `independent evidence rejected before model inference`로 종료됐고 출력 report를
생성하지 않았다. 따라서 개발 재현 증거는 강화됐지만 새 독립 test blocker는 해소되지 않았다.

## 2026-08-19 1.1.0 소유자 승인 승격

프로젝트 소유자는 Classifier 데이터 계보와 독립 test 부재를 확인한 뒤에도 현재 v3를
Worker·Detector·Classifier `1.1.0` 운영 기준선으로 우선 승격하도록 명시적으로 지시했다. 이에
다음 두 실패를 제거하거나 통과로 바꾸지 않고 production package의 `manual_waiver`로 기록한다.

- 최종 LDA head가 `single_objects` 200장만 사용하지 않고 E/M/H ROI 1,410개 전체로 fit됐다.
- v3가 보지 않은 새 독립 잠금 test가 없다.

승격 package는 `artifacts/packages/bread-worker-1.1.0`, 입력 결정은
`configs/releases/bread_1.1.0_owner_waiver.json`이다. E/M/H 1,410/1,410과 운영 재사용본
504/504는 계속 개발 결과이며 독립 성능으로 재분류하지 않는다.

이 예외는 1.1.0 한정이다. 1.1.1부터 Classifier의 모든 파라미터, fitted statistic,
calibration, threshold와 ranking/TTA 선택은 `single_objects` 200장과 그 결정적 파생 입력만
사용한다. 버전별 최소 실험과 종료 조건은
[Classifier 200장 전용 1.1.1+ 계획](bread-classifier-200-only-1.1.1-plan.md)에 고정한다.

```powershell
bixolon model promote `
  --candidate-dir artifacts/packages/bread-zero-error-1.1.0-candidate `
  --release-report configs/releases/bread_1.1.0_owner_waiver.json `
  --output-dir artifacts/packages/bread-worker-1.1.0 `
  --decided-on 2026-08-19 `
  --approve-known-limitations
```
