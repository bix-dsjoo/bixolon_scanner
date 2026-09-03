# 1-class SSDLite·고정 224·독립 verifier 단순화 실험

## 결론

제안한 구조는 구현과 재현이 가능하고, 소스 ground truth로 교정한 세 진단 세트에서
목표 5개를 모두 만족했다. 다중 객체 원본 300장에서 초기에 오승인으로 보인 2건은
모두 stale registry manifest의 주석 오류였다. 원본 COCO, 실제 이미지, 모델 예측은 모두
`bread_16 Grain Campagne`에 일치했지만 registry의 두 annotation만 `bread_17 Almond Campagne`로
변경돼 있었다.

교정 후 후보는 현재 회귀 415장, 운영 촬영 69장, 다중 객체 원본 300장에서 오승인,
Top-3 누락, FP, FN이 모두 0이다. 그러나 활성 `0.1.12`보다 정답 승인 커버리지와 속도가
낮고 독립 held-out 증거가 없다. 따라서 회귀 후보로는 유효하지만 새 제품 버전으로 올리지
않았고 활성 설정·Runtime·Catalog를 변경하지 않았다.

이 결과는 모두 학습·임계값 선택과 겹치는 회귀 데이터의 재현 진단이며 독립 일반화 성능,
인증 또는 SLA 증거가 아니다.

## 구현한 구조

1. `detector_class_mode=class_agnostic`인 1-class SSDLite320이 ROI box와 영상 품질을 판정한다.
2. DINOv3 ConvNeXt-Tiny의 기존 224 ONNX graph를 fallback에서 고정 primary로 옮겨 모든 정상 ROI를
   224×224로 분류한다.
3. 회전 verifier와 Frozen DINOv3 ViT-B/16 160을 별도로 실행한다.
4. primary가 `UNKNOWN`이거나 verifier 불일치로 승인이 차단된 ROI에서 verifier 중 하나라도
   OOD·품질 거부를 반환하면 `CLASSIFIER_TOP3_UNSAFE` `SEGMENT_RECAPTURE`로 보수 처리한다.
5. 공개 상태, reason code, Top-3, `IMAGE_RECAPTURE`, `SEGMENT_RECAPTURE`, `ERROR` 도메인은 변경하지
   않는다.

재현 빌더는
`python -m bixolon_scanner.experiments.bread.build_fixed224_classifier_runtime`이다. 소스 ONNX graph와
weight를 수정하지 않고 기존 224 graph를 primary 파일로 배치한 뒤 checksum을 재생성한다.

## 평가 결과

CPU ONNX Runtime, detector worker 1, detector·embedder thread 4에서 측정했다.

| 진단 세트 | 객체 | 정답 `APPROVED` | 정답 승인율 | 오승인 | Top-3 누락 | FP/FN | 결과 |
|---|---:|---:|---:|---:|---:|---:|---|
| 현재 회귀 415장 | 1,914 | 1,896 | 99.0596% | 0 | 0 | 0/0 | 통과 |
| 운영 촬영 69장 | 138 | 137 | 99.2754% | 0 | 0 | 0/0 | 통과 |
| 다중 객체 원본 300장 | 1,410 | 1,396 | 99.0071% | 0 | 0 | 0/0 | 통과 |

기본 후보의 독립 verifier 실행 상한은 approval score 0.55였다. stale registry로 평가했을
때 image 200의 annotation 907과 image 296의 annotation 1377이 각각 `bread_17`을 정답으로 요구했지만,
소스 `multi_object_instances.json`의 category ID는 둘 다 16이다. 이미지 296의 ROI는 평평한 타원형
밑면과 스탬프까지 `bread_16` flipped 기준 이미지와 일치한다. 기존 source audit도 동일한 두
건만 registry 17 → source 16 교정으로 기록하고 있다.

평가 산출물은 Git에 포함하지 않는 대형 실험 경로
`artifacts/experiments/simplified-pipeline-0.1.12` 아래에 보존한다.

## 검토 판정

- 검출 recall은 세 진단 세트에서 FN 0이었지만, 포함되지 않은 부분 미검출을 런타임에서
  증명하는 completeness 구성요소가 없으므로 100% 보증으로 해석하지 않는다.
- Frozen ViT는 주 분류기와 다른 backbone이지만 같은 crop·Catalog·촬영 분포를 사용하므로
  오류가 독립이라고 가정할 수 없다.
- class-aware Detector primary routing을 유지한 활성 `0.1.12`는 다중 객체 300장에서 오승인 0,
  정답 승인율 99.3617%로 단순화 후보의 99.0071%보다 높다.
- 활성 교체를 다시 검토하려면 학습·calibration·threshold 선택과 물리적으로 분리된 group-aware
  독립 데이터와, Frozen ViT의 공통 오류를 감지할 추가 증거가 필요하다.
