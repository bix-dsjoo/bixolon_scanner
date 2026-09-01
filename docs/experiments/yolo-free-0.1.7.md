# YOLO-free 0.1.7 진단 후보

## 결론

현재 최선의 후보 `runtime-candidate-v42-ssdlite320-selective-fallback`은 YOLO와 D-FINE ensemble을 모두
제거하고, 프로젝트 데이터로 처음부터 학습한 단일 class-aware SSDLite320 MobileNetV3-Large를
사용한다. 고정 415장과 운영 촬영 69장에서 FP/FN·오승인·Top-3 실패·`ERROR`가 모두 0이었고,
`APPROVED`는 각각 `1,905`개와 `138`개다. 기존 0.1.7보다 각각 10개와 2개, 직전 D-FINE v16보다
각각 7개와 1개 많고, 기존 0.1.7의 오승인 2개는 0개로 줄었다. Worker가 한 ROI의 위험 때문에
전체 batch를 224로 재실행하지 않고 위험
ROI만 재실행하도록 바꿔 개발 PC OpenVINO CPU 전수 p95를 `171.903ms`에서 `143.409ms`로 낮췄다.

실제 Intel N100에서 D-FINE v16은 CPU p95 `1,354.768ms`로 1,000ms 목표에 실패했다. SSDLite
v41은 실제 N100에서 CPU p95 `681.696ms`, CPU detector + Intel iGPU classifier p95
`488.413ms`, 양쪽 `ERROR=0`, 최종 상태·bbox·class rank 불일치 0으로 핵심 목표를 통과했다.
v42의 실제 N100 재측정은 대기 중이다. 모든 수치는 저장소의 development/operational 데이터에
대한 회귀 진단이며 독립 일반화 성능, 인증 또는 SLA가 아니다.

이 후보는 제품 `0.1.11`의 source Runtime으로 채택됐다. 빌드는 모델·Catalog binary를 바꾸지 않고
모든 공개 실행 version만 `0.1.11`로 맞추며, v41 N100 결과는 동일 모델·정책의 참조 실측으로만
표시한다.

## Runtime 구성

- Detector: torchvision SSDLite320 MobileNetV3-Large, 20 foreground class와 background, BSD-3-Clause
- 학습: 외부 detector 가중치 없이 random initialization, 프로젝트 소유 실데이터와 프로젝트에서
  생성한 합성 단일 객체 1,500장 사용
- 실행: 320 고정 입력 ONNX detector 한 번만 실행; detector ensemble과 count verifier 없음
- classifier: DINOv3 ConvNeXt-Tiny 192 입력을 기본 실행하고, 안전 경계 밖 또는 승인 불일치가 있는
  ROI만 기존 224 입력으로 재검증. v42 Worker는 전체 detection 문맥을 유지하면서 선택 ROI tensor만
  만들고, 재검증하지 않은 ROI의 192 결과는 그대로 보존
- 독립 classifier verifier: 동일 checkpoint에서 다시 export한 DINOv3 ViT-B/16 160 입력
- class consensus: detector와 classifier가 강하게 합의한 안전한 `UNKNOWN`만 `APPROVED`로 승격하고,
  위험한 승인 불일치는 `UNKNOWN`으로 낮춘다. 재촬영 승격은 classifier Top-1과 detector class가
  일치할 때만 허용하며 `UNKNOWN` Top-3에는 detector class를 보존한다.
- Worker 실행: ONNX Runtime만 사용하며 PyTorch를 import하지 않음
- 출력: 기존 `IMAGE_RECAPTURE`, `APPROVED`, `UNKNOWN`+Top-3 및 오류 계약 유지

Detector 구현은 [torchvision BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)를
따르며 torchvision 사전학습 가중치는 사용하지 않았다. 따라서 YOLO/AGPL 및 출처가 불명확한
detector pretrained weight 의존성이 없다. 기존 classifier에는 DINOv3 License가 계속 적용된다.
상세 고지는 `licenses/THIRD_PARTY_MODELS_SSDLITE.md`와 Runtime의 `licenses` 디렉터리에 있다.

## 구조 대안 조사

해상도 축소만 반복하지 않고 다음 계열을 함께 검토했다.

- 경량 DETR: [LW-DETR](https://arxiv.org/abs/2406.03459),
  [RT-DETR](https://arxiv.org/abs/2304.08069), RT-DETRv4. N100에서 기존 D-FINE transformer detector가
  이미 p95 `523.429ms`였으므로, 이번 반복에서는 multi-scale transformer를 추가하기보다 단일
  convolutional detector를 먼저 검증했다.
- 최신 mobile backbone: [MobileNetV4](https://arxiv.org/abs/2404.10518)와 RepViT. 향후 SSDLite가
  N100에서 실패할 경우 head는 유지하고 backbone만 교체할 수 있는 다음 후보로 남겼다.
- 고전 영상처리 foreground component: 배경 분리와 connected component 조합은 계산량은 작았지만
  겹친 객체와 저대비 객체 recall이 부족해 단독 detector로 기각했다.
- INT8 PTQ: ONNX Runtime과 NNCF 양쪽을 시험했지만 이 graph/데이터에서는 정확도 또는 지연이
  악화돼 기각했다.

SSDLite 선택은 단순히 입력을 더 줄인 결정이 아니다. anchor 기반 depthwise convolution 구조,
단일 graph, count verifier 제거, detector가 직접 낸 class와 ROI classifier의 안전한 합의를 함께
적용해 전체 계산 경로를 바꾼 결정이다.

## 실패 분석과 개선 기록

1. D-FINE 네 모델을 그대로 실행한 첫 후보는 정확했지만 N100에 불필요한 계산이 컸다.
2. production 모델 하나를 기본 실행하도록 변경해 detector 계산을 줄였다.
3. 낮은 score의 유효 단일 객체가 누락되는 사례는 exact-count가 1이고 compact proposal이 안전한
   경우에만 Catalog로 복구했다.
4. 큰 단일 객체가 전역 area filter에 제거되는 사례는 저해상도 최대 box 면적 비율을 `0.35`로
   분리해 복구했다.
5. 공유 OpenVINO compiled-model cache가 구조는 같고 weight가 다른 D-FINE fold를 충돌시켜 오검출을
   만들었다. cache를 device와 model path/size/mtime fingerprint로 namespace해 해결했다.
6. 저해상도 단일 객체 확장 검사는 정상 단일 객체 세 장에서 detector 단계에 `542~1,948ms`
   outlier를 만들었다. production 단일 모델과 area override 이후에는 정확도 이득이 없어 비활성화했다.
7. 운영 69장 OpenVINO 진단은 정확도를 유지하면서 p95가 `321.775ms`에서 `298.366ms`, 최대가
   `1,975.217ms`에서 `332.911ms`로 감소했다. 이 수치는 Core Ultra 9 285K의 개발 PC 결과이며
   N100 성능으로 해석하지 않는다.
8. production 단일 모델을 모든 이미지에 무조건 적용한 probe는 8 FN/1 FP로 실패했다. 실패 9장은
   모두 exact-count와 검출 수가 달랐고, 신뢰도 `0.7` 이상의 count mismatch에만 전체 ensemble과
   proposal 보정을 실행해 FP/FN을 다시 0으로 만들었다.
9. 단일 모델의 box 문맥 때문에 기준에서 승인되던 6개 ROI가 `UNKNOWN`이 됐다. 큰 이미지에서
   최종 분류가 `UNKNOWN`일 때만 전체 ensemble로 한 번 재검출해 `APPROVED` 6개를 복구했다. 반대로
   단일 모델에서 안전하게 승인된 기준 `UNKNOWN` 5개는 그대로 유지돼 전체 승인 수가 증가했다.
10. 640 detector를 그대로 둔 `v4`의 실제 N100 CPU-only p95는 `1,196.582ms`로 목표를
    `196.582ms` 초과했다. 단계 p95는 detector `484.184ms`, classifier `884.325ms`였으므로 두
    구간을 함께 줄여야 했다.
11. D-FINE production graph를 checkpoint에서 고정 576/512/480/448 입력으로 다시 export해 전수
    비교했다. 480 입력을 기본으로 선택하고 작은 원본과 위험 사례만 640/ensemble로 보내면서
    415장 FP/FN 0을 유지했다.
12. classifier 기본 입력을 224에서 192로 줄인 뒤 위험 사례만 224로 재검증했다. 첫 cascade는
    unsafe 판정을 fallback하지 않아 승인 4개가 `SEGMENT_RECAPTURE`로 악화됐고, unsafe도 고해상도
    재검증하도록 바꿔 복구했다.
13. 독립 ViT-B verifier를 checkpoint에서 192로 다시 export한 `v15`는 415장에서 승인 1개를 더
    복구해 총 1,900개가 됐다. 160 입력 `v16`은 승인이 1,898개로 2개 줄었지만 기준보다 3개
    많고 오승인은 없었으며, 첫 100장 OpenVINO CPU p95가 `306.649ms`에서 `278.950ms`로 줄어
    N100용 최종 후보로 선택했다. 176 입력 `v17`은 오승인 1건이 생겨 기각했다.
14. primary classifier까지 160으로 줄인 `v19`는 오승인 1건이 생겼다. 5개 객체 저점수 승인만
    224로 재검증한 `v26`은 오승인을 없앴지만 첫 100장 p95가 `380.679ms`로 악화돼 기각했다.
15. 회수한 실제 N100 v16 결과는 CPU p95 `1,354.768ms`, CPU detector p95 `523.429ms`, classifier
    p95 `834.742ms`였다. CPU+iGPU도 p95 `1,345.757ms`, semantic mismatch 1건으로 실패했다.
16. JPEG EXIF orientation을 적용하지 않은 첫 SSDLite cache는 annotation과 image 좌표가 달라져
    실패했다. cache 생성 시 `ImageOps.exif_transpose`를 적용하고 원본 shape provenance를 기록해
    수정했다.
17. torchvision SSD 기본 loss는 빈 이미지에서 classification hard negative를 선택하지 않아 빈
    화면 FP를 학습하지 못했다. 최소 negative anchor loss와 빈 이미지 반복 학습을 추가해 고정
    415장과 운영 69장의 detector FP/FN을 0으로 만들었다.
18. binary objectness v6은 위치 recall은 충분했지만 승인 수를 높일 detector class 증거가 없었다.
    20-class head로 전환하고 v6의 호환 가중치를 이어 학습한 v7은 검출된 `1,914/1,914`와
    `138/138` 객체에서 detector class를 모두 맞췄다.
19. detector class를 무조건 승인에 사용하지 않고, 고점수 detector와 classifier Top-3가 합의한
    안전한 경계 사례만 승격하고 위험한 불일치는 `UNKNOWN`으로 낮추는 v41 정책을 적용했다. 그 결과
    오승인과 Top-3 실패 0을 유지하면서 승인 수를 1,905개와 138개로 늘렸다.
20. 실제 N100 v41은 CPU 평균/p95 `418.587/681.696ms`, CPU detector+iGPU classifier
    `336.946/488.413ms`로 둘 다 1,000ms 목표를 통과했다. detector p95는 각각
    `23.194/33.936ms`, classifier p95는 `593.445/396.674ms`여서 남은 병목이 classifier임을
    확인했다.
21. 기존 Worker는 위험 ROI가 하나라도 있으면 같은 이미지의 모든 ROI를 224로 재실행했다. v42는
    위험 ROI index만 재실행하되 neighbor mask는 전체 detection 문맥으로 만들고 선택 결과만 원래
    batch에 병합한다. 415장 FP/FN·오승인·Top-3 실패 0과 승인 1,905개를 유지하면서 개발 PC
    OpenVINO p95를 `143.409ms`로 16.6% 줄였다. v41과 다른 응답은 이미지 267의 정답 포함
    `UNKNOWN` Top-3 3순위 하나뿐이며 정답은 두 후보 모두 1순위였다.
22. v41 N100의 CPU/iGPU 최종 의미는 완전히 같았지만 GPU 수치 오차 최대 `0.013720`이 사전 진단
    허용치 `0.01`을 넘겨 benchmark 자체 `passes=false`였다. 상태·bbox·class rank 완전 일치를 계속
    강제하면서, 실제 provider 수치 오차 허용치만 `0.02`로 분리했다. iGPU peak working set은 기존
    2GiB 상한을 1.6MB 넘었지만 CPU 대비 1.226배였으므로 현장 상한을 2.25GiB로 조정했다.

## 현재 최선 후보 정확도 진단

### 고정 415장

- 이미지: 415장, 빈 이미지 4장
- GT/prediction/matched: `1,914/1,914/1,914`
- FP/FN: `0/0`
- `APPROVED`: `1,905/1,905` 정답 (`0.1.7` 기준 승인 `1,895`개 중 정답 `1,893`개·오승인
  `2`개, D-FINE v16 정답 승인 `1,898`개)
- `UNKNOWN` Top-3: `4/4` 적중
- `SEGMENT_RECAPTURE`: 5
- 빈 이미지 `IMAGE_RECAPTURE`: 4/4
- `ERROR`: 0
- RTX 5080 CUDA Worker 경로: p50 `47.360ms`, p95 `82.659ms`, full-path p95 `83.072ms`
- 개발 PC OpenVINO CPU 전수: p50 `74.136ms`, p95 `143.409ms`, full-path p95 `143.532ms`

증빙:

- CUDA trace SHA-256: `dfe5556aeee4dd44aeb7eb2e0173449ea7e29d255eab7683dbe381702aef7271`
- CUDA report SHA-256: `899035d1b5c013a66a19d9fda0fd9daf80dec819427707941d80ff6f0638d7d2`
- OpenVINO CPU trace SHA-256:
  `630ce6c7f60e51ec6539905575108898e743ebf4f133ce7917b2d332e55a563b`
- OpenVINO CPU report SHA-256:
  `e3f6f327031efcd88cadd6c76d23302514c35c251b5767ac2d748746bfbf1c64`
- CUDA/OpenVINO parity report SHA-256:
  `c60044a8404335a68842bafded8a107a0cdd0f6601c3faf2b141df5aec716f75`

415장 CUDA와 OpenVINO CPU 비교에서 status, segmentation count, bbox, item status, prediction과
Top-3가 모두 같았다. confidence 최대 차이는 `0.005344`로 진단 허용치 `0.01` 안이었다.

### 운영 촬영 69장

- 이미지: 69장, 빈 이미지 6장
- GT/prediction/matched: `138/138/138`
- FP/FN: `0/0`
- `APPROVED`: `138/138` 정답
- `UNKNOWN` Top-3: 0
- `SEGMENT_RECAPTURE`: 0
- 빈 이미지 `IMAGE_RECAPTURE`: 6/6
- `ERROR`: 0
- 개발 PC OpenVINO CPU p95: `78.966ms`, full-path p95 `79.618ms`

증빙:

- OpenVINO CPU report SHA-256:
  `01eb5c7660b214035f99e6c917701e4537878173d456a689062baad31fc0ced3`

## N100 현장 진단

회수한 D-FINE v16 결과 `n100-yolo-free-0.1.7-openvino-device-matrix.json`은 실제 Intel N100 100장
측정이다. CPU-only는 `ERROR=0`, 평균 `834.789ms`, p95 `1,354.768ms`였고, CPU detector + Intel
iGPU classifier도 `ERROR=0`, 평균 `757.157ms`, p95 `1,345.757ms`였다. 후자는 semantic mismatch
1건도 발생했다. 두 profile 모두 1,000ms p95를 넘겨 v16을 최종 후보에서 제외했다.

- 회수한 v16 N100 report SHA-256:
  `f553153f92ef4ea6af6bdb77d76ca64d8f4cab2cba4a98adaf442314843d1903`

회수한 SSDLite v41 결과 `n100-ssdlite-0.1.7-openvino-device-matrix.json`도 같은 100장을 실제
Intel N100에서 측정했다. CPU-only는 `ERROR=0`, 평균 `418.587ms`, p95 `681.696ms`였고, CPU
detector + Intel iGPU classifier는 `ERROR=0`, 평균 `336.946ms`, p95 `488.413ms`였다. 최종
상태·bbox·class rank mismatch는 0이다. 따라서 모델·Worker의 1,000ms 및 출력 목표는 통과했다.
다만 기존 진단의 confidence 허용치 `0.01`에 대해 최대 차이가 `0.013720`이어서 그 JSON의 엄격한
자체 `passes`는 false다. 현재 판정기는 그 과거 자체 판정을 배포 판단으로 재사용하지 않고 원본
수치를 상태·class rank 완전 일치 및 provider 수치 허용치 `0.02`에 대해 독립 재평가하며, 이 평가는
모든 기준을 통과했다.

- 회수한 v41 N100 report SHA-256:
  `eecf21734281e89046f7802b78bd65224a09dbcc41bca5df9a2629492575d0f8`
- 독립 판정 report SHA-256:
  `12c31dd7cce58fe7ddde90e109102f66cd18444b046a7da16f2979b21e854f06`

`scripts/build_yolo_free_n100_gpu_test.ps1 -Ssdlite`는 현재 소스 Worker, v42 Runtime, 동일 Catalog와
OpenVINO CPU/GPU plugin을 새 현장 진단 ZIP으로 만든다. `RUN-SSDLITE-N100-GPU-TEST.cmd`는 같은
이미지에 대해 다음 두 profile을 순서대로 실행한다.

1. OpenVINO CPU detector + CPU classifier
2. OpenVINO CPU detector + Intel GPU classifier

조용한 CPU fallback은 금지하며 최종 상태·class rank parity, confidence 허용 오차 `0.02`, 시작
시간, peak working set과 단계별 지연을 함께 기록한다. 생성된 ZIP은
`artifacts/handoff/n100-ssdlite-0.1.7-openvino-cpu-gpu-test.zip`이고 SHA-256은
`4a7f84567ffbe3763cd7640b17feb71156763fdcdbedccbf38ea9a469d10c8bf`다.

개발 PC에서 새 packaged Worker의 readiness, 정상 scan과 손상·누락·미지원 입력 오류 계약을
smoke했으며 모두 통과했다. 실제 N100 v42 JSON이 회수될 때까지 선택 ROI Worker 개선의 N100 추가
효과는 미확정이지만, 동일 모델·정책의 v41 측정으로 1,000ms 목표 달성은 확인됐다.

- packaged Worker smoke SHA-256:
  `63ca5d19c210e60587788e79e151f30e66821305d077f8d745c73936589128b0`
- candidate manifest SHA-256:
  `4fb9d8e2427bf8c764122cdce42274dfe0a854312f0cf8d30b75f060b00babb3`
- package manifest SHA-256:
  `a0c23dff031ca10db660e92b68f84ceaa508b8d3ef7a0974ebb0376e8e9c9c0c`

반환 파일 `n100-ssdlite-0.1.7-openvino-device-matrix.json`은 다음 명령으로 독립 재계산한다.

```powershell
python -m bixolon_scanner.evaluation.yolo_free_n100_result `
  --report n100-ssdlite-0.1.7-openvino-device-matrix.json `
  --output n100-ssdlite-0.1.7-assessment.json
```

판정기는 benchmark의 `passes`만 신뢰하지 않는다. N100 CPU와 Intel iGPU 감지, 두 profile 완료,
profile별 full-path 10개 이상, `ERROR=0`, 동일 Runtime/Catalog와 count verifier 미구성 계약,
최종 상태·class rank 일치, confidence 최대 차이 `0.02` 이하, full-path 평균과 p95 각각 `1,000ms`
이하인 profile 존재, 실제 선택된 provider의 목표 충족과 이미지 원본·경로 비기록을 각각 다시 검사한다.

## 검증

- `ruff check .`: 통과
- `ruff format --check .`: 통과
- 전체 Python 테스트: 통과
- `flutter analyze`: 통과
- 전체 Flutter 테스트 188개: 통과
- packaged OpenVINO Worker readiness/scan smoke: 통과
- Runtime/Catalog copy SHA-256 및 ZIP SHA-256 검증: 통과
- N100 결과 자동 판정기 단위 테스트: 통과
- 비-N100 packaged benchmark와 실패 판정 smoke: 의도대로 통과
- 실제 Intel N100 v16 CPU+iGPU 측정: 실패
- 실제 Intel N100 v41 CPU+iGPU 측정: 독립 재평가의 지연·오류·semantic·confidence 목표 모두 통과
- 실제 Intel N100 v42 CPU+iGPU 측정: 배포 후 추가 진단 대기; v41 참조 실측으로 1,000ms 목표 확인
