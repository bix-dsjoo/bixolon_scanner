# Class-agnostic Detector + 확장 가능한 Classifier 실험

## 목적과 결론

활성 `0.1.11`의 20-class SSDLite Detector를 SKU와 독립적인 1-class `bread/object` Detector로
바꾸고, 새 빵 추가 시 Detector 재학습 없이 Catalog와 Classifier만 확장할 수 있는지 진단했다.
최종 class-agnostic 실험 후보 `runtime-candidate-v6-dense-fallback`은 Detector 출력이
`[1, 3234, 1]`인 class-agnostic SSDLite320 MobileNetV3-Large와 기존 `0.1.11` Classifier/Catalog를
사용한다.

415장, 운영 촬영 69장과 multi-object 300장에서 FP/FN·오승인·`UNKNOWN` Top-3 누락이 모두
0이었다. 정답 `APPROVED`는 415장에서 1,898개로 기존 `0.1.7`의 1,893개보다 5개 많고,
multi-object 300장에서는 1,398개로 기존 `0.1.7`의 1,394개보다 4개 많다. 운영 촬영은
138/138개가 모두 정답 `APPROVED`였다. CPU/CUDA 415장 최종 status·reason·bbox·prediction·Top-3도
완전히 일치했다.

이후 운영 조건을 다시 확인한 결과, 매장별로 Detector와 Classifier를 함께 갱신·배포할 수 있으므로
Detector를 SKU 독립적으로 고정할 필요는 없다. 이 조건에서는 활성 class-aware `0.1.11`이 415장
정답 `APPROVED` 1,905개로 v6보다 7개 많고, 실제 N100 hybrid Worker p95도 `469.471ms`로
1,000ms 목표를 이미 충족한다. 따라서 운영 권고안은 활성 class-aware `0.1.11` 유지이며, v6는
Detector 재학습 없이 SKU를 확장해야 하는 경우의 대안으로 보존한다.

매장에는 PyTorch 학습 checkpoint인 `.pt`를 Worker 실행 모델로 배포하지 않는다. SSDLite와
DINOv3를 각각 ONNX로 export한 `detector.onnx`와 `embedder*.onnx`, Catalog, metadata와 checksum을
하나의 매장별 제품 버전으로 묶는다. 새 SKU 추가 시 class-aware 방식을 선택하면 Detector 재학습,
Classifier/Catalog 갱신, 전체 정확도·provider parity 검증을 함께 수행한다.

## 구조

- Detector: torchvision SSDLite320 MobileNetV3-Large, 1 foreground objectness class, BSD-3-Clause
- 학습 초기값: 활성 class-aware checkpoint에서 shape-compatible backbone·box head 464개 tensor를
  전달하고 1-class classification head는 새로 학습
- 외부 pretrained Detector weight: 사용하지 않음
- Classifier/Catalog: 활성 `0.1.11` DINOv3 192 기본 경로, 선택 ROI만 224 fallback, append-only Catalog
- Detector class consensus/corroboration: 비활성
- Worker 보완:
  - 기존 class-aware 불일치 규칙 중 객체가 6개 이상이고 승인 점수가 0.76 이하인 dense-scene
    규칙만 유지
  - 승인 점수 0.85 이하이면서 bbox aspect ratio 3.0 이상인 ROI만 추가 224 재검사
  - 192→224 승인 점수 하락이 0.2를 넘으면 `UNKNOWN`
  - `UNKNOWN` Top-3는 192/224 ranking score의 원소별 최댓값으로 결합
  - fallback 승인 점수 0.08 미만은 provider 공통 `BELOW_APPROVAL_THRESHOLD`

이 정책은 SKU ID나 상품쌍을 참조하지 않는다. 새 빵이 기존 bread/object 범위에 속하면 Detector는
그대로 두고 Catalog support와 Classifier calibration만 갱신할 수 있다. 빵이 아닌 객체 종류,
새로운 촬영 장치·배경 또는 현재 objectness 분포 밖의 형태가 추가되면 Detector recall을 다시
검증하고 필요할 때만 재학습한다.

## 실패 분석 → 개선

1. 기존 binary objectness checkpoint를 그대로 export한 v1은 검출 FP/FN이 0이었지만 오승인
   2건이 남았다.
2. class-aware `0.1.11` checkpoint의 호환 tensor를 이어 학습한 v2는 오승인을 1건으로 줄였으나,
   224 fallback이 192의 올바른 3순위 후보를 덮어써 Top-3 누락 2건이 생겼다.
3. v3는 극단적인 ROI 한 건만 추가 재검사하고 해상도 간 승인 점수 급락을 `UNKNOWN`으로 낮췄다.
   승인 1건을 안전하게 낮추면서 오승인과 Top-3 누락이 모두 0이 됐다.
4. 첫 CPU/CUDA 비교에서는 한 ROI의 승인 점수 `0.0624/0.0748`이 class별 임계값 양쪽에 걸려
   CPU `UNKNOWN`, CUDA `APPROVED`가 됐다. fallback 승인 하한 0.08과 reason 정규화를 추가한 v4는
   두 provider의 최종 의미를 완전히 일치시켰다.
5. class-aware 시절의 세 fallback 규칙을 모두 제거한 v5는 CPU p95를 낮췄지만 오승인 3건이
   재발했다. 실패 3건은 모두 객체가 6개 이상인 dense scene이어서 해당 규칙 하나만 복원한 v6는
   오승인 0을 유지하고 정답 `APPROVED`를 1개 늘렸다.

## 최종 진단

| 진단 | 객체 | 정답 APPROVED | 오승인 | UNKNOWN/Top-3 누락 | SEGMENT_RECAPTURE | FP/FN | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 415 CPU | 1,914 | 1,898 | 0 | 11/0 | 5 | 0/0 | 357.139ms |
| 415 CUDA | 1,914 | 1,898 | 0 | 11/0 | 5 | 0/0 | 84.539ms |
| 운영69 CPU | 138 | 138 | 0 | 0/0 | 0 | 0/0 | 277.539ms |
| multi300 CPU | 1,410 | 1,398 | 0 | 8/0 | 4 | 0/0 | 330.657ms |

개발 PC CPU 수치는 ONNX Runtime CPU provider이며 활성 `0.1.11`의 OpenVINO CPU 수치와 직접
비교하지 않는다. CUDA/CPU parity는 415장 1,914 segmentation에서 decision mismatch 0,
bbox mismatch 0, minimum bbox IoU 1.0, maximum confidence delta `0.0123723`으로 허용치 0.02 안이다.

## 증빙

- Runtime: `artifacts/experiments/class-agnostic-detector-0.1.11/runtime-candidate-v6-dense-fallback`
- 빌드 보고서: `runtime-candidate-v6-dense-fallback-build.json`
- 415 CPU/CUDA: `v6-scanner415-cpu.json`, `v6-scanner415-cuda.json`
- parity: `v6-scanner415-cpu-cuda-parity.json`
- 운영69: `v6-operational69-cpu.json`
- multi300: `v6-multi300-cpu.json`
- N100 동일 입력 개발 PC 비교: `n100-sample100-base-cpu-trace.jsonl`,
  `n100-sample100-v4-cpu-trace.jsonl`
- 기존 v4 N100 진단 ZIP은 v6 이전 실험 패키지이므로 운영 배포 근거로 사용하지 않는다.
- packaged OpenVINO Worker smoke: `v4-packaged-worker-openvino-smoke.json`, 통과

이 결과는 저장소의 development/operational/stress regression 데이터에 대한 진단이며 독립 일반화
성능, 인증 또는 SLA가 아니다. 활성 `0.1.11` Runtime이나 배포 번들은 이 실험에서 변경하지 않았고,
운영에는 실제 N100 실측까지 완료된 활성 class-aware `0.1.11`을 권고한다.
