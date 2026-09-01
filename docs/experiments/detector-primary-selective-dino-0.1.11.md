# Detector-first 선택 DINOv3 실험

## 결론

class-aware SSDLite320이 bbox와 SKU class를 함께 출력하는 현재 운영 조건에서는 DINOv3를 모든 ROI에
실행할 필요가 없다. DINOv3 ConvNeXt-Tiny를 freeze해도 학습 gradient만 없어질 뿐 ONNX graph,
parameter 수와 추론 FLOPs는 그대로라서 실행 속도나 배포 크기가 줄지 않는다. 따라서 모델을 다시
동결하는 대신, 검증된 Detector 예측은 직접 승인하고 위험 ROI만 기존 DINOv3 192→224 cascade로
보내는 Worker 라우팅을 구현했다.

이 후보는 활성 `0.1.11`의 SSDLite·DINOv3·Catalog binary를 하나도 바꾸지 않는다. Runtime metadata의
`detector_primary_classifier_routing`만 다음과 같이 추가한다.

- 직접 승인 class index: `2,3,4,5,6,9,10,11,12,13,14,17,18`
- 최소 detector score: `0.98`
- 한 이미지에서 같은 detector class가 중복되면 모든 중복 ROI를 DINOv3로 전달
- 목록에 없는 신규 class, class가 없는 후보와 저신뢰 후보는 항상 DINOv3로 전달

415장 추적에서 class-aware Detector Top-1은 1,914개 중 1,912개가 맞았다. 오분류 2건은 detector
score가 거의 `1.0`이어서 score threshold만으로 분리할 수 없었지만, 잘못 예측된 class가 각각
`bread_02`, `bread_08`이므로 두 class를 직접 승인 목록에서 제외하면 모두 DINOv3가 복구했다.
직접 승인 1,243개의 detector score 최솟값은 `0.992792`였고 동일 class 중복은 없었다.

## 실패 분석 → 개선

1. DINOv3를 완전히 제거하면 415장에서 detector가 `bread_04→bread_02`,
   `bread_13→bread_08`로 잘못 승인하는 2건이 생겼다. 두 오분류의 detector score가 거의 1.0이므로
   전역 confidence threshold 방식은 기각했다.
2. Detector 오류 class와 기존 비승인 class를 위험군으로 묶어 13개 class만 직접 승인했다. 이
   정책은 415장 ROI의 35.06%, 운영69의 34.06%, multi300의 35.32%만 DINOv3로 전달한다.
3. 최초 후보는 원본 실험 Runtime의 공개 version `0.1.7`을 유지해 활성 `0.1.11` trace 비교에서
   300장 version mismatch가 발생했다. 모델 오류가 아니며, 동일 binary의 `0.1.11` 번들을 기반으로
   후보 v2를 다시 만들어 공개 version parity를 복원했다.
4. 기본 개발 Python에는 OpenVINO EP가 없고, 별도 OpenVINO 환경은 `openvino.dll` 검색 경로가
   빠져 초기화가 두 번 실패했다. N100 빌드 환경의 OpenVINO library 경로를 명시해 같은 소스로
   평가했으며 모델·Worker 오류와 환경 초기화 오류를 구분했다.

## 전체 검증

| 진단 | 객체 | DINO ROI | 정답 APPROVED | 오승인 | UNKNOWN/Top-3 누락 | SEGMENT_RECAPTURE | FP/FN | full-path p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 415 OpenVINO CPU | 1,914 | 671 | 1,905 | 0 | 4/0 | 5 | 0/0 | 99.139ms |
| 415 CUDA | 1,914 | 671 | 1,905 | 0 | 4/0 | 5 | 0/0 | 66.131ms |
| 운영69 OpenVINO CPU | 138 | 47 | 138 | 0 | 0/0 | 0 | 0/0 | 37.102ms |
| multi300 OpenVINO CPU | 1,410 | 498 | 1,401 | 0 | 4/0 | 5 | 0/0 | 96.934ms |

활성 경로와 비교한 개발 PC OpenVINO CPU p95는 415장 `143.532→99.139ms`(-30.9%), 운영69
`79.618→37.102ms`(-53.4%), multi300 `132.087→96.934ms`(-26.6%)다. 415장 OpenVINO/CUDA
1,914 segmentation은 status·reason·bbox·prediction·Top-3가 모두 일치했고 최대 confidence 차이는
`0.0000114`였다. 활성 multi300과 후보도 1,410 segmentation의 최종 status·class rank가 일치한다.
직접 승인 ROI의 공개 `confidence`는 기존 DINOv3 approval score가 아니라 detector score이므로,
활성 경로와 confidence 값 자체는 비교 대상이 아니다.

packaged OpenVINO Worker smoke는 readiness, 정상 3-object `SEGMENTATION`, 누락 multipart 422
`ERROR`, 손상 이미지 422 `ERROR`, 미지원 형식 415 `ERROR`와 privacy 검사를 모두 통과했다.

## N100 지연 근거

기존 `D:/n100-0.1.11-openvino-device-matrix.json`의 입력 SHA-256 100개는 로컬
`multi_object_scenes/easy/easy_001.jpg`부터 `easy_100.jpg`까지와 모두 일치한다. 원본 `0.1.11`이
모든 ROI에 DINOv3를 실행한 실제 N100 p95는 OpenVINO CPU `676.568ms`, CPU Detector+iGPU
Embedder `469.471ms`였고 두 구성 모두 `1,000ms` 목표를 통과했다.

동일 100장 후보 trace의 410개 ROI 중 267개는 Detector에서 직접 끝났고 143개(34.88%)만 DINOv3를
실행했다. 개발 PC의 동일 샘플 p95도 활성 경로 `97.644ms`에서 후보 `72.857ms`로 25.39% 감소했다.
후보는 원본 경로보다 모델 추론을 추가하지 않고 DINOv3 batch만 줄이므로 기존 N100 측정은
`1,000ms` 목표에 대해 CPU `323.432ms`, hybrid `530.529ms`의 보수적 여유를 제공한다. 다만 이 값은
동일 입력·동일 모델의 실행량 근거이지 후보 자체를 N100에서 측정한 수치가 아니다. 최종 N100
실측은 아래 진단 ZIP으로 별도 확정해야 한다.

## 증빙과 한계

- Runtime: `artifacts/experiments/detector-primary-0.1.11/runtime-candidate-v2-0.1.11-selective-dino`
- 빌드: `runtime-candidate-v2-build.json`
- 415 CPU/CUDA: `v2-scanner415-openvino-cpu.json`, `v2-scanner415-cuda.json`
- provider parity: `v2-scanner415-cuda-openvino-parity.json`
- 운영69: `v2-operational69-openvino-cpu.json`
- multi300: `v2-multi300-openvino-cpu.json`
- 활성 multi300 parity: `v2-active-multi300-decision-parity.json`
- 개발 Worker smoke: `v2-worker-cuda-smoke.json`
- packaged OpenVINO Worker smoke: `v2-packaged-worker-openvino-smoke.json`
- N100 진단 ZIP: `artifacts/handoff/n100-detector-primary-0.1.11-openvino-cpu-gpu-test.zip`
- ZIP SHA-256: `7d4df32196d2ae58a928a1b346d16bd816b346d17ed6a0dadf1dd19ac8106ddd`

직접 승인 목록은 현재 development·operational·stress regression에서 관찰된 Detector 오류를 기준으로
고정한 store-specific calibration이다. Detector를 재학습하거나 SKU를 추가하면 신규 class는 먼저
DINOv3 경로로 배포하고 독립 validation에서 Detector 오승인과 비승인이 모두 0임을 확인한 뒤에만
직접 승인 목록에 추가해야 한다. 이 결과는 독립 일반화 인증이나 SLA가 아니며, 활성 `0.1.11` 배포
Runtime은 변경하지 않았다. 실제 N100에서 후보를 재측정하기 전까지는 배포가 아닌 진단 후보다.
