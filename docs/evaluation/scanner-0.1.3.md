# BIXOLON Scanner 0.1.3 개발 검증

## 데이터와 분리

- 데이터 루트: `datasets/bread_dataset`
- 전체 유효 이미지: 415장, 객체 annotation: 1,914개
- Detector: 전체 유효 데이터 사용, class-agnostic objectness 한 클래스
- Classifier 후보: `single_objects` 200장과 `single_objects_3` 240장을 동일 grouped-fold 조건으로 비교
- 선택 소스: `single_objects_3`만 사용하며 두 소스를 혼합하지 않음
- multi-object 상품 label: Classifier 학습·보정에 사용하지 않음
- split: 원본 SHA-256, source image와 capture session을 포함한 provenance group 단위 분리
- 별도 고정 test set: 없음

두 Classifier 소스 비교에서 `single_objects`는 Top-1 오류 282개·Top-3 이탈 48개,
`single_objects_3`은 Top-1 오류 167개·Top-3 이탈 22개여서 후자를 선택했습니다. 평가 대상과
Classifier 학습 source 사이의 정확한 파일 SHA-256 중복은 0개입니다.

annotation 시각 검수에서 `multi_object_instances.json`의 annotation `907`과 `1377`이 실제 상품과
다른 category로 기록된 것을 바로잡은 뒤 registry와 고정 manifest를 다시 생성했습니다. 이는 특정
이미지 실행 예외가 아니라 ground truth 정정이며 Runtime에는 image ID나 예외 목록이 들어가지 않습니다.

## 실패 분석과 선택

초기 0.1.2는 Detector FN 6/FP 4, 승인 오분류 5, 올바른 승인 98.6938%였습니다. D-FINE
objectness 재학습, 여러 DINOv3 backbone·augmentation·metric head, label-free target neighbor
adaptation을 비교했습니다. 최종 Detector는 YOLO26의 class-agnostic objectness 모델을 선택했고,
Classifier는 DINOv3 ConvNeXt-Tiny soup의 180° 일관성과 별도 DINOv3 ViT-B/16 frozen verifier를
전역 규칙으로 결합했습니다. 상품·클래스·상품쌍·객체 수·난이도·이미지별 예외는 없습니다.

MobileCLIP2는 공개 weight의 research-only 사용 조건 때문에 제외했습니다. ConvNeXt V2 공개
weight도 비상업 조건 때문에 제외했습니다. SigLIP2 B/16 frozen, DINOv2 small, DINOv3 ConvNeXt
Tiny frozen과 label-free AdaContrast 계열 시도는 이 데이터의 안전 목표를 충족하지 못해 선택하지
않았습니다.

## 전체 개발 회귀 결과

| 지표 | 결과 |
|---|---:|
| `ERROR` | 0 |
| `IMAGE_RECAPTURE` | 4/415 (0.9639%) |
| Detector FP / FN, IoU@0.5 | 0 / 0 |
| 올바른 `APPROVED` | 1,895/1,914 (99.0073%) |
| 승인 오분류 | 0 |
| `UNKNOWN` | 14/1,914 (0.7315%) |
| `UNKNOWN` 정답 Top-3 이탈 | 0 |
| `SEGMENT_RECAPTURE` | 5/1,914 (0.2612%) |
| OpenVINO full-path 내부 평균 / p95 | 112.29ms / 183.56ms |
| packaged Worker HTTP 평균 / p95 | 144.95ms / 210.23ms |

속도 표본은 `IMAGE_RECAPTURE` 조기 종료 4장을 제외한 411장입니다. 개발 PC는 Windows와
OpenVINOExecutionProvider 1.24.1을 사용했습니다. packaged 수치는 standalone PyInstaller EXE에
JPEG/PNG를 multipart로 전송한 왕복시간이며 Worker 내부 처리시간은 평균 127.83ms, p95 194.52ms였습니다.
고정 증빙은 `artifacts/evaluations/scanner-0.1.3`의 `full-valid-openvino.json`, trace,
`packaged-worker-full-valid-http-openvino.json`입니다.

## 신규 SKU와 한계

Detector는 상품 label을 입력·출력·후처리에 사용하지 않으므로 신규 SKU 추가에 Detector 재학습,
교체 또는 threshold 변경이 필요하지 않습니다. Classifier embedder와 전역 판정 threshold도 Runtime에
고정되고 SKU 데이터는 Catalog payload로만 추가됩니다. 기존 Catalog payload와 Runtime checksum을
비교해 모델·정책의 비의도 변경을 차단합니다.

신규 SKU 등록은 기존 adapter coefficient를 bit-for-bit 유지하고 새 SKU 출력만 one-vs-rest Ridge로
추가합니다. 새 출력은 주 분류기, 180° 평균 view와 독립 DINOv3 ViT-B/16 verifier의 Top-1이 모두
같을 때만 사용하고, 미합의 시 확장 전 Catalog 판정을 그대로 복원합니다. 상품별 threshold나 예외는
없습니다.

20개 기존 클래스를 한 번씩 신규 SKU로 가정한 class-holdout 개발 회귀에서 `single_objects_3`만
등록 support로 사용하고 multi-object label은 사후 평가에만 사용했습니다. source와 평가 이미지의
정확한 SHA-256 중복은 0개입니다.

| 증분 Catalog 지표 | 결과 |
|---|---:|
| pseudo-new-SKU 시나리오 | 20 |
| 기존 클래스 판정 비교 | 36,366 |
| 기존 출력 변화 | 0 |
| correct-to-incorrect 전환 | 0 |
| 세 verifier가 선택한 신규 SKU 객체 | 1,577/1,914 (82.3929%) |

결과는 `artifacts/evaluations/scanner-0.1.3/incremental-new-sku.json`에 고정했습니다. 8-view 결정론적
증강 router도 multi-object label 없이 비교했지만 신규 선택이 1,871건으로 원본 support joint-router의
1,875건보다 낮아 채택하지 않았습니다.

이는 지정 데이터 안의 class-holdout 무회귀 증빙이며 데이터셋 외부 SKU의 독립 일반화 보장은
아닙니다. 전체 결과 역시 인증 또는 SLA가 아닙니다. 300ms 목표는 개발 PC packaged Worker 기준이며
N100 설치 생성 조건으로 사용하지 않습니다.

N100의 Intel 내장 GPU 효과는 `scripts/build_n100_gpu_test.ps1 -Version 0.1.3`으로 만드는 별도
진단 ZIP에서 비교합니다. CPU-only profile과 `CPU Detector + Intel GPU Embedder` profile은 같은
Worker·Runtime·Catalog·이미지 순서를 사용합니다. 하이브리드 profile에서는 주 ConvNeXt-Tiny,
180도 회전과 독립 ViT-B/16 embedding graph를 모두 GPU에 배치하며 Detector는 CPU에 고정합니다.
비교 실행에서는 전체 GPU graph를 초기화할 수 없으면 CPU fallback 없이 해당 profile을 실패로
기록합니다. N100 100장 측정에서 하이브리드는 CPU-only 대비 평균 1.315배, p95 1.286배 빨랐고
semantic 불일치는 0건이었습니다. confidence 최대 차이는 0.0085055232였습니다. 최종 Worker는
GPU 초기화 실패를 경고 로그에 남긴 뒤 CPU Embedder로 한 번 재조립하며 `/health/ready`에 실제
provider를 표시합니다.

## 기술 근거

- [YOLO26 공식 문서](https://docs.ultralytics.com/models/yolo26/)
- [YOLO OpenVINO 통합 문서](https://docs.ultralytics.com/integrations/openvino/)
- [DINOv3 공식 저장소](https://github.com/facebookresearch/dinov3)
- [DINOv3 논문](https://arxiv.org/abs/2508.10104)
- [OpenVINO CPU 성능 힌트와 스레드 설정](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html)
- [ONNX Runtime OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
