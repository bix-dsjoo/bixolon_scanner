# 0.1.12·0.1.13 운영 일관성 성능 최적화

## 결론

분류 정책은 `0.1.13`을 유지하고 실행 프로필만 최적화한다.

`1-class SSDLite320 → ConvNeXt-Tiny 192 전체 ROI → ConvNeXt-Tiny 224 선택 상세 → Frozen
ViT-B/16 160 선택 검증 → 단일 Safety Arbiter`

개발 PC의 최적 프로필은 `OpenVINO CPU`, detector 1 worker, detector intra-op `4` 또는 자동,
embedder intra-op 자동(`0`)이다. 두 detector 설정의 p95 차이는 측정 변동보다 작았고 판정 결과는
완전히 같았다. 반면 embedder를 4 thread로 고정하면 p95가 크게 나빠졌다. 따라서 다음 원칙을
고정한다.

- 모델 graph·weight, 192/224/160 입력, Catalog, threshold와 routing 조건은 변경하지 않는다.
- 매장·SKU별 예외와 Detector SKU 직접 승인을 사용하지 않는다.
- OpenVINO CPU fallback은 detector `1×4`, embedder thread 자동을 사용한다.
- CPU 전용 전달 패키지의 detector·embedder 자동 설정도 유지할 수 있다.
- Intel GPU 분리는 실제 대상 장비에서 같은 `0.1.13` 번들로 device matrix와 parity를 다시 얻은
  뒤에만 성능 근거로 사용한다.

Windows 설치형이 이미 사용하는 detector `1×4`, GPU embedder 자동 및 명시적 OpenVINO CPU
fallback 계약과 일치하므로 제품 판정 동작이나 버전을 바꾸지 않았다.

## 동일 300장 실행 프로필 비교

동일한 packaged Worker와 multi-object 300장, 1,410개 ROI를 multipart HTTP로 실행했다. 모든 행의
결과는 `APPROVED 1,398`, `UNKNOWN 8`, `SEGMENT_RECAPTURE 4`, `ERROR 0`으로 같았다.

| Provider와 thread | Worker 평균 | Worker p95 | HTTP p95 | 판정 차이 |
|---|---:|---:|---:|---:|
| CPU, detector/embedder 자동 | 986.298ms | 2,093.327ms | 2,105.905ms | 0 |
| CPU, detector/embedder 4 | 196.004ms | 389.119ms | 408.571ms | 0 |
| OpenVINO CPU, detector/embedder 4 | 170.427ms | 327.759ms | 352.698ms | 0 |
| OpenVINO CPU, detector/embedder 자동, 1차 | 109.523ms | 183.315ms | 199.648ms | 기준 |
| OpenVINO CPU, detector/embedder 자동, 2차 | 108.739ms | 183.479ms | 201.373ms | 0 |
| OpenVINO CPU, detector 4/embedder 자동 | 110.288ms | 181.290ms | 200.032ms | 0 |

OpenVINO detector 4/embedder 자동은 CPU 4/4보다 Worker p95가 `53.4%` 낮다. OpenVINO 내부에서도
embedder 4 thread 고정보다 `44.7%` 낮다. 자동 프로필의 두 번 측정 p95는 `0.164ms` 차이여서
재현됐다.

CPU 4/4와 OpenVINO 자동의 상태, reason code, segmentation 수와 순서, bbox, item status, class,
Top-3와 버전 null 패턴은 300장 모두 일치했다. confidence 최대 차이는 `0.00090116`이며 명시적
수치 허용치 `0.001` 안이다. OpenVINO 자동 반복과 detector 4/embedder 자동 비교는 confidence까지
정확히 같았다. 이 허용치는 상태나 class 차이를 허용하지 않으며 provider 부동소수점 차이만
분리한다.

## 정확도 회귀

OpenVINO detector 1 worker, detector·embedder 자동 프로필로 추가 확인했다.

| 진단 세트 | GT/예측/매칭 | 정답 APPROVED | UNKNOWN | SEGMENT_RECAPTURE | FP/FN | 오승인 | Top-3 누락 | Worker p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 개발 회귀 415장 | 1,914/1,914/1,914 | 1,898 | 11 | 5 | 0/0 | 0 | 0 | 191.085ms |
| 운영 촬영 69장 | 138/138/138 | 138 | 0 | 0 | 0/0 | 0 | 0 | 143.720ms |

415장의 빈 이미지 4장과 운영 세트의 빈 이미지 6장은 모두 `IMAGE_RECAPTURE`였고, 객체가 있는
이미지의 잘못된 전체 재촬영은 없었다. 이 데이터는 개발과 정책 선택에 사용된 회귀 세트이므로
독립 일반화 성능이나 SLA 증거는 아니다.

## 0.1.12에서 가져오지 않은 최적화

`0.1.12`의 Detector 직접 승인 경로는 multi-object 300장에서 OpenVINO p95 `96.934ms`로 빨랐지만,
13개 SKU whitelist와 Detector class score를 최종 승인에 사용한다. 새 매장·새 상품마다 routing
분포가 달라지므로 운영 일관성 목표와 맞지 않아 가져오지 않았다.

이전 전역 후보도 다음 이유로 기각 상태를 유지한다.

- primary 160은 오승인 1건이 생겼고, 저점수 5개만 224로 재검증한 보완안은 첫 100장 p95가
  `380.679ms`로 악화됐다.
- INT8 PTQ는 ONNX Runtime과 NNCF 양쪽에서 정확도 또는 지연이 악화됐다.
- 승인 후보 전수 ViT는 오승인을 추가로 찾지 못하면서 정답 승인율을 `97.858%` 또는 `95.402%`로
  낮추고 p95를 `894.7~1,122.7ms`로 늘렸다.
- 이 개발 PC에는 지원되는 Intel GPU가 없어 `openvino+openvino_gpu` 초기화가 실패했다. 조용한
  CPU 대체로 성공처럼 기록하지 않았으며, N100에서는 제공된 device matrix 진단을 다시 실행해야
  한다.

## 평가 도구와 재현

`packaged_worker_http`에 provider 분리와 CPU thread 인자를 추가했다. 한 실행 보고서가 사용한
프로필을 `execution_profile`에 남기므로 서로 다른 장비 설정을 같은 결과로 혼동하지 않는다.

```powershell
python -m bixolon_scanner.evaluation.packaged_worker_http `
  --worker-artifact artifacts/handoff/n100-0.1.13-openvino-cpu-gpu-test/worker `
  --executable artifacts/handoff/n100-0.1.13-openvino-cpu-gpu-test/worker/bixolon-worker.exe `
  --runtime artifacts/handoff/n100-0.1.13-openvino-cpu-gpu-test/worker/model-package `
  --catalog artifacts/handoff/n100-0.1.13-openvino-cpu-gpu-test/worker/store-catalog `
  --dataset-root datasets/bread_dataset `
  --manifest manifests/bread-0.1.2-single3-detector415/detector_manifest.jsonl `
  --output artifacts/experiments/consistent-performance-0.1.13/openvino-default-detector415-http.json `
  --trace-output artifacts/experiments/consistent-performance-0.1.13/openvino-default-detector415-http-trace.jsonl `
  --store-id bread-dev --provider openvino --embedder-provider same `
  --cpu-detector-workers 1 --cpu-detector-intra-op-threads 0 `
  --cpu-embedder-intra-op-threads 0 --expected-version 0.1.13 `
  --expected-image-count 415 --expected-full-path-count 411 --warmup-count 20
```

`packaged_response_diff --confidence-tolerance 0.001`은 부동소수점 값 차이의 개수와 허용치 초과를
별도로 기록한다. 핵심 산출물은 `artifacts/experiments/consistent-performance-0.1.13`에 있다.
