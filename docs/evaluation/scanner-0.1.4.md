# BIXOLON Scanner 0.1.4 개발 검증

`0.1.4`는 겹침이 심해 detector가 객체 경계를 합치거나 일부 객체를 누락하는 장면을 이미지 전체
재촬영으로 조기 종료합니다. 겹침 자체가 아니라 detector 결과의 불완전 가능성을 판정합니다. ONNX
graph, weight, Classifier와 Catalog payload는 `0.1.3`에서 변경하지 않고 Runtime metadata와 후처리
정책만 변경했습니다.

## 정책

한 번의 detector 추론에서 raw query를 score `0.05`까지 관찰합니다.

- score `0.145` 이상 NMS 제안 중 최대 면적 비율이 `0.21` 이상이면 재촬영합니다.
- 그렇지 않으면 NMS 박스의 최소 정규화 중심 거리가 `0.48` 이하이면서 IoU `0.7` query cluster의
  중복률이 `0.93` 이상일 때만 재촬영합니다.
- 위 조건이 정상이고 이미지 가로/세로 비율이 `1.0` 이상이며 기본 검출이 정확히 5개일 때 후보가
  됩니다. 기본 bbox 합산 면적 비율 `0.4` 이상과 최소 정규화 중심 거리 `0.63` 이상도 요구해 불필요한
  회전 실행을 제한합니다.
- 후보 장면만 90°와 180° 입력을 순서대로 검사합니다.
- 회전 검출 수가 기본보다 하나 이상 많고, 기본 bbox 전부가 IoU `0.5` 이상인 서로 다른 회전 bbox와
  대응할 때만 누락 가능성으로 재촬영합니다.
- 조건은 Runtime `detector_crowding` metadata에서 읽으며 코드 기본값으로 활성화하지 않습니다.
- 조건이 맞으면 `DETECTOR_UNCERTAIN_OBJECT` hard gate를 사용하므로 classifier를 호출하지 않습니다.

단순 bbox 겹침·근접 또는 회전 검출 수만 사용하면 정상 장면도 거부할 수 있습니다. 따라서 raw query
조건에는 중복률을, 회전 조건에는 기존 bbox 전체의 일대일 대응을 함께 요구합니다. 회전해도 객체 수가
늘지 않으면 겹침이 있어도 분류를 계속합니다.

## Worker 실행 최적화

모델 graph, weight, threshold와 판정 정책을 바꾸지 않고 다음 실행 경로를 최적화했습니다.

- RGB 입력은 HWC `float32` 중간 배열을 만들지 않고 NCHW `float32` 연속 배열로 한 번만 변환합니다.
- 고정 shape 단일 CUDA detector는 CUDA graph와 고정 I/O binding을 재사용합니다.
- 독립 classifier verifier는 전체 detection을 이웃 mask 문맥으로 유지하되 실제 검증 후보 ROI만
  crop·resize·mask 처리합니다.
- 180° detector 복구는 원본 640×640 detector tensor의 정확한 180° 회전 결과를 재사용합니다.

100장 trace에서 상태, reason code, bbox, 객체 상태, prediction과 Top-3 변경은 0건이었고 detector
415장 회귀의 추가 재촬영도 0건이었습니다.

## 결과

| 표본 | 결과 |
|---|---|
| 사용자 제공 겹침 이미지 | 5/5 `IMAGE_RECAPTURE` 경로 |
| 운영 검수 표본 | 확정 uncertain 16/16 탐지 |
| 정상 통과 확정 표본 | `054`, `087` 추가 재촬영 0/2 |
| 운영 미확정 표본 | `094`, `095` 재촬영; 오재촬영 분모에서 제외 |
| accepted detector 개발 회귀 | 추가 재촬영 0/415 |
| 운영 100장 full pipeline | `SEGMENTATION` 73, `IMAGE_RECAPTURE` 27, `ERROR` 0 |
| 확인 요청 표본 | `072`, `077`, `078` 재촬영; `054`, `087` 정상 분류 |
| detector ONNX SHA-256 | `800299fad5dee3389c93f7f00477a851557893f67e514f4f2977691d91de17e5` |

| CUDA 100장 속도 | 0.1.3 기준 | 최적화 전 0.1.4 | 최종 0.1.4 | 최적화 전 대비 |
|---|---:|---:|---:|---:|
| 평균 | 60.30ms | 61.59ms | 54.03ms | -12.3% |
| p50 | 50.30ms | 51.03ms | 46.86ms | -8.2% |
| p95 | 101.08ms | 106.63ms | 86.43ms | -18.9% |
| p99 | 106.28ms | 111.19ms | 96.53ms | -13.2% |

실제 Intel N100 100장 OpenVINO 진단 결과는 다음과 같습니다.

| N100 full-path | CPU-only | CPU Detector + Intel GPU Embedder | GPU speedup |
|---|---:|---:|---:|
| 평균 | 575.547ms | 438.422ms | 1.313× |
| p50 | 540.965ms | 427.174ms | - |
| p95 | 936.381ms | 727.238ms | 1.288× |
| p99 | 1,072.344ms | 830.369ms | - |
| 시작 | 10,633.835ms | 12,181.455ms | - |
| peak working set | 1,394,323,456 bytes | 1,933,832,192 bytes | - |

`0.1.3`과 같은 N100 100장 결과를 비교하면 CPU-only 평균/p95는 각각 1.75%/3.10%, 하이브리드
평균/p95는 각각 1.58%/3.23% 줄었습니다. 하이브리드 p50은 417.517ms에서 427.174ms로 2.31%
늘었습니다. 따라서 N100 개선 폭은 작으며, Worker 경로 최적화보다 모델 추론 시간이 전체 지연의
대부분을 차지하는 장비 특성이 그대로 남아 있습니다.

두 profile 모두 오류 0건이고 100장 최종 semantic mismatch는 0건입니다. 그러나 최대 confidence 차이
`0.0085055232`가 진단 허용치 `0.00001`을 초과해 `parity.safe=false`이며, 하이브리드 peak memory도
선택 기준을 초과했습니다. 앞으로의 N100 운영 진단 기준은 평균과 p95 모두 `500ms 이하`입니다.
하이브리드 평균 438.422ms는 새 기준을 충족하지만 p95 727.238ms는 초과하므로, 기존 실측을 새
기준으로 다시 보아도 `passes=false`, `recommended_provider=openvino`입니다. 배포 산출물은 GPU
Embedder 우선과 GPU 초기화 실패 시
CPU Embedder fallback을 포함하지만, 이 진단은 성능·자원·confidence parity 승인이나 SLA를
제공하지 않습니다. 원본 결과는 `docs/diagnostics/n100-0.1.4-openvino-device-matrix.json`이며
SHA-256은 `1cbbe83b9a58db5b0d4c5be6a36d458555cf8393cb7cfb42339c7f2992cdefba`입니다.

원본 device matrix JSON은 측정 당시 적용된 300ms 선택 정책과 해시를 보존합니다. 이후 생성되는
N100 CPU 및 CPU/GPU 비교 진단은 평균·p95 500ms 기준을 기록합니다.

고정 정책 진단은 `artifacts/evaluations/scanner-0.1.4/detector-crowding-policy-targeted-rotation-v2.json`,
binary 불변 변환 기록은 `runtime-targeted-rotation-v2-conversion.json`입니다. 운영 full pipeline 결과와
객체별 trace는 각각 `bread-project-4-100-targeted-rotation-v2.json`,
`bread-project-4-100-targeted-rotation-v2-trace.jsonl`입니다. 변환 전후 detector, embedder, verifier와 license
payload의 SHA-256 집합이 동일함을 확인했습니다. 정책 진단과 full pipeline 모두 Worker의 JPEG draft
decode 및 EXIF 처리 경로를 사용했습니다.

운영 100장 중 재촬영·정상 통과 어느 쪽으로도 확정하지 않은 행은 false-recapture 분모에서
제외했습니다. full pipeline의 draft annotation도 최종 정답이 아니므로 객체 정확도 수치는 정책 채택
근거로 사용하지 않습니다. 이 표본과 415장 회귀는 정책 선택에 사용한 같은 도메인의 개발 자료이며
독립 test가 아닙니다. 제공되지 않은 겹침 형태의 recall, 외부 매장·카메라 일반화와 장기 신뢰성을
입증하지 않습니다. N100 수치는 제공된 한 장비의 진단이며 지연시간 또는 SLA 목표를 충족했다는
의미가 아닙니다.
