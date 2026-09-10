# N100 1초 목표 최적화 실험

2026-09-09에 받은 실제 N100 결과에서 396건의 HTTP p50/p95/p99는
1,236.50 / 10,532.82 / 19,074.87ms, 최대 29,982.58ms였다. 1초 이내는 50건이었다.
목표는 HTTP 1초 이내이며 p95만 낮아졌다고 전체 요청이 목표를 달성한 것으로 표시하지 않는다.
0.2.0 재측정은 하지 않는다.

## 변경과 확인 범위

- CPU 검출기와 모든 판정 임계값, 선택적 상세 분류·회전·독립 검증 조건은 유지한다.
- primary·detail·verifier의 ONNX batch shape를 1로 특수화한 별도 후보를 만든다.
  모델 연산과 모든 initializer의 직렬화 bytes가 동일한지 확인한다. 원본 모델은 수정하지 않는다.
- GPU primary/detail에만 명시적으로 `f16` 실행 힌트를 적용한다. ONNX 가중치는 FP32 그대로다.
  CPU detector·CPU verifier·CPU fallback은 FP32를 유지한다.
- 전체 검출은 이웃 마스크 문맥에 계속 포함한다. 모든 선택 ROI를 처리하며 객체를 삭제하지 않는다.
  고정 batch 호출로 나눠도 ROI 순서와 최종 판정 로직은 같다.
- CPU 검출 스레드 4, CPU verifier 스레드 4를 시험한다. GPU는 LATENCY, stream 1이다.
- `BIXOLON_LOG_MODEL_TIMINGS=true`로 모델별 실행 시간·실제 provider·batch 크기·성공 여부를
  같은 `request_id`에 기록한다. 이미지, tensor, 전체 logits, 입력 경로는 기록하지 않는다.
- 측정기는 HTTP 요청 종료 후 Worker working set·private bytes·page fault 누적값과 시스템 전체·가용 RAM을 기록한다.
  계측 호출 자체는 HTTP 시간에 포함하지 않는다. page fault는 hard fault와 동의어가 아니다.
- USB에서 실행하더라도 모델과 Worker를 `%LOCALAPPDATA%\BixolonN100Benchmark`의 checksum별
  캐시에 복사·검증해 실행한다. 복사와 준비 시간은 HTTP 측정에서 제외한다.
  캐시는 시험 종료 후 사용자가 해당 폴더를 삭제해 정리할 수 있다.

기본 설정은 여전히 GPU `f32`, 모델별 세부 계측 비활성이다. CPU fallback은 명시된 설정으로
유지하고 실패한 요청은 `ERROR`를 반환한다. 장애 요청을 승인이나 재촬영으로 변환하지 않는다.

## 실험 결과와 해석

개발 PC(Core Ultra 9 285K)의 Intel GPU에서 동적 모델은 OpenCL 실행 오류가 발생했지만
고정 shape 모델은 실행됐다. 이는 고정 shape 후보를 평가할 근거이며 N100의 10초 지연이
같은 원인이라는 증명은 아니다. 실제 N100의 이전 로그에는 모델별 세부 시간과 memory가 없다.

고정 batch 1 + GPU FP16 후보의 개발 로그132장은 정답 승인 1,078/1,096, 오승인0, 미검출0,
UNKNOWN15, SEGMENT_RECAPTURE19, 추가 검출16이다. 원본302장도 정답 승인646/649,
오승인0, 미검출0, 추가 검출5를 유지했다. 후보를 고정한 뒤 시행한 과거 final300 진단은
정답 승인1,356/1,410, 기존 오승인4, 미검출0을 유지했다. final300의 기존 오류를 숨기지 않는다.
final300 이미지134의 UNKNOWN Top-3 세 번째 후보는 CPU와 차이가 있으나 기존 CUDA 결과와 같다.
로그132장과 원본302장은 상태·reason·상품 순위가 모두 기존 결과와 같다.

CPU 검출을 OpenVINO로 바꾼 결합 후보는 로그 정답 승인1,077로 악화되어 제외했다.
batch4 후보도 판정 기준은 통과했지만 개발 PC p95와 불필요한 padding을 고려해 batch1을 선택했다.
FP16에 대한 다른 데이터·장비의 결과까지 보장하지 않으며 N100 반환 결과를 다시 GT와 대조한다.

## 실행

`D:\N100\2_MEASURE.cmd`에서 실행할 최적화 키트는 실험용 Worker다.
PowerShell·Python 설치가 필요 없다. Scanner Lite와 다른 Worker를 닫고 실행한다.
파일 검증 → 로컬 복사 → GPU 준비 → warmup10회 → 로그132장 3회 → 결과 ZIP 순서다.
실제 provider, 모든 응답, p50/p95/p99/max, 1초 초과 개수와 모델별 로그를 보존한다.
기존 0.2.1 설치 배포물은 그대로 보관하며 이 실험을 새 제품 출시나 N100 목표 달성으로 부르지 않는다.
N100에서 검증이 끝나 배포할 앱·Worker를 새로 만들 때 단일 제품 patch 버전을 올린다.

재현 함수는 `operations.static_batch_candidate.specialize_runtime`이며 목적지는 새 디렉터리여야 한다.
실험 산출물과 원본 결과는 `artifacts/n100/optimization`, `artifacts/n100/0.2.1/target-analysis`에 보존한다.

동적 shape의 GPU 커널 컴파일·메모리 특성은
[OpenVINO GPU 문서](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html),
정밀도 힌트는 [ONNX Runtime OpenVINO provider 문서](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)를 참고했다.
