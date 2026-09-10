# N100 1초 목표: 전체 파이프라인 실험 R2

2026-09-09. 정식 0.2.1 설치본과 별도로 실행하는 실험용 Worker다. 제품 버전 확정이나
N100 1초 달성 선언이 아니다. `D:\N100\4_EXPERIMENTS.cmd`가 별도 폴더의 새 EXE를 실행한다.
앱 재설치가 불필요한 이유는 설치된 앱을 시험하는 것이 아니라 이 Worker를 직접 시험하기 때문이다.

## 출발점과 범위

반환된 `N100-20260909-143527-afe437`은 HTTP 396회에서 p50/p95/p99
790.74/1004.74/1205.79ms, 최대 1430.45ms, 1초 이내 374/396회였다.
각 반복의 log132는 정답 승인 1078/1096, 오승인·미검출 0, UNKNOWN 15,
SEGMENT_RECAPTURE 19, 추가 검출 16이었다. N100은 4GB 메모리 장비다.

느린 이미지별 예외나 판정 임계값 조정을 적용하지 않는다. 다음 공통 실행 경로를 시험한다.

- Worker: 요청 안에서 같은 검증 모델·dtype·shape·입력 bytes의 임베딩 재사용.
  요청이 끝나면 제거하며 최대 16개, 입력 1MiB 이하 batch만 보관한다.
  오류를 캐시하지 않고, 출력 복사로 캐시 변조를 방지한다.
- 파이프라인 실행: GPU 회전 검증과 독립 CPU 검증을 동시에 계산한 뒤 두 결과를 모두 기다린다.
  검증 대상을 줄이거나 판정 순서를 바꾸지 않는다. GPU 실행 오류 시에도 CPU 작업 종료 후 복구한다.
- 모델: 정적 verifier 그래프의 상수 계산 정리, CPU MatMul 가중치 INT8 변환을 별도 후보로 만든다.
  INT8은 실제 가중치 표현 변경이므로 동일 가중치라고 설명하지 않는다.
- 장비 활용: GPU primary batch2, CPU 2/4스레드, CPU 전체 ROI batch,
  CPU ONNX Runtime 1.24.1/1.29.0을 비교한다.

## 설정과 fallback

`BIXOLON_REUSE_VERIFIER_EMBEDDINGS` 기본값은 true,
`BIXOLON_PARALLEL_VERIFICATION` 기본값은 false다.
`BIXOLON_VERIFIER_PROVIDER`는 same/cpu/openvino를 지원한다.
openvino는 CPU FP32 검증이며 GPU는 기존 openvino_gpu 설정이다.
동시 검증은 GPU primary와 독립 CPU verifier 조합에서만 활성화한다.
GPU 실패 후 native CPU로 복구할 때 verifier도 native CPU로 전환한다.
실패 요청은 ERROR이며 임의 승인이나 재촬영으로 변환하지 않는다.

별도 CPU 1.29 Worker를 이용하는 비교는 ORT DLL을 혼합하지 않는다.
GPU Worker의 ORT는 OpenVINO EP가 있는 1.24.1을 유지한다.
큰 모델을 USB에 여러 번 저장하지 않도록 공통 모델과 작은 변경 묶음을 사용한다.
측정 전에 로컬 디스크에 완전한 Runtime을 구성하고 Worker가 전체 checksum을 검증한다.
로컬 hard link를 사용할 수 없는 환경에서는 일반 파일 복사를 사용한다.

## 기각한 변경

CPU detector를 OpenVINO FP32로 바꾼 조합은 개발 PC에서 빨라졌으나 original302의
UNKNOWN 하나가 SEGMENT_RECAPTURE로 변해 재촬영이 4→5개로 증가했다.
정답 승인 646개와 오승인·미검출 0만 보고 채택하지 않았다.
전체 CPU OpenVINO 조합도 같은 문제가 있어 제외했다. 이 후보들은 N100 매트릭스에 없다.
이미지 241의 원본·GT·기존/후보 예측을 실패 비교 자료에 보존한다.

원본 verifier 그래프를 바로 OpenVINO CPU로 실행하면 dynamic-rank Reshape 로딩 오류가
발생했다. 상수 계산을 정리한 별도 그래프가 이 오류를 해결한다.
모델 원본과 정식 번들은 수정하지 않는다.

## 재현과 결과

생성: `scripts/prepare_n100_matrix_kit.py`.
분석: `scripts/analyze_n100_matrix.py --results <반환된 결과 폴더> --manifest <log-inputs.jsonl> --output <새 분석 폴더>`.
코드와 도구는 Python canonical 경로를 이용한다. Worker에는 학습·평가·양자화 의존성이 없다.

11개 설정 각각 warmup10회 뒤 132장 1회, 총 1452개 HTTP 요청을 측정한다.
각 설정은 새 프로세스에서 시작한다. 시작·컴파일 시간은 HTTP 시간과 별도로 기록하며,
오류와 1초 초과 요청을 통계에서 제외하지 않는다. 시작과 종료에 같은 기준 설정을 넣어
시간 경과에 따른 변화를 확인한다. 개별 후보 확인은 `--profile <ID> --repetitions 3`이다.

결과는 상태별 개수, GT 정답 승인·오승인·미검출·추가 검출, p50/p95/p99/max,
1초 이내 수, 실제 provider, 모델별 시간, 메모리를 포함한다.
동시 실행된 모델 시간의 합은 HTTP 경과시간과 같지 않다.
스모크 3장 결과를 정확도나 지연 성능 증거로 사용하지 않는다.

개발 PC 분석·검증 원문은 `artifacts/n100/optimization-r2`에 보존한다.
historical final300에는 기존 오승인 4개가 있으므로 전체 데이터에서 오승인 0이라고 표현하지 않는다.
N100 후보 채택과 1초 달성 여부는 반환된 실측과 GT 대조 이후에 결정한다.

## R2 N100 실측과 R3

2026-09-09 `N100-20260909-161411-94c149`의 11개 설정은 각각 로그 132장을 측정했다.
모두 정답 승인 1078/1096, 오승인 0, 미검출 0, APPROVED 1078 / UNKNOWN 15 /
SEGMENT_RECAPTURE 19, 추가 검출 16을 유지했다. batch2의 HTTP p50/p95/p99는
764.5/920.7/1107.6ms, 최대 1203.3ms, 130/132장이 1초 이내였다.
INT8 verifier는 778.7/935.3/1063.3ms, 최대 1128.3ms, 129/132장이 1초 이내였다.
전 요청 1초 목표는 미달이다. CPU 동적 batch와 ORT 1.29 변경은 N100에서 개선을 재현하지 못했다.

R3는 batch2와 INT8의 결합, 그리고 남은 ROI 수로 정적 graph를 선택하는 공용 실행 경로를
시험한다. `EmbedderMetadata.batch_variants`는 선택 필드이며 각 항목은 `filename`과
`batch_size`를 가진다. 주 graph의 `fixed_batch_size`보다 작은 중복 없는 크기만 허용하며
batch 1을 포함해야 한다. 모든 variant는 Runtime checksum 대상이다. 가장 큰 맞는 크기부터
실행하고 모든 실제 ROI 순서, TTA 평균, integrity 출력과 주변 객체 mask 문맥을 보존한다.
11개 ROI는 4+4+2+1, 단일 검증 ROI는 1로 실행한다. variant가 없는 기존 설정은 종전 padding을
유지한다. 각 session은 시작 시 warmup하고 종료·초기화 실패 때 함께 해제한다.
CPU fallback에도 같은 graph와 규칙을 적용한다. 추가 GPU session의 메모리는 N100에서 확인한다.

`scripts/prepare_n100_r3_kit.py`는 새 Worker와 6개 설정을 별도 패키지로 만든다.
기본 실행은 설정별 132장 3회, 총 2376회이며 시작/종료에 R2 batch2 기준을 반복한다.
R3 산출물과 검증은 `artifacts/n100/optimization-r3`에 보존한다.
실험 동안 정식 0.2.1 설치본과 모델 원본은 변경하지 않는다.

INT8는 FP32와 모든 개별 판정이 같지는 않다. historical final300에서 197·296번의
정답 승인 각 1개가 UNKNOWN으로, 227·230번의 UNKNOWN 각 1개가 정답 승인으로 바뀐다.
총 정답 승인 1356, 기존 오승인 4, 미검출 0은 같지만 이 상쇄를 판정 parity로 표현하지 않는다.
R3의 batch 선택은 R2 INT8와 개별 상태·bbox·Top-3를 유지한다.
