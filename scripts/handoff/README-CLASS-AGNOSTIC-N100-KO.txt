BIXOLON class-agnostic Detector 0.1.11 후보 N100 CPU/iGPU 비교
================================================================

목적
----
- SKU와 독립적인 1-class bread/object SSDLite320 Detector 후보를 실제 Intel N100에서 측정합니다.
- 같은 Worker, Runtime, Catalog로 CPU-only와 Intel GPU Classifier 구성을 비교합니다.
- 전체 경로 평균과 p95가 1,000ms 이하인지 확인합니다.
- 두 provider의 최종 상태, reason, bbox와 class rank가 같은지 확인합니다.

실행 방법
---------
1. ZIP을 N100 PC의 로컬 디스크에 완전히 풉니다.
2. C:\easy에 기존 진단과 같은 JPEG/PNG 100장을 넣습니다.
3. RUN-CLASS-AGNOSTIC-N100-GPU-TEST.cmd를 더블클릭합니다.
4. 완료 후 n100-class-agnostic-0.1.11-openvino-device-matrix.json을 가져옵니다.

이미지가 다른 폴더에 있으면 그 폴더를 CMD 파일 위로 끌어다 놓아도 됩니다.

합격 조건
---------
- N100 CPU와 Intel GPU가 감지됩니다.
- 실행된 profile의 ERROR가 0입니다.
- CPU/iGPU 최종 의미 parity와 confidence 오차 0.02 조건을 통과합니다.
- 선택할 profile의 full-path 평균과 p95가 모두 1,000ms 이하입니다.

주의
----
- 이 패키지는 현장 진단용이며 설치 프로그램이나 활성 배포물이 아닙니다.
- 이미지 bytes와 로컬 경로는 결과 JSON에 남기지 않고 SHA-256만 기록합니다.
- GPU 초기화 또는 graph 실행 실패를 CPU fallback으로 숨기지 않습니다.
- 측정 결과는 SLA, 인증 또는 독립 일반화 성능을 의미하지 않습니다.
