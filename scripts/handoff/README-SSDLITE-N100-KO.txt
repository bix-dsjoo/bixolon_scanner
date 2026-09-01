BIXOLON SSDLite 0.1.7 후보 N100 CPU/iGPU 비교
================================================

목적
----
- YOLO와 D-FINE ensemble을 제거한 단일 SSDLite320 detector 후보를 실제 Intel N100에서 측정합니다.
- 정확히 같은 Worker, Runtime, Catalog와 판정 정책으로 CPU-only와 Intel GPU classifier 구성을 비교합니다.
- 전체 경로 평균과 p95가 1,000ms 이하인지 확인합니다.
- 두 provider의 최종 상태, bbox, class rank를 정확히 비교하고 confidence parity는 0.02 이내인지 확인합니다.

실행 방법
---------
1. ZIP을 N100 PC의 로컬 디스크에 완전히 풉니다.
2. C:\easy에 실제 촬영 JPEG/PNG 이미지 30장 이상을 넣습니다.
   SEGMENTATION 결과가 나오는 이미지가 10장 이상 필요합니다.
3. RUN-SSDLITE-N100-GPU-TEST.cmd를 더블클릭합니다.
4. 완료 후 n100-ssdlite-0.1.7-openvino-device-matrix.json을 가져옵니다.

이미지가 다른 폴더에 있으면 그 폴더를 CMD 파일 위로 끌어다 놓아도 됩니다.

합격 조건
---------
- hardware.target_cpu_detected와 target_intel_gpu_detected가 true입니다.
- 실행된 profile의 error_count가 0입니다.
- CPU/iGPU를 모두 실행하면 parity.safe가 true입니다.
- comparison.target_met_by_any_profile이 true입니다.
- 선택할 profile의 full-path 평균과 p95가 모두 1,000ms 이하입니다.

주의
----
- 이 패키지는 현장 진단용이며 설치 프로그램 또는 배포 승격물이 아닙니다.
- 이미지 bytes와 로컬 경로는 결과 JSON에 남기지 않고 SHA-256만 기록합니다.
- Intel 그래픽 드라이버와 OpenVINO GPU plugin이 필요합니다.
- GPU 초기화나 graph 실행 실패를 CPU fallback으로 숨기지 않습니다.
- Intel GPU 수치 오차 허용치 0.02는 상태·class rank가 정확히 같은 경우에만 적용합니다.
- GPU Worker peak working set 상한은 2.25GiB이고 CPU 대비 증가율 1.35배도 동시에 지켜야 합니다.
- 측정 결과는 SLA, 인증 또는 독립 일반화 성능을 의미하지 않습니다.
