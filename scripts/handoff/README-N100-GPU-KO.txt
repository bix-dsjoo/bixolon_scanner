BIXOLON Bakery AI Scanner 0.1.12 N100 OpenVINO CPU/GPU classifier 비교
===========================================================

목적
----
- 0.1.12 Detector-first SSDLite Runtime, Catalog와 판정 정책을 변경하지 않습니다.
- 같은 이미지로 OpenVINO CPU-only와 CPU Detector + Intel GPU classifier를 비교합니다.
- Detector는 양쪽 모두 OpenVINO CPU 1 worker x 4 threads입니다.
- count verifier와 object_presence verifier는 구성하지 않습니다.
- 후보는 주 Embedder, 180도 Embedder와 독립 ViT-B/16 검증 Embedder를 Intel GPU에서 실행합니다.
- GPU가 전체 classifier graph를 실행하지 못하면 CPU로 조용히 전환하지 않고 실패합니다.

실행 방법
---------
1. ZIP을 N100 PC의 로컬 디스크에 완전히 풉니다.
2. C:\easy 폴더에 실제 촬영 JPEG/PNG 이미지 30장 이상을 넣습니다.
   정상 SEGMENTATION 결과가 나오는 이미지가 10장 이상 필요합니다.
3. RUN-N100-GPU-TEST.cmd를 더블클릭합니다.
4. 완료 후 n100-0.1.12-openvino-device-matrix.json을 가져옵니다.

이미지가 다른 폴더에 있으면 그 폴더를 RUN-N100-GPU-TEST.cmd 위로 끌어다 놓아도 됩니다.

결과 해석
---------
- 두 profile의 최종 상태, bbox와 class rank는 정확히 같아야 합니다.
- confidence 최대 차이는 0.02 이하여야 합니다.
- 평균과 p95 1,000ms 이하인 profile이 하나 이상 있어야 합니다.
- GPU profile이 CPU보다 평균과 p95 모두 5% 이상 빠르면 GPU 구성을 우선 추천합니다.
- peak working set 절대 상한은 2.25GiB, CPU 대비 증가 상한은 35%입니다.

주의
----
- 이 패키지는 현장 진단용이며 설치 프로그램이 아닙니다.
- 이미지 bytes와 로컬 경로는 결과 JSON에 기록하지 않고 동일 입력 확인용 SHA-256만 기록합니다.
- Intel 그래픽 드라이버와 OpenVINO GPU plugin 지원이 필요합니다.
- 지연시간 또는 SLA를 보장하지 않습니다.
