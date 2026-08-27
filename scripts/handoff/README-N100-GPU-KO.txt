BIXOLON Bakery AI Scanner 0.1.6 N100 OpenVINO CPU/GPU Embedder 비교
================================================

목적
----
- 0.1.6 Runtime, Catalog와 판정 정책은 변경하지 않습니다.
- OpenVINO CPU-only 기준을 같은 이미지로 다시 측정합니다.
- Detector는 양쪽 모두 OpenVINO CPU 1 worker x 4 threads입니다.
- CPU-only 기준의 전체 프레임 object_presence verifier는 OpenVINO CPU에서 실행합니다.
- 후보는 CPU Detector와 object_presence verifier를 병렬 실행하며, object_presence verifier와
  주 Embedder, 180도 회전 Embedder, 독립 ViT-B/16 검증 Embedder를 OpenVINO Intel GPU에서
  실행합니다. threshold 0.54, ONNX checksum, Catalog adapter와 최종 판정은 동일합니다.
- GPU가 전체 Embedder graph를 실행하지 못하면 CPU로 조용히 전환하지 않고 실패합니다.

실행 방법
---------
1. ZIP을 N100 PC의 로컬 디스크에 완전히 풉니다.
2. C:\easy 폴더에 실제 촬영 JPEG/PNG 이미지 30장 이상을 넣습니다.
   정상 SEGMENTATION 결과가 나오는 이미지가 10장 이상 필요합니다.
3. RUN-N100-GPU-TEST.cmd를 더블클릭합니다.
4. 완료 후 n100-0.1.6-openvino-device-matrix.json을 가져옵니다.

이미지가 다른 폴더에 있으면 그 폴더를 RUN-N100-GPU-TEST.cmd 위로 끌어다 놓아도 됩니다.

결과 해석
---------
- profiles.openvino_cpu_only는 Detector와 모든 Embedder를 OpenVINO CPU로 실행한 결과입니다.
- profiles.openvino_cpu_detector_intel_gpu_embedder는 Detector는 CPU, object_presence verifier와
  모든 Embedder는 Intel GPU로 실행한 결과입니다. Detector와 object_presence verifier는 병렬입니다.
- comparison.full_path_p95_speedup_ratio가 1보다 크면 하이브리드 p95가 더 빠릅니다.
- semantic_mismatch_count는 0이어야 합니다.
- maximum_confidence_delta는 0.00001 이하여야 합니다.
- 각 profile의 target.mean_within_target과 target.p95_within_target은 500ms 운영 진단 기준 판정입니다.
- recommended_provider가 openvino+openvino_gpu가 되려면 평균과 p95가 모두 5% 이상
  개선되고, 평균·p95 500ms 이내, 시작 30초/메모리 2GB/CPU 대비 메모리 증가 35%
  이내여야 합니다.
- 하나라도 만족하지 않으면 검증된 openvino CPU 구성을 유지합니다.

주의
----
- 이 패키지는 현장 진단용이며 설치 프로그램이나 배포 승격물이 아닙니다.
- 이미지 bytes와 로컬 경로는 결과 JSON에 기록하지 않고 동일 입력 확인용 SHA-256만 기록합니다.
- Intel 그래픽 드라이버와 OpenVINO GPU plugin 지원이 필요합니다.
- 1.9GB 표시는 전용 VRAM이 아니라 시스템 공유 메모리일 수 있습니다.
- 지연시간 또는 SLA를 보장하지 않습니다.
