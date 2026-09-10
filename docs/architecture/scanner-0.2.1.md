# 0.2.1 최종 구성과 변경점

2026-09-10 N100 실측 후 RepViT GPU 구성을 최종 선택했다. 0.2.1 개발 중 실험을
완료한 것이므로 최종 제품 버전은0.2.1, Flutter 내부 build24로 유지한다.

- Detector: 기존 D-FINE-S640 ONNX·가중치 유지, CPU4스레드.
- Primary: DINOv3 ConvNeXt-Tiny192 → RepViT-M0.9 192 supervised 학습 모델.
  모든 출력 ROI를 GPU FP16 정적 batch2/1로 분류한다. ONNX 가중치는 FP32다.
- Primary·회전 Catalog: 원래 support 이미지에서 RepViT 특징을 재계산한다.
- Detail: 기존 DINOv3 ConvNeXt-Tiny224와 전용 DINO Catalog로 선택 ROI만 재분류한다.
- 독립 verifier: 기존 Frozen DINOv3 ViT-B/16 160 CPU. 검증 범위·합의 정책은 유지한다.
- Worker: 별도 detail Catalog의 경로·checksum·특징 공간·상품 순서·공통 verifier를 검증한다.
  N100에서 검증한 EXE를 최종 N100 배포물에 바이트 동일하게 포함한다.
- N100 앱·실행문: 실측과 동일한 CPU4스레드, GPU FP16, 입력 재사용·병렬 검증으로 연결한다.
  Lite 앱과 독립 Worker CMD 실행에 PowerShell/Python 설치가 필요하지 않다.

입력검증 → 검출·프레임 조기종료 → primary·선택적 detail/검증 → 최종 상태의 순서,
공개 API·reason code, 검출 candidate context0.025/출력0.04, 승인·재촬영 정책은 유지한다.
ERROR는 재촬영으로 바꾸지 않는다. 객체를 삭제해 정확도를 올리지 않는다.
모델 점수가 달라져 detail/verifier로 라우팅되는 ROI는 달라질 수 있다.
별도 detail Catalog의 metadata 버전과 Runtime의 checksum 연결도 번들 변환 시 함께 검증한다.

소형 detector, detector 특징공유, INT8/PTQ/QAT 후보는 정확도·속도 검증 후 제외했다.
N100 실측은 GPU396/396건≤1초, 정답승인1085/1096·오승인0·미검출0이다.
남은 재촬영18개·추가검출16개, 개별 승인손실2개, CPU1초초과4건, GPU최악950ms와
별도 final300 기존 오승인4개·독립 validation 부재를 공개한다.

[N100 결과와 사용법](../experiments/n100-0.2.1.md),
[개발 실험](../operations/n100-structural-experiments.md),
[이전 개발 구성 보관](../archive/repvit-final-20260910/docs--architecture--scanner-0.2.1.md.txt).
