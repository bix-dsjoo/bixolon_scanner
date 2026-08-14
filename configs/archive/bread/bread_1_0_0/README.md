# Bread 1.0.0 configuration archive

- `bread_10shot_1.0.0.json`: 과거 10-shot 비교 구성이다.
- `bread_official_1.0.0.json`: release composition 도입 전 통합 experiment 구성이다.
- `detector_12shot_rejected.json`: `single_objects_3` 12-shot native Detector 재학습의 KPI 실패와 미승격 증거다.
- `detector_12shot_native_evidence.json`: 거절 후보의 native stage hash-chain이다.
- `dfine_detector_12shot_rejected.yml`: 거절 후보의 D-FINE 학습 overlay다.

두 파일은 재현 참고용이며 active CLI 또는 운영 승격 입력으로 사용하지 않는다. 정식
조합 버전과 artifact lock은 `configs/releases/bixolon_scanner_1.0.0.json`에서 관리한다.
