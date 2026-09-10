# BIXOLON Bakery AI Scanner 현재 상태

활성 제품은 `0.2.0`, Flutter 앱은 `0.2.0+23`, SDK는1.2.0입니다.
선택 모델은 `dfine-margin-dense-20260908-local-recapture`입니다.
이전 `ssdlite-margin-dense-20260908`의 분류기·Catalog를 유지하고 검출기·전역 검출 정책을 변경했습니다.

로그132장의 모델 학습 미사용 개발 진단에서 정답1,078/1,096(98.36%), 오승인0, 미검출0입니다.
추가 박스37개와 UNKNOWN17·SEGMENT_RECAPTURE38이 있으며 이미지 완전 성공은99/132입니다.
CUDA 소스3회 p95는84~85ms, CPU는200ms 미달입니다. N100 장비는 연결되지 않아 실측하지 않았습니다.
로그는 이번 정책 선택에 사용했으므로 독립 일반화 검증이 아닙니다.

[버전 설정](../../configs/versions/0.2.0.json), [변경과 한계](../architecture/scanner-0.2.0.md),
[실험 기록](../experiments/n100-0.2.0.md)을 참조하십시오.
