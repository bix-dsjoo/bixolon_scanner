# Bread 1.0.0 probe archive

이 디렉터리는 Bread 1.0.0 모델 선택 과정에서 사용한 일회성 `*_probe.py`를 보존한다.
운영 CLI와 package 생성은 이 모듈에 의존하지 않으며, 새 기능은 `training` 또는
`evaluation`의 canonical 모듈에 구현한다.

과거 10-shot과 통합 experiment 설정은
`configs/archive/bread/bread_1_0_0`에 함께 보존한다. 정식 release 조합은
`configs/releases/bixolon_scanner_1.0.0.json`이 소유한다.
