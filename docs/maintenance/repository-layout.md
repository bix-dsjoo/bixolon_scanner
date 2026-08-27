# 저장소와 로컬 산출물 관리

## Git에 보존하는 것

- `src/bixolon_scanner`: Python canonical 구현
- `apps/product_scanner`: Flutter canonical 구현과 골든 테스트
- `configs/versions/0.1.5.json`: 유일한 활성 제품 설정
- `configs/experiments`: 재현 가능한 실험 설정
- `configs/archive`, `docs/archive`: 과거 설정·계약·판단 원문
- `docs/evaluation`, `docs/diagnostics`: 작은 JSON/Markdown 평가 증빙과 한계
- `tests`: 운영 계약과 과거 호환 reader 회귀

## Git 밖에 보존하는 것

- `datasets`: 원본 데이터와 촬영 provenance
- `artifacts/experiments`, `runs`: checkpoint, ONNX, 예측, trace와 실험 보고서
- `artifacts/evaluations`, `artifacts/reports`: 평가 결과와 오류 분석
- `artifacts/packages/bread-scanner-0.1.5-large-proposal-corroboration-runtime`: 활성 Runtime 원본
- `artifacts/catalogs/bread-store-0.1.5-consensus`: 활성 Catalog 원본
- `artifacts/versions/0.1.5`, `artifacts/installers/0.1.5`, `artifacts/handoff/0.1.5`: 재생성 가능한
  운영 전달물

대형 binary는 Git에 커밋하지 않습니다. 버전 설정과 평가 문서에는 원본 경로, SHA-256, 데이터
범위와 독립 test set 여부를 기록합니다. 실험 결과를 정리할 때는 보고서·trace·provenance와 그
결과가 참조하는 checkpoint/ONNX를 함께 보존합니다.

## 삭제해도 되는 생성물

`build`, `tmp`, `.pytest_cache`, `.ruff_cache`, `.codex_tmp`, Flutter `build`와 `.dart_tool`,
`artifacts/cache`, `artifacts/tmp`, `artifacts/build-envs`는 다시 만들 수 있는 캐시·중간물입니다.
다른 제품 버전의 `artifacts/versions`, `installers`, `handoff`와 중복 압축 해제 디렉터리도 현재
`0.1.5` 운영에는 사용하지 않습니다.

정리 전에는 `configs/versions/0.1.5.json`이 가리키는 Runtime, Catalog, CUDA와 평가 증빙이 실제로
존재하고 고정 SHA-256과 일치하는지 확인합니다. `artifacts/experiments`, `artifacts/evaluations`,
`artifacts/reports`, `datasets`, `runs`는 일반 캐시 정리 대상으로 취급하지 않습니다.

## 버전 변경 순서

1. 배포할 실행 내용이 바뀔 때만 새 patch 버전과 Flutter build 번호를 정합니다.
2. 새 `configs/versions/<version>.json`에 원본과 평가 증빙 hash를 고정합니다.
3. Python, Flutter, 문서, 예시와 테스트의 공개 버전을 같은 변경에서 맞춥니다.
4. 전체 테스트 뒤 Windows 번들을 실제 생성하고 `bixolon bundle verify`로 검증합니다.
5. 이전 활성 설정과 계약은 `archive`로 이동하고 실험 결과는 삭제하지 않습니다.
