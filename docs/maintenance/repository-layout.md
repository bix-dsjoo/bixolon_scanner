# 로컬 산출물 보존과 정리 기준

## 보존

- `configs/versions/0.1.6.json`: 유일한 활성 제품 설정
- `configs/archive/versions/0.1.5.json`: 이전 운영 설정 원문
- `artifacts/packages/bread-scanner-0.1.5-presence-consensus-runtime`: 0.1.6 source Runtime
- `artifacts/catalogs/bread-store-0.1.5-consensus`: 0.1.6 source Catalog
- `artifacts/versions/0.1.6`, `artifacts/installers/0.1.6`: 재생성 가능한 정식 배포 산출물
- `artifacts/evaluations/scanner-0.1.5`: 0.1.6에 고정된 모델·정책 회귀 원본
- `docs/diagnostics/n100-0.1.5-openvino-device-matrix.json`: 동일 source 후보의 N100 실측 원본

source artifact 경로의 `0.1.5`는 제품 표시 버전이 아니라 변경하지 않은 입력 payload provenance입니다.
배포 번들의 Runtime·Catalog metadata와 공개 API 버전은 모두 `0.1.6`으로 다시 작성됩니다.

## 정리 가능

빌드 환경, staging, PyInstaller·Flutter 중간 산출물과 실험 handoff ZIP은 위 source manifest,
평가 증빙, 최종 Setup/Worker ZIP의 SHA-256을 확인한 뒤 재생성 가능한 범위에서 정리할 수 있습니다.
대형 모델·데이터·빌드 산출물은 Git에 커밋하지 않습니다.

정리 전에는 `configs/versions/0.1.6.json`이 가리키는 Runtime, Catalog, CUDA와 평가 증빙이 실제로
존재하고 고정 해시와 일치하는지 확인합니다. N100은 제품명이 아니라 하드웨어 진단 provenance로만
유지합니다.
