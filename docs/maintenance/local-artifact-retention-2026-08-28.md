# 0.1.7 로컬 산출물 보존·정리 기록

`python -m bixolon_scanner.operations.local_artifacts --repository-root .`은 dry-run이 기본입니다.
실제 삭제는 동일 명령에 `--apply`를 명시한 경우에만 수행합니다. 도구는 `.git`, `pyproject.toml`,
활성 버전 설정을 저장소 표식으로 확인하고 workspace 밖 경로를 거부하며 `git clean`을 사용하지
않습니다.

## 보존 계약

- `datasets` 전체와 Git 추적 `manifests`, configs, archive 문서
- 활성·archive JSON 설정이 직접 참조하는 Runtime, Catalog, CUDA와 평가 증빙
- `artifacts/evaluations`, `artifacts/reports`, 실험 report/result/evidence/provenance/metadata
- 최종 `artifacts/versions/0.1.7`과 `artifacts/installers/0.1.7`

설정에 제공된 SHA-256은 로컬 manifest의 각 보존 항목에 함께 기록합니다. model graph·weight,
Catalog payload와 데이터셋은 정리 대상이 아닙니다.

## 삭제 계약

Python·Ruff·Flutter cache, 앱 중간 build, build environment, 임시 handoff, `runs`, 임시 weight,
과거 재생성 가능한 버전/설치 산출물, vendor clone과 보존 근거가 없는 실험 ONNX/NPY/checkpoint/
cache/candidate binary만 대상으로 합니다. 삭제된 중간 파일은 Git으로 복구되지 않으며 코드·설정·
데이터로 재생성합니다.

## 실행 결과

- dry-run manifest:
  `artifacts/reports/maintenance/local-artifact-cleanup-0.1.7-dry-run.json`
- apply manifest:
  `artifacts/reports/maintenance/local-artifact-cleanup-0.1.7-applied.json`
- 계획: 57,663개 파일, 471,322,707,927 bytes(438.953 GiB)
- 본 정리 실제 확보: 471,280,209,920 bytes(438.914 GiB), 오류 0건
- post-check cache 12개 추가 정리: 192,512 bytes
- 합계 manifest 기준 삭제: 471,322,904,578 bytes, 디스크 여유 공간 실측 증가는
  471,280,402,432 bytes(438.914 GiB)

정리 뒤 `datasets` 1,268개 파일, `artifacts/evaluations/scanner-0.1.7`, 최종 번들·설치물이 남아
있음을 확인했습니다. `bixolon bundle verify`는 bundle manifest
`9936d98fcdd874857fe99895c50999f4f67914928a8c7621c19d56496581731c`로 통과했습니다. 최종 Setup
SHA-256은 `dd096fb2deb9d66a657c5395ab6a3abbc2c7688fde2cfbb1e0a9fa31f3a54c44`, Worker ZIP은
`b2472f459f4a660cad9aab0f3ecb6c99dd771aa4f4d3c95d083e68961c2ccb45`입니다.

삭제된 재생성 가능 중간 파일은 Git으로 복구되지 않습니다. 보존 자료와 개별 삭제 후보·사유,
설정에 제공된 SHA-256은 위 로컬 manifest에서 확인할 수 있습니다.
