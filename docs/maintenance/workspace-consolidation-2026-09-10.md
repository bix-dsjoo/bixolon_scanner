# main 통합과 작업 폴더 정리

2026-09-10 사용자 요청으로 main의 진행 작업과 별도 release 폴더를 통합했다.
현재 제품은 `0.2.1`, Flutter build는 `0.2.1+24`이며 새 제품 버전이나 모델을 만들지 않는다.
현재 작업 기준은 저장소 루트 하나다. [현재 상태](../status/current.md),
[RepViT 최종 구성](../architecture/scanner-0.2.1.md),
[과거 버전 인덱스](../archive/version-history.md)를 함께 참조한다.

## 소스 보존

네 release 작업트리는 모두 `ba59cfa`에서 시작했지만 이후 변경을 커밋하지 않은 상태였다.
`0.1.19`는 Git에 연결되지 않은 부분 소스 폴더였다. 삭제 전 여섯 폴더의 소스 파일을
SHA-256으로 기록하고 ZIP 스냅샷을 만든 뒤 아래 순서로 main에 커밋했다.

| 보존 대상 | 커밋 | 소스 파일 수 |
|---|---|---:|
| main 진행 작업: 세 매장 학습·ROI 무결성·Lite 승인 상품 표시 | `4306123` | 1,487 |
| release 0.1.17 | `0b79e69` | 1,502 |
| release 0.1.18 | `06f7270` | 1,532 |
| release 0.1.19 부분 소스 | `1294298` | 1,205 |
| release 0.2.0 | `8e03793` | 1,579 |
| release 0.2.1 최종 RepViT 구성 | `d263adf` | 1,658 |

누적 소스는 버전 순서대로 반영했고, 다른 폴더에만 있는 파일은 통합 중 유지했다.
이전 버전의 변경 내용은 해당 커밋에서 확인할 수 있다. `0.1.19`의 생성된
`.flutter-plugins-dependencies`는 소스에 포함하지 않았다. main에만 있던 Lite 승인 상품
Windows 통합 테스트도 유지했다. 단계별 커밋은 작업 이력 보존이며, 각 과거 버전을 새로
빌드하거나 검증했다는 뜻은 아니다. 검증 대상은 최종 통합 트리다.

로컬 원본 스냅샷과 파일별 해시, 비교 결과, 검증 로그는 Git에서 제외되는
`artifacts/workspace-consolidation/20260910`에 보존한다. Git 복원은
`git show <commit>:<path>` 또는 별도 디렉터리로 `git archive <commit>`를 사용한다.
작업 중인 main에 과거 소스를 직접 덮어쓰지 않는다.

## 활성 파일과 과거 기록

- 활성 버전 설정은 `configs/versions/0.2.1.json` 하나만 유지한다.
- 이전 버전 설정은 `configs/archive/versions`, 연동 명세와 예시는
  `docs/archive/contracts`에 보존하고 활성 경로의 중복을 제거한다.
- 새로 발견한 0.1.19 설정도 archive에 보존한다.
- 학습·평가·실험 코드와 재현 스크립트는 유지한다. 과거 버전 이름이 포함되었다는 이유로
  실행 코드나 실험 증거를 삭제하지 않는다.
- 제거할 release 폴더를 참조하던 두 N100 실험 설정은 저장소 기준 상대 경로로 바꾼다.
- 과거 `bread-10shot-0.2.1`은 현재 제품 0.2.1과 별개의 실험이다.

## 로컬 파일 보존과 제거

`C:/workspace`의 `bixolon_scanner_release_0_1_17`, `0_1_18`, `0_1_19`,
`0_2_0`, `0_2_1` 접미사 폴더만 이번 정리 대상으로 한정한다.
다른 프로젝트, 원본 이미지, 사용자 로그, 데이터셋은 정리 대상이 아니다.

0.1.17·0.1.18·0.2.0·0.2.1의 `artifacts`와 `datasets`는 main의 실제 디렉터리를
가리키는 junction이다. 정리 시 연결만 제거하며 대상 디렉터리에는 삭제를 적용하지 않는다.
실제 배포물·Runtime·Catalog·CUDA·평가 증거는 main의 `artifacts`에 유지한다.
0.1.19의 독립 artifacts 616개를 파일별 SHA-256으로 main과 비교했다.
고유 파일 558개(3,549,991,522 bytes)는 main artifacts의 같은 상대 경로로 옮긴 후
해시를 재검증했다. 중복 58개(2,011,865,164 bytes)는 main 원본을 보존한다.
소스 통합·검증·원격 푸시를 확인한 뒤 release 폴더와
해당 Git worktree 등록을 제거한다.

최종 전달물은 `artifacts/distributions/0.2.1-final-n100`이다. 99개 배포 파일의
크기·SHA-256을 `distribution-manifest.json`과 대조했다. 일반 앱·CPU/CUDA 전달물과
과거 평가 자료도 유지한다. 대형 모델·이미지·EXE·ZIP은 Git에 추가하지 않는다.

## 검증

최종 통합 소스에서 아래 검증을 실행했다. 기계 판독용 요약과 증거 해시는
[검증 기록](../diagnostics/workspace-consolidation-2026-09-10.json)에 보존한다.

| 검증 | 결과 |
|---|---|
| `ruff check .` | 통과 |
| `ruff format --check .` | 782개 파일 통과 |
| 전체 Python 테스트 | 1,256개 통과 |
| Flutter analyze | 앱 2개, SDK, SDK example, mock Worker, 카메라 도구 등 6개 프로젝트 통과 |
| Flutter 테스트 | Product 193개, Lite 28개, SDK 6개, 카메라 도구 4개 통과 |
| Windows release 빌드 | Product·Lite 성공, 두 EXE ProductVersion 0.2.1 |
| `bixolon bundle verify` | staging·최종 번들 version/checksum 검증 통과 |
| 최종 N100 배포물 | manifest의 99개 파일 크기·SHA-256 일치 |
| packaged Worker CPU smoke | 일반 Lite·최종 N100 Worker 모두 통과 |
| `git diff --check` | 통과 |

Worker smoke는 readiness 200, 정상 이미지 `SEGMENTATION`, 손상 이미지 422 `ERROR`,
image 필드 누락 422 `ERROR`, 미지원 GIF 415 `ERROR`와 공개 버전 일치를 검사했다.
최종 N100 Worker EXE SHA-256은
`0c7f412dc8f2584ab3633b349819f7ad5610715f32adedaf2f0e967bc08dbd99`이다.
빌드는 main의 소스 검증용이며 기존 최종 설치본·Worker ZIP을 교체하지 않았다.
Intel N100/UHD GPU 실기기 재측정 및 새 설치본 생성은 이번 정리 범위에 포함하지 않는다.

첫 Python 실행은 작성 중이던 이 문서의 링크만 실패했다. 문서 추가 후 전체 테스트를 다시
실행해 1,256개 모두 통과했다. 이전 실패를 생략하거나 통과로 취급하지 않는다.

현재 N100 실측의 성능·남은 한계는 [기존 실측 기록](../experiments/n100-0.2.1.md)에 있으며,
이 정리 작업에서 N100 성능을 새로 측정했다고 해석하지 않는다.
