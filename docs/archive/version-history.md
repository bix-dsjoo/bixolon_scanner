# 버전 이력

이 문서는 현재 단일 제품 버전 `0.2.1` 외의 버전·평가·판단을 삭제하지 않고 찾을 수 있게 정리한
archive 인덱스입니다. 아래의 `production`, promotion, waiver, certification, release lock 표현은
당시 기록의 용어이며 현재 빌드 수명주기나 활성 기본값이 아닙니다. Git 밖의 실험·평가 결과는
보존하되 재생성 가능한 과거 전달물과 캐시는 정리할 수 있습니다.

## Scanner 0.1.16~0.2.1 통합

2026-09-10에 별도 release 폴더의 미커밋 소스를 버전별로 main 이력에 보존했다.
현재 구성은 [0.2.1 RepViT](../architecture/scanner-0.2.1.md)이며,
[통합·보존·검증 기록](../maintenance/workspace-consolidation-2026-09-10.md)에 복원 방법을 남긴다.

- [0.1.16 설정](../../configs/archive/versions/0.1.16.json) · [연동 명세](contracts/worker-integration-0.1.16.md)
- [0.1.17 설정](../../configs/archive/versions/0.1.17.json) · [연동 명세](contracts/worker-integration-0.1.17.md)
- [0.1.18 설정](../../configs/archive/versions/0.1.18.json) · [연동 명세](contracts/worker-integration-0.1.18.md)
- [0.1.19 설정](../../configs/archive/versions/0.1.19.json): Git에 연결되지 않았던 부분 소스 폴더에서 보존
- [0.2.0 설정](../../configs/archive/versions/0.2.0.json) · [연동 명세](contracts/worker-integration-0.2.0.md)
- [0.2.1 RepViT 이전 구성](../../configs/archive/versions/0.2.1-before-repvit-final.json)

과거 `bread-10shot-0.2.1` 실험은 현재 제품 0.2.1과 별개다.

## Scanner 0.1.15

0.1.15 build 18의 [설정](../../configs/archive/versions/0.1.15.json),
[실행 결과](status/0.1.15.md), [연동 명세](contracts/worker-integration-0.1.15.md)를 보존한다.
0.1.16은 원본 220장 학습 후보와 절차적 배경 추가 학습, 추론 오류·동시성 보완을 반영한다.

## Scanner 0.1.13

`0.1.13` 앱 build 16은 1-class SSDLite320으로 위치와 촬영 품질을 판단하고, 모든 정상 ROI를
ConvNeXt-Tiny 192에 전달한 뒤 전역 위험 ROI만 224 detail과 ViT-B/16 160 verifier로 선택
검증했습니다. `0.1.14`는 이 모델·판정 계약을 유지하면서 정식 Windows 빌드에서 N100 device
matrix 필수 의존을 제거했습니다. 당시 [설정](../../configs/archive/versions/0.1.13.json),
[아키텍처](architecture/scanner-0.1.13.md), [API 연동 명세](contracts/worker-integration-0.1.13.md)를
보존합니다.

## Scanner 0.1.12

`0.1.12` 앱 build 15는 검증된 Detector class를 직접 승인에 사용했으나, 여러 매장·상품에 같은
판정 흐름을 적용하기 위해 `0.1.13`에서 class-agnostic 분류 흐름으로 대체했습니다. 당시 설정과
계약은 archive에 보존합니다.

## Scanner 0.1.11

`0.1.11` 앱 build 14는 YOLO 계열을 제거하고 외부 detector pretrained weight 없이 프로젝트
데이터로 학습한 class-aware SSDLite320 MobileNetV3-Large를 배포했습니다. 모든 정상 ROI를
DINOv3로 분류하고 위험 ROI만 224로 다시 실행했으며 415장·운영69장 회귀에서 FP/FN·오승인·Top-3
실패·`ERROR` 0을 기록했습니다. `0.1.12`에서 모델 binary는 유지하고 검증된 Detector class를 직접
판정하는 Worker routing으로 변경했습니다.

## Scanner 0.1.10

`0.1.10` 앱 build 13은 `camera_windows`가 라이브 텍스처에만 적용하는 좌우 미러링을 보정해,
촬영 전 미리보기와 촬영 후 저장 JPEG의 방향을 일치시킨 앱 patch입니다. 왼쪽·오른쪽 위치 촬영이
저장 파일에서도 같은 구역에 유지됩니다. Worker 모델·Catalog·threshold와 판정 순서는 `0.1.9`에서
변경하지 않았습니다.

## Scanner 0.1.9

`0.1.9` 앱 build 12는 빵 원본 촬영 작업공간을 추가하고 카메라 미리보기가 가로·세로 크기를 뒤집어
가로 영상 좌우를 손실할 수 있던 문제를 수정한 앱 patch입니다. 카메라 고유 종횡비를 그대로
사용하고 3×3 촬영 가이드를 실제 영상 영역에 맞췄습니다. Worker 모델·Catalog·threshold와 판정
순서는 `0.1.8`에서 변경하지 않았습니다.

## Scanner 0.1.8

`0.1.8` 앱 build 11은 `0.1.7` payload를 기반으로 승인 차단 ROI의 회전·독립 verifier가 모두 품질
실패를 반환할 때 `SEGMENT_RECAPTURE`로 처리하는 classifier 안전 정책을 활성화했습니다. 당시
설정·평가·packaged smoke는 canonical `0.1.8` archive에 보존합니다.

## 0.x와 1.x

초기 `bread-worker-0.x`, detector `0.2.5`, Python/API 1.0 계열과 Bread 1.1 기록은 실험과 과거
복구 조합의 증거로 보존합니다.

- [Bread 기록](experiments/bread/README.md)
- [Detector 0.2.5](experiments/detector/detector-target-0.2.5.md)
- [Bread zero-error 1.1.0](experiments/bread/bread-zero-error-1.1.0.md)
- [과거 학습 파이프라인](guides/training-pipeline-1.0.0.md)
- [과거 release 설정](../../configs/archive/releases/bixolon_scanner_1.1.0.json)

이 계열의 수치와 예외는 현재 `0.1.5`의 독립 성능 근거로 사용하지 않습니다.

## Scanner 2.0.0

`2.0.0-rc.7`은 Bread Project 5 파생 3,000장 평가에서 `SEGMENTATION` 이미지 FN 0.9278%,
FP 0.8591%, 전체 GT 대비 승인 오인 0.1371%로 당시 세 point gate를 실패해 반려됐습니다.

이후 `2.0.0-rc.8`은 새 owner-private locked test 없이 소유자 예외로 과거 `2.0.0` 조합이
됐습니다. 300장과 운영 115장은 개발 계보이며 독립 일반화 성능으로 표현하지 않는다는 제한이
남아 있습니다. Scanner 2.x Catalog는 키·HMAC·`signature.json` 없이
`CHECKSUM-SHA256`을 사용하도록 정리됐습니다.

- [Scanner 2.0.0 과거 설계](releases/scanner-2.0.0-architecture.md)
- [300장 개발 평가](experiments/bread/scanner-2.0.0-development-300.md)
- [owner-private test 가이드](releases/scanner-2.0-private-test.md)

## Scanner 2.0.1 후보

원본 `single_objects` 200장 Catalog를 쓰는 `2.0.1-rc.3`는 ridge/retrieval Top-1 합의와
retrieval similarity 하한을 승인 조건에 추가했습니다. 같은 300장 개발 재평가의 주요 수치는
다음과 같습니다.

| 지표 | rc.3 결과 |
|---|---|
| 정답 승인 coverage | 1,311/1,410 (92.9787%) |
| 승인 오인 | 0/1,410 |
| Candidate out | 0 |
| 연속 warm CUDA 평균/P95/P99 | 84.89/95.57/100.63ms |
| 0/1,311 승인 오인 단측 95% 상한 | 0.22825% |

이는 선택에 사용한 개발 데이터 재평가이며 독립 test, 전체 CPU/CUDA parity, 1 IPS와 10,000회
reliability를 완료한 인증 결과가 아닙니다. 과거에는 2026-08-20 소유자 waiver로 `2.0.1`이라고
표시했지만, 현재는 그 수명주기를 사용하지 않습니다.

- [rc.3 전체 평가와 당시 판단](releases/scanner-2.0.1-rc.3-single-objects.md)
- [과거 2.0.1 조합 설정](../../configs/archive/releases/scanner_2.0.1.json)
- [과거 2.0.1 waiver](../../configs/archive/releases/scanner_2.0.1_owner_waiver.json)

## `0.0.1` 매핑

`0.0.1`은 `2.0.1-rc.3`의 model graph, weight, adapter, support, prototype와 판정 정책을 바꾸지
않고 하나의 실행 버전으로 다시 패키징한 기준선입니다.

| 항목 | 원본 | 현재 표시 |
|---|---|---|
| Python·Worker | 과거 독립 버전 | `0.0.1` |
| Detector·Embedder·정책 | `2.0.1-rc.3` 계보 | `0.0.1` |
| Catalog | `2.0.1-rc.3-development-single-objects-validated` | `0.0.1` |
| Flutter | `2.0.1+5` | `0.0.1+1` |
| 사용자 화면·EXE ProductVersion | 과거 2.x | `0.0.1` |

원본 Runtime manifest SHA-256은
`7b7364c381782493d597520da4b4bc55993068ae97d7ba405e30be4b2d92389a`, Catalog manifest는
`36afc8946f3661135f575f6c03fd968bd4cfd59cd36f3713e90359a6f8442c41`입니다. 새 번들은 이 값을
`provenance.json`에 보존하고 binary payload SHA-256 집합이 동일한지 빌드 전에 확인합니다.

`0.0.1`은 특정 실행 조합의 식별자이지 독립 인증, SLA 또는 장기 `APPROVED ≥99%` 목표 달성
선언이 아닙니다. 이후 EXE 내용이 바뀌면 `0.0.2`처럼 단일 patch 버전만 올립니다.

## `0.0.2` CPU 전달 경로

`0.0.2`는 model graph, weight, Catalog payload와 판정 정책을 유지하면서 Worker에 CPU thread와
detector 제한 병렬 실행 설정을 추가한 patch입니다. Flutter 내부 build는 `0.0.2+2`입니다.

CPU 전달 ZIP은 `onnxruntime` CPU wheel만 포함하고 CUDA DLL을 포함하지 않습니다. N100 성능
측정은 선택 진단이며 결과를 인증이나 SLA로 해석하지 않습니다. 4코어 N100의 100장 진단에서는
parity-safe `4 workers × 1 thread`가 선택됐고 p50/p95/p99는
2,025.864/2,405.074/2,714.368ms였습니다. `0.0.1` 기준 설정은
[`configs/archive/versions/0.0.1.json`](../../configs/archive/versions/0.0.1.json), 당시 번들 설명은
[`scanner-0.0.1.md`](scanner-0.0.1.md)에 보존합니다.

## `0.1.1` N100 경량 후보

`0.1.1`은 N100 처리시간을 줄이기 위해 같은 DINOv3 계열에서 Embedder를 ViT-B/16에서
ConvNeXt-Tiny last-stage fine-tuned 모델로 경량화했습니다. Detector 모델 graph와 weight는
`fold1+production` 2개를 유지하지만 `production`을 먼저 실행하고 selected count가 6이면서
`uncertain`, 최소 점수 `≤0.5` 또는 `≥0.87`일 때만 `fold1`을 추가하는 선택적 cascade로
바꿨습니다. 두 detector의 합의 수가 1 이하이면서 선택 박스 종횡비가 2.1 이상인 경우 재촬영하는
안전 조건도 유지합니다. ConvNeXt ONNX는 약 111.4MB이며 기존 ViT-B ONNX 약 327MB보다 작습니다.

300장 개발 재평가에서는 segmentation 292건, prediction 1,367건, detector false
negative/positive 각 0건과 승인 오인 0건을 유지했습니다. 정답 승인 coverage는
1,294/1,410(91.7730%)이고 CUDA 연속 warm 평균/P95/P99는 65.08/82.02/90.92ms였습니다. 이
결과는 후보 선택에 사용한 개발 데이터 진단이며 독립 일반화 성능이나 N100 SLA를 의미하지
않습니다.

같은 300장의 선택적 cascade 로컬 CPU `1×4` 평가는 평균/P95/P99
225.95/317.12/335.98ms였고 OpenVINO `1×4`는 169.53/228.13/234.90ms였습니다. 이 비교 장비는
N100이 아닙니다. N100에서는 이미 수행한 `1×4`, `2×2`, `2×1` 조합 비교를 반복하지 않고 동일한
`1×4`에서 CPU와 OpenVINO를 비교하며, OpenVINO 평균과 p95가 각각 1,000ms 이내인지
`n100-0.1.1-result.json`으로 판정합니다. 현장 JSON이 돌아오기 전에는 목표 달성을 확정하지 않고
최종 Setup을 만들지 않습니다.

## `0.1.3` 기준선

`0.1.3`은 one-class YOLO26 detector, `single_objects_3` DINOv3 ConvNeXt-Tiny soup와 독립
ViT-B/16 verifier를 하나의 제품 버전으로 묶었습니다. 전체 유효 415장 개발 회귀와 packaged
Worker/N100 진단은 [당시 평가 문서](diagnostics/scanner-0.1.3-evaluation.md)에 보존합니다.
`0.1.4`는 model binary와 Catalog payload를 유지하고 detector raw-query 기하 및 회전 입력
객체 수·bbox 합의 후처리를 추가한 patch입니다. 겹침만으로 재촬영하지 않고 추가 객체가 안정적으로
복구될 때 누락 가능성으로 처리합니다. 당시 기준 설정은
[`configs/archive/versions/0.1.3.json`](../../configs/archive/versions/0.1.3.json)에 있습니다.

## `0.1.4+7` 기준선

채택된 `0.1.4+7`은 raw query의 큰 proposal·근접·중복 조건과 선택적 회전 복구를 추가했습니다.
큰 proposal의 크기만으로 재촬영하던 분기가 2026-08-27 큰 빵 표본을 오거부해 `0.1.5+8`에서
보강 증거를 요구하도록 수정했습니다. 같은 날짜 주석 자료의 빈 장면 오검출은 기존 415장으로
학습된 전체 프레임 `object_presence` verifier를 추가해 차단했습니다. 기존 415장 packaged 응답은
`0.1.4+7`과 완전히 동일하지만 verifier 학습 자료와 겹치므로 독립 성능 근거로 사용하지 않습니다.

- [0.1.4 번들 구조](architecture/scanner-0.1.4.md)
- [0.1.4 Worker 연동 계약](contracts/worker-integration-0.1.4.md)
- [0.1.4 개발 평가](diagnostics/scanner-0.1.4-evaluation.md)
- [0.1.4 기준 설정](../../configs/archive/versions/0.1.4.json)

## `0.1.5+8` 운영 기준선

운영에 배포된 `0.1.5+8`은 큰 detector proposal의 크기만으로 재촬영하지 않고 raw-query surplus
또는 복수 detection 중심점의 보강 증거를 요구했습니다. 이후 같은 제품 버전 이름으로 진행하던
`object_presence` verifier와 병렬 Intel GPU 후보는 운영 배포물에 덮어쓰지 않았으며, 최종적으로
제품·Worker·Runtime 배포 내용이 바뀌므로 `0.1.6+9`로 승격했습니다.

- [0.1.5 번들 구조](architecture/scanner-0.1.5.md)
- [0.1.5 Worker 연동 계약](contracts/worker-integration-0.1.5.md)
- [0.1.5 개발 평가](diagnostics/scanner-0.1.5-evaluation.md)
- [0.1.5 packaged smoke](diagnostics/packaged-worker-0.1.5-build8-smoke.json)
- [0.1.5 운영 설정](../../configs/archive/versions/0.1.5.json)

## `0.1.6+9` 운영 기준선

`0.1.6+9`는 기존 crowding 정책을 유지하면서 전체 프레임 `object_presence` verifier를 추가하고,
Windows OpenVINO 구성에서 CPU Detector와 Intel GPU verifier를 병렬 실행했습니다. GPU 초기화에
실패하면 명시적 CPU fallback으로 재조립했습니다. 2026-08-27 주석 69장과 기존 415장 packaged
회귀 결과는 당시 활성 평가에 고정했으며, 이는 개발·비열화 방지 자료이지 독립 일반화 성능이나
SLA 근거가 아닙니다. `0.1.7+10`에서 모델·정책을 바꾸지 않고 코드 책임을 분리했으므로 이 버전은
archive로 이동했습니다.

- [0.1.6 번들 구조](architecture/scanner-0.1.6.md)
- [0.1.6 Worker 연동 계약](contracts/worker-integration-0.1.6.md)
- [0.1.6 개발 평가](diagnostics/scanner-0.1.6-evaluation.md)
- [0.1.6 packaged smoke](diagnostics/packaged-worker-0.1.6-build9-smoke.json)
- [0.1.6 운영 설정](../../configs/archive/versions/0.1.6.json)

## `0.1.4` 이름의 더 이른 미채택 앱 후보

현재 detector crowding 제품과 무관하게 과거에 `0.1.4` 이름으로 만든 로컬 앱 후보가 있었으나,
당시에는 제품 버전으로 채택하거나 배포하지 않았습니다. Runtime graph, weight, Catalog와 판정
threshold가 `0.1.3`과 같았고 앱 준비·상호작용 변경은 당시 `0.1.3+6` 소스에 통합했습니다.
후보 설정·계약·전달물은 활성 경로에서 제거했습니다.

당시 단일 packaged Worker 관찰 JSON은 실험 결과 보존 원칙에 따라
[`archive/diagnostics`](diagnostics/packaged-worker-0.1.4-smoke.json)에 남겼습니다. 이는 N100 실측,
현재 detector crowding `0.1.4` binary 증빙 또는 SLA가 아닙니다.
