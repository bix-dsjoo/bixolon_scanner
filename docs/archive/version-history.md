# 버전 이력

이 문서는 현재 단일 제품 버전 `0.1.4` 외의 버전·평가·판단을 삭제하지 않고 찾을 수 있게 정리한
archive 인덱스입니다. 아래의 `production`, promotion, waiver, certification, release lock 표현은
당시 기록의 용어이며 현재 빌드 수명주기나 활성 기본값이 아닙니다. Git 밖의 실험·평가 결과는
보존하되 재생성 가능한 과거 전달물과 캐시는 정리할 수 있습니다.

## 0.x와 1.x

초기 `bread-worker-0.x`, detector `0.2.5`, Python/API 1.0 계열과 Bread 1.1 기록은 실험과 과거
복구 조합의 증거로 보존합니다.

- [Bread 기록](experiments/bread/README.md)
- [Detector 0.2.5](experiments/detector/detector-target-0.2.5.md)
- [Bread zero-error 1.1.0](experiments/bread/bread-zero-error-1.1.0.md)
- [과거 학습 파이프라인](guides/training-pipeline-1.0.0.md)
- [과거 release 설정](../../configs/archive/releases/bixolon_scanner_1.1.0.json)

이 계열의 수치와 예외는 현재 `0.1.4`의 독립 성능 근거로 사용하지 않습니다.

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
현재 `0.1.4`는 model binary와 Catalog payload를 유지하고 detector raw-query 기하 및 회전 입력
객체 수·bbox 합의 후처리를 추가한 patch입니다. 겹침만으로 재촬영하지 않고 추가 객체가 안정적으로
복구될 때 누락 가능성으로 처리합니다. 당시 기준 설정은
[`configs/archive/versions/0.1.3.json`](../../configs/archive/versions/0.1.3.json)에 있습니다.

## `0.1.4` 이름의 과거 미채택 앱 후보

현재 detector crowding 제품과 무관하게 과거에 `0.1.4` 이름으로 만든 로컬 앱 후보가 있었으나,
당시에는 제품 버전으로 채택하거나 배포하지 않았습니다. Runtime graph, weight, Catalog와 판정
threshold가 `0.1.3`과 같았고 앱 준비·상호작용 변경은 당시 `0.1.3+6` 소스에 통합했습니다.
후보 설정·계약·전달물은 활성 경로에서 제거했습니다.

당시 단일 packaged Worker 관찰 JSON은 실험 결과 보존 원칙에 따라
[`archive/diagnostics`](diagnostics/packaged-worker-0.1.4-smoke.json)에 남겼습니다. 이는 N100 실측,
현재 detector crowding `0.1.4` binary 증빙 또는 SLA가 아닙니다.
