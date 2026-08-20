# 버전 이력

이 문서는 단일 제품 버전 `0.0.1` 이전의 버전·평가·판단을 삭제하지 않고 찾을 수 있게 정리한
archive 인덱스입니다. 아래의 `production`, promotion, waiver, certification, release lock 표현은
당시 기록의 용어이며 현재 빌드 수명주기나 활성 기본값이 아닙니다. 기존 ignored binary 산출물도
원래 위치에서 이동하거나 삭제하지 않았습니다.

## 0.x와 1.x

초기 `bread-worker-0.x`, detector `0.2.5`, Python/API 1.0 계열과 Bread 1.1 기록은 실험과 과거
복구 조합의 증거로 보존합니다.

- [Bread 기록](experiments/bread/README.md)
- [Detector 0.2.5](experiments/detector/detector-target-0.2.5.md)
- [Bread zero-error 1.1.0](experiments/bread/bread-zero-error-1.1.0.md)
- [과거 학습 파이프라인](guides/training-pipeline-1.0.0.md)
- [과거 release 설정](../../configs/archive/releases/bixolon_scanner_1.1.0.json)

이 계열의 수치와 예외는 현재 `0.0.1`의 독립 성능 근거로 사용하지 않습니다.

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
