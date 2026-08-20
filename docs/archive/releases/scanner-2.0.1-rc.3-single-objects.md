# Scanner `2.0.1-rc.1` 대 `2.0.1-rc.3` 안전 정책 비교

기준일은 2026-08-20이다. 이 문서는 DINOv3 ViT-B/16과 원본 `single_objects` 200장 조합을
유지한 `2.0.1-rc.1`과 `2.0.1-rc.3`의 개발 회귀 비교다. `single_objects`와
`single_objects_2`의 데이터셋 A/B가 아니다. 300장은 이미 정책 선택에 사용된 development
데이터이므로 독립 test나 승격 증거가 아니다.

저장된 두 CUDA trace의 `SEGMENTATION` 출력 1,375개를 대조하면 classifier Top-1,
`approval_score`, `top3_safety_score` 불일치는 각각 0건이다. Detector와 ONNX embedder 파일도
SHA-256이 동일하다. 따라서 아래 차이는 classifier 순위 개선이 아니라 rc.3 승인 방어 정책의
효과다.

## 진단

`2.0.1-rc.1`은 `single_objects`로 Catalog ridge head를 다시 fit하면서도
`single_objects_2`에서 고른 approval threshold `0.15510578930378`을 그대로 사용했다. 그 결과
1,410 GT 중 정답 승인 1,328개와 오승인 6개가 발생했다. 원본 support만 source-image fold로
나눈 내부 OOF는 Top-1
195/200이었고, 오승인 0건을 만드는 단일 threshold는 `0.5054142475`, 승인 coverage는
169/200이었다. 이 OOF는 SKU별 10장이 같은 물리 상품의 view이므로 새 상품 instance에 대한 독립
일반화 증거로 사용하지 않는다.

단일 threshold만 높이면 승인 coverage가 운영 gate 아래로 내려가므로 다음 결정을 결합했다.

- ridge approval margin `>= 0.212`
- ridge Top-1과 exact support/prototype retrieval Top-1이 같아야 승인
- retrieval Top-1 similarity `>= 0.39`; 미만은 `CLASSIFIER_OUT_OF_CATALOG`
- Top-3 safety threshold `-2.960296869277954`

`0.212`는 개발 trace에서 `0.2050926387`과 `0.2192806751` 사이의 빈 구간에 두었다. 기존
CPU/CUDA 최대 confidence 차이 `0.0041305423`보다 양쪽 여유가 크다. `0.39`도 가장 가까운 개발
retrieval score와 `0.0126` 이상 떨어져 있다. 이것은 provider guard 설계 근거이며 RC.3 자체의
CPU/CUDA 전체 parity를 대체하지 않는다.

| 정책 | `2.0.1-rc.1` | `2.0.1-rc.3` |
|---|---:|---:|
| Ridge approval margin | `0.15510578930378` | `0.212` |
| Ridge/검색 Top-1 일치 필수 | 비활성 | 활성 |
| 검색 최소 similarity | 비활성 | `0.39` |
| Top-3 safety threshold | `-2.960296869277954` | 동일 |

## 300장 CUDA 결과

아래 객체 상태 세 비율은 모두 `SEGMENTATION` 이미지가 반환한 객체 1,375개를 같은 분모로
사용한다.

| 지표 | `2.0.1-rc.1` | `2.0.1-rc.3` | 변화 |
|---|---:|---:|---:|
| `SEGMENTATION` | 294/300 = 98.0000% | 294/300 = 98.0000% | 동일 |
| `IMAGE_RECAPTURE` | 6/300 = 2.0000% | 6/300 = 2.0000% | 동일 |
| `APPROVED` / segmentation | 1,334/1,375 = 97.0182% | 1,311/1,375 = 95.3455% | -1.6727%p, -23개 |
| `UNKNOWN` Top-3 / segmentation | 16/1,375 = 1.1636% | 29/1,375 = 2.1091% | +0.9455%p, +13개 |
| `SEGMENT_RECAPTURE` / segmentation | 25/1,375 = 1.8182% | 35/1,375 = 2.5455% | +0.7273%p, +10개 |
| 정답 `APPROVED` / 전체 GT | 1,328/1,410 = 94.1844% | 1,311/1,410 = 92.9787% | -1.2057%p, -17개 |
| `APPROVED` 오인 / 전체 GT | 6/1,410 = 0.4255% | 0/1,410 = 0% | -0.4255%p, -6건 |
| `UNKNOWN` Candidate out / 전체 GT | 0/1,410 = 0% | 0/1,410 = 0% | 동일 |
| segmentation 이미지 FN/FP | 0/294, 0/294 | 0/294, 0/294 | 동일 |
| Classifier Top-1 | 1,356/1,375 = 98.6182% | 1,356/1,375 = 98.6182% | 동일 |
| 평균 / p50 / p95 / p99 | 90.06 / 87.86 / 106.46 / 134.23ms | 84.89 / 84.08 / 95.57 / 100.63ms | 이번 실행은 성능 gate 통과 |

rc.3는 rc.1의 오승인 6건을 모두 방어했다. MEDIUM 1건과 HARD 4건은
`SEGMENT_RECAPTURE`, HARD 1건은 `UNKNOWN`으로 전환됐다. 그 대가로 정답 승인도 17개 줄었다.
Classifier Top-1은 policy 이전 단계이므로 동일하며, 정책은 분류 순위 자체를 바꾸지 않는다.

| 난이도 | 후보 | `APPROVED` | `UNKNOWN` | `SEGMENT_RECAPTURE` | 오승인 / GT | 정답 승인 / GT | Classifier Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| EASY | rc.1 | 405/405 = 100% | 0/405 = 0% | 0/405 = 0% | 0/410 | 405/410 = 98.7805% | 405/405 = 100% |
| EASY | rc.3 | 404/405 = 99.7531% | 1/405 = 0.2469% | 0/405 = 0% | 0/410 | 404/410 = 98.5366% | 405/405 = 100% |
| MEDIUM | rc.1 | 482/493 = 97.7688% | 1/493 = 0.2028% | 10/493 = 2.0284% | 1/500 = 0.2000% | 481/500 = 96.2000% | 488/493 = 98.9858% |
| MEDIUM | rc.3 | 477/493 = 96.7546% | 5/493 = 1.0142% | 11/493 = 2.2312% | 0/500 | 477/500 = 95.4000% | 488/493 = 98.9858% |
| HARD | rc.1 | 447/477 = 93.7107% | 15/477 = 3.1447% | 15/477 = 3.1447% | 5/500 = 1.0000% | 442/500 = 88.4000% | 463/477 = 97.0650% |
| HARD | rc.3 | 430/477 = 90.1468% | 23/477 = 4.8218% | 24/477 = 5.0314% | 0/500 | 430/500 = 86.0000% | 463/477 = 97.0650% |

난이도별 이미지 상태는 두 후보가 동일하다. EASY와 MEDIUM은 각각
`SEGMENTATION` 99/100, `IMAGE_RECAPTURE` 1/100이고 HARD는 각각 96/100, 4/100이다. 모든
난이도에서 `SEGMENTATION` 이미지 FN/FP 포함 비율은 0%다.

## 운영 승격과 남은 인증

결론적으로 rc.1이 아니라 rc.3를 최종 `2.0.1` 운영 원본으로 선택했다. 프로젝트 소유자는
2026-08-20 rc.3의 즉시 운영 승격과 EXE 반영을 명시적으로 지시했다. 300장 point gate와 연속
CUDA p95는 통과했고, 새 production package의 CPU/CUDA packaged Worker smoke와 Windows 앱
CUDA readiness도 통과했다.

다만 오승인 0/1,311의 단측 95% 상한 0.22825%는 0.1% gate를 통과하지 못한다. 비공개 test,
rc.3 전체 CPU/CUDA parity, 1 IPS cadence, 10,000회 reliability와 release별 vulnerability/SBOM도
미완료다. 이 항목은 통과로 바꾸지 않고
`configs/releases/scanner_2.0.1_owner_waiver.json`에 고정했다. 따라서 `production`은 현재 배포
상태이며 `independent_certified=false`다. 다음 일반 승격에는 이 waiver를 상속하지 않는다.

고정 artifact는 다음과 같다.

- Runtime metadata SHA-256: `09dc5a405b89a94425af9d11ca02fcf9a1a1fa987b309896edd885b5dc59bdbd`
- Catalog metadata SHA-256: `b60e75f14c5245b2fe09de6fe28b5ce1ddd0383234f5e62a76be8b995d99e2b0`
- Catalog source manifest SHA-256: `700dbdc49e4a8b4cb5ccad09516dc143c4864279b411a4ae4f4b325ec0273add`
- RC.1 CUDA report SHA-256: `248df550f5cb81a5166c0d4284da7aa7798dd01f1eeed06561cc279d8af85ce4`
- RC.1 CUDA trace SHA-256: `9ab11b257a5c5f852aca1ce8caa2cdb47934c381e5ca3a4b8d8bc7bec1f37097`
- CUDA report SHA-256: `be7d009d5de1d41ae3ddf10333b510755016301efc54abf027b1b710e4bfe736`
- CUDA trace SHA-256: `b06e42514ace6d93e3d67a5fe429462f8f0a5bf49edb28339e342ee900f732cb`
- difficulty breakdown SHA-256: `1aee711f75f6f33051504391d16b20802f19e456e20755cc4cb20d8e74c6eddc`
- Production runtime manifest SHA-256: `d0905c6bbe2c118a08549cd6b777284313148c7051d3a4df567136d902cde7b4`
- Production Catalog manifest SHA-256: `1df2d6ef56eb818656eed029869020383980101e5cee43ce2d4643b152f4b73a`
- Worker EXE SHA-256: `63d386a8e106803fb2fb21ce9b4a4fbb746fc377081153a392ffb0aab6416559`
- Windows 앱 EXE SHA-256: `c3d65429ae1e02518bc56cb2de9efba4492e8eecaa3a29d2e17c4c09b35b0700`
- Windows bundle manifest SHA-256: `f230626160608134986f3d95876eb75c743c0a4f00848847e81435973bcb6f30`
