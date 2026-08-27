# 현재 버전

기준일: 2026-08-27

현재 실행 조합은 `0.1.5` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.5` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.5` |
| Store Catalog | `0.1.5`, `CHECKSUM-SHA256` |
| Flutter 내부 빌드 | `0.1.5+8` |
| Detector | class-agnostic YOLO26 objectness, score `0.65`, NMS IoU `0.4` |
| 겹침 hard gate | 큰 proposal 보강 증거 + 기존 raw-query 기하 + 선택적 회전 합의 |
| Classifier | DINOv3 ConvNeXt-Tiny soup + 180° 검증 + DINOv3 ViT-B/16 독립 검증 |
| source candidate | `yolo26-objectness-single3-consensus-large-proposal-corroboration` |
| Runtime 원본 manifest SHA-256 | `219057533c25a5165f0591493d959e3fa2b0bfb33d982068b214c04f9a53654e` |
| Catalog 원본 manifest SHA-256 | `085612ddd781a1879ae3bb2867c32c2174247ff8703e1641367d8c37226888b7` |
| 겹침 정책 report SHA-256 | `accc3622ac2b033218ee45455e3f7f5093253b366cb7a715202cc10a15aa2707` |

## 0.1.5 변경

`0.1.4+7`은 score `0.145` 이상인 NMS proposal의 면적 비율이 `0.21` 이상이면 그 크기만으로
`IMAGE_RECAPTURE`를 반환했습니다. 큰 빵 한 개도 이 조건을 만족해 정상 이미지를 거부할 수 있었습니다.

`0.1.5`는 큰 proposal을 후보로만 취급합니다. proposal 안의 raw query 수가 IoU cluster 수보다
설정된 수만큼 더 많거나, 제한된 개수의 선택 detection 중심점이 둘 이상 포함되는 보강 증거가 있을
때만 기존 crowding 재촬영을 유지합니다. 기존 근접·query 중복 및 회전 복구 분기는 바꾸지 않았습니다.
새 metadata가 없는 과거 Runtime은 이전 큰 proposal 동작을 그대로 유지합니다.

## 개발 회귀

| 표본 | `0.1.4+7` | `0.1.5+8` 후보 |
|---|---:|---:|
| 20260827 사용자 이미지 70장 | `SEGMENTATION` 43, `IMAGE_RECAPTURE` 27 | `SEGMENTATION` 66, `IMAGE_RECAPTURE` 4 |
| 실제 빈 트레이 | 4/4 재촬영 | 4/4 재촬영 |
| 큰 빵 오거부 | 23장 | 0장 |
| 객체 상태 | A 59 / U 3 / SR 4 | A 136 / U 4 / SR 4 |

운영 검수의 확정 severe overlap은 16/16 계속 재촬영했고, 정상 통과 확정 2장의 추가 재촬영은
0/2였습니다. 로컬에 남아 있는 accepted detector 개발 회귀 300장도 추가 재촬영 0건입니다. 기존
`0.1.4` 고정 평가에는 415장 모두 crowding 추가 재촬영 0건이 기록돼 있습니다. 새 정책은 기존 큰
proposal 참 분기를 보강 조건으로 좁힐 뿐 다른 분기를 넓히지 않으므로 그 415장의 상태를 새로
재촬영으로 바꿀 수 없습니다. 다만 현재 PC에는 그중 운영 수집 115장 원본이 없어 새 코드로 415장
전체를 다시 실행하지 못했으며, 이를 독립 일반화 성능으로 표현하지 않습니다.

동일한 현재 PC packaged OpenVINO CPU에서 70장 전체 평균은 40.916ms에서 55.933ms로 늘었습니다.
이는 오거부 23장이 detector 조기 종료 대신 classifier까지 실행하는 workload 변화입니다. 상태가
동일한 47장만 짝지으면 평균은 47.831→48.832ms(+1.001ms, +2.093%)이고, 기존 정상 43장의
p50/p95는 37.195/120.581ms에서 36.912/119.093ms로 유지됐습니다. N100의 `0.1.5+8` 실측은
사용자가 새 패키지로 수행해야 하며, 기존 `0.1.4+7` N100 결과는 참고 기준으로만 보존합니다.

Windows `0.1.5+8` 번들은 157개 파일과 구성요소 version/checksum 검증을 통과했고 bundle manifest
SHA-256은 `fa711eaa9ab7576e23cdda2c31b1f48cc26d844842eb533e943748b6388a748b`입니다. OpenVINO packaged
Worker는 readiness, 정상 scan, 손상·누락·미지원 입력의 `ERROR` 계약을 모두 통과했습니다.

상세 결과와 한계는 [0.1.5 개발 검증 보고서](../evaluation/scanner-0.1.5.md), 과거 판단은
[버전 이력](../archive/version-history.md)에 기록합니다.
