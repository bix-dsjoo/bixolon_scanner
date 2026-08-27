# BIXOLON Scanner 0.1.5 개발 검증

`0.1.5`는 `0.1.4+7`에서 큰 빵 한 개를 detector crowding으로 잘못 거부하는 문제를 수정합니다.
Detector·Embedder·Verifier ONNX와 Catalog의 adapter·prototype payload는 바꾸지 않았습니다. Runtime
metadata와 detector 후처리 정책만 변경했습니다.

## 원인과 정책

`0.1.4`는 score `0.145` 이상 NMS proposal의 최대 면적 비율이 `0.21` 이상이면 proposal 크기만으로
`DETECTOR_UNCERTAIN_OBJECT`를 반환했습니다. 2026-08-27 사용자 이미지에서 27장
`IMAGE_RECAPTURE` 중 빈 트레이 4장을 제외한 23장이 이 분기의 오거부였습니다.

`0.1.5`의 큰 proposal 분기는 다음 중 하나의 보강 증거를 요구합니다.

- proposal 안의 raw query containment 수가 IoU query cluster 수보다 metadata의 최소 surplus 이상 큼
- 선택 detection 수가 metadata의 상한 이하이고, proposal 안에 선택 detection 중심점이 최소 2개 있음

기존 proximity+query duplication과 선택적 90°·180° 회전 복구 조건은 그대로입니다. 정책 수치는
Runtime `detector_crowding.large_proposal_corroboration`에서 읽습니다. 이 metadata가 없는 과거
Runtime은 기존의 무조건 큰 proposal 분기를 유지합니다.

## 결과

| 표본 | 결과 |
|---|---|
| 사용자 제공 20260827 70장, 정확한 +7 기준 | `SEGMENTATION` 43, `IMAGE_RECAPTURE` 27 |
| 같은 70장, 0.1.5 CPU HTTP | `SEGMENTATION` 66, `IMAGE_RECAPTURE` 4, `ERROR` 0 |
| 실제 빈 트레이 | 4/4 `IMAGE_RECAPTURE` |
| 비어 있지 않은 66장 상태 기대 불일치 | 0건 |
| 후보 객체 상태 | `APPROVED` 136, `UNKNOWN` 4, `SEGMENT_RECAPTURE` 4 |
| 운영 확정 severe overlap | 16/16 재촬영 유지 |
| 운영 정상 통과 확정 | 0/2 추가 재촬영 |
| accepted detector 회귀, 로컬 가용분 | 0/300 추가 재촬영 |

70장 파일별 입력 SHA-256, 상태, 객체 수와 처리시간은
`artifacts/evaluations/scanner-0.1.5/large-bread-20260827-cpu-http.json`에 기록했습니다. 정책 회귀는
`detector-crowding-large-proposal-corroboration.json`, binary 불변 변환은
`runtime-large-proposal-corroboration-conversion.json`에 기록했습니다.

## 속도와 한계

동일한 현재 PC packaged OpenVINO CPU에서 각 이미지를 한 번씩 직렬 실행했습니다. 70장 전체 평균은
`0.1.4+7` 40.916ms에서 `0.1.5+8` 55.933ms로 늘었습니다. 이는 오거부 23장이 detector 조기 종료
대신 정상 classifier 경로를 실행하는 workload 변화입니다. 상태가 동일한 47장만 짝지으면 평균은
47.831→48.832ms(+1.001ms, +2.093%)이고, 기존 정상 43장의 p50/p95는
37.195/120.581ms에서 36.912/119.093ms로 유지됐습니다. N100 성능은 새 CPU/GPU 비교 ZIP으로
다시 측정해야 합니다.

기존 accepted 415장 중 115장 원본이 현재 PC에 없어 새 코드로 전체를 다시 실행하지 못했습니다.
기존 고정 0.1.4 평가의 crowding 추가 재촬영은 0/415이고, 새 분기는 그 참 조건을 좁히기만 하므로
새 재촬영을 만들 수 없다는 단조성 근거와 실제 300장 결과를 함께 사용합니다.

사용자 70장과 운영 겹침 표본은 정책 개발·회귀 자료이지 독립 test가 아닙니다. 객체별 class 정답도
제공되지 않아 `APPROVED`의 정답 여부는 평가하지 않았습니다. 이 결과를 외부 매장·카메라 일반화,
인증 또는 SLA로 표현하지 않습니다.
