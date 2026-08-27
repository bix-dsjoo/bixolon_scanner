# BIXOLON Bakery AI Scanner 0.1.6 개발 검증

`0.1.6`는 `0.1.4+7`에서 큰 빵 한 개를 detector crowding으로 잘못 거부하는 문제와, detector가
빈 장면의 주변 물체를 빵으로 오검출하는 문제를 함께 수정합니다. YOLO26 Detector, 분류 Embedder·
Verifier와 Catalog의 adapter·prototype payload는 바꾸지 않았습니다. Runtime metadata와 detector
후처리 정책을 변경하고, 기존 415장으로 학습돼 있던 DINOv3 ViT-S/16 전체 프레임
`object_presence` verifier를 Runtime에 추가했습니다.

## 원인과 정책

`0.1.4`는 score `0.145` 이상 NMS proposal의 최대 면적 비율이 `0.21` 이상이면 proposal 크기만으로
`DETECTOR_UNCERTAIN_OBJECT`를 반환했습니다. 2026-08-27 사용자 이미지에서 27장
`IMAGE_RECAPTURE` 중 빈 트레이 4장을 제외한 23장이 이 분기의 오거부였습니다.

`0.1.6`의 큰 proposal 분기는 다음 중 하나의 보강 증거를 요구합니다.

- proposal 안의 raw query containment 수가 IoU query cluster 수보다 metadata의 최소 surplus 이상 큼
- 선택 detection 수가 metadata의 상한 이하이고, proposal 안에 선택 detection 중심점이 최소 2개 있음

기존 proximity+query duplication과 선택적 90°·180° 회전 복구 조건은 그대로입니다. 정책 수치는
Runtime `detector_crowding.large_proposal_corroboration`에서 읽습니다. 이 metadata가 없는 과거
Runtime은 기존의 무조건 큰 proposal 분기를 유지합니다.

detector가 하나 이상의 객체를 정상 선택하면 전체 프레임 verifier가 빵 존재 여부 `[0, 1]`를
독립 판정합니다. 최대 class confidence가 `0.54` 미만이면 `DETECTOR_COUNT_UNCERTAIN`, 빈 장면
판정과 detection 존재 여부가 불일치하면 `DETECTOR_COUNT_MISMATCH` 진단으로 classifier 전에
`IMAGE_RECAPTURE`를 반환합니다. 외부 reason은 기존과 같은 `IMAGE_RECAPTURE_REQUIRED`입니다.
detector의 기존 hard 조건이나 detection 0건은 해당 조건으로 이미 조기 종료합니다. CPU-only는
정상 detection 뒤 CPU verifier를 실행합니다. Intel GPU 구성은 GPU verifier를 CPU detector와
병렬로 미리 시작하지만 hard 경로에서는 verifier 결과나 오류를 판정에 사용하지 않습니다.

verifier의 3-fold group-aware OOF는 학습 415장(빈 4, 비어 있지 않음 411)에서 415/415를
분리했고, 비어 있지 않은 표본의 최소 presence 확률은 `0.5505207`이었습니다. 2026-08-27의 두
오검출 빈 장면 중 더 높은 presence 확률은 `0.519718`이어서, 기존 OOF 하한보다 낮고 해당 값을
넘는 최소 운영 경계로 `0.54`를 선택했습니다. 이 선택 자체가 개발 데이터에 맞춘 것이므로 독립
test 근거가 아닙니다.

## 결과

| 표본 | 최종 packaged OpenVINO 결과 |
|---|---|
| 20260827 COCO 주석 69장 | `SEGMENTATION` 63, `IMAGE_RECAPTURE` 6, `ERROR` 0 |
| 빈 장면 / 비어 있지 않은 장면 | 빈 6/6 재촬영, 비어 있지 않은 63/63 segmentation |
| bbox IoU `0.5` 매칭 | GT 138, prediction 138, match 138, FP 0, FN 0 |
| 객체 판정 | `APPROVED` 136/136 정답, `UNKNOWN` 2/2 Top-3 정답, `SEGMENT_RECAPTURE` 0 |
| 기존 415장 응답 | `0.1.5` 후보 trace 대비 semantic diff 0/415, 최대 confidence 차이 0 |
| 기존 415장 상태 | `SEGMENTATION` 411, `IMAGE_RECAPTURE` 4; A 1,895 / U 14 / SR 5 |
| 운영 확정 severe overlap | 16/16 재촬영 유지 |
| 운영 정상 통과 확정 | 0/2 추가 재촬영 |

69장의 입력 SHA-256과 GT는 `operational-20260827-annotated69-manifest.jsonl`, 최종 0.1.6 HTTP
응답은 `operational-20260827-packaged-cpu-fallback-trace.jsonl`, COCO 판정은
`operational-20260827-packaged-cpu-fallback-coco.json`에 기록했습니다. 기존 415장과 69장 비교는
각각 `detector415-0.1.5-vs-0.1.6-diff.json`,
`operational-20260827-0.1.5-vs-0.1.6-diff.json`이 상태, reason, segmentation 수, bbox, 객체 상태,
prediction, Top-3, confidence와 version null pattern을 모두 검사합니다.

최종 ONNX는 원본 verifier weight를 유지한 채 public batch를 1로 고정하고 shape inference 후
`ORT_ENABLE_EXTENDED`로 최적화했습니다. 동적 원본과 public batch만 고정한 graph는 packaged
OpenVINO 초기화에 실패했고, 최종 graph는 fallback 없이 실제 OpenVINO Worker readiness와 69장·
415장 HTTP 실행을 통과했습니다. 원본·중간·최종 SHA-256과 `weights_modified: false`는
`presence-verifier-fixed-batch-optimization-v2.json`, Runtime 파일 불변성과 manifest는
`runtime-presence-verifier-conversion.json`에 기록했습니다.

## 속도와 한계

동일한 현재 PC packaged OpenVINO CPU에서 warm-up 5회 뒤 각 이미지를 한 번씩 직렬 실행했습니다.
2026-08-27 비어 있지 않은 63장 full path의 worker 평균/p50/p95/p99는
84.238/68.497/154.429/162.262ms, HTTP 왕복 평균/p95는 98.413/168.830ms였습니다.

기존 415장의 411개 full path는 최종 worker 평균/p50/p95/p99
155.989/149.142/213.226/267.096ms입니다. 동일한 0.1.5 후보 trace보다 각각
18.368/18.015/23.358/10.466ms 감소했고 상태와 confidence는 완전히 동일하며, 현재 PC 500ms
평균·p95 진단 범위 안입니다. 69장도 동일 기준보다 평균 8.232ms, p95 6.861ms 감소했습니다. 이
수치는 N100 측정이나 SLA가 아닙니다.

동일 source 후보의 N100 100장 실측에서 CPU Detector와 GPU presence·Embedder 병렬 구성의 worker
평균/p95는 `440.767/729.284ms`, CPU-only는 `637.597/1003.835ms`였습니다. CPU 대비 속도비는 평균
`1.443배`, p95 `1.375배`이고 semantic mismatch와 오류는 0건입니다. 최대 confidence 차이는
`0.0085055232`로 strict tolerance `1e-5`를 넘었고 하이브리드 p95도 500ms 진단 기준을 넘으므로
원본 결과의 `passes=false`를 유지합니다. 이 실측은 버전 metadata를 0.1.6으로 바꾸기 전 동일 모델·
정책의 0.1.5 후보에서 수행됐으며, 0.1.6 Runtime/Catalog의 binary payload 불변성 검증과 함께 참조합니다.

최종 0.1.6 Worker를 GPU 요청과 CPU fallback 설정으로 시작한 smoke에서 GPU 초기화 실패 뒤
OpenVINO CPU로 `/ready`와 정상 scan을 완료했습니다. 같은 배포 실행 파일의 명시적 CPU fallback
경로로 69장과 415장도 다시 실행했습니다. 0.1.5 후보 trace 대비 status, reason, bbox, item status,
prediction, Top-3, confidence와 version null pattern 차이는 모두 0이었습니다. 이 현재 PC 수치는
N100 성능 비교 근거로 사용하지 않습니다.

presence verifier는 동일한 415장으로 학습됐으므로 415장 완전 동일 결과는 비열화 방지 회귀로만
사용합니다. 2026-08-27 69장과 운영 겹침 표본도 정책·threshold 선택에 사용한 개발 자료입니다.
따라서 이 결과를 외부 매장·카메라 일반화, 독립 test, 인증 또는 SLA로 표현하지 않습니다.
