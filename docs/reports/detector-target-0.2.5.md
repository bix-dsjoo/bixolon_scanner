# Detector 안전성 우선 목표 모드 `0.2.5`

## 상태

`0.2.5` detector 목표 모드는 전체 단계를 실행한 실험 경로이며 최종 승격 상태는 `experiment_only`다. ONNX parity와 RTX 5080 benchmark는 통과했지만 독립성·risk·hard recall·UNKNOWN Top-3 gate를 통과하지 못했으므로 운영 기본 package를 변경하지 않는다.

## 선택 계약

평가 단위는 이미지 한 장이다. 예측과 GT가 IoU 0.5에서 1:1로 완전히 매칭되고 FP/FN과 count 차이가 없을 때만 `detector_correct=true`다. IoU 0.75 exact-image 결과도 별도로 기록한다.

후보 선택 순서는 다음과 같다.

1. Natural development의 `Detector PASS Risk U95 ≤ 0.5%`
2. Natural development의 `E2E APPROVED Risk U95 ≤ 0.5%`
3. Hard development의 `Error Catch Recall ≥ 99%`
4. `Safe Auto-Pass Rate` 최대
5. 표본 30장 이상 그룹의 worst approval coverage 최대
6. AUGRC 최소, seed와 canonical policy JSON 순

단측 95% Clopper–Pearson 상한을 사용한다. 오류가 0건이어도 PASS 또는 APPROVED 표본 수가 부족하면 risk 인증은 실패한다. 적격 후보가 없으면 Silent Failure 수, 두 risk 상한, Safe Auto-Pass 순으로 진단 후보를 남기지만 production 후보로 취급하지 않는다.

## 탐색 범위

- detector: RT-DETRv2-R18, seed `20260812/20260813/20260814`, capture-session 3-fold OOF
- score threshold: `0.05–0.95`, 간격 `0.01`
- NMS IoU: `0.5/0.6/0.7/0.8`
- uncertainty score: disabled 또는 `0.10/0.20/0.30/0.40`
- uncertainty 최소 면적: `0.020/0.039/0.050`
- uncertainty match IoU: `0.5`
- classifier: `0.2.4` ONNX·calibration 동결
- border/count/quality gate: 기존 Worker 정책 동결

각 model/policy family는 risk–coverage, AUGRC, 호환용 AURC와 failure AUROC를 기록한다. 운영점에서는 Useful Reject, Wasted Reject, Silent Failure, Safe Pass, risk U95, detector/approval coverage와 Safe Auto-Pass를 보고한다. 객체 진단에는 TP/FP/FN, exact-count, exact-image IoU 0.5/0.75, fixed-point LRP, localization/duplicate/background FP/missed GT/count mismatch를 포함한다.

## 실행 결과 (`2026-08-13`)

고정 grid의 4개 모델(baseline과 세 seed) × 3,724개 정책, 총 14,896개 후보를 모두 평가했다. development 적격 후보는 0개였고 fallback 진단 후보로 baseline, score `0.68`, NMS IoU `0.5`, uncertainty disabled가 선택됐다. 선택 보고서에 기록된 sweep hash와 682,719,482-byte sweep 파일의 실제 SHA-256은 `7e72170706ae4a8d6c0673a3021716b2e45089e20c4e883d08354a41cb86df7d`로 일치한다.

| development set | 이미지 | Useful / Wasted / Silent / Safe | Detector risk U95 | APPROVED 오류/수 | E2E risk U95 | Safe Auto-Pass | Error Catch Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural | 247 | 0 / 7 / 0 / 240 | 1.2405% | 0/149 | 1.9905% | 149 (60.32%) | 분모 없음 |
| Hard | 232 | 6 / 0 / 218 / 8 | 98.2266% | 2/2 | 100% | 0 (0%) | 2.68% |

Natural development는 관측 Silent Failure와 승인 오류가 모두 0건이어도 PASS 240장과 APPROVED 149장뿐이므로 0.5% U95를 인증할 수 없다. zero-error 인증에는 각각 최소 598개 독립 표본이 필요하다.

pre-test lock `7ed44ee7b28fa214ae49e7f2436698e0509b15a639c5e14b17dd01421e3a6dab` 이후 threshold나 정책은 변경하지 않았다. locked test 결과는 다음과 같다.

| test set | 이미지 | Useful / Wasted / Silent / Safe | Detector risk (U95) | APPROVED 오류/수 (U95) | Safe Auto-Pass | Error Catch Recall | UNKNOWN Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural | 94 | 0 / 0 / 3 / 91 | 3.19% (8.04%) | 0/38 (7.58%) | 38 (40.43%) | 0% | 83/88 (94.32%) |
| Hard | 77 | 0 / 0 / 68 / 9 | 88.31% (93.77%) | 0/0 (100%) | 0 | 0% | 1/474 (0.21%) |
| Shift | 80 | 4 / 0 / 74 / 2 | 97.37% (99.53%) | 0/0 (100%) | 0 | 5.13% | 2/374 (0.53%) |

Natural 객체 진단은 precision `99.8031%`, recall `99.2172%`, exact-image IoU 0.5 `96.8085%`, IoU 0.75 `94.6809%`다. Hard/Shift는 현재 regression 진단 manifest이며 독립 promotion evidence가 아니다. 모든 set의 E/M/H, 객체 수, small/border, capture session, camera/store/light, blur/exposure, novel/overlap 그룹별 표본 수·gate 네 칸·두 risk U95·coverage·Safe Auto-Pass는 `reports/locked-test.json`의 `sets.*.metrics.groups`에 기록했다. Hard는 Natural coverage 분모에 포함하지 않았다.

데이터 감사에서 Natural 341장은 `perceptual_group_id`가 없고 299장은 `physical_target_group_id`가 없다. Hard 309장과 Shift 303장도 `perceptual_group_id`가 없어 세 set의 `promotion_evidence_ready`가 모두 `false`다.

## ONNX parity와 성능

- detector ONNX SHA-256: `b4f22a766c995239d927ae20b79490a7ddd2179ff9918a34ef3316ba4bd1bf6b`
- 동결 classifier ONNX SHA-256: `f0af021c271721f702061cb8d7042f39bf35b2050e6acace9e7dc386b133cbee`
- detector matched minimum IoU: CPU `0.9999996`, CUDA `0.9975026`; count·좌표·score parity 통과
- classifier 886개 tensor: PyTorch/CPU ONNX/CUDA ONNX tolerance, Top-1, 정렬 Top-3, 최종 상태 parity 모두 통과; mismatch 0건
- RTX 5080, CUDA 13.1, ORT 1.28.0, warm-up 30회, full-path 1,000회: p50 `69.51ms`, p95 `96.10ms`, p99 `104.62ms`
- stage p95: decode `47.11ms`, detector `34.83ms`, classifier `20.74ms`, decision overhead `0.31ms`

지연 gate `p95 ≤ 100ms`와 parity gate는 통과했다.

## 최종 승격 gate

- locked natural test의 detector/e2e risk U95 각각 0.5% 이하
- detector Silent Failure 0건, 잘못된 E2E APPROVED 0건
- hard challenge Error Catch Recall 99% 이상
- 정답 class가 있는 UNKNOWN Top-3 accuracy 95% 이상
- PyTorch/CPU ONNX/CUDA ONNX 허용오차와 상태·Top-3 순위 parity
- RTX 5080, warm-up 30회, full-path 1,000회 p95 100ms 이하
- natural/hard/shift manifest의 독립성 필드와 split 누수 검사 통과

수동 waiver는 없다. 모든 gate가 통과한 보고서가 생성되기 전에는 package metadata를 production으로 바꾸거나 앱의 기본 package를 전환하지 않는다.

실제 최종 실패 항목은 `independent_data_ready`, `detector_pass_risk_u95`, `e2e_approved_risk_u95`, `detector_silent_failure_zero`, `hard_error_catch_recall`, `unknown_top3_accuracy`다. `e2e_approved_error_zero`, `parity`, `full_path_latency`는 통과했다.

## 버전과 provenance

공개 응답 구조는 변경하지 않는다. 실행된 detector와 classifier는 전체 추론 모델 버전 `0.2.5`를 반환하고 detector hard gate의 classifier 값만 `null`이다. package의 `bundle_provenance`에는 동결 classifier의 원본 `0.2.4` 버전·SHA-256, detector selection 보고서 SHA-256, Natural/Hard/Shift 평가 데이터 버전을 저장한다.
