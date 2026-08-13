# Detector 안전성 우선 목표 모드 `0.2.5`

## 상태

`0.2.5` detector 목표 모드는 구현된 실험 경로이며 현재 승격 상태는 `experiment_only`다. 저장소에는 독립 natural prevalence, hard challenge, shift test와 실행된 ONNX parity·RTX 5080 benchmark 증거가 모두 존재하지 않으므로 운영 기본 package를 변경하지 않는다.

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

## 최종 승격 gate

- locked natural test의 detector/e2e risk U95 각각 0.5% 이하
- detector Silent Failure 0건, 잘못된 E2E APPROVED 0건
- hard challenge Error Catch Recall 99% 이상
- 정답 class가 있는 UNKNOWN Top-3 accuracy 95% 이상
- PyTorch/CPU ONNX/CUDA ONNX 허용오차와 상태·Top-3 순위 parity
- RTX 5080, warm-up 30회, full-path 1,000회 p95 100ms 이하
- natural/hard/shift manifest의 독립성 필드와 split 누수 검사 통과

수동 waiver는 없다. 모든 gate가 통과한 보고서가 생성되기 전에는 package metadata를 production으로 바꾸거나 앱의 기본 package를 전환하지 않는다.

## 버전과 provenance

공개 응답 구조는 변경하지 않는다. 실행된 detector와 classifier는 전체 추론 모델 버전 `0.2.5`를 반환하고 detector hard gate의 classifier 값만 `null`이다. package의 `bundle_provenance`에는 동결 classifier의 원본 `0.2.4` 버전·SHA-256, detector selection 보고서 SHA-256, Natural/Hard/Shift 평가 데이터 버전을 저장한다.
