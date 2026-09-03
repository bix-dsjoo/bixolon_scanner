# 0.1.13 일관 추론 파이프라인 결정 기록

## 결론

0.1.13은 다음 구조를 활성화합니다.

`1-class SSDLite320 → ConvNeXt-Tiny 192 전체 ROI → ConvNeXt-Tiny 224 선택 상세 → Frozen
ViT-B/16 160 선택 검증 → 단일 Safety Arbiter`

운영 일관성은 모든 모델을 항상 실행한다는 뜻이 아닙니다. 모든 매장·SKU에서 모델 순서, 입력,
전처리, Catalog 계약, 전역 위험 조건과 최종 상태 우선순위가 같다는 뜻입니다. 매장별·SKU별 분기와
Detector SKU 직접 승인은 사용하지 않습니다.

## 후보 비교

현재 개발 회귀 415장, 1,914개 객체에서 비교했습니다. 이 데이터는 학습·정책 선택과 겹치므로
독립 test가 아닙니다.

| 정책 | 정답 승인 | 정답 승인율 | UNKNOWN | SEGMENT_RECAPTURE | FP/FN | 오승인 | Top-3 누락 | CPU full-path p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 선택 224 + 선택 ViT | 1,898 | 99.164% | 11 | 5 | 0/0 | 0 | 0 | 440.2ms |
| 승인 전수 ViT, class 불일치 veto | 1,873 | 97.858% | 36 | 5 | 0/0 | 0 | 0 | 894.7ms |
| 승인 전수 ViT, 단일 품질 거부 재촬영 | 1,826 | 95.402% | 20 | 68 | 0/0 | 0 | 0 | 1,122.7ms |

전수 verifier는 잘못된 승인을 추가로 발견하지 못하면서 99% 정답 승인 목표를 깨고 지연과
재촬영을 늘렸습니다. Frozen backbone이라는 사실만으로 오류가 독립이 되는 것은 아니며, 같은 crop,
Catalog와 개발 분포를 공유하는 verifier를 hard veto로 과신하지 않습니다.

활성 선택 정책의 추가 진단은 다음과 같습니다.

| 진단 세트 | 객체 | 정답 승인율 | FP/FN | 오승인 | Top-3 누락 | 비고 |
|---|---:|---:|---:|---:|---:|---|
| 개발 회귀 415장 | 1,914 | 99.164% | 0/0 | 0 | 0 | 현 환경 재실행 |
| 운영 촬영 69장 | 138 | 100% | 0/0 | 0 | 0 | 현 환경 재실행 |
| multi-object 300장 | 1,410 | 99.149% | 0/0 | 0 | 0 | 동일 binary·정책의 기존 v6 진단 |

## 연구와 운영 원칙의 연결

- [DINOv3 기술 보고서](https://arxiv.org/abs/2508.10104)는 frozen backbone과 여러 해상도·모델
  크기의 후처리 유연성을 보여주지만, 이 프로젝트의 매장 분포에서 오류 0을 보증하지는 않습니다.
- [Selective Classification Under Distribution Shifts](https://arxiv.org/abs/2405.05160)는 배포
  분포 이동에서 confidence 기반 선택을 별도로 다뤄야 함을 보입니다. 따라서 `UNKNOWN`과
  `RECAPTURE`를 실패가 아니라 명시적 abstention으로 유지합니다.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)은 정해진 calibration 절차와 교환가능성
  같은 가정 아래 위험을 제어하는 방법입니다. 현재 표본은 그 보증 조건을 충족하지 않으므로 0건
  진단을 확률적 보증으로 표현하지 않습니다.
- Google의 [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/)
  과 [production monitoring 지침](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)은
  단순한 end-to-end 계약, training-serving skew 측정과 시간 이후 데이터 평가를 강조합니다.
- NIST의 [배포 AI 모니터링 보고서](https://www.nist.gov/news-events/news/2026/03/new-report-challenges-monitoring-deployed-ai-systems)는
  배포 후 incident·field monitoring을 별도 활동으로 봅니다. 0.1.13도 개발 회귀 통과와 운영 일반화
  확인을 구분합니다.

## 새 매장 확정 절차

여러 매장 이미지가 아직 없으므로 파이프라인은 고정하되 일반화 성능은 확정하지 않습니다.

1. 0.1.13 모델·threshold·전처리를 바꾸지 않고 새 매장에서 shadow capture를 수집합니다.
2. 물리 상품과 촬영 session을 기준으로 group-aware split을 만들고 매장별/전체 지표를 함께 냅니다.
3. detector recall, 정답 승인율, 오승인, Top-3 누락, `IMAGE_RECAPTURE`, `SEGMENT_RECAPTURE`, p95를
   매장 slice별로 기록합니다.
4. 특정 매장만 임계값을 바꾸지 않습니다. 공통 정책 변경이 필요하면 모든 매장 회귀를 다시 돌리고
   새 patch 버전으로 묶습니다.
5. 로그의 상태·reason code 분포와 입력 통계를 감시하되 이미지 bytes와 로컬 경로는 남기지 않습니다.

## 재현

Runtime source는 다음 명령으로 만듭니다. ONNX graph·weight와 Catalog payload는 변경하지 않습니다.

```powershell
python -m bixolon_scanner.operations.consistent_evidence_runtime `
  --source-runtime artifacts/experiments/class-agnostic-detector-0.1.11/runtime-candidate-v6-dense-fallback `
  --output-dir artifacts/experiments/consistent-evidence-0.1.13/runtime-source `
  --report artifacts/experiments/consistent-evidence-0.1.13/runtime-source-build.json
```

평가 산출물은 `artifacts/experiments/consistent-evidence-0.1.13`에 보존합니다. 활성 설정은 각
source manifest와 평가 파일의 SHA-256을 고정합니다.
