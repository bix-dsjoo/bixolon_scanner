# 모델 실험과 승격

## 버전 구분

- Python 배포 버전: 패키지와 CLI의 호환성
- 모델 패키지 버전: detector·classifier·metadata의 운영 단위
- 데이터셋 버전: manifest, label, split과 촬영 provenance
- Flutter 앱 버전: 작업자 UI와 Windows bundle

네 버전은 서로 독립적으로 올립니다. Python 또는 앱을 배포했다고 모델 후보가 자동 승격되지 않습니다.

## 실험 수명주기

```text
proposal → active → promoted ┐
                    rejected ├→ archive
```

- `proposal`: 가설, 데이터, KPI, 중단 조건을 기록한 상태
- `active`: 재현 가능한 설정과 실행 경로가 있는 현재 실험
- `promoted`: 모든 gate 통과 후 운영 package에 명시적으로 채택
- `rejected`: KPI 또는 안전 gate 실패와 이유를 기록
- `archive`: 재실행 대상이 아닌 설정·보고서를 보존

`experiment_only`는 검증 일부가 끝났지만 승격되지 않은 결과입니다. 활성 CLI에는 최신 재현 가능한 실험만 노출하고 prototype·완료·거절 설정은 `configs/archive`에 둡니다.

## 승격 순서

1. PyTorch checkpoint 생성
2. 고정 validation KPI와 calibration 검증
3. ONNX export
4. PyTorch/ONNX와 CPU/CUDA parity 검증
5. checksum과 metadata를 포함한 package 생성
6. 잠긴 test split의 정확도 gate 검증
7. warm-up 뒤 RTX 5080 full-path benchmark
8. 보고서에 `promoted` 또는 `rejected` 결정 기록

정확도 gate는 재촬영 대상 recall 99% 이상, `APPROVED` precision 99.5% 이상, 정답 클래스가 있는 `UNKNOWN` Top-3 accuracy 95% 이상입니다. 성능 gate는 detector와 classifier가 모두 실행되는 CUDA full-path p95 100ms 이하입니다.

원본 데이터, checkpoint, ONNX와 benchmark artifact는 Git에 넣지 않습니다. 결과 문서에는 모델·데이터셋·ONNX Runtime·CUDA·driver·hardware·warm-up과 표본 수를 기록합니다.
