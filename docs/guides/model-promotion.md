# 모델 실험과 승격

## 버전 구분

- Python 배포 버전: 패키지와 CLI의 호환성
- Worker 버전: detector·classifier·정책 조합의 운영 단위
- Detector 버전: detector 모델과 후처리 metadata
- Classifier 버전: classifier 모델과 calibration metadata
- 데이터셋 버전: manifest, label, split과 촬영 provenance
- Flutter 앱 버전: 작업자 UI와 Windows bundle

모든 버전은 서로 독립적으로 올립니다. 정식 Worker·Detector·Classifier 버전은 `1.0.0`부터 시작하며 Python 또는 앱을 배포했다고 모델 후보가 자동 승격되지 않습니다.

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

정식 1.0 정확도 gate는 전체 ground-truth 인식률 99% 이상, `APPROVED` 오인율 및 단측 95% 상한 0.1% 이하, segmentation IoU@0.5 recall/precision 99% 이상, 이미지 재촬영 recall 99% 이상입니다. 정상 이미지의 `IMAGE_RECAPTURE`와 정상 segment의 `SEGMENT_RECAPTURE`는 각각 1% 이하이고 기준 모델보다 증가할 수 없습니다. 성능 gate는 detector와 classifier가 실행되는 CUDA full-path 평균과 p95가 모두 100ms 이하입니다. 오인 0건으로 0.1% 상한을 만족하려면 최소 2,995개의 독립 승인 표본이 필요합니다.

승인 threshold 또는 재촬영 정책을 조정할 때는 risk뿐 아니라 coverage와 RECAPTURE 비회귀를 동시에 잠급니다. test/평가 결과로 threshold를 다시 맞추지 않는 것이 원칙입니다. `bread-worker-1.0.0`은 2026-08-14 프로젝트 소유자의 명시적 운영 지시에 따라 관측 오인 0건과 모든 point·segmentation·RECAPTURE·속도 gate 통과를 확인한 뒤, 표본 부족으로 실패한 `approved_misrecognition_rate_upper_95` 하나만 감사 가능한 manual waiver로 기록했습니다. 독립 이미지 사후 검증 의무와 수치 `0.2669% > 0.1%`를 package에서 제거하지 않습니다.

schema 2.1 hardening package도 동일한 통계 gate를 새로 우회하지 않는다. 모델 바이너리가 같더라도 새 production 디렉터리 생성에는 명시적 통계 위험 승인이 필요하며, 승인 전 후보는 `development`로 유지한다. 승격은 같은 볼륨의 임시 디렉터리에서 검증한 뒤 rename하고 이미 존재하는 version 디렉터리는 덮어쓰지 않는다.

원본 데이터, checkpoint, ONNX와 benchmark artifact는 Git에 넣지 않습니다. 결과 문서에는 모델·데이터셋·ONNX Runtime·CUDA·driver·hardware·warm-up과 표본 수를 기록합니다.
