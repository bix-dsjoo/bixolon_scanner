# 모델 실험과 승격

## 버전 구분

- Python 배포 버전: 패키지와 CLI의 호환성
- Worker 버전: detector·classifier·정책 조합의 운영 단위
- Detector 버전: detector 모델과 후처리 metadata
- Classifier 버전: classifier 모델과 calibration metadata
- 데이터셋 버전: manifest, label, split과 촬영 provenance
- Flutter 앱 버전: 작업자 UI와 Windows bundle

모든 버전은 서로 독립적으로 올립니다. 정식 Worker·Detector·Classifier 버전은 `1.0.0`부터 시작하며 Python 또는 앱을 배포했다고 모델 후보가 자동 승격되지 않습니다.

## 현재 Scanner 2.0.1 운영 예외

프로젝트 소유자는 2026-08-20 `2.0.1-rc.3`를 최종 `2.0.1` 운영 release로 명시적으로
승격했다. 일반 승격 gate를 통과한 것으로 소급 표현하지 않으며, 독립 비공개 test·통계 상한·
300장 CPU/CUDA 전체 parity·1 IPS cadence·장시간 reliability·SBOM/취약점 scan과 Catalog HMAC
미충족은 `configs/releases/scanner_2.0.1_owner_waiver.json`에 보존한다. 재현 가능한 최종 조합은
`configs/releases/scanner_2.0.1.json`이고 Windows 앱 버전은 `2.0.1+5`다. 이 예외는 후속 버전에
자동 상속되지 않는다.

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

### Bread 1.1 공식 override

2026-08-18 프로젝트 소유자가 `bread-zero-error-1.1.0`과 Worker·Detector·Classifier
`1.1.0` 후보의 정확도 승격 기준을 아래처럼 지정했다. 1.1 후보에는 위 1.0 정확도 gate 대신
이 point-rate gate를 적용하되, 잠긴 test·PyTorch/ONNX·CPU/CUDA 상태 parity·package 및
RTX 5080 full-path 성능 gate는 생략하지 않는다.

| 지표 | 기준 |
| --- | ---: |
| `SEGMENTATION` | ≥ 90% |
| `APPROVED` | ≥ 90% |
| `SEGMENTATION` 이미지 FN | ≤ 0.1% |
| `SEGMENTATION` 이미지 FP | ≤ 0.1% |
| `APPROVED` 객체 오인율 | ≤ 0.1% |
| `UNKNOWN` Top-3 Candidate out | ≤ 0.1% |

`UNKNOWN + Top-3` 비율과 `SEGMENT_RECAPTURE` 비율은 진단 지표로 계속 보고하지만 1.1 승격
합격 여부에는 사용하지 않는다. 경계값은 통과로 처리한다. 정확한 분모,
`IMAGE_RECAPTURE` 비회귀 조건과 현재 결과의 재판정은
[Bread zero-error 목표 실험 1.1.0](../experiments/bread/bread-zero-error-1.1.0.md)에 고정한다.

1.1의 공식 평가 집합은 `multi_object_scenes` EASY/MEDIUM/HARD 300장이다.
`SEGMENTATION` 비율의 분모는 이 300장 전체이고, 이미지 FN/FP의 분모는 최종
응답이 `SEGMENTATION`인 이미지다. 한 이미지에 IoU@0.5 FN 또는 FP가 하나 이상이면 해당
이미지 오류 1건으로 센다. raw 객체 FP/FN은 별도 진단으로 계속 공개한다.

승인 threshold 또는 재촬영 정책을 조정할 때는 risk뿐 아니라 coverage와 RECAPTURE 비회귀를 동시에 잠급니다. test/평가 결과로 threshold를 다시 맞추지 않는 것이 원칙입니다. `bread-worker-1.0.0`은 2026-08-14 프로젝트 소유자의 명시적 운영 지시에 따라 관측 오인 0건과 모든 point·segmentation·RECAPTURE·속도 gate 통과를 확인한 뒤, 표본 부족으로 실패한 `approved_misrecognition_rate_upper_95` 하나만 감사 가능한 manual waiver로 기록했습니다. 독립 이미지 사후 검증 의무와 수치 `0.2669% > 0.1%`를 package에서 제거하지 않습니다. 이 1.0 이력은 위 1.1 point-rate override의 합격 여부에 포함하지 않지만 통계 위험 진단으로 계속 보고합니다.

schema 2.1 hardening package도 동일한 통계 gate를 새로 우회하지 않는다. 모델 바이너리가 같더라도 새 production 디렉터리 생성에는 명시적 통계 위험 승인이 필요하며, 승인 전 후보는 `development`로 유지한다. 승격은 같은 볼륨의 임시 디렉터리에서 검증한 뒤 rename하고 이미 존재하는 version 디렉터리는 덮어쓰지 않는다.

2026-08-19 프로젝트 소유자는 Bread 1.1.0 v3를 bridge release로 별도 승인했다. 이 package는
Classifier 최종 LDA가 `single_objects` 200장 외의 E/M/H ROI 1,410개로 fit된 점과 새 독립
잠금 test가 없다는 점을 각각 `classifier_training_source_restriction`,
`evaluation_set_independence` waiver로 기록한다. 두 개발 세트를 독립 증거로 바꾸지 않으며 이
예외는 1.1.1 이상에 상속되지 않는다. 승격 입력은
`configs/releases/bread_1.1.0_owner_waiver.json`이다.

원본 데이터, checkpoint, ONNX와 benchmark artifact는 Git에 넣지 않습니다. 결과 문서에는 모델·데이터셋·ONNX Runtime·CUDA·driver·hardware·warm-up과 표본 수를 기록합니다.
