# Scanner 2.0 선택적 독립 인증 가이드

Scanner `2.0.0`은 2026-08-20 프로젝트 소유자의 명시적 예외 승인으로 production에 승격됐다.
새 owner-private test는 운영 승격이나 신규 매장 onboarding의 필수 조건이 아니다. 고객에게 다중
상품 사진, hidden test, Catalog key 또는 서명을 요구하지 않는다.

현재 release는 다음 사실을 함께 공개한다.

- `production=true`
- `promotion_basis=owner_manual_waiver`
- `independent_certified=false`
- 300장과 2026-08-18 운영 115장은 모두 개발·캘리브레이션 계보
- Catalog authentication은 영구 무키 `CHECKSUM-SHA256`

향후 소유자가 독립 일반화 성능을 별도로 알고 싶을 때만 이 절차를 사용한다. 결과는 현재 release의
추가 evidence일 뿐 Catalog 활성화나 운영 지속 여부를 자동으로 바꾸지 않는다.

## 독립 bundle 계약

이미지 원본과 review가 끝난 GT를 다음 두 잠금 파일로 만든다.

- `private-manifest.jsonl`: image SHA-256, dHash, capture session, physical object, bbox, class와 기대 상태
- `private-plan.json`: immutable dataset revision, manifest SHA-256과 사전 선언한 certification trial

같은 촬영 burst/session 또는 같은 물리 객체의 파생 이미지는 하나의 group으로 묶는다. RC.8 개발
identity 전체와 exact 또는 dHash Hamming distance 2 이하로 겹치는 이미지는 독립 표본에서 제외한다.
preflight는 모델이나 ONNX session을 만들기 전에 이 조건과 release attestation 정합성을 검사해야 한다.

## 권장 지표

| 영역 | 보고 기준 |
|---|---:|
| `SEGMENTATION / eligible image` | point ≥ 90% |
| `correct APPROVED / judgeable GT` | point ≥ 95%, 장기 목표 99% 별도 |
| `wrong APPROVED / approved output` | point와 단측 95% upper ≤ 0.1% |
| `SEGMENTATION` 이미지 FN·FP | 각각 point와 단측 95% upper ≤ 0.1% |
| forced Top-3 Candidate out | point와 단측 95% upper ≤ 0.1% |
| OOD false `APPROVED` | point와 단측 95% upper ≤ 0.1% |
| image recapture recall | point와 단측 95% lower ≥ 99% |
| 불필요 `IMAGE_RECAPTURE` | point와 단측 95% upper ≤ 1% |
| invalid-ROI `SEGMENT_RECAPTURE` recall | point와 단측 95% lower ≥ 99% |

상관된 ROI를 독립 표본으로 부풀리지 않는다. 0건 오류로 단측 95% upper 0.1%를 입증하려면 해당
endpoint에 최소 2,995개의 독립 trial이 필요하다. 표본이 부족하면 point metric과 별도로
`insufficient_evidence`를 기록한다.

## 실행 원칙

1. production attestation과 Runtime·Catalog content manifest를 먼저 검증한다.
2. 전체 개발 identity와의 exact/dHash 중복을 모델 실행 전에 차단한다.
3. frozen `2.0.0` Runtime, decision policy와 production Catalog를 변경하지 않는다.
4. Catalog는 키 없이 로딩하고 모든 파일별 SHA-256과 source manifest를 검증한다.
5. 결과에는 aggregate metric, confidence bound, version과 artifact hash만 기록한다.
6. 이미지 bytes/path, per-image logits와 embedding은 보고서나 기본 로그에 남기지 않는다.
7. 결과를 보고 model, crop, threshold 또는 ranking을 바꾸면 그 bundle은 이후 개발 evidence다.

독립 인증을 추가하더라도 `configs/releases/scanner_2.0.0.json`의 production 구성과 영구 무키
정책은 유지한다. 새 모델 정책을 선택하면 별도 semantic version과 새 release record를 만든다.
