# Scanner 2.0 owner-private test 실행 가이드

> 현재 `scanner-2.0.1`은 2026-08-20 프로젝트 소유자의 명시적 `manual_waiver`로 이미 운영
> 승격됐다. 이 문서는 독립 certification을 추가하거나 다음 정상 승격을 수행할 때의 엄격한
> 절차다. 현재 production 상태를 취소하지 않지만 `independent_certified=false` 제한도 제거하지
> 않는다. 운영 `2.0.1` Catalog는 영구 무키 `CHECKSUM-SHA256` 계약을 사용한다.

이 절차는 공용 Runtime `2.0.0-rc.7`의 마지막 승격 gate다. 신규 매장 onboarding 절차가 아니며,
고객에게 다중 상품 사진을 요구하지 않는다. private bundle은 프로젝트 소유자가 한 번 제공하고,
model-free preflight 뒤 각 이미지를 정확히 한 번만 추론한다.

## 제공할 것

소유자는 이미지 원본과 확정 GT만 준비하면 된다. 저장소 작업자는 이를 다음 두 잠금 파일로 만든다.

- `private-manifest.jsonl`: 이미지당 한 줄
- `private-plan.json`: dataset revision과 사전 선언된 certification trial

manifest 한 줄의 계약은 다음과 같다.

```json
{
  "image_id": "private-000001",
  "image_path": "images/private-000001.jpg",
  "image_sha256": "64자리 sha256",
  "perceptual_hash": "16자리 dhash",
  "store_id": "bread-project-2",
  "capture_session_id": "session-000001",
  "expected_image_status": "SEGMENTATION",
  "annotations": [
    {
      "annotation_id": "object-000001",
      "bbox_xywh": [100, 200, 300, 400],
      "target_class_id": "bread_01",
      "physical_object_id": "physical-000001",
      "catalog_membership": "in_catalog",
      "expected_item_status": "APPROVED"
    }
  ]
}
```

이미지 전체 재촬영 GT는 `expected_image_status=IMAGE_RECAPTURE`, `annotations=[]`로 기록한다.
미등록/OOD 객체는 `catalog_membership=ood`, `expected_item_status=SEGMENT_RECAPTURE`여야 한다.
잘못된 ROI·국소 품질 대상도 `expected_item_status=SEGMENT_RECAPTURE`로 명시한다.

plan은 release lock과 manifest SHA-256을 먼저 고정하고 trial을 선언한다.

```json
{
  "schema_version": "2.0",
  "dataset_id": "scanner-2-owner-private-v1",
  "immutable_revision": "revision-0001",
  "review_status": "locked",
  "release_lock_sha256": "release-lock의 lock_sha256",
  "manifest_sha256": "private-manifest.jsonl의 sha256",
  "image_count": 3300,
  "store_count": 1,
  "trials": [
    {
      "endpoint": "approval_safety",
      "group_id": "physical-000001",
      "image_id": "private-000001",
      "annotation_id": "object-000001"
    }
  ]
}
```

숫자 `3000`은 형식 예시일 뿐 고정 이미지 수가 아니다. 필요한 수는 독립 certification group 수와
한 이미지의 객체 수에 따라 달라진다.

## Certification trial

| endpoint | 단위 | zero-error 최소 trial | 성공 조건 |
|---|---|---:|---|
| `approval_safety` | 물리 객체 | 2,995 | 실제 `APPROVED` 출력의 class가 GT와 같음 |
| `detector_fn` | 촬영 session | 2,995 | eligible 이미지가 `SEGMENTATION`이고 FN 없음 |
| `detector_fp` | 촬영 session | 2,995 | unmatched detection 없음 |
| `top3_safety` | 물리 객체 | 2,995 | forced Top-3에 GT class 포함 |
| `ood_false_approval` | 물리 객체 | 2,995 | OOD가 `APPROVED`되지 않음 |
| `image_recapture_recall` | 촬영 session | 299 | recapture GT가 `IMAGE_RECAPTURE` |
| `unnecessary_image_recapture` | 촬영 session | 299 | eligible 이미지가 `IMAGE_RECAPTURE`가 아님 |
| `invalid_roi_action` | 물리 객체 | 299 | 대상 ROI가 `SEGMENT_RECAPTURE` |

같은 `capture_session_id` 또는 `physical_object_id`를 여러 group ID로 나눌 수 없다. 반대로 서로
다른 provenance를 같은 group ID로 합칠 수도 없다. preflight가 endpoint별 일대일 관계를 검사한다.
`approval_safety`의 통계 분모는 선언된 trial 중 실제 `APPROVED`가 나온 독립 group이므로, 승인
coverage가 낮아 2,995개보다 작으면 사전 trial 수를 채웠더라도 `insufficient_evidence`다.

## Model-free preflight

```powershell
bixolon evaluate scanner-2.0-private-preflight `
  --release-lock artifacts/releases/scanner-2.0.0-rc.7-pre-private/release-lock.json `
  --plan C:/private/scanner-2/private-plan.json `
  --manifest C:/private/scanner-2/private-manifest.jsonl `
  --dataset-root C:/private/scanner-2 `
  --development-identity-manifest artifacts/experiments/scanner-2.0.0/catalog-single-objects-2-manifest/detector_manifest.jsonl `
  --development-identity-manifest artifacts/experiments/scanner-2.0.0/catalog-single-objects-2-manifest/classifier_manifest.jsonl `
  --development-identity-manifest manifests/bread-zero-error-1.1/classifier_manifest.jsonl `
  --development-identity-manifest artifacts/experiments/bread-training-source-comparison/manifests/single_objects_1/manifest.jsonl `
  --development-identity-manifest artifacts/experiments/bread-training-source-comparison/manifests/single_objects_3/manifest.jsonl `
  --development-identity-manifest manifests/bread-zero-error-1.1/rejected_operational_v3_development_identity.jsonl `
  --development-identity-manifest artifacts/releases/scanner-2.0.0-rc.7-pre-private/development-identity-lineage.jsonl `
  --output C:/private/scanner-2/private-preflight.json
```

이 단계는 먼저 실행 중인 preflight source가 release lock의 hash와 같은지 확인하며 모델이나 ONNX
session을 생성하지 않는다. 모든 이미지 SHA-256을 확인하고 dHash를 실제
이미지에서 다시 계산하며, 전체 개발 계보와 Hamming distance 2 이하인 이미지를 거부한다. 이미지
경로·ID나 bytes는 report에 쓰지 않는다.

## 단 한 번의 production evaluation

서명 key는 명령 인수가 아니라 접근 통제된 환경 변수로 주입한다. `run-state`는 exclusive create로
선점되며 시작·실패·완료 어느 상태에서도 같은 private bundle 재실행을 거부하는 감사 기록이다.

```powershell
$env:BIXOLON_CATALOG_SIGNING_KEY = "<trusted-secret>"

bixolon evaluate scanner-2.0-private `
  --release-lock artifacts/releases/scanner-2.0.0-rc.7-pre-private/release-lock.json `
  --preflight-report C:/private/scanner-2/private-preflight.json `
  --plan C:/private/scanner-2/private-plan.json `
  --manifest C:/private/scanner-2/private-manifest.jsonl `
  --dataset-root C:/private/scanner-2 `
  --runtime artifacts/packages/bread-scanner-2.0.0-rc.7-runtime `
  --catalog artifacts/catalogs/bread-project-2/2.0.0-rc.7 `
  --warmup-image datasets/bread_dataset/multi_object_scenes/easy/easy_001.jpg `
  --output C:/private/scanner-2/private-production-report.json `
  --run-state C:/private/scanner-2/private-run-state.json `
  --store-id bread-project-2 `
  --key-id development-2026-08 `
  --provider cuda `
  --cuda-dll-dir C:/trusted/cuda-runtime
```

평가 report에는 요청 지표, production point gate, endpoint별 one-sided 95% Clopper-Pearson bound,
latency 집계와 최종 `production_eligible`만 기록한다. per-image trace는 생성하지 않는다. 실패하면
RC.7을 `rejected`로 보존하고 같은 private bundle을 후속 후보에 재사용하지 않는다.

## 통과 후 2.0.0 finalization

private report가 모든 gate를 통과했을 때만 다음 명령이 작동한다. model graph, weight, threshold,
crop이나 decision policy는 바꾸지 않고 component version, promotion status와 production Catalog
서명만 결정적으로 갱신한다.

```powershell
$env:BIXOLON_PRODUCTION_CATALOG_SIGNING_KEY = "<production-secret>"

bixolon release promote-scanner-2.0 `
  --release-lock artifacts/releases/scanner-2.0.0-rc.7-pre-private/release-lock.json `
  --private-report C:/private/scanner-2/private-production-report.json `
  --runtime artifacts/packages/bread-scanner-2.0.0-rc.7-runtime `
  --catalog artifacts/catalogs/bread-project-2/2.0.0-rc.7 `
  --output-root artifacts/releases/scanner-2.0.0-production `
  --store-id bread-project-2 `
  --production-key-id "<production-key-id>"
```

출력에는 `runtime/`, `catalog/`, `promotion-attestation.json`이 생성된다. 기존 경로를 덮어쓰지 않으며,
private gate와 다른 release lock을 참조하면 실패한다.
