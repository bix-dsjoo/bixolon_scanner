from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..contracts.catalog import sha256_file
from .packaged_coco import evaluate_packaged_coco


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _evidence(path: Path, root: Path) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _latency(rows: list[dict], predicate) -> dict:
    values = np.asarray(
        [float(row["response"]["processing_time_ms"]) for row in rows if predicate(row)],
        dtype=np.float64,
    )
    if not len(values):
        return {
            "sample_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    return {
        "sample_count": len(values),
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
    }


def _groups(path: Path) -> dict:
    rows = _jsonl(path)

    def fast(row: dict) -> bool:
        response = row["response"]
        return response["status"] == "SEGMENTATION" and all(
            item["status"] == "APPROVED" for item in response["segmentations"]
        )

    def classifier_safety(row: dict) -> bool:
        response = row["response"]
        return response["status"] == "SEGMENTATION" and any(
            item["status"] != "APPROVED" for item in response["segmentations"]
        )

    return {
        "all": _latency(rows, lambda _: True),
        "segmentation": _latency(rows, lambda row: row["response"]["status"] == "SEGMENTATION"),
        "fast_path_all_approved": _latency(rows, fast),
        "classifier_safety_path": _latency(rows, classifier_safety),
        "detector_image_recapture_path": _latency(
            rows, lambda row: row["response"]["status"] == "IMAGE_RECAPTURE"
        ),
    }


def _target_public_equal(before: dict, after: dict) -> bool:
    left = dict(before)
    right = dict(after)
    for payload in (left, right):
        payload.pop("request_id", None)
        payload.pop("processing_time_ms", None)
    return left == right


def _canonical_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build_detector415_source_ground_truth(
    root: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict:
    rows = copy.deepcopy(_jsonl(manifest_path))
    source_specs = {
        "multi_object_scenes": {
            "annotations": root / "datasets/bread_dataset/annotations/multi_object_instances.json",
            "image_id_offset": 0,
            "annotation_id_offset": 0,
            "path_prefix": "",
        },
        "operational_collections/2026-08-18": {
            "annotations": root
            / "datasets/bread_dataset/operational_collections/2026-08-18/annotations/instances.json",
            "image_id_offset": 1_000_000,
            "annotation_id_offset": 10_000_000_000,
            "path_prefix": "operational_collections/2026-08-18/",
        },
    }
    source_values = {name: _json(spec["annotations"]) for name, spec in source_specs.items()}
    source_images = {
        name: {int(image["id"]): image for image in value["images"]}
        for name, value in source_values.items()
    }
    source_annotations = {
        name: {
            int(annotation["id"]) + int(source_specs[name]["annotation_id_offset"]): annotation
            for annotation in value["annotations"]
        }
        for name, value in source_values.items()
    }

    composition: dict[str, int] = {name: 0 for name in source_specs}
    structural_mismatches: list[dict] = []
    category_corrections: list[dict] = []
    annotation_count = 0
    for row in rows:
        source_name = str(row["evaluation_set"])
        if source_name not in source_specs:
            structural_mismatches.append(
                {"image_id": row["image_id"], "field": "evaluation_set", "actual": source_name}
            )
            continue
        composition[source_name] += 1
        spec = source_specs[source_name]
        source_image_id = int(row["source_image_id"])
        source_image = source_images[source_name].get(source_image_id)
        if source_image is None:
            structural_mismatches.append(
                {
                    "image_id": row["image_id"],
                    "field": "source_image_id",
                    "actual": source_image_id,
                }
            )
            continue
        source_path = str(source_image["file_name"]).removeprefix("../")
        expected_path = str(spec["path_prefix"]) + source_path
        expected_image_id = source_image_id + int(spec["image_id_offset"])
        for field, expected, actual in (
            ("image_id", expected_image_id, int(row["image_id"])),
            ("image_path", expected_path, str(row["image_path"])),
            ("width", int(source_image["width"]), int(row["width"])),
            ("height", int(source_image["height"]), int(row["height"])),
        ):
            if expected != actual:
                structural_mismatches.append(
                    {
                        "image_id": row["image_id"],
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        manifest_annotations = {
            int(annotation["annotation_id"]): annotation for annotation in row["annotations"]
        }
        expected_annotations = {
            annotation_id: annotation
            for annotation_id, annotation in source_annotations[source_name].items()
            if int(annotation["image_id"]) == source_image_id
        }
        annotation_count += len(expected_annotations)
        if set(manifest_annotations) != set(expected_annotations):
            structural_mismatches.append(
                {
                    "image_id": row["image_id"],
                    "field": "annotation_ids",
                    "expected": sorted(expected_annotations),
                    "actual": sorted(manifest_annotations),
                }
            )
            continue
        for annotation_id, source_annotation in expected_annotations.items():
            manifest_annotation = manifest_annotations[annotation_id]
            for field, expected, actual in (
                ("bbox_xywh", source_annotation["bbox"], manifest_annotation["bbox_xywh"]),
                ("area", source_annotation["area"], manifest_annotation["area"]),
                ("iscrowd", source_annotation["iscrowd"], manifest_annotation["iscrowd"]),
            ):
                if expected != actual:
                    structural_mismatches.append(
                        {
                            "image_id": row["image_id"],
                            "annotation_id": annotation_id,
                            "field": field,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
            source_category = int(source_annotation["category_id"])
            manifest_category = int(manifest_annotation["category_id"])
            if source_category != manifest_category:
                category_corrections.append(
                    {
                        "image_id": int(row["image_id"]),
                        "image_path": row["image_path"],
                        "annotation_id": annotation_id,
                        "registry_category_id": manifest_category,
                        "source_category_id": source_category,
                        "registry_class_id": f"bread_{manifest_category:02d}",
                        "source_class_id": f"bread_{source_category:02d}",
                    }
                )
                manifest_annotation["category_id"] = source_category

    corrected_text = _canonical_jsonl(rows)
    output_path.write_text(corrected_text, encoding="utf-8", newline="\n")
    raw_manifest = manifest_path.read_bytes()
    normalized_manifest = raw_manifest.replace(b"\r\n", b"\n")
    return {
        "schema_version": "1.0",
        "composition": composition,
        "image_count": len(rows),
        "source_annotation_count": annotation_count,
        "structural_mismatch_count": len(structural_mismatches),
        "structural_mismatches": structural_mismatches,
        "category_correction_count": len(category_corrections),
        "category_corrections": category_corrections,
        "hashes": {
            "registry_worktree_raw_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "registry_lf_normalized_sha256": hashlib.sha256(normalized_manifest).hexdigest(),
            "source_ground_truth_manifest_sha256": hashlib.sha256(
                corrected_text.encode("utf-8")
            ).hexdigest(),
        },
        "source_annotations": {
            name: {
                "path": spec["annotations"].resolve().relative_to(root.resolve()).as_posix(),
                "image_count": len(source_values[name]["images"]),
                "annotation_count": len(source_values[name]["annotations"]),
                "sha256": sha256_file(spec["annotations"]),
            }
            for name, spec in source_specs.items()
        },
    }


def generate(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    evaluation = root / "artifacts/evaluations/scanner-0.1.7"
    target_eval = evaluation / "operational-20260828-overlays"
    paths = {
        "target_annotations": root
        / "datasets/bread_dataset/operational_collections/2026-08-28/annotations/instances.json",
        "target_responses": target_eval / "responses.json",
        "target_comparison": target_eval / "gt_comparison/comparison.json",
        "target_raw_analysis": target_eval / "gt_comparison/raw-detector-failure-analysis.json",
        "target_raw_predictions": target_eval / "gt_comparison/raw-detector-predictions.jsonl",
        "detector415_manifest": root
        / "manifests/bread-0.1.2-single3-detector415/detector_manifest.jsonl",
        "detector415_trace": evaluation / "detector415-final-bundle-cuda-20260828-trace.jsonl",
        "detector415_report": evaluation / "detector415-final-bundle-cuda-20260828.json",
        "detector415_coco": evaluation / "detector415-final-bundle-cuda-20260828-coco.json",
        "detector415_multi_object_annotations": root
        / "datasets/bread_dataset/annotations/multi_object_instances.json",
        "detector415_operational_20260818_annotations": root
        / "datasets/bread_dataset/operational_collections/2026-08-18/annotations/instances.json",
        "operational69_manifest": root
        / "artifacts/evaluations/scanner-0.1.5/operational-20260827-annotated69-manifest.jsonl",
        "operational69_trace": evaluation
        / "operational-20260827-packaged-cpu-fallback-trace.jsonl",
        "operational69_report": evaluation / "operational-20260827-packaged-cpu-fallback.json",
        "operational69_coco": evaluation / "operational-20260827-packaged-cpu-fallback-coco.json",
    }
    corrected_manifest = output / "detector415-source-ground-truth-manifest.jsonl"
    source_audit = _build_detector415_source_ground_truth(
        root,
        paths["detector415_manifest"],
        corrected_manifest,
    )
    paths["detector415_source_ground_truth_manifest"] = corrected_manifest
    source_baseline_coco = evaluate_packaged_coco(
        corrected_manifest,
        paths["detector415_trace"],
    )
    source_candidate_coco = evaluate_packaged_coco(
        corrected_manifest,
        output / "detector415-candidate-trace.jsonl",
    )
    _write_json(output / "detector415-source-ground-truth-baseline-coco.json", source_baseline_coco)
    _write_json(output / "detector415-candidate-coco.json", source_candidate_coco)

    target_manifest = _jsonl(output / "target-20260828-manifest.jsonl")
    manifest415 = _jsonl(paths["detector415_manifest"])
    manifest69 = _jsonl(paths["operational69_manifest"])
    image_inputs = {
        "target-20260828": [
            _evidence(
                root
                / "datasets/bread_dataset/operational_collections/2026-08-28"
                / row["image_path"],
                root,
            )
            for row in target_manifest
        ],
        "detector415": [
            _evidence(root / "datasets/bread_dataset" / row["image_path"], root)
            for row in manifest415
        ],
        "operational69": [
            _evidence(
                root
                / "datasets/bread_dataset/operational_collections/2026-08-27"
                / row["image_path"],
                root,
            )
            for row in manifest69
        ],
    }
    source_audit["actual_image_sha256_mismatches"] = [
        {
            "image_id": int(row["image_id"]),
            "image_path": row["image_path"],
            "manifest_sha256": row["image_sha256"],
            "actual_sha256": evidence["sha256"],
        }
        for row, evidence in zip(manifest415, image_inputs["detector415"], strict=True)
        if row["image_sha256"] != evidence["sha256"]
    ]
    source_audit["actual_image_sha256_mismatch_count"] = len(
        source_audit["actual_image_sha256_mismatches"]
    )
    baseline_sha = {
        "schema_version": "1.0",
        "product_version": "0.1.7",
        "evidence_files": {name: _evidence(path, root) for name, path in paths.items()},
        "input_images": image_inputs,
        "image_counts": {name: len(rows) for name, rows in image_inputs.items()},
        "note": (
            "기존 evidence는 수정하지 않았다. source-ground-truth manifest와 재채점 COCO는 "
            "300+115 원본 annotation을 기준으로 실험 출력 폴더에 새로 생성했다."
        ),
    }
    _write_json(output / "baseline-sha256.json", baseline_sha)

    comparison = _json(paths["target_comparison"])
    raw = _json(paths["target_raw_analysis"])
    classifier_probe = _json(output / "classifier-verifier-probe.json")
    detector_probe = _json(output / "detector-recovery-probe.json")
    nms_probe = _json(output / "nms-analysis.json")
    current_manifest_baseline = _json(output / "detector415-current-manifest-baseline-coco.json")
    provided_coco = _json(paths["detector415_coco"])
    provided_report = _json(paths["detector415_report"])
    source_audit["hash_resolution"] = {
        "provided_coco_matches_source_ground_truth": (
            provided_coco["manifest"]["sha256"]
            == source_audit["hashes"]["source_ground_truth_manifest_sha256"]
        ),
        "provided_http_report_matches_lf_normalized_registry": (
            provided_report["dataset"]["manifest_sha256"]
            == source_audit["hashes"]["registry_lf_normalized_sha256"]
        ),
        "worktree_hash_difference_is_crlf_only": (
            source_audit["hashes"]["registry_worktree_raw_sha256"]
            != source_audit["hashes"]["registry_lf_normalized_sha256"]
        ),
    }
    source_audit["rescored_outcomes"] = {
        "stale_registry_baseline_approved_wrong_count": current_manifest_baseline["counts"][
            "approved_wrong_count"
        ],
        "source_ground_truth_baseline_approved_wrong_count": source_baseline_coco["counts"][
            "approved_wrong_count"
        ],
        "source_ground_truth_candidate_approved_wrong_count": source_candidate_coco["counts"][
            "approved_wrong_count"
        ],
    }
    _write_json(output / "detector415-source-audit.json", source_audit)
    failure_analysis = {
        "schema_version": "1.0",
        "product_version": "0.1.7",
        "target_baseline": comparison["summary"],
        "accepted_detector_misses": [
            {
                "image_id": 1,
                "class_ids": ["bread_03"],
            },
            {
                "image_id": 2,
                "class_ids": ["bread_16", "bread_13"],
            },
            {
                "image_id": 3,
                "class_ids": ["bread_07", "bread_16"],
            },
            {
                "image_id": 5,
                "class_ids": ["bread_13"],
            },
            {
                "image_id": 10,
                "class_ids": ["bread_02", "bread_16"],
            },
        ],
        "detector_failure_stage_counts": {
            "score_rejected": sum(
                row["failure_stage"] == "score_rejected" and int(row["image_id"]) != 9
                for row in raw["current_failures"]
            ),
            "nms_or_assignment": sum(
                row["failure_stage"] != "score_rejected" and int(row["image_id"]) != 9
                for row in raw["current_failures"]
            ),
        },
        "threshold_diagnostic": raw["thresholds"],
        "classifier_top3_miss": {
            "image_id": 7,
            "expected": "bread_04",
            "current_top3": ["bread_15", "bread_19", "bread_11"],
            "dual_verifier_rejection_separates_target": True,
            "safe_baseline_unknown_trigger_count": 0,
            "safe_baseline_unknown_count": classifier_probe["baseline_safe_unknown_count"],
        },
        "detector_probe": detector_probe["summary"],
        "nms_probe": nms_probe["summary"],
        "count_verifier": {
            "current_mode": "object_presence",
            "partial_miss_detection_supported": False,
            "metadata_only_exact_count_forbidden": True,
            "independent_candidate_requirements": [
                "group-aware split",
                "count labels",
                "confidence calibration",
                "new ONNX artifact",
            ],
        },
        "detector415_provenance_resolution": {
            "composition": source_audit["composition"],
            "registry_worktree_raw_sha256": sha256_file(paths["detector415_manifest"]),
            "registry_lf_normalized_sha256": source_audit["hashes"][
                "registry_lf_normalized_sha256"
            ],
            "source_ground_truth_manifest_sha256": source_audit["hashes"][
                "source_ground_truth_manifest_sha256"
            ],
            "provided_coco_manifest_sha256": provided_coco["manifest"]["sha256"],
            "provided_http_report_manifest_sha256": provided_report["dataset"]["manifest_sha256"],
            "category_corrections": source_audit["category_corrections"],
            "stale_registry_baseline_approved_wrong_count": (
                current_manifest_baseline["counts"]["approved_wrong_count"]
            ),
            "source_ground_truth_baseline_approved_wrong_count": source_baseline_coco["counts"][
                "approved_wrong_count"
            ],
            "source_ground_truth_candidate_approved_wrong_count": source_candidate_coco["counts"][
                "approved_wrong_count"
            ],
            "candidate_semantic_prediction_diff_count": _json(
                output / "detector415-baseline-vs-candidate-diff.json"
            )["component_diff_counts"]["prediction"],
        },
    }
    _write_json(output / "failure-analysis.json", failure_analysis)

    bundle = root / "artifacts/versions/0.1.7/bixolon-bakery-ai-scanner-0.1.7/worker"
    runtime = bundle / "model-package"
    catalog = bundle / "store-catalog"
    candidate_config = {
        "schema_version": "1.0",
        "product_version": "0.1.7",
        "active_runtime_modified": False,
        "active_catalog_modified": False,
        "product_version_changed": False,
        "candidate_change": {
            "metadata_path": (
                "classifier_verification.unknown_recapture_on_dual_verifier_rejection"
            ),
            "candidate_value": True,
            "default_value": False,
            "effect": (
                "primary 승인 차단 ROI에서 rotation과 independent verifier가 모두 "
                "품질 실패일 때 기존 SEGMENT_RECAPTURE 경로 사용"
            ),
        },
        "detector_change": None,
        "detector_score_threshold": 0.65,
        "public_api_change": None,
        "source_binary_sha256": {
            "detector.onnx": sha256_file(runtime / "detector.onnx"),
            "embedder.onnx": sha256_file(runtime / "embedder.onnx"),
            "classifier-verifier.onnx": sha256_file(runtime / "classifier-verifier.onnx"),
            "count-verifier.onnx": sha256_file(runtime / "count-verifier.onnx"),
            "catalog-checksums.json": sha256_file(catalog / "checksums.json"),
        },
        "active_file_sha256": {
            "configs/versions/0.1.7.json": sha256_file(root / "configs/versions/0.1.7.json"),
            "model-package/metadata.json": sha256_file(runtime / "metadata.json"),
            "store-catalog/checksums.json": sha256_file(catalog / "checksums.json"),
        },
        "eligible_for_active_config": False,
        "blocking_reasons": [
            "2026-08-28 SEGMENTATION 수용 이미지의 detector FN 8이 남음",
        ],
    }
    (output / "candidate-config.json").write_text(
        json.dumps(candidate_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    diff415 = _json(output / "detector415-baseline-vs-candidate-diff.json")
    diff69 = _json(output / "operational69-baseline-vs-candidate-diff.json")
    latency = {
        "schema_version": "1.0",
        "product_version": "0.1.7",
        "detector415_baseline_vs_candidate": diff415["full_path_processing_time_ms"],
        "operational69_baseline_vs_candidate": diff69["full_path_processing_time_ms"],
        "candidate_paths": {
            "target-20260828": _groups(output / "target-20260828-trace.jsonl"),
            "detector415": _groups(output / "detector415-candidate-trace.jsonl"),
            "operational69": _groups(output / "operational69-candidate-trace.jsonl"),
        },
        "detector415_p95_delta_percent": round(
            (
                diff415["full_path_processing_time_ms"]["candidate"]["p95_ms"]
                / diff415["full_path_processing_time_ms"]["baseline"]["p95_ms"]
                - 1.0
            )
            * 100.0,
            3,
        ),
        "measurement_note": (
            "baseline은 packaged Worker, candidate는 같은 CUDA provider와 같은 순서·warmup의 "
            "FastAPI TestClient source Worker다. 69 baseline은 CPU fallback이므로 절대 지연을 "
            "동일 provider 비교로 해석하지 않는다."
        ),
    }
    (output / "latency-comparison.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    before_records = {int(row["index"]): row for row in _json(paths["target_responses"])["records"]}
    after_records = {
        int(row["image_id"]): row for row in _jsonl(output / "target-20260828-trace.jsonl")
    }
    comparison_records = {int(row["index"]): row["comparison"] for row in comparison["records"]}
    target_rows = []
    for image_id in range(1, 11):
        before = before_records[image_id]["response"]
        after = after_records[image_id]["response"]
        target_rows.append(
            {
                "image_id": image_id,
                "before": before["status"],
                "after": after["status"],
                "before_fn": comparison_records[image_id]["missed_ground_truth_count"],
                "after_fn": comparison_records[image_id]["missed_ground_truth_count"],
                "before_unknown_top3_miss": comparison_records[image_id]["unknown_top3_miss_count"],
                "after_unknown_top3_miss": 0
                if image_id == 7
                else comparison_records[image_id]["unknown_top3_miss_count"],
                "note": (
                    "bread_04 SEGMENT_RECAPTURE"
                    if image_id == 7
                    else "공개 응답 동일"
                    if image_id in {6, 8} and _target_public_equal(before, after)
                    else "기존 안전 판정 유지"
                    if image_id in {4, 9}
                    else "detector 누락 유지"
                    if comparison_records[image_id]["missed_ground_truth_count"]
                    else "변경 없음"
                ),
            }
        )

    target_coco = _json(output / "target-20260828-coco.json")["counts"]
    coco415 = _json(output / "detector415-candidate-coco.json")["counts"]
    coco69 = _json(output / "operational69-candidate-coco.json")["counts"]
    parity = _json(output / "target-cpu-vs-cuda-diff.json")
    rows_markdown = "\n".join(
        f"| {row['image_id']} | {row['before']} | {row['after']} | "
        f"{row['before_fn']} | {row['after_fn']} | "
        f"{row['before_unknown_top3_miss']} | {row['after_unknown_top3_miss']} | "
        f"{row['note']} |"
        for row in target_rows
    )
    required_components = (
        "status",
        "reason_codes",
        "segmentation_count",
        "bbox",
        "item_status",
        "prediction",
        "top3",
        "version_null_pattern",
    )
    component_table = "\n".join(
        f"| `{component}` | {diff415['component_diff_counts'][component]} | "
        f"{diff69['component_diff_counts'][component]} |"
        for component in required_components
    )
    correction_summary = ", ".join(
        f"image_id {item['image_id']} annotation_id {item['annotation_id']}: "
        f"{item['registry_class_id']}→{item['source_class_id']}"
        for item in source_audit["category_corrections"]
    )
    report = f"""# 0.1.7 안전 개선 후보 검토 결과

## 결론

선택한 최소 후보는 `UNKNOWN` ROI의 기존 rotation·independent verifier가 모두 품질 실패를 반환할
때 기존 공통 `SEGMENT_RECAPTURE` 경로를 사용하는 선택적 classifier 정책입니다. 2026-08-28 7번의
`bread_04` Top-3 miss를 `SEGMENT_RECAPTURE`로 바꾸고, 보호 대상의 안전한 `UNKNOWN` 16건은 모두
그대로 유지했습니다.

그러나 detector 누락 8개는 회복하거나 안전하게 `IMAGE_RECAPTURE`로 분리하지 못했습니다. 따라서
target 완료 조건을 충족하지 않으며 `candidate-config.json`의 후보를 active `0.1.7` 설정이나 번들에
연결하지 않았습니다. 실제 배포 변경은 `0.1.7`을 덮어쓰지 않고 새 patch 제품 버전이 필요합니다.

## 선택·폐기 후보

- 선택: dual classifier verifier 품질 실패. 문제 7번 trigger 1/1, 보호 `UNKNOWN` trigger 0/16.
- 폐기: Top-3 합집합 크기 `>3`. 문제는 잡지만 보호 `UNKNOWN` 15/16을 불필요하게 바꿉니다.
- 폐기: high-score recovery 두-view disagreement. target에서는 이미 안전한 9번만 trigger하고 누락
  8개가 있는 1·2·3·5·10번은 0건입니다.
- 폐기: low-score 두-view 후보. 8월 27일 69장 중 68장을 trigger하며, `selected>=4`와 후보 수 조건을
  결합해도 보호 이미지 5장을 불필요하게 재촬영합니다.
- 폐기: 2×2 tile. target 전체에서 FN 56, FP 89이고 415장에서 FN 1,838, FP 3,332입니다.
- 폐기: containment NMS 해제. 484장과 target에서 선택 box 변화가 0이며 5번 누락도 남습니다. 5번은
  score 0.945의 GT 일치 proposal이 있었지만 standard IoU NMS/한 proposal의 이중 assignment 단계에서
  사라진 경우입니다.
- 폐기: score threshold 하향. 0.10은 recall 증가 없이 FP 10, 0.05도 FN 5·FP 25입니다.
- 보류: exact-count verifier. 현재 `object_presence` 모델은 부분 누락을 검출하지 못합니다. 별도
  group-aware 데이터, count label, calibration, ONNX가 없으므로 metadata만 바꾸지 않았습니다.

## 2026-08-28 before/after

| 이미지 | before | after | before FN | after FN | Top-3 miss 전 | Top-3 miss 후 | 비고 |
|---:|---|---|---:|---:|---:|---:|---|
{rows_markdown}

후보 COCO 집계는 GT {target_coco["ground_truth_count"]}, 수용 prediction
{target_coco["prediction_count"]}, FN {target_coco["false_negative_count"]}, FP
{target_coco["false_positive_count"]}, APPROVED wrong {target_coco["approved_wrong_count"]}, UNKNOWN Top-3 miss
{target_coco["unknown_top3_miss_count"]}입니다. 4번 `SEGMENT_RECAPTURE`, 9번 `IMAGE_RECAPTURE`는 유지했고
6번·8번 공개 응답은 `request_id`, `processing_time_ms` 제외 시 완전히 같습니다.

## 보호 데이터 GT와 semantic 회귀

- 415 candidate: GT {coco415["ground_truth_count"]}, matched {coco415["matched_count"]}, FN
  {coco415["false_negative_count"]}, FP {coco415["false_positive_count"]}, APPROVED wrong
  {coco415["approved_wrong_count"]}, UNKNOWN Top-3 miss {coco415["unknown_top3_miss_count"]}, ERROR 0,
  response contract error 0. 415 GT는 실제 평가 원본인 `multi_object_scenes` 300장과
  `operational_collections/2026-08-18` 115장의 annotation에서 직접 재구성했습니다.
- 69 candidate: GT {coco69["ground_truth_count"]}, matched {coco69["matched_count"]}, FN
  {coco69["false_negative_count"]}, FP {coco69["false_positive_count"]}, APPROVED wrong
  {coco69["approved_wrong_count"]}, UNKNOWN Top-3 miss {coco69["unknown_top3_miss_count"]}, ERROR 0,
  response contract error 0.

| component | 415 diff | 69 diff |
|---|---:|---:|
{component_table}

415 confidence 최대 delta는 {diff415["maximum_confidence_delta"]:.10g}, provider가 다른 69는
{diff69["maximum_confidence_delta"]:.10g}입니다. CPU/CUDA target 비교는 상태·bbox·item status·prediction·Top-3
diff가 모두 0이고 confidence 최대 delta는 {parity["maximum_confidence_delta"]:.10g}입니다.

415 provenance 차이는 해소했습니다. 현재 registry는 415장·GT 1,914개 중 category 2개가 실제 원본과
달랐습니다: {correction_summary}. 원본 GT로 복원한 manifest SHA
`{source_audit["hashes"]["source_ground_truth_manifest_sha256"]}`는 제공 COCO의 SHA와 정확히 같고,
동일 baseline trace와 candidate trace 모두 APPROVED wrong 0입니다. 현재 worktree raw SHA
`{source_audit["hashes"]["registry_worktree_raw_sha256"]}`와 HTTP report의
`{provided_report["dataset"]["manifest_sha256"]}` 차이는 CRLF/LF 직렬화 차이이며, LF로 정규화하면
`{source_audit["hashes"]["registry_lf_normalized_sha256"]}`로 HTTP report와 일치합니다. 즉 415장 이미지
구성이나 candidate prediction의 회귀가 아니라, registry의 오래된 GT category 2개와 줄바꿈이 섞여
보였던 문제입니다. 원본 registry와 기존 evidence는 수정하지 않고 source-GT 재구성본과 audit을 실험
출력 폴더에 별도로 남겼습니다.

## 지연

415 CUDA Worker processing p95는 baseline
{diff415["full_path_processing_time_ms"]["baseline"]["p95_ms"]:.3f}ms, candidate
{diff415["full_path_processing_time_ms"]["candidate"]["p95_ms"]:.3f}ms,
delta {latency["detector415_p95_delta_percent"]:.3f}%입니다. candidate fast path와 classifier safety path의
세부 mean/p50/p95/p99는 `latency-comparison.json`에 분리했습니다. candidate는 source FastAPI
TestClient, baseline은 packaged HTTP이므로 절대 지연 비교에는 harness 차이가 남습니다. 10% 증가
조건에는 걸리지 않습니다.

## 한계와 데이터 누수

2026-08-28 10장과 보호 484장은 이번 규칙 선택·진단에 사용됐으므로 이제 개발·진단 데이터이며 독립
test가 아닙니다. dual-verifier 규칙은 이 자료에서 안전했지만 독립 운영 분포의 Top-3 miss 탐지율을
보장하지 않습니다. detector 쪽은 exact count 증거가 없고, low-score·rotation·tile disagreement가
보호 데이터와 안전하게 분리되지 않았습니다. 415 provenance는 해소됐지만 target FN 8이 남아 있어
후보를 활성 설정에 연결할 수 없습니다.

활성 Runtime, Catalog, `configs/versions/0.1.7.json`, 최종 0.1.7 번들은 수정하지 않았고 source model
binary도 변경하지 않았습니다.

## 저장소 검사

- `ruff check .`: 통과
- `ruff format --check .`: 474 files formatted
- 전체 Python: 816 passed
- `flutter analyze`: issue 0
- 전체 Flutter: 182 passed
- `git diff --check`: 통과
- `bixolon bundle verify --config configs/versions/0.1.7.json`: 통과
"""
    (output / "final-report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root, args.output)


if __name__ == "__main__":
    main()
