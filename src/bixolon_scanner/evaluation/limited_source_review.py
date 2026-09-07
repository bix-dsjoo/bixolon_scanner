"""Turn full-Worker measurements into bounded, traceable improvement experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..training.limited_source import verify_sources, write_json


def review_measurement(report: dict, source: dict, config: dict) -> dict:
    dataset = report["dataset"]
    if dataset["manifest_sha256"] != source["detector_manifest_sha256"]:
        raise ValueError("diagnostic measurement used an unlocked or different image set")
    if dataset["image_count"] != source["multi_count"] or dataset.get("held_out_test_set"):
        raise ValueError(
            "same-item diagnostics must use the locked scene count and cannot claim held-out test"
        )
    counts, metrics = report["counts"], report["metrics"]
    full_path = report["performance"]["full_path"]
    values = {
        "approved_error_count": counts["approved_misrecognition_count"]
        + counts["approved_false_positive_count"],
        "false_negative_count": counts["false_negative_count"],
        "unknown_top3_miss_count": counts["unknown_candidate_out_count"],
        "correct_approved_rate": metrics["correct_approved_rate"],
        "full_path_p95_ms": full_path["p95_ms"],
    }
    missing_latency = values["full_path_p95_ms"] is None
    if missing_latency and counts["segmentation_image_count"] != 0:
        raise ValueError("missing full-path latency for measured segmentations")
    if any(
        not math.isfinite(float(value)) or value < 0
        for value in values.values()
        if value is not None
    ):
        raise ValueError("invalid diagnostic metric")
    if values["correct_approved_rate"] > 1:
        raise ValueError("correct approval rate exceeds one")
    provider = report["environment"]["provider"]
    if provider not in {"cpu", "cuda"}:
        raise ValueError("this experiment compares the CPU and CUDA Worker profiles")
    targets = config["diagnostic_targets"]
    failures = []
    for metric in ("approved_error_count", "false_negative_count", "unknown_top3_miss_count"):
        if values[metric] > targets[f"maximum_{metric}"]:
            failures.append(metric)
    if values["correct_approved_rate"] < targets["minimum_correct_approved_rate"]:
        failures.append("correct_approved_rate")
    if (
        missing_latency
        or values["full_path_p95_ms"] > targets[f"maximum_{provider}_full_path_p95_ms"]
    ):
        failures.append("full_path_p95_ms")
    # Zero full-path samples cannot satisfy latency or coverage targets.
    if counts["segmentation_image_count"] == 0:
        failures.append("no_full_path_samples")
    actions = []
    if "approved_error_count" in failures:
        actions.append(
            "오승인 ROI의 정답 bbox·classifier Top-3·verifier 불일치를 분리하고, 승인 차단 및 잘못된 crop 원인을 먼저 수정한다."
        )
    if "false_negative_count" in failures:
        actions.append(
            "검출 전 raw FN과 조기 재촬영을 구분한다. 원본 220장 안에서 크기·겹침 분포와 합성 visible bbox를 개선하고 Detector를 같은 고정 epoch로 다시 학습한다."
        )
    if "unknown_top3_miss_count" in failures:
        actions.append(
            "정답 ROI와 Detector ROI를 비교하여 localization 오류와 표현 오류를 분리한다. 기존 원본의 scene crop·neighbor mask 일치 학습을 후보로 추가한다."
        )
    if "correct_approved_rate" in failures:
        actions.append(
            "전수 UNKNOWN/RECAPTURE로 안전성을 확보한 후보를 성공으로 간주하지 않는다. detail 전환·품질 거부·오분류를 나누고 head-only와 낮은 학습률 후보를 비교 진단한다."
        )
    if "full_path_p95_ms" in failures or "no_full_path_samples" in failures:
        actions.append(
            "primary/detail/verifier 호출 ROI 수와 decode 시간을 측정한다. 같은 이미지·모델로 draft 크기 및 전처리 재사용 후보를 CPU/CUDA 각각 반복 측정한다."
        )
    rank = [
        values["approved_error_count"],
        values["false_negative_count"],
        values["unknown_top3_miss_count"],
        -values["correct_approved_rate"],
        values["full_path_p95_ms"] if not missing_latency else 1e308,
    ]
    return {
        "schema_version": "1.0",
        "provider": provider,
        "metrics": values,
        "diagnostic_rank": rank,
        "failed_targets": failures,
        "all_development_targets_met": not failures,
        "next_experiments": actions,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "evaluation_role": source["evaluation_role"],
        "automatic_model_selection": False,
        "threshold_selection_allowed": False,
        "limitation": "같은 실물의 학습 이미지 재측정이다. 독립 validation/test와 일반화 성능을 대신하지 않는다.",
    }


def record_review(work: Path, measurement: Path, history_directory: Path | None = None) -> dict:
    source = verify_sources(work / "sources")
    plan = load_json_config(work / "plan.json")
    config = load_json_config(Path(plan["config"]))
    review = review_measurement(load_json_config(measurement), source, config)
    directory = history_directory or work / "reviews"
    previous = [load_json_config(path) for path in sorted(directory.glob("iteration-*.json"))]
    if any(row["source_manifest_sha256"] != source["source_manifest_sha256"] for row in previous):
        raise ValueError("review history contains different originals")
    digest = sha256_file(measurement)
    if any(row["measurement_sha256"] == digest for row in previous):
        raise ValueError("this measurement was already reviewed")
    comparable = [row for row in previous if row["provider"] == review["provider"]]
    improved = not comparable or review["diagnostic_rank"] < min(
        r["diagnostic_rank"] for r in comparable
    )
    non_improvements = 0 if improved else comparable[-1]["consecutive_non_improvements"] + 1
    iteration = config["iteration"]
    review.update(
        {
            "iteration": len(previous) + 1,
            "measurement_sha256": digest,
            "measurement": str(measurement.resolve()),
            "work_dir": str(work.resolve()),
            "experiment_config_sha256": sha256_file(Path(plan["config"])),
            "development_diagnostic_improved": improved,
            "consecutive_non_improvements": non_improvements,
            "stop_and_reassess": (
                len(comparable) + 1 >= iteration["maximum_candidates"]
                or non_improvements >= iteration["maximum_consecutive_non_improvements"]
            ),
        }
    )
    write_json(directory / f"iteration-{len(previous) + 1:03d}.json", review)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a locked-source Worker diagnostic and suggest next experiments"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument(
        "--history-dir", type=Path, help="Shared history across candidate work directories"
    )
    args = parser.parse_args()
    print(record_review(args.work_dir.resolve(), args.measurement.resolve(), args.history_dir))


if __name__ == "__main__":
    main()
