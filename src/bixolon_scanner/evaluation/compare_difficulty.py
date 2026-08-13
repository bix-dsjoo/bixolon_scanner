from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DIFFICULTIES = ("E", "M", "H", "ALL")


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("by_difficulty"), dict) or not isinstance(
        report.get("overall"), dict
    ):
        raise ValueError(f"invalid difficulty report: {path}")
    return report


def flatten_report(condition: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        values = report["overall"] if difficulty == "ALL" else report["by_difficulty"][difficulty]
        statuses = values["response_status"]
        classified = int(values["classified_matched_boxes"])
        ground_truth = int(values["ground_truth_boxes"])
        rows.append(
            {
                "condition": condition,
                "package_version": report["package_version"],
                "difficulty": difficulty,
                "images": int(values["images"]),
                "ground_truth_boxes": ground_truth,
                "approved_images": int(statuses.get("APPROVED", 0)),
                "unknown_images": int(statuses.get("UNKNOWN", 0)),
                "recapture_images": int(statuses.get("RECAPTURE", 0)),
                "detector_matched_boxes": int(values["matched_boxes"]),
                "detector_missed_boxes": int(values["missed_boxes"]),
                "detector_false_positive_boxes": int(values["false_positive_boxes"]),
                "classifier_top1_accuracy_excluding_recapture": values["rates"].get(
                    "classifier_top1_accuracy_excluding_recapture",
                    values["rates"].get("top1_recognition_accuracy"),
                ),
                "unknown_top3_accuracy": values["rates"].get("unknown_top3_accuracy"),
                "recognized_approved_correct": int(values["approved_correct"]),
                "top3_candidate": int(values["unknown_top3_correct"]),
                "candidate_out": int(values["unknown_top3_missing"]),
                "approved_misclassification": int(values["approved_wrong"]),
                "recapture_ground_truth_boxes": int(
                    values.get("recapture_ground_truth_boxes", ground_truth - classified)
                ),
                "unblocked_segmentation_missed": int(
                    values.get(
                        "unblocked_missed_boxes",
                        max(0, int(values["missed_boxes"]) - (ground_truth - classified)),
                    )
                ),
                "mean_latency_ms": float(values["end_to_end_latency_ms"]["mean"]),
            }
        )
    return rows


def compare(
    reports: list[tuple[str, Path]],
    *,
    output_json: Path,
    output_csv: Path,
    output_markdown: Path,
    dataset_note: str,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one difficulty report is required")
    loaded = [(name, _read_report(path)) for name, path in reports]
    rows = [row for name, report in loaded for row in flatten_report(name, report)]
    summary = {
        "schema_version": "1.0",
        "evaluation": "bread_project_2_worker_difficulty_comparison",
        "dataset_root": loaded[0][1]["dataset_root"],
        "provider": loaded[0][1]["provider"],
        "match_iou_threshold": loaded[0][1]["match_iou_threshold"],
        "evaluation_passes_per_image": 1,
        "reference_condition": reports[0][0],
        "diagnostic_set_note": dataset_note,
        "selected_n": None,
        "promotion_status": "experiment_only",
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# bread_project_2 E/M/H Worker 비교",
        "",
        f"- 평가 성격: {dataset_note}",
        "- 이미지별 실행: 1회",
        "- 자동 N 선택 및 production 승격: 없음",
        "",
        "| 조건 | 난이도 | Top-1 | UNKNOWN Top-3 | Candidate out | APPROVED/UNKNOWN/RECAPTURE | 평균 지연 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        top1 = row["classifier_top1_accuracy_excluding_recapture"]
        top3 = row["unknown_top3_accuracy"]
        lines.append(
            f"| {row['condition']} | {row['difficulty']} | "
            f"{'N/A' if top1 is None else f'{top1:.2%}'} | "
            f"{'N/A' if top3 is None else f'{top3:.2%}'} | "
            f"{row['candidate_out']} | {row['approved_images']}/"
            f"{row['unknown_images']}/{row['recapture_images']} | "
            f"{row['mean_latency_ms']:.3f} ms |"
        )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _report_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must use CONDITION=PATH")
    condition, path = value.split("=", 1)
    if not condition or not path:
        raise argparse.ArgumentTypeError("report must use CONDITION=PATH")
    return condition, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E/M/H Worker reports")
    parser.add_argument("--report", type=_report_argument, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--dataset-note", required=True)
    args = parser.parse_args()
    compare(
        args.report,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_markdown=args.output_markdown,
        dataset_note=args.dataset_note,
    )


if __name__ == "__main__":
    main()
