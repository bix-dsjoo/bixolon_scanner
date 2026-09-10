"""Apply a user-revised GT-object objective to unchanged HTTP measurements."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import ceil
from pathlib import Path

from ..configuration import load_json_config


def load_objective(path: Path) -> dict:
    settings = load_json_config(path)
    if (
        settings.get("schema_version") != 1
        or settings.get("accuracy_unit") != "ground_truth_object"
        or settings.get("maximum_wrong_approved_count") != 0
    ):
        raise ValueError("unsupported object approval objective")
    for key in ("development_ground_truth_count", "final_ground_truth_count"):
        required_correct_count(settings[key], settings["minimum_correct_approved_rate"])
    return settings


def required_correct_count(ground_truth_count: int, minimum_rate: float) -> int:
    rate = Fraction(str(minimum_rate))
    if ground_truth_count < 1 or not 0 < rate <= 1:
        raise ValueError("object accuracy requires positive GT count and a rate in (0, 1]")
    return ceil(ground_truth_count * rate)


def object_accuracy(summary: dict, settings: dict, *, final: bool) -> dict:
    expected = settings["final_ground_truth_count" if final else "development_ground_truth_count"]
    total = summary["ground_truth_count"]
    correct = summary["correct_approved_count"]
    if total != expected or not 0 <= correct <= total:
        raise ValueError("GT object denominator or correct approval count changed")
    minimum = required_correct_count(total, settings["minimum_correct_approved_rate"])
    return {
        "accuracy_unit": "ground_truth_object",
        "ground_truth_count": total,
        "correct_approved_count": correct,
        "correct_approved_rate": correct / total,
        "minimum_correct_approved_rate": settings["minimum_correct_approved_rate"],
        "minimum_correct_approved_count": minimum,
        "accuracy_target_met": correct >= minimum
        and summary["wrong_approved_count"] == settings["maximum_wrong_approved_count"],
    }


def rank_candidate(measurement: dict, settings: dict) -> dict:
    result = deepcopy(measurement)
    real, stress = (measurement["diagnostics"][key] for key in ("real", "stress"))
    if any(r["image_status_counts"].get("ERROR", 0) for r in (real, stress)):
        raise ValueError("a candidate with execution errors cannot be ranked")
    accuracy = object_accuracy(real, settings, final=False)
    wrong = real["wrong_approved_count"] + stress["wrong_approved_count"]
    latency = measurement["cpu_latency"]
    eligible = accuracy["accuracy_target_met"] and wrong == 0 and latency["target_met"]
    result["image_objective_rank"] = result["rank"]
    result["rank"] = (
        0 if eligible else 1,
        wrong,
        -real["correct_approved_count"],
        -stress["correct_approved_count"],
        real["missed_count"] + stress["missed_count"],
        real["unknown_top3_miss_count"] + stress["unknown_top3_miss_count"],
        latency["worst_full_path_p95_ms"],
    )
    result["object_objective"] = {**accuracy, "development_target_met": eligible}
    return result


def apply_final_objective(measurement: dict, settings: dict) -> dict:
    """Retain raw responses/timing and annotate each repeat with the frozen objective."""
    result = deepcopy(measurement)
    for summary in result["repetitions"]:
        summary.update(object_accuracy(summary, settings, final=True))
        summary["target_met"] = (
            summary["accuracy_target_met"] and summary["speed_target_met"] is not False
        )
    result["summary"] = result["repetitions"][0]
    result["target_met_all_repetitions"] = all(r["target_met"] for r in result["repetitions"])
    result["objective"] = settings
    return result
