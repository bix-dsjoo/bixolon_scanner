from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassAssistedSelectionPolicy:
    base_match_iou: float = 0.9
    candidate_minimum_support: int = 3
    group_relation_iou: float = 0.3
    group_area_ratio: float = 0.8
    group_margin_ratio: float = 0.5
    group_novel_margin: float = 0.2
    group_minimum_score: float = 0.04
    independent_maximum_iou: float = 0.3
    independent_margin: float = 0.5
    independent_minimum_score: float = 0.04
    proposal_minimum_score: float = 0.03
    add_maximum_iou: float = 0.5
    refinement_containment: float = 0.75
    refinement_minimum_score: float = 0.035
    refinement_maximum_support: int = 3
    refinement_maximum_area_ratio: float = 0.95
    refinement_minimum_iou: float = 0.25
    refinement_minimum_separation_gain: float = 0.3
    single_object_expansion_minimum_containment: float | None = None
    single_object_expansion_minimum_area_ratio: float | None = None
    single_object_expansion_minimum_approval_score: float | None = None
    single_object_expansion_minimum_approval_margin: float | None = None


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_intersection(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def box_iou(left: list[float], right: list[float]) -> float:
    intersection = box_intersection(left, right)
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def containment(outer: list[float], inner: list[float]) -> float:
    inner_area = box_area(inner)
    return box_intersection(outer, inner) / inner_area if inner_area > 0.0 else 0.0


def _entry_margin(entry: dict[str, Any]) -> float:
    return float(entry.get("class_margin", entry.get("approval_score", 0.0)))


def _entry_index(entry: dict[str, Any]) -> int:
    return int(entry["proposal_index"])


def select_single_object_recovery(
    entries: list[dict[str, Any]],
    *,
    minimum_score: float,
    minimum_support: int,
    maximum_aspect_ratio: float,
    minimum_approval_score: float,
) -> dict[str, Any] | None:
    """Choose a compact whole-object proposal when the strict selector found none."""
    candidates = []
    for entry in entries:
        box = entry["box"]
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        if width <= 0.0 or height <= 0.0:
            continue
        aspect_ratio = max(width / height, height / width)
        if (
            float(entry["detector_score"]) >= minimum_score
            and int(entry.get("support_count", 0)) >= minimum_support
            and aspect_ratio <= maximum_aspect_ratio
            and _entry_margin(entry) >= minimum_approval_score
        ):
            candidates.append(entry)
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda entry: (
            box_area(entry["box"]),
            _entry_margin(entry),
            float(entry["detector_score"]),
        ),
    )
    return {
        "boxes_xyxy": [list(map(float, selected["box"]))],
        "scores": [float(selected["detector_score"])],
        "class_ids": [int(selected["predicted_class"])],
        "proposal_indices": [_entry_index(selected)],
    }


def _map_base_entries(
    base: dict[str, Any],
    raw: dict[str, Any],
    entries_by_index: dict[int, dict[str, Any]],
    policy: ClassAssistedSelectionPolicy,
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for base_box in base["boxes_xyxy"]:
        overlaps = [box_iou(base_box, raw_box) for raw_box in raw["boxes_xyxy"]]
        if not overlaps:
            continue
        raw_index = max(range(len(overlaps)), key=overlaps.__getitem__)
        if overlaps[raw_index] < policy.base_match_iou:
            raise ValueError("base detector box cannot be mapped to the proposal union")
        entry = entries_by_index.get(raw_index)
        if entry is not None and int(raw["support_counts"][raw_index]) >= (
            policy.candidate_minimum_support
        ):
            mapped.append(entry)
    return mapped


def class_verified_select(
    base: dict[str, Any],
    raw: dict[str, Any],
    entries: list[dict[str, Any]],
    policy: ClassAssistedSelectionPolicy,
    *,
    maximum_count: int | None = None,
) -> list[dict[str, Any]]:
    """Select spatial proposals using detector support and classifier identity.

    The function is label-free and operates only on deployable detector and
    classifier outputs. It preserves at most one base proposal per predicted
    class, then splits a likely group box or adds one independent novel class.
    """

    entries_by_index = {_entry_index(entry): entry for entry in entries}
    mapped = _map_base_entries(base, raw, entries_by_index, policy)
    by_class: dict[int, dict[str, Any]] = {}
    for entry in mapped:
        class_id = int(entry["predicted_class"])
        current = by_class.get(class_id)
        if current is None or float(entry["detector_score"]) > float(current["detector_score"]):
            by_class[class_id] = entry
    selected = list(by_class.values())
    used = {_entry_index(entry) for entry in selected}

    for base_entry in list(selected):
        if maximum_count is not None and len(selected) >= maximum_count:
            break
        base_entry_area = box_area(base_entry["box"])
        if base_entry_area <= 0.0:
            continue
        current_classes = {int(entry["predicted_class"]) for entry in selected}
        alternatives = []
        novel = []
        for candidate in entries:
            if _entry_index(candidate) in used:
                continue
            if box_iou(candidate["box"], base_entry["box"]) < policy.group_relation_iou:
                continue
            if box_area(candidate["box"]) > base_entry_area * policy.group_area_ratio:
                continue
            if candidate["predicted_class"] == base_entry["predicted_class"]:
                if _entry_margin(candidate) >= (
                    _entry_margin(base_entry) * policy.group_margin_ratio
                ):
                    alternatives.append(candidate)
            elif (
                int(candidate["predicted_class"]) not in current_classes
                and _entry_margin(candidate) >= policy.group_novel_margin
                and float(candidate["detector_score"]) >= policy.group_minimum_score
            ):
                novel.append(candidate)
        if not alternatives or not novel:
            continue
        alternative = max(alternatives, key=lambda row: float(row["detector_score"]))
        compatible = [
            row for row in novel if box_iou(row["box"], alternative["box"]) < policy.add_maximum_iou
        ]
        if not compatible:
            continue
        novel_entry = max(compatible, key=lambda row: float(row["detector_score"]))
        selected.remove(base_entry)
        selected.extend((alternative, novel_entry))
        used.discard(_entry_index(base_entry))
        used.update((_entry_index(alternative), _entry_index(novel_entry)))

    current_classes = {int(entry["predicted_class"]) for entry in selected}
    independent = [
        entry
        for entry in entries
        if _entry_index(entry) not in used
        and int(entry["predicted_class"]) not in current_classes
        and _entry_margin(entry) >= policy.independent_margin
        and float(entry["detector_score"]) >= policy.independent_minimum_score
        and (
            not selected
            or max(box_iou(entry["box"], other["box"]) for other in selected)
            < policy.independent_maximum_iou
        )
    ]
    if independent and (maximum_count is None or len(selected) < maximum_count):
        best_class = int(max(independent, key=_entry_margin)["predicted_class"])
        same_class = [entry for entry in independent if int(entry["predicted_class"]) == best_class]
        selected.append(max(same_class, key=lambda row: box_area(row["box"])))

    selected.sort(key=lambda row: float(row["detector_score"]), reverse=True)
    return selected if maximum_count is None else selected[:maximum_count]


def _add_until_count(
    selected: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    target_count: int,
    policy: ClassAssistedSelectionPolicy,
) -> None:
    used = {_entry_index(entry) for entry in selected}
    while len(selected) < target_count:
        candidates = [
            entry
            for entry in entries
            if _entry_index(entry) not in used
            and float(entry["detector_score"]) >= policy.proposal_minimum_score
            and int(entry.get("support_count", 0)) >= policy.candidate_minimum_support
            and (
                not selected
                or max(box_iou(entry["box"], other["box"]) for other in selected)
                < policy.add_maximum_iou
            )
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda entry: (
                float(entry["detector_score"])
                * (0.25 + float(entry.get("approval_score", 0.0)))
                * (0.75 + 0.25 * int(entry.get("support_count", 0)))
            ),
        )
        selected.append(best)
        used.add(_entry_index(best))


def _refine_merged_localization(
    selected: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    policy: ClassAssistedSelectionPolicy,
) -> bool:
    used = {_entry_index(entry) for entry in selected}
    best_replacement: tuple[float, int, dict[str, Any]] | None = None
    for index, base_entry in enumerate(selected):
        others = [entry for other_index, entry in enumerate(selected) if other_index != index]
        if not others:
            continue
        current_containment = max(containment(base_entry["box"], other["box"]) for other in others)
        if current_containment < (policy.refinement_containment):
            continue
        base_area = box_area(base_entry["box"])
        for candidate in entries:
            if _entry_index(candidate) in used:
                continue
            if candidate["predicted_class"] != base_entry["predicted_class"]:
                continue
            if float(candidate["detector_score"]) < policy.refinement_minimum_score:
                continue
            if int(candidate.get("support_count", 0)) > policy.refinement_maximum_support:
                continue
            if box_area(candidate["box"]) >= base_area * policy.refinement_maximum_area_ratio:
                continue
            if box_iou(candidate["box"], base_entry["box"]) <= policy.refinement_minimum_iou:
                continue
            candidate_containment = max(
                containment(candidate["box"], other["box"]) for other in others
            )
            gain = current_containment - candidate_containment
            if gain <= policy.refinement_minimum_separation_gain:
                continue
            ranked = (gain, index, candidate)
            if best_replacement is None or ranked[0] > best_replacement[0]:
                best_replacement = ranked
    if best_replacement is None:
        return False
    _, index, candidate = best_replacement
    selected[index] = candidate
    return True


def _refine_single_object_expansion(
    selected: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    policy: ClassAssistedSelectionPolicy,
) -> bool:
    if len(selected) != 1 or any(
        value is None
        for value in (
            policy.single_object_expansion_minimum_containment,
            policy.single_object_expansion_minimum_area_ratio,
            policy.single_object_expansion_minimum_approval_score,
            policy.single_object_expansion_minimum_approval_margin,
        )
    ):
        return False
    base = selected[0]
    base_area = box_area(base["box"])
    base_approval = _entry_margin(base)
    candidates = [
        candidate
        for candidate in entries
        if _entry_index(candidate) != _entry_index(base)
        and candidate["predicted_class"] == base["predicted_class"]
        and float(candidate["detector_score"]) >= policy.proposal_minimum_score
        and int(candidate.get("support_count", 0)) >= policy.candidate_minimum_support
        and containment(candidate["box"], base["box"])
        >= policy.single_object_expansion_minimum_containment
        and box_area(candidate["box"])
        >= base_area * policy.single_object_expansion_minimum_area_ratio
        and _entry_margin(candidate) >= policy.single_object_expansion_minimum_approval_score
        and _entry_margin(candidate)
        >= base_approval + policy.single_object_expansion_minimum_approval_margin
    ]
    if not candidates:
        return False
    selected[0] = max(
        candidates,
        key=lambda candidate: (
            _entry_margin(candidate),
            containment(candidate["box"], base["box"]),
            box_area(candidate["box"]),
            float(candidate["detector_score"]),
        ),
    )
    return True


def count_and_class_assisted_select(
    base: dict[str, Any],
    raw: dict[str, Any],
    entries: list[dict[str, Any]],
    target_count: int,
    policy: ClassAssistedSelectionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair an ambiguous detector result using count and per-proposal classes."""

    active_policy = policy or ClassAssistedSelectionPolicy()
    entries_by_index = {_entry_index(entry): entry for entry in entries}
    if len(base["boxes_xyxy"]) != target_count:
        selected = class_verified_select(
            base,
            raw,
            entries,
            active_policy,
            maximum_count=target_count,
        )
        mode = "count_mismatch"
    else:
        selected = _map_base_entries(base, raw, entries_by_index, active_policy)
        mode = "localization_check"
        if len(selected) != len(base["boxes_xyxy"]):
            return base | {"proposal_indices": []}, {
                "mode": "base_preserved",
                "base_count": len(base["boxes_xyxy"]),
                "target_count": target_count,
                "selected_count": len(base["boxes_xyxy"]),
                "localization_refined": False,
            }
    _add_until_count(selected, entries, target_count, active_policy)
    refined = False
    if len(selected) == target_count:
        refined = _refine_single_object_expansion(selected, entries, active_policy)
        refined = _refine_merged_localization(selected, entries, active_policy) or refined
    selected.sort(key=lambda row: float(row["detector_score"]), reverse=True)
    result = {
        "boxes_xyxy": [list(map(float, row["box"])) for row in selected],
        "scores": [float(row["detector_score"]) for row in selected],
        "class_ids": [int(row["predicted_class"]) for row in selected],
        "proposal_indices": [_entry_index(row) for row in selected],
    }
    diagnostics = {
        "mode": mode,
        "base_count": len(base["boxes_xyxy"]),
        "target_count": target_count,
        "selected_count": len(selected),
        "localization_refined": refined,
    }
    return result, diagnostics
