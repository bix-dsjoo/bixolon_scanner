from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ids(bundle: np.lib.npyio.NpzFile) -> list[tuple[int, int]]:
    return list(zip(bundle["image_ids"].tolist(), bundle["annotation_ids"].tolist(), strict=True))


def _take(
    bundle: np.lib.npyio.NpzFile,
    identifiers: list[tuple[int, int]],
    array_name: str,
) -> np.ndarray:
    positions = {identifier: index for index, identifier in enumerate(_ids(bundle))}
    missing = [identifier for identifier in identifiers if identifier not in positions]
    if missing:
        raise ValueError(f"score bundle is missing {len(missing)} objects")
    return bundle[array_name][[positions[identifier] for identifier in identifiers]]


def _visibility(manifest: Path) -> dict[tuple[int, int], float]:
    output = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        for annotation in record.get("annotations", []):
            output[(int(record["image_id"]), int(annotation["annotation_id"]))] = float(
                annotation.get("visible_fraction", 1.0)
            )
    return output


def _baseline(
    paths: list[Path], arrays: list[str]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    bundles = [np.load(path) for path in paths]
    identifiers = sorted(set.intersection(*(set(_ids(bundle)) for bundle in bundles)))
    labels = _take(bundles[0], identifiers, "labels").astype(np.int64)
    predictions = []
    for bundle, array_name in zip(bundles, arrays, strict=True):
        if not np.array_equal(labels, _take(bundle, identifiers, "labels").astype(np.int64)):
            raise ValueError("baseline labels do not align")
        predictions.append(_take(bundle, identifiers, array_name).argmax(axis=1))
    votes = np.stack(predictions, axis=1)
    majority = np.apply_along_axis(lambda row: np.bincount(row, minlength=20).argmax(), 1, votes)
    return identifiers, labels, majority


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a binary quality score using source validation only"
    )
    parser.add_argument("--source-quality", type=Path, required=True)
    parser.add_argument("--target-quality", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-baseline", type=Path, action="append", required=True)
    parser.add_argument("--baseline-array", action="append", required=True)
    parser.add_argument("--partial-visible-threshold", type=float, default=0.45)
    parser.add_argument("--complete-visible-threshold", type=float, default=0.75)
    parser.add_argument("--maximum-complete-recap-rate", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.target_baseline) != 3 or len(args.baseline_array) != 3:
        parser.error("exactly three target baseline bundles and arrays are required")

    source = np.load(args.source_quality)
    target = np.load(args.target_quality)
    source_ids = _ids(source)
    visibility_by_id = _visibility(args.source_manifest)
    visibility = np.asarray([visibility_by_id[identifier] for identifier in source_ids])
    source_risk = source["quality_scores"].astype(np.float64)
    partial = visibility <= args.partial_visible_threshold
    complete = visibility >= args.complete_visible_threshold
    complete_risk = np.sort(source_risk[complete])
    allowed = int(np.floor(args.maximum_complete_recap_rate * len(complete_risk)))
    threshold = float(complete_risk[-(allowed + 1)])
    source_recapture = source_risk > threshold

    baseline_ids, labels, majority = _baseline(args.target_baseline, args.baseline_array)
    target_ids = sorted(set(_ids(target)) & set(baseline_ids))
    target_risk = _take(target, target_ids, "quality_scores").astype(np.float64)
    baseline_positions = {identifier: index for index, identifier in enumerate(baseline_ids)}
    indexes = [baseline_positions[identifier] for identifier in target_ids]
    labels = labels[indexes]
    majority = majority[indexes]
    errors = majority != labels
    recapture = target_risk > threshold
    report = {
        "schema_version": "1.0",
        "experiment": "source_calibrated_binary_quality_threshold",
        "development_target_used_for_selection": False,
        "selection": "source validation complete-object recap budget only",
        "partial_visible_threshold": args.partial_visible_threshold,
        "complete_visible_threshold": args.complete_visible_threshold,
        "maximum_complete_recap_rate": args.maximum_complete_recap_rate,
        "selected_threshold": threshold,
        "source_validation": {
            "object_count": int(len(source_ids)),
            "partial_count": int(partial.sum()),
            "complete_count": int(complete.sum()),
            "partial_recapture_count": int((source_recapture & partial).sum()),
            "complete_recapture_count": int((source_recapture & complete).sum()),
            "partial_recall": float((source_recapture & partial).sum() / max(1, partial.sum())),
            "complete_recap_rate": float(
                (source_recapture & complete).sum() / max(1, complete.sum())
            ),
        },
        "target_diagnostic": {
            "object_count": int(len(target_ids)),
            "baseline_error_count": int(errors.sum()),
            "recapture_count": int(recapture.sum()),
            "caught_error_count": int((recapture & errors).sum()),
            "remaining_error_count": int((~recapture & errors).sum()),
            "correct_recapture_count": int((recapture & ~errors).sum()),
            "correct_approved_count": int((~recapture & ~errors).sum()),
        },
        "target_recaptures": [
            {
                "image_id": int(target_ids[index][0]),
                "annotation_id": int(target_ids[index][1]),
                "was_error": bool(errors[index]),
                "expected": int(labels[index] + 1),
                "predicted": int(majority[index] + 1),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(recapture)
        ],
        "target_remaining_errors": [
            {
                "image_id": int(target_ids[index][0]),
                "annotation_id": int(target_ids[index][1]),
                "expected": int(labels[index] + 1),
                "predicted": int(majority[index] + 1),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(errors & ~recapture)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
