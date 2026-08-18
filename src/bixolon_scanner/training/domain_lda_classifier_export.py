from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..contracts.model_package import sha256_file
from .fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from .models import require_torch


def transfer_margin_thresholds_by_quantile(
    oof_scores: np.ndarray,
    final_training_scores: np.ndarray,
    thresholds: Sequence[float | None],
) -> tuple[list[float | None], list[dict[str, Any]]]:
    oof = np.asarray(oof_scores, dtype=np.float64)
    final = np.asarray(final_training_scores, dtype=np.float64)
    if oof.ndim != 2 or final.ndim != 2 or oof.shape[1] != final.shape[1]:
        raise ValueError("OOF and final scores must have the same class dimension")
    if len(thresholds) != oof.shape[1]:
        raise ValueError("threshold count must match the classifier class count")

    def predictions_and_margins(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-values, axis=1, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=1)
        return order[:, 0], sorted_values[:, 0] - sorted_values[:, 1]

    oof_predictions, oof_margins = predictions_and_margins(oof)
    final_predictions, final_margins = predictions_and_margins(final)
    transferred: list[float | None] = []
    diagnostics = []
    for class_id, threshold in enumerate(thresholds):
        if threshold is None:
            transferred.append(None)
            continue
        oof_class_margins = oof_margins[oof_predictions == class_id]
        final_class_margins = final_margins[final_predictions == class_id]
        if not len(oof_class_margins) or not len(final_class_margins):
            raise ValueError(f"class {class_id} has no score margins for threshold transfer")
        rejection_count = int(np.count_nonzero(oof_class_margins <= float(threshold)))
        rejection_fraction = rejection_count / len(oof_class_margins)
        final_rank = max(0, min(len(final_class_margins) - 1, rejection_count - 1))
        final_threshold = float(np.sort(final_class_margins)[final_rank])
        transferred.append(float(np.nextafter(np.float32(final_threshold), np.float32(np.inf))))
        diagnostics.append(
            {
                "class_id": class_id,
                "oof_threshold": float(threshold),
                "oof_predicted_class_count": len(oof_class_margins),
                "oof_rejection_count": rejection_count,
                "oof_rejection_fraction": rejection_fraction,
                "final_training_predicted_class_count": len(final_class_margins),
                "final_training_threshold": transferred[-1],
            }
        )
    return transferred, diagnostics


def export(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classifier = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=f"facebookresearch/dinov3:{checkpoint['source_revision']}",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    classifier.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    classifier = classifier.eval()

    feature_cache = np.load(args.training_features)
    features = np.asarray(feature_cache[args.feature_family], dtype=np.float64)
    targets = np.asarray(feature_cache["targets"], dtype=np.int64)
    image_ids = np.asarray(feature_cache["image_ids"], dtype=np.int64)
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=args.shrinkage)
    lda.fit(features, targets)
    final_training_scores = lda.decision_function(features)

    oof_cache = np.load(args.oof_logits)
    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    final_thresholds, threshold_diagnostics = transfer_margin_thresholds_by_quantile(
        np.asarray(oof_cache["scores"], dtype=np.float64),
        final_training_scores,
        policy_report["policy"]["thresholds"],
    )

    class DomainLdaClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_model = classifier
            self.linear = torch.nn.Linear(lda.coef_.shape[1], lda.coef_.shape[0])
            with torch.no_grad():
                self.linear.weight.copy_(torch.from_numpy(lda.coef_.astype(np.float32)))
                self.linear.bias.copy_(torch.from_numpy(lda.intercept_.astype(np.float32)))

        def forward(self, pixel_values):
            raw = self.feature_model.extract_features(pixel_values)
            if isinstance(raw, tuple):
                adapted = self.feature_model.classifier.adapt(*raw)
            else:
                adapted = self.feature_model.classifier.adapt(raw)
            return self.linear(adapted)

    model = DomainLdaClassifier().eval()
    image_size = int(checkpoint["image_size"])
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(args.output))
    tensors = np.load(args.parity_tensors, mmap_mode="r")[: args.parity_samples].astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(tensors)).float().numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"pixel_values": tensors})[0]
    difference = np.abs(expected - actual)
    training_predictions = np.argmax(final_training_scores, axis=1)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_same_domain_lda_final_classifier_export",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "feature_cache": str(args.training_features),
        "training_sample_count": len(targets),
        "training_image_count": len(np.unique(image_ids)),
        "feature_family": args.feature_family,
        "lda_shrinkage": args.shrinkage,
        "training_top1_error_count_diagnostic": int(
            np.count_nonzero(training_predictions != targets)
        ),
        "model": str(args.output),
        "model_sha256": sha256_file(args.output),
        "model_size_bytes": args.output.stat().st_size,
        "dynamic_batch": True,
        "approval_metric": "logit_margin",
        "approval_thresholds": final_thresholds,
        "approval_threshold_transfer": "preserve_per_predicted_class_OOF_rejection_count",
        "threshold_diagnostics": threshold_diagnostics,
        "parity": {
            "provider": "CPUExecutionProvider",
            "sample_count": len(tensors),
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "top1_equal": bool(np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))),
            "top3_equal": bool(
                np.array_equal(
                    np.argsort(-expected, axis=1, kind="stable")[:, :3],
                    np.argsort(-actual, axis=1, kind="stable")[:, :3],
                )
            ),
        },
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "package integration, CUDA parity, latency, and locked test pending",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the final same-domain LDA classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--oof-logits", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--feature-family", choices=["raw", "adapted"], default="adapted")
    parser.add_argument("--shrinkage", type=float, default=0.01)
    parser.add_argument("--parity-tensors", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=20)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
