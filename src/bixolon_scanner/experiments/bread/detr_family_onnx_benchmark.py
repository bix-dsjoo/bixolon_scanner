from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np

from ...training.models import require_torch
from ...training.ssdlite_objectness_detector import CachedObjectnessDataset
from .detr_family_comparison import HfObjectnessDataset, _latency_summary, _process_images


def run(args: argparse.Namespace) -> dict:
    import onnxruntime as ort
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    torch = require_torch()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, source_model) -> None:
            super().__init__()
            self.source_model = source_model

        def forward(self, pixel_values):
            outputs = self.source_model(pixel_values=pixel_values)
            return outputs.logits, outputs.pred_boxes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "detector.onnx"
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForObjectDetection.from_pretrained(args.model_dir).eval()
    dataset = HfObjectnessDataset(
        CachedObjectnessDataset(
            args.real_manifest,
            args.real_root,
            args.real_cache,
            training=False,
        )
    )
    first_image, _ = dataset[0]
    example = _process_images(processor, [first_image])["pixel_values"]
    fixed_position_encoding = False
    position_encoding_parity = None
    if model.config.model_type == "rf_detr":
        embeddings = model.model.backbone.backbone.embeddings
        original_interpolate = embeddings.interpolate_pos_encoding
        captured = {}

        def capture_position_encoding(self, values, height, width):
            result = original_interpolate(values, height, width)
            captured["value"] = result.detach()
            return result

        embeddings.interpolate_pos_encoding = MethodType(capture_position_encoding, embeddings)
        with torch.inference_mode():
            original_outputs = model(pixel_values=example)
        position_encoding = captured["value"]

        def use_fixed_position_encoding(self, values, height, width):
            del self, height, width
            return position_encoding.to(device=values.device, dtype=values.dtype)

        embeddings.interpolate_pos_encoding = MethodType(use_fixed_position_encoding, embeddings)
        with torch.inference_mode():
            fixed_outputs = model(pixel_values=example)
        position_encoding_parity = {
            "logits_max_abs_error": float(
                torch.max(torch.abs(original_outputs.logits - fixed_outputs.logits))
            ),
            "boxes_max_abs_error": float(
                torch.max(torch.abs(original_outputs.pred_boxes - fixed_outputs.pred_boxes))
            ),
        }
        fixed_position_encoding = True

    wrapper = ExportWrapper(model).eval()
    with torch.inference_mode():
        reference_logits, reference_boxes = wrapper(example)
    torch.onnx.export(
        wrapper,
        (example,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=args.dynamo,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.cpu_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx_logits, onnx_boxes = session.run(
        None,
        {"pixel_values": example.detach().cpu().numpy()},
    )
    parity = {
        "logits_max_abs_error": float(
            np.max(np.abs(reference_logits.detach().cpu().numpy() - onnx_logits))
        ),
        "boxes_max_abs_error": float(
            np.max(np.abs(reference_boxes.detach().cpu().numpy() - onnx_boxes))
        ),
    }

    def infer(index: int) -> None:
        image, _ = dataset[index]
        values = _process_images(processor, [image])["pixel_values"].numpy()
        session.run(None, {"pixel_values": values})

    for index in range(args.warmup_count):
        infer(index % len(dataset))
    latency = []
    for index in range(len(dataset)):
        started = time.perf_counter()
        infer(index)
        latency.append((time.perf_counter() - started) * 1000.0)

    report = {
        "model_dir": str(args.model_dir),
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": args.opset,
        "dynamo_exporter": args.dynamo,
        "provider": session.get_providers()[0],
        "cpu_threads": args.cpu_threads,
        "warmup_count": args.warmup_count,
        "scope": "processor+ONNX Runtime model forward, fixed 320x320, batch=1",
        "fixed_position_encoding": fixed_position_encoding,
        "position_encoding_parity": position_encoding_parity,
        "parity": parity,
        "latency": _latency_summary(latency),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and benchmark a trained DETR candidate")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path(
            "artifacts/experiments/scanner-0.1.7-safety-20260828/"
            "detector415-source-ground-truth-manifest.jsonl"
        ),
    )
    parser.add_argument("--real-root", type=Path, default=Path("datasets/bread_dataset"))
    parser.add_argument(
        "--real-cache", type=Path, default=Path("artifacts/cache/ssdlite320-real415")
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--warmup-count", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
