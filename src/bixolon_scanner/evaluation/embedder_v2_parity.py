from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..runtime.catalog import OnnxEmbedder
from ..runtime.onnx import prepare_rgb
from ..training.models import (
    DINO_V3_CONVNEXT_TINY,
    DINO_V3_HUB_REPOSITORY,
    DINO_V3_VIT_BASE_16,
    build_dino_classifier,
)

DINO_V2_BASE = "dinov2_base"


def _representative_images(manifest: Path, dataset_root: Path) -> list[Image.Image]:
    records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    selected = {}
    for row in records:
        selected.setdefault(str(row["class_id"]), row)
    images = []
    for row in selected.values():
        with Image.open(dataset_root / row["image_path"]) as source:
            images.append(source.convert("RGB"))
    return images


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    runtime = load_runtime_package_v2(args.runtime)
    images = _representative_images(args.manifest, args.dataset_root)
    metadata = runtime.metadata.embedder
    tensors = np.stack(
        [
            prepare_rgb(
                image,
                metadata.input_size,
                metadata.mean,
                metadata.std,
                reducing_gap=metadata.resize_reducing_gap,
            )
            for image in images
        ]
    )
    if args.backbone == DINO_V2_BASE:
        from transformers import AutoModel

        if args.model_dir is None:
            raise ValueError("DINOv2 parity requires --model-dir")
        model = (
            AutoModel.from_pretrained(
                args.model_dir.resolve().as_posix(),
                local_files_only=True,
                attn_implementation="eager",
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            pytorch = (
                model(pixel_values=torch.from_numpy(tensors).cuda())
                .last_hidden_state[:, 0]
                .float()
                .cpu()
                .numpy()
            )
    else:
        if args.weights is None:
            raise ValueError("DINOv3 parity requires --weights")
        model = (
            build_dino_classifier(
                args.backbone,
                1,
                weights_path=args.weights,
                hub_repository=f"facebookresearch/dinov3:{args.source_revision}",
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            pytorch = model.extract_features(torch.from_numpy(tensors).cuda()).float().cpu().numpy()
    cpu = OnnxEmbedder(runtime, "cpu").embed_images_raw(images)
    cuda = OnnxEmbedder(runtime, "cuda", args.cuda_dll_dir).embed_images_raw(images)
    for image in images:
        image.close()

    def comparison(left: np.ndarray, right: np.ndarray) -> dict:
        left_norm = left / np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = right / np.linalg.norm(right, axis=1, keepdims=True)
        return {
            "maximum_absolute_error": float(np.max(np.abs(left - right))),
            "mean_absolute_error": float(np.mean(np.abs(left - right))),
            "minimum_cosine_similarity": float(np.min(np.sum(left_norm * right_norm, axis=1))),
        }

    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_embedder_parity",
        "backbone_kind": args.backbone,
        "sample_count": len(images),
        "pytorch_vs_onnx_cuda": comparison(pytorch, cuda),
        "onnx_cpu_vs_cuda": comparison(cpu, cuda),
    }
    report["passes"] = (
        report["pytorch_vs_onnx_cuda"]["maximum_absolute_error"] <= args.maximum_error
        and report["onnx_cpu_vs_cuda"]["maximum_absolute_error"] <= args.maximum_error
    )
    report["maximum_absolute_error_tolerance"] = args.maximum_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare DINO PyTorch/CPU/CUDA features")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=(DINO_V2_BASE, DINO_V3_CONVNEXT_TINY, DINO_V3_VIT_BASE_16),
        default=DINO_V2_BASE,
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--source-revision",
        default=DINO_V3_HUB_REPOSITORY.rsplit(":", maxsplit=1)[1],
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-error", type=float, default=1e-4)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
