"""Source-only fixed-epoch QAT after evidence-constrained PTQ restored every layer."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .structural_student import build_student, cpu_session, evidence_loss
from .three_bakery_data import write_json
from .three_bakery_detector import seed_everything


def normalize_per_tensor_qparams(model):
    """Represent scalar QDQ parameters as rank zero for ORT QLinearMul fusion."""
    from onnx import numpy_helper

    result = copy.deepcopy(model)
    tensors = {tensor.name: tensor for tensor in result.graph.initializer}
    for node in result.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    tensors[node.output[0]] = attribute.t
        elif node.op_type == "Identity" and node.input[0] in tensors:
            tensors[node.output[0]] = tensors[node.input[0]]
    aliases = {}
    for node in result.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if any(attribute.name == "axis" for attribute in node.attribute):
            continue
        for index in range(1, min(3, len(node.input))):
            name = node.input[index]
            if name not in tensors:
                continue
            value = numpy_helper.to_array(tensors[name])
            if value.size != 1 or value.ndim == 0:
                continue
            if name not in aliases:
                alias = name + ".scalar"
                result.graph.initializer.append(numpy_helper.from_array(value.reshape(()), alias))
                aliases[name] = alias
            node.input[index] = aliases[name]
    return result, aliases


def run(config_path: Path, *, export_only: bool = False):
    import onnx
    import torch
    from timm.utils import reparameterize_model
    from torch.ao.quantization import (
        FakeQuantize,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        QConfig,
        QConfigMapping,
        disable_observer,
    )
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    settings = load_json_config(config_path)
    base = load_json_config(Path(settings["base_config"]))
    seed_everything(settings["seed"])
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(settings["source_student"])
    images = np.load(Path(base["output"]) / "student-cache/images.npy", mmap_mode="r")
    labels = np.load(Path(base["output"]) / "student-cache/targets.npz")
    model = build_student("repvit_m0_9.dist_450e_in1k", base["classifier"], pretrained=False)
    model.load_state_dict(torch.load(source / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    model.backbone = reparameterize_model(model.backbone)
    teacher = copy.deepcopy(model).cuda().eval()
    config = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=255,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            quant_min=-128,
            quant_max=127,
            ch_axis=0,
        ),
    )
    mapping = QConfigMapping().set_global(config)
    # Exclude named head modules from the global QAT configuration.
    for name in ["classifier", "integrity"]:
        mapping.set_module_name(name, None)
    model = prepare_qat_fx(model.train(), mapping, (torch.zeros(1, *images.shape[1:]),)).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.0001
    )
    history = []
    if export_only:
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        if len(history) != settings["epochs"] or history[-1]["epoch"] != settings["epochs"]:
            raise ValueError("Export resume requires every fixed training epoch")
        model.load_state_dict(
            torch.load(output / "model.pt", map_location="cuda", weights_only=True)
        )
    start = time.perf_counter()
    for epoch in range(0 if export_only else settings["epochs"]):
        model.train()
        if epoch >= settings["observer_epochs"]:
            model.apply(disable_observer)
        losses = []
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(images))
        for offset in range(0, len(order), settings["batch_size"]):
            indices = order[offset : offset + settings["batch_size"]]
            values = torch.from_numpy(np.array(images[indices], dtype=np.float32)).cuda()
            with torch.inference_mode():
                reference, quality = teacher(values)
            optimizer.zero_grad(set_to_none=True)
            embeddings, probability = model(values)
            # This is the exact two-class log-probability representation of the public quality output.
            probability = probability.clamp(1e-6, 1 - 1e-6)
            logits = torch.stack([torch.log1p(-probability), probability.log()], -1)
            loss = evidence_loss(
                embeddings,
                logits,
                torch.from_numpy(labels["categories"][indices]).cuda(),
                torch.from_numpy(labels["multiplicity"][indices]).cuda(),
                reference.clone(),
                quality.clone(),
                base["classifier"],
                "distilled",
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite QAT training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        write_json(output / "history.json", history)
        print(
            f"RepViT QAT epoch {epoch + 1}/{settings['epochs']}: {history[-1]['loss']:.6f}",
            flush=True,
        )
    model = model.cpu().eval()
    model.apply(disable_observer)
    if not export_only:
        torch.save(model.state_dict(), output / "model.pt")
    probe = torch.from_numpy(np.array(images[:16], dtype=np.float32))
    torch.onnx.export(
        model,
        probe[:1],
        output / "model-export-raw.onnx",
        opset_version=18,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["embeddings", "multi_object_probabilities"],
        dynamic_axes={
            "pixel_values": {0: "rois"},
            "embeddings": {0: "rois"},
            "multi_object_probabilities": {0: "rois"},
        },
    )
    graph, scalar_aliases = normalize_per_tensor_qparams(
        onnx.load(output / "model-export-raw.onnx")
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, output / "model.onnx")
    runner = cpu_session(output / "model.onnx")
    with torch.inference_mode():
        expected = [value.numpy() for value in model(probe)]
    actual = runner.run(None, {"pixel_values": probe.numpy()})
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    raw_runner = ort.InferenceSession(
        str(output / "model-export-raw.onnx"), options, providers=["CPUExecutionProvider"]
    )
    raw = raw_runner.run(None, {"pixel_values": probe.numpy()})
    parity_passed = all(
        np.allclose(a, b, atol=1e-3, rtol=1e-3) for a, b in zip(expected, actual, strict=True)
    )
    np.savez(
        output / "parity-probe.npz",
        pytorch_embeddings=expected[0],
        onnx_embeddings=actual[0],
        raw_onnx_embeddings=raw[0],
    )
    graph = onnx.load(output / "model.onnx")
    write_json(
        output / "report.json",
        {
            "config_sha256": sha256_file(config_path),
            "source_sha256": sha256_file(source / "model.pt"),
            "onnx_sha256": sha256_file(output / "model.onnx"),
            "history": history,
            "elapsed_seconds": time.perf_counter() - start,
            "sample_count": len(images),
            "elapsed_scope": "export resume only" if export_only else "training and export",
            "scalar_qparam_aliases": scalar_aliases,
            "quantize_nodes": sum(node.op_type == "QuantizeLinear" for node in graph.graph.node),
            "parity_probe_count": len(probe),
            "parity_max_error": [
                float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)
            ],
            "raw_onnx_parity_max_error": [
                float(np.max(np.abs(a - b))) for a, b in zip(expected, raw, strict=True)
            ],
            "parity_passed": parity_passed,
            "top1_differences": int(np.sum(expected[0].argmax(-1) != actual[0].argmax(-1))),
            "scope": settings["checkpoint_selection"],
            "deployment_selected": False,
        },
    )
    if not parity_passed:
        raise ValueError("QAT ONNX parity failed; candidate is not deployable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    run(args.config, export_only=args.export_only)
