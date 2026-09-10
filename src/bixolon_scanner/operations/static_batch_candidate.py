"""Create an experimental static-batch copy, preserving every model weight."""

import json
import shutil
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file


def specialize_runtime(
    source: Path,
    destination: Path,
    *,
    primary_batch_size: int = 1,
    primary_batch_variants: tuple[int, ...] = (),
) -> dict:
    import onnx

    if primary_batch_size < 1:
        raise ValueError("Batch size must be positive")
    if primary_batch_variants and (
        1 not in primary_batch_variants
        or len(set(primary_batch_variants)) != len(primary_batch_variants)
        or any(size < 1 or size >= primary_batch_size for size in primary_batch_variants)
    ):
        raise ValueError("Variants must be unique smaller positive sizes including one")
    if destination.exists():
        raise ValueError("Candidate destination must be new")
    metadata = load_json_config(source / "metadata.json")
    shutil.copytree(source, destination)
    changes = []
    entries = [(metadata["embedder"], primary_batch_size)]
    variants = []
    for size in primary_batch_variants:
        original = metadata["embedder"]
        filename = str(Path(original["filename"]).with_suffix(f".batch{size}.onnx")).replace(
            "\\", "/"
        )
        if (destination / filename).exists():
            raise ValueError("Variant file already exists")
        shutil.copy2(source / original["filename"], destination / filename)
        entries.append(({**original, "filename": filename}, size))
        variants.append({"filename": filename, "batch_size": size})
    if variants:
        metadata["embedder"]["batch_variants"] = variants
    if metadata.get("classifier_resolution_fallback"):
        entries.append((metadata["classifier_resolution_fallback"]["embedder"], 1))
    if metadata.get("classifier_verification"):
        entries.append((metadata["classifier_verification"]["independent_embedder"], 1))
    for entry, batch in entries:
        path = destination / entry["filename"]
        original_hash = sha256_file(path)
        model = onnx.load(path)
        weights = [value.SerializeToString() for value in model.graph.initializer]
        nodes = [value.SerializeToString() for value in model.graph.node]
        model_input = next(i for i in model.graph.input if i.name == entry["input_name"])
        dimension = model_input.type.tensor_type.shape.dim[0]
        symbol = dimension.dim_param
        if not symbol:
            raise ValueError("Expected a symbolic batch dimension")
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param == symbol:
                    dim.dim_value = batch
        onnx.checker.check_model(model)
        onnx.save(model, path)
        reloaded = onnx.load(path)
        if weights != [v.SerializeToString() for v in reloaded.graph.initializer]:
            raise ValueError("Specialization changed weights")
        if nodes != [v.SerializeToString() for v in reloaded.graph.node]:
            raise ValueError("Specialization changed operators")
        entry["fixed_batch_size"] = batch
        entry["warmup_batch_sizes"] = [batch]
        metadata["checksums"][entry["filename"]] = sha256_file(path)
        changes.append(
            {
                "model": entry["filename"],
                "batch_size": batch,
                "source_sha256": original_hash,
                "candidate_sha256": sha256_file(path),
                "weights_identical": True,
                "operators_identical": True,
            }
        )
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"models": changes, "classification_policy_changed": False}
