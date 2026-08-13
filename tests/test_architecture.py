from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "bixolon_scanner"


def _relative_imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            modules.add(node.module or "")
    return modules


def test_canonical_layers_do_not_import_forbidden_dependencies() -> None:
    forbidden = {
        "contracts": {"worker", "runtime", "training", "evaluation", "experiments"},
        "pipeline": {"worker", "runtime", "training", "evaluation", "experiments"},
        "runtime": {"worker", "training", "evaluation", "experiments"},
        "worker": {"training", "evaluation", "experiments"},
    }
    for layer, blocked in forbidden.items():
        for path in (PACKAGE / layer).rglob("*.py"):
            imported = _relative_imports(path)
            assert not any(
                module == name or module.startswith(f"{name}.")
                for module in imported
                for name in blocked
            ), f"{path.relative_to(ROOT)} crosses the {layer} dependency boundary"


def test_worker_import_does_not_load_training_frameworks() -> None:
    source = str(ROOT / "src")
    code = (
        "import sys; "
        f"sys.path.insert(0, {source!r}); "
        "import bixolon_scanner.worker; "
        "blocked={'torch','torchvision','transformers','sklearn','scipy'}; "
        "loaded=blocked.intersection(sys.modules); "
        "assert not loaded, sorted(loaded)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_imports_resolve_to_canonical_objects() -> None:
    from bixolon_scanner.api import create_app as legacy_create_app
    from bixolon_scanner.contracts import ScanResponse
    from bixolon_scanner.contracts.model_package import ModelPackageMetadata
    from bixolon_scanner.inference import Detection as legacy_detection
    from bixolon_scanner.package import ModelPackageMetadata as legacy_metadata
    from bixolon_scanner.pipeline import DecisionPipeline
    from bixolon_scanner.pipeline.decision import DecisionPipeline as canonical_pipeline
    from bixolon_scanner.pipeline.ports import Detection
    from bixolon_scanner.worker.api import create_app

    assert legacy_create_app is create_app
    assert legacy_detection is Detection
    assert legacy_metadata is ModelPackageMetadata
    assert DecisionPipeline is canonical_pipeline
    assert ScanResponse.__module__ == "bixolon_scanner.contracts.api"
