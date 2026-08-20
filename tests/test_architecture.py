from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "bixolon_scanner"
_DART_IMPORT = re.compile(r"(?:import|export)\s+['\"]([^'\"]+)['\"]")


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    package_parts = path.relative_to(PACKAGE).with_suffix("").parts[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.removeprefix("bixolon_scanner.") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parent_count = node.level - 1
                base = package_parts[: len(package_parts) - parent_count]
                resolved = (*base, *filter(None, module.split(".")))
            else:
                normalized = module.removeprefix("bixolon_scanner.")
                resolved = () if normalized == "bixolon_scanner" else tuple(normalized.split("."))
            if resolved:
                modules.add(".".join(resolved))
            elif not module or module == "bixolon_scanner":
                modules.update(alias.name for alias in node.names)
    return modules


def _is_compatibility_alias(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    statements = tree.body[1:] if ast.get_docstring(tree) else tree.body
    return (
        (ast.get_docstring(tree) or "").startswith("Compatibility alias")
        and len(statements) == 3
        and isinstance(statements[0], ast.Import)
        and isinstance(statements[1], ast.ImportFrom)
        and isinstance(statements[2], ast.Assign)
    )


def test_canonical_layers_do_not_import_forbidden_dependencies() -> None:
    forbidden = {
        "contracts": {"worker", "runtime", "training", "evaluation", "experiments"},
        "pipeline": {"worker", "runtime", "training", "evaluation", "experiments"},
        "runtime": {"worker", "training", "evaluation", "experiments", "torch", "torchvision"},
        "worker": {"training", "evaluation", "experiments", "torch", "torchvision"},
        "training": {"evaluation", "experiments", "operations"},
        "evaluation": {"experiments"},
    }
    for layer, blocked in forbidden.items():
        for path in (PACKAGE / layer).rglob("*.py"):
            if _is_compatibility_alias(path):
                continue
            imported = _imports(path)
            assert not any(
                module == name or module.startswith(f"{name}.")
                for module in imported
                for name in blocked
            ), f"{path.relative_to(ROOT)} crosses the {layer} dependency boundary"


def test_flutter_features_do_not_import_each_other() -> None:
    features = ROOT / "apps" / "product_scanner" / "lib" / "features"
    feature_names = {path.name for path in features.iterdir() if path.is_dir()}
    for feature in feature_names:
        for path in (features / feature).rglob("*.dart"):
            for uri in _DART_IMPORT.findall(path.read_text(encoding="utf-8")):
                if uri.startswith("package:") and "/features/" in uri:
                    imported_feature = uri.split("/features/", maxsplit=1)[1].split(
                        "/", maxsplit=1
                    )[0]
                elif uri.startswith("."):
                    target = (path.parent / uri).resolve()
                    try:
                        imported_feature = target.relative_to(features.resolve()).parts[0]
                    except ValueError:
                        continue
                else:
                    continue
                assert imported_feature == feature or imported_feature not in feature_names, (
                    f"{path.relative_to(ROOT)} imports feature {imported_feature}"
                )


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
