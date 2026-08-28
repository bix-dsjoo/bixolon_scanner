from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_WHOLE_TREE_TARGETS = (
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "weight",
    "checkpoints",
    "models",
    "apps/product_scanner/.dart_tool",
    "apps/product_scanner/.idea",
    "apps/product_scanner/.ruff_cache",
    "apps/product_scanner/build",
    "apps/product_scanner/test/failures",
    "artifacts/build-envs",
    "artifacts/handoff",
    "artifacts/vendor",
)
_EXPERIMENT_DELETE_SUFFIXES = {
    ".bin",
    ".cache",
    ".ckpt",
    ".dll",
    ".exe",
    ".joblib",
    ".lib",
    ".npy",
    ".npz",
    ".obj",
    ".onnx",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".safetensors",
    ".so",
    ".whl",
    ".zip",
}
_EXPERIMENT_DELETE_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "cache",
    "caches",
    "candidate",
    "candidates",
    "checkpoint",
    "checkpoints",
    "dist",
    "handoff",
    "runs",
    "venv",
    "weights",
    "worker-build",
}
_EVIDENCE_PARTS = {
    "analysis",
    "audit",
    "diagnostic",
    "diagnostics",
    "evaluation",
    "evaluations",
    "evidence",
    "figure",
    "figures",
    "metrics",
    "plots",
    "report",
    "reports",
    "results",
}
_EVIDENCE_NAME_TOKENS = {
    "audit",
    "comparison",
    "diff",
    "evaluation",
    "history",
    "manifest",
    "metadata",
    "metric",
    "provenance",
    "report",
    "result",
    "summary",
}


@dataclass(frozen=True)
class CleanupEntry:
    path: str
    size_bytes: int
    file_count: int
    reason: str


@dataclass(frozen=True)
class PreserveEntry:
    path: str
    size_bytes: int
    file_count: int
    reason: str
    provided_sha256: str | None = None


@dataclass(frozen=True)
class CleanupPlan:
    repository_root: Path
    active_version: str
    candidates: tuple[CleanupEntry, ...]
    preserved: tuple[PreserveEntry, ...]

    @property
    def planned_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.candidates)

    @property
    def planned_file_count(self) -> int:
        return sum(entry.file_count for entry in self.candidates)


def _relative(root: Path, path: Path) -> str:
    return path.absolute().relative_to(root).as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_repository_root(repository_root: Path) -> Path:
    root = repository_root.resolve()
    required = (root / ".git", root / "pyproject.toml", root / "configs")
    if not all(path.exists() for path in required):
        raise ValueError(f"저장소 루트 표식이 없습니다: {root}")
    return root


def _validate_candidate(root: Path, path: Path) -> None:
    absolute = path.absolute()
    if absolute == root or not _is_relative_to(absolute, root):
        raise ValueError(f"workspace 밖 경로는 정리할 수 없습니다: {absolute}")


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _configured_references(root: Path) -> dict[Path, tuple[str, str | None]]:
    references: dict[Path, tuple[str, str | None]] = {}
    config_roots = (root / "configs" / "versions", root / "configs" / "archive")
    for config_root in config_roots:
        if not config_root.exists():
            continue
        for config_path in sorted(config_root.rglob("*.json")):
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in _walk_json(payload):
                raw_path = item.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                absolute = candidate.absolute()
                if not _is_relative_to(absolute, root) or not absolute.exists():
                    continue
                digest = item.get("sha256") or item.get("manifest_sha256")
                references[absolute] = (
                    f"{_relative(root, config_path)} 직접 참조",
                    digest if isinstance(digest, str) else None,
                )
    return references


def _active_version(root: Path) -> str:
    configs = sorted((root / "configs" / "versions").glob("*.json"))
    if len(configs) != 1:
        raise ValueError("configs/versions에는 활성 JSON 하나만 있어야 합니다")
    payload = json.loads(configs[0].read_text(encoding="utf-8-sig"))
    version = payload.get("version")
    if not isinstance(version, str) or configs[0].stem != version:
        raise ValueError("활성 버전 설정의 파일명과 version이 일치하지 않습니다")
    return version


def _path_stats(path: Path) -> tuple[int, int]:
    if not path.exists() and not path.is_symlink():
        return 0, 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size, 1
    size = 0
    count = 0
    for directory, directories, files in os.walk(path, followlinks=False):
        base = Path(directory)
        retained_directories: list[str] = []
        for name in directories:
            child = base / name
            if child.is_symlink() or _is_reparse_point(child):
                size += child.lstat().st_size
                count += 1
            else:
                retained_directories.append(name)
        directories[:] = retained_directories
        for name in files:
            child = base / name
            try:
                size += child.lstat().st_size
                count += 1
            except FileNotFoundError:
                continue
    return size, count


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attributes & flag)


def _is_preserved(path: Path, roots: set[Path], files: set[Path]) -> bool:
    absolute = path.absolute()
    return absolute in files or any(
        absolute == preserved or _is_relative_to(absolute, preserved) for preserved in roots
    )


def _has_preserved_descendant(path: Path, roots: set[Path], files: set[Path]) -> bool:
    absolute = path.absolute()
    return any(
        preserved == absolute or _is_relative_to(preserved, absolute) for preserved in roots | files
    )


def _is_experiment_evidence(relative: Path) -> bool:
    lowered_parts = {part.lower() for part in relative.parts[:-1]}
    if lowered_parts & _EVIDENCE_PARTS:
        return True
    stem = relative.stem.lower().replace("-", "_")
    tokens = set(stem.split("_"))
    return bool(tokens & _EVIDENCE_NAME_TOKENS)


def _delete_experiment_file(relative: Path) -> bool:
    lowered_parts = {part.lower() for part in relative.parts[:-1]}
    disposable_container = any(
        part in _EXPERIMENT_DELETE_PARTS
        or part.endswith(("-venv", "-worker-build", "-handoff", "-candidate", "-candidates"))
        for part in lowered_parts
    )
    return relative.suffix.lower() in _EXPERIMENT_DELETE_SUFFIXES or disposable_container


def build_cleanup_plan(repository_root: Path) -> CleanupPlan:
    root = _validate_repository_root(repository_root)
    version = _active_version(root)
    references = _configured_references(root)
    preserved_roots = {
        path for path in references if path.is_dir() and _is_relative_to(path.absolute(), root)
    }
    preserved_files = {
        path for path in references if path.is_file() and _is_relative_to(path.absolute(), root)
    }
    policy_roots = {
        root / "datasets",
        root / "artifacts" / "evaluations",
        root / "artifacts" / "reports",
        root / "artifacts" / "versions" / version,
        root / "artifacts" / "installers" / version,
    }
    preserved_roots.update(path for path in policy_roots if path.exists())

    candidates: list[CleanupEntry] = []
    preserved: list[PreserveEntry] = []
    candidate_paths: set[Path] = set()

    def add_candidate(path: Path, reason: str) -> None:
        absolute = path.absolute()
        if absolute in candidate_paths or not absolute.exists():
            return
        _validate_candidate(root, absolute)
        size, count = _path_stats(absolute)
        candidates.append(
            CleanupEntry(
                path=_relative(root, absolute),
                size_bytes=size,
                file_count=count,
                reason=reason,
            )
        )
        candidate_paths.add(absolute)

    for relative in _WHOLE_TREE_TARGETS:
        target = root / relative
        if target.exists() and not _has_preserved_descendant(
            target, preserved_roots, preserved_files
        ):
            add_candidate(target, "재생성 가능한 cache/build/handoff")

    for parent_name in ("versions", "installers"):
        parent = root / "artifacts" / parent_name
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.name == version or _is_preserved(child, preserved_roots, preserved_files):
                continue
            add_candidate(child, f"활성 {version} 외 재생성 가능한 과거 배포 산출물")

    root_weight = root / "yolo26n.pt"
    if root_weight.exists():
        add_candidate(root_weight, "재다운로드 가능한 임시 weight")

    evidence_size = 0
    evidence_count = 0
    experiments = root / "artifacts" / "experiments"
    if experiments.exists():
        for directory, directories, files in os.walk(experiments, followlinks=False):
            base = Path(directory)
            retained_directories: list[str] = []
            for name in directories:
                child = base / name
                if child.is_symlink() or _is_reparse_point(child):
                    if not _is_preserved(child, preserved_roots, preserved_files):
                        add_candidate(child, "실험 경로의 재생성 가능한 link/junction")
                else:
                    retained_directories.append(name)
            directories[:] = retained_directories
            for name in files:
                path = base / name
                if _is_preserved(path, preserved_roots, preserved_files):
                    continue
                relative = path.relative_to(experiments)
                if _is_experiment_evidence(relative):
                    try:
                        evidence_size += path.lstat().st_size
                        evidence_count += 1
                    except FileNotFoundError:
                        pass
                    continue
                if _delete_experiment_file(relative):
                    add_candidate(path, "재생성 가능한 실험 binary/cache/candidate")

    for search_root in (root / "src", root / "tests", root / "scripts", root / "apps"):
        if not search_root.exists():
            continue
        for directory, directories, _ in os.walk(search_root, followlinks=False):
            base = Path(directory)
            for name in list(directories):
                if name == "__pycache__":
                    add_candidate(base / name, "Python bytecode cache")
                    directories.remove(name)

    for path, (reason, digest) in sorted(
        references.items(), key=lambda item: _relative(root, item[0])
    ):
        size, count = _path_stats(path)
        preserved.append(
            PreserveEntry(
                path=_relative(root, path),
                size_bytes=size,
                file_count=count,
                reason=reason,
                provided_sha256=digest,
            )
        )
    for path, reason in (
        (root / "datasets", "전체 원본·파생 데이터셋"),
        (root / "artifacts" / "evaluations", "모든 평가 결과"),
        (root / "artifacts" / "reports", "정리 manifest와 보고서"),
        (root / "artifacts" / "versions" / version, f"최종 {version} 번들"),
        (root / "artifacts" / "installers" / version, f"최종 {version} 설치 산출물"),
    ):
        if not path.exists() or path in references:
            continue
        size, count = _path_stats(path)
        preserved.append(
            PreserveEntry(
                path=_relative(root, path),
                size_bytes=size,
                file_count=count,
                reason=reason,
            )
        )
    if evidence_count:
        preserved.append(
            PreserveEntry(
                path="artifacts/experiments/**/{reports,results,evidence,metadata,provenance}",
                size_bytes=evidence_size,
                file_count=evidence_count,
                reason="실험 결과·보고서·provenance·metadata",
            )
        )

    candidates.sort(key=lambda item: item.path)
    preserved.sort(key=lambda item: item.path)
    return CleanupPlan(
        repository_root=root,
        active_version=version,
        candidates=tuple(candidates),
        preserved=tuple(preserved),
    )


def _make_writable(path: Path) -> None:
    try:
        path.chmod(stat.S_IWRITE | stat.S_IREAD)
    except FileNotFoundError:
        pass


def _remove_path(root: Path, path: Path) -> None:
    _validate_candidate(root, path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or _is_reparse_point(path):
        if path.is_dir():
            path.rmdir()
        else:
            _make_writable(path)
            path.unlink()
        return
    if path.is_dir():
        with os.scandir(path) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            _remove_path(root, child)
        _make_writable(path)
        path.rmdir()
        return
    _make_writable(path)
    path.unlink()


def _remove_empty_experiment_directories(root: Path) -> None:
    experiments = root / "artifacts" / "experiments"
    if not experiments.exists():
        return
    for directory, _, _ in os.walk(experiments, topdown=False, followlinks=False):
        path = Path(directory)
        if path == experiments or path.is_symlink() or _is_reparse_point(path):
            continue
        try:
            path.rmdir()
        except OSError:
            continue


def _manifest_payload(
    plan: CleanupPlan,
    *,
    apply: bool,
    status: str,
    disk_free_before: int | None = None,
    disk_free_after: int | None = None,
    deleted_bytes: int = 0,
    deleted_file_count: int = 0,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_root": str(plan.repository_root),
        "active_version": plan.active_version,
        "mode": "apply" if apply else "dry-run",
        "status": status,
        "safety": {
            "workspace_only": True,
            "git_clean_used": False,
            "datasets_deletable": False,
            "configured_references_preserved": True,
        },
        "summary": {
            "candidate_entry_count": len(plan.candidates),
            "planned_file_count": plan.planned_file_count,
            "planned_bytes": plan.planned_bytes,
            "deleted_file_count": deleted_file_count,
            "deleted_bytes_from_manifest": deleted_bytes,
            "disk_free_before_bytes": disk_free_before,
            "disk_free_after_bytes": disk_free_after,
            "actual_freed_bytes": (
                max(0, disk_free_after - disk_free_before)
                if disk_free_before is not None and disk_free_after is not None
                else None
            ),
        },
        "preserved": [asdict(entry) for entry in plan.preserved],
        "deletion_candidates": [asdict(entry) for entry in plan.candidates],
        "errors": errors or [],
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def execute_cleanup(plan: CleanupPlan, manifest_path: Path) -> dict[str, Any]:
    root = plan.repository_root
    manifest = manifest_path.absolute()
    _validate_candidate(root, manifest)
    if not _is_relative_to(manifest, root / "artifacts" / "reports"):
        raise ValueError("manifest는 artifacts/reports 아래에만 기록할 수 있습니다")
    disk_free_before = shutil.disk_usage(root).free
    write_manifest(
        manifest,
        _manifest_payload(
            plan,
            apply=True,
            status="planned-before-delete",
            disk_free_before=disk_free_before,
        ),
    )
    deleted_bytes = 0
    deleted_file_count = 0
    errors: list[str] = []
    for entry in sorted(
        plan.candidates,
        key=lambda value: (value.path.count("/"), value.path),
        reverse=True,
    ):
        path = root / entry.path
        try:
            _remove_path(root, path)
        except OSError as error:
            errors.append(f"{entry.path}: {error}")
            continue
        deleted_bytes += entry.size_bytes
        deleted_file_count += entry.file_count
    _remove_empty_experiment_directories(root)
    disk_free_after = shutil.disk_usage(root).free
    payload = _manifest_payload(
        plan,
        apply=True,
        status="completed" if not errors else "partial",
        disk_free_before=disk_free_before,
        disk_free_after=disk_free_after,
        deleted_bytes=deleted_bytes,
        deleted_file_count=deleted_file_count,
        errors=errors,
    )
    write_manifest(manifest, payload)
    return payload


def _default_manifest(root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        root
        / "artifacts"
        / "reports"
        / "maintenance"
        / (f"local-artifact-cleanup-{timestamp}.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="보존 계약을 검증하고 재생성 가능한 로컬 산출물을 정리합니다"
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    plan = build_cleanup_plan(args.repository_root)
    manifest = args.manifest or _default_manifest(plan.repository_root)
    if not manifest.is_absolute():
        manifest = plan.repository_root / manifest
    if args.apply:
        payload = execute_cleanup(plan, manifest)
    else:
        _validate_candidate(plan.repository_root, manifest)
        if not _is_relative_to(manifest.absolute(), plan.repository_root / "artifacts" / "reports"):
            raise ValueError("manifest는 artifacts/reports 아래에만 기록할 수 있습니다")
        payload = _manifest_payload(plan, apply=False, status="planned")
        write_manifest(manifest, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"manifest: {manifest}")
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
