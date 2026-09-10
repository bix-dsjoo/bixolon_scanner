"""Add R4 on a nearly full USB drive while preserving older kits and results."""

import argparse
import copy
import shutil
from pathlib import Path

from bixolon_scanner.operations.n100_test_kit import child, digest, verify_kit, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("D:/N100"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    source = Path("artifacts/n100/structural-r4/kit").resolve()
    root = source.parent
    manifest = verify_kit(source)
    prefix = "Experiments-R4"
    shared = "Experiments-R2/Experiment/worker"
    planned = {}
    references = {}
    reference_overlay = []
    for row in manifest["files"]:
        name = row["path"]
        if name.startswith("log132/"):
            target = "Experiments-R3/" + name
            if digest(child(destination, target)) != row["sha256"]:
                raise ValueError("Shared log input differs from R4")
            references[target] = row | {"path": target}
        elif name.startswith("Experiment/worker/model-package/"):
            relative = name.removeprefix("Experiment/worker/model-package/")
            target = shared + "/model-package/" + relative
            existing = child(destination, target)
            if not existing.is_file():
                raise ValueError("Missing shared runtime payload")
            references[target] = {
                "path": target,
                "sha256": digest(existing),
                "size_bytes": existing.stat().st_size,
            }
            if references[target]["sha256"] != row["sha256"]:
                new = f"{prefix}/Experiment/overlays/reference/{relative}"
                planned[new] = child(source, name)
                reference_overlay.append(relative)
        elif name.startswith("Experiment/worker/store-catalog/"):
            target = (
                shared + "/store-catalog/" + name.removeprefix("Experiment/worker/store-catalog/")
            )
            if digest(child(destination, target)) != row["sha256"]:
                raise ValueError("Shared reference Catalog differs from R4")
            references[target] = row | {"path": target}
        else:
            planned[prefix + "/" + name] = child(source, name)
    required = sum(path.stat().st_size for path in planned.values())
    if shutil.disk_usage(destination).free < required + 20_000_000:
        raise ValueError(
            f"Insufficient free space for non-destructive R4 addition: {required} bytes"
        )
    addition = child(destination, prefix)
    addition.mkdir(exist_ok=False)
    for name, path in planned.items():
        target = child(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if digest(path) != digest(target):
            raise ValueError("USB copy checksum mismatch")
    for name, arguments in [
        ("1_RUN_ALL.cmd", "--matrix --repetitions 3 %*"),
        ("2_VERIFY_FILES.cmd", "--verify-only %*"),
    ]:
        (addition / name).write_bytes(
            (
                '@echo off\r\nsetlocal\r\n"%~dp0N100-EXPERIMENTS.exe" '
                f'--kit-root "%~dp0.." {arguments}\r\n'
                'set "RESULT=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RESULT%\r\n'
            ).encode("ascii")
        )
    payload = copy.deepcopy(manifest)
    payload["benchmark"] = prefix + "/Experiment"
    payload["benchmark_dependencies"] = [shared + "/model-package", shared + "/store-catalog"]
    for row in payload["inputs"]:
        row["path"] = "Experiments-R3/" + row["path"]
    for trial in payload["experiments"]:
        trial["runtime"] = shared + "/model-package"
        if trial.get("catalog"):
            trial["catalog"] = prefix + "/" + trial["catalog"]
        else:
            trial["catalog"] = shared + "/store-catalog"
        trial["runtime_overlay"] = (
            prefix + "/" + trial["runtime_overlay"]
            if trial.get("runtime_overlay")
            else prefix + "/Experiment/overlays/reference"
        )
    payload["files"] = [
        *references.values(),
        *[
            {
                "path": name,
                "sha256": digest(child(destination, name)),
                "size_bytes": child(destination, name).stat().st_size,
            }
            for name in sorted(planned)
        ],
    ]
    backup = destination / "previous-measurement-before-r4"
    backup.mkdir(exist_ok=False)
    for name in [
        "KIT-MANIFEST.json",
        "2_MEASURE.cmd",
        "3_VERIFY_FILES.cmd",
        "4_EXPERIMENTS.cmd",
        "CURRENT-RESULTS-KO.md",
    ]:
        existing = destination / name
        if existing.exists():
            shutil.copy2(existing, backup / name)
    write_json(destination / "KIT-MANIFEST.json", payload)
    verify_kit(destination)
    for name, arguments in [
        ("2_MEASURE.cmd", "--matrix --repetitions 3 %*"),
        ("3_VERIFY_FILES.cmd", "--verify-only %*"),
        ("4_EXPERIMENTS.cmd", "--matrix --repetitions 3 %*"),
    ]:
        (destination / name).write_bytes(
            (
                '@echo off\r\nsetlocal\r\n"%~dp0Experiments-R4\\N100-EXPERIMENTS.exe" '
                f'--kit-root "%~dp0." {arguments}\r\n'
                'set "RESULT=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RESULT%\r\n'
            ).encode("ascii")
        )
    write_json(
        root / "usb-verification.json",
        {
            "destination": str(destination),
            "manifest_sha256": digest(destination / "KIT-MANIFEST.json"),
            "added_payload_bytes": required,
            "reused_files": len(references),
            "reference_overlay_files": reference_overlay,
            "previous_launchers_backed_up": str(backup),
            "old_model_inputs_modified": False,
            "free_bytes_after": shutil.disk_usage(destination).free,
        },
    )
    print(f"R4 activated at {destination}; added {required:,} bytes; old kits/results preserved")


if __name__ == "__main__":
    main()
