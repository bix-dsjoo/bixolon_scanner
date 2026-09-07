"""Build the standalone camera utility's license inventory and payload manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


def main():
    payload = Path(sys.argv[1]).resolve()
    licenses = payload / "licenses"
    licenses.mkdir(exist_ok=True)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    shutil.copy2(python_license, licenses / "Python-LICENSE.txt")
    for name in ("numpy", "Pillow", "opencv-python", "PyInstaller"):
        distribution = importlib.metadata.distribution(name)
        for path in distribution.files or []:
            if "license" in str(path).lower() or "copying" in str(path).lower():
                source = Path(distribution.locate_file(path))
                if source.is_file():
                    destination = licenses / name / str(path).replace("..", "_")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    for source in (Path(sys.base_prefix) / "tcl").rglob("license*"):
        if source.is_file():
            target = licenses / "TclTk" / source.relative_to(Path(sys.base_prefix) / "tcl")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = {
        "application": "Camera Crop 2048",
        "version": "1.0.0",
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "Pillow", "opencv-python", "PyInstaller")
        },
        "files": {
            path.relative_to(payload).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(payload.rglob("*"))
            if path.is_file() and path.name != "payload-manifest.json"
        },
    }
    (payload / "payload-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
