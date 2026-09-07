"""Camera Crop 2048: standalone utility, independent of the Scanner product bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bixolon_scanner.operations.camera_capture import CameraApp, center_crop_frame, save_frame

APP_VERSION = "1.0.0"
CROP_SIZE = 2048


def verify_package(directory: Path, *, live_camera: bool = False, camera_index: int = 0):
    """Explicit local packaging diagnostic; never runs during normal startup."""
    directory.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    try:
        if live_camera:
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            try:
                if not camera.isOpened():
                    raise RuntimeError("Camera is unavailable")
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
                for _ in range(3):
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError("Camera read failed")
            finally:
                camera.release()
        else:
            frame = np.random.default_rng(7).integers(0, 256, (2448, 3264, 3), dtype=np.uint8)
        crop = center_crop_frame(frame, CROP_SIZE)
        target = save_frame(crop, directory)
        with Image.open(target) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), crop[:, :, ::-1])
            if decoded.size != (CROP_SIZE, CROP_SIZE):
                raise RuntimeError("Incorrect output size")
        report = {
            "app_version": APP_VERSION,
            "frozen": bool(getattr(sys, "frozen", False)),
            "camera": live_camera,
            "source_size": [frame.shape[1], frame.shape[0]],
            "crop_size": list(decoded.size),
            "png_pixel_equality": True,
            "tk_version": str(tk.TkVersion),
            "output": target.name,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (directory / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description="중앙 2048×2048 카메라 미리보기 / PNG 촬영")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path.home() / "Pictures" / "CameraCrop2048")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.add_argument("--verify-package", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--live-camera", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_package:
        try:
            verify_package(
                args.verify_package, live_camera=args.live_camera, camera_index=args.camera
            )
        except Exception as error:
            args.verify_package.mkdir(parents=True, exist_ok=True)
            (args.verify_package / "failure.txt").write_text(str(error), encoding="utf-8")
            raise SystemExit(1) from error
        return
    root = tk.Tk()
    CameraApp(root, args.output.resolve(), args.camera, crop_size=CROP_SIZE)
    root.title(f"Camera Crop 2048 · {APP_VERSION}")
    root.mainloop()


if __name__ == "__main__":
    main()
