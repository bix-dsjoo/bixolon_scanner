from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from bixolon_scanner.operations import camera_capture  # noqa: E402
from bixolon_scanner.operations.camera_capture import fitted_size, save_frame  # noqa: E402


@pytest.mark.parametrize(
    ("source", "viewport", "expected"),
    [
        ((3264, 2448), (1920, 1080), (1440, 1080)),
        ((1920, 1080), (800, 800), (800, 450)),
        ((640, 480), (1920, 1080), (640, 480)),
        ((3264, 2448), (1, 1), (1, 1)),
    ],
)
def test_preview_contains_entire_frame(source, viewport, expected):
    assert fitted_size(source, viewport) == expected


def test_png_keeps_every_pixel_and_color_and_does_not_overwrite(tmp_path: Path):
    frame = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    directory = tmp_path / "원본 사진"
    first = save_frame(frame, directory)
    second = save_frame(frame, directory)
    assert first != second
    with Image.open(first) as image:
        assert image.size == (640, 480)
        np.testing.assert_array_equal(np.array(image), frame[:, :, ::-1])
    assert first.read_bytes() == second.read_bytes()


def test_save_failure_is_reported(tmp_path: Path):
    blocked = tmp_path / "file"
    blocked.write_text("occupied")
    with pytest.raises(OSError):
        save_frame(np.zeros((4, 6, 3), dtype=np.uint8), blocked)


def test_camera_keeps_actual_frame_and_releases_after_disconnect(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class Camera:
        released = False
        reads = 0

        def isOpened(self):
            return True

        def set(self, *_):
            return False  # Device refuses requested 8MP mode.

        def read(self):
            self.reads += 1
            return (True, frame) if self.reads == 1 else (False, None)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda *_: camera)
    session = camera_capture.CameraSession(0, (3264, 2448))
    session.thread.start()
    session.thread.join(timeout=2)
    assert not session.thread.is_alive()
    assert camera.released
    _, received = session.frames.get_nowait()
    assert received is frame
    assert session.errors.get_nowait()


def test_2048_crop_is_exact_native_center_and_png_keeps_corners(tmp_path):
    frame = np.random.default_rng(8).integers(0, 256, size=(2448, 3264, 3), dtype=np.uint8)
    crop = camera_capture.center_crop_frame(frame, 2048)
    assert crop.shape == (2048, 2048, 3)
    np.testing.assert_array_equal(crop, frame[200:2248, 608:2656])
    target = save_frame(crop, tmp_path)
    with Image.open(target) as image:
        np.testing.assert_array_equal(np.asarray(image), frame[200:2248, 608:2656, ::-1])


@pytest.mark.parametrize("shape", [(1080, 1920, 3), (1944, 2592, 3), (2448, 1024, 3)])
def test_2048_crop_rejects_small_frames_without_upscaling(shape):
    with pytest.raises(ValueError):
        camera_capture.center_crop_frame(np.zeros(shape, dtype=np.uint8), 2048)


def test_portrait_and_odd_dimensions_keep_center_without_reflection():
    frame = np.arange(9 * 7 * 3).reshape(9, 7, 3)
    crop = camera_capture.center_crop_frame(frame, 4)
    np.testing.assert_array_equal(crop, frame[2:6, 1:5])
