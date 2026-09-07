"""Standalone full-frame camera viewer; no Worker or model dependencies."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

import cv2
from PIL import Image, ImageTk

RESOLUTIONS = {
    "8.0MP · 4:3 · 3264×2448": (3264, 2448),
    "5.0MP · 4:3 · 2592×1944": (2592, 1944),
    "2.1MP · 16:9 · 1920×1080": (1920, 1080),
    "1.9MP · 4:3 · 1600×1200": (1600, 1200),
    "0.9MP · 16:9 · 1280×720": (1280, 720),
    "0.3MP · 4:3 · 640×480": (640, 480),
}


def fitted_size(source: tuple[int, int], viewport: tuple[int, int]) -> tuple[int, int]:
    """Contain the complete image without cropping, stretching or upscaling."""
    width, height = source
    if min(width, height, *viewport) <= 0:
        raise ValueError("Image and viewport dimensions must be positive")
    scale = min(viewport[0] / width, viewport[1] / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale))


def center_crop_frame(frame, size: int):
    """Return native center pixels, rejecting undersized input instead of enlarging it."""
    height, width = frame.shape[:2]
    if size <= 0 or min(width, height) < size:
        raise ValueError(f"{width}×{height} 영상에서는 {size}×{size} 크롭을 만들 수 없습니다.")
    x, y = (width - size) // 2, (height - size) // 2
    return frame[y : y + size, x : x + size]


def save_frame(frame, directory: Path) -> Path:
    """Save all BGR pixels losslessly; never use the resized preview."""
    directory.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = directory / f"capture_{stamp}_{width}x{height}_{uuid4().hex[:8]}.png"
    ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError("PNG encoding failed")
    try:
        with target.open("xb") as stream:
            stream.write(encoded.tobytes())
    except FileExistsError:
        raise
    except OSError:
        target.unlink(missing_ok=True)
        raise
    return target


class CameraSession:
    """The reader thread exclusively owns the camera, including its release."""

    def __init__(self, index: int, resolution: tuple[int, int]):
        self.index = index
        self.resolution = resolution
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.errors: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def _read(self):
        camera = None
        try:
            camera = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                raise RuntimeError("camera unavailable")
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            while not self.stop.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("camera read failed")
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait((time.monotonic(), frame))
        except Exception:
            self.errors.put(
                "카메라 영상을 받을 수 없습니다. 다른 카메라 앱을 닫고, "
                "USB 연결과 카메라 번호를 확인한 뒤 다시 연결하세요."
            )
        finally:
            if camera is not None:
                camera.release()


class CameraApp:
    def __init__(
        self, root: tk.Tk, directory: Path, camera_index: int = 0, *, crop_size: int | None = None
    ):
        self.root = root
        self.directory = directory
        self.crop_size = crop_size
        self.source_size: tuple[int, int] | None = None
        self.session: CameraSession | None = None
        self.frame = None
        self.frame_time = 0.0
        self.saving = False
        self.closing = False
        self.save_results: queue.SimpleQueue = queue.SimpleQueue()
        self.photo = None
        self.last_render = None
        self.root.title(
            f"Camera Crop {crop_size} · 중앙 크롭" if crop_size else "전체 화면 카메라 · 원본 촬영"
        )
        self.root.geometry("1180x880")
        self.root.minsize(800, 600)
        self.root.configure(bg="#101216")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f4f6")
        style.configure("TLabel", background="#f3f4f6", font=("맑은 고딕", 10))
        style.configure("TButton", font=("맑은 고딕", 10), padding=(12, 8))
        style.configure("Capture.TButton", font=("맑은 고딕", 12, "bold"))

        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="카메라 번호").pack(side="left")
        self.index = tk.StringVar(value=str(camera_index))
        self.index_input = ttk.Spinbox(toolbar, from_=0, to=9, width=3, textvariable=self.index)
        self.index_input.pack(side="left", padx=(8, 16))
        self.resolution = tk.StringVar(value=next(iter(RESOLUTIONS)))
        self.resolution_input = ttk.Combobox(
            toolbar,
            textvariable=self.resolution,
            values=[
                label
                for label, size in RESOLUTIONS.items()
                if crop_size is None or min(size) >= crop_size
            ],
            state="readonly",
            width=31,
        )
        self.resolution_input.pack(side="left", padx=(0, 12))
        self.connect_button = ttk.Button(toolbar, text="연결", command=self.toggle_connection)
        self.connect_button.pack(side="left")
        ttk.Button(toolbar, text="전체 화면 · F11", command=self.toggle_fullscreen).pack(
            side="right"
        )

        self.info = tk.StringVar(value="카메라 연결 준비 중")
        ttk.Label(root, textvariable=self.info, padding=(16, 8)).pack(fill="x")
        self.canvas = tk.Canvas(root, bg="#101216", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas_image = self.canvas.create_image(0, 0, anchor="center")
        self.placeholder = self.canvas.create_text(
            0,
            0,
            text=f"중앙 {crop_size}×{crop_size} 미리보기" if crop_size else "전체 프레임 미리보기",
            fill="#aeb6c2",
            font=("맑은 고딕", 18),
        )
        footer = ttk.Frame(root, padding=12)
        footer.pack(fill="x")
        self.capture_button = ttk.Button(
            footer,
            text="크롭 PNG 촬영 · Space" if crop_size else "원본 PNG 촬영 · Space",
            style="Capture.TButton",
            command=self.capture,
            state="disabled",
        )
        self.capture_button.pack(side="left")
        ttk.Button(footer, text="저장 위치 변경", command=self.choose_directory).pack(
            side="left", padx=8
        )
        ttk.Button(footer, text="저장 폴더 열기", command=self.open_directory).pack(side="left")
        self.status = tk.StringVar(
            value=(
                f"중앙 {crop_size}×{crop_size} 픽셀 그대로 저장 · Space 촬영 / F11 전체 화면"
                if crop_size
                else "미리보기만 축소 · 자르기 / 좌우 반전 / 판정 없음"
            )
        )
        ttk.Label(root, textvariable=self.status, padding=(16, 6)).pack(fill="x")
        self.path_text = tk.StringVar(value=f"저장 위치: {directory}")
        ttk.Label(root, textvariable=self.path_text, padding=(16, 6), wraplength=1050).pack(
            fill="x"
        )
        root.bind("<F11>", lambda _: self.toggle_fullscreen())
        root.bind("<Escape>", lambda _: root.attributes("-fullscreen", False))
        root.bind("<space>", self.on_space)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.poll)
        root.after(150, self.toggle_connection)

    def toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def on_space(self, event):
        if event.widget not in (self.index_input, self.resolution_input):
            self.capture()
            return "break"
        return None

    def toggle_connection(self):
        if self.session is not None:
            self.session.stop.set()
            self.connect_button.configure(state="disabled", text="연결 해제 중…")
            self.clear_frame()
            return
        try:
            index = int(self.index.get())
            if index < 0 or index > 9:
                raise ValueError
        except ValueError:
            messagebox.showerror("카메라 번호", "0~9 사이의 카메라 번호를 입력하세요.")
            return
        self.clear_frame()
        self.session = CameraSession(index, RESOLUTIONS[self.resolution.get()])
        self.session.thread.start()
        self.index_input.configure(state="disabled")
        self.resolution_input.configure(state="disabled")
        self.connect_button.configure(text="연결 해제")
        self.info.set("카메라 연결 중… 실제 수신 해상도를 확인합니다.")

    def clear_frame(self):
        self.frame = None
        self.source_size = None
        self.last_render = None
        self.canvas.itemconfigure(self.canvas_image, image="")
        self.canvas.itemconfigure(self.placeholder, state="normal")
        self.capture_button.configure(state="disabled")

    def poll(self):
        if self.closing:
            return
        session = self.session
        if session is not None:
            try:
                frame_time, frame = session.frames.get_nowait()
                if not session.stop.is_set():
                    try:
                        source_size = (frame.shape[1], frame.shape[0])
                        cropped = (
                            center_crop_frame(frame, self.crop_size) if self.crop_size else frame
                        )
                        self.source_size = source_size
                        self.frame_time, self.frame = frame_time, cropped
                    except ValueError as error:
                        self.status.set(f"{error} 8MP를 지원하는 카메라를 연결하세요.")
                        session.stop.set()
                        self.clear_frame()
            except queue.Empty:
                pass
            try:
                error = session.errors.get_nowait()
                self.status.set(error)
                session.stop.set()
                self.clear_frame()
            except queue.Empty:
                pass
            if not session.thread.is_alive():
                self.session = None
                self.clear_frame()
                self.info.set("연결 해제됨 · 해상도를 선택하고 연결하세요.")
                self.index_input.configure(state="normal")
                self.resolution_input.configure(state="readonly")
                self.connect_button.configure(state="normal", text="연결")
        fresh = self.frame is not None and time.monotonic() - self.frame_time < 2.0
        self.capture_button.configure(state="normal" if fresh and not self.saving else "disabled")
        viewport = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        center = (viewport[0] // 2, viewport[1] // 2)
        self.canvas.coords(self.placeholder, *center)
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            requested = RESOLUTIONS[self.resolution.get()]
            source_width, source_height = self.source_size or (width, height)
            info = f"요청 {requested[0]}×{requested[1]}  |  실제 {source_width}×{source_height}"
            info += f"  |  중앙 크롭 {width}×{height}" if self.crop_size else "  |  전체 영역 표시"
            if requested != (source_width, source_height):
                info += "  · 요청과 다른 해상도 수신 중"
            if not fresh:
                info += "  · 영상 수신 지연: 촬영 불가"
            self.info.set(info)
            render_key = (self.frame_time, viewport)
            if self.last_render != render_key:
                size = fitted_size((width, height), viewport)
                preview = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas.coords(self.canvas_image, *center)
                self.canvas.itemconfigure(self.canvas_image, image=self.photo)
                self.canvas.itemconfigure(self.placeholder, state="hidden")
                self.last_render = render_key
        try:
            saved, error = self.save_results.get_nowait()
            self.saving = False
            if error:
                self.status.set("저장 실패 · 저장 폴더의 권한과 디스크 여유 공간을 확인하세요.")
                messagebox.showerror("저장 실패", self.status.get())
            else:
                self.status.set(f"{'크롭' if self.crop_size else '원본'} 저장 완료: {saved.name}")
        except queue.Empty:
            pass
        self.root.after(40, self.poll)

    def capture(self):
        if self.saving or self.frame is None or time.monotonic() - self.frame_time >= 2.0:
            return
        frame = self.frame.copy()
        directory = self.directory
        self.saving = True
        self.capture_button.configure(state="disabled")
        self.status.set("크롭 PNG 저장 중…" if self.crop_size else "전체 해상도 PNG 저장 중…")

        def write():
            try:
                self.save_results.put((save_frame(frame, directory), None))
            except Exception:
                self.save_results.put((None, True))

        threading.Thread(target=write, daemon=True).start()

    def choose_directory(self):
        chosen = filedialog.askdirectory(title="원본 사진 저장 폴더", initialdir=self.directory)
        if chosen:
            self.directory = Path(chosen)
            self.path_text.set(f"저장 위치: {self.directory}")

    def open_directory(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            os.startfile(self.directory)
        except OSError:
            messagebox.showerror("폴더 열기 실패", "저장 폴더에 접근할 수 없습니다.")

    def close(self):
        if self.saving:
            self.status.set("사진 저장을 마친 뒤 창을 닫습니다…")
            self.root.after(100, self.close)
            return
        self.closing = True
        if self.session is not None:
            self.session.stop.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="전체 프레임 카메라 미리보기 / 원본 PNG 촬영")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path.home() / "Pictures" / "FullFrameCamera")
    args = parser.parse_args()
    root = tk.Tk()
    CameraApp(root, args.output.resolve(), args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
