"""Exercise the shipped PowerShell measurement client without model inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell handoff client")
def test_n100_measurement_preserves_all_repetitions_errors_and_input_hashes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    hashes = []
    for index in range(132):
        content = f"synthetic transport test {index}".encode()
        (images / f"{index:03d}.jpg").write_bytes(content)
        hashes.append(hashlib.sha256(content).hexdigest())
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"image_sha256": hashes}), encoding="utf-8")
    count = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, value):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send(200, {"status": "ready", "provider": "test_transport_only"})

        def do_POST(self):
            nonlocal count
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert b'name="image"' in body or b"name=image" in body
            count += 1
            # The first measured request fails; it must remain in the output.
            self.send(
                500 if count == 11 else 200, {"status": "ERROR" if count == 11 else "SEGMENTATION"}
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "results"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(Path(__file__).resolve().parents[1] / "installer/windows/measure-n100.ps1"),
        "-ImageDirectory",
        str(images),
        "-InputManifest",
        str(manifest),
        "-BaseUrl",
        f"http://127.0.0.1:{server.server_port}",
        "-OutputDirectory",
        str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        rows = [
            json.loads(line)
            for line in (output / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert count == 406
        assert len(rows) == 396
        for repeat in range(3):
            group = [r for r in rows if r["repetition"] == repeat]
            assert sorted(r["image_sha256"] for r in group) == sorted(hashes)
            assert all(r["elapsed_ms"] > 0 for r in group)
        assert (
            sum(r["http_status"] == 500 and r["response"]["status"] == "ERROR" for r in rows) == 1
        )
        environment = json.loads((output / "environment.json").read_text(encoding="utf-8-sig"))
        assert environment["warmup"] == 10
        assert environment["concurrent_requests"] == 1
        # A different image must be rejected before making another HTTP request.
        (images / "000.jpg").write_bytes(b"changed")
        rejected = subprocess.run(command, capture_output=True, timeout=30)
        assert rejected.returncode != 0
        assert count == 406
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
