"""Generate the Windows app icon from the BIXOLON Scanner brand mark.

The icon intentionally uses the product's orange focus symbol rather than
reproducing the official BIXOLON wordmark. It only relies on Python's standard
library so the committed ICO can be regenerated without design tooling.
"""

from __future__ import annotations

import argparse
import binascii
import struct
import zlib
from pathlib import Path

ORANGE = (0xEE, 0x72, 0x03, 0xFF)
INK = (0x17, 0x17, 0x17, 0xFF)
TRANSPARENT = (0, 0, 0, 0)
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SCALE = 4


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png(width: int, height: int, pixels: bytes) -> bytes:
    rows = b"".join(b"\x00" + pixels[y * width * 4 : (y + 1) * width * 4] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows, 9))
        + _chunk(b"IEND", b"")
    )


def _inside_rounded_square(x: int, y: int, size: int, inset: int, radius: int) -> bool:
    left = top = inset
    right = bottom = size - inset - 1
    if not (left <= x <= right and top <= y <= bottom):
        return False
    inner_left = left + radius
    inner_right = right - radius
    inner_top = top + radius
    inner_bottom = bottom - radius
    if inner_left <= x <= inner_right or inner_top <= y <= inner_bottom:
        return True
    cx = inner_left if x < inner_left else inner_right
    cy = inner_top if y < inner_top else inner_bottom
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def _render_high_resolution(size: int) -> bytearray:
    pixels = bytearray(TRANSPARENT * (size * size))
    inset = max(1, round(size * 0.035))
    radius = max(2, round(size * 0.19))
    stroke = max(2, round(size * 0.07))
    bracket_start = round(size * 0.28)
    bracket_end = round(size * 0.45)
    opposite_start = size - bracket_end
    opposite_end = size - bracket_start
    dot_radius = max(1, round(size * 0.045))
    center = (size - 1) / 2

    for y in range(size):
        for x in range(size):
            color = TRANSPARENT
            if _inside_rounded_square(x, y, size, inset, radius):
                color = ORANGE

                left_vertical = bracket_start <= x < bracket_start + stroke
                left_horizontal = bracket_start <= x < bracket_end
                right_vertical = opposite_end - stroke < x <= opposite_end
                right_horizontal = opposite_start < x <= opposite_end
                top_vertical = bracket_start <= y < bracket_end
                top_horizontal = bracket_start <= y < bracket_start + stroke
                bottom_vertical = opposite_start < y <= opposite_end
                bottom_horizontal = opposite_end - stroke < y <= opposite_end

                in_bracket = (
                    (left_vertical and (top_vertical or bottom_vertical))
                    or (right_vertical and (top_vertical or bottom_vertical))
                    or (top_horizontal and (left_horizontal or right_horizontal))
                    or (bottom_horizontal and (left_horizontal or right_horizontal))
                )
                in_dot = (x - center) ** 2 + (y - center) ** 2 <= dot_radius**2
                if in_bracket or in_dot:
                    color = INK

            offset = (y * size + x) * 4
            pixels[offset : offset + 4] = bytes(color)
    return pixels


def _render(size: int) -> bytes:
    high_size = size * SCALE
    source = _render_high_resolution(high_size)
    output = bytearray(size * size * 4)
    sample_count = SCALE * SCALE
    for y in range(size):
        for x in range(size):
            totals = [0, 0, 0, 0]
            for sy in range(SCALE):
                for sx in range(SCALE):
                    offset = (((y * SCALE + sy) * high_size) + x * SCALE + sx) * 4
                    for channel in range(4):
                        totals[channel] += source[offset + channel]
            target = (y * size + x) * 4
            output[target : target + 4] = bytes(round(total / sample_count) for total in totals)
    return _png(size, size, bytes(output))


def build_icon() -> bytes:
    images = [_render(size) for size in SIZES]
    header_size = 6 + len(images) * 16
    offset = header_size
    entries = []
    for size, image in zip(SIZES, images, strict=True):
        encoded_size = 0 if size == 256 else size
        entries.append(
            struct.pack(
                "<BBBBHHII",
                encoded_size,
                encoded_size,
                0,
                0,
                1,
                32,
                len(image),
                offset,
            )
        )
        offset += len(image)
    return struct.pack("<HHH", 0, 1, len(images)) + b"".join(entries) + b"".join(images)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = (
        Path(__file__).resolve().parents[1] / "windows" / "runner" / "resources" / "app_icon.ico"
    )
    generated = build_icon()
    if args.check:
        if not target.exists() or target.read_bytes() != generated:
            raise SystemExit(f"Windows icon is out of date: {target}")
        print(f"Windows icon is current: {target}")
        return 0
    target.write_bytes(generated)
    print(f"Generated {target} ({len(generated)} bytes, {len(SIZES)} sizes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
