from __future__ import annotations

import struct
import zlib
from pathlib import Path

BRAND_RGB = (225, 54, 98)


def _rgba_pixels(path: Path) -> tuple[tuple[int, int], list[tuple[int, ...]]]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    compressed = bytearray()
    width = height = 0
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += length + 12
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            assert (bit_depth, color_type, interlace) == (8, 6, 0)
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break

    raw = zlib.decompress(compressed)
    stride = width * 4
    previous = bytearray(stride)
    pixels: list[tuple[int, ...]] = []
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        scanline = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        for index, value in enumerate(scanline):
            left = scanline[index - 4] if index >= 4 else 0
            above = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 1:
                scanline[index] = (value + left) & 255
            elif filter_type == 2:
                scanline[index] = (value + above) & 255
            elif filter_type == 3:
                scanline[index] = (value + ((left + above) // 2)) & 255
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[distances.index(min(distances))]
                scanline[index] = (value + predictor) & 255
            else:
                assert filter_type == 0
        pixels.extend(
            tuple(scanline[index : index + 4]) for index in range(0, stride, 4)
        )
        previous = scanline
    return (width, height), pixels


def test_brand_assets_are_transparent_and_use_the_primary_color() -> None:
    assets = {
        Path("public/localchat-logo.png"): (2087, 753),
        Path("public/localchat-icon.png"): (1254, 1254),
        Path("public/avatars/localchat.png"): (512, 512),
        Path("public/favicon.png"): (256, 256),
    }

    for path, expected_size in assets.items():
        size, pixels = _rgba_pixels(path)
        visible = [pixel for pixel in pixels if pixel[3] > 0]
        exact_brand = [pixel for pixel in visible if pixel[:3] == BRAND_RGB]
        assert size == expected_size
        assert len(visible) < len(pixels)
        assert len(exact_brand) / len(visible) >= 0.20


def test_theme_wordmarks_match_the_canonical_logo() -> None:
    canonical = Path("public/localchat-logo.png").read_bytes()

    assert Path("public/logo_dark.png").read_bytes() == canonical
    assert Path("public/logo_light.png").read_bytes() == canonical
