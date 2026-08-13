from __future__ import annotations

from PIL import Image

ORIGINAL_SIZE_INFO_KEY = "bixolon_original_size"


def image_original_size(image: Image.Image) -> tuple[int, int]:
    """Return the source pixel size preserved by the decode boundary."""

    value = image.info.get(ORIGINAL_SIZE_INFO_KEY, image.size)
    return int(value[0]), int(value[1])
