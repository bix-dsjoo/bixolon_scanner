from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from ..contracts.errors import CorruptImageError, ImageTooLargeError, UnsupportedImageFormatError
from ..contracts.image import (
    ORIGINAL_SIZE_INFO_KEY as ORIGINAL_SIZE_INFO_KEY,
)
from ..contracts.image import (
    image_original_size as image_original_size,
)

ALLOWED_FORMATS = {"JPEG", "MPO", "PNG"}


def decode_image(
    data: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
    jpeg_draft_size: int | None = None,
) -> Image.Image:
    if len(data) > max_bytes:
        raise ImageTooLargeError
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise UnsupportedImageFormatError
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageTooLargeError
            orientation = int(source.getexif().get(274, 1))
            if source.format in {"JPEG", "MPO"} and jpeg_draft_size is not None:
                source.draft("RGB", (jpeg_draft_size, jpeg_draft_size))
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode != "RGB":
                image = image.convert("RGB")
            original_size = (height, width) if orientation in {5, 6, 7, 8} else (width, height)
            image.info[ORIGINAL_SIZE_INFO_KEY] = original_size
    except UnsupportedImageFormatError:
        raise
    except ImageTooLargeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CorruptImageError from exc
    return image
