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
SOURCE_BYTES_INFO_KEY = "bixolon_source_bytes"


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
            if image.size != original_size:
                image.info[SOURCE_BYTES_INFO_KEY] = data
    except UnsupportedImageFormatError:
        raise
    except ImageTooLargeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CorruptImageError from exc
    return image


def restore_original_resolution(image: Image.Image) -> Image.Image:
    """Restore a draft-decoded image for rare content-sensitive secondary inference.

    The returned object is ``image`` when restoration is unnecessary or the image did
    not originate at the decode boundary. Callers must close a different returned image.
    """

    if image.size == image_original_size(image):
        return image
    encoded = image.info.get(SOURCE_BYTES_INFO_KEY)
    if not isinstance(encoded, bytes):
        return image
    with Image.open(BytesIO(encoded)) as source:
        source.load()
        restored = ImageOps.exif_transpose(source).convert("RGB")
        restored.load()
    restored.info[ORIGINAL_SIZE_INFO_KEY] = image_original_size(image)
    return restored


def redraft_image(image: Image.Image, draft_size: int) -> Image.Image:
    """Decode source JPEG bytes at a larger intermediate draft resolution.

    The returned object is ``image`` when the decode boundary did not retain source
    bytes or Pillow resolves the requested draft to the current pixel dimensions.
    Callers must close a different returned image.
    """

    encoded = image.info.get(SOURCE_BYTES_INFO_KEY)
    if not isinstance(encoded, bytes):
        return image
    with Image.open(BytesIO(encoded)) as source:
        if source.format not in {"JPEG", "MPO"}:
            return image
        source.draft("RGB", (draft_size, draft_size))
        source.load()
        redrafted = ImageOps.exif_transpose(source)
        if redrafted.mode != "RGB":
            redrafted = redrafted.convert("RGB")
        redrafted.load()
    if redrafted.size == image.size:
        redrafted.close()
        return image
    original_size = image_original_size(image)
    redrafted.info[ORIGINAL_SIZE_INFO_KEY] = original_size
    if redrafted.size != original_size:
        redrafted.info[SOURCE_BYTES_INFO_KEY] = encoded
    return redrafted
