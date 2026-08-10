from io import BytesIO

from PIL import Image

from bixolon_scanner.imaging import decode_image, image_original_size


def test_jpeg_draft_decode_preserves_original_coordinate_size():
    stream = BytesIO()
    Image.new("RGB", (2048, 1536), (128, 128, 128)).save(stream, format="JPEG")
    decoded = decode_image(
        stream.getvalue(),
        max_bytes=10_000_000,
        max_pixels=10_000_000,
        jpeg_draft_size=500,
    )
    assert decoded.size != (2048, 1536)
    assert image_original_size(decoded) == (2048, 1536)


def test_mpo_encoded_jpeg_uses_primary_frame(monkeypatch):
    stream = BytesIO()
    Image.new("RGB", (64, 48), (128, 128, 128)).save(stream, format="JPEG")
    original_open = Image.open

    def open_as_mpo(*args, **kwargs):
        image = original_open(*args, **kwargs)
        image.format = "MPO"
        return image

    monkeypatch.setattr("bixolon_scanner.imaging.Image.open", open_as_mpo)
    decoded = decode_image(
        stream.getvalue(),
        max_bytes=1_000_000,
        max_pixels=1_000_000,
        jpeg_draft_size=32,
    )

    assert decoded.mode == "RGB"
    assert image_original_size(decoded) == (64, 48)
