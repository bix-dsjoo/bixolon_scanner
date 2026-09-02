from __future__ import annotations

from bixolon_scanner.experiments.bread.detr_family_comparison import _process_images


class RecordingProcessor:
    def __init__(self) -> None:
        self.options = None

    def __call__(self, **options):
        self.options = options
        return {"pixel_values": options["images"]}


def test_cached_float_images_are_not_rescaled_twice() -> None:
    processor = RecordingProcessor()
    images = [object()]
    annotations = [{"image_id": 1, "annotations": []}]

    encoded = _process_images(processor, images, annotations=annotations)

    assert processor.options == {
        "images": images,
        "annotations": annotations,
        "return_tensors": "pt",
        "do_rescale": False,
    }
    assert encoded["pixel_values"] == images
