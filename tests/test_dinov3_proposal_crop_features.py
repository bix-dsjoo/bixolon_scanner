import torch

from bixolon_scanner.experiments.bread.dinov3_proposal_crop_features import expand_boxes


def test_expand_boxes_adds_context_and_clips() -> None:
    boxes = expand_boxes(
        torch.tensor([[0.0, 10.0, 20.0, 30.0], [80.0, 80.0, 100.0, 100.0]]),
        ratio=0.1,
        maximum=100.0,
    )

    assert boxes.tolist() == [[0.0, 8.0, 22.0, 32.0], [78.0, 78.0, 100.0, 100.0]]
