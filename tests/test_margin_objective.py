import pytest

from bixolon_scanner.training.margin_objective import normalized_margin_loss


def test_confident_wrong_prediction_receives_more_penalty_than_correct_prediction():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[4.0, 0.0, 0.0]])
    correct = normalized_margin_loss(logits, torch.tensor([0]), target_margin=0.9)
    wrong = normalized_margin_loss(logits, torch.tensor([1]), target_margin=0.9)
    assert correct.item() == 0
    assert wrong.item() > 1


def test_scaling_logits_cannot_game_normalized_margin_objective():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, 0.8, 0.3]])
    labels = torch.tensor([0])
    a = normalized_margin_loss(logits, labels, target_margin=0.9)
    b = normalized_margin_loss(logits * 100, labels, target_margin=0.9)
    assert a.item() == pytest.approx(b.item())


def test_margin_loss_gradient_improves_signed_ground_truth_margin():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.5, 0.8, 0.1]], requires_grad=True)
    labels = torch.tensor([0])
    loss = normalized_margin_loss(logits, labels, target_margin=0.9)
    loss.backward()
    updated = logits.detach() - 0.01 * logits.grad
    assert normalized_margin_loss(updated, labels, target_margin=0.9).item() < loss.item()
