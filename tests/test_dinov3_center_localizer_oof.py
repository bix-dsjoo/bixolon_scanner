from bixolon_scanner.experiments.bread.dinov3_center_localizer_oof import run


def test_center_localizer_oof_runner_is_importable() -> None:
    assert callable(run)
