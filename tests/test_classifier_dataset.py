import json

from bixolon_scanner.training.data import ClassifierDataset


def test_classifier_records_follow_group_fold_for_train_and_validation(tmp_path) -> None:
    manifest = tmp_path / "classifier.jsonl"
    rows = [
        {
            "record_type": "classification",
            "split": "development",
            "fold": fold,
            "image_path": f"single_objects/{fold}.jpg",
            "category_id": fold + 1,
        }
        for fold in range(3)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    training = ClassifierDataset(manifest, tmp_path, mode="train", fold=1)
    validation = ClassifierDataset(manifest, tmp_path, mode="validation", fold=1)

    assert [sample[1] for sample in training.samples] == [0, 2]
    assert [sample[1] for sample in validation.samples] == [1]
