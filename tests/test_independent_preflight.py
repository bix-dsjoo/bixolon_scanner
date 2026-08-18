import hashlib
import json
from pathlib import Path

from PIL import Image

from bixolon_scanner.experiments.bread.independent_preflight import (
    audit_independent_dataset,
)
from bixolon_scanner.training.bread_cv import difference_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, same_as_source: bool, review_status: str) -> dict:
    dataset = tmp_path / "incoming"
    images = dataset / "images"
    annotations = dataset / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    candidate_image = images / "candidate.png"
    source_image = tmp_path / "source.png"
    Image.new("RGB", (24, 24), (220, 120, 40)).save(candidate_image)
    if same_as_source:
        source_image.write_bytes(candidate_image.read_bytes())
    else:
        source = Image.new("RGB", (24, 24))
        source.putdata([(255 - x * 10, y * 10, (x + y) * 5) for y in range(24) for x in range(24)])
        source.save(source_image)
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "candidate.png",
                "width": 24,
                "height": 24,
                "exported_image_sha256": _sha(candidate_image),
            }
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 20, 20]}],
        "categories": [{"id": 1, "name": "bread"}],
    }
    annotation = annotations / "instances.json"
    annotation.write_text(json.dumps(coco), encoding="utf-8")
    metadata = dataset / "metadata.json"
    metadata.write_text(json.dumps({"annotation_review_status": review_status}), encoding="utf-8")
    record_manifest = dataset / "manifest.jsonl"
    record_manifest.write_text(
        json.dumps(
            {
                "image_id": 1,
                "annotation_review_status": review_status,
                "capture_session_id": "new-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "image_sha256": _sha(source_image),
                "perceptual_hash": difference_hash(source_image),
                "image_path": "development/source.png",
                "evaluation_set": "development",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_manifest = tmp_path / "candidate.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-v3",
                "lifecycle": "active_development",
                "independent_preflight": {
                    "fixed_git_commit": "adfae95",
                    "required_source_manifests": [
                        {"path": "source.jsonl", "sha256": _sha(source_manifest)}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "dataset_root": dataset,
        "annotation_path": annotation,
        "metadata_path": metadata,
        "record_manifest_path": record_manifest,
        "source_manifest_paths": [source_manifest],
        "candidate_manifest_path": candidate_manifest,
        "dataset_version": "independent-v1",
        "candidate_id": "candidate-v3",
        "candidate_commit": "adfae95",
    }


def test_independent_preflight_accepts_final_disjoint_dataset(tmp_path: Path) -> None:
    report = audit_independent_dataset(
        **_fixture(tmp_path, same_as_source=False, review_status="finalized")
    )

    assert report["eligible_for_independent_lock"] is True
    assert report["development_overlap_audit"]["exact_overlap_image_count"] == 0
    assert report["model_inference_executed"] is False


def test_independent_preflight_rejects_overlap_and_pending_review(tmp_path: Path) -> None:
    report = audit_independent_dataset(
        **_fixture(tmp_path, same_as_source=True, review_status="pending_user_review")
    )

    assert report["eligible_for_independent_lock"] is False
    assert report["development_overlap_audit"]["exact_overlap_image_count"] == 1
    assert report["checks"]["dataset_review_final"] is False
    assert report["checks"]["record_reviews_final"] is False


def test_independent_preflight_rejects_incomplete_candidate_lineage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, same_as_source=False, review_status="finalized")
    candidate_path = fixture["candidate_manifest_path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["independent_preflight"]["required_source_manifests"].append(
        {"path": "missing-development.jsonl", "sha256": "0" * 64}
    )
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    report = audit_independent_dataset(**fixture)

    assert report["eligible_for_independent_lock"] is False
    assert report["checks"]["candidate_source_scope_complete"] is False
