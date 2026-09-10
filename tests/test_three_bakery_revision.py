from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_three_bakery import GT, item, response

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery import score_response, summarize
from bixolon_scanner.evaluation.three_bakery_cpu import developmental_rank, profile_order
from bixolon_scanner.evaluation.three_bakery_parity import matched_detector_outputs
from bixolon_scanner.training.three_bakery_data import audit_sources, write_jsonl


@pytest.mark.parametrize("milliseconds,passed", [(300.0, True), (300.001, False)])
def test_cpu_speed_boundary_requires_all_and_full_path(milliseconds, passed):
    row = {"metrics": score_response(response([item()]), GT), "elapsed_ms": milliseconds}
    report = summarize([row] * 300, minimum_complete_images=297, maximum_p95_ms=300)
    assert report["accuracy_target_met"] is True
    assert report["speed_target_met"] is passed
    assert report["target_met"] is passed
    assert report["latency"]["all"]["count"] == 300
    assert report["latency"]["full_path"]["max_ms"] == milliseconds


def test_fast_early_exits_cannot_hide_slow_full_path():
    fast = {"metrics": score_response(response(status="IMAGE_RECAPTURE"), GT), "elapsed_ms": 1}
    slow = {"metrics": score_response(response([item()]), GT), "elapsed_ms": 301}
    report = summarize([fast] * 299 + [slow], maximum_p95_ms=300)
    assert report["latency"]["all"]["p95_ms"] == 1
    assert report["speed_target_met"] is False
    assert summarize([fast], maximum_p95_ms=300)["speed_target_met"] is False


def test_nonfinite_latency_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize([{"metrics": {}, "elapsed_ms": float("nan")}])


def test_thread_profile_ties_use_p99_then_thread_sum_then_tuple():
    def row(profile, p99=220):
        return {"profile": profile, "latency": {"full_path": {"p95_ms": 200, "p99_ms": p99}}}

    assert profile_order(row([4, 8])) < profile_order(row([8, 4]))
    assert profile_order(row([4, 4])) < profile_order(row([8, 8]))
    assert profile_order(row([8, 8], 219)) < profile_order(row([4, 4]))


def test_development_targets_precede_quality_ranking():
    real = {
        "image_status_counts": {},
        "wrong_approved_count": 0,
        "by_difficulty": {"multi": {"complete_images": 99}},
        "missed_count": 0,
        "unknown_top3_miss_count": 0,
        "latency": {"full_path": {"p95_ms": 200}},
    }
    stress = {**real, "complete_images": 310}
    settings = {"minimum_real_multi_complete": 99}
    good = developmental_rank(
        real, stress, {"target_met": True, "worst_full_path_p95_ms": 299}, settings
    )
    slow = developmental_rank(
        real,
        {**stress, "complete_images": 320},
        {"target_met": False, "worst_full_path_p95_ms": 301},
        settings,
    )
    assert good < slow


def test_bbox_parity_separates_query_permutation_from_raw_arrays():
    boxes = np.array([[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]])
    logits = np.array([[[1.0], [2.0]]])
    actual = [logits[:, ::-1], boxes[:, ::-1]]
    report = matched_detector_outputs(["logits", "pred_boxes"], actual, [logits, boxes])
    assert not np.array_equal(actual[0], logits)
    batch = report["batches"][0]
    assert batch["reordered_query_count"] == 2
    assert batch["complete_correspondence"]
    assert all(value["allclose"] for value in batch["outputs"].values())


def test_audit_guard_blocks_reads_and_enumeration_without_prefix_false_positive(
    tmp_path, monkeypatch
):
    from bixolon_scanner.experiments import input_isolation

    forbidden = tmp_path / "held-out"
    forbidden.mkdir()
    secret = forbidden / "image.jpg"
    secret.write_bytes(b"image")
    allowed = tmp_path / "held-out-source"
    allowed.mkdir()
    monkeypatch.setattr(input_isolation, "_active_roots", (forbidden.resolve(),))
    for event, value in [("open", secret), ("os.listdir", forbidden), ("os.scandir", forbidden)]:
        with pytest.raises(PermissionError, match="held-out"):
            input_isolation._audit(event, (value,))
    input_isolation._audit("open", (allowed / "image.jpg",))
    input_isolation._audit("open", (1,))


def test_revision_ids_are_sha_bound_not_renumbered_after_new_multi(tmp_path):
    root = tmp_path / "sources"
    records = []
    for index, (relative, kind, image_id) in enumerate(
        [
            ("single_object/bread_01_bread/1.png", "single", 1),
            ("multi_object/old.png", "multi", 223),
            ("multi_object/added.png", "multi", 284),
            ("background/empty.png", "background", 251),
        ]
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), (index * 40, 0, 0)).save(path)
        records.append(
            {
                "image_id": image_id,
                "image_path": relative,
                "kind": kind,
                "image_sha256": sha256_file(path),
                "capture_session_id": "original-session",
            }
        )
    snapshot = tmp_path / "revision.jsonl"
    write_jsonl(snapshot, records)
    config = {
        "dataset_root": str(root),
        "class_count": 1,
        "shots_per_class": 1,
        "expected_counts": {"single": 1, "multi": 2, "background": 1},
        "source_group": "same-item",
        "source_revision": {
            "snapshots": [{"path": str(snapshot), "sha256": sha256_file(snapshot)}]
        },
    }
    rows, _ = audit_sources(config)
    assert [row["image_id"] for row in rows] == [1, 223, 251, 284]
    assert all(row["capture_session_id"] == "original-session" for row in rows)
    Image.new("RGB", (10, 10), "blue").save(root / "multi_object/added.png")
    with pytest.raises(ValueError, match="authorized revision"):
        audit_sources(config)


def test_revised_configuration_prespecifies_corrections_and_budgets():
    from bixolon_scanner.configuration import load_json_config

    config = load_json_config(Path("configs/experiments/bread/three_bakery_revised300.json"))
    assert config["source_revision"]["expected_objects"] == 649
    assert config["expected_counts"] == {"single": 200, "multi": 100, "background": 2}
    assert config["seeds"] == [20260908, 20260909, 20260910]
    assert len(config["cpu_optimization"]["profiles"]) == 6
    assert config["evaluation"]["maximum_cpu_p95_ms"] == 300


def test_both_user_label_corrections_and_missing_object_are_required():
    from bixolon_scanner.training.three_bakery_revision import validate_corrected_labels

    rows = [
        {"image_id": 223, "annotations": [{"category_id": v} for v in [1, 2, 3, 7]]},
        {"image_id": 284, "annotations": [{"category_id": v} for v in [1, 2, 3]]},
        {"image_id": 271, "annotations": [{"category_id": v} for v in [1, 2, 3, 4, 5, 5]]},
    ]
    validate_corrected_labels(rows)
    for row, index in [(rows[0], 3), (rows[1], 2), (rows[2], 5)]:
        original = row["annotations"][index]["category_id"]
        row["annotations"][index]["category_id"] = 10
        with pytest.raises(ValueError, match="correction"):
            validate_corrected_labels(rows)
        row["annotations"][index]["category_id"] = original
    rows[2]["annotations"].pop()
    with pytest.raises(ValueError, match="correction"):
        validate_corrected_labels(rows)


def test_development_guard_is_inherited_by_python_children(tmp_path):
    import subprocess
    import sys

    from bixolon_scanner.training.three_bakery_data import write_json

    forbidden = tmp_path / "held-out"
    forbidden.mkdir()
    target = forbidden / "image.jpg"
    target.write_bytes(b"not an input")
    config = tmp_path / "config.json"
    write_json(config, {"input_isolation": {"forbidden_roots": [str(forbidden)]}})
    script = (
        "from pathlib import Path; import subprocess,sys; "
        "from bixolon_scanner.experiments.input_isolation import protect_development; "
        "protect_development(Path(sys.argv[1]),Path(sys.argv[2])); "
        "child=subprocess.run([sys.executable,'-c',"
        "'from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()',sys.argv[3]],"
        "capture_output=True,text=True); print(child.stderr); sys.exit(child.returncode)"
    )
    run = subprocess.run(
        [sys.executable, "-c", script, str(config), str(tmp_path / "work"), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "held-out benchmark access is prohibited" in run.stdout


def test_extra_seed_fixes_verified_cpu_profile_and_measures_both_populations(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bixolon_scanner.evaluation import three_bakery_cpu
    from bixolon_scanner.training.three_bakery_data import write_json

    config_path = tmp_path / "config.json"
    config = {
        "product_version": "0.1.16",
        "seeds": [20260908, 20260909, 20260910],
        "cpu_optimization": {
            "warmup": 10,
            "repetitions": 3,
            "maximum_p95_ms": 300,
            "minimum_real_multi_complete": 99,
        },
        "evaluation": {"match_iou": 0.5},
        "expected_counts": {"multi": 100},
        "synthetic": {
            "diagnostic": {"seed": 20261908, "image_count": 400},
            "empty_probability": 0.2,
        },
    }
    write_json(config_path, config)
    real_path = tmp_path / "prepared/original_detection.jsonl"
    stress_path = tmp_path / "prepared/diagnostic-20261908/manifest.jsonl"
    write_jsonl(
        real_path, [{"image_id": i, "difficulty": "multi", "annotations": [{}]} for i in range(100)]
    )
    write_jsonl(
        stress_path, [{"image_id": i, "annotations": [{}] if i < 320 else []} for i in range(400)]
    )
    candidate = {"id": "ssdlite-frozen-basic"}
    base = tmp_path / "candidates/ssdlite-frozen-basic-20260908/cpu-profile.json"
    profile = {
        "selected_profile": [8, 4],
        "input_identity": {
            "real_manifest_sha256": sha256_file(real_path),
            "stress_manifest_sha256": sha256_file(stress_path),
            "configuration_sha256": sha256_file(config_path),
        },
    }
    write_json(base, profile)
    write_json(base.parent / "comparison.json", {"cpu_profile_sha256": sha256_file(base)})
    assembled = tmp_path / "candidates/ssdlite-frozen-basic-20260909"
    write_json(assembled / "runtime/metadata.json", {})
    write_json(assembled / "catalog/checksums.json", {})
    calls = []

    def fake_measure(candidate_path, records, output, **kwargs):
        calls.append((output.name, len(records), kwargs["repetitions"], kwargs["cpu_profile"]))
        summary = {
            "image_status_counts": {},
            "wrong_approved_count": 0,
            "by_difficulty": {"multi": {"complete_images": 100}},
            "complete_images": 100 if len(records) == 100 else 320,
            "missed_count": 0,
            "unknown_top3_miss_count": 0,
            "latency": {"full_path": {"p95_ms": 200}},
        }
        return {"summary": summary, "repetitions": [summary] * kwargs["repetitions"]}

    monkeypatch.setattr(three_bakery_cpu, "measure", fake_measure)
    args = SimpleNamespace(config=config_path, work=tmp_path, cpu_python=Path("python.exe"))
    result = three_bakery_cpu.diagnose_cpu(args, candidate, 20260909, assembled)
    assert result["rank"][0] == 0
    assert calls == [
        ("diagnostic-real", 100, 1, (8, 4)),
        ("diagnostic-stress", 400, 1, (8, 4)),
        ("latency-real-multi", 100, 3, (8, 4)),
        ("latency-stress-positive", 320, 3, (8, 4)),
    ]
    profile["selected_profile"] = [4, 4]
    write_json(base, profile)
    with pytest.raises(ValueError, match="CPU profile"):
        three_bakery_cpu.diagnose_cpu(args, candidate, 20260909, assembled)


def test_provider_parity_checks_later_repetitions(tmp_path):
    from bixolon_scanner.evaluation.three_bakery_http import provider_parity

    correct = response([item()]).model_dump(mode="json")
    wrong = response([item(category=2)]).model_dump(mode="json")
    cpu = [
        {"image_id": 1, "image_sha256": "same", "repetition": i, "response": correct}
        for i in range(3)
    ]
    cuda = [dict(row) for row in cpu]
    cuda[2]["response"] = wrong
    write_jsonl(tmp_path / "cpu.jsonl", cpu)
    write_jsonl(tmp_path / "cuda.jsonl", cuda)
    report = provider_parity(tmp_path / "cpu.jsonl", tmp_path / "cuda.jsonl")
    assert report["compared_request_count"] == 3
    assert not report["status_rank_parity"]
    assert report["mismatches"] == [{"repetition": 2, "image_id": 1}]


def test_repeated_shared_training_command_preserves_previous_logs(tmp_path):
    import sys

    from bixolon_scanner.experiments.bread.three_bakery import command

    command(tmp_path, "shared", [sys.executable, "-c", "print('first execution')"])
    command(tmp_path, "shared", [sys.executable, "-c", "print('second execution')"])
    logs = list((tmp_path / "stages").glob("*.log"))
    assert len(logs) == 2
    assert (tmp_path / "stages/shared.log").read_text().strip() == "first execution"
    assert {p.read_text().strip() for p in logs} == {"first execution", "second execution"}
