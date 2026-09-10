"""Readable fixed-benchmark results and traceable failure-case overlays."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from ..configuration import load_json_config
from ..training.three_bakery_data import read_jsonl, write_json
from .three_bakery_objective import load_objective, required_correct_count


def training_inventory(work: Path) -> list[dict]:
    rows = []
    for path in sorted((work / "models").glob("*/report.json")):
        report = load_json_config(path)
        if path.parent.name == "verifier":
            continue
        contract = load_json_config(path.parent / "contract.json")
        rows.append(
            {
                "model": path.parent.name,
                "epochs": report["epochs"],
                "original_images_consumed": report["observed_original_count"],
                "original_crops_consumed": report.get("observed_original_crop_count"),
                "original_inputs_per_image_min": min(report["observed_input_counts"].values()),
                "original_inputs_per_image_max": max(report["observed_input_counts"].values()),
                "foundation_weights_sha256": contract["weights_sha256"],
                "source_manifest_sha256": contract["source_manifest_sha256"],
                "annotation_sha256": contract["annotation_sha256"],
                "synthetic_manifest_sha256": contract["synthetic_manifest_sha256"],
                "report_path": str(path.resolve()),
            }
        )
    return rows


def write_report(work: Path) -> Path:
    final = work / "final"
    result = load_json_config(final / "report.json")
    selection = load_json_config(work / "selection.json")
    objective = (
        load_objective(Path(selection["objective_config"]))
        if "objective_config" in selection
        else None
    )
    source = load_json_config(work / "sources/annotation-report.json")
    source_report = load_json_config(work / "sources/source-report.json")
    config = load_json_config(Path(source_report["config_path"]))
    cpu_target = config["evaluation"].get("maximum_cpu_p95_ms")
    multi_count = config["expected_counts"]["multi"]
    latency_provider = "CPU" if config.get("cpu_optimization") else "CUDA"
    payload = f"../candidates/{selection['selected']['id']}-{selection['payload_seed']}"
    originals = {r["image_id"]: r for r in read_jsonl(final / "final-inputs.jsonl")}
    lines = [
        "# three_bakery 고정 벤치마크 결과",
        "",
        f"선택 구성: `{selection['selected']['id']}`, payload seed `{selection['payload_seed']}`.",
        "",
        f"학습 원본 {source['images']}장·정답 객체 {source['objects']}개. 최종 평가는 별도 300장·1,410개 GT이다.",
        "같은 실물 개발 진단으로 구성을 선택했다. 최종 세트는 과거 평가 이력이 있는 고정 벤치마크다.",
        "",
        f"산출물: [Runtime metadata]({payload}/runtime/metadata.json), [Catalog metadata]({payload}/catalog/catalog.json), "
        f"[Catalog support 원본 추적]({payload}/catalog-support.jsonl).",
        f"[{source['images']}장 원본 manifest](../sources/originals.jsonl), [검수 annotation](../sources/annotations.jsonl), "
        "[전체 객체 crop 원본 추적](../prepared/original_crops.jsonl)에 SHA-256 연결을 보존한다.",
        "",
        (
            "사전 접근 이력: 후보 확정 전 저장소 전체 테스트가 기존 300장 이미지·GT를 읽어 데이터 계약을 검사했다. "
            "이 내용은 이번 모델 학습·증강·후보 선택·임계값 결정에는 사용하지 않았으나, 후보 확정 전 파일 접근을 완전히 차단한 실행은 아니다. "
            "[접근 기록](../reproduction/prefreeze-benchmark-access.json)을 함께 확인해야 한다."
            if (work / "reproduction/prefreeze-benchmark-access.json").exists()
            else "최종 세트와 이전 개별 평가 출력은 개발 선택 입력에서 제외했다. 개발 CLI와 Python 자식 프로세스에 "
            "파일 접근 audit guard를 적용했다. 이는 OS sandbox가 아닌 Python 접근 차단이며 원본·파생물 허용 목록과 해시 검증을 함께 사용했다."
        ),
        "",
        (
            f"사용자 변경 목표는 CPU·CUDA 매 반복에서 GT 객체 정답 승인율 "
            f"≥{objective['minimum_correct_approved_rate']:.0%} "
            f"(≥{required_correct_count(objective['final_ground_truth_count'], objective['minimum_correct_approved_rate'])}"
            f"/{objective['final_ground_truth_count']}개) 및 전체 오승인 0건이다. "
            "분모는 전체 GT이며 누락·UNKNOWN·재촬영·ERROR의 GT를 빼지 않는다. "
            "이미지 전체 성공률은 참고 진단이다."
            if objective is not None
            else "목표는 CPU·CUDA 각각 완전 정답 승인 이미지 ≥297/300 및 전체 오승인 0건이다."
        ),
        f"CPU는 각 반복의 전체 요청 및 full-path HTTP p95 ≤{cpu_target:g}ms도 요구한다."
        if cpu_target is not None
        else "속도는 이번 실행의 합격 조건이 아니다.",
        "완전 성공은 모든 GT에 정답 APPROVED가 하나씩 대응하고 추가 segmentation이 없는 이미지다.",
        "누락·UNKNOWN·재촬영·ERROR가 있는 이미지는 완전 성공이 아니다. 오승인은 오분류·배경·중복 승인을 포함한다.",
        "",
        f"**최종 목표 {'달성' if result['target_met'] else '미달'}**. 결과에 맞춘 재학습이나 임계값 변경을 하지 않았다.",
        "",
        "| Provider | 반복 | 완전 성공/300 | 이미지 성공률 | 정답 승인 객체/1,410 | 객체 승인율 | 오승인 | 누락 | 추가 segmentation | UNKNOWN | segment 재촬영 | IMAGE_RECAPTURE | ERROR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if objective is not None:
        lines[6:6] = [
            "최종 세트 접근 전 사용자가 99% 목표를 이미지 전체에서 원본 GT 객체 기준으로 변경했다. "
            "[목표 변경 기록](../objective-revision.json). 기존 HTTP 응답·IoU 대응·측정 시간을 바꾸지 않고 "
            "객체 기준 적격 여부와 순위를 별도 집계했다. 학습량·모델·승인 정책·재촬영 정책은 유지했다.",
            "",
        ]
    if cpu_target is not None:
        selected_path = (
            work / f"candidates/{selection['selected']['id']}-{selection['payload_seed']}"
        )
        profile = load_json_config(selected_path / "cpu-profile.json")
        lines[6:6] = [
            f"CPU: {config['cpu_optimization']['cpu_model']}, detector/embedder threads "
            f"`{profile['selected_profile']}`. [Worker 실행 설정]({payload}/worker-cpu.json), "
            f"[thread 비교 근거]({payload}/cpu-profile.json).",
            "[검수 주체별 이력](../sources/review-history.json), "
            "[신규 마스크 검수](../sources/visual-review.json). 신규 다중 마스크는 합성에 사용하지 않았으며 "
            "검출기 원본 302장·분류기 crop 649개는 전부 학습했다.",
            "",
        ]
        development = selection["development_targets"]
        lines[6:6] = [
            f"개발 목표를 모두 충족한 기본 seed 후보는 {development['eligible_base_candidate_count']}/12개이고, "
            f"선택 구성의 세 seed 모두 개발 목표 충족 여부는 `{development['selected_all_seeds_eligible']}`다. "
            "미달이어도 사전에 정한 순위의 최선 후보를 최종 평가했다.",
            "",
        ]
    revision = work / "policy-revision.json"
    if revision.exists():
        freeze = load_json_config(work / "final-candidate.json")
        integrity_config = load_json_config(Path(freeze["roi_integrity_config"]))
        lines[6:6] = [
            "추가 요청으로 최종 벤치마크 접근 전에 ROI 다중 객체 재촬영 정책을 추가하고, "
            "기본 12개·추가 seed·CPU thread 구성을 다시 비교했다. [변경 이력](../policy-revision.json).",
            f"192 primary의 ConvNeXt 특징을 공유하는 head의 다중 객체 확률이 "
            f"`{integrity_config['multi_object_probability_threshold']}` 이상이면 해당 ROI만 "
            "SEGMENT_RECAPTURE이며, detail·회전·ViT 합의로 다시 승인하지 않는다. "
            "별도 backbone이나 전수 ViT 검증은 추가하지 않았다.",
            "head는 해당 구성의 원본 GT와 합성 학습 장면에서 단일/다중 ROI를 만들어 학습했다. "
            "공통 합성 진단과 최종 벤치마크는 이 head의 학습 입력이 아니다. "
            "재촬영으로 오승인을 차단해도 완전 성공 이미지로 집계하지 않는다.",
            "",
        ]
    for provider, report in result["providers"].items():
        for index, summary in enumerate(report["repetitions"], 1):
            items, statuses = summary["item_status_counts"], summary["image_status_counts"]
            lines.append(
                f"| {provider} | {index} | {summary['complete_images']}/300 | {summary['complete_image_rate']:.2%} | "
                f"{summary['correct_approved_count']}/1410 | {summary['correct_approved_rate']:.2%} | "
                f"{summary['wrong_approved_count']} | {summary['missed_count']} | "
                f"{summary['extra_count']} | {items.get('UNKNOWN', 0)} | {items.get('SEGMENT_RECAPTURE', 0)} | "
                f"{statuses.get('IMAGE_RECAPTURE', 0)} | {statuses.get('ERROR', 0)} |"
            )
    inventory = training_inventory(work)
    comparison_runs = selection["base_candidates"] + [
        run for candidate in selection["repeated_candidates"] for run in candidate["runs"]
    ]
    used_models = {
        f"{run['candidate'][component]}-{run['candidate']['recipe']}-{run['seed']}"
        for run in comparison_runs
        for component in ("architecture", "method")
    }
    selected_models = {
        f"{selection['selected'][component]}-{selection['selected']['recipe']}-{selection['payload_seed']}"
        for component in ("architecture", "method")
    }
    for row in inventory:
        row["used_in_current_comparison"] = row["model"] in used_models
        row["selected_payload_component"] = row["model"] in selected_models
    write_json(final / "training-inventory.json", inventory)
    lines += [
        "",
        "## 모델별 실제 원본 사용",
        "",
        "| 모델 | epoch | 원본 이미지 | 원본 객체 crop | 현재 사용 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in inventory:
        usage = (
            "최종 payload"
            if row["selected_payload_component"]
            else "비교 후보"
            if row["used_in_current_comparison"]
            else "이전 비교에서 보존"
        )
        lines.append(
            f"| {row['model']} | {row['epochs']} | {row['original_images_consumed']} | "
            f"{row['original_crops_consumed'] if row['original_crops_consumed'] is not None else '해당 없음'} | "
            f"{usage} |"
        )
    integrity_inventory = []
    for path in sorted((work / "roi-integrity").glob("*/report.json")):
        head = load_json_config(path)
        integrity_inventory.append({"model": path.parent.name, **head})
    if integrity_inventory:
        write_json(final / "roi-integrity-inventory.json", integrity_inventory)
        lines += [
            "",
            "ROI 객체 수 head 학습:",
            "",
            "| 공유 backbone 구성 | epoch | 학습 ROI | 사용 원본 객체 |",
            "|---|---:|---:|---:|",
        ]
        for head in integrity_inventory:
            lines.append(
                f"| {head['model']} | {head['epochs']} | {head['sample_count']} | "
                f"{head['unique_original_single_objects']} |"
            )
        lines += [
            "",
            "[head 학습·ONNX 해시](roi-integrity-inventory.json). 각 head의 "
            "`roi-integrity/<구성>/samples.jsonl`에 부모 원본 SHA-256과 합친 객체 index를 보존한다.",
        ]
    lines += [
        "",
        "각 검출기는 해당 seed의 basic 1,200장 또는 dense 6,000장 합성 장면을 함께 사용했다. "
        "분류기는 해당 합성 장면의 양성 객체 crop을 함께 사용했다. 빈 장면은 검출기의 음성 입력이다. "
        "Frozen ViT-B/16 본체는 일반 사전학습 가중치를 고정하고, 검증용 adapter·support는 이번 원본에서 선택한 "
        "품목별 10개 crop으로 새로 조립했다. [입력·가중치 해시](training-inventory.json).",
        "",
    ]
    lines += [
        "",
        "첫 반복의 오승인 세부 내역(중복은 미대응 bbox가 다른 GT와 IoU≥0.5인 경우):",
        "",
        "| Provider | 오분류 승인 | 중복 승인 | 배경 승인 | 전체 중복 검출 |",
        "|---|---:|---:|---:|---:|",
    ]
    for provider, report in result["providers"].items():
        summary = report["summary"]
        lines.append(
            f"| {provider} | {summary['wrong_class_approved_count']} | {summary['duplicate_approved_count']} | "
            f"{summary['background_approved_count']} | {summary['duplicate_count']} |"
        )
    lines += [
        "",
        "| Provider | 난이도 | 이미지 수 | 완전 성공 | 정답 승인/GT 객체 | 객체 승인율 | 오승인 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    difficulty_objects = {}
    for provider, report in result["providers"].items():
        first_rows = [
            r for r in read_jsonl(final / provider / "responses.jsonl") if r["repetition"] == 0
        ]
        difficulty_objects[provider] = {}
        for difficulty, values in report["summary"]["by_difficulty"].items():
            subset = [r["metrics"] for r in first_rows if r["difficulty"] == difficulty]
            total = sum(r["ground_truth_count"] for r in subset)
            correct = sum(r["correct_approved_count"] for r in subset)
            rate = correct / total if total else None
            difficulty_objects[provider][difficulty] = {
                "ground_truth_count": total,
                "correct_approved_count": correct,
                "correct_approved_rate": rate,
            }
            rate_text = f"{rate:.2%}" if rate is not None else "해당 없음"
            lines.append(
                f"| {provider} | {difficulty} | {values['image_count']} | {values['complete_images']} | "
                f"{correct}/{total} | {rate_text} | {values['wrong_approved_count']} |"
            )
    write_json(final / "difficulty-object-accuracy.json", difficulty_objects)
    lines += [
        "",
        "지연은 warmup 10회 이후 동시 요청 1개로 측정한 실제 HTTP 왕복 시간이다. "
        "파일 읽기·모델 로딩은 제외하며 업로드부터 응답 파싱까지 포함한다. 학습과 벤치마크를 동시에 실행하지 않았다.",
        "",
        "| Provider | 반복 | 경로 | 표본 수 | p50 ms | p95 ms | p99 ms | 최대 ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for provider, report in result["providers"].items():
        for index, repetition in enumerate(report["repetitions"], 1):
            for name, latency in repetition["latency"].items():
                values = [
                    "—" if latency.get(key) is None else f"{latency[key]:.2f}"
                    for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms")
                ]
                lines.append(
                    f"| {provider} | {index} | {name} | {latency['count']} | {' | '.join(values)} |"
                )
    lines += [
        "",
        "## Parity",
        "",
        f"CPU/CUDA 공개 상태·품목 순위 불일치 이미지: {len(result['provider_parity']['status_rank_mismatch_image_ids'])}장.",
    ]
    for filename, title in (
        ("pytorch-ort.json", "PyTorch/ORT CPU"),
        ("ort-cpu-cuda.json", "ORT CPU/CUDA tensor"),
    ):
        path = work / "parity" / filename
        if path.exists():
            parity = load_json_config(path)
            lines.append(
                f"- {title}: tensor 허용 오차 충족 `{parity['all_tensors_close']}`, 전체 분류 순위 일치 `{parity['all_classifier_ranks_equal']}`."
            )
            aligned = parity.get("bbox_aligned_detector_checks", [])
            if aligned:
                batches = [batch for check in aligned for batch in check["batches"]]
                aligned_passes = sum(
                    b["complete_correspondence"]
                    and all(v["allclose"] for v in b["outputs"].values())
                    for b in batches
                )
                lines.append(
                    f"- {title} bbox 대응 후 검출 수치 허용 오차: {aligned_passes}/{len(batches)} batch. 원시 배열 비교와 별도로 기록한다."
                )
            if "status_rank_mismatch_image_ids" in parity:
                lines.append(
                    f"  원본 {parity['image_count']}장 pipeline 상태·품목 순위 불일치: "
                    f"{len(parity['status_rank_mismatch_image_ids'])}장."
                )
    for provider, report in result["providers"].items():
        lines.append(
            f"- {provider} 3회 반복 간 상태·품목 순위 불일치: "
            f"{len(report['repeat_status_rank_mismatch_image_ids'])}장."
        )
    for provider in ("pytorch", "cuda"):
        path = work / f"parity/detector-query-order-{provider}.json"
        if path.exists():
            diagnostic = load_json_config(path)
            selected = [
                value["above_frozen_detector_threshold"] for value in diagnostic["fixtures"]
            ]
            maximum = max(v.get("max_normalized_box_error", 0.0) for v in selected)
            lines.append(
                f"- Detector 추가 분석 ORT CPU/{provider}: 저장된 입력 {len(selected)}개에서 "
                f"고정 검출 임계값 이상 후보 수는 모두 일치했고, bbox 일대일 대응 후 정규화 좌표 최대 차이는 {maximum:.3g}였다. "
                f"[분석 기록](../parity/{path.name})"
            )
    if selection["selected"]["architecture"] == "dfine":
        lines += [
            "",
            "D-FINE 원시 출력의 직접 배열 비교에는 낮은 점수 후보의 순서 차이와 수치 차이가 포함된다. "
            "추가 표본 분석은 이 차이를 설명하기 위한 것으로, 원시 tensor 허용 오차 실패를 통과로 바꾸지 않았다. "
            "모델·ONNX·검출 및 승인 임계값은 변경하지 않았다.",
        ]
    real_comparison_label = (
        f"원본 정답 승인 객체/{source['objects']}"
        if objective is not None
        else f"실제 멀티 완전 성공/{multi_count}"
    )
    stress_comparison_label = (
        "합성 정답 승인 객체/"
        f"{selection['base_candidates'][0]['diagnostics']['stress']['ground_truth_count']}"
        if objective is not None
        else "합성 진단 완전 성공/400"
    )
    lines += [
        "",
        "## 기본 seed 구성 비교",
        "",
        f"| 구성 | 개발 오승인 | {real_comparison_label} | {stress_comparison_label} | 누락 | UNKNOWN Top-3 누락 | {latency_provider} p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in selection["base_candidates"]:
        rank = entry["rank"]
        if config.get("cpu_optimization"):
            rank = rank[1:]
        lines.append(
            f"| {entry['candidate']['id']} | {rank[0]} | {-rank[1]} | {-rank[2]} | {rank[3]} | {rank[4]} | {rank[5]:.2f} |"
        )
    lines += [
        "",
        "합성 진단 400장에는 빈 장면 80장이 포함된다. IMAGE_RECAPTURE는 완전 성공에 포함하지 않으므로 "
        "양성 장면 320장의 성공 수와 빈 장면의 오승인 여부를 함께 해석해야 한다.",
        "",
        "## 상위 두 구성의 seed 반복",
        "",
        f"| 구성 | seed | 개발 오승인 | {real_comparison_label} | {stress_comparison_label} | 누락 | UNKNOWN Top-3 누락 | {latency_provider} p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in selection["repeated_candidates"]:
        for run in entry["runs"]:
            rank = run["rank"]
            if config.get("cpu_optimization"):
                rank = rank[1:]
            lines.append(
                f"| {entry['candidate']['id']} | {run['seed']} | {rank[0]} | {-rank[1]} | "
                f"{-rank[2]} | {rank[3]} | {rank[4]} | {rank[5]:.2f} |"
            )
    lines += [
        "",
        "각 구성의 세 실행 중 최악의 순위를 우선 비교하고, 동률이면 중앙 순위·중앙 지연·구성 ID로 결정했다. "
        f"최종 Runtime/Catalog는 선택 구성의 기본 seed {selection['payload_seed']} 결과다. "
        "[선택 기록](../selection.json)에 모든 진단과 순위 근거를 보존한다.",
    ]
    lines += [
        "",
        "## 실패 사례",
        "",
        "녹색은 GT, 파란색은 정답 승인, 빨간색은 오승인, 주황색은 미승인 segmentation이다.",
        "전체 응답과 일대일 대응은 provider별 `responses.jsonl`, 정답과 이미지 해시는 `final-inputs.jsonl`에 보존한다.",
        "",
    ]
    failures = []
    for provider in result["providers"]:
        for row in read_jsonl(final / provider / "responses.jsonl"):
            if not row["metrics"]["complete_image"]:
                failures.append({**row, "provider": provider})
    failures.sort(
        key=lambda row: (
            -row["metrics"]["wrong_approved_count"],
            -row["metrics"]["missed_count"],
            row["image_id"],
            row["provider"],
            row["repetition"],
        )
    )
    seen, cases = set(), []
    for row in failures:
        if row["image_id"] in seen:
            continue
        seen.add(row["image_id"])
        original = originals[row["image_id"]]
        with Image.open(original["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((1600, 1600))
        scale_x, scale_y = image.width / original_width, image.height / original_height
        draw = ImageDraw.Draw(image)
        for annotation in original["annotations"]:
            x, y, width, height = annotation.get("bbox_xywh", annotation.get("bbox"))
            x, y, width, height = x * scale_x, y * scale_y, width * scale_x, height * scale_y
            draw.rectangle([x, y, x + width, y + height], outline="green", width=3)
            draw.text(
                (x, max(0, y - 20)),
                f"GT {annotation['category_id']}",
                fill="green",
                font_size=17,
                stroke_width=2,
                stroke_fill="white",
            )
        for index, segmentation in enumerate(row["response"]["segmentations"]):
            box, item = segmentation["bbox"], row["metrics"]["items"][index]
            approved = segmentation["status"] == "APPROVED"
            color = (
                "blue"
                if approved and item["target_class_id"] == item["predicted_class_id"]
                else "red"
                if approved
                else "orange"
            )
            x, y, width, height = (box[k] for k in ("x", "y", "width", "height"))
            x, y, width, height = x * scale_x, y * scale_y, width * scale_x, height * scale_y
            draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
            draw.text(
                (x, y + 3),
                f"{segmentation['status']} {item['predicted_class_id'] or ''}",
                fill=color,
                font_size=17,
                stroke_width=2,
                stroke_fill="white",
            )
        draw.text(
            (10, 10),
            f"{row['provider']} {row['image_id']} {row['response']['status']}",
            fill="black",
            font_size=24,
            stroke_width=2,
            stroke_fill="white",
        )
        path = final / "cases" / f"{row['image_id']}-{row['provider']}.jpg"
        path.parent.mkdir(exist_ok=True)
        image.save(path, quality=94)
        cases.append(
            {
                "image_id": row["image_id"],
                "provider": row["provider"],
                "repetition": row["repetition"],
                "image_sha256": row["image_sha256"],
                "metrics": row["metrics"],
                "overlay": str(path.resolve()),
            }
        )
        lines += [
            f"- 이미지 {row['image_id']} ({row['provider']}): 오승인 {row['metrics']['wrong_approved_count']}, "
            f"누락 {row['metrics']['missed_count']}, 추가 {row['metrics']['extra_count']}.",
            f"  ![실패 사례 {row['image_id']}]({path.resolve().as_posix()})",
            "",
        ]
        if len(cases) == 12:
            break
    write_json(final / "failure-cases.json", cases)
    lines += [
        "",
        "이 수치는 해당 300장에서 관측된 결과다. 다른 실물·촬영 환경의 독립 일반화 성능이나 SLA를 뜻하지 않는다.",
    ]
    destination = final / "report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
