"""Score every matrix response against frozen GT; retain errors and all slow requests."""

import argparse
import json
from pathlib import Path

from bixolon_scanner.contracts import ScanResponse
from bixolon_scanner.evaluation.three_bakery import score_response
from bixolon_scanner.evaluation.three_bakery_http import summarize
from bixolon_scanner.operations.n100_test_kit import digest, timing, write_json


def signature(response):
    return {
        "status": response["status"],
        "reason_codes": response["reason_codes"],
        "segments": [
            {
                "bbox": segment["bbox"],
                "status": segment["status"],
                "reason_codes": segment["reason_codes"],
                "prediction": segment.get("prediction"),
                "top3": [item["class_id"] for item in segment["top3"]],
            }
            for segment in response["segmentations"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-profile", default="A0_gpu_reference")
    parser.add_argument("--repeat-profile", default="A9_gpu_reference_repeat")
    args = parser.parse_args()
    records = {
        row["image_id"]: row
        for row in map(json.loads, args.manifest.read_text(encoding="utf-8").splitlines())
    }
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    reference = {}
    for path in sorted(
        args.results.glob("*/responses.jsonl"),
        key=lambda p: (p.parent.name.startswith("A9_"), p.parent.name),
    ):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            record = records[row["image_id"]]
            if row["image_sha256"] != record["image_sha256"]:
                raise ValueError("Response input differs from frozen GT")
            response = ScanResponse.model_validate(row["response"])
            row["metrics"] = score_response(response, record["annotations"], threshold=0.5)
        repetitions = {}
        complete_repetitions = {}
        for repeat in sorted({row["repetition"] for row in rows}):
            selected = [row for row in rows if row["repetition"] == repeat]
            complete_repetitions[str(repeat)] = len(selected) == len(records) and {
                row["image_id"] for row in selected
            } == set(records)
            repetitions[str(repeat)] = summarize(selected)
        if path.parent.name == args.reference_profile:
            reference = {row["image_id"]: signature(row["response"]) for row in rows}
        parity = [
            {"image_id": row["image_id"], "repetition": row["repetition"]}
            for row in rows
            if row["image_id"] in reference
            and signature(row["response"]) != reference[row["image_id"]]
        ]
        report = {
            "reference_profile": args.reference_profile,
            "reference_available": bool(reference),
            "source_sha256": digest(path),
            "http_latency": timing([row["elapsed_ms"] for row in rows]),
            "repetitions": repetitions,
            "complete_repetitions": complete_repetitions,
            "http_errors": sum(row["http_status"] != 200 for row in rows),
            "strict_bbox_status_reason_rank_differences": parity,
            "over_1000ms": [
                {
                    "image_id": row["image_id"],
                    "repetition": row["repetition"],
                    "elapsed_ms": row["elapsed_ms"],
                }
                for row in rows
                if row["elapsed_ms"] > 1000
            ],
            "environment": json.loads(
                (path.parent / "environment.json").read_text(encoding="utf-8")
            ),
            "memory_max": {
                key: max(
                    (row.get("memory_after_request", {}).get(key, 0) for row in rows), default=0
                )
                for key in ("working_set_bytes", "private_bytes", "peak_working_set_bytes")
            },
        }
        reports[path.parent.name] = report
        write_json(args.output / f"{path.parent.name}.json", {"rows": rows, "report": report})
    failures = {
        p.parent.name: p.read_text(encoding="utf-8") for p in args.results.glob("*/FAILED.txt")
    }
    write_json(
        args.output / "analysis.json",
        {
            "profiles": reports,
            "failed_profiles": failures,
            "manifest_sha256": digest(args.manifest),
        },
    )
    cpu_names = sorted(
        {
            cpu.get("ProcessorNameString", "unknown")
            for report in reports.values()
            for cpu in report["environment"]["hardware"].get("cpu", [])
        }
    )
    lines = [
        "# N100 실험 매트릭스 검증",
        "",
        "측정 CPU: **"
        + ", ".join(cpu_names)
        + "**. 기록된 장비의 결과이며 N100 성능으로 환산하지 않습니다.",
        "",
        "HTTP 전체 요청 기준입니다. 오류·느린 요청을 제외하지 않습니다. 수집 장비와 실제 provider는 각 JSON에 보존합니다.",
        "",
        "| 설정 | p50 / p95 / p99 / 최대 ms | 1초 이내 | 정답 승인 / 오승인 / 미검출 | 승인 / UNKNOWN / 재촬영 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        latency = report["http_latency"]
        if not report["repetitions"]:
            lines.append(f"| {name} | 수집된 요청 없음 | 0 | 확인 불가 | 확인 불가 |")
            continue
        accuracy = next(iter(report["repetitions"].values()))
        states = accuracy["item_status_counts"]
        lines.append(
            f"| {name} | "
            + " / ".join(f"{latency[key]:.1f}" for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms"))
            + f" | {latency['within_1000ms_count']}/{latency['count']} | {accuracy['correct_approved_count']} / {accuracy['wrong_approved_count']} / {accuracy['missed_count']} | {states.get('APPROVED', 0)} / {states.get('UNKNOWN', 0)} / {states.get('SEGMENT_RECAPTURE', 0)} |"
        )
    lines += [
        "",
        "정확도·상태 수는 첫 반복 기준이며 모든 반복의 GT 결과는 analysis.json에 있습니다. 엄격한 bbox·상태·reason·Top-3 순위 차이는 정답 승인 수와 별도로 보존합니다.",
        "",
        f"실패 설정: {len(failures)}개. 매트릭스 1회 결과로 1초 보장이나 최종 후보 확정을 주장하지 않습니다.",
    ]
    if args.reference_profile in reports and args.repeat_profile in reports:
        before = reports[args.reference_profile]["http_latency"]["p95_ms"]
        after = reports[args.repeat_profile]["http_latency"]["p95_ms"]
        lines += [
            "",
            f"동일 기준 설정의 시작/종료 p95: {before:.2f} → {after:.2f}ms ({(after / before - 1) * 100:+.2f}%). 변동이 크면 단순 순차 비교로 후보의 속도 우열을 확정하지 않습니다.",
        ]
    (args.output / "REPORT-KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "REPORT-KO.md")


if __name__ == "__main__":
    main()
