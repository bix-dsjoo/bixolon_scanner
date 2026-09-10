"""Render original/GT/before/after evidence for all 38 baseline recapture regions."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def box_key(seg):
    return tuple(seg["bbox"][key] for key in ("x", "y", "width", "height"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()
    work = Path("artifacts/retraining/recapture-0.2.1")
    output = work / ("visual-context-report" if args.context else "visual-report")
    measurement = work / (
        "measurements/context-study/context040-cpu"
        if args.context
        else "measurements/development/score040-cpu"
    )
    output.mkdir(exist_ok=True)
    records = {
        r["image_id"]: r
        for r in read_jsonl(
            Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
        )
    }
    baseline = {
        r["image_id"]: r
        for r in json.loads((work / "baseline-trace.json").read_text(encoding="utf-8"))
    }
    candidate = {r["image_id"]: r for r in read_jsonl(measurement / "responses.jsonl")}
    cases = json.loads((work / "recapture-causes.json").read_text(encoding="utf-8"))
    assert len(cases) == 38
    colors = {"APPROVED": "#2465dc", "UNKNOWN": "#d08300", "SEGMENT_RECAPTURE": "#e02132"}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    scenes = {}
    for image_id in sorted({r["image_id"] for r in cases}):
        record = records[image_id]
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        size = (600, 600)
        preview = image.copy()
        preview.thumbnail(size)
        sx, sy = preview.width / image.width, preview.height / image.height
        page = Image.new("RGB", (1800, 650), "white")
        for index, title in enumerate(("Original + GT", "0.2.0 + GT", "0.2.1 + GT")):
            view = preview.copy()
            draw = ImageDraw.Draw(view)
            for j, target in enumerate(record["annotations"]):
                x, y, w, h = target["bbox_xywh"]
                draw.rectangle(
                    (x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline="#00a65e", width=2
                )
                draw.text(
                    (x * sx, y * sy),
                    f"G{j}/C{target['category_id']}",
                    font=font,
                    fill="#006633",
                    stroke_width=1,
                    stroke_fill="white",
                )
            if index:
                response = (baseline if index == 1 else candidate)[image_id]["response"]
                for seg in response["segmentations"]:
                    x, y, w, h = box_key(seg)
                    draw.rectangle(
                        (x * sx, y * sy, (x + w) * sx, (y + h) * sy),
                        outline=colors[seg["status"]],
                        width=3,
                    )
            page.paste(view, (index * 600, 40))
            ImageDraw.Draw(page).text((index * 600 + 8, 8), title, font=font, fill="black")
        name = f"scene-{image_id:03d}.jpg"
        page.save(output / name, quality=93)
        scenes[image_id] = name
    rows = []
    for case in cases:
        image_id = case["image_id"]
        old = baseline[image_id]["response"]["segmentations"][case["ordinal"] - 1]
        new = next(
            (
                seg
                for seg in candidate[image_id]["response"]["segmentations"]
                if box_key(seg) == box_key(old)
            ),
            None,
        )
        assert case["target_index"] is None or new is not None
        rows.append(
            {
                **case,
                "after_status": new["status"] if new else "PROPOSAL_REMOVED",
                "after_prediction": new["prediction"] if new else None,
                "scene_image": scenes[image_id],
            }
        )
    write_json(output / "38-region-comparison.json", rows)
    before = load_json_config(work / "measurements/development/baseline-cpu/report.json")["summary"]
    after = load_json_config(measurement / "report.json")["summary"]
    changes = []
    for image_id, row in baseline.items():
        b = {m["target_index"]: m for m in row["metrics"]["items"] if m["target_index"] is not None}
        c = {
            m["target_index"]: m
            for m in candidate[image_id]["metrics"]["items"]
            if m["target_index"] is not None
        }
        assert b.keys() == c.keys()
        for j, m in b.items():
            if m["status"] != c[j]["status"]:
                changes.append(
                    {"image_id": image_id, "target_index": j, "before": m, "after": c[j]}
                )
    write_json(
        output / "comparison.json",
        {
            "before": before,
            "after": after,
            "gt_status_changes": changes,
            "recapture_transitions": dict(Counter(r["after_status"] for r in rows)),
            "all_gt_preserved": True,
        },
    )
    table = "".join(
        f"<tr><td>{label}</td><td>{before[key]}</td><td>{after[key]}</td></tr>"
        for label, key in [
            ("GT 객체", "ground_truth_count"),
            ("정답 승인", "correct_approved_count"),
            ("오승인", "wrong_approved_count"),
            ("미검출", "missed_count"),
            ("추가 검출", "extra_count"),
        ]
    )
    cards = []
    for row in rows:
        label = f"로그 {row['image_id']:03d} / 영역 {row['ordinal']:02d}"
        target = (
            "GT 미대응 추가 박스"
            if row["target_index"] is None
            else f"실제 GT {row['target_index']}"
        )
        cards.append(
            f'<article><h2>{label} · {target}</h2><p>검출 점수 {row["detector_score"]:.5f} · 원인: {html.escape(", ".join(row["causes"]))} · 이후: <strong>{row["after_status"]}</strong></p><img loading="lazy" src="../{row["image"]}" alt="원본 영역과 GT 대조"><details><summary>전체 원본·GT·0.2.0·0.2.1 비교</summary><a href="{row["scene_image"]}"><img loading="lazy" src="{row["scene_image"]}" alt="전체 장면 비교"></a></details></article>'
        )
    output.joinpath("comparison.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>0.2.1 재촬영 38개 비교</title><style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;background:#f5f5f4;color:#202020}table{border-collapse:collapse}td,th{padding:10px 22px;border-bottom:1px solid #ccc}article{background:white;padding:20px;margin:24px 0}img{max-width:100%;height:auto}summary{cursor:pointer;padding:14px}h2{font-size:20px}</style><h1>0.2.1 재촬영 38개 영역 비교</h1><p>고정 로그 132장 / GT 1,096개. 원본 GT는 변경하지 않았으며 전체 GT의 일대일 대응을 보존했습니다. 초록 GT · 파랑 승인 · 주황 UNKNOWN · 빨강 재촬영.</p><p>이 자료는 노출된 로그의 개발 진단입니다. 속도와 배포 검증은 최종 보고서를 참조하십시오.</p><table><tr><th>지표</th><th>0.2.0</th><th>0.2.1 후보</th></tr>'
        + table
        + "</table>"
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )
    print("Rendered", len(rows), "regions; GT changes:", changes, flush=True)


if __name__ == "__main__":
    main()
