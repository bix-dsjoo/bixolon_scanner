"""Capture local diagnostic evidence; never installs hooks in the HTTP Worker."""

from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery import score_response
from bixolon_scanner.pipeline import decision
from bixolon_scanner.runtime.imaging import decode_image
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json
from bixolon_scanner.worker.runtime_factory import build_worker_runtime
from bixolon_scanner.worker.settings import WorkerSettings


def main():
    root = Path("artifacts/retraining/recapture-0.2.1")
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    baseline = Path("artifacts/versions/0.2.0/staging")
    runtime = build_worker_runtime(
        WorkerSettings(
            package_dir=baseline / "runtime",
            catalog_dir=baseline / "catalog",
            provider="cpu",
            cpu_detector_intra_op_threads=8,
            cpu_embedder_intra_op_threads=12,
        )
    )
    captured = []
    original = decision.build_scan_items

    def capture(detections, batch, metadata, **kwargs):
        captured.append(
            {
                "detections": [asdict(d) for d in detections],
                "batch": {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in asdict(batch).items()
                },
                "border_indices": sorted(kwargs["border_indices"]),
                "duplicate_review_indices": sorted(kwargs["duplicate_review_indices"]),
            }
        )
        return original(detections, batch, metadata, **kwargs)

    decision.build_scan_items = capture
    rows, cases = [], []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    try:
        for record in records:
            path = Path(record["image_path"])
            assert sha256_file(path) == record["image_sha256"]
            with decode_image(
                path.read_bytes(),
                max_bytes=30_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.jpeg_draft_size,
            ) as decoded:
                response = runtime.scan(decoded, f"diagnosis-{record['image_id']}")
            metrics = score_response(response, record["annotations"], threshold=0.5)
            row = {
                "image_id": record["image_id"],
                "response": response.model_dump(mode="json"),
                "metrics": metrics,
                **captured[-1],
            }
            rows.append(row)
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            for i, (seg, item) in enumerate(
                zip(row["response"]["segmentations"], metrics["items"], strict=True)
            ):
                if seg["status"] != "SEGMENT_RECAPTURE":
                    continue
                d = row["detections"][i]
                b = row["batch"]
                reasons = []
                if d["score"] < 0.25:
                    reasons.append("low_detector_score")
                if b["segment_recapture_reasons"] and b["segment_recapture_reasons"][i]:
                    reasons.append(b["segment_recapture_reasons"][i])
                if i in row["border_indices"] and not b["approved"][i]:
                    reasons.append("border_low_confidence")
                if b["top3_unsafe"][i]:
                    reasons.append("top3_unsafe")
                case = {
                    "image_id": record["image_id"],
                    "ordinal": i + 1,
                    "target_index": item["target_index"],
                    "detector_score": d["score"],
                    "causes": reasons,
                    "classifier_approved": b["approved"][i],
                    "classifier_top1": b["decision_indices"][i][0],
                    "approval_score": b["approval_scores"][i],
                }
                cases.append(case)
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for j, target in enumerate(record["annotations"]):
                    x, y, w, h = target["bbox_xywh"]
                    draw.rectangle((x, y, x + w, y + h), outline="#00aa55", width=5)
                    draw.text(
                        (x, y),
                        f"GT{j} C{target['category_id']}",
                        fill="black",
                        stroke_width=2,
                        stroke_fill="white",
                        font=font,
                    )
                x, y, w, h = (seg["bbox"][k] for k in ("x", "y", "width", "height"))
                draw.rectangle((x, y, x + w, y + h), outline="red", width=7)
                pad = max(w, h) * 0.18
                box = (
                    max(0, x - pad),
                    max(0, y - pad),
                    min(image.width, x + w + pad),
                    min(image.height, y + h + pad),
                )
                tile = Image.new("RGB", (800, 340), "white")
                for offset, view in ((0, image.crop(box)), (400, annotated.crop(box))):
                    view.thumbnail((395, 270))
                    tile.paste(
                        view, (offset + (395 - view.width) // 2, 25 + (270 - view.height) // 2)
                    )
                title = f"{record['image_id']:03d}/S{i + 1:02d} GT={item['target_index']} det={d['score']:.4f}"
                ImageDraw.Draw(tile).text((5, 5), title, fill="black", font=font)
                ImageDraw.Draw(tile).text((5, 300), ", ".join(reasons), fill="black", font=font)
                name = f"case-{record['image_id']:03d}-{i + 1:02d}.jpg"
                tile.save(root / name, quality=93)
                case["image"] = name
            if len(rows) % 20 == 0:
                print("diagnosis", len(rows), flush=True)
    finally:
        decision.build_scan_items = original
        runtime.close()
    write_json(root / "baseline-trace.json", rows)
    write_json(root / "recapture-causes.json", cases)
    for start in range(0, len(cases), 10):
        page = Image.new("RGB", (1600, 1700), "white")
        for index, case in enumerate(cases[start : start + 10]):
            with Image.open(root / case["image"]) as tile:
                page.paste(tile, ((index % 2) * 800, (index // 2) * 340))
        page.save(root / f"review-{start // 10 + 1}.jpg", quality=90)
    print("cases", len(cases), flush=True)


if __name__ == "__main__":
    main()
