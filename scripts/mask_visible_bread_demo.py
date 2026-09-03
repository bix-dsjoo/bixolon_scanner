from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from transformers import SamModel, SamProcessor


@dataclass(frozen=True)
class AutomaticPrompt:
    bread_seed: np.ndarray
    hand_seed: np.ndarray
    positive_point: tuple[int, int]
    negative_point: tuple[int, int] | None
    box: tuple[int, int, int, int]
    hand_distance_px: float | None


def _largest_component(mask: np.ndarray, *, minimum_area: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("no connected foreground component was found")
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    label = int(np.argmax(areas))
    if int(areas[label]) < minimum_area:
        raise ValueError("the largest foreground component is too small")
    return labels == label


def _top_connected_hand_seed(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    red_green = red - green
    green_blue = green - blue
    skin = (
        (red > 65)
        & (red_green > 12)
        & (red_green < 70)
        & (green_blue > 3)
        & (green_blue < 55)
        & ((red - blue) < 100)
    )
    skin = ndimage.binary_closing(skin, iterations=2)
    labels, count = ndimage.label(skin)
    hand = np.zeros(skin.shape, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        ys, _ = np.nonzero(component)
        if len(ys) >= 500 and int(ys.min()) < 20:
            hand |= component
    return hand


def automatic_prompt(image: Image.Image) -> AutomaticPrompt:
    rgb = np.asarray(image, dtype=np.uint8)
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    bread_seed = (red > 110) & ((red - green) > 38) & ((green - blue) > 18) & ((red - blue) > 85)
    bread_seed = ndimage.binary_opening(bread_seed, iterations=2)
    bread_seed = _largest_component(bread_seed, minimum_area=100)
    hand_seed = _top_connected_hand_seed(rgb)

    interior_distance = ndimage.distance_transform_edt(bread_seed)
    positive_y, positive_x = np.unravel_index(
        int(np.argmax(interior_distance)), interior_distance.shape
    )
    ys, xs = np.nonzero(bread_seed)
    padding = max(8, round(max(int(np.ptp(xs)), int(np.ptp(ys))) * 0.04))
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(image.width, int(xs.max()) + padding + 1)
    bottom = min(image.height, int(ys.max()) + padding + 1)

    negative_point = None
    hand_distance_px = None
    if hand_seed.any():
        distance_to_bread = ndimage.distance_transform_edt(~bread_seed)
        candidate = hand_seed & (distance_to_bread > 2.0)
        if candidate.any():
            candidate_distance = np.where(candidate, distance_to_bread, np.inf)
            negative_y, negative_x = np.unravel_index(
                int(np.argmin(candidate_distance)), candidate_distance.shape
            )
            negative_point = (int(negative_x), int(negative_y))
        distance_to_hand = ndimage.distance_transform_edt(~hand_seed)
        hand_distance_px = float(distance_to_hand[bread_seed].min())

    return AutomaticPrompt(
        bread_seed=bread_seed,
        hand_seed=hand_seed,
        positive_point=(int(positive_x), int(positive_y)),
        negative_point=negative_point,
        box=(left, top, right, bottom),
        hand_distance_px=hand_distance_px,
    )


def _select_mask(
    masks: np.ndarray,
    predicted_ious: np.ndarray,
    prompt: AutomaticPrompt,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    diagnostics = []
    best_index = 0
    best_score = -np.inf
    seed_area = int(prompt.bread_seed.sum())
    left, top, right, bottom = prompt.box
    allowed = np.zeros(prompt.bread_seed.shape, dtype=bool)
    allowed[top:bottom, left:right] = True
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=bool) & allowed
        intersection = int(np.count_nonzero(mask & prompt.bread_seed))
        coverage = intersection / seed_area
        area = max(1, int(mask.sum()))
        seed_precision = intersection / area
        score = float(predicted_ious[index]) + 2.0 * coverage + 0.25 * seed_precision
        diagnostics.append(
            {
                "candidate": index,
                "predicted_iou": float(predicted_ious[index]),
                "seed_coverage": coverage,
                "seed_precision": seed_precision,
                "selection_score": score,
            }
        )
        if score > best_score:
            best_index = index
            best_score = score
    selected = np.asarray(masks[best_index], dtype=bool) & allowed
    selected |= prompt.bread_seed
    if prompt.hand_seed.any():
        selected &= ~prompt.hand_seed
        selected |= prompt.bread_seed
    selected = ndimage.binary_fill_holes(selected)
    selected = _largest_component(selected, minimum_area=100)
    return selected, best_index, diagnostics


def _save_outputs(
    image: Image.Image,
    mask: np.ndarray,
    prompt: AutomaticPrompt,
    output_dir: Path,
    stem: str,
    status: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{stem}.mask.png"
    cutout_path = output_dir / f"{stem}.cutout.png"
    overlay_path = output_dir / f"{stem}.overlay.jpg"

    alpha = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    alpha.save(mask_path)
    cutout = image.convert("RGBA")
    cutout.putalpha(alpha)
    cutout.save(cutout_path)

    source = np.asarray(image, dtype=np.uint8)
    overlay = source.copy()
    overlay[mask] = np.rint(
        source[mask].astype(np.float32) * 0.6 + np.asarray([0, 230, 80], dtype=np.float32) * 0.4
    ).astype(np.uint8)
    panel = Image.fromarray(overlay)
    draw = ImageDraw.Draw(panel)
    draw.rectangle(prompt.box, outline=(255, 210, 0), width=4)
    x, y = prompt.positive_point
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(0, 90, 255))
    if prompt.negative_point is not None:
        x, y = prompt.negative_point
        draw.line((x - 8, y - 8, x + 8, y + 8), fill=(255, 0, 0), width=4)
        draw.line((x - 8, y + 8, x + 8, y - 8), fill=(255, 0, 0), width=4)
    draw.rectangle((12, 12, 245, 58), fill=(0, 0, 0))
    draw.text((24, 23), status, fill=(255, 255, 255))
    panel.save(overlay_path, quality=92)
    return {
        "mask": str(mask_path),
        "cutout": str(cutout_path),
        "overlay": str(overlay_path),
    }


def _save_contact_sheet(records: list[dict[str, object]], output_dir: Path) -> Path:
    columns = 2
    panel_size = (640, 360)
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (panel_size[0] * columns, panel_size[1] * rows), "white")
    for index, record in enumerate(records):
        outputs = record["outputs"]
        if not isinstance(outputs, dict):
            raise TypeError("record outputs must be a mapping")
        with Image.open(str(outputs["overlay"])) as opened:
            panel = opened.convert("RGB").resize(panel_size, Image.Resampling.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))
    path = output_dir / "contact-sheet.jpg"
    sheet.save(path, quality=92)
    return path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visible-bread masks with automatic color prompts and SAM refinement"
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    processor = SamProcessor.from_pretrained(args.model, local_files_only=not args.allow_download)
    model = (
        SamModel.from_pretrained(args.model, local_files_only=not args.allow_download)
        .to(device)
        .eval()
    )

    records = []
    for ordinal, source_path in enumerate(args.inputs, start=1):
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        prompt = automatic_prompt(image)
        points = [[list(prompt.positive_point)]]
        labels = [[1]]
        if prompt.negative_point is not None:
            points[0].append(list(prompt.negative_point))
            labels[0].append(0)
        inputs = processor(
            image,
            input_points=points,
            input_labels=labels,
            input_boxes=[[[*prompt.box]]],
            return_tensors="pt",
        )
        original_sizes = inputs.pop("original_sizes")
        reshaped_input_sizes = inputs.pop("reshaped_input_sizes")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes, reshaped_input_sizes
        )[0][0].numpy()
        predicted_ious = outputs.iou_scores.detach().cpu()[0, 0].numpy()
        mask, selected_index, candidates = _select_mask(masks, predicted_ious, prompt)
        status = (
            "REVIEW_HAND_CONTACT"
            if prompt.hand_distance_px is not None and prompt.hand_distance_px <= 2.0
            else "AUTO_ACCEPT"
        )
        stem = f"{ordinal:02d}_{source_path.stem}"
        paths = _save_outputs(image, mask, prompt, args.output_dir, stem, status)
        records.append(
            {
                "source": str(source_path),
                "status": status,
                "hand_distance_px": prompt.hand_distance_px,
                "prompt": {
                    "positive_point": prompt.positive_point,
                    "negative_point": prompt.negative_point,
                    "box_xyxy": prompt.box,
                },
                "selected_candidate": selected_index,
                "mask_area_pixels": int(mask.sum()),
                "mask_area_ratio": float(mask.mean()),
                "candidates": candidates,
                "outputs": paths,
            }
        )
    contact_sheet_path = _save_contact_sheet(records, args.output_dir)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "contact_sheet": str(contact_sheet_path),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "contact_sheet": str(contact_sheet_path),
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
