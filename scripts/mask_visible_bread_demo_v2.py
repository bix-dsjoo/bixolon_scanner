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
    bread_point: tuple[int, int]
    bread_box: tuple[int, int, int, int]
    hand_point: tuple[int, int] | None
    hand_box: tuple[int, int, int, int] | None
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


def _mask_box(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    minimum_padding: int,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot build a box around an empty mask")
    padding = max(
        minimum_padding,
        round(max(int(np.ptp(xs)), int(np.ptp(ys))) * padding_ratio),
    )
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(image_width, int(xs.max()) + padding + 1),
        min(image_height, int(ys.max()) + padding + 1),
    )


def _deepest_point(mask: np.ndarray) -> tuple[int, int]:
    distance = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


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

    bread_point = _deepest_point(bread_seed)
    bread_box = _mask_box(
        bread_seed,
        image_width=image.width,
        image_height=image.height,
        padding_ratio=0.04,
        minimum_padding=8,
    )

    hand_point = None
    hand_box = None
    hand_distance_px = None
    if hand_seed.any():
        clean_hand_seed = hand_seed & ~ndimage.binary_dilation(bread_seed, iterations=5)
        clean_hand_seed = ndimage.binary_opening(clean_hand_seed, iterations=1)
        if clean_hand_seed.any():
            hand_point = _deepest_point(clean_hand_seed)
            hand_box = _mask_box(
                hand_seed,
                image_width=image.width,
                image_height=image.height,
                padding_ratio=0.02,
                minimum_padding=8,
            )
        distance_to_hand = ndimage.distance_transform_edt(~hand_seed)
        hand_distance_px = float(distance_to_hand[bread_seed].min())

    return AutomaticPrompt(
        bread_seed=bread_seed,
        hand_seed=hand_seed,
        bread_point=bread_point,
        bread_box=bread_box,
        hand_point=hand_point,
        hand_box=hand_box,
        hand_distance_px=hand_distance_px,
    )


def _sam_candidates(
    image: Image.Image,
    processor: SamProcessor,
    model: SamModel,
    device: str,
    *,
    points: list[tuple[int, int]],
    labels: list[int],
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inputs = processor(
        image,
        input_points=[[list(point) for point in points]],
        input_labels=[labels],
        input_boxes=[[[*box]]],
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
    return masks, predicted_ious


def _select_mask(
    masks: np.ndarray,
    predicted_ious: np.ndarray,
    *,
    seed: np.ndarray,
    box: tuple[int, int, int, int],
    exclusion: np.ndarray | None = None,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    allowed = np.zeros(seed.shape, dtype=bool)
    left, top, right, bottom = box
    allowed[top:bottom, left:right] = True
    seed_area = max(1, int(seed.sum()))
    diagnostics = []
    best_index = 0
    best_score = -np.inf
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=bool) & allowed
        area = max(1, int(mask.sum()))
        intersection = int(np.count_nonzero(mask & seed))
        coverage = intersection / seed_area
        seed_precision = intersection / area
        exclusion_overlap = (
            0.0 if exclusion is None else int(np.count_nonzero(mask & exclusion)) / area
        )
        score = (
            float(predicted_ious[index])
            + 2.0 * coverage
            + 0.25 * seed_precision
            - 2.0 * exclusion_overlap
        )
        diagnostics.append(
            {
                "candidate": index,
                "predicted_iou": float(predicted_ious[index]),
                "seed_coverage": coverage,
                "seed_precision": seed_precision,
                "exclusion_overlap": exclusion_overlap,
                "selection_score": score,
            }
        )
        if score > best_score:
            best_index = index
            best_score = score
    return np.asarray(masks[best_index], dtype=bool) & allowed, best_index, diagnostics


def _contact_negative_points(
    hand_mask: np.ndarray,
    bread_seed: np.ndarray,
    *,
    maximum: int = 4,
) -> list[tuple[int, int]]:
    distance_to_bread = ndimage.distance_transform_edt(~bread_seed)
    candidate = hand_mask & (distance_to_bread > 2.0) & (distance_to_bread <= 35.0)
    coordinates = np.argwhere(candidate)
    if not len(coordinates):
        return []
    order = np.argsort(distance_to_bread[candidate])
    selected: list[tuple[int, int]] = []
    for coordinate_index in order:
        y, x = coordinates[int(coordinate_index)]
        point = (int(x), int(y))
        if all(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 >= 24**2 for other in selected
        ):
            selected.append(point)
        if len(selected) == maximum:
            break
    return selected


def _keep_seeded_components(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        if int(component.sum()) >= 50 and np.any(component & seed):
            output |= component
    if not output.any():
        return _largest_component(mask, minimum_area=50)
    return output


def _visible_bread_mask(
    bread_mask: np.ndarray,
    hand_mask: np.ndarray,
    bread_seed: np.ndarray,
    *,
    hand_margin_px: int,
) -> np.ndarray:
    hand_margin = ndimage.binary_dilation(hand_mask, iterations=hand_margin_px)
    visible = bread_mask & ~hand_margin
    clean_seed = bread_seed & ~hand_margin
    visible = _keep_seeded_components(visible, clean_seed)
    filled = ndimage.binary_fill_holes(visible) & ~hand_margin
    return _keep_seeded_components(filled, clean_seed)


def _save_outputs(
    image: Image.Image,
    bread_mask: np.ndarray,
    hand_mask: np.ndarray,
    prompt: AutomaticPrompt,
    negative_points: list[tuple[int, int]],
    output_dir: Path,
    stem: str,
    status: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{stem}.mask.png"
    hand_mask_path = output_dir / f"{stem}.hand-mask.png"
    cutout_path = output_dir / f"{stem}.cutout.png"
    overlay_path = output_dir / f"{stem}.overlay.jpg"

    alpha = Image.fromarray(bread_mask.astype(np.uint8) * 255, mode="L")
    alpha.save(mask_path)
    Image.fromarray(hand_mask.astype(np.uint8) * 255, mode="L").save(hand_mask_path)
    cutout = image.convert("RGBA")
    cutout.putalpha(alpha)
    cutout.save(cutout_path)

    source = np.asarray(image, dtype=np.uint8)
    overlay = source.copy()
    overlay[hand_mask] = np.rint(
        source[hand_mask].astype(np.float32) * 0.72
        + np.asarray([255, 55, 35], dtype=np.float32) * 0.28
    ).astype(np.uint8)
    overlay[bread_mask] = np.rint(
        source[bread_mask].astype(np.float32) * 0.58
        + np.asarray([0, 230, 80], dtype=np.float32) * 0.42
    ).astype(np.uint8)
    panel = Image.fromarray(overlay)
    draw = ImageDraw.Draw(panel)
    draw.rectangle(prompt.bread_box, outline=(255, 210, 0), width=4)
    if prompt.hand_box is not None:
        draw.rectangle(prompt.hand_box, outline=(0, 210, 255), width=3)
    x, y = prompt.bread_point
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(0, 90, 255))
    for x, y in negative_points:
        draw.line((x - 7, y - 7, x + 7, y + 7), fill=(255, 0, 0), width=4)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=(255, 0, 0), width=4)
    draw.rectangle((12, 12, 315, 58), fill=(0, 0, 0))
    draw.text((24, 23), status, fill=(255, 255, 255))
    panel.save(overlay_path, quality=92)
    return {
        "mask": str(mask_path),
        "hand_mask": str(hand_mask_path),
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
        description="Create visible-bread masks with separate SAM hand and bread masks"
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hand-margin-px", type=int, default=5)
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

        hand_mask = np.zeros(prompt.bread_seed.shape, dtype=bool)
        hand_selected_index = None
        hand_candidates: list[dict[str, float]] = []
        if prompt.hand_point is not None and prompt.hand_box is not None:
            hand_masks, hand_ious = _sam_candidates(
                image,
                processor,
                model,
                device,
                points=[prompt.hand_point, prompt.bread_point],
                labels=[1, 0],
                box=prompt.hand_box,
            )
            hand_mask, hand_selected_index, hand_candidates = _select_mask(
                hand_masks,
                hand_ious,
                seed=prompt.hand_seed,
                box=prompt.hand_box,
                exclusion=prompt.bread_seed,
            )
            hand_mask = _largest_component(hand_mask, minimum_area=500)

        negative_points = _contact_negative_points(hand_mask, prompt.bread_seed)
        bread_masks, bread_ious = _sam_candidates(
            image,
            processor,
            model,
            device,
            points=[prompt.bread_point, *negative_points],
            labels=[1, *([0] * len(negative_points))],
            box=prompt.bread_box,
        )
        raw_bread_mask, bread_selected_index, bread_candidates = _select_mask(
            bread_masks,
            bread_ious,
            seed=prompt.bread_seed,
            box=prompt.bread_box,
            exclusion=hand_mask,
        )
        bread_mask = _visible_bread_mask(
            raw_bread_mask,
            hand_mask,
            prompt.bread_seed,
            hand_margin_px=args.hand_margin_px,
        )
        status = (
            "REVIEW_HAND_OCCLUSION"
            if prompt.hand_distance_px is not None and prompt.hand_distance_px <= 2.0
            else "AUTO_ACCEPT"
        )
        stem = f"{ordinal:02d}_{source_path.stem}"
        paths = _save_outputs(
            image,
            bread_mask,
            hand_mask,
            prompt,
            negative_points,
            args.output_dir,
            stem,
            status,
        )
        records.append(
            {
                "source": str(source_path),
                "status": status,
                "hand_distance_px": prompt.hand_distance_px,
                "prompt": {
                    "bread_point": prompt.bread_point,
                    "bread_box_xyxy": prompt.bread_box,
                    "hand_point": prompt.hand_point,
                    "hand_box_xyxy": prompt.hand_box,
                    "bread_negative_points": negative_points,
                },
                "bread_selected_candidate": bread_selected_index,
                "hand_selected_candidate": hand_selected_index,
                "mask_area_pixels": int(bread_mask.sum()),
                "mask_area_ratio": float(bread_mask.mean()),
                "hand_overlap_removed_pixels": int(np.count_nonzero(raw_bread_mask & hand_mask)),
                "bread_candidates": bread_candidates,
                "hand_candidates": hand_candidates,
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
                "hand_margin_px": args.hand_margin_px,
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
