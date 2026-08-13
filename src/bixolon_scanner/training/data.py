from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageEnhance


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class ClassifierDataset:
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        *,
        mode: Literal["train", "final_train", "validation", "test"],
        fold: int = 0,
        image_size: int = 224,
        cache_dir: Path | None = None,
    ):
        import torchvision.transforms as transforms

        records = read_manifest(manifest_path)
        self.samples: list[tuple[Path, int, list[float] | None, str]] = []
        for record in records:
            if record["record_type"] == "classification":
                if mode in ("train", "final_train"):
                    self.samples.append(
                        (
                            dataset_root / record["image_path"],
                            record["category_id"] - 1,
                            None,
                            f"classification:{record['image_path']}",
                        )
                    )
                continue
            include = (
                mode == "test"
                and record["split"] == "test"
                or mode == "validation"
                and record["split"] == "development"
                and record["fold"] == fold
                or mode == "train"
                and record["split"] == "development"
                and record["fold"] != fold
                or mode == "final_train"
                and record["split"] == "development"
            )
            if include:
                for annotation in record["annotations"]:
                    self.samples.append(
                        (
                            dataset_root / record["image_path"],
                            annotation["category_id"] - 1,
                            annotation["bbox_xywh"],
                            f"roi-{'train' if mode in ('train', 'final_train') else 'exact'}:{record['image_id']}:{annotation['annotation_id']}",
                        )
                    )
        augmentation = (
            [
                transforms.RandomRotation(180, fill=(255, 255, 255)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            ]
            if mode in ("train", "final_train")
            else []
        )
        cached_crop = (
            [transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1))]
            if mode in ("train", "final_train") and cache_dir is not None
            else []
        )
        self.transform = transforms.Compose(
            augmentation
            + cached_crop
            + [
                *([] if cached_crop else [transforms.Resize((image_size, image_size))]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.training = mode in ("train", "final_train")
        self.cache_dir = cache_dir
        self.cache_index: dict[str, int | str] = {}
        self.cache_images = None
        if cache_dir is not None:
            metadata = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
            self.cache_index = metadata["index"]
            if "array_filename" in metadata:
                self.cache_images = np.load(cache_dir / metadata["array_filename"], mmap_mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target, bbox, cache_key = self.samples[index]
        cached_entry = self.cache_index.get(cache_key)
        if cached_entry is not None:
            array = (
                np.asarray(self.cache_images[int(cached_entry)])
                if self.cache_images is not None
                else np.load(self.cache_dir / str(cached_entry))
            )
            image = Image.fromarray(array, mode="RGB")
            bbox = None
        else:
            with Image.open(path) as source:
                image = source.convert("RGB")
        if bbox is not None:
            x, y, width, height = bbox
            if self.training:
                jitter = random.uniform(-0.05, 0.05)
                x -= width * jitter
                y -= height * jitter
                width *= 1.0 + 2.0 * jitter
                height *= 1.0 + 2.0 * jitter
            image = image.crop(
                (
                    max(0, int(x)),
                    max(0, int(y)),
                    min(image.width, int(x + width)),
                    min(image.height, int(y + height)),
                )
            )
        return self.transform(image), target


class DetectionDataset:
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        *,
        mode: Literal["train", "final_train", "validation", "test"],
        fold: int = 0,
        cache_dir: Path | None = None,
    ):
        records = read_manifest(manifest_path)
        self.records = []
        for record in records:
            if record["record_type"] != "detection":
                continue
            if record.get("exclude_from_detector_training", False) and mode != "test":
                continue
            if record.get("adaptation_replay_only", False) and mode in (
                "validation",
                "test",
            ):
                continue
            include = (
                mode == "test"
                and record["split"] == "test"
                or mode == "validation"
                and record["split"] == "development"
                and record["fold"] == fold
                or mode == "train"
                and record["split"] == "development"
                and record["fold"] != fold
                or mode == "final_train"
                and record["split"] == "development"
            )
            if include:
                self.records.append(record)
        self.dataset_root = dataset_root
        self.training = mode in ("train", "final_train")
        self.cache_index: dict[str, int] = {}
        self.cache_images = None
        if cache_dir is not None:
            metadata = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
            self.cache_index = metadata["index"]
            self.cache_images = np.load(cache_dir / metadata["array_filename"], mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        cached_row = self.cache_index.get(str(record["image_id"]))
        if cached_row is None:
            with Image.open(self.dataset_root / record["image_path"]) as source:
                image = source.convert("RGB")
            scale_x = scale_y = 1.0
        else:
            image = Image.fromarray(np.asarray(self.cache_images[cached_row]), mode="RGB")
            scale_x = image.width / record["width"]
            scale_y = image.height / record["height"]
        annotations = [dict(annotation) for annotation in record["annotations"]]
        if cached_row is not None:
            for annotation in annotations:
                x, y, width, height = annotation["bbox_xywh"]
                annotation["bbox_xywh"] = [
                    x * scale_x,
                    y * scale_y,
                    width * scale_x,
                    height * scale_y,
                ]
                annotation["area"] = width * scale_x * height * scale_y
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for annotation in annotations:
                x, y, width, height = annotation["bbox_xywh"]
                annotation["bbox_xywh"] = [image.width - x - width, y, width, height]
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        target = {
            "image_id": record["image_id"],
            "annotations": [
                {
                    "id": annotation["annotation_id"],
                    "image_id": record["image_id"],
                    "category_id": 0,
                    "bbox": annotation["bbox_xywh"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                }
                for annotation in annotations
            ],
        }
        return image, target
