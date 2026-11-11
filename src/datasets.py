"""Image lists for MVTec AD and VisA.

MVTec AD (Bergmann et al., CVPR 2019) is read from its release folders; VisA
(Zou et al., ECCV 2022) through its one-class split, split_csv/1cls.csv, as
in amazon-science/spot-diff utils/prepare_data.py. An image ID is
"<dataset>/<path inside the dataset folder>". Lists are sorted by ID, so
seeded sampling sees the same order every run.
"""

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

# Dataset folders, relative to the repository root.
DATA_ROOTS = {"mvtec_ad": Path("data/MVTec-AD"), "visa": Path("data/VisA")}
VISA_SPLIT_FILE = "split_csv/1cls.csv"


class Sample(NamedTuple):
    image_id: str
    image_path: Path
    mask_path: Path | None  # None for normal images
    label: int  # 0 = normal, 1 = anomalous


def list_categories(dataset: str, root: str | Path) -> list[str]:
    """Category names found in a dataset folder."""
    root = Path(root)
    if dataset == "mvtec_ad":
        return sorted(path.name for path in root.iterdir() if (path / "train").is_dir())
    if dataset == "visa":
        return sorted({row["object"] for row in _visa_rows(root)})
    raise ValueError(f"Unknown dataset: {dataset}")


def list_samples(
    dataset: str, root: str | Path, category: str, split: str
) -> list[Sample]:
    """Images of one category and split ("train" or "test"), sorted by ID."""
    root = Path(root)
    if dataset == "mvtec_ad":
        samples = _mvtec_samples(root, category, split)
    elif dataset == "visa":
        samples = _visa_samples(root, category, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not samples:
        raise ValueError(f"No {split} images for {dataset}/{category} in {root}")
    return sorted(samples, key=lambda sample: sample.image_id)


def read_image(path: str | Path) -> Image.Image:
    """Read an image as 8-bit RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")


def read_mask(sample: Sample) -> np.ndarray:
    """Boolean defect mask at image size; all False for a normal image.

    Non-zero pixels are defects (MVTec AD stores 255, VisA a class number),
    as in spot-diff utils/prepare_data.py.
    """
    if sample.mask_path is None:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        return np.zeros((height, width), dtype=bool)
    with Image.open(sample.mask_path) as mask:
        return np.asarray(mask) > 0


class ImageList:
    """Samples as a dataset for torch's DataLoader: item i is transform(image i)."""

    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.transform(read_image(self.samples[index].image_path))


def _mvtec_samples(root: Path, category: str, split: str) -> list[Sample]:
    samples = []
    for image_path in (root / category / split).glob("*/*.png"):
        defect = image_path.parent.name
        if defect.startswith(".") or image_path.name.startswith("."):
            continue  # hidden files, e.g. "._000.png" copies made by macOS
        label = int(defect != "good")
        mask_path = None
        if label:
            mask_dir = root / category / "ground_truth" / defect
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
        image_id = f"mvtec_ad/{image_path.relative_to(root).as_posix()}"
        samples.append(Sample(image_id, image_path, mask_path, label))
    return samples


def _visa_samples(root: Path, category: str, split: str) -> list[Sample]:
    return [
        Sample(
            image_id=f"visa/{row['image']}",
            image_path=root / row["image"],
            mask_path=root / row["mask"] if row["mask"] else None,
            label=int(row["label"] == "anomaly"),
        )
        for row in _visa_rows(root)
        if row["object"] == category and row["split"] == split
    ]


def _visa_rows(root: Path) -> list[dict[str, str]]:
    split_file = root / VISA_SPLIT_FILE
    with open(split_file, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        anomalous = row["label"] == "anomaly"
        if (
            row["split"] not in ("train", "test")
            or row["label"] not in ("normal", "anomaly")
            or (row["split"] == "train" and anomalous)
            or anomalous != bool(row["mask"])
        ):
            raise ValueError(f"Unexpected row in {split_file}: {row}")
    return rows
