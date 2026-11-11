"""Detector metrics: image level, and pixel level on a shared region.

Each detector saves, with every map, the box of the original image the map
covers (crop_box). Pixels are scored where all boxes overlap (common_box),
sampled onto a 256 x 256 grid: masks by nearest neighbour, maps bilinearly at
the cell centres. Normal images take part with an empty mask.

Per category: image AUROC, AP and F1-max; pixel AUROC, AP and F1-max; and
AU-PRO up to a false-positive rate of 0.3 with 8-connected regions (Bergmann
et al., CVPR 2020). F1-max picks its threshold on the test labels.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix_at_thresholds,
    roc_auc_score,
    roc_curve,
)

GRID_SIZE = 256
FPR_LIMIT = 0.3

Box = tuple[float, float, float, float]  # left, top, right, bottom in pixels


def crop_box(size: tuple[int, int], resize: int | tuple[int, int], crop: int) -> Box:
    """Part of an image kept by Resize(resize) and then CenterCrop(crop).

    size is (width, height); resize is an int (shorter side) or a
    (height, width) pair, as in torchvision.
    """
    width, height = size
    if isinstance(resize, int):
        short = min(width, height)
        new_width = resize * width // short
        new_height = resize * height // short
    else:
        new_height, new_width = resize

    # Same rounding as torchvision's CenterCrop.
    left = round((new_width - crop) / 2)
    top = round((new_height - crop) / 2)
    return (
        left * width / new_width,
        top * height / new_height,
        (left + crop) * width / new_width,
        (top + crop) * height / new_height,
    )


def common_box(boxes: list[Box]) -> Box:
    """Part of the image covered by all the given map boxes."""
    lefts, tops, rights, bottoms = zip(*boxes)
    return max(lefts), max(tops), min(rights), min(bottoms)


def sample_mask(mask: np.ndarray, box: Box) -> np.ndarray:
    """Boolean mask on the evaluation grid (nearest-neighbour sampling)."""
    return _nearest(np.uint8(mask > 0), box) > 0


def sample_map(anomaly_map: np.ndarray, map_box: Box, box: Box) -> np.ndarray:
    """Anomaly map on the evaluation grid (bilinear sampling, float32).

    map_box is the part of the image that the map covers.
    """
    left, top, right, bottom = box
    map_left, map_top, map_right, map_bottom = map_box
    centres = torch.arange(GRID_SIZE, dtype=torch.float32) + 0.5
    x = left + centres * ((right - left) / GRID_SIZE)
    y = top + centres * ((bottom - top) / GRID_SIZE)

    # grid_sample puts -1 and 1 at the outer edges of the map.
    grid_x = 2 * (x - map_left) / (map_right - map_left) - 1
    grid_y = 2 * (y - map_top) / (map_bottom - map_top) - 1
    rows, cols = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack((cols, rows), dim=-1)[None]

    values = np.ascontiguousarray(anomaly_map, dtype=np.float32)
    sampled = F.grid_sample(
        torch.from_numpy(values)[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, 0].numpy()


def lost_regions(mask: np.ndarray, box: Box) -> tuple[int, int]:
    """Defect regions of a full-size mask that the evaluation grid loses.

    Returns (cropped, missed): regions lying wholly outside the box, and
    regions reaching into it that no grid cell samples.
    """
    regions = measure.label(mask, connectivity=2)  # 8-connected
    regions = regions.astype(np.int32)  # PIL cannot sample int64
    height, width = regions.shape
    left, top, right, bottom = box

    # Pixel (x, y) covers [x, x+1) x [y, y+1); keep pixels overlapping the box.
    rows, cols = np.arange(height), np.arange(width)
    inside = regions[
        np.ix_((rows + 1 > top) & (rows < bottom), (cols + 1 > left) & (cols < right))
    ]
    kept = set(np.unique(inside)) - {0}
    sampled = set(np.unique(_nearest(regions, box)))
    return int(regions.max()) - len(kept), len(kept - sampled)


def _nearest(array: np.ndarray, box: Box) -> np.ndarray:
    """Values of the box on the evaluation grid, nearest neighbour."""
    image = Image.fromarray(array).resize(
        (GRID_SIZE, GRID_SIZE), Image.Resampling.NEAREST, box=box
    )
    return np.asarray(image)


def f1_max(targets: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 over all thresholds; a score >= threshold means anomalous."""
    _, fp, fn, tp, _ = confusion_matrix_at_thresholds(targets, scores, pos_label=1)
    return float(np.max(2 * tp / (2 * tp + fp + fn)))


# aupro() is adapted from Anomalib, src/anomalib/metrics/aupro.py
# (_AUPRO.compute_pro), commit 0be85508edde53a2b1062ee6d1732c85c5b7172e,
# Copyright (C) 2022-2026 Intel Corporation, Apache License 2.0.
# Changed: regions are labelled with scikit-image instead of OpenCV, and the
# curve comes from scikit-learn's roc_curve instead of torch cumulative sums.
def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = FPR_LIMIT) -> float:
    """Normalised area under the PRO curve up to fpr_limit.

    masks and maps hold the images of one category on the evaluation grid.
    """
    masks = np.asarray(masks, dtype=bool)

    # With weight 1/area on each defect pixel, the weighted true-positive
    # rate is the mean overlap of all defect regions.
    weights = np.ones(masks.shape)
    for weight, mask in zip(weights, masks):
        regions = measure.label(mask, connectivity=2)  # 8-connected
        areas = np.bincount(regions.ravel())
        weight[mask] = 1 / areas[regions[mask]]
    fpr, pro, _ = roc_curve(
        masks.ravel(),
        maps.ravel(),
        sample_weight=weights.ravel(),
        drop_intermediate=False,
    )

    # Keep the curve up to the limit and end it exactly there.
    end = np.searchsorted(fpr, fpr_limit, side="right")
    x, y = fpr[:end], pro[:end]
    if x[-1] < fpr_limit:
        fraction = (fpr_limit - x[-1]) / (fpr[end] - x[-1])
        y = np.append(y, y[-1] + fraction * (pro[end] - y[-1]))
        x = np.append(x, fpr_limit)
    return float(np.trapezoid(y, x) / fpr_limit)


def category_metrics(
    labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray
) -> dict:
    """Seven metrics of one category, as fractions.

    labels and scores hold one value per test image; masks and maps hold the
    same images on the evaluation grid. A category needs normal and anomalous
    images and pixels, or some metrics are undefined.
    """
    pixels, values = masks.ravel(), maps.ravel()
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "image_f1_max": f1_max(labels, scores),
        "pixel_auroc": float(roc_auc_score(pixels, values)),
        "pixel_ap": float(average_precision_score(pixels, values)),
        "pixel_f1_max": f1_max(pixels, values),
        "aupro": aupro(masks, maps),
    }
