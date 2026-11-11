"""Compare the three detectors with the same metrics on the same pixels.

Reads the outputs of 01 to 03. Image metrics use each detector's own image
scores; pixel metrics use the area all three maps cover, sampled onto a
256 x 256 grid (see src/metrics.py). A dataset's value is the unweighted mean
of its unrounded category values.

Writes outputs/detector/<detector>/metrics.csv and summary.csv (percent) and
prints how many defect regions the shared area loses.
Run from the repository root: python -m scripts.04_compare_detectors
"""

import csv
from statistics import mean

import numpy as np

from src.datasets import DATA_ROOTS, list_categories, list_samples, read_mask
from src.metrics import (
    category_metrics,
    common_box,
    lost_regions,
    sample_map,
    sample_mask,
)
from src.predictions import OUTPUTS, load_category

DETECTORS = ("patchcore", "efficientad-s", "dinomaly")
METRICS = (
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "pixel_f1_max",
    "aupro",
)


def evaluate_category(dataset, category):
    """Metrics of each detector, and the defect regions the grid loses."""
    samples = list_samples(dataset, DATA_ROOTS[dataset], category, "test")
    saved = {name: load_category(name, dataset, category) for name in DETECTORS}
    expected = sorted(sample.image_id for sample in samples)
    for name in DETECTORS:
        if sorted(saved[name]["image_ids"].tolist()) != expected:
            raise ValueError(
                f"{name}, {dataset}/{category}: the saved image "
                "IDs differ from the test images"
            )
    rows = {
        name: {
            image_id: i for i, image_id in enumerate(saved[name]["image_ids"].tolist())
        }
        for name in DETECTORS
    }

    masks, maps = [], {name: [] for name in DETECTORS}
    lost = np.zeros(3, dtype=int)  # cropped, missed, empty anomalous masks
    for sample in samples:
        index = {name: rows[name][sample.image_id] for name in DETECTORS}
        boxes = {name: saved[name]["map_boxes"][index[name]] for name in DETECTORS}
        box = common_box(list(boxes.values()))
        mask = read_mask(sample)
        grid = sample_mask(mask, box)
        masks.append(grid)
        cropped, missed = lost_regions(mask, box)
        lost += [cropped, missed, sample.label == 1 and not grid.any()]
        for name in DETECTORS:
            anomaly_map = saved[name]["anomaly_maps"][index[name]]
            maps[name].append(sample_map(anomaly_map, boxes[name], box))

    labels = np.array([sample.label for sample in samples])
    results = {}
    for name in DETECTORS:
        scores = saved[name]["image_scores"][
            [rows[name][sample.image_id] for sample in samples]
        ]
        results[name] = category_metrics(
            labels, scores, np.stack(masks), np.stack(maps[name])
        )
    return results, lost


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percent(metrics):
    """Metric values as percentages with two decimals, as text."""
    return {name: f"{100 * metrics[name]:.2f}" for name in METRICS}


def main():
    rows = {name: [] for name in DETECTORS}
    summary = {name: [] for name in DETECTORS}
    for dataset, root in DATA_ROOTS.items():
        results = {name: {} for name in DETECTORS}
        lost = np.zeros(3, dtype=int)
        for category in list_categories(dataset, root):
            by_detector, category_lost = evaluate_category(dataset, category)
            for name in DETECTORS:
                results[name][category] = by_detector[name]
            lost += category_lost
            print(f"{dataset}/{category}: evaluated")

        print(f"\n{dataset}, mean over categories (%)")
        print(f"{'detector':14s}" + "".join(f"{m:>13s}" for m in METRICS))
        for name in DETECTORS:
            rows[name] += [
                {"dataset": dataset, "category": category, **percent(metrics)}
                for category, metrics in results[name].items()
            ]
            means = {m: mean(r[m] for r in results[name].values()) for m in METRICS}
            summary[name].append({"dataset": dataset, **percent(means)})
            print(f"{name:14s}" + "".join(f"{100 * means[m]:13.2f}" for m in METRICS))
        print(
            f"Defect regions outside the shared region: {lost[0]}; "
            f"too small for the grid: {lost[1]}; anomalous images with "
            f"no defect on the grid: {lost[2]}\n"
        )

    for name in DETECTORS:
        write_csv(OUTPUTS / name / "metrics.csv", rows[name])
        write_csv(OUTPUTS / name / "summary.csv", summary[name])


if __name__ == "__main__":
    main()
