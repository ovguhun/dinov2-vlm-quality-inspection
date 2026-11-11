"""What the detector scripts save, and how later scripts read it back.

Per dataset, a detector script writes to outputs/detector/<detector>/<dataset>/:

    <category>.npz    image_ids, image_scores, anomaly_maps and map_boxes
                      (the part of the original image each map covers)
    <category>.png    the first normal and the first anomalous test image
                      by name: image, ground truth and anomaly map
    predictions.csv   one row per test image: category, image_id, label,
                      score
"""

import csv
from pathlib import Path

import numpy as np

from src.visualize import save_examples


OUTPUTS = Path("outputs/detector")


def save_category(detector, dataset, category, samples, scores, maps, boxes):
    """Write <category>.npz and <category>.png; return the CSV rows."""
    folder = OUTPUTS / detector / dataset
    folder.mkdir(parents=True, exist_ok=True)
    scores = np.array(scores, np.float32)  # the same values in npz and CSV
    np.savez_compressed(
        folder / f"{category}.npz",
        image_ids=np.array([s.image_id for s in samples]),
        image_scores=scores,
        anomaly_maps=np.stack(maps),
        map_boxes=np.array(boxes),
    )

    labels = [sample.label for sample in samples]
    picks = [labels.index(0), labels.index(1)]  # first normal, anomalous
    save_examples(
        folder / f"{category}.png",
        f"{detector}, {dataset}/{category}",
        [(samples[i], maps[i], boxes[i], scores[i]) for i in picks],
        vmax=max(float(m.max()) for m in maps),
    )
    print(f"{dataset}/{category}: {len(samples)} images -> {folder}")
    return [
        [
            category,
            sample.image_id,
            "anomalous" if sample.label else "normal",
            float(score),
        ]
        for sample, score in zip(samples, scores)
    ]


def save_predictions(detector, dataset, rows):
    """Write predictions.csv for one dataset."""
    path = OUTPUTS / detector / dataset / "predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["category", "image_id", "label", "score"])
        writer.writerows(rows)


def load_category(detector, dataset, category):
    """The arrays saved by save_category, by name."""
    with np.load(OUTPUTS / detector / dataset / f"{category}.npz") as saved:
        return {name: saved[name] for name in saved.files}
