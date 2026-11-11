"""Select the report images and set a detector cutoff per category.

Per category, one normal and one anomalous test image are drawn with
random.Random(seed). Images under excluded_images in configs/reporting.json
are skipped: they were used for trial runs while the prompt and settings were
prepared. The cutoff is the 0.98 quantile ("higher" method) of the Dinomaly
scores of the category's other normal test images; a score above it counts as
anomalous. This applies the p-quantile threshold of Bergmann et al. (IJCV
2021, Section 5) to image scores and guarantees no 2% false-positive rate.

Reads the scores saved by 03. Writes outputs/reports/cases.csv.
Run from the repository root: python -m scripts.05_select_report_images
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

from src.datasets import DATA_ROOTS, list_samples
from src.predictions import load_category

CONFIG = Path("configs/reporting.json")


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    detector = config["detector"]
    excluded = set(config["excluded_images"])
    rng = random.Random(config["seed"])

    rows = []
    for dataset, categories in config["categories"].items():
        root = DATA_ROOTS[dataset]
        for category in categories:
            samples = list_samples(dataset, root, category, "test")
            picked = []
            for label in (0, 1):  # normal, then anomalous
                pool = [
                    sample
                    for sample in samples
                    if sample.label == label and sample.image_id not in excluded
                ]
                picked.append(rng.choice(pool))

            saved = load_category(detector["method"], dataset, category)
            scores = dict(
                zip(saved["image_ids"].tolist(), saved["image_scores"].tolist())
            )
            normal = [
                scores[sample.image_id]
                for sample in samples
                if sample.label == 0 and sample not in picked
            ]
            cutoff = float(
                np.quantile(
                    normal,
                    detector["cutoff_quantile"],
                    method=detector["quantile_method"],
                )
            )

            for sample in picked:
                score = scores[sample.image_id]
                rows.append(
                    {
                        "case_id": f"case-{len(rows) + 1:03d}",
                        "dataset": dataset,
                        "category": category,
                        "image_id": sample.image_id,
                        "label": "anomalous" if sample.label else "normal",
                        "score": score,
                        "cutoff": cutoff,
                        "prediction": "anomalous" if score > cutoff else "normal",
                    }
                )

    path = Path(config["files"]["cases"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases to {path}")


if __name__ == "__main__":
    main()
