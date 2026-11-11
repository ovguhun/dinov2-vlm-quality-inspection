"""Tests for predictions: what the detector scripts save.

Run from the repository root: python -m unittest discover tests
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src import predictions
from src.datasets import Sample


class SaveAndLoad(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.addCleanup(setattr, predictions, "OUTPUTS", predictions.OUTPUTS)
        predictions.OUTPUTS = root / "outputs"

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "a.png")
        Image.fromarray(mask).save(root / "a_mask.png")
        self.samples = [
            Sample("mvtec_ad/toy/test/crack/000.png", root / "a.png",
                   root / "a_mask.png", 1),
            Sample("mvtec_ad/toy/test/good/000.png", root / "a.png", None, 0),
        ]

    def test_saved_files_read_back(self):
        maps = [np.full((4, 4), 0.9, np.float32), np.zeros((4, 4), np.float32)]
        boxes = [(1.0, 1.0, 7.0, 7.0), (0.0, 0.0, 8.0, 8.0)]
        rows = predictions.save_category("toy", "mvtec_ad", "toy",
                                         self.samples, [0.9, 0.1], maps,
                                         boxes)
        predictions.save_predictions("toy", "mvtec_ad", rows)

        saved = predictions.load_category("toy", "mvtec_ad", "toy")
        self.assertEqual(saved["image_ids"].tolist(),
                         [sample.image_id for sample in self.samples])
        np.testing.assert_array_equal(saved["image_scores"],
                                      np.float32([0.9, 0.1]))
        np.testing.assert_array_equal(saved["anomaly_maps"], np.stack(maps))
        np.testing.assert_array_equal(saved["map_boxes"], np.array(boxes))

        folder = predictions.OUTPUTS / "toy" / "mvtec_ad"
        self.assertTrue((folder / "toy.png").is_file())
        with open(folder / "predictions.csv", encoding="utf-8") as file:
            table = list(csv.reader(file))
        self.assertEqual(table, [
            ["category", "image_id", "label", "score"],
            ["toy", self.samples[0].image_id, "anomalous",
             str(float(np.float32(0.9)))],
            ["toy", self.samples[1].image_id, "normal",
             str(float(np.float32(0.1)))],
        ])


if __name__ == "__main__":
    unittest.main()
