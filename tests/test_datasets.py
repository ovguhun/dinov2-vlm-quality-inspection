"""Tests for src.datasets.

TinyDatasets builds a few MVTec AD- and VisA-style files in a temporary
folder, so it runs without the real data. OfficialSplits counts the real
datasets in data/ and is skipped when they are not there.

Run from the repository root: python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets import DATA_ROOTS, list_categories, list_samples, read_mask


REPO = Path(__file__).resolve().parents[1]


def save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class TinyDatasets(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1, 2] = 1

        self.mvtec = root / "mvtec"
        for name in ("train/good/000.png", "test/good/000.png",
                     "test/good/._000.png", "test/crack/001.png",
                     "test/crack/000.png"):
            save(self.mvtec / "bottle" / name, image)
        for name in ("000_mask.png", "001_mask.png"):
            save(self.mvtec / "bottle/ground_truth/crack" / name, mask * 255)

        self.visa = root / "visa"
        anomaly = "candle/Data/Images/Anomaly/000.JPG"
        anomaly_mask = "candle/Data/Masks/Anomaly/000.png"
        for name in ("Normal/0001.JPG", "Normal/0002.JPG", "Anomaly/000.JPG"):
            save(self.visa / "candle/Data/Images" / name, image)
        save(self.visa / anomaly_mask, mask * 5)
        rows = [
            "object,split,label,image,mask",
            "candle,train,normal,candle/Data/Images/Normal/0002.JPG,",
            "candle,test,normal,candle/Data/Images/Normal/0001.JPG,",
            f"candle,test,anomaly,{anomaly},{anomaly_mask}",
        ]
        (self.visa / "split_csv").mkdir()
        (self.visa / "split_csv/1cls.csv").write_text("\n".join(rows) + "\n")

    def test_categories(self):
        self.assertEqual(list_categories("mvtec_ad", self.mvtec), ["bottle"])
        self.assertEqual(list_categories("visa", self.visa), ["candle"])

    def test_mvtec_test_split(self):
        samples = list_samples("mvtec_ad", self.mvtec, "bottle", "test")
        self.assertEqual([s.image_id for s in samples], [
            "mvtec_ad/bottle/test/crack/000.png",
            "mvtec_ad/bottle/test/crack/001.png",
            "mvtec_ad/bottle/test/good/000.png",
        ])
        self.assertEqual([s.label for s in samples], [1, 1, 0])
        self.assertEqual(samples[1].mask_path.name, "001_mask.png")
        self.assertIsNone(samples[2].mask_path)

    def test_visa_splits_follow_the_split_file(self):
        # Train and test normals share one folder; only the split file
        # tells them apart.
        train = list_samples("visa", self.visa, "candle", "train")
        test = list_samples("visa", self.visa, "candle", "test")
        self.assertEqual([s.image_id for s in train],
                         ["visa/candle/Data/Images/Normal/0002.JPG"])
        self.assertEqual([s.image_id for s in test], [
            "visa/candle/Data/Images/Anomaly/000.JPG",
            "visa/candle/Data/Images/Normal/0001.JPG",
        ])
        self.assertEqual([s.label for s in test], [1, 0])

    def test_masks_mark_nonzero_pixels(self):
        # MVTec AD masks store 255; VisA masks store a defect class (1-8).
        mvtec = list_samples("mvtec_ad", self.mvtec, "bottle", "test")
        visa = list_samples("visa", self.visa, "candle", "test")
        for anomalous in (mvtec[0], visa[0]):
            self.assertEqual(np.argwhere(read_mask(anomalous)).tolist(),
                             [[1, 2]])
        for normal in (mvtec[2], visa[1]):
            mask = read_mask(normal)
            self.assertEqual(mask.shape, (4, 6))
            self.assertFalse(mask.any())

    def test_unexpected_visa_rows_stop(self):
        split_file = self.visa / "split_csv/1cls.csv"
        text = split_file.read_text()
        bad_rows = {
            "anomalous train image": "candle,train,anomaly,x.JPG,x.png",
            "unknown split": "candle,val,normal,x.JPG,",
            "anomaly without a mask": "candle,test,anomaly,x.JPG,",
            "unknown label": "candle,test,good,x.JPG,",
        }
        for problem, row in bad_rows.items():
            split_file.write_text(text + row + "\n")
            with self.assertRaises(ValueError, msg=problem):
                list_samples("visa", self.visa, "candle", "test")


class OfficialSplits(unittest.TestCase):
    """Image counts: (train, normal test, anomalous test)."""

    def check(self, dataset, expected):
        root = REPO / DATA_ROOTS[dataset]
        if not root.is_dir():
            self.skipTest(f"{root} not found")
        counts = [0, 0, 0]
        for category in list_categories(dataset, root):
            counts[0] += len(list_samples(dataset, root, category, "train"))
            for sample in list_samples(dataset, root, category, "test"):
                counts[1 + sample.label] += 1
        self.assertEqual(tuple(counts), expected)

    def test_mvtec_ad(self):
        # Bergmann et al. (CVPR 2019), Table 1.
        self.check("mvtec_ad", (3629, 467, 1258))

    def test_visa(self):
        # 9,621 normal and 1,200 anomalous images (spot-diff README);
        # the official 1cls split keeps 962 of the normal images for testing.
        self.check("visa", (8659, 962, 1200))


if __name__ == "__main__":
    unittest.main()
