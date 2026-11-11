"""Tests for metrics, on small hand-made examples.

Run from the repository root: python -m unittest discover tests
"""

import unittest

import numpy as np

from src.metrics import (aupro, category_metrics, common_box, crop_box, f1_max,
                     lost_regions, sample_map, sample_mask)


BOX = (64.0, 64.0, 960.0, 960.0)  # 448 -> crop 392 on a 1024 x 1024 image
VISA_BOX = (93.75, 62.5, 1406.25, 937.5)  # 256 -> crop 224 on 1500 x 1000


class EvaluationRegion(unittest.TestCase):
    def test_crop_box(self):
        # A centre crop of 392 from 448, or of 224 from 256, keeps 7/8 of
        # each side and drops 1/16 on every edge.
        self.assertEqual(crop_box((1024, 1024), (448, 448), 392), BOX)
        self.assertEqual(crop_box((1500, 1000), (256, 256), 224), VISA_BOX)
        # An int resize sets the shorter side: 900 x 600 becomes 384 x 256.
        self.assertEqual(crop_box((900, 600), 256, 224),
                         (187.5, 37.5, 712.5, 562.5))
        # A pair is (height, width), as in torchvision.
        self.assertEqual(crop_box((1000, 500), (250, 500), 200),
                         (300.0, 50.0, 700.0, 450.0))
        # Offsets of 2.5 and 1.5 pixels both round to 2, as in torchvision.
        for size in ((229, 227), (227, 229)):
            self.assertEqual(crop_box(size, size[::-1], 224),
                             (2.0, 2.0, 226.0, 226.0))

    def test_common_box_is_the_overlap(self):
        boxes = [BOX, (0.0, 0.0, 1024.0, 1024.0), (70.0, 0.0, 1000.0, 900.0)]
        self.assertEqual(common_box(boxes), (70.0, 64.0, 960.0, 900.0))

    def test_maps_are_read_at_grid_cell_centres(self):
        # Full-image maps whose values are the x and the y position of each
        # pixel centre, read on the non-square VisA box.
        rows, cols = np.mgrid[0:1000, 0:1500].astype(np.float32) + 0.5
        whole_image = (0.0, 0.0, 1500.0, 1000.0)
        centres = np.arange(256) + 0.5
        x = 93.75 + centres * 1312.5 / 256
        y = 62.5 + centres * 875 / 256
        np.testing.assert_allclose(sample_map(cols, whole_image, VISA_BOX),
                                   np.tile(x, (256, 1)), atol=1e-3)
        np.testing.assert_allclose(sample_map(rows, whole_image, VISA_BOX),
                                   np.tile(y[:, None], (1, 256)), atol=1e-3)

        # A 256 x 256 map that covers exactly the box comes back unchanged.
        values = np.random.default_rng(0).random((256, 256), dtype=np.float32)
        np.testing.assert_allclose(sample_map(values, BOX, BOX), values,
                                   atol=1e-6)
        # Cells at the edge repeat the map's edge instead of fading to 0.
        ones = np.ones((224, 224), dtype=np.float32)
        np.testing.assert_array_equal(sample_map(ones, VISA_BOX, VISA_BOX), 1)

    def test_regions_lost_by_cropping_and_sampling(self):
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[500:520, 500:520] = True    # inside the box
        mask[0:10, 0:10] = True          # in the cropped border
        mask[1000, 0] = mask[1001, 1] = True  # cropped too, 8-connected
        mask[68, 68] = True              # inside, but between grid samples
        grid = sample_mask(mask, BOX)
        self.assertEqual(grid.shape, (256, 256))
        self.assertTrue(grid[126:129, 126:129].all())
        self.assertEqual(lost_regions(mask, BOX), (2, 1))

    def test_mask_on_a_non_square_box(self):
        mask = np.zeros((1000, 1500), dtype=bool)
        mask[:62] = True                # above the box: cropped
        mask[70:80, 700:720] = True     # inside, near the top edge
        grid = sample_mask(mask, VISA_BOX)
        self.assertFalse(grid[0].any())
        self.assertTrue(grid[3:5, 118:121].all())
        self.assertEqual(lost_regions(mask, VISA_BOX), (1, 0))

    def test_a_grid_cell_takes_the_pixel_under_its_centre(self):
        # Cell (0, 0) of BOX has its centre at 64 + 3.5 / 2 = 65.75.
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[65, 65] = True
        self.assertEqual(np.argwhere(sample_mask(mask, BOX)).tolist(),
                         [[0, 0]])
        self.assertEqual(lost_regions(mask, BOX), (0, 0))

    def test_a_pixel_on_the_box_edge_is_inside(self):
        # Row 62 spans 62 to 63 and so overlaps the box, which starts at
        # 62.5; the first grid row reads row 64, so the region is missed.
        mask = np.zeros((1000, 1500), dtype=bool)
        mask[62, 700] = True
        self.assertEqual(lost_regions(mask, VISA_BOX), (0, 1))


class Metrics(unittest.TestCase):
    def test_f1_max(self):
        # Thresholds 0.8, 0.4, 0.35, 0.1 give F1 2/3, 1/2, 4/5, 2/3.
        self.assertAlmostEqual(
            f1_max(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])),
            0.8)
        # Tied scores are always predicted together.
        self.assertAlmostEqual(f1_max(np.array([0, 1]), np.array([0.5, 0.5])),
                               2 / 3)

    def test_aupro_weights_each_region_equally(self):
        # Region A is one pixel scoring 0.9. Region B is three diagonally
        # touching pixels, one 8-connected region, scoring 0.5. Of the ten
        # background pixels one scores 0.7 (FPR 0.1), the rest 0.1.
        masks = np.zeros((1, 2, 7), dtype=bool)
        maps = np.full((1, 2, 7), 0.1)
        masks[0, 0, 0], maps[0, 0, 0] = True, 0.9
        for row, col in ((0, 2), (1, 3), (0, 4)):
            masks[0, row, col], maps[0, row, col] = True, 0.5
        maps[0, 1, 6] = 0.7
        # PRO is 0.5 from FPR 0 to 0.1, then 1.0 up to the limit 0.3: the
        # area 0.05 + 0.2 = 0.25 is normalised to 0.25 / 0.3 = 5/6.
        self.assertAlmostEqual(aupro(masks, maps), 5 / 6)
        self.assertAlmostEqual(aupro(masks.astype(np.uint8), maps), 5 / 6)

    def test_aupro_keeps_the_regions_of_each_image_apart(self):
        # Image 0 holds regions A (0.9) and C (0.5), one pixel each; image 1
        # holds region B, three touching pixels at A's position (0.5). Of 23
        # background pixels one scores 0.7 (FPR 1/23), the rest 0.1. PRO is
        # 1/3 (A of the three regions) up to FPR 1/23, then 1.
        masks = np.zeros((2, 2, 7), dtype=bool)
        maps = np.full((2, 2, 7), 0.1)
        masks[0, 0, 0], maps[0, 0, 0] = True, 0.9
        masks[0, 0, 4], maps[0, 0, 4] = True, 0.5
        for row, col in ((0, 0), (1, 1), (0, 2)):
            masks[1, row, col], maps[1, row, col] = True, 0.5
        maps[0, 1, 6] = 0.7
        self.assertAlmostEqual(aupro(masks, maps),
                               (1 / 3 / 23 + 0.3 - 1 / 23) / 0.3)

    def test_aupro_interpolates_at_the_limit(self):
        # Ten background pixels; the one defect pixel ties with two of them
        # at 0.5. PRO climbs from 0 at FPR 0.2 to 1 at FPR 0.4, so it is 0.5
        # at the limit 0.3, and the area is 0.1 * 0.25 = 0.025.
        maps = np.array([[[0.9, 0.8, 0.5, 0.5, 0.5] + [0.1] * 6]])
        masks = np.zeros(maps.shape, dtype=bool)
        masks[0, 0, 2] = True
        self.assertAlmostEqual(aupro(masks, maps), 0.025 / 0.3)

    def test_category_metrics(self):
        # Two images on a 2 x 2 grid; the image scores separate them. The one
        # defect pixel (0.5) outranks 6 of the 7 background pixels and ranks
        # second after a 0.6, so pixel AP is 1/2 and F1-max 2/3 (tp 1, fp 1).
        # PRO reaches 1 at FPR 1/7, so AU-PRO is (0.3 - 1/7) / 0.3 = 11/21.
        labels, scores = np.array([0, 1]), np.array([0.2, 0.9])
        masks = np.zeros((2, 2, 2), dtype=bool)
        masks[1, 0, 0] = True
        maps = np.array([[[0.1, 0.2], [0.3, 0.6]],
                         [[0.5, 0.1], [0.1, 0.1]]])
        expected = {"image_auroc": 1, "image_ap": 1, "image_f1_max": 1,
                    "pixel_auroc": 6 / 7, "pixel_ap": 1 / 2,
                    "pixel_f1_max": 2 / 3, "aupro": 11 / 21}
        metrics = category_metrics(labels, scores, masks, maps)
        self.assertEqual(metrics.keys(), expected.keys())
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value, msg=name)


if __name__ == "__main__":
    unittest.main()
