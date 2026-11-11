"""Tests for visualize.draw_overlay, on a tiny image.

Run from the repository root: python -m unittest discover tests
"""

import unittest

import numpy as np

from src.visualize import draw_overlay


class Overlay(unittest.TestCase):
    def test_overlay(self):
        rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
        heat = np.zeros((4, 4), dtype=np.float32)
        heat[1:3, 1:3] = 1.0  # the map's maximum, in its middle
        heat[0, 0] = -5.0     # negative values count as 0
        out = draw_overlay(rgb, heat, (2.0, 2.0, 14.0, 14.0), (255, 0, 0),
                           0.6, 128)

        # Nothing changes outside the box.
        for edge in (out[:2], out[14:], out[:, :2], out[:, 14:]):
            self.assertTrue((edge == 100).all())
        # At the maximum the colour has its full 0.6 opacity:
        # 100 * 0.4 + 255 * 0.6 = 193 for red and 100 * 0.4 = 40 otherwise.
        np.testing.assert_array_equal(out[8, 8], [193, 40, 40])
        # The dotted outline starts at each corner of the box and skips
        # every other pixel; next to it the negative map leaves no tint.
        np.testing.assert_array_equal(out[2, 2], [128, 128, 128])
        np.testing.assert_array_equal(out[13, 2], [128, 128, 128])
        np.testing.assert_array_equal(out[2, 3], [100, 100, 100])

    def test_overlay_follows_the_map_and_the_box(self):
        rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
        heat = np.zeros((4, 4), dtype=np.float32)
        heat[0, 3] = 1.0  # top-right cell only
        out = draw_overlay(rgb, heat, (2.0, 2.0, 14.0, 14.0), (255, 0, 0),
                           0.6, 128)
        np.testing.assert_array_equal(out[3, 12], [193, 40, 40])
        np.testing.assert_array_equal(out[12, 3], [100, 100, 100])
        # Pixel (3, 10) reads the map a third of the way from cell 2 to cell
        # 3: opacity 0.2, so 100 * 0.8 + 255 * 0.2 = 131.
        np.testing.assert_array_equal(out[3, 10], [131, 80, 80])

        # Pixels whose centre is on the left edge are in the box, those on
        # the right edge are not.
        out = draw_overlay(rgb, np.ones((4, 4), dtype=np.float32),
                           (2.5, 2.5, 13.5, 13.5), (255, 0, 0), 0.6, 128)
        np.testing.assert_array_equal(out[8, 2], [193, 40, 40])
        np.testing.assert_array_equal(out[8, 13], [100, 100, 100])


if __name__ == "__main__":
    unittest.main()
