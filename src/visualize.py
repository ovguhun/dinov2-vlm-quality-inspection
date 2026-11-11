"""Example figures, and the overlay image shown to the VLM.

save_examples draws image, ground-truth mask and anomaly map per test image,
cropped to the part the map covers. draw_overlay blends a map in red over the
original image inside that part and outlines it with a dotted line.
"""

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.datasets import read_image, read_mask


PANEL = (256, 256)  # pixels per panel


def save_examples(path, title, examples, vmax):
    """Save one row per (sample, anomaly map, map box, score) example.

    The anomaly maps share one colour scale, from 0 to vmax.
    """
    figure = Figure(figsize=(7.5, 0.5 + 2.6 * len(examples)), layout="constrained")
    axes = figure.subplots(len(examples), 3, squeeze=False)
    for row, (sample, anomaly_map, box, score) in zip(axes, examples):
        image = read_image(sample.image_path).resize(
            PANEL, Image.Resampling.BILINEAR, box=box
        )
        mask = Image.fromarray(read_mask(sample).astype(np.uint8) * 255)
        mask = mask.resize(PANEL, Image.Resampling.NEAREST, box=box)
        heat = Image.fromarray(np.asarray(anomaly_map, np.float32))
        heat = heat.resize(PANEL, Image.Resampling.BILINEAR)

        name = sample.image_id.split("/", 2)[2]  # drop dataset/category
        row[0].imshow(np.asarray(image))
        row[0].set_title(f"{name}\nscore {score:.4g}", fontsize=8)
        row[1].imshow(np.asarray(mask), cmap="gray", vmin=0, vmax=255)
        row[1].set_title("Ground truth", fontsize=8)
        row[2].imshow(np.asarray(image))
        shown = row[2].imshow(
            np.asarray(heat), cmap="jet", alpha=0.5, vmin=0, vmax=vmax
        )
        row[2].set_title("Anomaly map", fontsize=8)
        for axis in row:
            axis.set_axis_off()

    figure.colorbar(shown, ax=axes[:, 2], shrink=0.8)
    figure.suptitle(title)
    figure.savefig(path, dpi=100)


def draw_overlay(rgb, anomaly_map, box, color, max_opacity, outline_gray):
    """The original RGB image with the map blended in, as a uint8 array.

    The map is scaled by its own maximum (negatives count as 0). Inside the
    box, each pixel takes the bilinear map value times max_opacity as the
    colour's opacity; a dotted line outlines the box.
    """
    height, width = rgb.shape[:2]
    rows, cols = anomaly_map.shape
    left, top, right, bottom = box
    positive = np.maximum(anomaly_map, 0)
    maximum = float(positive.max())
    normalized = positive / maximum if maximum > 0 else positive

    # Map coordinates of the pixel centres inside the box.
    x_centres, y_centres = np.arange(width) + 0.5, np.arange(height) + 0.5
    xs = np.flatnonzero((x_centres >= left) & (x_centres < right))
    ys = np.flatnonzero((y_centres >= top) & (y_centres < bottom))
    mx = np.clip((xs + 0.5 - left) * cols / (right - left) - 0.5, 0, cols - 1)
    my = np.clip((ys + 0.5 - top) * rows / (bottom - top) - 0.5, 0, rows - 1)
    x0, y0 = np.floor(mx).astype(int), np.floor(my).astype(int)
    x1, y1 = np.minimum(x0 + 1, cols - 1), np.minimum(y0 + 1, rows - 1)
    dx, dy = mx - x0, (my - y0)[:, None]
    upper = normalized[np.ix_(y0, x0)] * (1 - dx) + normalized[np.ix_(y0, x1)] * dx
    lower = normalized[np.ix_(y1, x0)] * (1 - dx) + normalized[np.ix_(y1, x1)] * dx
    alpha = (max_opacity * (upper * (1 - dy) + lower * dy))[..., None]

    overlay = rgb.copy()
    pixels = rgb[np.ix_(ys, xs)].astype(np.float64)
    overlay[np.ix_(ys, xs)] = np.rint(
        pixels * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)

    # Every other pixel along the edges of the whole pixels in the box.
    x_start, y_start = int(np.ceil(left)), int(np.ceil(top))
    x_stop, y_stop = int(np.floor(right)), int(np.floor(bottom))
    if x_start < x_stop and y_start < y_stop:
        overlay[y_start, x_start:x_stop:2] = outline_gray
        overlay[y_stop - 1, x_start:x_stop:2] = outline_gray
        overlay[y_start:y_stop:2, x_start] = outline_gray
        overlay[y_start:y_stop:2, x_stop - 1] = outline_gray
    return overlay
