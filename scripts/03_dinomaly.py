"""Dinomaly scores and anomaly maps from the released checkpoints.

Dinomaly (Guo et al., CVPR 2025), guojiajeremy/Dinomaly (commit 1f252be): the
unified Dinomaly-B models trained for 20,000 iterations, used without
training; the model code is the unchanged copy in third_party/dinomaly. The
steps follow dinomaly_mvtec_uni.py and evaluation_batch in utils.py: resize
to 448, centre crop 392, cosine-distance map resized to 256 and smoothed with
a 5 x 5 Gaussian (sigma 4); the score is the mean of the map's largest 1%.
Images are scored one at a time (upstream: batches of 16).

Needs weights/dinomaly/mvtec_vitb_it20k.pth and visa_vitb_it20k.pth.
Writes outputs/detector/dinomaly/<dataset>/ (see src/predictions.py).
Run from the repository root: python -m scripts.03_dinomaly
"""

import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from src.datasets import DATA_ROOTS, list_categories, list_samples, read_image
from src.metrics import crop_box
from src.predictions import save_category, save_predictions

CHECKPOINTS = {
    "mvtec_ad": Path("weights/dinomaly/mvtec_vitb_it20k.pth"),
    "visa": Path("weights/dinomaly/visa_vitb_it20k.pth"),
}
RESIZE, CROP = 448, 392  # image_size, crop_size in dinomaly_*_uni.py
MAP_SIZE = 256  # evaluation_batch(..., resize_mask=256)
TOP_RATIO = 0.01  # evaluation_batch(..., max_ratio=0.01)

# dataset.py get_data_transforms(448, 392), ImageNet mean and std.
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.CenterCrop(CROP),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sys.path.insert(0, "third_party/dinomaly")
os.environ["XFORMERS_DISABLED"] = "1"  # plain PyTorch attention in DINOv2


def build_model(checkpoint):
    """Dinomaly-B as built in dinomaly_mvtec_uni.py, with released weights."""
    from dinov2.models.vision_transformer import vit_base
    from models.uad import ViTill
    from models.vision_transformer import Block, LinearAttention2, bMlp

    # Encoder settings for "dinov2reg_vit_base_14" in models/vit_encoder.py.
    encoder = vit_base(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
    decoder = nn.ModuleList(
        [
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(8)
        ]
    )
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        mask_neighbor_size=0,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


# gaussian_kernel() and cosine_map() are adapted from Dinomaly utils.py
# (get_gaussian_kernel, cal_anomaly_maps), commit 1f252be, Copyright 2025
# Jia Guo, Apache License 2.0. The arithmetic is unchanged; the kernel is
# returned as a convolution weight instead of a Conv2d layer.
def gaussian_kernel(size=5, sigma=4):
    """Normalised size x size Gaussian, shaped as a convolution weight."""
    x_grid = torch.arange(size).repeat(size).view(size, size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (size - 1) / 2.0
    variance = sigma**2.0
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )
    kernel = kernel / torch.sum(kernel)
    return kernel.view(1, 1, size, size)


def cosine_map(encoder_features, decoder_features, size):
    """1 - cosine similarity per feature group, resized, then averaged."""
    maps = []
    for fs, ft in zip(encoder_features, decoder_features):
        a_map = torch.unsqueeze(1 - F.cosine_similarity(fs, ft), dim=1)
        maps.append(
            F.interpolate(a_map, size=(size, size), mode="bilinear", align_corners=True)
        )
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)


@torch.no_grad()
def predict(model, kernel, image, device):
    """Anomaly map (256 x 256) and image score of one RGB image."""
    encoder_features, decoder_features = model(TRANSFORM(image).unsqueeze(0).to(device))
    anomaly_map = cosine_map(encoder_features, decoder_features, CROP)
    anomaly_map = F.interpolate(
        anomaly_map, size=MAP_SIZE, mode="bilinear", align_corners=False
    )
    anomaly_map = F.conv2d(anomaly_map, kernel, padding=kernel.shape[-1] // 2)
    values = torch.sort(anomaly_map.flatten(1), dim=1, descending=True)[0]
    score = values[:, : int(values.shape[1] * TOP_RATIO)].mean(dim=1)
    return anomaly_map[0, 0].cpu().numpy(), score.item()


def run_category(model, kernel, dataset, category, device):
    """Score the test images of one category and save them."""
    test = list_samples(dataset, DATA_ROOTS[dataset], category, "test")
    scores, maps, boxes = [], [], []
    for sample in test:
        image = read_image(sample.image_path)
        anomaly_map, score = predict(model, kernel, image, device)
        scores.append(score)
        maps.append(anomaly_map)
        boxes.append(crop_box(image.size, (RESIZE, RESIZE), CROP))
    return save_category("dinomaly", dataset, category, test, scores, maps, boxes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # As setup_seed() in dinomaly_mvtec_uni.py: fixed cuDNN algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kernel = gaussian_kernel().to(device)

    for dataset, checkpoint in CHECKPOINTS.items():
        model = build_model(checkpoint).to(device)
        rows = []
        for category in list_categories(dataset, DATA_ROOTS[dataset]):
            rows += run_category(model, kernel, dataset, category, device)
        save_predictions("dinomaly", dataset, rows)


if __name__ == "__main__":
    main()
