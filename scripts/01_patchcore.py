"""PatchCore scores and anomaly maps (Roth et al., CVPR 2022).

IM224 setting of sample_training.sh in amazon-science/patchcore-inspection
(commit fcaa92f): WideResNet-50-2, layer2 and layer3, 1024-dimensional
patches, 10% coreset, one model per category (seed 0). The image score is the
largest patch score; the map is the patch scores at 224 x 224 after a Gaussian
(sigma 4). Both are saved unscaled; upstream rescales them to 0-1 per
category, which changes no metric. The model code is the unchanged copy in
third_party/patchcore, with FAISS on the CPU. Images: shorter side to 256,
centre crop 224 (VisA images are first resized to 256 x 256).

Writes outputs/detector/patchcore/<dataset>/ (see src/predictions.py).
Run from the repository root: python -m scripts.01_patchcore
"""

import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import wide_resnet50_2

from src.datasets import DATA_ROOTS, ImageList, list_categories, list_samples
from src.metrics import crop_box
from src.predictions import save_category, save_predictions

RESIZE, CROP = 256, 224
BATCH_SIZE, SEED = 2, 0

sys.path.insert(0, "third_party/patchcore")


def build_model(device):
    """PatchCore with the IM224 settings of sample_training.sh."""
    import patchcore.common
    import patchcore.patchcore
    import patchcore.sampler

    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=wide_resnet50_2(weights="IMAGENET1K_V1"),
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, CROP, CROP),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=0.1, device=device
        ),
        anomaly_score_num_nn=1,
        nn_method=patchcore.common.FaissNN(on_gpu=False),
    )
    return model


def make_transform(resize):
    """The image transform of datasets/mvtec.py."""
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def run_category(dataset, category, device):
    """Fit PatchCore on one category, score its test images, save them."""
    import patchcore.utils

    root = DATA_ROOTS[dataset]
    resize = (RESIZE, RESIZE) if dataset == "visa" else RESIZE
    transform = make_transform(resize)
    train = list_samples(dataset, root, category, "train")
    test = list_samples(dataset, root, category, "test")

    # Seeds are set per category, and the images go through a DataLoader,
    # as in run_patchcore.py; both affect the random coreset.
    patchcore.utils.fix_seeds(SEED)
    model = build_model(device)
    with torch.no_grad():
        model.fit(DataLoader(ImageList(train, transform), batch_size=BATCH_SIZE))
    scores, maps, _, _ = model.predict(
        DataLoader(ImageList(test, transform), batch_size=BATCH_SIZE)
    )

    boxes = []
    for sample in test:
        with Image.open(sample.image_path) as image:
            boxes.append(crop_box(image.size, resize, CROP))
    return save_category("patchcore", dataset, category, test, scores, maps, boxes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for dataset, root in DATA_ROOTS.items():
        rows = []
        for category in list_categories(dataset, root):
            rows += run_category(dataset, category, device)
        save_predictions("patchcore", dataset, rows)


if __name__ == "__main__":
    main()
