"""EfficientAD-S scores and anomaly maps (Batzner et al., WACV 2024).

Unofficial implementation nelson1425/EfficientAD (commit fcab514, Apache
License 2.0): the networks and imported functions are unchanged copies in
third_party/efficientad; train() and score() adapt its main() and test(). Per
category, a student and an autoencoder are trained for 70,000 steps on 90% of
the normal training images; the other 10% set the map-normalisation
quantiles. Unlike main(): penalty images come from Imagenette instead of
ImageNet, images load without worker processes, cuDNN uses fixed algorithms,
and there are no intermediate evaluations or checkpoints.

Needs weights/efficientad/teacher_small.pth and data/imagenette2/train.
Writes outputs/detector/efficientad-s/<dataset>/ (see src/predictions.py);
maps are saved at 64 x 64, padded but not resized, covering the whole image.
Run from the repository root: python -m scripts.02_efficientad
"""

import itertools
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.datasets import (
    DATA_ROOTS,
    ImageList,
    list_categories,
    list_samples,
    read_image,
)
from src.predictions import save_category, save_predictions

TEACHER = "weights/efficientad/teacher_small.pth"
PENALTY_IMAGES = "data/imagenette2/train"
STEPS, SEED = 70_000, 42  # efficientad.py defaults

# The penalty transform of main() in efficientad.py.
PENALTY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomGrayscale(0.3),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sys.path.insert(0, "third_party/efficientad")


def train(samples, device):
    """Train a student and an autoencoder, as main() in efficientad.py.

    Returns the arguments of predict(): the three networks, the teacher's
    feature mean and std, and the four map-normalisation quantiles.
    """
    from common import (
        ImageFolderWithoutTarget,
        InfiniteDataloader,
        get_autoencoder,
        get_pdn_small,
    )
    from efficientad import (
        map_normalization,
        out_channels,
        teacher_normalization,
        train_transform,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    normals = ImageList(samples, train_transform)
    train_size = int(0.9 * len(normals))
    train_set, validation_set = random_split(
        normals,
        [train_size, len(normals) - train_size],
        torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=1)
    penalty_loader = DataLoader(
        ImageFolderWithoutTarget(PENALTY_IMAGES, transform=PENALTY_TRANSFORM),
        batch_size=1,
        shuffle=True,
    )

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)
    teacher.load_state_dict(torch.load(TEACHER, map_location="cpu"))
    autoencoder = get_autoencoder(out_channels)
    teacher.eval()
    student.train()
    autoencoder.train()
    teacher.to(device)
    student.to(device)
    autoencoder.to(device)

    mean, std = teacher_normalization(teacher, train_loader)
    optimizer = torch.optim.Adam(
        itertools.chain(student.parameters(), autoencoder.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * STEPS), gamma=0.1
    )

    steps = tqdm(range(STEPS))
    for step, (image_st, image_ae), image_penalty in zip(
        steps, InfiniteDataloader(train_loader), InfiniteDataloader(penalty_loader)
    ):
        image_st, image_ae = image_st.to(device), image_ae.to(device)
        image_penalty = image_penalty.to(device)

        # Student vs teacher, on the hardest 0.1% of the distances,
        # plus the penalty on the unrelated image.
        with torch.no_grad():
            teacher_output_st = (teacher(image_st) - mean) / std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])
        student_output_penalty = student(image_penalty)[:, :out_channels]
        loss_st = loss_hard + torch.mean(student_output_penalty**2)

        # Autoencoder vs teacher, and student vs autoencoder.
        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = (teacher(image_ae) - mean) / std
        student_output_ae = student(image_ae)[:, out_channels:]
        loss_ae = torch.mean((teacher_output_ae - ae_output) ** 2)
        loss_stae = torch.mean((ae_output - student_output_ae) ** 2)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0:
            steps.set_description(f"Current loss: {loss_total.item():.4f}")

    student.eval()
    autoencoder.eval()
    quantiles = map_normalization(
        validation_loader, teacher, student, autoencoder, mean, std
    )
    return teacher, student, autoencoder, mean, std, *quantiles


@torch.no_grad()
def score(model, samples, device):
    """Image scores, 64 x 64 anomaly maps and map boxes of some samples."""
    from efficientad import default_transform, predict

    scores, maps, boxes = [], [], []
    for sample in samples:
        image = read_image(sample.image_path)
        anomaly_map, _, _ = predict(
            default_transform(image).unsqueeze(0).to(device), *model
        )
        # As test() in efficientad.py: pad by 4, resize to the image and
        # take the largest value.
        anomaly_map = F.pad(anomaly_map, (4, 4, 4, 4))
        full_size = F.interpolate(
            anomaly_map, (image.height, image.width), mode="bilinear"
        )
        scores.append(full_size.max().item())
        maps.append(anomaly_map[0, 0].cpu().numpy())
        boxes.append((0.0, 0.0, float(image.width), float(image.height)))
    return scores, maps, boxes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Fixed cuDNN algorithms; two GPU runs can still differ in the last digits.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for dataset, root in DATA_ROOTS.items():
        rows = []
        for category in list_categories(dataset, root):
            model = train(list_samples(dataset, root, category, "train"), device)
            test = list_samples(dataset, root, category, "test")
            scores, maps, boxes = score(model, test, device)
            rows += save_category(
                "efficientad-s", dataset, category, test, scores, maps, boxes
            )
        save_predictions("efficientad-s", dataset, rows)


if __name__ == "__main__":
    main()
