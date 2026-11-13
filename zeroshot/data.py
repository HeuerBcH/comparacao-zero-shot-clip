from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as TF

from . import config


@dataclass(frozen=True)
class ShiftedDataset:
    images: np.ndarray
    labels: np.ndarray
    shifts: np.ndarray
    classnames: list[str]


def load_shifted_cifar100() -> ShiftedDataset:
    dataset = CIFAR100(root=str(config.DATA_DIR), train=False, download=True)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(config.SHIFT_ASSIGNMENT_SEED)
    shifts = np.empty_like(labels)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        balanced = np.resize(np.arange(len(config.SHIFTS)), len(idx))
        shifts[idx] = rng.permutation(balanced)
    classnames = [name.replace("_", " ") for name in dataset.classes]
    return ShiftedDataset(dataset.data, labels, shifts, classnames)


def stratified_sample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in np.unique(labels)]
    return np.sort(np.concatenate(picks))


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def _sketch(x: torch.Tensor) -> torch.Tensor:
    gray = TF.rgb_to_grayscale(x).unsqueeze(0)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx)
    gy = F.conv2d(padded, kx.transpose(2, 3))
    magnitude = torch.sqrt(gx**2 + gy**2).squeeze(0)
    magnitude = magnitude / magnitude.max().clamp_min(1e-8)
    return (1.0 - magnitude).expand(3, -1, -1)


def apply_shift(x: torch.Tensor, shift: str, image_index: int) -> torch.Tensor:
    if shift == "ruido_gaussiano":
        generator = torch.Generator().manual_seed(image_index)
        return (x + config.NOISE_SIGMA * torch.randn(x.shape, generator=generator)).clamp(0.0, 1.0)
    if shift == "desfoque_gaussiano":
        return TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[config.BLUR_SIGMA, config.BLUR_SIGMA])
    if shift == "cinza_baixo_contraste":
        gray = _grayscale(x)
        return ((gray - gray.mean()) * config.CONTRAST_FACTOR + gray.mean()).clamp(0.0, 1.0)
    if shift == "esboco":
        return _sketch(x)
    raise ValueError(f"Shift desconhecido: {shift}")


def build_shifted_batch(data: ShiftedDataset, indices: np.ndarray) -> torch.Tensor:
    raw = torch.from_numpy(data.images[indices]).permute(0, 3, 1, 2).float() / 255.0
    return torch.stack([apply_shift(img, config.SHIFTS[data.shifts[i]], int(i)) for img, i in zip(raw, indices)])


def to_clip_input(images: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(images, size=config.IMAGE_SIZE, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    mean = torch.tensor(config.CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.CLIP_STD).view(1, 3, 1, 1)
    return (resized - mean) / std
