import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F


def _load_image(path: Path) -> torch.Tensor:
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3:
        img = img[:3]
    return img


def _extract_patches(
    img: torch.Tensor,
    patch_size: int,
    stride: int,
    max_patches: int,
    seed: int,
) -> torch.Tensor:
    c, h, w = img.shape
    if h < patch_size or w < patch_size:
        img = F.resize(img, [patch_size, patch_size])
        c, h, w = img.shape

    patches = (
        img.unfold(1, patch_size, stride)
        .unfold(2, patch_size, stride)
        .permute(1, 2, 0, 3, 4)
        .reshape(-1, c, patch_size, patch_size)
    )

    total = patches.shape[0]
    g = torch.Generator().manual_seed(seed)
    if total >= max_patches:
        idx = torch.randperm(total, generator=g)[:max_patches]
        patches = patches[idx]
    else:
        if total == 0:
            patches = torch.zeros((1, c, patch_size, patch_size), dtype=img.dtype)
            total = 1
        pad_idx = torch.randint(0, total, (max_patches - total,), generator=g)
        patches = torch.cat([patches, patches[pad_idx]], dim=0)

    return patches


class PatchMILDatasetSingle(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size: int = 224,
        stride: int = 112,
        max_patches_per_image: int = 32,
        seed: int = 42,
        class_to_idx=None,
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches_per_image = max_patches_per_image
        self.seed = seed

        if class_to_idx is None:
            self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.classes = [idx_to_class[i] for i in range(len(idx_to_class))]

        self.samples = []
        for cls in self.classes:
            class_dir = self.root_dir / cls
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.is_file():
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        img = _load_image(img_path)
        patches = _extract_patches(
            img,
            patch_size=self.patch_size,
            stride=self.stride,
            max_patches=self.max_patches_per_image,
            seed=self.seed + idx,
        )
        meta = {"image_path": str(img_path)}
        return patches, label_idx, meta


class MultiImageWindowDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        patch_size: int = 224,
        stride: int = 112,
        max_patches_per_image: int = 32,
        seed: int = 42,
        class_names=None,
    ):
        self.json_path = Path(json_path)
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches_per_image = max_patches_per_image
        self.seed = seed

        self.records = json.loads(self.json_path.read_text())
        if class_names is None:
            class_names = sorted(list({r["label"] for r in self.records}))
        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        paths = [Path(p) for p in rec["paths"]]
        label_idx = self.class_to_idx[rec["label"]]

        patches_per_image = []
        for j, p in enumerate(paths):
            img = _load_image(p)
            patches = _extract_patches(
                img,
                patch_size=self.patch_size,
                stride=self.stride,
                max_patches=self.max_patches_per_image,
                seed=self.seed + idx * 1000 + j,
            )
            patches_per_image.append(patches)

        patches_per_image = torch.stack(patches_per_image, dim=0)
        meta = {"paths": [str(p) for p in paths]}
        return patches_per_image, label_idx, meta
