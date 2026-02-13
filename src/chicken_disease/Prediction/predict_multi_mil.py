import json
import sys
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms import functional as F

from chicken_disease.utils.config import read_config
from chicken_disease.utils.params import read_params
from chicken_disease.models.attention_mil import MultiImageMILClassifier


def _load_image(path: Path) -> torch.Tensor:
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3:
        img = img[:3]
    return img


def _extract_patches(img, patch_size, stride, max_patches, seed):
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
            patches = torch.zeros((1, c, patch_size, patch_size))
            total = 1
        pad_idx = torch.randint(0, total, (max_patches - total,), generator=g)
        patches = torch.cat([patches, patches[pad_idx]], dim=0)

    return patches


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_multi_mil.py img1.jpg img2.jpg ...")
        sys.exit(1)

    cfg = read_config()
    p = read_params()
    mil_cfg = p["patch_mil"]

    model_dir = Path(cfg["artifacts"]["model_dir"]) / "patch_mil_multi"
    model_path = model_dir / "model.pt"
    class_map_path = model_dir / "class_map.json"

    class_map = json.loads(class_map_path.read_text())
    class_names = [class_map[str(i)] for i in range(len(class_map))]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiImageMILClassifier(
        num_classes=len(class_names),
        model_name=mil_cfg["model_name"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    paths = [Path(p) for p in sys.argv[1:]]
    patches_per_image = []
    for i, path in enumerate(paths):
        img = _load_image(path)
        patches = _extract_patches(
            img,
            patch_size=mil_cfg["patch_size"],
            stride=mil_cfg["stride"],
            max_patches=mil_cfg["max_patches_per_image"],
            seed=mil_cfg["seed"] + i,
        )
        patches_per_image.append(patches)

    images_as_patches = torch.stack(patches_per_image, dim=0).to(device)

    k = mil_cfg.get("mc_dropout_passes", 20)
    probs_all = []
    image_weights_all = []

    model.train()
    with torch.no_grad():
        for _ in range(k):
            logits, image_weights, _ = model(images_as_patches)
            probs = torch.softmax(logits, dim=0)
            probs_all.append(probs)
            image_weights_all.append(image_weights)

    probs_all = torch.stack(probs_all, dim=0)
    mean_probs = probs_all.mean(dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8)).item()

    image_weights = torch.stack(image_weights_all, dim=0).mean(dim=0).cpu().tolist()

    topk = torch.topk(mean_probs, k=min(2, len(class_names)))
    print("Top predictions:")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {class_names[idx]}: {score:.4f}")
    print(f"Uncertainty (entropy): {entropy:.4f}")
    print("Per-image attention weights:")
    for p, w in zip(paths, image_weights):
        print(f"  {p}: {w:.4f}")


if __name__ == "__main__":
    main()
