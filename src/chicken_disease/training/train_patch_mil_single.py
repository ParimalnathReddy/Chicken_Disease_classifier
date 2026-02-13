import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from chicken_disease.utils.config import read_config
from chicken_disease.utils.params import read_params
from chicken_disease.utils.seed import seed_everything
from chicken_disease.datasets.patch_mil_dataset import PatchMILDatasetSingle
from chicken_disease.models.attention_mil import PatchMILClassifier
from chicken_disease.losses.metrics import compute_macro_f1, compute_per_class_f1_and_recall


def _make_dataloader(ds, cfg, train: bool):
    tfm = None
    if train:
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)], p=0.7
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 5.0))], p=0.4
            ),
        ])

    def _collate(batch):
        patches, labels, meta = zip(*batch)
        patches = torch.stack(patches, dim=0)
        if tfm is not None:
            b, n, c, h, w = patches.shape
            patches = patches.view(b * n, c, h, w)
            patches = torch.stack([tfm(p) for p in patches], dim=0)
            patches = patches.view(b, n, c, h, w)
        labels = torch.tensor(labels, dtype=torch.long)
        return patches, labels, meta

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader


@torch.no_grad()
def _evaluate(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    for patches, labels, _ in loader:
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        b = patches.shape[0]
        logits_all = []
        for i in range(b):
            logits, _ = model(patches[i])
            logits_all.append(logits)
        logits = torch.stack(logits_all, dim=0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    macro_f1 = compute_macro_f1(y_true, y_pred, class_names)
    per_class = compute_per_class_f1_and_recall(y_true, y_pred, class_names)
    return acc, macro_f1, per_class


def train_patch_mil_single():
    cfg = read_config()
    p = read_params()
    mil_cfg = p["patch_mil"]
    seed_everything(mil_cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(cfg["data"]["processed_dir"]) / "single"
    model_dir = Path(cfg["artifacts"]["model_dir"]) / "patch_mil_single"
    reports_dir = Path(cfg["artifacts"]["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Training Dataset (to define class map)
    train_ds = PatchMILDatasetSingle(
        root_dir=data_root / "train",
        patch_size=mil_cfg["patch_size"],
        stride=mil_cfg["stride"],
        max_patches_per_image=mil_cfg["max_patches_per_image"],
        seed=mil_cfg["seed"],
    )
    
    # 2. Initialize Validation Dataset with SAME class map
    val_ds = PatchMILDatasetSingle(
        root_dir=data_root / "val",
        patch_size=mil_cfg["patch_size"],
        stride=mil_cfg["stride"],
        max_patches_per_image=mil_cfg["max_patches_per_image"],
        seed=mil_cfg["seed"],
        class_to_idx=train_ds.class_to_idx
    )

    train_loader = _make_dataloader(train_ds, mil_cfg, train=True)
    val_loader = _make_dataloader(val_ds, mil_cfg, train=False)

    class_map = {i: c for i, c in enumerate(train_ds.classes)}
    (model_dir / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    print(f"Class map: {class_map}")

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    model = PatchMILClassifier(
        num_classes=len(train_ds.classes),
        model_name=mil_cfg["model_name"],
    ).to(device)

    # 3a. Weighted CrossEntropyLoss (Sqrt Inverse Freq)
    counts = Counter([s[1] for s in train_ds.samples])
    weights = []
    for i in range(len(train_ds.classes)):
        weights.append(math.sqrt(1.0 / (counts[i] + 1e-6))) # Sqrt for milder weighting
    weights = torch.tensor(weights, dtype=torch.float, device=device)
    weights = weights / weights.sum() * len(train_ds.classes)
    print(f"Class weights: {dict(zip(train_ds.classes, weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # 3b. Initial Optimizer (Backbone Frozen)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        [
            {"params": model.mil.parameters(), "lr": mil_cfg["lr"]},
            {"params": model.classifier.parameters(), "lr": mil_cfg["lr"]},
        ],
        weight_decay=mil_cfg["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_f1 = -1.0
    epochs_no_improve = 0
    epoch_logs = []
    backbone_unfrozen = False

    # 4. Training Loop
    freeze_epochs = 5
    accum_steps = 2  # Effective batch size = 4 * 2 = 8
    
    for epoch in range(1, mil_cfg["epochs"] + 1):
        # Unfreeze backbone after freeze_epochs
        if not backbone_unfrozen and epoch > freeze_epochs:
            print("🔓 Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            # Re-initialize optimizer with differential learning rates
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": mil_cfg["lr"] * 0.1},
                    {"params": model.mil.parameters(), "lr": mil_cfg["lr"]},
                    {"params": model.classifier.parameters(), "lr": mil_cfg["lr"]},
                ],
                weight_decay=mil_cfg["weight_decay"],
            )
            # Re-initialize scheduler to match new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5
            )
            backbone_unfrozen = True

        model.train()
        total_loss = 0.0

        for patches, labels, _ in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # optimizer.zero_grad(set_to_none=True)  # Move inside accumulation loop
            logits_batch = []
            for i in range(patches.shape[0]):
                logits_img, _ = model(patches[i])
                logits_batch.append(logits_img)
            
            logits = torch.stack(logits_batch, dim=0)
            logits = torch.stack(logits_batch, dim=0)
            loss = criterion(logits, labels)
            loss = loss / accum_steps
            loss.backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accum_steps

        avg_loss = total_loss / max(1, len(train_loader))
        val_acc, val_macro_f1, per_class = _evaluate(model, val_loader, device, train_ds.classes)

        ncd = per_class.get("ncd", {"f1": 0.0, "recall": 0.0})
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_ncd_f1": ncd["f1"],
            "val_ncd_recall": ncd["recall"],
            "lr_main": optimizer.param_groups[-1]["lr"]
        })

        print(
            f"[Epoch {epoch}/{mil_cfg['epochs']}] "
            f"loss={avg_loss:.4f} acc={val_acc:.4f} f1={val_macro_f1:.4f} "
            f"ncd_f1={ncd['f1']:.4f} "
            f"lr={optimizer.param_groups[-1]['lr']:.6f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_dir / "model.pt")
            print(f"✅ Saved best model (f1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
        
        scheduler.step(val_macro_f1)

        if epochs_no_improve >= mil_cfg["patience"]:
            print(f"🛑 Early stopping after {epoch} epochs")
            break

    (reports_dir / "patch_mil_single_epoch_metrics.json").write_text(
        json.dumps(epoch_logs, indent=2), encoding="utf-8"
    )


def main():
    train_patch_mil_single()


if __name__ == "__main__":
    main()
