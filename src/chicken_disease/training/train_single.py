import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import timm
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score

from chicken_disease.utils.config import read_config
from chicken_disease.utils.params import read_params
from chicken_disease.utils.seed import seed_everything


def get_loaders(data_root: Path, image_size: int, batch_size: int, num_workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    tfm_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)], p=0.7
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 5.0))], p=0.4
        ),
        transforms.ToTensor(),
    ])

    tfm_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(train_dir, transform=tfm_train)
    val_ds = ImageFolder(val_dir, transform=tfm_eval)
    test_ds = ImageFolder(test_dir, transform=tfm_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, idx_to_class):
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ncd_idx = None
    for i, name in idx_to_class.items():
        if name == "ncd":
            ncd_idx = i
            break
    if ncd_idx is None:
        ncd_f1 = 0.0
        ncd_recall = 0.0
    else:
        ncd_f1 = f1_score(
            y_true, y_pred, labels=[ncd_idx], average=None, zero_division=0
        )[0]
        ncd_recall = recall_score(
            y_true, y_pred, labels=[ncd_idx], average=None, zero_division=0
        )[0]
    report = classification_report(
        y_true, y_pred,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True,
        zero_division=0,
    )
    return acc, macro_f1, ncd_f1, ncd_recall, report


def train_single():
    cfg = read_config()
    p = read_params()

    train_cfg = p["train_single"]
    seed_everything(train_cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parent.parent.parent.parent
    data_root = root / cfg["data"]["processed_dir"] / "single"
    model_dir = root / cfg["artifacts"]["model_dir"] / "baseline_single"
    reports_dir = root / cfg["artifacts"]["reports_dir"]
    
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = get_loaders(
        data_root=data_root,
        image_size=train_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    num_classes = len(train_ds.classes)
    class_map = {i: c for i, c in enumerate(train_ds.classes)}
    (model_dir / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")

    model = timm.create_model(
        train_cfg["model_name"],
        pretrained=True,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"]
    )

    best_val_f1 = -1.0
    epochs_no_improve = 0
    epoch_metrics = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        val_acc, val_macro_f1, val_ncd_f1, val_ncd_recall, val_report = evaluate(
            model, val_loader, device, class_map
        )

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_ncd_f1": val_ncd_f1,
            "val_ncd_recall": val_ncd_recall,
        })

        print(
            f"[Epoch {epoch}/{train_cfg['epochs']}] "
            f"train_loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_dir / "model.pt")
            (reports_dir / "baseline_val_report.json").write_text(
                json.dumps(val_report, indent=2), encoding="utf-8"
            )
            print(f"✅ Saved best model (val_macro_f1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                print("🛑 Early stopping: no macro-F1 improvement for 8 epochs")
                break

    # final test evaluation on best checkpoint
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    test_acc, _, _, _, test_report = evaluate(model, test_loader, device, class_map)

    out = {
        "best_val_macro_f1": best_val_f1,
        "test_acc": test_acc,
        "test_report": test_report
    }
    (reports_dir / "baseline_test_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (reports_dir / "baseline_epoch_metrics.json").write_text(
        json.dumps(epoch_metrics, indent=2), encoding="utf-8"
    )

    print("✅ Baseline training complete")
    print(f"Best val macro-F1: {best_val_f1:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
