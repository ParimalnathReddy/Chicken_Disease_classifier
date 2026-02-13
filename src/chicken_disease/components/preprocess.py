import json
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def preprocess(
    interim_dir: str,
    processed_dir: str,
    window_size: int = 5,
    split=(0.7, 0.15, 0.15),
):
    random.seed(RANDOM_SEED)

    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)

    # reset processed
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)

    # ---------- SINGLE IMAGE SPLITS ----------
    single_dir = processed_dir / "single"
    for s in ["train", "val", "test"]:
        (single_dir / s).mkdir(parents=True)

    # ---------- MULTI IMAGE METADATA ----------
    multi_dir = processed_dir / "multi"
    multi_dir.mkdir(parents=True)

    train_meta, val_meta, test_meta = [], [], []

    for class_dir in interim_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        if len(images) < window_size:
            print(f"[WARN] Not enough images for class {class_dir.name}")
            continue

        train_imgs, temp_imgs = train_test_split(
            images, test_size=(1 - split[0]), random_state=RANDOM_SEED
        )

        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=split[2] / (split[1] + split[2]),
            random_state=RANDOM_SEED,
        )

        def copy_single(imgs, split_name):
            out = single_dir / split_name / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, out / img.name)

        copy_single(train_imgs, "train")
        copy_single(val_imgs, "val")
        copy_single(test_imgs, "test")

        def make_windows(imgs, target):
            imgs = list(imgs)
            random.shuffle(imgs)
            for i in range(0, len(imgs) - window_size + 1, window_size):
                window = imgs[i:i + window_size]
                target.append({
                    "paths": [str(p) for p in window],
                    "label": class_dir.name
                })

        make_windows(train_imgs, train_meta)
        make_windows(val_imgs, val_meta)
        make_windows(test_imgs, test_meta)

    # save metadata
    (multi_dir / "train.json").write_text(json.dumps(train_meta, indent=2))
    (multi_dir / "val.json").write_text(json.dumps(val_meta, indent=2))
    (multi_dir / "test.json").write_text(json.dumps(test_meta, indent=2))

    print("✅ Stage 02 complete")
    print(f"Single-image data → {single_dir}")
    print(f"Multi-image metadata → {multi_dir}")
