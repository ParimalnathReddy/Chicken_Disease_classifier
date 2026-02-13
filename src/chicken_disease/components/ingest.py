import shutil
from pathlib import Path
from PIL import Image
import hashlib

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Map whatever raw folder names are -> our standard names
CLASS_MAP = {
    "healthy": "healthy",
    "cocci": "cocci",
    "salmo": "salmo",
    "ncd": "ncd",
    # If your folders are named differently, add here, e.g.
    # "Coccidiosis": "cocci",
    # "Salmonella": "salmo",
    # "Newcastle": "ncd",
}

def _is_valid_image(path: Path) -> bool:
    """
    Returns True if file is a readable image.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # verifies header integrity
        return True
    except Exception:
        return False

def _hash_file(path: Path, block_size=65536) -> str:
    """
    Hash file content to detect duplicates.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def ingest(raw_dir: str, interim_dir: str, remove_duplicates: bool = True):
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    # Reset interim (fresh, reproducible)
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Track duplicates across whole dataset
    seen_hashes = set()

    # Find class folders in raw
    raw_folders = [p for p in raw_dir.iterdir() if p.is_dir()]

    kept = 0
    skipped = 0
    corrupted = 0
    dupes = 0

    for folder in raw_folders:
        raw_name = folder.name
        if raw_name not in CLASS_MAP:
            # Ignore non-class folders (e.g., annotations)
            continue

        std_class = CLASS_MAP[raw_name]
        out_class_dir = interim_dir / std_class
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for file in folder.rglob("*"):
            if not file.is_file():
                continue

            ext = file.suffix.lower()
            if ext not in ALLOWED_EXT:
                skipped += 1
                continue

            if not _is_valid_image(file):
                corrupted += 1
                continue

            file_hash = None
            if remove_duplicates:
                file_hash = _hash_file(file)
                if file_hash in seen_hashes:
                    dupes += 1
                    continue
                seen_hashes.add(file_hash)

            # Copy to interim with stable naming
            dest = out_class_dir / file.name
            # Avoid overwriting if same filename repeats
            if dest.exists():
                if file_hash is None:
                    file_hash = _hash_file(file)
                dest = out_class_dir / f"{file.stem}_{file_hash[:8]}{file.suffix}"

            shutil.copy2(file, dest)
            kept += 1

    print("✅ Ingestion complete")
    print(f"Kept images:      {kept}")
    print(f"Skipped non-img:  {skipped}")
    print(f"Corrupted removed:{corrupted}")
    print(f"Duplicates removed:{dupes}")
    print(f"Output -> {interim_dir}")
