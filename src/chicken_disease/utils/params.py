import yaml
from pathlib import Path

def read_params(path: str = "params.yaml") -> dict:
    # Resolve project root (src/chicken_disease/utils -> root)
    root = Path(__file__).resolve().parent.parent.parent.parent
    full_path = root / path
    
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
