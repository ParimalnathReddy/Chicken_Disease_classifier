import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from chicken_disease.training.train_patch_mil_multi import train_patch_mil_multi


def main():
    train_patch_mil_multi()


if __name__ == "__main__":
    main()
