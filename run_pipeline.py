import sys
import os
from pathlib import Path

# Add src to python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from chicken_disease.pipeline.stage_01_ingest import main as ingest_main
from chicken_disease.pipeline.stage_02_preprocess import main as preprocess_main
from chicken_disease.pipeline.stage_03_train_single import main as train_main

if __name__ == "__main__":
    print("Running Stage 01: Ingestion...")
    ingest_main()
    print("\nRunning Stage 02: Preprocess...")
    preprocess_main()
    print("\nRunning Stage 03: Training...")
    train_main()
    print("\nPipeline execution completed successfully.")
