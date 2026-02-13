from chicken_disease.utils.config import read_config
from chicken_disease.components.ingest import ingest

def main():
    cfg = read_config()
    ingest(cfg["data"]["raw_dir"], cfg["data"]["interim_dir"], remove_duplicates=True)

if __name__ == "__main__":
    main()
