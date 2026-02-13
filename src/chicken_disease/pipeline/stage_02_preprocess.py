from chicken_disease.utils.config import read_config
from chicken_disease.components.preprocess import preprocess

def main():
    cfg = read_config()
    preprocess(
        interim_dir=cfg["data"]["interim_dir"],
        processed_dir=cfg["data"]["processed_dir"],
        window_size=5
    )

if __name__ == "__main__":
    main()
