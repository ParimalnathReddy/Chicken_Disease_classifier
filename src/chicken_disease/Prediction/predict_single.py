# save as test_one.py and run: python test_one.py /path/to/image.jpg
import json
import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm

IMG_PATH = Path(sys.argv[1])
MODEL_PATH = Path(r"/mnt/scratch/kodumuru/pari/chicken_disease/models/baseline_single/model.pt")
CLASS_MAP = Path(r"/mnt/scratch/kodumuru/pari/chicken_disease/models/baseline_single/class_map.json")
IMAGE_SIZE = 224  # from params.yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

idx_to_class = json.loads(CLASS_MAP.read_text())
num_classes = len(idx_to_class)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

tfm = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

img = Image.open(IMG_PATH).convert("RGB")
x = tfm(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()

print("Predicted class:", idx_to_class[str(pred)])





# how to run
# python /mnt/scratch/kodumuru/pari/chicken_disease/src/chicken_disease/Prediction/predict_single.py /path/to/image.jpg
# python /mnt/scratch/kodumuru/pari/chicken_disease/src/chicken_disease/Prediction/predict_single.py /mnt/scratch/kodumuru/pari/chicken_disease/data/raw/ncd/ncd.3.jpg
