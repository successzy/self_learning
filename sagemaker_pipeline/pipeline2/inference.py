"""
Pipeline 2 - Step 2: DINOv3 binary inference on new images + distractors.

Build a test set: new target images + fresh distractors from Flowers-102.
For each image, compute cosine distance to gallery, predict target/distractor.

Input:  /opt/ml/processing/input/model/model.tar.gz
        /opt/ml/processing/input/generated/  (new target images)
        /opt/ml/processing/input/gallery/gallery.pt
        /opt/ml/processing/input/pretrained/
Output: /opt/ml/processing/output/predictions.json
        /opt/ml/processing/output/test_images/  (PNG files for VLM)
"""
import os, json, tarfile, subprocess
subprocess.check_call(["pip", "install", "transformers"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
from transformers import AutoModel

DISTRACTOR_CLASSES = [27, 51, 64, 76]
DISTRACTORS_PER_CLASS = 2  # 4 x 2 = 8 distractors

# --- Extract model ---
with tarfile.open("/opt/ml/processing/input/model/model.tar.gz") as tar:
    tar.extractall(path="/opt/ml/processing/input/model")

checkpoint = torch.load(
    "/opt/ml/processing/input/model/model.pth",
    weights_only=False, map_location="cpu",
)
backbone_dim = checkpoint["backbone_dim"]
embed_dim = checkpoint["embed_dim"]
threshold = checkpoint["threshold"]

print(f"Model: embed_dim={embed_dim}, threshold={threshold:.2f}")

# --- Build model ---
projector = nn.Sequential(
    nn.Linear(backbone_dim, embed_dim),
    nn.BatchNorm1d(embed_dim),
)
projector.load_state_dict(checkpoint["projector_state"])

backbone = AutoModel.from_pretrained("/opt/ml/processing/input/pretrained")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device).eval()
projector = projector.to(device).eval()

# --- Load gallery ---
gallery_data = torch.load("/opt/ml/processing/input/gallery/gallery.pt", weights_only=False)
with torch.no_grad():
    g_feat = backbone(gallery_data["images"].to(device)).last_hidden_state[:, 0]
    gallery_emb = F.normalize(projector(g_feat), dim=1)  # (1, 128)

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load new target images ---
gen_dir = "/opt/ml/processing/input/generated"
with open(os.path.join(gen_dir, "labels.json")) as f:
    gen_labels = json.load(f)

img_dir = os.path.join(gen_dir, "images")

# Prepare test image output dir (save PNGs for VLM step)
test_img_dir = "/opt/ml/processing/output/test_images"
os.makedirs(test_img_dir, exist_ok=True)

# Collect all test images: new target + fresh distractors
test_items = []  # (filename, tensor, true_label, png_path)

for filename in sorted(gen_labels.keys()):
    filepath = os.path.join(img_dir, filename)
    if not os.path.exists(filepath):
        continue
    img = Image.open(filepath).convert("RGB")
    # Save PNG copy for VLM
    png_path = os.path.join(test_img_dir, filename)
    img.save(png_path)
    test_items.append((filename, transform(img), 1, png_path))

# Add distractors from Flowers-102 val split (different from Pipeline 1)
print("Loading distractors from Flowers-102 val split...")
dataset = Flowers102(root="/tmp/flowers", split="val", download=True)
distractor_counts = {c: 0 for c in DISTRACTOR_CLASSES}

for idx in range(len(dataset)):
    img, label = dataset[idx]
    if label in distractor_counts and distractor_counts[label] < DISTRACTORS_PER_CLASS:
        distractor_counts[label] += 1
        fname = f"distractor_c{label}_{distractor_counts[label]}.png"
        img_rgb = img.convert("RGB") if hasattr(img, 'convert') else img
        png_path = os.path.join(test_img_dir, fname)
        img_rgb.save(png_path)
        test_items.append((fname, transform(img_rgb), 0, png_path))

n_target = sum(1 for _, _, l, _ in test_items if l == 1)
n_distract = sum(1 for _, _, l, _ in test_items if l == 0)
print(f"Test set: {n_target} target + {n_distract} distractors = {len(test_items)} total")

# --- Predict ---
predictions = {}
with torch.no_grad():
    for filename, tensor, true_label, png_path in test_items:
        emb = F.normalize(
            projector(backbone(tensor.unsqueeze(0).to(device)).last_hidden_state[:, 0]),
            dim=1,
        )
        sim = (emb @ gallery_emb.T).squeeze().item()
        distance = 1.0 - sim
        pred_label = 1 if distance < threshold else 0

        predictions[filename] = {
            "true_label": true_label,
            "predicted_label": pred_label,
            "distance": round(distance, 4),
            "similarity": round(sim, 4),
        }
        true_str = "TARGET" if true_label == 1 else "DISTRACT"
        pred_str = "TARGET" if pred_label == 1 else "DISTRACT"
        ok = "OK" if pred_label == true_label else "WRONG"
        print(f"  [{ok}] {filename}: true={true_str} pred={pred_str} dist={distance:.4f}")

# --- Save ---
os.makedirs("/opt/ml/processing/output", exist_ok=True)
with open("/opt/ml/processing/output/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

correct = sum(1 for v in predictions.values() if v["predicted_label"] == v["true_label"])
print(f"\nDone! {correct}/{len(predictions)} correct")
