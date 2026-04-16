"""
Pipeline 1 - Step 2: Preprocess for binary verification (is / is not target).

- Train: target images (label=1) + distractor images (label=0)
- Gallery: 1 original target image (tensor + PNG for VLM)
- Val: real target images + distractors (for threshold tuning)
- Test: different real target images + distractors (for evaluation)

Uses multiple OTHER classes from Flowers-102 as distractors.

Input:  /opt/ml/processing/input/generated/  (images/, labels.json)
Output: /opt/ml/processing/train/train.pt
        /opt/ml/processing/gallery/gallery.pt
        /opt/ml/processing/gallery_png/       (PNG reference for VLM)
        /opt/ml/processing/val/val.pt
        /opt/ml/processing/test/test.pt
"""
import os, json, shutil
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Flowers102

INPUT_DIR = "/opt/ml/processing/input/generated"
OUTPUT_DIR = "/opt/ml/processing"

TARGET_CLASS = 12  # our target identity
DISTRACTOR_CLASSES = [27, 51, 64, 76]  # other flowers as distractors

# Per split: how many real images per class
TRAIN_DISTRACTORS_PER_CLASS = 3   # 4 classes x 3 = 12 distractors in train
VAL_TARGET = 5                     # 5 real target images for val
VAL_DISTRACTORS_PER_CLASS = 3      # 4 x 3 = 12 distractors in val
TEST_TARGET = 5                    # 5 real target images for test
TEST_DISTRACTORS_PER_CLASS = 3     # 4 x 3 = 12 distractors in test

# DINOv3 preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load generated target images as train positives ---
with open(os.path.join(INPUT_DIR, "labels.json")) as f:
    labels_map = json.load(f)

img_dir = os.path.join(INPUT_DIR, "images")
train_images, train_labels = [], []

for filename, label in labels_map.items():
    filepath = os.path.join(img_dir, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping")
        continue
    img = Image.open(filepath).convert("RGB")
    train_images.append(transform(img))
    train_labels.append(1)  # target = 1

print(f"Train positives (generated + original): {len(train_labels)}")

# --- Build gallery (first original target image) ---
gallery_png_dir = os.path.join(OUTPUT_DIR, "gallery_png")
os.makedirs(gallery_png_dir, exist_ok=True)

gallery_src = os.path.join(img_dir, "target_orig0.png")
gallery_img = Image.open(gallery_src).convert("RGB")
gallery_tensor = transform(gallery_img).unsqueeze(0)

shutil.copy2(gallery_src, os.path.join(gallery_png_dir, "target_ref.png"))

# Save gallery metadata
with open(os.path.join(gallery_png_dir, "gallery_meta.json"), "w") as f:
    json.dump({"target_class": TARGET_CLASS, "distractor_classes": DISTRACTOR_CLASSES}, f)

print(f"Gallery: 1 reference image (target_orig0.png)")

# --- Load Flowers-102 for distractors and real target images ---
print("Downloading Flowers-102...")
train_dataset = Flowers102(root="/tmp/flowers", split="train", download=True)
test_dataset = Flowers102(root="/tmp/flowers", split="test", download=True)

# Collect images per class from train split (for train distractors)
train_class_imgs = {c: [] for c in DISTRACTOR_CLASSES}
for idx in range(len(train_dataset)):
    img, label = train_dataset[idx]
    if label in train_class_imgs and len(train_class_imgs[label]) < TRAIN_DISTRACTORS_PER_CLASS:
        train_class_imgs[label].append(img)

# Add train distractors (label=0)
for cls_id, imgs in train_class_imgs.items():
    for img in imgs:
        train_images.append(transform(img.convert("RGB")))
        train_labels.append(0)  # distractor = 0
    print(f"  Train distractors from class {cls_id}: {len(imgs)}")

train_images = torch.stack(train_images)
train_labels = torch.tensor(train_labels, dtype=torch.long)
print(f"Train total: {len(train_labels)} ({train_labels.sum().item()} positive, {(train_labels == 0).sum().item()} negative)")

# --- Collect from test split for val/test ---
test_target_imgs = []
test_distractor_imgs = {c: [] for c in DISTRACTOR_CLASSES}
needed_target = VAL_TARGET + TEST_TARGET
needed_distractor = VAL_DISTRACTORS_PER_CLASS + TEST_DISTRACTORS_PER_CLASS

for idx in range(len(test_dataset)):
    img, label = test_dataset[idx]
    if label == TARGET_CLASS and len(test_target_imgs) < needed_target:
        test_target_imgs.append(img)
    elif label in test_distractor_imgs and len(test_distractor_imgs[label]) < needed_distractor:
        test_distractor_imgs[label].append(img)

# --- Build val set ---
val_images, val_labels = [], []

for img in test_target_imgs[:VAL_TARGET]:
    val_images.append(transform(img.convert("RGB")))
    val_labels.append(1)

for cls_id in DISTRACTOR_CLASSES:
    for img in test_distractor_imgs[cls_id][:VAL_DISTRACTORS_PER_CLASS]:
        val_images.append(transform(img.convert("RGB")))
        val_labels.append(0)

val_images = torch.stack(val_images)
val_labels = torch.tensor(val_labels, dtype=torch.long)
print(f"Val: {len(val_labels)} ({val_labels.sum().item()} positive, {(val_labels == 0).sum().item()} negative)")

# --- Build test set ---
test_images, test_labels = [], []

for img in test_target_imgs[VAL_TARGET:VAL_TARGET + TEST_TARGET]:
    test_images.append(transform(img.convert("RGB")))
    test_labels.append(1)

for cls_id in DISTRACTOR_CLASSES:
    for img in test_distractor_imgs[cls_id][VAL_DISTRACTORS_PER_CLASS:needed_distractor]:
        test_images.append(transform(img.convert("RGB")))
        test_labels.append(0)

test_images = torch.stack(test_images)
test_labels = torch.tensor(test_labels, dtype=torch.long)
print(f"Test: {len(test_labels)} ({test_labels.sum().item()} positive, {(test_labels == 0).sum().item()} negative)")

# --- Save ---
common = {"target_class": TARGET_CLASS, "distractor_classes": DISTRACTOR_CLASSES}

for split_name, imgs, lbls in [("train", train_images, train_labels),
                                 ("val", val_images, val_labels),
                                 ("test", test_images, test_labels)]:
    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"images": imgs, "labels": lbls, **common},
               os.path.join(out_dir, f"{split_name}.pt"))
    print(f"Saved {split_name}.pt")

gallery_dir = os.path.join(OUTPUT_DIR, "gallery")
os.makedirs(gallery_dir, exist_ok=True)
torch.save({"images": gallery_tensor, "labels": torch.tensor([1], dtype=torch.long), **common},
           os.path.join(gallery_dir, "gallery.pt"))
print("Saved gallery.pt")
