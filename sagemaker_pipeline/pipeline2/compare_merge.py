"""
Pipeline 2 - Step 4: Compare DINOv3 vs VLM, correct mistakes, merge data.

Logic:
  - Compare DINOv3's prediction vs VLM's YES/NO for each test image
  - VLM is ground truth
  - Add ALL test images with VLM-corrected labels to training set
  - Merge with Pipeline 1's training data

Input:  /opt/ml/processing/input/predictions/predictions.json
        /opt/ml/processing/input/predictions/test_images/
        /opt/ml/processing/input/vl_labels/vl_labels.json
        /opt/ml/processing/input/prev_train/train.pt
Output: /opt/ml/processing/train/train.pt    (merged)
        /opt/ml/processing/val/val.pt        (pass-through)
        /opt/ml/processing/test/test.pt      (pass-through)
        /opt/ml/processing/output/comparison_report.json
"""
import os, json
import torch
from PIL import Image
from torchvision import transforms

# --- Load inputs ---
with open("/opt/ml/processing/input/predictions/predictions.json") as f:
    predictions = json.load(f)

with open("/opt/ml/processing/input/vl_labels/vl_labels.json") as f:
    vl_labels = json.load(f)

prev_train = torch.load(
    "/opt/ml/processing/input/prev_train/train.pt",
    weights_only=False,
)

print(f"Previous training data: {len(prev_train['labels'])} samples")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_img_dir = "/opt/ml/processing/input/predictions/test_images"

# --- Compare and collect ---
report = {"correct": [], "incorrect": [], "skipped": []}
new_images = []
new_labels = []

for filename in sorted(predictions.keys()):
    pred_info = predictions[filename]
    vl_info = vl_labels.get(filename, {})

    dino_pred = pred_info["predicted_label"]
    vl_label = vl_info.get("vl_label")

    if vl_label is None:
        report["skipped"].append({"filename": filename, "reason": "VL failed"})
        print(f"  SKIP {filename}: VL could not determine")
        continue

    if dino_pred == vl_label:
        report["correct"].append({
            "filename": filename,
            "label": vl_label,
            "distance": pred_info.get("distance"),
        })
        status = "CORRECT"
    else:
        report["incorrect"].append({
            "filename": filename,
            "dino_pred": dino_pred,
            "vl_label": vl_label,
            "distance": pred_info.get("distance"),
        })
        dino_str = "TARGET" if dino_pred == 1 else "DISTRACT"
        vl_str = "TARGET" if vl_label == 1 else "DISTRACT"
        status = f"INCORRECT (DINOv3={dino_str}, VL={vl_str})"

    # Add image with VLM's corrected label
    filepath = os.path.join(test_img_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found")
        continue

    img = Image.open(filepath).convert("RGB")
    new_images.append(transform(img))
    new_labels.append(vl_label)
    print(f"  {status}: {filename} -> label={vl_label}")

print(f"\nComparison: {len(report['correct'])} correct, "
      f"{len(report['incorrect'])} incorrect, {len(report['skipped'])} skipped")

# --- Merge ---
if new_images:
    new_images_tensor = torch.stack(new_images)
    new_labels_tensor = torch.tensor(new_labels, dtype=torch.long)
    merged_images = torch.cat([prev_train["images"], new_images_tensor], dim=0)
    merged_labels = torch.cat([prev_train["labels"], new_labels_tensor], dim=0)
else:
    merged_images = prev_train["images"]
    merged_labels = prev_train["labels"]

n_pos = (merged_labels == 1).sum().item()
n_neg = (merged_labels == 0).sum().item()
print(f"Merged: {len(merged_labels)} samples ({n_pos} positive, {n_neg} negative)")

# --- Save ---
for split_name in ["train", "val", "test"]:
    os.makedirs(f"/opt/ml/processing/{split_name}", exist_ok=True)

torch.save({
    "images": merged_images,
    "labels": merged_labels,
    "target_class": prev_train["target_class"],
    "distractor_classes": prev_train["distractor_classes"],
}, "/opt/ml/processing/train/train.pt")

# Pass through val/test
for split in ["val", "test"]:
    prev = torch.load(f"/opt/ml/processing/input/prev_{split}/{split}.pt", weights_only=False)
    torch.save(prev, f"/opt/ml/processing/{split}/{split}.pt")
    print(f"Passed through {split}.pt: {len(prev['labels'])} samples")

os.makedirs("/opt/ml/processing/output", exist_ok=True)
with open("/opt/ml/processing/output/comparison_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Done!")
