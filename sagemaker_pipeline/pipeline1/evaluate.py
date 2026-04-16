"""
Pipeline 1 - Step 4: Evaluate with binary metrics (TPR, FPR, F1).

For each test image, compute cosine distance to gallery reference.
Apply learned threshold to predict "target" or "distractor".

Input:  /opt/ml/processing/model/model.tar.gz
        /opt/ml/processing/test/test.pt
        /opt/ml/processing/gallery/gallery.pt
        /opt/ml/processing/pretrained/
Output: /opt/ml/processing/evaluation/evaluation.json
"""
import json, os, tarfile, subprocess
subprocess.check_call(["pip", "install", "transformers"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

# --- Extract model ---
with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
    tar.extractall(path="/opt/ml/processing/model")

checkpoint = torch.load("/opt/ml/processing/model/model.pth", weights_only=False, map_location="cpu")
backbone_dim = checkpoint["backbone_dim"]
embed_dim = checkpoint["embed_dim"]
threshold = checkpoint["threshold"]

print(f"Model: backbone_dim={backbone_dim}, embed_dim={embed_dim}, threshold={threshold:.2f}")

projector = nn.Sequential(
    nn.Linear(backbone_dim, embed_dim),
    nn.BatchNorm1d(embed_dim),
)
projector.load_state_dict(checkpoint["projector_state"])

backbone = AutoModel.from_pretrained("/opt/ml/processing/pretrained")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device).eval()
projector = projector.to(device).eval()

# --- Load data ---
test_data = torch.load("/opt/ml/processing/test/test.pt", weights_only=False)
gallery_data = torch.load("/opt/ml/processing/gallery/gallery.pt", weights_only=False)

def extract_embeddings(images):
    embs = []
    loader = DataLoader(images, batch_size=16)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = backbone(batch).last_hidden_state[:, 0]
            emb = F.normalize(projector(features), dim=1)
            embs.append(emb)
    return torch.cat(embs, dim=0)

gallery_emb = extract_embeddings(gallery_data["images"])  # (1, 128)
test_emb = extract_embeddings(test_data["images"])         # (N, 128)
test_labels = test_data["labels"].to(device)

# --- Compute distances and predict ---
sim = (test_emb @ gallery_emb.T).squeeze(1)
distances = 1.0 - sim
predictions = (distances < threshold).long()  # 1=target, 0=distractor

# --- Binary metrics ---
tp = ((predictions == 1) & (test_labels == 1)).sum().item()
fp = ((predictions == 1) & (test_labels == 0)).sum().item()
tn = ((predictions == 0) & (test_labels == 0)).sum().item()
fn = ((predictions == 0) & (test_labels == 1)).sum().item()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tpr
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (tp + tn) / (tp + fp + tn + fn)

print(f"Threshold: {threshold:.2f}")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"TPR (recall): {tpr:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# --- Per-sample details ---
for i in range(len(test_labels)):
    label = "TARGET" if test_labels[i] == 1 else "DISTRACT"
    pred = "TARGET" if predictions[i] == 1 else "DISTRACT"
    correct = "OK" if predictions[i] == test_labels[i] else "WRONG"
    print(f"  [{correct}] true={label} pred={pred} dist={distances[i]:.4f}")

# --- Save (use "accuracy" key for pipeline condition compatibility) ---
os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump({"metrics": {
        "accuracy": f1,  # use F1 as the gate metric
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "threshold": threshold,
    }}, f)
