"""
Pipeline 1 - Step 3: Train DINOv3 embedding for binary verification.

Frozen DINOv3 backbone + trainable projector (384 -> 128d).
Triplet loss: pull target images together, push distractors away.
After training, search optimal distance threshold on val set.

Input:  /opt/ml/input/data/train/train.pt     (target + distractors)
        /opt/ml/input/data/val/val.pt         (for threshold tuning)
        /opt/ml/input/data/gallery/gallery.pt (1 reference image)
        /opt/ml/input/data/pretrained/        (DINOv3 model)
Output: /opt/ml/model/model.pth
        (contains projector weights + optimal threshold)
"""
import os, subprocess
subprocess.check_call(["pip", "install", "transformers"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

EMBED_DIM = 128
MARGIN = 0.3


def batch_hard_triplet_loss(embeddings, labels, margin):
    """Batch Hard Triplet Mining for binary labels (1=target, 0=distractor)."""
    # Pairwise squared L2 via dot product (keeps grad through embeddings)
    dot = embeddings @ embeddings.T
    sq_norms = torch.diag(dot)
    dist_mat = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * dot
    dist_mat = torch.clamp(dist_mat, min=1e-12).sqrt()

    n = embeddings.size(0)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

    losses = []
    for i in range(n):
        pos_mask = label_eq[i].clone()
        pos_mask[i] = False
        if pos_mask.sum() == 0:
            continue
        hardest_pos = dist_mat[i][pos_mask].max()

        neg_mask = ~label_eq[i]
        if neg_mask.sum() == 0:
            continue
        hardest_neg = dist_mat[i][neg_mask].min()

        loss = F.relu(hardest_pos - hardest_neg + margin)
        losses.append(loss)

    if not losses:
        # Return zero that still has grad_fn
        return (embeddings * 0).sum()
    return torch.stack(losses).mean()


def find_best_threshold(gallery_emb, query_embs, query_labels):
    """Search optimal cosine distance threshold on val set.

    Returns threshold that maximizes F1 score.
    """
    # Cosine distance to gallery
    gallery_norm = F.normalize(gallery_emb, dim=1)  # (1, 128)
    query_norm = F.normalize(query_embs, dim=1)
    sim = (query_norm @ gallery_norm.T).squeeze(1)   # (N,)
    distances = 1.0 - sim  # cosine distance

    best_f1, best_thresh = 0.0, 0.5
    for thresh in [i * 0.05 for i in range(1, 20)]:  # 0.05 to 0.95
        preds = (distances < thresh).long()
        tp = ((preds == 1) & (query_labels == 1)).sum().float()
        fp = ((preds == 1) & (query_labels == 0)).sum().float()
        fn = ((preds == 0) & (query_labels == 1)).sum().float()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


# --- Load data ---
train_data = torch.load("/opt/ml/input/data/train/train.pt", weights_only=False)
val_data = torch.load("/opt/ml/input/data/val/val.pt", weights_only=False)
gallery_data = torch.load("/opt/ml/input/data/gallery/gallery.pt", weights_only=False)

n_pos = (train_data["labels"] == 1).sum().item()
n_neg = (train_data["labels"] == 0).sum().item()
print(f"Train: {n_pos} positive + {n_neg} negative = {n_pos + n_neg}")
print(f"Val: {len(val_data['labels'])}, Gallery: {len(gallery_data['labels'])}")

train_loader = DataLoader(TensorDataset(train_data["images"], train_data["labels"]), batch_size=16, shuffle=True)

# --- Build model ---
backbone = AutoModel.from_pretrained("/opt/ml/input/data/pretrained")
backbone_dim = backbone.config.hidden_size

for param in backbone.parameters():
    param.requires_grad = False

projector = nn.Sequential(
    nn.Linear(backbone_dim, EMBED_DIM),
    nn.BatchNorm1d(EMBED_DIM),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
projector = projector.to(device)

optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=1e-4)

# --- Pre-extract backbone features for gallery & val ---
def extract_backbone(images):
    feats = []
    loader = DataLoader(images, batch_size=16)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = backbone(batch).last_hidden_state[:, 0]
            feats.append(f)
    return torch.cat(feats, dim=0)

gallery_feats = extract_backbone(gallery_data["images"])
val_feats = extract_backbone(val_data["images"])

# --- Train ---
epochs = int(os.environ.get("SM_HP_EPOCHS", "20"))
best_f1 = 0.0
best_threshold = 0.5

for epoch in range(epochs):
    projector.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = backbone(images).last_hidden_state[:, 0]
        embeddings = F.normalize(projector(features), dim=1)

        loss = batch_hard_triplet_loss(embeddings, labels, MARGIN)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validate: find best threshold on val set
    projector.eval()
    with torch.no_grad():
        gallery_emb = F.normalize(projector(gallery_feats), dim=1)
        val_emb = F.normalize(projector(val_feats), dim=1)

    thresh, f1 = find_best_threshold(gallery_emb, val_emb, val_data["labels"].to(device))
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_f1: {f1:.4f} - threshold: {thresh:.2f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nBest val F1: {best_f1:.4f}, Best threshold: {best_threshold:.2f}")

# --- Save ---
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(model_dir, exist_ok=True)
torch.save({
    "projector_state": projector.state_dict(),
    "backbone_dim": backbone_dim,
    "embed_dim": EMBED_DIM,
    "threshold": best_threshold,
    "target_class": train_data["target_class"],
}, os.path.join(model_dir, "model.pth"))
print(f"Model saved.")
